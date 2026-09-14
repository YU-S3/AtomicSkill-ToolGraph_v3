"""Public pre-publication diagnostics must not become binding authority."""

import copy
from types import SimpleNamespace

import pytest

from atomic_skillgraph.agents.runtime_policy_projection import (
    pack_downstream_context, unpack_downstream_context,
)
from atomic_skillgraph.core.bindings import (
    BindingExpression, BindingExprKind, BindingResolution, BindingSource,
    BindingStatus, RuntimeBinding,
)
from atomic_skillgraph.core.contracts import ParameterSpec, SemanticPredicate, TaskContract
from atomic_skillgraph.core.edges import GraphEdge, GraphEdgeType
from atomic_skillgraph.core.results import RuntimeLinearPlan, RuntimeOccurrence
from atomic_skillgraph.runtime.binding_store import RuntimeBindingStore
from atomic_skillgraph.runtime.node_executor import NodeExecutor
from atomic_skillgraph.runtime.plan_context import RuntimePlanContextBuilder
from atomic_skillgraph.validation.engine import ValidationEngine
from test_r92_support_and_public_memory import _atomic, _input, _Registry


def _context(navigate=None, output_role="destination"):
    navigate = navigate or _atomic(
        "navigate_candidate", inputs=[ParameterSpec("destination", "location")],
        outputs=[ParameterSpec(output_role, "location")],
    )
    take = _atomic("take_candidate", inputs=[ParameterSpec("object", "object"),
                                             ParameterSpec("source", "location")],
                   preconditions=[SemanticPredicate("object.at_location", {
                       "object": _input("object"), "location": _input("source"),
                   })])
    nav = RuntimeOccurrence("navigate", "nav_occ", navigate.ref, [], {}, [], list(navigate.effects))
    consumers = [RuntimeOccurrence(step, f"{step}_occ", take.ref, [], {
        "object": _input("task_object"),
        "source": BindingExpression(BindingExprKind.DATA_FLOW, source_step="navigate",
                                    source_role=output_role),
    }, [], []) for step in ("take", "take_other")]
    plan = RuntimeLinearPlan("candidate_test", "stored_composite", None,
                             [nav, *consumers], ["navigate", "take", "take_other"],
                             [GraphEdge(f"edge_{item.step_id}", GraphEdgeType.DATA_FLOW,
                                        "navigate", item.step_id, output_role, "source")
                              for item in consumers], [], TaskContract(), {})
    store = RuntimeBindingStore()
    store.seed_task_bindings(SimpleNamespace(task_id="candidate_test", context={
        "semantic_bindings": {"task_object": "pencil_3"},
    }), TaskContract(), 0)
    for consumer in consumers:
        store.resolve_occurrence_specs(consumer, 2)
    facts = [{"predicate": "object.at_location",
              "args": {"object": "pencil_3", "location": "desk_2"},
              "public_evidence_ref": "action_catalog:take:revision:2"}]
    registry = _Registry(navigate, take)
    executor = NodeExecutor(SimpleNamespace(skills=registry), ValidationEngine(), lambda *a: None)
    executor.plan_context_builder = RuntimePlanContextBuilder(registry)
    ctx = SimpleNamespace(
        plan=plan, binding_store=store, world_revision=2,
        harness=SimpleNamespace(public_runtime_relation_facts=lambda: facts),
        action_catalog=[{"revision": 2, "action_type": "GOTO",
                         "arguments": {"destination": "cabinet_3"}}],
        trace_builder=SimpleNamespace(trace=SimpleNamespace(metadata={}, validations=[])),
    )
    return executor, ctx, nav, navigate, facts


def _candidate(value="desk_2", source="agent_proposal", revision=2, resolution="concrete"):
    return {"value": value, "source": source, "revision": revision, "resolution": resolution}


@pytest.mark.parametrize("source", ["agent_proposal", "effect_resolution", "input_identity"])
@pytest.mark.parametrize("value,status", [("desk_2", "supported"), ("cabinet_3", "contradicted")])
def test_unpublished_candidate_assesses_real_edges_without_committing(source, value, status):
    executor, ctx, nav, _, _ = _context()
    before = copy.deepcopy(vars(ctx.binding_store))
    view = executor._downstream_plan_context(
        ctx, nav, producer_output_candidates={"destination": _candidate(value, source)},
    )
    assert vars(ctx.binding_store) == before
    assert ctx.binding_store.validated_outputs(nav.occurrence_id) == {}
    assert {item["consumer_step"] for item in view["output_obligations"]} == {"take", "take_other"}
    for item in view["output_obligations"]:
        assert item["public_relation_status"] == status
        assert item["producer_value_context"] == {
            "status": "candidate", **_candidate(value, source), "is_binding_authority": False,
        }
        assert "source" not in item["consumer_known_semantic_anchors"]
    assert ctx.trace_builder.trace.validations == []


@pytest.mark.parametrize("candidate,private,stale_fact", [
    (None, False, False), (_candidate(revision=1), False, False),
    (_candidate(resolution="semantic"), False, False),
    (_candidate(source="validator_private"), False, False),
    (_candidate(value="hidden_99"), False, False),
    (_candidate(), True, False), (_candidate(), False, True),
])
def test_missing_semantic_stale_or_private_candidates_remain_unknown(candidate, private, stale_fact):
    executor, ctx, nav, _, facts = _context()
    if private:
        facts[0].pop("public_evidence_ref")
    if stale_fact:
        facts[0]["observed_at_revision"] = 1
    view = executor._downstream_plan_context(
        ctx, nav, producer_output_candidates={"destination": candidate} if candidate else {},
    )
    for item in view["output_obligations"]:
        assert item["public_relation_status"] == "unknown"
        assert item["producer_value_context"] is None
        assert item["public_evidence_refs"] == []
    assert ctx.binding_store.validated_outputs(nav.occurrence_id) == {}


def test_identity_derivation_requires_explicit_contract_and_current_grounding():
    executor, ctx, nav, atomic, _ = _context()
    ctx.binding_store.commit_grounded(nav.occurrence_id, {"destination": RuntimeBinding(
        "destination", "desk_2", "location", BindingSource.HARNESS_EVIDENCE,
        BindingStatus.GROUNDED, BindingResolution.CONCRETE, ["public:2"], 2,
    )})
    view = executor._downstream_plan_context(ctx, nav)
    assert view["output_obligations"][0]["public_relation_status"] == "unknown"
    atomic.validator_spec["output_derivations"] = {
        "destination": {"kind": "input_identity", "input_role": "destination"},
    }
    before = copy.deepcopy(vars(ctx.binding_store))
    view = executor._downstream_plan_context(ctx, nav)
    assert view["output_obligations"][0]["public_relation_status"] == "supported"
    assert view["output_obligations"][0]["producer_value_context"]["source"] == "input_identity"
    assert vars(ctx.binding_store) == before
    ctx.world_revision = 3
    assert executor._downstream_plan_context(ctx, nav)["output_obligations"][0]["public_relation_status"] == "unknown"


def test_published_output_takes_precedence_over_agent_candidate():
    executor, ctx, nav, _, _ = _context()
    ctx.binding_store.publish_validated_outputs(nav, {"destination": "desk_2"}, ["valid:2"], 2)
    before = copy.deepcopy(vars(ctx.binding_store))
    view = executor._downstream_plan_context(
        ctx, nav, producer_output_candidates={"destination": _candidate("cabinet_3")},
    )
    assert view["output_obligations"][0]["public_relation_status"] == "supported"
    assert view["output_obligations"][0]["producer_value_context"]["status"] == "published"
    assert vars(ctx.binding_store) == before


def test_effect_resolution_is_assessed_before_output_publication():
    from test_effect_reconciliation import _different_name_effect_output_atomic, _fake_at, _grounded

    atomic = _different_name_effect_output_atomic()
    executor, ctx, nav, _, _ = _context(atomic, "arrived_location")
    ctx.world_revision = 1
    ctx.harness.validator_channel = _fake_at
    ctx.harness.public_runtime_relation_facts = lambda: [{
        "predicate": "object.at_location", "args": {"object": "pencil_3", "location": "cabinet_3"},
        "public_evidence_ref": "action_catalog:take:revision:1",
    }]
    ctx.binding_store.commit_grounded(nav.occurrence_id, {"destination": _grounded("destination", "cabinet_3")})
    result = executor._complete_from_current_effect(nav, ctx, mode="preparation", preferred_values=["cabinet_3"])
    assert result.atomic_effect_passed
    assert result.validated_outputs == {"arrived_location": "cabinet_3"}
    assert ctx.binding_store.validated_outputs(nav.occurrence_id) == {}
    audit = ctx.trace_builder.trace.metadata["producer_output_candidate_assessments"][0]
    assert audit["before_output_publication"] is True
    obligation = audit["context"]["output_obligations"][0]
    assert obligation["public_relation_status"] == "supported"
    assert obligation["producer_value_context"]["source"] == "effect_resolution"
    assert "arrived_location" not in ctx.binding_store.snapshot_for_node(nav)


def test_candidate_provenance_survives_lossless_policy_compaction():
    from test_v32_r5_runtime_projection import _repeated_consumer_context

    raw = _repeated_consumer_context()
    for item in raw["output_obligations"]:
        item.update(relation_predicate="object.at_location", effect_domain="world",
                    relevant_anchor_roles=["object"], public_relation_status="unknown",
                    public_evidence_refs=[], producer_value_context={
                        "status": "candidate", **_candidate(), "is_binding_authority": False,
                    })
    packed, reason = pack_downstream_context(raw)
    assert reason == "deduplicated"
    assert unpack_downstream_context(packed) == raw
