from __future__ import annotations

from types import SimpleNamespace

import pytest

from atomic_skillgraph.core.bindings import BindingExpression, BindingExprKind
from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    ParameterSpec,
    SemanticPredicate,
    TaskContract,
)
from atomic_skillgraph.core.edges import GraphEdge, GraphEdgeType
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.results import (
    ImplementationExecutionResult,
    RuntimeLinearPlan,
    RuntimeOccurrence,
    ToolCallPreflightResult,
)
from atomic_skillgraph.core.status import SkillStatus
from atomic_skillgraph.core.status import RuntimeMode, skill_status_usable
from atomic_skillgraph.harness.alfworld import AlfWorldAdapter
from atomic_skillgraph.runtime.node_executor import NodeExecutor
from atomic_skillgraph.runtime.output_obligations import assess_output_obligation
from atomic_skillgraph.runtime.binding_store import RuntimeBindingStore
from atomic_skillgraph.runtime.plan_context import RuntimePlanContextBuilder
from atomic_skillgraph.runtime.state import ExplorationMemory
from atomic_skillgraph.runtime.support_retriever import SupportAtomicRetriever
from atomic_skillgraph.validation.engine import ValidationEngine


def _input(role: str) -> BindingExpression:
    return BindingExpression(BindingExprKind.SKILL_INPUT, source_role=role)


def _atomic(
    name: str,
    *,
    inputs: list[ParameterSpec] | None = None,
    outputs: list[ParameterSpec] | None = None,
    preconditions: list[SemanticPredicate] | None = None,
    effects: list[SemanticPredicate] | None = None,
) -> AbstractAtomicSkill:
    return AbstractAtomicSkill(
        SkillRef(name, "1.0.0"),
        name,
        list(inputs or ()),
        list(outputs or ()),
        list(preconditions or ()),
        list(effects or ()),
        {},
        [],
        {},
        {},
        SkillStatus.ACTIVE,
    )


class _Registry:
    def __init__(self, *atomics: AbstractAtomicSkill) -> None:
        self._values = {str(item.ref): item for item in atomics}

    def get_atomic(self, ref: SkillRef | str) -> AbstractAtomicSkill:
        return self._values[str(ref)]


def test_entity_discovered_at_retains_location_as_historical_clue() -> None:
    memory = ExplorationMemory()
    memory.observe_catalog(
        [{
            "action_id": "r002_a001",
            "arguments": {"object": "pencil_3", "source": "desk_2"},
        }],
        revision=2,
        current_facts=[{
            "predicate": "entity.discovered_at",
            "args": {"entity": "pencil_3", "location": "desk_2"},
        }],
    )

    observed = memory.policy_view()["observed_discoveries"]["pencil_3"]
    assert observed["last_known_location"] == "desk_2"
    assert observed["observed_at_revision"] == 2
    assert observed["source_kind"] == "public_action_catalog"
    assert observed["location_evidence_status"] == "observed"

    memory.observe_catalog(
        [{"arguments": {"entity": "pencil_3"}}],
        revision=9,
        current_facts=[],
    )
    historical = memory.policy_view()["historical_discoveries"]["pencil_3"]
    assert historical["last_known_location"] == "desk_2"
    assert historical["observed_at_revision"] == 2
    assert historical["location_evidence_status"] == "historical"


def test_support_projection_has_separate_input_output_contracts_and_availability() -> None:
    blocked = _atomic(
        "blocked",
        inputs=[ParameterSpec("location", "location")],
    )
    producer = _atomic(
        "producer",
        inputs=[ParameterSpec("destination", "location")],
        outputs=[ParameterSpec("location", "location")],
        effects=[SemanticPredicate(
            "entity.discovered_at",
            {"entity": "target", "location": _input("location")},
            effect_domain="evidence",
        )],
    )
    retriever = SupportAtomicRetriever()

    candidates = retriever.retrieve(
        blocked_atomic=blocked,
        missing_roles=["location"],
        atomics=[producer],
        execution_availability={str(producer.ref): True},
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.inputs == ({
        "name": "destination",
        "semantic_type": "location",
        "required": True,
        "runtime_resolvable": False,
        "required_resolution": "semantic",
    },)
    assert candidate.outputs == ({
        "name": "location",
        "semantic_type": "location",
        "required": True,
    },)
    assert candidate.execution_available is True
    assert candidate.missing_required_inputs == ("destination",)
    assert retriever.retrieve(
        blocked_atomic=blocked,
        missing_roles=["location"],
        atomics=[producer],
        execution_availability={str(producer.ref): False},
    ) == []

    schema = NodeExecutor._support_tool(candidates).input_schema
    argument_schema = schema["properties"]["arguments"]
    assert set(argument_schema["properties"]) == {"destination"}
    assert argument_schema["additionalProperties"] is False


def test_support_retriever_exposes_untruncated_formal_candidate_set() -> None:
    blocked = _atomic(
        "blocked_many_supports",
        inputs=[ParameterSpec("location", "location")],
    )
    producers = [
        _atomic(
            f"producer_{index}",
            outputs=[ParameterSpec("location", "location")],
        )
        for index in range(5)
    ]
    availability = {str(item.ref): True for item in producers}
    retriever = SupportAtomicRetriever()

    all_candidates = retriever.retrieve(
        blocked_atomic=blocked,
        missing_roles=["location"],
        atomics=producers,
        execution_availability=availability,
        top_k=None,
    )
    displayed_candidates = retriever.retrieve(
        blocked_atomic=blocked,
        missing_roles=["location"],
        atomics=producers,
        execution_availability=availability,
    )

    assert len(all_candidates) == 5
    assert displayed_candidates == all_candidates[:3]


def test_frozen_support_uses_compiler_mode_and_filters_candidate_only_atomic() -> None:
    blocked = _atomic(
        "blocked_frozen",
        inputs=[ParameterSpec("location", "location")],
    )
    producer = _atomic(
        "candidate_support",
        outputs=[ParameterSpec("location", "location")],
    )
    producer.status = SkillStatus.CANDIDATE

    class Skills:
        @staticmethod
        def atomics(*, mode=None):
            values = [blocked, producer]
            if mode is None:
                return values
            return [
                item for item in values
                if skill_status_usable(item.status, mode)
            ]

        @staticmethod
        def implementations_for(_ref, *, mode):
            RuntimeMode(mode)
            return []

    compiler = SimpleNamespace(
        skills=Skills(),
        mode=RuntimeMode.FROZEN,
    )
    executor = NodeExecutor(compiler, ValidationEngine(), lambda *_args: None)
    ctx = SimpleNamespace(
        trace_builder=SimpleNamespace(
            trace=SimpleNamespace(metadata={}),
        ),
    )

    candidates = executor._retrieve_runtime_support_candidates(
        blocked_atomic=blocked,
        missing_roles=["location"],
        ctx=ctx,
    )

    assert candidates == []
    funnel = ctx.trace_builder.trace.metadata["runtime_support_funnel"]
    assert funnel["retrieved_count"] == 1
    assert funnel["filtered_by_mode_count"] == 1


def test_dataflow_obligation_projects_supported_and_unknown_public_relation() -> None:
    navigate = _atomic(
        "navigate",
        inputs=[ParameterSpec("destination", "location")],
        outputs=[ParameterSpec("destination", "location")],
        effects=[SemanticPredicate(
            "agent.at_location", {"location": _input("destination")},
        )],
    )
    take = _atomic(
        "take",
        inputs=[
            ParameterSpec("object", "object"),
            ParameterSpec("source", "location"),
        ],
        preconditions=[SemanticPredicate(
            "object.at_location",
            {"object": _input("object"), "location": _input("source")},
        )],
    )
    nav_occurrence = RuntimeOccurrence(
        "navigate",
        "occ_nav",
        navigate.ref,
        [],
        {},
        [],
        list(navigate.effects),
    )
    take_occurrence = RuntimeOccurrence(
        "take",
        "occ_take",
        take.ref,
        [],
        {
            "object": _input("task_object"),
            "source": BindingExpression(
                BindingExprKind.DATA_FLOW,
                source_step="navigate",
                source_role="destination",
            ),
        },
        [],
        [],
    )
    plan = RuntimeLinearPlan(
        task_id="r92_obligation",
        source="stored_composite",
        source_composite_ref=None,
        occurrences=[nav_occurrence, take_occurrence],
        control_sequence=["navigate", "take"],
        data_edges=[GraphEdge(
            "nav_take",
            GraphEdgeType.DATA_FLOW,
            "navigate",
            "take",
            "destination",
            "source",
        )],
        dependency_edges=[],
        task_contract=TaskContract(),
        planner_audit={},
    )
    bindings = RuntimeBindingStore()
    bindings.seed_task_bindings(
        SimpleNamespace(
            task_id="r92_obligation",
            context={"semantic_bindings": {"task_object": "pencil_3"}},
        ),
        TaskContract(),
        0,
    )
    bindings.resolve_occurrence_specs(take_occurrence, 0)
    bindings.publish_validated_outputs(
        nav_occurrence,
        {"destination": "desk_2"},
        ["validator:nav"],
        2,
    )
    builder = RuntimePlanContextBuilder(_Registry(navigate, take))

    supported = builder.build(
        plan,
        "navigate",
        bindings,
        public_facts=[{
            "predicate": "object.at_location",
            "args": {"object": "pencil_3", "location": "desk_2"},
            "public_evidence_ref": "action_catalog:take_pencil:revision:2",
        }],
        public_revision=2,
    ).policy_view()["output_obligations"][0]
    assert supported["relation_predicate"] == "object.at_location"
    assert supported["effect_domain"] == "world"
    assert supported["relevant_anchor_roles"] == ["object"]
    assert supported["public_relation_status"] == "supported"
    assert supported["public_evidence_refs"]

    unknown = builder.build(
        plan,
        "navigate",
        bindings,
        public_facts=[],
        public_revision=2,
    ).policy_view()["output_obligations"][0]
    assert unknown["public_relation_status"] == "unknown"
    assert unknown["public_evidence_refs"] == []


def test_output_obligation_selects_later_publicly_supported_predicate() -> None:
    consumer = _atomic(
        "consumer_with_multiple_relations",
        inputs=[
            ParameterSpec("object", "object"),
            ParameterSpec("source", "location"),
        ],
        preconditions=[
            SemanticPredicate(
                "entity.discovered_at",
                {"entity": _input("object"), "location": _input("source")},
                effect_domain="evidence",
            ),
            SemanticPredicate(
                "object.at_location",
                {"object": _input("object"), "location": _input("source")},
            ),
        ],
    )

    assessment = assess_output_obligation(
        consumer_atomic=consumer,
        consumer_input_role="source",
        known_anchors={
            "object": {
                "value": "pencil_3",
                "semantic_type": "object",
                "source": "task",
            },
        },
        producer_output_value="desk_2",
        public_facts=[{
            "predicate": "object.at_location",
            "args": {"object": "pencil_3", "location": "desk_2"},
            "observed_at_revision": 7,
            "public_evidence_ref": "action_catalog:take_pencil:revision:7",
        }],
        revision=7,
    )

    assert assessment.relation_predicate == "object.at_location"
    assert assessment.public_relation_status == "supported"


def test_output_obligation_contradiction_requires_current_exact_object() -> None:
    consumer = _atomic(
        "consumer_exact_relation",
        inputs=[
            ParameterSpec("object", "object"),
            ParameterSpec("source", "location"),
        ],
        preconditions=[SemanticPredicate(
            "object.at_location",
            {"object": _input("object"), "location": _input("source")},
        )],
    )
    anchor = {
        "object": {
            "value": "pencil_3",
            "semantic_type": "object",
            "source": "task",
        },
    }

    contradicted = assess_output_obligation(
        consumer_atomic=consumer,
        consumer_input_role="source",
        known_anchors=anchor,
        producer_output_value="desk_1",
        public_facts=[{
            "predicate": "object.at_location",
            "args": {"object": "pencil_3", "location": "desk_2"},
            "observed_at_revision": 8,
            "public_evidence_ref": "action_catalog:take_pencil:revision:8",
        }],
        revision=8,
    )
    assert contradicted.public_relation_status == "contradicted"
    assert contradicted.public_evidence_refs

    for fact, revision in (
        ({
            "predicate": "object.at_location",
            "args": {"object": "pencil_3", "location": "desk_2"},
            "observed_at_revision": 7,
            "public_evidence_ref": "action_catalog:stale:revision:7",
        }, 8),
        ({
            "predicate": "object.at_location",
            "args": {"object": "pencil_4", "location": "desk_2"},
            "observed_at_revision": 8,
            "public_evidence_ref": "action_catalog:other:revision:8",
        }, 8),
    ):
        unknown = assess_output_obligation(
            consumer_atomic=consumer,
            consumer_input_role="source",
            known_anchors=anchor,
            producer_output_value="desk_1",
            public_facts=[fact],
            revision=revision,
        )
        assert unknown.public_relation_status == "unknown"
        assert unknown.public_evidence_refs == ()

    class_anchor = {
        "object": {**anchor["object"], "value": "pencil"},
    }
    class_unknown = assess_output_obligation(
        consumer_atomic=consumer,
        consumer_input_role="source",
        known_anchors=class_anchor,
        producer_output_value="desk_1",
        public_facts=[{
            "predicate": "object.at_location",
            "args": {"object": "pencil_3", "location": "desk_2"},
            "observed_at_revision": 8,
            "public_evidence_ref": "action_catalog:instance:revision:8",
        }],
        revision=8,
    )
    assert class_unknown.public_relation_status == "unknown"

    nonformal_unknown = assess_output_obligation(
        consumer_atomic=consumer,
        consumer_input_role="source",
        known_anchors={
            "object": {**anchor["object"], "source": "agent_proposed"},
        },
        producer_output_value="desk_1",
        public_facts=[{
            "predicate": "object.at_location",
            "args": {"object": "pencil_3", "location": "desk_2"},
            "observed_at_revision": 8,
            "public_evidence_ref": "action_catalog:untrusted:revision:8",
        }],
        revision=8,
    )
    assert nonformal_unknown.public_relation_status == "unknown"


def test_historical_discovery_never_becomes_location_contradiction() -> None:
    consumer = _atomic(
        "consumer_historical_relation",
        inputs=[
            ParameterSpec("object", "object"),
            ParameterSpec("source", "location"),
        ],
        preconditions=[SemanticPredicate(
            "entity.discovered_at",
            {"entity": _input("object"), "location": _input("source")},
            effect_domain="evidence",
        )],
    )

    assessment = assess_output_obligation(
        consumer_atomic=consumer,
        consumer_input_role="source",
        known_anchors={
            "object": {
                "value": "pencil_3",
                "semantic_type": "object",
                "source": "task",
            },
        },
        producer_output_value="desk_1",
        public_facts=[{
            "predicate": "entity.discovered_at",
            "args": {"entity": "pencil_3", "location": "desk_2"},
            "observed_at_revision": 8,
            "public_evidence_ref": "action_catalog:discovery:revision:8",
        }],
        revision=8,
    )

    assert assessment.public_relation_status == "unknown"
    assert assessment.public_evidence_refs == ()


def test_downstream_relation_never_projects_validator_only_truth() -> None:
    navigate = _atomic(
        "navigate_hidden_boundary",
        inputs=[ParameterSpec("destination", "location")],
        outputs=[ParameterSpec("destination", "location")],
        effects=[SemanticPredicate(
            "agent.at_location", {"location": _input("destination")},
        )],
    )
    take = _atomic(
        "take_hidden_boundary",
        inputs=[
            ParameterSpec("object", "object"),
            ParameterSpec("source", "location"),
        ],
        preconditions=[SemanticPredicate(
            "object.at_location",
            {"object": _input("object"), "location": _input("source")},
        )],
    )
    nav_occurrence = RuntimeOccurrence(
        "navigate", "occ_nav_hidden", navigate.ref, [], {}, [],
        list(navigate.effects),
    )
    take_occurrence = RuntimeOccurrence(
        "take", "occ_take_hidden", take.ref, [],
        {
            "object": _input("task_object"),
            "source": BindingExpression(
                BindingExprKind.DATA_FLOW,
                source_step="navigate",
                source_role="destination",
            ),
        },
        [], [],
    )
    plan = RuntimeLinearPlan(
        task_id="r92_hidden_boundary",
        source="stored_composite",
        source_composite_ref=None,
        occurrences=[nav_occurrence, take_occurrence],
        control_sequence=["navigate", "take"],
        data_edges=[GraphEdge(
            "nav_take_hidden", GraphEdgeType.DATA_FLOW,
            "navigate", "take", "destination", "source",
        )],
        dependency_edges=[],
        task_contract=TaskContract(),
        planner_audit={},
    )
    bindings = RuntimeBindingStore()
    bindings.seed_task_bindings(
        SimpleNamespace(
            task_id="r92_hidden_boundary",
            context={"semantic_bindings": {"task_object": "pencil_3"}},
        ),
        TaskContract(),
        0,
    )
    bindings.resolve_occurrence_specs(take_occurrence, 0)
    bindings.publish_validated_outputs(
        nav_occurrence,
        {"destination": "desk_2"},
        ["validator:nav"],
        2,
    )
    executor = NodeExecutor.__new__(NodeExecutor)
    executor.plan_context_builder = RuntimePlanContextBuilder(
        _Registry(navigate, take)
    )
    hidden_fact = {
        "predicate": "object.at_location",
        "args": {"object": "pencil_3", "location": "desk_2"},
    }
    harness = SimpleNamespace(
        validator_channel=lambda: SimpleNamespace(
            snapshot=lambda: {"facts": [hidden_fact]},
        ),
    )
    ctx = SimpleNamespace(
        plan=plan,
        binding_store=bindings,
        harness=harness,
        world_revision=2,
        tool_evidence_snapshot=lambda: {"semantic_facts": [hidden_fact]},
    )

    hidden = executor._downstream_plan_context(ctx, nav_occurrence)
    assert hidden["output_obligations"][0]["public_relation_status"] == (
        "unknown"
    )

    harness.public_runtime_relation_facts = lambda: [{
        **hidden_fact,
        "public_evidence_ref": "action_catalog:take_pencil:revision:2",
    }]
    public = executor._downstream_plan_context(ctx, nav_occurrence)
    obligation = public["output_obligations"][0]
    assert obligation["public_relation_status"] == "supported"
    assert obligation["public_evidence_refs"] == [
        "action_catalog:take_pencil:revision:2"
    ]


def test_alfworld_public_relation_projection_uses_only_take_catalog() -> None:
    adapter = AlfWorldAdapter(split="train")
    adapter._catalog.replace(
        [
            "take pencil 3 from desk 2",
            "go to bed 1",
        ],
        revision=4,
    )
    adapter._validator.snapshot = lambda: {
        "facts": [{
            "predicate": "object.at_location",
            "args": {
                "object": "hidden_object_99",
                "location": "hidden_location_99",
                "private_marker": "must_not_cross",
            },
        }],
    }

    facts = adapter.public_runtime_relation_facts()

    assert {item["predicate"] for item in facts} == {
        "entity.discovered_at", "object.at_location",
    }
    assert {
        tuple(sorted(item["args"].items())) for item in facts
    } == {
        (("entity", "pencil_3"), ("location", "desk_2")),
        (("location", "desk_2"), ("object", "pencil_3")),
    }
    assert all(
        item["public_evidence_ref"] == "action_catalog:r004_a001:revision:4"
        for item in facts
    )
    assert all(item["observed_at_revision"] == 4 for item in facts)
    assert all(item["source_kind"] == "public_action_catalog" for item in facts)
    assert all(item["evidence_status"] == "observed" for item in facts)
    assert "hidden_object_99" not in repr(facts)
    assert "private_marker" not in repr(facts)


def test_discovery_location_survives_sessions_but_is_not_current_authority() -> None:
    memory = ExplorationMemory()
    memory.observe_catalog(
        [{
            "action_id": "r002_a001",
            "arguments": {"object": "pencil_3", "source": "desk_2"},
        }],
        revision=2,
        current_facts=[{
            "predicate": "entity.discovered_at",
            "args": {"entity": "pencil_3", "location": "desk_2"},
        }],
    )

    # A later session/revision can still use the clue, but absence from the
    # current public facts demotes it instead of preserving current authority.
    memory.observe_catalog([], revision=10, current_facts=[])
    view = memory.policy_view()
    assert "pencil_3" not in view["observed_discoveries"]
    clue = view["historical_discoveries"]["pencil_3"]
    assert clue["last_known_location"] == "desk_2"
    assert clue["observed_at_revision"] == 2
    assert clue["location_evidence_status"] == "historical"

    # A new public relation supersedes the old clue without rewriting history
    # into an unsupported absence claim.
    memory.record_action(
        {
            "accepted": True,
            "action_id": "r012_a004",
            "action_type": "PUT",
            "arguments": {"object": "pencil_3", "destination": "shelf_1"},
        },
        metadata={},
        catalog=[],
        revision=12,
        current_facts=[{
            "predicate": "object.at_location",
            "args": {"object": "pencil_3", "location": "shelf_1"},
        }],
    )
    current = memory.policy_view()["observed_discoveries"]["pencil_3"]
    assert current["last_known_location"] == "shelf_1"
    assert current["observed_at_revision"] == 12
    assert current["location_evidence_status"] == "observed"


def test_private_validator_location_without_public_evidence_is_not_projected() -> None:
    memory = ExplorationMemory()
    memory.observe_catalog(
        [],
        revision=4,
        current_facts=[{
            "predicate": "entity.discovered_at",
            "args": {"entity": "hidden_key_9", "location": "safe_1"},
        }],
    )

    view = memory.policy_view()
    assert "hidden_key_9" not in view["observed_discoveries"]
    assert "hidden_key_9" not in view["historical_discoveries"]


class _SupportCallContext(SimpleNamespace):
    def begin_occurrence(self, occurrence) -> None:
        self.active_occurrence_id = str(occurrence.occurrence_id)


class _SupportCallCompiler:
    mode = RuntimeMode.ONLINE

    def __init__(self, support_atomic, *, executable: bool, preflight_passed: bool):
        implementation = SimpleNamespace(ref=SkillRef("impl_support", "1.0.0"))
        self.skills = SimpleNamespace(
            get_atomic=lambda _ref: support_atomic,
            implementations_for=lambda _ref, *, mode: [implementation],
        )
        self.compiled = SimpleNamespace(
            atomic=support_atomic, implementation=implementation, tools=[],
            spec=SimpleNamespace(
                name="invoke_impl_support",
                input_schema={
                    "type": "object",
                    "properties": {"destination": {"type": "string"}},
                    "required": ["destination"],
                    "additionalProperties": False,
                },
            ),
        )
        self.executable = executable
        self.preflight_passed = preflight_passed

    def compile_candidates(self, *_args, **_kwargs):
        return [self.compiled] if self.executable else []

    def prepare_arguments(self, *_args, arguments, **_kwargs):
        if self.preflight_passed:
            return ToolCallPreflightResult(
                True,
                "impl_support@1.0.0",
                normalized_arguments=dict(arguments),
            )
        return ToolCallPreflightResult(
            False,
            "impl_support@1.0.0",
            failure_layer="runtime_binding",
            failure_code="runtime_binding_unresolved",
            message="required binding unresolved: destination",
        )

    def validate_execution_context(self, _compiled, prepared, **_kwargs):
        return prepared


class _SupportCallRunner:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls = 0
        self.error = error
        self.result = None
        self.on_run = None

    def run(self, *_args, **_kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        if callable(self.on_run):
            self.on_run()
        if self.result is not None:
            return self.result
        raise AssertionError("focused rejection tests must not execute support")


def _support_call_fixture(*, executable: bool, preflight_passed: bool):
    support = _atomic(
        "support",
        inputs=[ParameterSpec("destination", "location")],
        outputs=[ParameterSpec("location", "location")],
        effects=[SemanticPredicate(
            "entity.discovered_at",
            {"entity": "target", "location": _input("location")},
            effect_domain="evidence",
        )],
    )
    blocked = _atomic(
        "blocked_for_support",
        inputs=[ParameterSpec("source", "location")],
        preconditions=[SemanticPredicate(
            "object.at_location",
            {"object": "target", "location": _input("source")},
        )],
    )
    support.validator_spec = {"output_derivations": {"location": {"kind": "input_identity", "input_role": "destination"}}}
    candidate = SupportAtomicRetriever().retrieve(
        blocked_atomic=blocked,
        missing_roles=["source"],
        atomics=[support],
        execution_availability={str(support.ref): True},
    )[0]
    parent = RuntimeOccurrence(
        "blocked_step",
        "blocked_occurrence",
        blocked.ref,
        [],
        {},
        [],
        [],
    )
    compiler = _SupportCallCompiler(
        support,
        executable=executable,
        preflight_passed=preflight_passed,
    )
    runner = _SupportCallRunner()
    executor = NodeExecutor.__new__(NodeExecutor)
    executor.invocation_compiler = compiler
    executor.implementation_runner = runner
    # These tests isolate selected-Support preflight/transfer. Effect-first
    # resolution is covered with the real harness in the R10/R10.1 suites.
    executor._complete_from_current_effect = lambda *_args, **_kwargs: None
    executor._augment_runtime_payload = lambda *_args, **_kwargs: None
    executor._record_control_call = lambda *_args, **_kwargs: None
    executor._activate_occurrence_state = (
        lambda occurrence, _atomic, _invocations, ctx, **_kwargs:
        ctx.begin_occurrence(occurrence)
    )
    ctx = _SupportCallContext(
        task_id="support_task",
        task_contract=TaskContract(),
        world_revision=0,
        binding_store=RuntimeBindingStore(),
        budget=SimpleNamespace(current_occurrence_id=parent.occurrence_id),
        evidence_store=SimpleNamespace(),
        validated_outputs={},
        active_occurrence_id=parent.occurrence_id,
        last_failed_invocation=None,
        _after_action_refresh=None,
        trace_builder=SimpleNamespace(
            trace=SimpleNamespace(metadata={}),
        ),
    )
    from atomic_skillgraph.core.bindings import RuntimeBinding, BindingSource, BindingStatus, BindingResolution
    ctx.binding_store.commit_grounded(parent.occurrence_id, {"source": RuntimeBinding(
        "source", "desk_2", "location", BindingSource.DATA_FLOW, BindingStatus.GROUNDED,
        BindingResolution.CONCRETE, ["validated:prior-node"], 0)})
    session = SimpleNamespace(session_id="support_session")
    return executor, runner, ctx, session, parent, blocked, candidate


def _support_call(candidate, *, arguments, support_ref=None):
    return SimpleNamespace(
        call_id="support_call",
        arguments={
            "support_atomic_ref": support_ref or str(candidate.atomic_ref),
            "arguments": dict(arguments),
        },
    )


def test_unknown_or_missing_support_input_never_starts_execution() -> None:
    executor, runner, ctx, session, parent, blocked, candidate = (
        _support_call_fixture(executable=True, preflight_passed=False)
    )
    unknown = executor._invoke_support_atomic_call(
        _support_call(
            candidate,
            arguments={"destination": "desk_2"},
            support_ref="unknown_support@1.0.0",
        ),
        session,
        parent,
        ctx,
        blocked,
        [candidate],
    )
    assert unknown["error"] == "runtime_support_candidate_invalid"
    assert runner.calls == 0

    missing = executor._invoke_support_atomic_call(
        _support_call(candidate, arguments={}),
        session,
        parent,
        ctx,
        blocked,
        [candidate],
    )
    assert missing["error"] == "support_not_execution_ready"
    assert missing["missing_required_inputs"] == ["destination"]
    assert runner.calls == 0


def test_support_no_executable_and_not_ready_have_distinct_results() -> None:
    executor, runner, ctx, session, parent, blocked, candidate = (
        _support_call_fixture(executable=False, preflight_passed=False)
    )
    unavailable = executor._invoke_support_atomic_call(
        _support_call(candidate, arguments={"destination": "desk_2"}),
        session,
        parent,
        ctx,
        blocked,
        [candidate],
    )
    assert unavailable["error"] == "runtime_support_no_executable"
    assert runner.calls == 0

    executor, runner, ctx, session, parent, blocked, candidate = (
        _support_call_fixture(executable=True, preflight_passed=False)
    )
    not_ready = executor._invoke_support_atomic_call(
        _support_call(candidate, arguments={"destination": "desk_2"}),
        session,
        parent,
        ctx,
        blocked,
        [candidate],
    )
    assert not_ready["error"] == "support_not_execution_ready"
    assert not_ready["preflight_failure_code"] == "runtime_binding_unresolved"
    assert runner.calls == 0


def test_support_exception_restores_parent_occurrence_and_refresh_callback() -> None:
    executor, runner, ctx, session, parent, blocked, candidate = (
        _support_call_fixture(executable=True, preflight_passed=True)
    )
    runner.error = RuntimeError("support execution failed")
    refreshes = []
    ctx._after_action_refresh = lambda: refreshes.append(ctx.active_occurrence_id)
    ctx.last_failed_invocation = {
        "occurrence_id": parent.occurrence_id,
        "failure_code": "parent_failure",
    }

    with pytest.raises(RuntimeError, match="support execution failed"):
        executor._invoke_support_atomic_call(
            _support_call(candidate, arguments={"destination": "desk_2"}),
            session,
            parent,
            ctx,
            blocked,
            [candidate],
        )

    assert runner.calls == 1
    assert ctx.active_occurrence_id == parent.occurrence_id
    assert refreshes == [parent.occurrence_id]
    assert ctx.last_failed_invocation["failure_code"] == "parent_failure"


def test_support_execution_started_funnel_requires_real_start() -> None:
    executor, runner, ctx, session, parent, blocked, candidate = (
        _support_call_fixture(executable=True, preflight_passed=True)
    )
    runner.result = ImplementationExecutionResult(
        "skill://impl_support@1.0.0",
        str(candidate.atomic_ref),
        True,
        False,
        False,
        False,
    )

    payload = executor._invoke_support_atomic_call(
        _support_call(candidate, arguments={"destination": "desk_2"}),
        session,
        parent,
        ctx,
        blocked,
        [candidate],
    )

    assert payload["passed"] is False
    assert ctx.trace_builder.trace.metadata["runtime_support_funnel"].get(
        "execution_started_count", 0,
    ) == 0


def test_support_success_publishes_to_parent_and_refreshes_after_revision() -> None:
    executor, runner, ctx, session, parent, blocked, candidate = (
        _support_call_fixture(executable=True, preflight_passed=True)
    )
    published = []
    evidence = []
    refreshes = []
    retrievals = []
    ctx.binding_store.publish_validated_outputs = lambda *args: published.append(args)
    ctx.binding_store.runtime_prompt_projection = lambda *_args, **_kwargs: {
        "missing_or_insufficient_bindings": ["source"] if ctx.world_revision == 0 else []}
    ctx.evidence_store = SimpleNamespace(
        add_validated_tool_output=lambda *args: evidence.append(args),
    )
    ctx.trace_builder.trace.validations = []
    ctx._after_action_refresh = lambda: refreshes.append(
        ctx.active_occurrence_id
    )
    runner.on_run = lambda: setattr(ctx, "world_revision", 1)
    runner.result = ImplementationExecutionResult(
        "skill://impl_support@1.0.0",
        str(candidate.atomic_ref),
        True,
        True,
        True,
        True,
        validated_outputs={"location": "desk_2"},
    )

    payload = executor._invoke_support_atomic_call(
        _support_call(candidate, arguments={"destination": "desk_2"}),
        session,
        parent,
        ctx,
        blocked,
        [candidate],
    )

    assert payload["passed"] is True
    assert payload["new_revision"] == 1
    assert runner.calls == 1
    assert ctx.active_occurrence_id == parent.occurrence_id
    assert refreshes == [parent.occurrence_id]
    # R10.1 changes the contract: Support supplies INPUT, never parent OUTPUT.
    assert published == []
    assert ctx.binding_store.snapshot_for_node(parent)["source"].value == "desk_2"
    assert ctx.binding_store.snapshot_for_node(parent)["source"].world_revision == 1
    assert parent.occurrence_id not in ctx.validated_outputs
    assert evidence == []
    assert ctx.trace_builder.trace.metadata["runtime_support_funnel"][
        "validated_output_published_count"
    ] == 1

    executor._retrieve_runtime_support_candidates = lambda **kwargs: (
        retrievals.append(kwargs) or []
    )
    refreshed, state = executor._refresh_runtime_support_candidates(
        blocked_atomic=blocked,
        occurrence=parent,
        ctx=ctx,
        previous_state=(0, ("source",)),
        current_candidates=[candidate],
    )
    assert refreshed == []
    assert state == (1, ())
    assert retrievals[0]["missing_roles"] == []
