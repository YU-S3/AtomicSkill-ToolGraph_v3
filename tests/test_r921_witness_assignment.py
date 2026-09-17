"""Concrete effect witnesses are validation-local, never input mutations."""
import copy
from types import SimpleNamespace

import pytest

from atomic_skillgraph.core.contracts import AbstractAtomicSkill, ParameterSpec, SemanticPredicate
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.results import RuntimeOccurrence
from atomic_skillgraph.validation.atomic_validator import AtomicValidator
from experiments.self_tooling_targeted import CandidateHarness, RouteCase, fixture_task


@pytest.mark.parametrize("accepted,target,passed", [
    (True, "cabinet_1", True), (False, "cabinet_1", False),
    (True, "cabinet_2", False),
])
def test_resolved_step_local_is_not_rebound_by_predicate_argument_name(accepted, target, passed):
    from atomic_skillgraph.harness.alfworld import AlfWorldValidatorChannel, parse_alfworld_action
    from atomic_skillgraph.harness.action_catalog import HarnessActionCatalog
    from atomic_skillgraph.runtime.tool_runner import ToolRunner
    from atomic_skillgraph.tooling.ir import ToolExecutionState
    from atomic_skillgraph.validation.tool_validator import ToolValidator
    channel = AlfWorldValidatorChannel()
    catalog = HarnessActionCatalog(parse_alfworld_action)
    action = catalog.replace(["examine cabinet 1"], 0)[0]
    channel.record(action, accepted=accepted, revision=1, done=False, won=False)
    state = ToolExecutionState(bindings={"object": "egg"}, local={"examine_target": target})
    ctx = SimpleNamespace(harness=SimpleNamespace(validator_channel=lambda: channel))
    tool = SimpleNamespace(signature={"properties": {"object": {}}}, interface={})
    before = copy.deepcopy((state.bindings, state.local, channel.snapshot()))
    result = ToolRunner(ToolValidator())._validate_step_effects({
        "node_id": "examine_object", "expected_effects": [{
            "predicate": "object.observed", "effect_domain": "evidence",
            "args": {"object": {"kind": "local_variable", "source_role": "examine_target"}},
        }],
    }, ctx, state, tool=tool)
    assert result["step_effect_passed"] is passed
    assert (state.bindings, state.local, channel.snapshot()) == before


def discovered():
    case = RouteCase("witness", target="apple")
    harness = CandidateHarness(case)
    harness.reset(fixture_task(case))
    for action_type in ("GO_TO", "OPEN"):
        spec = next(a for a in harness.action_catalog() if a.action_type == action_type
                    and "cabinet_1" in a.arguments.values())
        harness.execute_action(spec.action_id, spec.revision)
    atomic = AbstractAtomicSkill(SkillRef("generic_discovery", "1.0.0"), "locate target",
        [ParameterSpec("query", "entity")], [ParameterSpec("place", "entity", required_resolution="concrete")], [],
        [SemanticPredicate("entity.discovered_at", {"entity": "$query", "location": "$place"}, effect_domain="evidence")],
        {"output_derivations": {"place": {"kind": "effect_witness", "predicate": "entity.discovered_at",
                                         "argument_role": "location"}}}, [], {}, {})
    occurrence = RuntimeOccurrence("parent", "parent", atomic.ref, [], {}, [], atomic.effects)
    return harness, atomic, occurrence


@pytest.mark.parametrize("target,location,revision,passed", [
    ("apple", "cabinet_1", 2, True), ("apple_1", "cabinet_1", 2, True),
    ("apple_2", "cabinet_1", 2, False), ("cup", "cabinet_1", 2, False),
    ("apple", "countertop_2", 2, False), ("apple", "cabinet_1", 1, False),
])
def test_real_witness_must_match_original_input_output_and_revision(target, location, revision, passed):
    harness, atomic, occurrence = discovered()
    inputs, outputs = {"query": target}, {"place": location}
    before = copy.deepcopy((inputs, outputs, harness.validator_channel().snapshot()))
    result = AtomicValidator().validate_execution_result(atomic, occurrence, inputs, outputs,
        harness.validator_channel(), current_revision=revision)
    assert result.passed is passed, result
    assert (inputs, outputs, harness.validator_channel().snapshot()) == before


def test_semantic_input_identity_cannot_be_promoted_into_concrete_output():
    harness, atomic, occurrence = discovered()
    atomic.outputs.append(ParameterSpec("query", "entity"))
    atomic.validator_spec["output_derivations"]["query"] = {"kind": "input_identity", "input_role": "query"}
    result = AtomicValidator().validate_execution_result(atomic, occurrence, {"query": "apple"},
        {"place": "cabinet_1", "query": "apple_1"}, harness.validator_channel(), current_revision=2)
    assert not result.passed
    assert "atomic_output_identity_mismatch" in result.failure_codes


def test_semantic_assignment_without_witness_is_rejected():
    harness, atomic, occurrence = discovered()
    authority = harness.validator_channel()

    class MissingWitnessChannel:
        def resolve_atomic_effect(self, request):
            result = authority.resolve_atomic_effect(request)
            result.witness_refs.clear()
            return result

        def validate_atomic_effect(self, request):
            raise AssertionError("unwitnessed assignment must be rejected before final validation")

    result = AtomicValidator().validate_execution_result(atomic, occurrence, {"query": "apple"},
        {"place": "cabinet_1"}, MissingWitnessChannel(), current_revision=2)
    assert not result.passed
    assert result.failure_codes == ["atomic_effect_witness_missing"]


def test_public_relation_schema_describes_existing_projection_without_episode_state():
    harness, _, _ = discovered()
    schema = harness.public_catalog_relation_schema()
    before = copy.deepcopy(schema)
    relation = schema[0]
    actions = [a for a in harness.action_catalog() if a.action_type == relation["action_type"]]
    facts = harness.public_runtime_relation_facts()
    assert facts
    for action in actions:
        for projection in relation["predicates"]:
            args = {role: action.arguments[source] for role, source in projection["argument_mapping"].items()}
            assert any(fact["predicate"] == projection["predicate"] and fact["args"] == args for fact in facts)
    move = next(a for a in harness.action_catalog() if a.action_type == "GO_TO" and
                a.arguments["destination"] == "countertop_1")
    harness.execute_action(move.action_id, move.revision)
    assert harness.public_runtime_relation_facts() == []
    assert harness.public_catalog_relation_schema() == before
    schema[0]["predicates"].clear()
    assert harness.public_catalog_relation_schema() == before


def test_explicit_fresh_output_is_checked_against_joint_fact_not_unused_input():
    """R10.2 validates the submitted value; an unused input is not an implicit constraint."""
    class SharedLocationHarness(CandidateHarness):
        def _replace_catalog(self):
            catalog = super()._replace_catalog()
            if self.opened and not self.held:
                raw = [item.display_text for item in catalog]
                raw.append("take cup 1 from " + self.location.replace("_", " "))
                return self._catalog.replace(raw, self._revision)
            return catalog

    case = RouteCase("ambiguous_fresh_output", target="apple")
    harness = SharedLocationHarness(case)
    harness.reset(fixture_task(case))
    for action_type in ("GO_TO", "OPEN"):
        action = next(a for a in harness.action_catalog() if a.action_type == action_type
                      and "cabinet_1" in a.arguments.values())
        harness.execute_action(action.action_id, action.revision)
    _, atomic, occurrence = discovered()
    ambiguous = copy.deepcopy(atomic)
    ambiguous.outputs.append(ParameterSpec("found", "entity", required_resolution="concrete"))
    ambiguous.effects[0].args["entity"] = "$found"
    ambiguous.validator_spec["output_derivations"]["found"] = {
        "kind": "effect_witness", "predicate": "entity.discovered_at", "argument_role": "entity"}
    inputs = {"query": "apple"}
    candidates = {"found": "apple_1", "place": "cabinet_1"}
    before = copy.deepcopy((inputs, candidates, harness.validator_channel().snapshot()))
    facts = harness.public_runtime_relation_facts()
    result = AtomicValidator().validate_execution_result(ambiguous, occurrence, inputs, candidates,
        harness.validator_channel(), current_revision=2, authoritative_evidence_facts=facts)
    assert result.passed
    forged = AtomicValidator().validate_execution_result(ambiguous, occurrence, inputs,
        {"found": "apple_99", "place": "cabinet_1"}, harness.validator_channel(),
        current_revision=2, authoritative_evidence_facts=facts)
    assert not forged.passed
    # Referencing the semantic input in the Effect gives the Harness the
    # intended constraint. No RETURN claim is promoted into witness authority.
    anchored = AtomicValidator().validate_execution_result(atomic, occurrence, inputs,
        {"place": "cabinet_1"}, harness.validator_channel(), current_revision=2,
        authoritative_evidence_facts=facts)
    assert anchored.passed, anchored
    assert (inputs, candidates, harness.validator_channel().snapshot()) == before
