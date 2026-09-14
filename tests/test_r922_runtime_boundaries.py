"""Explicit roles, witnessed concrete choices and revision-local traversal."""
import copy
import pytest

from atomic_skillgraph.core.bindings import BindingExpression, BindingExprKind
from atomic_skillgraph.core.contracts import ParameterSpec
from atomic_skillgraph.harness.alfworld import AlfWorldValidatorChannel
from atomic_skillgraph.validation.atomic_validator import AtomicValidator
from atomic_skillgraph.tooling.proposal import validate_output_semantic_constraints
from atomic_skillgraph.system import AtomicSkillGraphSystem
from experiments import self_tooling_targeted as route
from fixtures.r921_self_tooling_cases import fixture_config
from test_r921_witness_assignment import discovered


@pytest.mark.parametrize("value,expected", [
    ("cabinet_1", "cabinet_1"), ("object", "object"), ("$alias", "egg_1"),
    (BindingExpression(BindingExprKind.SKILL_INPUT, source_role="alias"), "egg_1"),
])
def test_predicate_argument_never_implies_lookup(value, expected):
    _, args, _, _ = AlfWorldValidatorChannel._expected_args(
        {"predicate": "object.observed", "args": {"object": value}},
        {"object": "egg", "alias": "egg_1"})
    assert args == {"object": expected}


@pytest.mark.parametrize("candidate,revision,passed", [
    ("apple_1", 2, True), ("apple_2", 2, True), ("apple_3", 2, False),
    ("cup_1", 2, False), ("apple_1", 1, False), ("apple", 2, False),
])
def test_explicit_concrete_output_filters_real_facts_without_creating_them(candidate, revision, passed):
    class MultiHarness(route.CandidateHarness):
        def _replace_catalog(self):
            catalog = super()._replace_catalog()
            if self.opened:
                return self._catalog.replace([a.display_text for a in catalog] + ["take apple 2 from cabinet 1"], self._revision)
            return catalog
    case = route.RouteCase("multi", target="apple")
    harness = MultiHarness(case)
    harness.reset(route.fixture_task(case))
    for action_type in ("GO_TO", "OPEN"):
        action = next(a for a in harness.action_catalog() if a.action_type == action_type and "cabinet_1" in a.arguments.values())
        harness.execute_action(action.action_id, action.revision)
    _, atomic, occurrence = discovered()
    atomic.outputs.append(ParameterSpec("found", "entity", required_resolution="concrete"))
    atomic.effects[0].args["entity"] = "$found"
    atomic.validator_spec["output_derivations"]["found"] = {"kind": "effect_witness", "predicate": "entity.discovered_at", "argument_role": "entity"}
    atomic.validator_spec["output_semantic_constraints"] = {"found": {"compatible_with_input": "query"}}
    original = copy.deepcopy(harness.validator_channel().snapshot())
    inputs = {"query": "apple"}
    result = AtomicValidator().validate_execution_result(atomic, occurrence, inputs,
        {"found": candidate, "place": "cabinet_1"}, harness.validator_channel(), current_revision=revision,
        semantic_compatible=harness.semantic_value_compatible, authoritative_evidence_facts=harness.public_runtime_relation_facts())
    assert result.passed is passed, result
    assert inputs == {"query": "apple"}
    assert harness.validator_channel().snapshot() == original


@pytest.mark.parametrize("constraints", [
    {"missing": {"compatible_with_input": "target"}},
    {"found": {"compatible_with_input": "missing"}},
    {"found": {"compatible_with_input": "target", "expression": "select_first"}},
])
def test_output_constraint_symbol_errors_fail_closed(constraints):
    with pytest.raises(ValueError):
        validate_output_semantic_constraints([ParameterSpec("target", "entity")],
            [ParameterSpec("found", "entity", required_resolution="concrete")], constraints)


@pytest.mark.parametrize("live,limit,passed", [(False, 3, False), (True, 3, True), (True, 2, False)])
def test_native_trial_traverses_current_candidates_and_keeps_snapshot_behavior(tmp_path, monkeypatch, live, limit, passed):
    class ChangingCatalog(route.CandidateHarness):
        def _replace_catalog(self):
            destinations = {0: ("countertop_1", "countertop_2", "shelf_1"),
                            1: ("shelf_1", "cabinet_1"), 2: ("cabinet_1",)}.get(self._revision, ())
            raw = ["go to " + value.replace("_", " ") for value in destinations]
            if self.location == "cabinet_1" and not self.held:
                raw.append("take egg 1 from cabinet 1")
            return self._catalog.replace(raw, self._revision)
    original = route.scripted_proposal
    def proposal(atomic, case):
        value = original(atomic, case)
        value["program"][0]["collection_source"]["refresh_each_iteration"] = live
        value["program"][0]["max_iterations"] = limit
        return value
    monkeypatch.setattr(route, "scripted_proposal", proposal)
    case = route.RouteCase("changing", openable=False)
    with AtomicSkillGraphSystem(fixture_config(tmp_path), harness=ChangingCatalog(case)) as system:
        outcome = route.run_node_case(system, case)
        draft = next(iter(outcome["ctx"].runtime_automation_drafts.values()))
        assert draft["r1_passed"] is passed, draft
        actions = outcome["trace"].environment_actions
        if passed:
            assert [a.arguments.get("destination") for a in actions[:-1]] == ["countertop_1", "shelf_1", "cabinet_1"]
            assert outcome["result"].atomic_effect_passed
        elif not live:
            assert len(actions) == 1
            assert draft["trial"]["result"]["failure_code"] == "tool_ir_action_unavailable"
        else:
            assert len(actions) == 2
            assert not draft["r1_passed"]


@pytest.mark.parametrize("source,value", [("tool_input", True), ("semantic_evidence", False),
                                         ("action_catalog", "true"), ("action_catalog", 1)])
def test_live_selector_schema_rejects_invalid_source_and_flag(source, value):
    from atomic_skillgraph.agents.protocol import validate_schema_instance, SchemaValidationError
    from atomic_skillgraph.agents.structured_submission import TOOL_IR_COLLECTION_SOURCE_SCHEMA
    from atomic_skillgraph.tooling.validator import _validate_selector
    selector = {"source": source, "field": "display_text", "refresh_each_iteration": value}
    with pytest.raises(SchemaValidationError):
        validate_schema_instance(selector, TOOL_IR_COLLECTION_SOURCE_SCHEMA)
    failures = []
    _validate_selector(selector, "loop", fail=lambda code, message: failures.append(code))
    assert "tool_ir_selector_invalid" in failures


@pytest.mark.parametrize("anchor,output", [
    (ParameterSpec("value", "entity"), ParameterSpec("value", "entity", required_resolution="concrete")),
    (ParameterSpec("anchor", "entity", required=False), ParameterSpec("found", "entity", required_resolution="concrete")),
    (ParameterSpec("anchor", "entity"), ParameterSpec("found", "entity")),
    (ParameterSpec("anchor", "entity"), ParameterSpec("found", "number", required_resolution="concrete")),
])
def test_semantic_constraint_cannot_upgrade_identity_or_mismatch_type(anchor, output):
    with pytest.raises(ValueError):
        validate_output_semantic_constraints([anchor], [output],
            {output.name: {"compatible_with_input": anchor.name}})


def test_public_predicate_authority_is_the_existing_harness_schema():
    from atomic_skillgraph.tooling.runtime_interface import public_predicate_schema
    harness = route.CandidateHarness(route.RouteCase("schema"))
    schema = {item["predicate"]: item for item in public_predicate_schema(harness)}
    assert schema["object.observed"]["validation_source"]
    assert schema["entity.discovered_at"]["validation_source"]
    assert schema["object.observed"]["validation_source"] != schema["entity.discovered_at"]["validation_source"]
