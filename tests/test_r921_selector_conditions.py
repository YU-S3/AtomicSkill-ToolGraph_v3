"""R9.2.1 current-catalog selector conditions and rejection boundaries."""
import copy

import pytest

from atomic_skillgraph.tooling.ir import ToolExecutionState, evaluate_condition


def query():
    return {"op": "exists", "match": {
        "source": "action_catalog", "where": {
            "action_type": "TAKE", "argument_role": "object",
            "semantic_compatible_with": {"source": "tool_input", "field": "target",
                                          "semantic_type": "entity"},
        }, "project": {"kind": "argument", "role": "object"}, "distinct": True,
    }}


def matcher(**kwargs):
    return kwargs["concrete_value"].split("_")[0] == kwargs["semantic_anchor"]


@pytest.mark.parametrize("objects,expected", [([], False), (["cup_1"], False), (["egg_1"], True)])
def test_target_query_is_readonly_and_target_specific(objects, expected):
    state = ToolExecutionState(bindings={"target": "egg"}, catalog=[{
        "action_id": f"action_{i}", "revision": 0, "action_type": "TAKE",
        "arguments": {"object": obj, "source": "desk_1"},
    } for i, obj in enumerate(objects)])
    before = copy.deepcopy(state)
    assert evaluate_condition(query(), state, semantic_compatible=matcher) is expected
    negative = {**query(), "op": "not_exists"}
    assert evaluate_condition(negative, state, semantic_compatible=matcher) is (not expected)
    assert state == before


@pytest.mark.parametrize("op,expected", [("exists", True), ("not_exists", False),
    ("equals", True), ("not_equals", False), ("contains", True), ("empty", False), ("non_empty", True)])
def test_legacy_operators_keep_their_original_behavior(op, expected):
    assert evaluate_condition({"source": "tool_input", "field": "flag", "op": op, "value": "abc"},
                              ToolExecutionState(bindings={"flag": "abc"})) is expected


def test_current_revision_and_local_scope():
    state = ToolExecutionState(local={"target": "egg"}, catalog_revision=1)
    condition = query()
    condition["match"]["where"]["semantic_compatible_with"]["source"] = "local_variable"
    assert not evaluate_condition(condition, state, semantic_compatible=matcher)
    state.catalog = [{"action_id": "a", "revision": 2, "action_type": "TAKE", "arguments": {"object": "egg_1"}}]
    with pytest.raises(ValueError, match="catalog_invalid"):
        evaluate_condition(condition, state, semantic_compatible=matcher)
    state.catalog_revision = 2
    assert evaluate_condition(condition, state, semantic_compatible=matcher)
    state.catalog = []
    state.catalog_revision = 3
    assert not evaluate_condition(condition, state, semantic_compatible=matcher)
    del state.local["target"]
    with pytest.raises(ValueError, match="reference_unavailable"):
        evaluate_condition(condition, state, semantic_compatible=matcher)


@pytest.mark.parametrize("change", ["mixed", "operator", "source", "project", "distinct", "nested", "reference_source",
                                    "operator_type", "reference_type", "null_reference", "wrapper", "missing_role", "missing_reference"])
def test_native_and_interpreter_reject_invalid_shape(change):
    from atomic_skillgraph.agents.protocol import validate_schema_instance, SchemaValidationError
    from atomic_skillgraph.agents.structured_submission import TOOL_IR_CONDITION_SCHEMA
    condition = query()
    if change == "mixed": condition["source"] = "tool_input"
    if change == "operator": condition["op"] = "equals"
    if change == "source": condition["match"]["source"] = "semantic_evidence"
    if change == "project": condition["match"]["project"]["kind"] = "field"
    if change == "distinct": condition["match"]["distinct"] = "yes"
    if change == "nested": condition["match"]["match"] = query()
    if change == "reference_source": condition["match"]["where"]["semantic_compatible_with"]["source"] = "semantic_evidence"
    if change == "operator_type": condition["op"] = {}
    if change == "reference_type": condition["match"]["where"]["semantic_compatible_with"]["source"] = []
    if change == "null_reference": condition["match"]["where"]["semantic_compatible_with"] = None
    if change == "wrapper": condition["match"]["where"]["arguments"] = {"object": "egg"}
    if change == "missing_role": del condition["match"]["where"]["argument_role"]
    if change == "missing_reference": del condition["match"]["where"]["semantic_compatible_with"]
    with pytest.raises(SchemaValidationError): validate_schema_instance(condition, TOOL_IR_CONDITION_SCHEMA)
    with pytest.raises(ValueError, match="condition_match_invalid"):
        evaluate_condition(condition, ToolExecutionState(), semantic_compatible=matcher)


def test_missing_callback_and_callback_exception_are_not_negative_matches():
    state = ToolExecutionState(bindings={"target": "egg"})
    with pytest.raises(ValueError, match="matcher_unavailable"): evaluate_condition(query(), state)
    state.catalog = [{"action_id": "a", "revision": 0, "action_type": "TAKE", "arguments": {"object": "egg_1"}}]
    def broken(**kwargs): raise RuntimeError("matcher transport broken")
    with pytest.raises(RuntimeError, match="transport broken"):
        evaluate_condition(query(), state, semantic_compatible=broken)
    state.catalog[0]["arguments"] = {}
    with pytest.raises(ValueError, match="projection_invalid"):
        evaluate_condition(query(), state, semantic_compatible=matcher)


def test_query_is_independent_of_alfworld_action_and_role_names():
    condition = query()
    condition["match"]["where"].update(action_type="RETRIEVE", argument_role="payload")
    condition["match"]["project"]["role"] = "payload"
    state = ToolExecutionState(bindings={"target": "parcel"}, catalog=[{
        "action_id": "a", "revision": 0, "action_type": "RETRIEVE", "arguments": {"payload": "parcel_2"},
    }])
    assert evaluate_condition(condition, state, semantic_compatible=matcher)


@pytest.mark.parametrize("mutation", ["valid", "primitive", "role", "local", "literal", "code"])
def test_nested_static_scope_roles_and_literals(mutation):
    from test_r92_tool_ir_public_contract import _catalog_atomic_and_proposal, _runtime_context
    from atomic_skillgraph.tooling.validator import ToolStaticValidator
    atomic, proposal = _catalog_atomic_and_proposal()
    _, ctx, _ = _runtime_context()
    condition = query()
    condition["match"]["where"]["semantic_compatible_with"]["field"] = "destination"
    condition["match"]["where"].update(action_type="GO_TO", argument_role="destination")
    condition["match"]["project"]["role"] = "destination"
    if mutation == "primitive": condition["match"]["where"]["action_type"] = "INVENTED"
    if mutation == "role": condition["match"]["project"]["role"] = "invented"
    if mutation == "local": condition["match"]["where"]["semantic_compatible_with"].update(source="local_variable", field="undefined")
    if mutation == "literal": condition["match"]["where"]["destination"] = "egg_987"
    if mutation == "code": condition["match"]["where"]["destination"] = "eval(secret)"
    proposal.program[0]["body"].insert(0, {"op": "IF", "node_id": "query", "condition": condition, "then_branch": []})
    result = ToolStaticValidator().validate_proposal(proposal, atomic, ctx.harness)
    assert result.passed is (mutation == "valid"), result.failure_codes


def test_output_derivation_explanation_is_shared_without_changing_r0():
    from atomic_skillgraph.agents.structured_submission import RUNTIME_AUTOMATION_ATOMIC_SCHEMA
    from atomic_skillgraph.agents.runtime_prompt_texts import PREPARATION_ONLY, SEEDED_ONLY
    from atomic_skillgraph.tooling.runtime_interface import RUNTIME_OUTPUT_DERIVATION_RULES
    assert RUNTIME_OUTPUT_DERIVATION_RULES in PREPARATION_ONLY
    assert RUNTIME_OUTPUT_DERIVATION_RULES in SEEDED_ONLY
    assert RUNTIME_OUTPUT_DERIVATION_RULES in RUNTIME_AUTOMATION_ATOMIC_SCHEMA["properties"]["effects"]["description"]
    assert "output_semantic_constraints" in RUNTIME_OUTPUT_DERIVATION_RULES
    assert "Candidates and prose never create witnesses" in RUNTIME_OUTPUT_DERIVATION_RULES
