"""Production expression projection; all execution authority stays outside it."""
import copy
import json

import pytest

from atomic_skillgraph.agents.protocol import NativeToolSpec, ONE_NATIVE_CALL_PREFIX
from atomic_skillgraph.agents.runtime_policy_projection import compact_native_interfaces, restore_native_interfaces
from atomic_skillgraph.agents.context_builder import ContextBuilder
from atomic_skillgraph.agents.skill_guidance import organize_guidance_view


def surface():
    return {"allowed_implementation_invocations": [{"name": "invoke_test",
        "description": "Perform the declared bounded work.",
        "input_schema": {"type": "object", "properties": {"x": {"const": 1}}, "required": ["x"], "additionalProperties": False},
        "ready": False, "scope": "parent", "unknown_public_semantics": {"value": None}}],
        "current_state_snapshot": {"current_atomic": {"skill_guidance": {
            "steps": ["Check.", "Act.", "Check."], "notes": ["Keep the identity."], "soft_reference": True}}},
        "execution_frame": {"last_step": {"call_id": "a", "started": True, "completed": False, "arguments": {"x": 4}}}}


def native(payload, prefix=""):
    row = payload["allowed_implementation_invocations"][0]
    return NativeToolSpec(row["name"], prefix + row["description"], copy.deepcopy(row["input_schema"]))


@pytest.mark.parametrize("prefix", ["", ONE_NATIVE_CALL_PREFIX])
def test_actual_native_exact_roundtrip_preserves_extra_fields(prefix):
    raw = surface()
    original = copy.deepcopy(raw)
    specs = [native(raw, prefix)]
    out, audit = compact_native_interfaces(raw, specs)
    assert raw == original
    assert "description" not in out["allowed_implementation_invocations"][0]
    assert restore_native_interfaces(out, specs, audit) == raw
    assert audit["required_surface_equal"]
    assert audit["guidance_input_hash"] == audit["guidance_output_hash"]
    assert out["execution_frame"] == raw["execution_frame"]
    assert out["allowed_implementation_invocations"][0]["ready"] is False
    assert len(json.dumps(out)) < len(json.dumps(raw))


@pytest.mark.parametrize("condition", ["absent", "empty", "duplicate", "bool", "float", "wrapper", "changed_description", "other_name", "required"])
def test_unknown_or_different_native_definition_is_never_removed(condition):
    raw = surface()
    tool = native(raw)
    specs = [tool]
    if condition == "absent": specs = None
    if condition == "empty": specs = []
    if condition == "duplicate": specs *= 2
    if condition in {"bool", "float"}: tool.input_schema["properties"]["x"]["const"] = True if condition == "bool" else 1.0
    if condition == "wrapper": specs = [native(raw, "Similar wrapper ")]
    if condition == "changed_description": specs = [NativeToolSpec(tool.name, tool.description + " ", tool.input_schema)]
    if condition == "other_name": specs = [NativeToolSpec("unoffered", tool.description, tool.input_schema)]
    if condition == "required": tool.input_schema["required"] = []
    out, audit = compact_native_interfaces(raw, specs)
    assert out == raw and not audit["removed_paths"]
    assert audit["original_fallback_reasons"]


def test_restoration_detects_payload_and_native_tampering():
    raw = surface()
    specs = [native(raw)]
    out, audit = compact_native_interfaces(raw, specs)
    with pytest.raises(ValueError):
        restore_native_interfaces(out, [native(raw, ONE_NATIVE_CALL_PREFIX)], audit)
    out["execution_frame"]["last_step"]["started"] = False
    with pytest.raises(ValueError):
        restore_native_interfaces(out, specs, audit)


def test_guidance_does_not_revise_or_deduplicate():
    view = surface()["current_state_snapshot"]["current_atomic"]["skill_guidance"]
    result = organize_guidance_view(view)
    assert result == view and result is not view
    assert result["steps"] == ["Check.", "Act.", "Check."]


def test_context_builder_projects_only_the_request_surface():
    raw = surface()
    spec = native(raw)
    audit = {}
    prompt = ContextBuilder().runtime_node(task_goal="public goal", atomic_contract={"summary": "a", "inputs": [], "outputs": [], "preconditions": [], "effects": []},
        observation="public", action_catalog=[], relevant_action_history=[], remaining_budget={},
        implementation_invocations=[spec], native_tool_specs=[spec], projection_audit=audit,
        runtime_step_mode="preparation", execution_frame=raw["execution_frame"])
    payload = json.loads(prompt.split("\n\nPOLICY_CONTEXT_JSON\n")[1])
    assert payload["allowed_implementation_invocations"] == [{"name": spec.name}]
    assert "expression_audit" not in payload
    assert audit["expression_audit"]["required_surface_equal"]
    assert payload["execution_frame"] == raw["execution_frame"]
