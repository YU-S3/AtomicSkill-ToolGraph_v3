"""Tool-local schema, safety, artifact, and output checks."""

from __future__ import annotations

from typing import Any

from ..core.contracts import ToolAsset
from ..core.results import ValidationResult


def validate_json_schema(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    errors: list[str] = []
    expected = schema.get("type")
    type_map = {
        "object": dict, "array": list, "string": str, "integer": int,
        "number": (int, float), "boolean": bool, "null": type(None),
    }
    if expected in type_map and not isinstance(value, type_map[expected]):
        return [f"{path}: expected {expected}"]
    if expected == "object" and isinstance(value, dict):
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}.{name}: required")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            errors.extend(f"{path}.{name}: additional property" for name in value if name not in properties)
        for name, item in value.items():
            if name in properties:
                errors.extend(validate_json_schema(item, properties[name], f"{path}.{name}"))
    elif expected == "array" and isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            errors.extend(validate_json_schema(item, schema["items"], f"{path}[{index}]"))
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: not in enum")
    return errors


class ToolValidator:
    SUPPORTED_ARTIFACTS = {"primitive_ir"}

    def validate_asset(self, tool: ToolAsset) -> ValidationResult:
        checks = {
            "signature_object": isinstance(tool.signature, dict),
            "interface_object": isinstance(tool.interface, dict),
            "artifact_supported": tool.artifact_kind in self.SUPPORTED_ARTIFACTS,
            "artifact_object": isinstance(tool.artifact, dict),
            "safety_declared": isinstance(tool.safety, dict) and tool.safety.get("reviewed") is True,
        }
        if tool.artifact_kind == "primitive_ir":
            steps = tool.artifact.get("steps")
            checks["primitive_steps"] = isinstance(steps, list) and bool(steps)
            checks["primitive_shape"] = bool(steps) and all(isinstance(step, dict) and bool(step.get("action_type")) for step in steps)
        passed = all(checks.values())
        return ValidationResult(
            "tool", passed, checks,
            [] if passed else ["tool_preflight_rejected"],
            [] if passed else ["ToolAsset failed local artifact validation"],
        )

    def validate_output(self, tool: ToolAsset, output: dict[str, Any]) -> ValidationResult:
        schema = tool.interface.get("output_schema") or tool.signature.get("returns") or {"type": "object"}
        errors = validate_json_schema(output, schema)
        return ValidationResult(
            "tool_output", not errors, {"output_schema": not errors},
            [] if not errors else ["tool_output_schema_error"], errors,
        )
