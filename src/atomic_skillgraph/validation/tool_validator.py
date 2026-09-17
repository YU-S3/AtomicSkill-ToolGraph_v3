"""Tool-local schema, safety, artifact, and output checks."""

from __future__ import annotations

from typing import Any

from ..core.contracts import ToolAsset
from ..core.results import ValidationResult


def validate_json_schema(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    from ..agents.protocol import validate_schema_instance, SchemaValidationError
    try:
        validate_schema_instance(value, schema, path=path)
        return []
    except (SchemaValidationError, TypeError, ValueError) as exc:
        return [str(exc)]


class ToolValidator:
    SUPPORTED_ARTIFACTS = {"primitive_ir", "tool_ir_v1"}

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
        elif tool.artifact_kind == "tool_ir_v1":
            program = tool.artifact.get("program")
            checks["tool_ir_program"] = isinstance(program, list) and bool(program)
            checks["tool_ir_bounded"] = isinstance(tool.artifact.get("max_actions"), int) and tool.artifact["max_actions"] > 0
            checks["tool_ir_final_effects"] = bool(tool.artifact.get("final_effects"))
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
