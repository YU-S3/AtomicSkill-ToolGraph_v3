"""Artifact-kind dispatch + replay admission before online Candidate use."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from ..core.bindings import BindingExprKind, BindingExpression
from ..core.contracts import AbstractAtomicSkill, ImplementationAtom, ToolAsset
from ..core.status import SkillStatus, ToolStatus
from ..tooling.ir import walk_program_nodes
from ..tooling.validator import ToolStaticValidator
from ..validation.tool_validator import ToolValidator


class Admission:
    def __init__(self, tool_validator: ToolValidator) -> None:
        self.tool_validator = tool_validator
        self.static_validator = ToolStaticValidator()

    def admit_tool(
        self,
        tool: ToolAsset,
        *,
        replay: Callable[[ToolAsset, dict[str, Any]], bool] | None,
        atomic: AbstractAtomicSkill | None = None,
        harness: Any | None = None,
    ) -> ToolAsset:
        if tool.artifact_kind == "tool_ir_v1":
            return self._admit_tool_ir_v1(
                tool, replay=replay, atomic=atomic, harness=harness,
            )
        if tool.artifact_kind == "primitive_ir":
            return self._admit_primitive_ir(tool, replay=replay)
        return replace(tool, status=ToolStatus.SHADOW, metadata={
            **tool.metadata, "admission_failure": ["unsupported_tool_artifact_kind"],
        })

    def _admit_primitive_ir(
        self, tool: ToolAsset, *, replay: Callable[[ToolAsset, dict[str, Any]], bool] | None,
    ) -> ToolAsset:
        local = self.tool_validator.validate_asset(tool)
        closure_failures = self._primitive_closure_failures(tool)
        if not local.passed or closure_failures:
            return replace(tool, status=ToolStatus.SHADOW, metadata={
                **tool.metadata,
                "admission_failure": list(dict.fromkeys([
                    *local.failure_codes, *closure_failures,
                ])),
            })
        replay_cases = [
            item for item in tool.tests
            if item.get("kind") in {"source_replay", "tool_proposal_replay"}
        ]
        if not replay_cases or replay is None:
            return replace(tool, status=ToolStatus.SHADOW, metadata={
                **tool.metadata, "admission_failure": ["source_replay_unavailable"],
            })
        results = [bool(replay(tool, item)) for item in replay_cases]
        if not all(results):
            return replace(tool, status=ToolStatus.SHADOW, metadata={
                **tool.metadata, "admission_failure": ["source_replay_failed"],
            })
        return replace(tool, status=ToolStatus.CANDIDATE, metadata={
            **tool.metadata, "admission": {"source_replay": results},
        })

    def _admit_tool_ir_v1(
        self,
        tool: ToolAsset,
        *,
        replay: Callable[[ToolAsset, dict[str, Any]], bool] | None,
        atomic: AbstractAtomicSkill | None,
        harness: Any | None,
    ) -> ToolAsset:
        local = self.tool_validator.validate_asset(tool)
        closure_failures = self._tool_ir_closure_failures(tool)
        static_failures: list[str] = []
        if atomic is not None and harness is not None:
            static = self.static_validator.validate_tool_asset(tool, atomic, harness)
            if not static.passed:
                static_failures = list(static.failure_codes)
        if not local.passed or closure_failures or static_failures:
            return replace(tool, status=ToolStatus.SHADOW, metadata={
                **tool.metadata,
                "admission_failure": list(dict.fromkeys([
                    *local.failure_codes, *closure_failures, *static_failures,
                ])),
            })
        replay_cases = [
            item for item in tool.tests
            if item.get("kind") in {"source_replay", "tool_proposal_replay"}
        ]
        if not replay_cases or replay is None:
            return replace(tool, status=ToolStatus.SHADOW, metadata={
                **tool.metadata, "admission_failure": ["tool_ir_replay_unavailable"],
            })
        results = [bool(replay(tool, item)) for item in replay_cases]
        if not all(results):
            return replace(tool, status=ToolStatus.SHADOW, metadata={
                **tool.metadata, "admission_failure": ["tool_ir_replay_failed"],
            })
        return replace(tool, status=ToolStatus.CANDIDATE, metadata={
            **tool.metadata,
            "admission": {"tool_ir_replay": results, "kind": "tool_ir_v1"},
        })

    def admit_implementation(
        self,
        implementation: ImplementationAtom,
        tool: ToolAsset | list[ToolAsset] | tuple[ToolAsset, ...],
        *,
        atomic: AbstractAtomicSkill,
        harness: Any,
    ) -> ImplementationAtom:
        reasons: list[str] = []
        supplied_tools = list(tool) if isinstance(tool, (list, tuple)) else [tool]
        tools_by_ref = {str(item.ref): item for item in supplied_tools}
        if len(tools_by_ref) != len(supplied_tools):
            reasons.append("duplicate_supplied_tool_ref")
        if implementation.abstract_ref != atomic.ref:
            reasons.append("abstract_ref_mismatch")
        if not implementation.tool_bindings:
            reasons.append("missing_tool_bindings")
        if implementation.execution_policy.get("mode", "serial") != "serial":
            reasons.append("unsupported_execution_policy")
        orders = [item.order for item in implementation.tool_bindings]
        if len(orders) != len(set(orders)):
            reasons.append("duplicate_tool_order")
        if sorted(orders) != list(range(len(orders))):
            reasons.append("non_contiguous_tool_order")

        atomic_inputs = {item.name: item for item in atomic.inputs}
        atomic_outputs = {item.name: item for item in atomic.outputs}
        available_tool_outputs: set[tuple[str, str]] = set()
        binding_roles: set[str] = set()
        for binding in sorted(implementation.tool_bindings, key=lambda item: item.order):
            bound_tool = tools_by_ref.get(str(binding.tool_ref))
            if bound_tool is None:
                reasons.append("tool_ref_missing")
                continue
            if bound_tool.status not in {
                ToolStatus.CANDIDATE, ToolStatus.ACTIVE, ToolStatus.PREFERRED,
            }:
                reasons.append("tool_not_admitted")
            if not binding.role or binding.role in binding_roles:
                reasons.append("duplicate_or_empty_tool_role")
            binding_roles.add(binding.role)
            properties = bound_tool.signature.get("properties") or {}
            required = set(bound_tool.signature.get("required") or [])
            if required - set(binding.parameter_mapping):
                reasons.append("required_tool_argument_unmapped")
            if set(binding.parameter_mapping) - set(properties):
                reasons.append("unknown_tool_argument_mapping")
            for expression in binding.parameter_mapping.values():
                if not self._mapping_closed(
                    expression,
                    atomic_inputs=set(atomic_inputs),
                    available_tool_outputs=available_tool_outputs,
                ):
                    reasons.append("implementation_mapping_not_closed")
            output_schema = bound_tool.interface.get("output_schema") or {}
            for name in (output_schema.get("properties") or {}):
                available_tool_outputs.add((binding.role, str(name)))
            if bound_tool.safety.get("blocked"):
                reasons.append("tool_safety_blocked")

        constraint_ids: set[str] = set()
        for constraint in implementation.grounding_constraints:
            if not constraint.constraint_id or constraint.constraint_id in constraint_ids:
                reasons.append("duplicate_or_empty_constraint_id")
            constraint_ids.add(constraint.constraint_id)
            if constraint.required_resolution not in {
                "semantic", "concrete", "relation_verified",
            }:
                reasons.append("unsupported_constraint_resolution")
            if not harness.supports_constraint(constraint.kind.value, constraint.verifier_id):
                reasons.append("unsupported_grounding_constraint")
            if any(
                not self._mapping_closed(
                    expression,
                    atomic_inputs=set(atomic_inputs),
                    available_tool_outputs=set(),
                    allow_tool_output=False,
                )
                for expression in constraint.argument_mapping.values()
            ):
                reasons.append("constraint_mapping_not_closed")

        profiles = list(implementation.compatibility.get("harness_profiles") or [])
        if not profiles or str(harness.profile_name) not in profiles:
            reasons.append("harness_profile_incompatible")
        output_mapping = implementation.execution_policy.get("output_mapping")
        if not isinstance(output_mapping, dict):
            reasons.append("output_mapping_missing")
            output_mapping = {}
        required_outputs = {
            name for name, spec in atomic_outputs.items() if spec.required
        }
        if required_outputs - set(output_mapping):
            reasons.append("required_atomic_output_unmapped")
        if set(output_mapping) - set(atomic_outputs):
            reasons.append("unknown_atomic_output_mapping")
        for expression in output_mapping.values():
            if not self._mapping_closed(
                expression,
                atomic_inputs=set(atomic_inputs),
                available_tool_outputs=available_tool_outputs,
            ):
                reasons.append("output_mapping_not_closed")
        return replace(
            implementation,
            status=SkillStatus.SHADOW if reasons else SkillStatus.CANDIDATE,
            quality={
                **implementation.quality,
                "admission_failure": list(dict.fromkeys(reasons)),
            } if reasons else dict(implementation.quality),
        )

    @staticmethod
    def _mapping_closed(
        raw: Any,
        *,
        atomic_inputs: set[str],
        available_tool_outputs: set[tuple[str, str]],
        allow_tool_output: bool = True,
    ) -> bool:
        try:
            expression = BindingExpression.from_dict(raw) if isinstance(raw, dict) else raw
        except (KeyError, TypeError, ValueError):
            return False
        if not isinstance(expression, BindingExpression):
            return False
        if expression.kind is BindingExprKind.CONSTANT:
            return True
        if expression.kind is BindingExprKind.SKILL_INPUT:
            return expression.source_role in atomic_inputs
        if expression.kind is BindingExprKind.TOOL_OUTPUT:
            return allow_tool_output and (
                expression.source_step, expression.source_role
            ) in available_tool_outputs
        return False

    @staticmethod
    def _common_closure_failures(tool: ToolAsset) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
        reasons: list[str] = []
        signature = tool.signature
        if not isinstance(signature, dict):
            return ["tool_signature_invalid"], {}, {}
        properties = signature.get("properties")
        required = signature.get("required")
        if signature.get("type") != "object" or not isinstance(properties, dict):
            return ["tool_signature_invalid"], {}, {}
        if not isinstance(required, list) or len(required) != len(set(required)) or set(required) - set(properties):
            reasons.append("tool_signature_required_invalid")
        output_schema = tool.interface.get("output_schema") if isinstance(tool.interface, dict) else None
        output_properties = output_schema.get("properties") if isinstance(output_schema, dict) else None
        output_required = output_schema.get("required") if isinstance(output_schema, dict) else None
        if (
            not isinstance(output_schema, dict)
            or output_schema.get("type") != "object"
            or not isinstance(output_properties, dict)
            or not isinstance(output_required, list)
            or set(output_required) - set(output_properties)
        ):
            reasons.append("tool_output_interface_invalid")
        return reasons, properties, {
            "output_properties": output_properties,
            "output_required": output_required,
        }

    @classmethod
    def _primitive_closure_failures(cls, tool: ToolAsset) -> list[str]:
        reasons, properties, output = cls._common_closure_failures(tool)
        if not isinstance(tool.artifact, dict):
            reasons.append("tool_artifact_invalid")
            return list(dict.fromkeys(reasons))
        steps = tool.artifact.get("steps")
        allowed = set(tool.safety.get("allowed_action_types") or [])
        for step in steps if isinstance(steps, list) else []:
            action_type = str(step.get("action_type", ""))
            if action_type not in allowed:
                reasons.append("tool_action_not_safety_allowlisted")
            mapping = step.get("argument_mapping")
            if not isinstance(mapping, dict):
                reasons.append("tool_primitive_mapping_invalid")
                continue
            for raw in mapping.values():
                expression = Admission._normalized_expression(raw)
                if not isinstance(expression, BindingExpression):
                    reasons.append("tool_primitive_mapping_invalid")
                elif expression.kind is BindingExprKind.SKILL_INPUT:
                    if expression.source_role not in properties:
                        reasons.append("tool_primitive_mapping_not_closed")
                elif expression.kind is not BindingExprKind.CONSTANT:
                    reasons.append("tool_primitive_mapping_not_closed")
        output_mapping = tool.artifact.get("output_mapping")
        if not isinstance(output_mapping, dict) or set(output_mapping) != set(output.get("output_required") or []):
            reasons.append("tool_output_mapping_incomplete")
        else:
            for raw in output_mapping.values():
                expression = Admission._normalized_expression(raw)
                if not isinstance(expression, BindingExpression):
                    reasons.append("tool_output_mapping_invalid")
                elif expression.kind is BindingExprKind.SKILL_INPUT:
                    if expression.source_role not in properties:
                        reasons.append("tool_output_mapping_not_closed")
                elif expression.kind is not BindingExprKind.CONSTANT:
                    reasons.append("tool_output_mapping_not_closed")
        return list(dict.fromkeys(reasons))

    @classmethod
    def _tool_ir_closure_failures(cls, tool: ToolAsset) -> list[str]:
        reasons, properties, output = cls._common_closure_failures(tool)
        if not isinstance(tool.artifact, dict):
            reasons.append("tool_artifact_invalid")
            return list(dict.fromkeys(reasons))
        program = tool.artifact.get("program")
        if not isinstance(program, list) or not program:
            reasons.append("tool_ir_program_invalid")
            return list(dict.fromkeys(reasons))
        max_actions = tool.artifact.get("max_actions")
        if not isinstance(max_actions, int) or max_actions <= 0:
            reasons.append("tool_ir_max_actions_invalid")
        allowed = set(tool.safety.get("allowed_action_types") or [])
        nodes = walk_program_nodes(program)
        action_types = {
            str(node.get("action_type", ""))
            for node in nodes
            if str(node.get("op", "")) == "ACTION"
        }
        if action_types - allowed:
            reasons.append("tool_ir_action_not_safety_allowlisted")
        return_roles: set[str] = set()
        for node in nodes:
            if str(node.get("op", "")) != "RETURN":
                continue
            return_roles.update(
                str(role)
                for role in dict(node.get("output_sources") or {})
            )
        required_outputs = set(output.get("output_required") or [])
        if required_outputs - return_roles:
            reasons.append("tool_ir_return_closure_invalid")
        return list(dict.fromkeys(reasons))

    @staticmethod
    def _normalized_expression(raw: Any) -> Any:
        try:
            return BindingExpression.from_dict(raw) if isinstance(raw, dict) else raw
        except (KeyError, TypeError, ValueError):
            return None
