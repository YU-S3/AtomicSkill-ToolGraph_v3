"""Compile ImplementationAtom into the only learned native tool visible to an Agent."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from ..agents.protocol import SchemaValidationError, validate_schema_instance
from ..core.bindings import (
    BindingExprKind, BindingExpression, BindingResolution, BindingSource,
    BindingStatus, GroundingConstraint, GroundingConstraintKind, RuntimeBinding,
    resolution_satisfies,
)
from ..core.contracts import (
    AbstractAtomicSkill, IdentityRelation, ImplementationAtom, ParameterSpec,
    TaskContract, ToolAsset,
)
from ..core.errors import AtomicSkillGraphError, FailureLayer
from ..core.results import ImplementationInvocationSpec, ToolCallPreflightResult
from ..core.status import RuntimeMode, skill_status_usable, tool_status_usable
from ..knowledge.skill_registry import SkillRegistry
from ..knowledge.tool_registry import ToolRegistry
from .binding_store import RuntimeBindingStore
from .evidence_store import GroundingEvidenceStore


_TYPE_SCHEMA = {
    "string": "string", "str": "string", "entity": "string", "object": "string",
    "integer": "integer", "int": "integer", "number": "number", "float": "number",
    "boolean": "boolean", "bool": "boolean", "array": "array", "list": "array",
    "object_map": "object", "dict": "object",
}


@dataclass
class CompiledInvocation:
    spec: ImplementationInvocationSpec
    atomic: AbstractAtomicSkill
    implementation: ImplementationAtom
    tools: list[ToolAsset]


class InvocationCompiler:
    def __init__(
        self, skills: SkillRegistry, tools: ToolRegistry, harness: Any,
        *, mode: RuntimeMode | str = RuntimeMode.ONLINE, candidate_policy: Any | None = None,
    ) -> None:
        self.skills, self.tools, self.harness = skills, tools, harness
        self.mode = RuntimeMode(mode)
        self.candidate_policy = candidate_policy
        self.compile_rejections: list[dict[str, Any]] = []

    def compile(
        self, atomic: AbstractAtomicSkill, implementation: ImplementationAtom,
        tools: list[ToolAsset], current_bindings: dict[str, RuntimeBinding],
    ) -> ImplementationInvocationSpec:
        if implementation.abstract_ref != atomic.ref:
            raise ValueError("implementation abstract_ref does not match Atomic")
        if not skill_status_usable(implementation.status, self.mode):
            raise ValueError("implementation lifecycle status is not usable")
        by_ref = {str(tool.ref): tool for tool in tools}
        if len(by_ref) != len(implementation.tool_bindings):
            raise ValueError("not all ToolBinding refs were loaded")
        orders = [item.order for item in implementation.tool_bindings]
        if len(orders) != len(set(orders)):
            raise ValueError("ToolBinding order must be unique")
        policy_mode = implementation.execution_policy.get("mode", "serial")
        if policy_mode != "serial":
            raise ValueError("v3 first release supports serial Implementation only")
        atomic_inputs = {item.name: item for item in atomic.inputs}
        tool_outputs: set[tuple[str, str]] = set()
        for binding in sorted(implementation.tool_bindings, key=lambda item: item.order):
            tool = by_ref.get(str(binding.tool_ref))
            if tool is None or not tool_status_usable(tool.status, self.mode):
                raise ValueError(f"Tool ref unavailable or unusable: {binding.tool_ref}")
            required = set(tool.signature.get("required", []))
            properties = tool.signature.get("properties", {})
            if not required and isinstance(properties, dict):
                required = {name for name, schema in properties.items() if schema.get("required") is True}
            missing = required - set(binding.parameter_mapping)
            if missing:
                raise ValueError(f"Tool required arguments have no mapping: {sorted(missing)}")
            for parameter, expression in binding.parameter_mapping.items():
                expression = BindingExpression.from_dict(expression)
                if expression.kind is BindingExprKind.SKILL_INPUT and expression.source_role not in atomic_inputs:
                    raise ValueError(f"mapping references unknown Atomic input: {expression.source_role}")
                if expression.kind is BindingExprKind.TOOL_OUTPUT and (expression.source_step, expression.source_role) not in tool_outputs:
                    raise ValueError(f"mapping references unavailable prior Tool output: {expression.source_step}.{expression.source_role}")
            output_schema = tool.interface.get("output_schema", {})
            for name in output_schema.get("properties", {}):
                tool_outputs.add((binding.role, name))
        for constraint in implementation.grounding_constraints:
            if not self.harness.supports_constraint(constraint.kind.value, constraint.verifier_id):
                raise ValueError(f"Harness cannot support constraint {constraint.constraint_id}")
        output_mapping = implementation.execution_policy.get("output_mapping", {})
        for output in atomic.outputs:
            if output.required and output.name not in output_mapping:
                # Identity output with the same Atomic input role is closed.
                if output.name not in atomic_inputs:
                    raise ValueError(f"required Atomic output has no output mapping: {output.name}")

        properties: dict[str, Any] = {}
        required: list[str] = []
        for parameter in atomic.inputs:
            schema_type = _TYPE_SCHEMA.get(parameter.semantic_type.casefold(), "string")
            schema: dict[str, Any] = {"type": schema_type, "description": parameter.description}
            properties[parameter.name] = schema
            current = current_bindings.get(parameter.name)
            if parameter.required and (
                current is None or current.status is not BindingStatus.GROUNDED
                or not resolution_satisfies(current.resolution, parameter.required_resolution)
            ):
                required.append(parameter.name)
        # Provider-facing native names are opaque routing identifiers.  Do not
        # embed the persisted Implementation logical id: long ids both crowd
        # the 64-character protocol limit and can be mistaken by a model for a
        # second, derivable tool name when the artifact ref is present in
        # policy context.  A stable 64-bit digest keeps names short and unique
        # for the invocation candidates exposed in one turn.
        name_digest = hashlib.sha256(
            str(implementation.ref).encode("utf-8")
        ).hexdigest()[:16]
        return ImplementationInvocationSpec(
            name=f"invoke_impl_{name_digest}", implementation_ref=implementation.ref,
            atomic_ref=atomic.ref, description=f"Execute learned implementation for: {atomic.summary}",
            input_schema={"type": "object", "properties": properties, "required": required, "additionalProperties": False},
            grounding_constraints=list(implementation.grounding_constraints),
            tool_refs=[item.tool_ref for item in sorted(implementation.tool_bindings, key=lambda item: item.order)],
            execution_policy=dict(implementation.execution_policy),
        )

    def compile_candidates(
        self, occurrence: Any, binding_store: RuntimeBindingStore,
        *, max_candidates: int = 3, task_id: str = "",
    ) -> list[CompiledInvocation]:
        atomic = self.skills.get_atomic(occurrence.node_ref)
        current = binding_store.snapshot_for_node(occurrence)
        refs = list(occurrence.implementation_candidates)
        implementations = []
        loaded = []
        for ref in refs:
            try:
                implementation = self.skills.get_implementation(ref)
                loaded.append(implementation)
            except KeyError:
                self.compile_rejections.append({"implementation_ref": str(ref), "code": "implementation_compile_rejected", "reason": "missing"})
        active_available = any(str(item.status.value) == "active" for item in loaded)
        for implementation in loaded:
            try:
                if self.candidate_policy is not None and not self.candidate_policy.allows(
                    artifact_ref=str(implementation.ref), artifact_kind="implementation",
                    status=implementation.status, mode=self.mode,
                    task_id=task_id or "unknown_task",
                    reliable_active_available=active_available,
                ):
                    self.compile_rejections.append({
                        "implementation_ref": str(implementation.ref),
                        "code": "implementation_compile_rejected",
                        "reason": "candidate_exploration_quota",
                    })
                    continue
                quality = float(implementation.quality.get("reliability", 0.0))
                preferred = 1 if implementation.quality.get("preferred") else 0
                implementations.append((preferred, quality, str(implementation.ref), implementation))
            except (TypeError, ValueError) as exc:
                self.compile_rejections.append({
                    "implementation_ref": str(implementation.ref),
                    "code": "implementation_compile_rejected", "reason": str(exc),
                })
        implementations.sort(key=lambda item: (-item[0], -item[1], item[2]))
        result: list[CompiledInvocation] = []
        for _, _, _, implementation in implementations:
            try:
                tools = [self.tools.get(binding.tool_ref) for binding in sorted(implementation.tool_bindings, key=lambda item: item.order)]
                for tool in tools:
                    active_tool_available = any(
                        other.ref.tool_id == tool.ref.tool_id
                        and str(other.status.value) in {"active", "preferred"}
                        for other in self.tools.tools()
                    )
                    if self.candidate_policy is not None and not self.candidate_policy.allows(
                        artifact_ref=str(tool.ref), artifact_kind="tool", status=tool.status,
                        mode=self.mode, task_id=task_id or "unknown_task",
                        reliable_active_available=active_tool_available,
                    ):
                        raise ValueError("candidate_tool_exploration_quota")
                spec = self.compile(atomic, implementation, tools, current)
                result.append(CompiledInvocation(spec, atomic, implementation, tools))
            except AtomicSkillGraphError as exc:
                if exc.layer is FailureLayer.INFRASTRUCTURE:
                    raise
                self.compile_rejections.append({
                    "implementation_ref": str(implementation.ref), "code": "implementation_compile_rejected",
                    "reason": str(exc),
                })
            except (KeyError, TypeError, ValueError) as exc:
                self.compile_rejections.append({
                    "implementation_ref": str(implementation.ref), "code": "implementation_compile_rejected",
                    "reason": str(exc),
                })
            if len(result) >= max_candidates:
                break
        return result

    def preflight(
        self, compiled: CompiledInvocation, *, call_name: str, call_id: str,
        arguments: dict[str, Any], occurrence: Any, binding_store: RuntimeBindingStore,
        evidence_store: GroundingEvidenceStore, revision: int,
        arguments_are_agent_proposals: bool = True,
        task_contract: TaskContract | None = None,
    ) -> ToolCallPreflightResult:
        ref = str(compiled.implementation.ref)
        def fail(layer: str, code: str, message: str) -> ToolCallPreflightResult:
            return ToolCallPreflightResult(False, ref, failure_layer=layer, failure_code=code, message=message)

        # 1. native name / call id
        if not call_id or call_name != compiled.spec.name:
            return fail("runtime_agent", "runtime_agent_schema_error", "invalid call_id or invocation name")
        # 2. JSON schema
        try:
            validate_schema_instance(arguments, compiled.spec.input_schema)
        except (SchemaValidationError, TypeError, ValueError) as exc:
            return fail("runtime_agent", "runtime_agent_schema_error", str(exc))
        # 3. semantic types are enforced by the compiled schema; booleans are not integers.
        by_parameter = {item.name: item for item in compiled.atomic.inputs}
        for role, value in arguments.items():
            parameter = by_parameter.get(role)
            if parameter and _TYPE_SCHEMA.get(parameter.semantic_type.casefold(), "string") == "string" and not isinstance(value, str):
                return fail("runtime_agent", "runtime_agent_schema_error", f"{role} has incompatible semantic type")
        current = binding_store.snapshot_for_node(occurrence)
        # 3b. A schema-valid concrete entity may still belong to the wrong
        # semantic family.  Compare before proposal grounding/commit, using
        # immutable Task/DataFlow intent as the anchor.
        if arguments_are_agent_proposals:
            compatibility = getattr(self.harness, "semantic_value_compatible", None)
            for role, value in arguments.items():
                parameter = by_parameter.get(role)
                anchor_binding = binding_store.semantic_anchor_for(occurrence, role)
                if parameter is None or anchor_binding is None:
                    continue
                if callable(compatibility):
                    compatible = bool(compatibility(
                        role=role,
                        concrete_value=value,
                        semantic_anchor=anchor_binding.value,
                        semantic_type=parameter.semantic_type,
                    ))
                else:
                    compatible = value == anchor_binding.value
                if not compatible:
                    return fail(
                        "runtime_binding",
                        "runtime_semantic_anchor_mismatch",
                        f"Agent proposal {role} is incompatible with its semantic anchor",
                    )
        # Occurrence-scoped identity is a pre-start obligation.  Task-scoped
        # cardinality/identity remains a terminal contract obligation because
        # a multi-object task may legitimately use one invocation per object.
        if task_contract is not None:
            proposal_values = {
                role: binding.value for role, binding in current.items()
                if binding.status is BindingStatus.GROUNDED
            }
            proposal_values.update(arguments)
            for constraint in task_contract.identity_constraints:
                if constraint.scope != "occurrence":
                    continue
                if (
                    constraint.left_role not in proposal_values
                    or constraint.right_role not in proposal_values
                ):
                    continue
                left = proposal_values[constraint.left_role]
                right = proposal_values[constraint.right_role]
                if (
                    constraint.relation is IdentityRelation.SAME_AS and left != right
                ) or (
                    constraint.relation is IdentityRelation.DISTINCT_FROM and left == right
                ):
                    return fail(
                        "runtime_binding", "runtime_identity_constraint_mismatch",
                        "Agent proposal violates occurrence identity/cardinality constraints",
                    )
        # 4. static mapping closure was validated at compile time. Recheck immutable refs.
        try:
            current_impl = self.skills.get_implementation(compiled.implementation.ref)
            if current_impl.abstract_ref != compiled.atomic.ref:
                return fail("implementation", "implementation_mapping_error", "implementation mapping no longer matches Atomic")
        except KeyError:
            return fail("implementation", "implementation_mapping_error", "implementation disappeared")

        grounded: dict[str, RuntimeBinding] = {}
        matched: list[str] = []
        if arguments_are_agent_proposals:
            proposals = binding_store.propose_agent_arguments(occurrence, arguments, revision, {
                item.name: item.semantic_type for item in compiled.atomic.inputs
            })
            # Agent proposals must be certified, even when schema-valid.
            for role, proposal in proposals.items():
                entity_constraint = GroundingConstraint(
                    f"proposal_concrete_{role}", GroundingConstraintKind.ARGUMENT_CONCRETE,
                    argument_mapping={role: BindingExpression(BindingExprKind.SKILL_INPUT, source_role=role)},
                    required_resolution="concrete",
                )
                local, refs = binding_store.ground_from_evidence(occurrence.occurrence_id, {role: proposal}, [entity_constraint], evidence_store)
                if not local:
                    return fail("runtime_binding", "runtime_binding_not_concrete", f"Agent proposal {role} has no current concrete evidence")
                grounded.update(local)
                matched.extend(refs)
        else:
            for role, value in arguments.items():
                binding = current.get(role)
                if binding is None or binding.value != value or binding.status is not BindingStatus.GROUNDED:
                    return fail("runtime_binding", "runtime_binding_unresolved", f"autonomous argument is not a certified current binding: {role}")
        merged = dict(current)
        merged.update(grounded)
        # 5. resolution
        for parameter in compiled.atomic.inputs:
            if not parameter.required:
                continue
            binding = merged.get(parameter.name)
            if binding is None or binding.status is not BindingStatus.GROUNDED:
                return fail("runtime_binding", "runtime_binding_unresolved", f"required binding unresolved: {parameter.name}")
            if not resolution_satisfies(binding.resolution, parameter.required_resolution):
                return fail("runtime_binding", "runtime_binding_not_concrete", f"binding resolution insufficient: {parameter.name}")
        # 6/7. Grounding relation + current revision
        values = {role: binding.value for role, binding in merged.items()}
        for constraint in compiled.spec.grounding_constraints:
            evidence = evidence_store.match_constraint(constraint, values, revision)
            if not evidence:
                return fail("runtime_binding", "runtime_relation_not_grounded", f"constraint not grounded: {constraint.constraint_id}")
            if any(not item.valid_at(revision) for item in evidence):
                return fail("runtime_binding", "stale_grounding_evidence", f"stale evidence: {constraint.constraint_id}")
            matched.extend(item.evidence_id for item in evidence)
            if constraint.required_resolution == "relation_verified":
                for expression in constraint.argument_mapping.values():
                    if expression.source_role in merged:
                        old = merged[expression.source_role]
                        upgraded = RuntimeBinding(
                            old.role, old.value, old.semantic_type, BindingSource.HARNESS_EVIDENCE,
                            BindingStatus.GROUNDED, BindingResolution.RELATION_VERIFIED,
                            list(dict.fromkeys(old.evidence_refs + [item.evidence_id for item in evidence])), revision,
                        )
                        grounded[old.role] = upgraded
                        merged[old.role] = upgraded
        # 8. compatibility
        profiles = compiled.implementation.compatibility.get("harness_profiles") or []
        if profiles and self.harness.profile_name not in profiles:
            return fail("implementation", "implementation_compatibility_error", "Harness profile incompatible")
        # 9. safety/lifecycle
        if not skill_status_usable(compiled.implementation.status, self.mode):
            return fail("implementation", "implementation_compatibility_error", "Implementation status unusable")
        for tool in compiled.tools:
            if not tool_status_usable(tool.status, self.mode) or tool.safety.get("blocked"):
                return fail("implementation", "implementation_compatibility_error", f"Tool unavailable or unsafe: {tool.ref}")
        binding_store.commit_grounded(occurrence.occurrence_id, grounded)
        normalized = {role: binding.value for role, binding in merged.items() if role in by_parameter}
        return ToolCallPreflightResult(
            True, ref, normalized, list(merged.values()), list(dict.fromkeys(matched)), "", "", "",
        )

    def autonomous_preflight(
        self, compiled: CompiledInvocation, occurrence: Any,
        binding_store: RuntimeBindingStore, evidence_store: GroundingEvidenceStore, revision: int,
        task_contract: TaskContract | None = None,
    ) -> ToolCallPreflightResult:
        current = binding_store.snapshot_for_node(occurrence)
        arguments = {
            parameter.name: current[parameter.name].value
            for parameter in compiled.atomic.inputs
            if parameter.name in current and current[parameter.name].status is BindingStatus.GROUNDED
        }
        return self.preflight(
            compiled, call_name=compiled.spec.name, call_id="autonomous",
            arguments=arguments, occurrence=occurrence, binding_store=binding_store,
            evidence_store=evidence_store, revision=revision,
            arguments_are_agent_proposals=False,
            task_contract=task_contract,
        )
