"""Compile ImplementationAtom into the only learned native tool visible to an Agent."""

from __future__ import annotations

import re
import hashlib
import copy
from dataclasses import dataclass
from typing import Any

from ..agents.protocol import SchemaValidationError, validate_schema_instance
from ..agents.native_call_contract import IMPLEMENTATION_HELP
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
from ..evolution.portability import contract_label, validate_portability
from ..tooling.entry_contract import normalize_entry_contract, check_tool_entry, required_tool_parameters
from .binding_store import RuntimeBindingStore
from .evidence_store import GroundingEvidenceStore


_TYPE_SCHEMA = {
    "string": "string", "str": "string", "entity": "string", "object": "string",
    "integer": "integer", "int": "integer", "number": "number", "float": "number",
    "boolean": "boolean", "bool": "boolean", "array": "array", "list": "array",
    "object_map": "object", "dict": "object",
}


def _binding_diagnostics(atomic, occurrence, store, arguments, revision, roles, constraint):
    from ..core.serialization import to_primitive
    parameters = {p.name: p for p in atomic.inputs}
    missing = []
    for role in dict.fromkeys(roles):
        if role not in parameters:
            continue
        anchor = store.semantic_anchor_for(occurrence, role)
        missing.append({'role': role, 'required_resolution': parameters[role].required_resolution,
            'submitted_value': to_primitive(arguments.get(role)),
            'semantic_anchor': to_primitive(anchor.value) if anchor else None,
            'evidence_revision': revision, 'missing': 'current_authorizing_evidence'})
    if not missing and constraint is None:
        return {}
    return {'bindings': missing, 'required_anchor_or_relation': to_primitive(constraint),
            'relevant_revision': revision}


@dataclass
class CompiledInvocation:
    spec: ImplementationInvocationSpec
    atomic: AbstractAtomicSkill
    implementation: ImplementationAtom
    tools: list[ToolAsset]


def _tool_arguments(mapping, atomic_values, tool_outputs, atomic, tool, *, first=False):
    """Resolve explicit mappings; absence is allowed only at two optional ends."""
    inputs = {p.name: p for p in atomic.inputs}
    properties = tool.signature.get("properties", {})
    required = required_tool_parameters(tool.signature)
    result = {}
    for role, raw in mapping.items():
        expression = BindingExpression.from_dict(raw)
        if role not in properties:
            raise ValueError(f"unknown Tool parameter {role}")
        if expression.kind is BindingExprKind.CONSTANT:
            result[role] = expression.constant
        elif expression.kind is BindingExprKind.SKILL_INPUT:
            source = expression.source_role
            if source not in inputs:
                raise ValueError(f"unknown Atomic input {source} mapped to {role}")
            if source not in atomic_values:
                if not inputs[source].required and role not in required:
                    continue
                raise ValueError(f"missing Atomic input {source} mapped to {role}: source or target is required")
            result[role] = atomic_values[source]
        elif not first and expression.kind is BindingExprKind.TOOL_OUTPUT:
            result[role] = tool_outputs[(expression.source_step, expression.source_role)]
        elif not first and expression.kind in {BindingExprKind.DATA_FLOW, BindingExprKind.ADAPTER_TRANSFORM}:
            result[role] = atomic_values[expression.source_role]
        else:
            raise ValueError(f"unresolved Tool parameter {role}: unsupported expression {expression.kind.value}")
    return result


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
        from .input_authorization import validate_declarations
        input_authorizations = validate_declarations(atomic)
        tool_outputs: set[tuple[str, str]] = set()
        for binding in sorted(implementation.tool_bindings, key=lambda item: item.order):
            tool = by_ref.get(str(binding.tool_ref))
            if tool is None or not tool_status_usable(tool.status, self.mode):
                raise ValueError(f"Tool ref unavailable or unusable: {binding.tool_ref}")
            normalize_entry_contract(tool.interface.get('entry_contract'), tool.signature.get('properties', {}))
            required = required_tool_parameters(tool.signature)
            properties = tool.signature.get("properties", {})
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
            authorization = input_authorizations.get(parameter.name, {})
            if authorization.get('kind') == 'ordered_entity_scope':
                schema.update(items={'type':'string'},minItems=authorization['min_items'],
                    maxItems=authorization['max_items'],uniqueItems=authorization['unique_items'])
            properties[parameter.name] = schema
            for binding in implementation.tool_bindings:
                tool = by_ref[str(binding.tool_ref)]
                if tool.artifact.get('value_contract_version') != 2:
                    continue
                for target, raw in binding.parameter_mapping.items():
                    expression = BindingExpression.from_dict(raw)
                    if expression.kind is BindingExprKind.SKILL_INPUT and expression.source_role == parameter.name:
                        constraint = tool.signature.get('properties', {}).get(target)
                        if constraint is not None:
                            schema.setdefault('allOf', []).append(copy.deepcopy(constraint))
            current = current_bindings.get(parameter.name)
            if parameter.required and (
                current is None or current.status is not BindingStatus.GROUNDED
                or not resolution_satisfies(current.resolution, parameter.required_resolution)
            ):
                required.append(parameter.name)
        # Keep provider-facing names semantic enough for reliable model tool
        # selection, but never embed the full persisted Implementation id.
        # Long ids crowd the 64-character protocol limit and can be mistaken
        # for a second, derivable tool name.  The bounded semantic prefix plus
        # stable digest is at most 49 characters and remains unique for the
        # invocation candidates exposed in one turn.
        canonical_intent = str(
            dict(implementation.metadata or {}).get("canonical_intent")
            or dict(atomic.metadata or {}).get("canonical_intent")
            or ""
        )
        if not validate_portability(
            canonical_intent,
            require_intent=True,
        ).passed:
            canonical_intent = contract_label(atomic.effects, atomic.outputs)
        name_id = re.sub(
            r"[^A-Za-z0-9_-]", "_", canonical_intent,
        )[:24].strip("_-")
        if not name_id:
            name_id = "implementation"
        name_digest = hashlib.sha256(
            str(implementation.ref).encode("utf-8")
        ).hexdigest()[:12]
        description = str(
            dict(implementation.metadata or {}).get("semantic_description")
            or canonical_intent.replace("_", " ")
        )
        if atomic.metadata.get('runtime_support_promotion'):
            description = contract_label(atomic.effects, atomic.outputs).replace('_', ' ')
            properties = {name: {key: value for key, value in spec.items() if key != 'description'}
                          for name, spec in properties.items()}
        return ImplementationInvocationSpec(
            name=f"invoke_impl_{name_id}_{name_digest}", implementation_ref=implementation.ref,
            atomic_ref=atomic.ref,
            description=IMPLEMENTATION_HELP + f"Execute learned implementation for: {description}. "
                f"Entry: {[(str(t.ref), t.interface['entry_contract']) for t in tools]}. "
                f"Execution results (not caller inputs): {[(p.name, p.semantic_type, p.required) for p in atomic.outputs]}",
            input_schema={"type": "object", "properties": properties, "required": required, "additionalProperties": False},
            grounding_constraints=list(implementation.grounding_constraints),
            tool_refs=[item.tool_ref for item in sorted(implementation.tool_bindings, key=lambda item: item.order)],
            execution_policy=dict(implementation.execution_policy),
        )

    def compile_candidates(
        self, occurrence: Any, binding_store: RuntimeBindingStore,
        *, max_candidates: int | None = None, task_id: str = "",
        evidence_store: GroundingEvidenceStore | None = None, revision: int | None = None,
        task_contract: TaskContract | None = None,
    ) -> list[CompiledInvocation]:
        if getattr(self, "r103", False):
            return self._compile_routes(occurrence, binding_store, max_candidates=max_candidates,
                task_id=task_id, evidence_store=evidence_store, revision=revision, task_contract=task_contract)
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
            if max_candidates is not None and len(result) >= max_candidates:
                break
        return result

    def route_availability(self, compiled, occurrence, binding_store, evidence_store, revision, task_contract=None):
        """Original compiler/preflight as a read-only diagnostic; never commits updates."""
        incompatible = self._compatibility_failure(compiled)
        if incompatible is not None:
            return {"state": "unusable", "gaps": [incompatible.failure_code], "message": incompatible.message}
        if evidence_store is None or revision is None:
            return {"state": "preparable", "gaps": ["current_evidence_unavailable"]}
        values = {role: binding.value for role,binding in binding_store.snapshot_for_node(occurrence).items()
                  if binding.status is BindingStatus.GROUNDED}
        conflict = self._identity_failure(compiled, occurrence, binding_store, values, task_contract)
        if conflict is not None:
            return {"state": "unusable", "gaps": [conflict.failure_code], "message": conflict.message}
        missing = compiled.spec.input_schema.get("required", [])
        if missing:
            return {"state": "preparable", "gaps": ["input:" + role for role in missing]}
        prepared = self.prepare_arguments(compiled, call_name=compiled.spec.name, call_id="route_readonly_probe",
            arguments={}, occurrence=occurrence, binding_store=binding_store, evidence_store=evidence_store,
            revision=revision, arguments_are_agent_proposals=False, task_contract=task_contract)
        checked = self.validate_execution_context(compiled, prepared, occurrence=occurrence,
            binding_store=binding_store, evidence_store=evidence_store, revision=revision) if prepared.passed else prepared
        if checked.passed:
            return {"state": "ready", "gaps": []}
        # The normal validators distinguish current evidence/entry gaps from
        # a proved semantic conflict. No validator result is made into PASS.
        conflict = checked.failure_layer == "implementation" or checked.failure_code in {
            "runtime_semantic_anchor_mismatch", "runtime_identity_constraint_mismatch"}
        return {"state": "unusable" if conflict else "preparable", "gaps": [checked.failure_code], "message": checked.message}

    def _compatibility_failure(self, compiled):
        reason = ""
        profiles = compiled.implementation.compatibility.get("harness_profiles") or []
        if profiles and self.harness.profile_name not in profiles:
            reason = "Harness profile incompatible"
        elif not skill_status_usable(compiled.implementation.status, self.mode):
            reason = "Implementation status unusable"
        elif any(not tool_status_usable(tool.status, self.mode) or tool.safety.get("blocked") for tool in compiled.tools):
            reason = "Tool unavailable or unsafe"
        if reason:
            return ToolCallPreflightResult(False, str(compiled.implementation.ref), failure_layer="implementation",
                failure_code="implementation_compatibility_error", message=reason)
        return None

    def _compile_routes(self, occurrence, binding_store, *, max_candidates, task_id, evidence_store, revision, task_contract):
        from ..evolution.identity_matching import match_implementation, raw_hash, MAX_SEARCH_STATES
        from ..governance.projections import ArtifactStats
        from ..knowledge.identity_index import IdentityIndex
        import json
        atomic = self.skills.get_atomic(occurrence.node_ref)
        if not skill_status_usable(atomic.status, self.mode):
            return []
        current = binding_store.snapshot_for_node(occurrence)
        routes = []
        remaining_identity_states = MAX_SEARCH_STATES
        identity_index = IdentityIndex(self.skills.database, self.skills.store.data_dir)
        for ref in occurrence.implementation_candidates:
            try:
                implementation = self.skills.get_implementation(ref)
                tools = [self.tools.get(b.tool_ref) for b in implementation.tool_bindings]
                compiled = CompiledInvocation(self.compile(atomic, implementation, tools, current), atomic, implementation, tools)
                availability = self.route_availability(compiled, occurrence, binding_store, evidence_store, revision, task_contract)
                if availability["state"] == "unusable":
                    continue
                duplicate = None
                for route in routes:
                    if remaining_identity_states <= 0:
                        break  # Unknown comparison keeps both legal routes.
                    other = route["compiled"]
                    proof = match_implementation(implementation, other.implementation, source_atomic=atomic, target_atomic=other.atomic,
                        source_tools={str(t.ref): t for t in tools}, target_tools={str(t.ref): t for t in other.tools},
                        max_states=remaining_identity_states)
                    remaining_identity_states -= proof.search_states
                    if proof.status == "exact":
                        duplicate = route
                        break
                candidate = any(str(item.status.value) == "candidate" for item in (atomic, implementation, *tools))
                row = self.skills.database.execute("SELECT projection_json FROM lifecycle_projection WHERE artifact_ref=?", (str(ref),)).fetchone()
                stats = ArtifactStats.from_dict(json.loads(row[0])) if row else ArtifactStats(str(ref), "implementation")
                route_key = identity_index.equivalence_key(str(implementation.ref))
                displays = self.skills.database.execute("SELECT COUNT(*) FROM candidate_route_exposures WHERE route_key=?",
                                                       (route_key,)).fetchone()[0]
                row = dict(compiled=compiled, availability=availability, candidate=candidate, stats=stats,
                           displays=displays, route_key=route_key)
                if duplicate is not None:
                    # Equivalent interfaces are one route. Prefer an available
                    # reliable representative, never create a new ref/status.
                    if duplicate["candidate"] and not candidate:
                        duplicate.update(row)
                    continue
                routes.append(row)
            except AtomicSkillGraphError as exc:
                if exc.layer is FailureLayer.INFRASTRUCTURE:
                    raise
                self.compile_rejections.append(dict(implementation_ref=str(ref), code="implementation_compile_rejected", reason=str(exc)))
            except (KeyError, TypeError, ValueError) as exc:
                self.compile_rejections.append(dict(implementation_ref=str(ref), code="implementation_compile_rejected", reason=str(exc)))
        reliable_ready = any(not r["candidate"] and r["availability"]["state"] == "ready" for r in routes)
        task_key = getattr(self, "independent_task_key", "") or task_id
        allowed = []
        for route in routes:
            if route["candidate"] and self.candidate_policy is not None and not self.candidate_policy.allows(
                artifact_ref=route["route_key"], artifact_kind="implementation", status="candidate", mode=self.mode,
                task_id=task_key, reliable_active_available=reliable_ready):
                continue
            allowed.append(route)
        def rank(route):
            from ..deployment.preferences import implementation_priority
            stats = route["stats"]
            return (route["availability"]["state"] != "ready",
                -implementation_priority(self.skills, route['compiled'].implementation.ref, route['availability']['state'] == 'ready'),
                stats.execution_support.get("intrinsic_failure_count", stats.intrinsic_failure_count),
                max(0, 2-stats.independent_execution_support_count), route["displays"],
                raw_hash([task_key, route["route_key"]]))
        reliable = sorted((r for r in allowed if not r["candidate"]), key=rank)
        candidates = sorted((r for r in allowed if r["candidate"]), key=rank)
        limit = min(3, max_candidates if max_candidates is not None else 3)
        selected = reliable[:limit]
        if candidates and limit > 0:
            selected = reliable[:limit-1] + candidates[:1]
        selected.sort(key=rank)
        from ..deployment.preferences import implementation_priority
        best_ready = any(r['availability']['state'] == 'ready' for r in selected)
        priorities = [implementation_priority(self.skills, r['compiled'].implementation.ref,
            r['availability']['state'] == 'ready') if (r['availability']['state'] == 'ready') == best_ready
            else 0 for r in selected]
        if priorities and max(priorities) > 0 and priorities.count(max(priorities)) == 1:
            # Detached read-only projection consumed by existing unique-route
            # selection. No historical quality/success statistic is persisted.
            import copy
            chosen = selected[priorities.index(max(priorities))]['compiled']
            chosen.implementation = copy.deepcopy(chosen.implementation)
            chosen.implementation.quality['preferred'] = True
        self.route_diagnostics = [{"implementation_ref": str(r["compiled"].implementation.ref),
            "route_equivalence_key": r["route_key"], "candidate": r["candidate"], **r["availability"]} for r in selected]
        return [r["compiled"] for r in selected]

    def record_display(self, ctx, session_id, invocations, native_tools, *, native_name=None):
        if not getattr(self, "r103", False) or self.mode is RuntimeMode.FROZEN:
            return
        from ..knowledge.identity_index import IdentityIndex
        names = {tool.name for tool in native_tools}
        index = IdentityIndex(self.skills.database, self.skills.store.data_dir)
        audit = ctx.trace_builder.trace.metadata.setdefault("candidate_route_exposures", [])
        for invocation in invocations:
            if (native_name or invocation.spec.name) not in names:
                continue
            key = index.equivalence_key(str(invocation.implementation.ref))
            row = {"session_id":session_id, "route_key":key, "implementation_ref":str(invocation.implementation.ref),
                   "native_name":native_name or invocation.spec.name,
                   "candidate":any(str(item.status.value) == "candidate" for item in
                       (invocation.atomic,invocation.implementation,*invocation.tools))}
            if row not in audit:
                audit.append(row)

    @staticmethod
    def _identity_failure(compiled, occurrence, binding_store, values, task_contract):
        def fail(code, message):
            return ToolCallPreflightResult(False, str(compiled.implementation.ref),
                failure_layer="runtime_binding", failure_code=code, message=message)
        if task_contract is not None:
            for constraint in task_contract.identity_constraints:
                if (constraint.scope != "occurrence" or constraint.left_role not in values
                        or constraint.right_role not in values):
                    continue
                left, right = values[constraint.left_role], values[constraint.right_role]
                if ((constraint.relation is IdentityRelation.SAME_AS and left != right)
                        or (constraint.relation is IdentityRelation.DISTINCT_FROM and left == right)):
                    return fail("runtime_identity_constraint_mismatch",
                                "Agent proposal violates occurrence identity/cardinality constraints")
        repeat = binding_store.preflight_repeat_bindings(occurrence.step_id, values)
        if not repeat.passed:
            return fail(repeat.failure_codes[0] if repeat.failure_codes else "runtime_repetition_distinctness_violation",
                        "Invocation arguments violate an effect-committed RepeatBlock binding")
        return None

    def prepare_arguments(
        self, compiled: CompiledInvocation, *, call_name: str, call_id: str,
        arguments: dict[str, Any], occurrence: Any, binding_store: RuntimeBindingStore,
        evidence_store: GroundingEvidenceStore, revision: int,
        arguments_are_agent_proposals: bool = True,
        task_contract: TaskContract | None = None,
    ) -> ToolCallPreflightResult:
        ref = str(compiled.implementation.ref)
        def fail(layer: str, code: str, message: str, *, roles=(), constraint=None) -> ToolCallPreflightResult:
            return ToolCallPreflightResult(False, ref, failure_layer=layer, failure_code=code, message=message,
                diagnostics=_binding_diagnostics(compiled.atomic, occurrence, binding_store, arguments,
                    revision, roles, constraint))

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
        from .input_authorization import validate_declarations, validate_committed
        control_declarations = validate_declarations(compiled.atomic)
        # 3b. A schema-valid concrete entity may still belong to the wrong
        # semantic family.  Compare before proposal grounding/commit, using
        # immutable Task/DataFlow intent as the anchor.
        if arguments_are_agent_proposals:
            compatibility = getattr(self.harness, "semantic_value_compatible", None)
            for role, value in arguments.items():
                if role in control_declarations:
                    continue  # The pure typed control authorizer below owns this port.
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
        repeat_values = {
            role: binding.value
            for role, binding in current.items()
            if binding.status is BindingStatus.GROUNDED
        }
        repeat_values.update(arguments)
        identity_failure = self._identity_failure(compiled, occurrence, binding_store, repeat_values, task_contract)
        if identity_failure is not None:
            return identity_failure
        grounded: dict[str, RuntimeBinding] = {}
        matched: list[str] = []
        if arguments_are_agent_proposals:
            proposals = binding_store.propose_agent_arguments(occurrence, arguments, revision, {
                item.name: item.semantic_type for item in compiled.atomic.inputs
            })
            # Agent proposals must be certified, even when schema-valid.
            for role, proposal in proposals.items():
                parameter = by_parameter[role]
                anchor = binding_store.semantic_anchor_for(occurrence, role)
                if role in control_declarations:
                    from .input_authorization import authorize
                    try:
                        grounded[role] = authorize(compiled.atomic, role, proposal.value,
                            call_id=call_id, evidence_store=evidence_store, revision=revision,
                            fixed_anchor=anchor)
                    except ValueError as exc:
                        return fail('runtime_binding', 'caller_input_authorization_failed', str(exc))
                    # Caller provenance is not an effect witness. It stays on
                    # the pending binding and existing transaction audit.
                    continue
                if (str(parameter.required_resolution) == "semantic"
                        and anchor is not None and anchor.value == proposal.value):
                    # A declared semantic input is not a concrete entity.
                    # Reusing the exact formal anchor proves this argument;
                    # it neither invents a new anchor nor upgrades resolution.
                    grounded[role] = anchor
                    matched.extend(anchor.evidence_refs)
                    continue
                entity_constraint = GroundingConstraint(
                    f"proposal_concrete_{role}", GroundingConstraintKind.ARGUMENT_CONCRETE,
                    argument_mapping={role: BindingExpression(BindingExprKind.SKILL_INPUT, source_role=role)},
                    required_resolution="concrete",
                )
                local, refs = binding_store.ground_from_evidence(occurrence.occurrence_id, {role: proposal}, [entity_constraint], evidence_store)
                if not local:
                    return fail("runtime_binding", "runtime_binding_not_concrete", f"Agent proposal {role} has no current concrete evidence",
                        roles=[role], constraint=entity_constraint)
                grounded.update(local)
                matched.extend(refs)
        else:
            for role, value in arguments.items():
                binding = current.get(role)
                if binding is None or binding.value != value or binding.status is not BindingStatus.GROUNDED:
                    return fail("runtime_binding", "runtime_binding_unresolved", f"autonomous argument is not a certified current binding: {role}")
        merged = dict(current)
        merged.update(grounded)
        for role in control_declarations:
            if role in merged:
                try:
                    validate_committed(compiled.atomic, role, merged[role],
                        evidence_store=evidence_store, revision=revision)
                except ValueError as exc:
                    return fail('runtime_binding', 'caller_input_authorization_failed', str(exc))
        # Parameter-only evidence is part of argument preparation.  It must
        # be certified before the effect short-circuit, while affordance and
        # current-context relations remain execution-context gates below.
        argument_kinds = {
            GroundingConstraintKind.ARGUMENT_EXISTS,
            GroundingConstraintKind.ARGUMENT_CONCRETE,
        }
        for constraint in compiled.spec.grounding_constraints:
            if constraint.kind not in argument_kinds:
                continue
            values = {role: binding.value for role, binding in merged.items()}
            evidence = evidence_store.match_constraint(
                constraint, values, revision,
            )
            if not evidence:
                return fail(
                    "runtime_binding",
                    "runtime_binding_not_concrete",
                    f"argument constraint not grounded: {constraint.constraint_id}",
                    roles=[e.source_role for e in constraint.argument_mapping.values()], constraint=constraint,
                )
            if any(not item.valid_at(revision) for item in evidence):
                return fail(
                    "runtime_binding",
                    "stale_grounding_evidence",
                    f"stale evidence: {constraint.constraint_id}",
                )
            refs = [item.evidence_id for item in evidence]
            matched.extend(refs)
            if constraint.kind is not GroundingConstraintKind.ARGUMENT_CONCRETE:
                continue
            required = BindingResolution.CONCRETE
            for expression in constraint.argument_mapping.values():
                role = expression.source_role
                if role not in merged:
                    continue
                old = merged[role]
                if resolution_satisfies(old.resolution, required):
                    continue
                upgraded = RuntimeBinding(
                    old.role,
                    old.value,
                    old.semantic_type,
                    BindingSource.HARNESS_EVIDENCE,
                    BindingStatus.GROUNDED,
                    required,
                    list(dict.fromkeys([*old.evidence_refs, *refs])),
                    revision,
                )
                grounded[old.role] = upgraded
                merged[old.role] = upgraded

        # Required input resolution.
        for parameter in compiled.atomic.inputs:
            if not parameter.required:
                continue
            binding = merged.get(parameter.name)
            if binding is None or binding.status is not BindingStatus.GROUNDED:
                return fail("runtime_binding", "runtime_binding_unresolved", f"required binding unresolved: {parameter.name}")
            if not resolution_satisfies(binding.resolution, parameter.required_resolution):
                # Authenticate this exact supplied value, never search for a
                # replacement. Upgrades remain provisional until transaction.
                concrete = GroundingConstraint(
                    "given_" + parameter.name, GroundingConstraintKind.ARGUMENT_CONCRETE,
                    argument_mapping={parameter.name: BindingExpression(
                        BindingExprKind.SKILL_INPUT, source_role=parameter.name)})
                local, refs = binding_store.ground_from_evidence(
                    occurrence.occurrence_id, {parameter.name: binding}, [concrete], evidence_store)
                if local:
                    grounded.update(local)
                    merged.update(local)
                    matched.extend(refs)
                    if resolution_satisfies(local[parameter.name].resolution, parameter.required_resolution):
                        continue
                # A role-specific execution-context constraint may provide the
                # missing concrete/relation authority immediately before the
                # implementation starts.  Defer only to such a declared
                # constraint; validate_execution_context must both match it
                # and perform the final required-resolution check.
                can_be_certified = any(
                    constraint.kind not in {
                        GroundingConstraintKind.ARGUMENT_EXISTS,
                        GroundingConstraintKind.ARGUMENT_CONCRETE,
                    }
                    and any(
                        expression.source_role == parameter.name
                        for expression in constraint.argument_mapping.values()
                    )
                    and resolution_satisfies(
                        constraint.required_resolution,
                        parameter.required_resolution,
                    )
                    for constraint in compiled.spec.grounding_constraints
                )
                if not can_be_certified:
                    return fail("runtime_binding", "runtime_binding_not_concrete", f"binding resolution insufficient: {parameter.name}",
                        roles=[parameter.name], constraint=concrete)
        normalized = {
            role: binding.value
            for role, binding in merged.items()
            if role in by_parameter
        }
        return ToolCallPreflightResult(
            True,
            ref,
            normalized,
            list(grounded.values()),
            list(dict.fromkeys(matched)),
        )

    def validate_execution_context(
        self,
        compiled: CompiledInvocation,
        prepared: ToolCallPreflightResult,
        *,
        occurrence: Any,
        binding_store: RuntimeBindingStore,
        evidence_store: GroundingEvidenceStore,
        revision: int,
    ) -> ToolCallPreflightResult:
        """Pure entry check. Only the selected invocation transaction commits."""

        ref = str(compiled.implementation.ref)

        def fail(layer: str, code: str, message: str, *, roles=(), constraint=None) -> ToolCallPreflightResult:
            return ToolCallPreflightResult(
                False,
                ref,
                failure_layer=layer,
                failure_code=code,
                message=message,
                diagnostics=_binding_diagnostics(compiled.atomic, occurrence, binding_store,
                    prepared.normalized_arguments, revision, roles, constraint),
            )

        if not prepared.passed:
            return prepared
        # Static mapping closure was validated at compile time. Recheck the
        # immutable relation immediately before execution-context admission.
        try:
            current_impl = self.skills.get_implementation(
                compiled.implementation.ref,
            )
            if current_impl.abstract_ref != compiled.atomic.ref:
                return fail(
                    "implementation",
                    "implementation_mapping_error",
                    "implementation mapping no longer matches Atomic",
                )
        except KeyError:
            return fail(
                "implementation",
                "implementation_mapping_error",
                "implementation disappeared",
            )

        current = binding_store.snapshot_for_node(occurrence)
        grounded = {item.role: item for item in prepared.binding_updates}
        merged = dict(current)
        merged.update(grounded)
        matched = list(prepared.matched_evidence_refs)
        # Grounding relation + current revision.  In particular,
        # entry_affordance remains a hard gate whenever the Atomic effect has
        # not already been proven by the validator short-circuit.
        values = {role: binding.value for role, binding in merged.items()}
        for constraint in compiled.spec.grounding_constraints:
            if constraint.kind in {
                GroundingConstraintKind.ARGUMENT_EXISTS,
                GroundingConstraintKind.ARGUMENT_CONCRETE,
            }:
                continue
            evidence = evidence_store.match_constraint(constraint, values, revision)
            if not evidence:
                return fail("runtime_binding", "runtime_relation_not_grounded", f"constraint not grounded: {constraint.constraint_id}",
                    roles=[e.source_role for e in constraint.argument_mapping.values()], constraint=constraint)
            if any(not item.valid_at(revision) for item in evidence):
                return fail("runtime_binding", "stale_grounding_evidence", f"stale evidence: {constraint.constraint_id}")
            matched.extend(item.evidence_id for item in evidence)
            required_resolution = BindingResolution(
                constraint.required_resolution
            )
            if required_resolution is not BindingResolution.SEMANTIC:
                for expression in constraint.argument_mapping.values():
                    if expression.source_role in merged:
                        old = merged[expression.source_role]
                        if resolution_satisfies(
                            old.resolution, required_resolution,
                        ):
                            continue
                        upgraded = RuntimeBinding(
                            old.role, old.value, old.semantic_type, BindingSource.HARNESS_EVIDENCE,
                            BindingStatus.GROUNDED, required_resolution,
                            list(dict.fromkeys(old.evidence_refs + [item.evidence_id for item in evidence])), revision,
                        )
                        grounded[old.role] = upgraded
                        merged[old.role] = upgraded
        # Direct and Agent-selected learned invocations share this final
        # authority check.  Matched evidence alone is never sufficient: each
        # required Atomic role must now carry the declared resolution.
        for parameter in compiled.atomic.inputs:
            if not parameter.required:
                continue
            binding = merged.get(parameter.name)
            if binding is None or binding.status is not BindingStatus.GROUNDED:
                return fail(
                    "runtime_binding",
                    "runtime_binding_unresolved",
                    f"required binding unresolved after context validation: {parameter.name}",
                )
            if not resolution_satisfies(
                binding.resolution, parameter.required_resolution,
            ):
                return fail(
                    "runtime_binding",
                    "runtime_binding_not_concrete",
                    f"binding resolution insufficient after context validation: {parameter.name}",
                    roles=[parameter.name],
                )
        incompatible = self._compatibility_failure(compiled)
        if incompatible is not None:
            return incompatible
        if compiled.atomic.preconditions:
            report = self.harness.validator_channel().validate_atomic_effect({
                'effects': compiled.atomic.preconditions, 'bindings': values, 'output_candidates': {}})
            if not report.passed:
                return fail('runtime_binding', 'runtime_preconditions_unsatisfied', '; '.join(report.messages))
        if not compiled.implementation.tool_bindings:
            return fail("implementation", "implementation_mapping_error", "implementation has no Tool bindings")
        first = min(compiled.implementation.tool_bindings, key=lambda item: item.order)
        tool = next((t for t in compiled.tools if t.ref == first.tool_ref), None)
        if tool is None:
            return fail('implementation', 'implementation_mapping_error', 'First Tool binding has no resolved Tool')
        try:
            arguments = _tool_arguments(first.parameter_mapping, values, {}, compiled.atomic, tool, first=True)
        except (KeyError, TypeError, ValueError) as exc:
            return fail('implementation', 'implementation_mapping_error', str(exc))
        entry = check_tool_entry(tool, arguments, self.harness, evidence_store, revision)
        if not entry.passed:
            return fail('tool', entry.failure_codes[0], '; '.join(entry.messages))
        return ToolCallPreflightResult(
            True,
            ref,
            dict(prepared.normalized_arguments),
            list(merged.values()),
            list(dict.fromkeys(matched)),
        )

    def preflight(
        self, compiled: CompiledInvocation, *, call_name: str, call_id: str,
        arguments: dict[str, Any], occurrence: Any, binding_store: RuntimeBindingStore,
        evidence_store: GroundingEvidenceStore, revision: int,
        arguments_are_agent_proposals: bool = True,
        task_contract: TaskContract | None = None,
    ) -> ToolCallPreflightResult:
        prepared = self.prepare_arguments(
            compiled,
            call_name=call_name,
            call_id=call_id,
            arguments=arguments,
            occurrence=occurrence,
            binding_store=binding_store,
            evidence_store=evidence_store,
            revision=revision,
            arguments_are_agent_proposals=arguments_are_agent_proposals,
            task_contract=task_contract,
        )
        if not prepared.passed:
            return prepared
        return self.validate_execution_context(
            compiled,
            prepared,
            occurrence=occurrence,
            binding_store=binding_store,
            evidence_store=evidence_store,
            revision=revision,
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
