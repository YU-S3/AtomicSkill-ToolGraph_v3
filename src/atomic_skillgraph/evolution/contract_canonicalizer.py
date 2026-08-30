"""One canonical Atomic role schema shared by every persistent layer.

The Atomic contract owns role identity.  Tool, Implementation, and Composite
artifacts are executable/reference views of that contract and therefore must
not retain trace-local aliases after alignment.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace
from typing import Any, Mapping

from ..core.bindings import BindingExpression, BindingExprKind
from ..core.contracts import (
    AbstractAtomicSkill,
    CompositeOccurrence,
    CompositeSkill,
    ImplementationAtom,
    ParameterSpec,
    SemanticPredicate,
    ToolAsset,
)
from ..core.edges import GraphEdge
from ..core.refs import SkillRef, content_hash
from ..core.serialization import to_primitive


@dataclass(frozen=True)
class CanonicalizedAtomicBundle:
    """A role-consistent Atomic and its optional executable views.

    ``role_map`` is the convenient unqualified original-to-canonical map.  The
    boundary-specific maps are authoritative when an input and output happen
    to share the same trace-local spelling.
    """

    atomic: AbstractAtomicSkill
    role_map: dict[str, str]
    tool: ToolAsset | None
    implementation: ImplementationAtom | None
    input_role_map: dict[str, str]
    output_role_map: dict[str, str]


def _as_expression(value: Any) -> BindingExpression | None:
    if isinstance(value, BindingExpression):
        return value
    if isinstance(value, dict) and "kind" in value:
        try:
            return BindingExpression.from_dict(value)
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _role_occurrences(
    atomic: AbstractAtomicSkill,
    role: str,
) -> tuple[tuple[str, str, str], ...]:
    """Describe semantic predicate use without depending on an alias."""

    occurrences: list[tuple[str, str, str]] = []
    for boundary, predicates in (
        ("pre", atomic.preconditions),
        ("effect", atomic.effects),
    ):
        for predicate in predicates:
            for argument_name, raw in predicate.args.items():
                expression = _as_expression(raw)
                source_role = expression.source_role if expression else ""
                if source_role == role or raw == f"${role}":
                    occurrences.append((
                        boundary,
                        predicate.predicate.casefold(),
                        str(argument_name),
                    ))
    return tuple(sorted(occurrences))


def _parameter_descriptor(
    atomic: AbstractAtomicSkill,
    spec: ParameterSpec,
) -> tuple[Any, ...]:
    return (
        str(spec.semantic_type).casefold(),
        bool(spec.required),
        bool(spec.runtime_resolvable),
        str(spec.required_resolution).casefold(),
        _role_occurrences(atomic, spec.name),
    )


def _boundary_role_map(
    atomic: AbstractAtomicSkill,
    specs: list[ParameterSpec],
    prefix: str,
) -> dict[str, str]:
    described = [
        (_parameter_descriptor(atomic, spec), spec.name)
        for spec in specs
    ]
    described.sort(key=lambda item: (
        json.dumps(item[0], ensure_ascii=False, sort_keys=True),
        item[1],
    ))
    return {
        original: f"{prefix}_{index:03d}"
        for index, (_descriptor, original) in enumerate(described)
    }


def _rewrite_expression(
    raw: BindingExpression | dict[str, Any],
    role_map: Mapping[str, str],
) -> BindingExpression:
    expression = BindingExpression.from_dict(raw)
    if expression.kind is BindingExprKind.CONSTANT:
        return expression
    return replace(
        expression,
        source_role=role_map.get(expression.source_role, expression.source_role),
    )


def _rewrite_nested(value: Any, role_map: Mapping[str, str]) -> Any:
    """Rewrite typed role references while preserving ordinary constants."""

    if isinstance(value, BindingExpression):
        return _rewrite_expression(value, role_map)
    if isinstance(value, SemanticPredicate):
        return _rewrite_predicate(value, role_map)
    if isinstance(value, list):
        return [_rewrite_nested(item, role_map) for item in value]
    if isinstance(value, tuple):
        return tuple(_rewrite_nested(item, role_map) for item in value)
    if isinstance(value, dict):
        expression = _as_expression(value)
        if expression is not None:
            return to_primitive(_rewrite_expression(expression, role_map))
        return {
            str(key): _rewrite_nested(item, role_map)
            for key, item in value.items()
        }
    if isinstance(value, str):
        if value in role_map:
            return role_map[value]
        if value.startswith("$"):
            role = value[1:]
            return "$" + role_map.get(role, role)
    return value


def _rewrite_validator_spec(
    validator_spec: Mapping[str, Any],
    input_role_map: Mapping[str, str],
    output_role_map: Mapping[str, str],
) -> dict[str, Any]:
    """Rewrite validator roles at their explicit input/output boundaries."""

    payload = copy.deepcopy(dict(validator_spec))
    raw_identity = payload.pop("output_identity", None)
    expression_roles = {**output_role_map, **input_role_map}
    rewritten = _rewrite_nested(payload, expression_roles)
    if raw_identity is not None:
        rewritten["output_identity"] = sorted([
            {
                **dict(item),
                "output_role": output_role_map.get(
                    str(item.get("output_role", "")),
                    str(item.get("output_role", "")),
                ),
                "input_role": input_role_map.get(
                    str(item.get("input_role", "")),
                    str(item.get("input_role", "")),
                ),
            }
            for item in raw_identity
        ], key=lambda item: (
            str(item.get("output_role", "")),
            str(item.get("input_role", "")),
        ))
    return rewritten


def _rewrite_predicate(
    predicate: SemanticPredicate,
    role_map: Mapping[str, str],
) -> SemanticPredicate:
    return replace(
        predicate,
        args={
            str(name): _rewrite_nested(value, role_map)
            for name, value in predicate.args.items()
        },
    )


def _rewrite_parameter(
    spec: ParameterSpec,
    role_map: Mapping[str, str],
) -> ParameterSpec:
    return replace(spec, name=role_map.get(spec.name, spec.name))


def _rewrite_schema_roles(
    schema: Mapping[str, Any],
    role_map: Mapping[str, str],
) -> dict[str, Any]:
    rewritten = copy.deepcopy(dict(schema))
    properties = rewritten.get("properties")
    if isinstance(properties, dict):
        rewritten["properties"] = {
            role_map.get(str(name), str(name)): value
            for name, value in properties.items()
        }
    required = rewritten.get("required")
    if isinstance(required, list):
        rewritten["required"] = [
            role_map.get(str(name), str(name)) for name in required
        ]
    return rewritten


def _identity_payload(
    atomic: AbstractAtomicSkill,
    input_role_map: Mapping[str, str],
    output_role_map: Mapping[str, str],
) -> dict[str, Any]:
    # Predicate BindingExpressions name Atomic inputs in the current IR.  If
    # a trace used the same spelling at both boundaries, input meaning wins.
    expression_roles = {**output_role_map, **input_role_map}

    def parameter(boundary: str, spec: ParameterSpec) -> dict[str, Any]:
        boundary_map = input_role_map if boundary == "input" else output_role_map
        return {
            "role": boundary_map[spec.name],
            "boundary": boundary,
            "semantic_type": str(spec.semantic_type).casefold(),
            "required": bool(spec.required),
            "runtime_resolvable": bool(spec.runtime_resolvable),
            "required_resolution": str(spec.required_resolution).casefold(),
        }

    def predicate(item: SemanticPredicate) -> dict[str, Any]:
        return {
            "predicate": str(item.predicate).casefold(),
            "args": {
                str(name): to_primitive(_rewrite_nested(value, expression_roles))
                for name, value in sorted(item.args.items())
            },
            "cardinality": int(item.cardinality),
            "distinct_by": str(item.distinct_by),
        }

    def predicates(items: list[SemanticPredicate]) -> list[dict[str, Any]]:
        result = [predicate(item) for item in items]
        return sorted(
            result,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    return {
        "inputs": sorted(
            (parameter("input", item) for item in atomic.inputs),
            key=lambda item: item["role"],
        ),
        "outputs": sorted(
            (parameter("output", item) for item in atomic.outputs),
            key=lambda item: item["role"],
        ),
        "preconditions": predicates(atomic.preconditions),
        "effects": predicates(atomic.effects),
        "validator_contract": to_primitive(
            _rewrite_validator_spec(
                atomic.validator_spec,
                input_role_map,
                output_role_map,
            )
        ),
    }


def canonical_atomic_contract(atomic: AbstractAtomicSkill) -> dict[str, Any]:
    """Return the one alpha-normalized Atomic identity payload."""

    inputs = _boundary_role_map(atomic, atomic.inputs, "input")
    outputs = _boundary_role_map(atomic, atomic.outputs, "output")
    return _identity_payload(atomic, inputs, outputs)


def atomic_contract_signature(atomic: AbstractAtomicSkill) -> str:
    """Stable cross-trace Atomic contract signature."""

    return content_hash(canonical_atomic_contract(atomic))


def aligned_role_maps(
    candidate: AbstractAtomicSkill,
    persisted: AbstractAtomicSkill,
) -> tuple[dict[str, str], dict[str, str]]:
    """Map candidate aliases onto an alpha-equivalent persisted schema."""

    if atomic_contract_signature(candidate) != atomic_contract_signature(persisted):
        raise ValueError("Atomic contracts are not alpha-equivalent")

    def align(
        candidate_specs: list[ParameterSpec],
        persisted_specs: list[ParameterSpec],
        prefix: str,
    ) -> dict[str, str]:
        candidate_neutral = _boundary_role_map(candidate, candidate_specs, prefix)
        persisted_neutral = _boundary_role_map(persisted, persisted_specs, prefix)
        persisted_by_neutral = {
            neutral: role for role, neutral in persisted_neutral.items()
        }
        if set(candidate_neutral.values()) != set(persisted_by_neutral):
            raise ValueError("alpha-equivalent Atomic role boundaries do not align")
        return {
            role: persisted_by_neutral[neutral]
            for role, neutral in candidate_neutral.items()
        }

    return (
        align(candidate.inputs, persisted.inputs, "input"),
        align(candidate.outputs, persisted.outputs, "output"),
    )


class AtomicContractCanonicalizer:
    """Canonicalize one Atomic and every artifact that names its roles."""

    def canonicalize(
        self,
        atomic: AbstractAtomicSkill,
        tool: ToolAsset | None = None,
        implementation: ImplementationAtom | None = None,
        *,
        input_role_map: Mapping[str, str] | None = None,
        output_role_map: Mapping[str, str] | None = None,
        atomic_ref: SkillRef | None = None,
    ) -> CanonicalizedAtomicBundle:
        input_roles = dict(input_role_map or _boundary_role_map(
            atomic, atomic.inputs, "input",
        ))
        output_roles = dict(output_role_map or _boundary_role_map(
            atomic, atomic.outputs, "output",
        ))
        expression_roles = {**output_roles, **input_roles}
        signature = content_hash(
            _identity_payload(atomic, input_roles, output_roles)
        )
        resolved_ref = atomic_ref or SkillRef(
            f"atomic_{signature[:24]}", "1.0.0",
        )
        canonical_atomic = replace(
            atomic,
            ref=resolved_ref,
            inputs=[_rewrite_parameter(item, input_roles) for item in atomic.inputs],
            outputs=[_rewrite_parameter(item, output_roles) for item in atomic.outputs],
            preconditions=[
                _rewrite_predicate(item, expression_roles)
                for item in atomic.preconditions
            ],
            effects=[
                _rewrite_predicate(item, expression_roles)
                for item in atomic.effects
            ],
            validator_spec=_rewrite_validator_spec(
                atomic.validator_spec,
                input_roles,
                output_roles,
            ),
        )
        canonical_tool = (
            None
            if tool is None
            else self._rewrite_tool(tool, input_roles, output_roles)
        )
        canonical_implementation = (
            None
            if implementation is None
            else self._rewrite_implementation(
                implementation,
                resolved_ref,
                input_roles,
                output_roles,
            )
        )
        # Unqualified lookup is convenient for disjoint boundary names.  When
        # an alias is shared, BindingExpression semantics are input-scoped;
        # callers rewriting output positions must use output_role_map.
        role_map = {**output_roles, **input_roles}
        return CanonicalizedAtomicBundle(
            canonical_atomic,
            role_map,
            canonical_tool,
            canonical_implementation,
            dict(input_roles),
            dict(output_roles),
        )

    @staticmethod
    def _rewrite_tool(
        tool: ToolAsset,
        input_roles: Mapping[str, str],
        output_roles: Mapping[str, str],
    ) -> ToolAsset:
        signature = _rewrite_schema_roles(tool.signature, input_roles)
        interface = copy.deepcopy(tool.interface)
        if isinstance(interface.get("output_schema"), dict):
            interface["output_schema"] = _rewrite_schema_roles(
                interface["output_schema"], output_roles,
            )
        artifact = copy.deepcopy(tool.artifact)
        steps = []
        for raw_step in artifact.get("steps", []) or []:
            step = dict(raw_step)
            step["argument_mapping"] = {
                str(argument): _rewrite_expression(expression, input_roles)
                for argument, expression in dict(
                    step.get("argument_mapping") or {}
                ).items()
            }
            steps.append(step)
        artifact["steps"] = steps
        artifact["output_mapping"] = {
            output_roles.get(str(role), str(role)): _rewrite_expression(
                expression, input_roles,
            )
            for role, expression in dict(
                artifact.get("output_mapping") or {}
            ).items()
        }
        tests: list[dict[str, Any]] = []
        for raw_case in tool.tests:
            case = copy.deepcopy(dict(raw_case))
            if isinstance(case.get("bindings"), dict):
                case["bindings"] = {
                    input_roles.get(str(role), str(role)): value
                    for role, value in case["bindings"].items()
                }
            if isinstance(case.get("effects"), list):
                case["effects"] = [
                    _rewrite_predicate(effect, input_roles)
                    if isinstance(effect, SemanticPredicate)
                    else _rewrite_nested(effect, input_roles)
                    for effect in case["effects"]
                ]
            tests.append(case)
        return replace(
            tool,
            signature=signature,
            interface=interface,
            artifact=artifact,
            tests=tests,
        )

    @staticmethod
    def _rewrite_implementation(
        implementation: ImplementationAtom,
        atomic_ref: SkillRef,
        input_roles: Mapping[str, str],
        output_roles: Mapping[str, str],
    ) -> ImplementationAtom:
        tool_bindings = []
        for binding in implementation.tool_bindings:
            tool_bindings.append(replace(
                binding,
                parameter_mapping={
                    input_roles.get(str(role), str(role)): _rewrite_expression(
                        expression, input_roles,
                    )
                    for role, expression in binding.parameter_mapping.items()
                },
            ))
        constraints = [
            replace(
                constraint,
                argument_mapping={
                    str(argument): _rewrite_expression(expression, input_roles)
                    for argument, expression in constraint.argument_mapping.items()
                },
            )
            for constraint in implementation.grounding_constraints
        ]
        policy = copy.deepcopy(implementation.execution_policy)
        output_mapping: dict[str, Any] = {}
        for role, raw_expression in dict(
            policy.get("output_mapping") or {}
        ).items():
            expression = BindingExpression.from_dict(raw_expression)
            role_source = (
                output_roles
                if expression.kind is BindingExprKind.TOOL_OUTPUT
                else input_roles
            )
            output_mapping[output_roles.get(str(role), str(role))] = (
                _rewrite_expression(expression, role_source)
            )
        policy["output_mapping"] = output_mapping
        return replace(
            implementation,
            abstract_ref=atomic_ref,
            tool_bindings=tool_bindings,
            grounding_constraints=constraints,
            execution_policy=policy,
        )

    def rewrite_canonical_occurrence(
        self,
        occurrence: Any,
        bundle: CanonicalizedAtomicBundle,
        *,
        atomic_ref: SkillRef | None = None,
    ) -> Any:
        """Rewrite the pre-E2 canonical occurrence without importing E1 IR."""

        expression_roles = {
            **bundle.output_role_map,
            **bundle.input_role_map,
        }
        changes = {
            "input_bindings": {
                bundle.input_role_map.get(str(role), str(role)): value
                for role, value in occurrence.input_bindings.items()
            },
            "output_bindings": {
                bundle.output_role_map.get(str(role), str(role)): value
                for role, value in occurrence.output_bindings.items()
            },
            "input_specs": [
                _rewrite_parameter(item, bundle.input_role_map)
                for item in occurrence.input_specs
            ],
            "output_specs": [
                _rewrite_parameter(item, bundle.output_role_map)
                for item in occurrence.output_specs
            ],
            "preconditions": [
                _rewrite_predicate(item, expression_roles)
                for item in occurrence.preconditions
            ],
            "effects": [
                _rewrite_predicate(item, expression_roles)
                for item in occurrence.effects
            ],
            "proposed_ref": atomic_ref or bundle.atomic.ref,
        }
        return replace(occurrence, **changes)

    @staticmethod
    def rewrite_graph_edge(
        edge: GraphEdge,
        *,
        source_role_map: Mapping[str, str],
        target_role_map: Mapping[str, str],
    ) -> GraphEdge:
        """Rewrite one DataFlow/dependency edge at its I/O boundaries."""

        return replace(
            edge,
            source_role=source_role_map.get(edge.source_role, edge.source_role),
            target_role=target_role_map.get(edge.target_role, edge.target_role),
        )

    @staticmethod
    def rewrite_edge_dict(
        edge: Mapping[str, Any],
        *,
        source_role_map: Mapping[str, str],
        target_role_map: Mapping[str, str],
    ) -> dict[str, Any]:
        """Mapping-form counterpart used for an E2 proposal before build."""

        result = dict(edge)
        source_role = str(result.get("source_role", ""))
        target_role = str(result.get("target_role", ""))
        result["source_role"] = source_role_map.get(source_role, source_role)
        result["target_role"] = target_role_map.get(target_role, target_role)
        return result

    def rewrite_composite(
        self,
        composite: CompositeSkill,
        role_bundles: Mapping[
            str, CanonicalizedAtomicBundle | Mapping[str, str]
        ],
    ) -> CompositeSkill:
        """Rewrite occurrence bindings, node refs, and all edge role endpoints.

        ``role_bundles`` may be keyed by step_id or occurrence_id.  Passing a
        full bundle is preferred; a plain map is supported for simple callers
        where input and output aliases are disjoint.
        """

        by_step = {item.step_id: item for item in composite.occurrences}

        def bundle_for(step_id: str) -> CanonicalizedAtomicBundle | Mapping[str, str]:
            occurrence = by_step[step_id]
            if step_id in role_bundles:
                return role_bundles[step_id]
            if occurrence.occurrence_id in role_bundles:
                return role_bundles[occurrence.occurrence_id]
            raise KeyError(f"missing canonical role bundle for {step_id}")

        def input_map(value: Any) -> Mapping[str, str]:
            return (
                value.input_role_map
                if isinstance(value, CanonicalizedAtomicBundle)
                else value
            )

        def output_map(value: Any) -> Mapping[str, str]:
            return (
                value.output_role_map
                if isinstance(value, CanonicalizedAtomicBundle)
                else value
            )

        occurrences: list[CompositeOccurrence] = []
        for occurrence in composite.occurrences:
            bundle = bundle_for(occurrence.step_id)
            target_inputs = input_map(bundle)
            bindings: dict[str, BindingExpression] = {}
            for target_role, raw_expression in occurrence.binding_specs.items():
                expression = BindingExpression.from_dict(raw_expression)
                source_roles = target_inputs
                if expression.kind is BindingExprKind.DATA_FLOW:
                    source_roles = output_map(bundle_for(expression.source_step))
                bindings[target_inputs.get(target_role, target_role)] = (
                    _rewrite_expression(expression, source_roles)
                )
            node_ref = (
                bundle.atomic.ref
                if isinstance(bundle, CanonicalizedAtomicBundle)
                else occurrence.node_ref
            )
            occurrences.append(replace(
                occurrence,
                node_ref=node_ref,
                binding_specs=bindings,
            ))

        def edge(raw: GraphEdge) -> GraphEdge:
            return self.rewrite_graph_edge(
                raw,
                source_role_map=output_map(bundle_for(raw.source_step)),
                target_role_map=input_map(bundle_for(raw.target_step)),
            )

        return replace(
            composite,
            occurrences=occurrences,
            data_edges=[edge(item) for item in composite.data_edges],
            dependency_edges=[edge(item) for item in composite.dependency_edges],
        )
