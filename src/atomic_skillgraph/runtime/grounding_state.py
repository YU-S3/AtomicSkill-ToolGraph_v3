"""Deterministic per-action Grounding Status authority.

This module narrows current harness evidence against formal Atomic contracts.
It may confirm a binding only when exactly one complete assignment survives all
hard checks.  Ambiguous assignments are projected to the Agent and never
selected here.
"""

from __future__ import annotations

import copy
from itertools import product
from typing import Any, Iterable, Mapping

from ..core.bindings import (
    BindingExprKind,
    BindingExpression,
    BindingResolution,
    BindingSource,
    BindingStatus,
    RuntimeBinding,
    resolution_satisfies,
)
from ..core.contracts import IdentityRelation, SemanticPredicate
from ..core.serialization import to_primitive
from .state import normalized_facts


_RESOLUTION_ORDER = {
    BindingResolution.SEMANTIC: 0,
    BindingResolution.CONCRETE: 1,
    BindingResolution.RELATION_VERIFIED: 2,
}


def _expression(value: Any) -> tuple[str, Any]:
    if isinstance(value, BindingExpression):
        expression = value
    elif isinstance(value, Mapping) and "kind" in value:
        expression = BindingExpression.from_dict(dict(value))
    elif isinstance(value, str) and value.startswith("$"):
        return "role", value[1:]
    else:
        return "constant", value
    if expression.kind is BindingExprKind.SKILL_INPUT:
        return "role", expression.source_role
    if expression.kind is BindingExprKind.CONSTANT:
        return "constant", expression.constant
    return "unsupported", None


def _predicate_parts(value: Any) -> tuple[str, dict[str, Any], int, str]:
    if isinstance(value, SemanticPredicate):
        return (
            value.predicate,
            dict(value.args),
            max(1, int(value.cardinality)),
            str(value.distinct_by),
        )
    mapping = dict(value or {})
    return (
        str(mapping.get("predicate", "")),
        dict(mapping.get("args") or {}),
        max(1, int(mapping.get("cardinality", 1))),
        str(mapping.get("distinct_by", "")),
    )


def _predicate_matches(
    predicate: Any,
    facts: Iterable[Mapping[str, Any]],
    bindings: Mapping[str, Any],
) -> list[dict[str, Any]]:
    name, raw_args, cardinality, distinct_by = _predicate_parts(predicate)
    matches: list[dict[str, Any]] = []
    for fact in facts:
        if str(fact.get("predicate", "")) != name:
            continue
        actual = dict(fact.get("args") or {})
        accepted = True
        for argument, raw in raw_args.items():
            kind, expected = _expression(raw)
            if kind == "unsupported":
                accepted = False
                break
            if kind == "role":
                expected = bindings.get(str(expected))
                if expected in (None, ""):
                    accepted = False
                    break
            if actual.get(argument) != expected:
                accepted = False
                break
        if accepted:
            matches.append(actual)
    if distinct_by:
        if len({
            item.get(distinct_by)
            for item in matches
            if item.get(distinct_by) not in (None, "")
        }) < cardinality:
            return []
    elif len(matches) < cardinality:
        return []
    return matches


def _fact_assignment(
    predicate: Any,
    fact: Mapping[str, Any],
    known: Mapping[str, Any],
) -> dict[str, Any] | None:
    name, raw_args, _cardinality, _distinct_by = _predicate_parts(predicate)
    if str(fact.get("predicate", "")) != name:
        return None
    actual = dict(fact.get("args") or {})
    assignment: dict[str, Any] = {}
    for argument, raw in raw_args.items():
        if argument not in actual:
            return None
        kind, expected = _expression(raw)
        if kind == "unsupported":
            return None
        if kind == "constant":
            if actual[argument] != expected:
                return None
            continue
        role = str(expected)
        if role in known and known[role] != actual[argument]:
            return None
        if role in assignment and assignment[role] != actual[argument]:
            return None
        assignment[role] = actual[argument]
    return assignment


class IncrementalGroundingAuthority:
    """Compute and optionally commit one unambiguous current assignment."""

    def __init__(self, invocation_compiler: Any, validation: Any) -> None:
        self.invocation_compiler = invocation_compiler
        self.validation = validation

    @staticmethod
    def _compatible(
        ctx: Any,
        occurrence: Any,
        role: str,
        value: Any,
        parameter: Any,
    ) -> bool:
        anchor = ctx.binding_store.semantic_anchor_for(
            role=role, occurrence=occurrence,
        )
        if anchor is None:
            return True
        check = getattr(ctx.harness, "semantic_value_compatible", None)
        if callable(check):
            return bool(check(
                role=role,
                concrete_value=value,
                semantic_anchor=anchor.value,
                semantic_type=parameter.semantic_type,
            ))
        return value == anchor.value

    @staticmethod
    def _add_candidate(
        candidates: dict[str, dict[Any, dict[str, Any]]],
        *,
        role: str,
        value: Any,
        evidence_ref: str,
        resolution: BindingResolution,
    ) -> None:
        if role not in candidates or value in (None, ""):
            return
        try:
            bucket = candidates[role].setdefault(value, {
                "evidence_refs": [],
                "resolution": resolution,
            })
        except TypeError:
            # Runtime entity bindings are expected to be JSON scalars.  An
            # unhashable catalog payload is not a concrete entity candidate.
            return
        if evidence_ref:
            bucket["evidence_refs"] = list(dict.fromkeys([
                *bucket["evidence_refs"], evidence_ref,
            ]))
        if _RESOLUTION_ORDER[resolution] > _RESOLUTION_ORDER[bucket["resolution"]]:
            bucket["resolution"] = resolution

    def _collect_candidates(
        self,
        occurrence: Any,
        atomic: Any,
        invocations: list[Any],
        ctx: Any,
        known: dict[str, Any],
        active_facts: list[dict[str, Any]],
    ) -> dict[str, dict[Any, dict[str, Any]]]:
        parameters = {item.name: item for item in atomic.inputs}
        candidates: dict[str, dict[Any, dict[str, Any]]] = {
            role: {} for role in parameters
        }
        occurrence_fact_roles: set[str] = set()
        for fact in active_facts:
            for predicate in [*atomic.preconditions, *atomic.effects]:
                assignment = _fact_assignment(predicate, fact, known)
                if assignment is None:
                    continue
                predicate_name = str(fact.get("predicate", ""))
                reference = (
                    f"occurrence_fact:{occurrence.occurrence_id}:"
                    f"r{ctx.world_revision}:{predicate_name}"
                )
                for role, value in assignment.items():
                    if role in parameters:
                        occurrence_fact_roles.add(role)
                        self._add_candidate(
                            candidates,
                            role=role,
                            value=value,
                            evidence_ref=reference,
                            resolution=BindingResolution.RELATION_VERIFIED,
                        )

        constraints = [
            constraint
            for invocation in invocations
            for constraint in invocation.spec.grounding_constraints
        ]
        for spec in ctx.action_catalog:
            arguments = dict(spec.arguments)
            for role in parameters:
                if role in arguments and role not in occurrence_fact_roles:
                    self._add_candidate(
                        candidates,
                        role=role,
                        value=arguments[role],
                        evidence_ref=(
                            f"entity:{ctx.world_revision}:{role}:"
                            f"{arguments[role]}:{spec.action_id}"
                        ),
                        resolution=BindingResolution.CONCRETE,
                    )
            for constraint in constraints:
                if constraint.action_type and constraint.action_type != spec.action_type:
                    continue
                mapped: list[tuple[str, Any]] = []
                valid = True
                for argument, raw_expression in constraint.argument_mapping.items():
                    if argument not in arguments:
                        valid = False
                        break
                    expression = BindingExpression.from_dict(raw_expression)
                    if expression.kind is BindingExprKind.CONSTANT:
                        if arguments[argument] != expression.constant:
                            valid = False
                            break
                    elif expression.kind is BindingExprKind.SKILL_INPUT:
                        mapped.append((expression.source_role, arguments[argument]))
                    else:
                        valid = False
                        break
                if not valid:
                    continue
                resolution = BindingResolution(constraint.required_resolution)
                reference = f"affordance:{ctx.world_revision}:{spec.action_id}"
                for role, value in mapped:
                    if role in parameters and role not in occurrence_fact_roles:
                        self._add_candidate(
                            candidates,
                            role=role,
                            value=value,
                            evidence_ref=reference,
                            resolution=resolution,
                        )

        for role, values in candidates.items():
            parameter = parameters[role]
            for value in list(values):
                if not self._compatible(
                    ctx, occurrence, role, value, parameter,
                ):
                    del values[value]
        return candidates

    @staticmethod
    def _identity_valid(ctx: Any, values: Mapping[str, Any]) -> bool:
        for constraint in ctx.task_contract.identity_constraints:
            if constraint.scope != "occurrence":
                continue
            if constraint.left_role not in values or constraint.right_role not in values:
                continue
            left = values[constraint.left_role]
            right = values[constraint.right_role]
            if constraint.relation is IdentityRelation.SAME_AS and left != right:
                return False
            if constraint.relation is IdentityRelation.DISTINCT_FROM and left == right:
                return False
        return True

    @staticmethod
    def _invocation_valid(
        invocation: Any,
        atomic: Any,
        ctx: Any,
        values: Mapping[str, Any],
        resolutions: dict[str, BindingResolution],
        refs: dict[str, list[str]],
    ) -> tuple[bool, dict[str, BindingResolution], dict[str, list[str]]]:
        resolved = dict(resolutions)
        evidence_refs = {role: list(value) for role, value in refs.items()}
        for constraint in invocation.spec.grounding_constraints:
            evidence = ctx.evidence_store.match_constraint(
                constraint, dict(values), ctx.world_revision,
            )
            if not evidence:
                return False, resolved, evidence_refs
            required = BindingResolution(constraint.required_resolution)
            for expression in constraint.argument_mapping.values():
                expression = BindingExpression.from_dict(expression)
                if expression.kind is not BindingExprKind.SKILL_INPUT:
                    continue
                role = expression.source_role
                if role not in values:
                    return False, resolved, evidence_refs
                if _RESOLUTION_ORDER[required] > _RESOLUTION_ORDER.get(
                    resolved.get(role, BindingResolution.SEMANTIC), 0,
                ):
                    resolved[role] = required
                evidence_refs.setdefault(role, []).extend(
                    item.evidence_id for item in evidence
                )
        for parameter in atomic.inputs:
            if not parameter.required:
                continue
            if parameter.name not in values:
                return False, resolved, evidence_refs
            if not resolution_satisfies(
                resolved.get(parameter.name, BindingResolution.SEMANTIC),
                parameter.required_resolution,
            ):
                return False, resolved, evidence_refs
        return True, resolved, {
            role: list(dict.fromkeys(value))
            for role, value in evidence_refs.items()
        }

    def _assignment_valid(
        self,
        occurrence: Any,
        atomic: Any,
        invocations: list[Any],
        ctx: Any,
        values: dict[str, Any],
        resolutions: dict[str, BindingResolution],
        refs: dict[str, list[str]],
        current_facts: list[dict[str, Any]],
    ) -> tuple[bool, dict[str, BindingResolution], dict[str, list[str]]]:
        if not self._identity_valid(ctx, values):
            return False, resolutions, refs
        if any(
            not _predicate_matches(item, current_facts, values)
            for item in atomic.preconditions
        ):
            return False, resolutions, refs
        admissible: tuple[
            bool, dict[str, BindingResolution], dict[str, list[str]]
        ] | None = None
        for invocation in invocations:
            passed, upgraded, upgraded_refs = self._invocation_valid(
                invocation, atomic, ctx, values, resolutions, refs,
            )
            if passed:
                admissible = (True, upgraded, upgraded_refs)
                break
        if invocations and admissible is None:
            return False, resolutions, refs
        # Required resolution is a property of the Atomic input contract, not
        # of whether a learned invocation happens to be available.  Seeded
        # execution has no invocation constraints to perform this check for
        # us, so enforce it at the common assignment boundary as well.
        effective_resolutions = (
            admissible[1] if admissible is not None else resolutions
        )
        for parameter in atomic.inputs:
            if not parameter.required:
                continue
            if parameter.name not in values or not resolution_satisfies(
                effective_resolutions.get(
                    parameter.name, BindingResolution.SEMANTIC,
                ),
                parameter.required_resolution,
            ):
                return False, resolutions, refs
        if not ctx.binding_store.repeat_candidate_compatible(
            occurrence.step_id, values,
        ):
            return False, resolutions, refs
        return admissible or (True, resolutions, refs)

    def refresh(
        self,
        occurrence: Any,
        atomic: Any,
        invocations: list[Any],
        ctx: Any,
        *,
        allow_auto_confirm: bool = True,
    ) -> dict[str, Any]:
        """Refresh all R3 Grounding fields and commit only a unique assignment."""

        current = ctx.binding_store.snapshot_for_node(occurrence)
        anchors = {
            parameter.name: anchor.value
            for parameter in atomic.inputs
            if (
                anchor := ctx.binding_store.semantic_anchor_for(
                    occurrence, parameter.name,
                )
            ) is not None
        }
        known = {
            role: binding.value
            for role, binding in current.items()
            if binding.status is BindingStatus.GROUNDED
        }
        invalidated = {
            role: {
                "value": binding.value,
                "resolution": binding.resolution.value,
                "invalidated_at_revision": int(binding.world_revision),
            }
            for role, binding in current.items()
            if binding.status is BindingStatus.INVALIDATED
        }
        try:
            active_facts = ctx.atomic_evidence_for(
                occurrence
            ).authoritative_facts()
        except (AttributeError, KeyError):
            active_facts = []
        current_facts = list(normalized_facts(
            ctx.harness.validator_channel().snapshot()
        ).values())
        candidates = self._collect_candidates(
            occurrence, atomic, invocations, ctx, known, active_facts,
        )
        missing_specs = [
            parameter
            for parameter in atomic.inputs
            if parameter.required
            and (
                parameter.name not in current
                or current[parameter.name].status is not BindingStatus.GROUNDED
                or not resolution_satisfies(
                    current[parameter.name].resolution,
                    parameter.required_resolution,
                )
            )
        ]

        valid_assignments: list[
            tuple[dict[str, Any], dict[str, BindingResolution], dict[str, list[str]]]
        ] = []
        candidate_lists = [
            list(candidates[item.name]) for item in missing_specs
        ]
        assignments_evaluated = bool(missing_specs and all(candidate_lists))
        if assignments_evaluated:
            for selected in product(*candidate_lists):
                proposed = dict(zip(
                    (item.name for item in missing_specs), selected,
                ))
                values = {**known, **proposed}
                resolutions = {
                    role: binding.resolution
                    for role, binding in current.items()
                    if binding.status is BindingStatus.GROUNDED
                }
                refs = {
                    role: list(binding.evidence_refs)
                    for role, binding in current.items()
                    if binding.status is BindingStatus.GROUNDED
                }
                for role, value in proposed.items():
                    detail = candidates[role][value]
                    resolutions[role] = detail["resolution"]
                    refs[role] = list(detail["evidence_refs"])
                passed, resolutions, refs = self._assignment_valid(
                    occurrence,
                    atomic,
                    invocations,
                    ctx,
                    values,
                    resolutions,
                    refs,
                    current_facts,
                )
                if passed:
                    valid_assignments.append((proposed, resolutions, refs))

        assignment_committed = bool(
            allow_auto_confirm and len(valid_assignments) == 1
        )
        if assignment_committed:
            proposed, resolutions, refs = valid_assignments[0]
            parameters = {item.name: item for item in atomic.inputs}
            ctx.binding_store.commit_grounded(
                occurrence.occurrence_id,
                {
                    role: RuntimeBinding(
                        role=role,
                        value=value,
                        semantic_type=parameters[role].semantic_type,
                        source=BindingSource.HARNESS_EVIDENCE,
                        status=BindingStatus.GROUNDED,
                        resolution=resolutions[role],
                        evidence_refs=list(dict.fromkeys(refs[role])),
                        world_revision=ctx.world_revision,
                    )
                    for role, value in proposed.items()
                },
            )
            current = ctx.binding_store.snapshot_for_node(occurrence)
            known = {
                role: binding.value
                for role, binding in current.items()
                if binding.status is BindingStatus.GROUNDED
            }

        confirmed = {
            parameter.name: current[parameter.name].value
            for parameter in atomic.inputs
            if parameter.name in current
            and current[parameter.name].status is BindingStatus.GROUNDED
            and resolution_satisfies(
                current[parameter.name].resolution,
                parameter.required_resolution,
            )
        }
        missing = [
            parameter.name
            for parameter in atomic.inputs
            if parameter.required and parameter.name not in confirmed
        ]
        if assignment_committed:
            projected_candidates: dict[str, list[Any]] = {}
        else:
            source_assignments = (
                [item[0] for item in valid_assignments]
                if assignments_evaluated
                else [
                    {role: value}
                    for role, values in candidates.items()
                    for value in values
                ]
            )
            projected_candidates = {
                role: sorted(
                    {
                        item[role]
                        for item in source_assignments
                        if role in item
                    },
                    key=lambda value: repr(value),
                )
                for role in missing
                if any(role in item for item in source_assignments)
            }

        preconditions: list[dict[str, Any]] = []
        for predicate in atomic.preconditions:
            matched = _predicate_matches(predicate, current_facts, known)
            status = "satisfied" if matched else "missing"
            if not matched and valid_assignments and any(
                _predicate_matches(predicate, current_facts, {**known, **item[0]})
                for item in valid_assignments
            ):
                status = "candidate_satisfied"
            preconditions.append({
                "predicate": _predicate_parts(predicate)[0],
                "status": status,
            })

        semantic_anchors = {
            parameter.name: anchor
            for parameter in atomic.inputs
            if (anchor := ctx.binding_store.semantic_anchor_for(
                occurrence, parameter.name,
            )) is not None
        }
        effect = self.validation.atomic.resolve_current_effect(
            atomic,
            occurrence,
            current,
            ctx.harness.validator_channel(),
            semantic_anchors=semantic_anchors,
            preferred_values=[],
            preferred_bindings={},
            authoritative_evidence_facts=active_facts,
            current_revision=ctx.world_revision,
        )
        effect_status = {
            "passed": bool(effect.passed),
            "failure_code": str(effect.failure_code),
            "resolved_bindings": copy.deepcopy(effect.resolved_bindings),
            "witness_refs": list(effect.witness_refs),
        }

        ready = False
        if invocations and not missing:
            resolutions = {
                role: binding.resolution
                for role, binding in current.items()
                if binding.status is BindingStatus.GROUNDED
            }
            refs = {
                role: list(binding.evidence_refs)
                for role, binding in current.items()
                if binding.status is BindingStatus.GROUNDED
            }
            ready = self._assignment_valid(
                occurrence,
                atomic,
                invocations,
                ctx,
                dict(known),
                resolutions,
                refs,
                current_facts,
            )[0]

        blocking: list[str] = []
        for role in missing:
            values = projected_candidates.get(role, [])
            if len(values) > 1:
                blocking.append(f"multiple_valid_{role}_candidates")
            elif not values:
                blocking.append(f"missing_{role}")
        if any(item["status"] == "missing" for item in preconditions):
            blocking.append("preconditions_not_satisfied")
        if invocations and not ready:
            blocking.append("learned_invocation_not_ready")

        state = {
            "revision": int(ctx.world_revision),
            "occurrence_id": str(occurrence.occurrence_id),
            "semantic_anchors": copy.deepcopy(anchors),
            "confirmed_bindings": copy.deepcopy(confirmed),
            "candidate_bindings": copy.deepcopy(projected_candidates),
            "missing_bindings": list(missing),
            "invalidated_bindings": copy.deepcopy(invalidated),
            "precondition_status": preconditions,
            "effect_witness_status": effect_status,
            "learned_invocation_ready": bool(ready),
            "blocking_reasons": list(dict.fromkeys(blocking)),
        }
        record = getattr(ctx, "record_grounding_state", None)
        if callable(record):
            record(occurrence.occurrence_id, to_primitive(state))
        return state


__all__ = ["IncrementalGroundingAuthority"]
