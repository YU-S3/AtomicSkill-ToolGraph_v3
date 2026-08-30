"""Atomic effect, witness identity, and output validation."""

from __future__ import annotations

from typing import Any

from ..core.bindings import BindingStatus, RuntimeBinding
from ..core.contracts import AbstractAtomicSkill
from ..core.results import AtomicEffectResolution, RuntimeOccurrence, ValidationResult


class AtomicValidator:
    @staticmethod
    def _plain(bindings: dict[str, RuntimeBinding | Any]) -> dict[str, Any]:
        return {
            role: value.value if isinstance(value, RuntimeBinding) else value
            for role, value in bindings.items()
        }

    def validate(
        self, atomic: AbstractAtomicSkill, occurrence: RuntimeOccurrence,
        bindings: dict[str, RuntimeBinding | Any], validator_channel: Any,
        output_candidates: dict[str, Any] | None = None,
    ) -> ValidationResult:
        plain = self._plain(bindings)
        request = {
            "atomic_ref": str(atomic.ref), "occurrence_id": occurrence.occurrence_id,
            "effects": atomic.effects, "bindings": plain,
            "output_candidates": dict(output_candidates or {}),
        }
        result = validator_channel.validate_atomic_effect(request)
        if not result.passed:
            return result
        checks = dict(result.checks)
        identity_values: dict[str, Any] = {}
        identity_ok = True
        for effect in atomic.effects:
            for role, expression in effect.args.items():
                source_role = getattr(expression, "source_role", role) or role
                if source_role not in plain:
                    continue
                if source_role in identity_values and identity_values[source_role] != plain[source_role]:
                    identity_ok = False
                identity_values[source_role] = plain[source_role]
        checks["effect_identity_consistent"] = identity_ok
        output_candidates = output_candidates or {}
        outputs_ok = all(not spec.required or spec.name in output_candidates or spec.name in plain for spec in atomic.outputs)
        checks["required_outputs_present"] = outputs_ok
        passed = identity_ok and outputs_ok
        return ValidationResult(
            "atomic", passed, checks,
            [] if passed else ["atomic_effect_violation"],
            [] if passed else ["Atomic witness identity or required outputs are inconsistent"],
            witness_refs=list(result.witness_refs), before_ref=result.before_ref, after_ref=result.after_ref,
        )

    def resolve_current_effect(
        self,
        atomic: AbstractAtomicSkill,
        occurrence: RuntimeOccurrence,
        bindings: dict[str, RuntimeBinding | Any],
        validator_channel: Any,
        *,
        semantic_anchors: dict[str, RuntimeBinding | Any],
        preferred_values: list[Any],
        current_revision: int,
    ) -> AtomicEffectResolution:
        """Resolve current action facts, then run the ordinary Atomic validator."""

        plain = {
            role: value.value if isinstance(value, RuntimeBinding) else value
            for role, value in bindings.items()
            if not isinstance(value, RuntimeBinding)
            or value.status is BindingStatus.GROUNDED
        }
        anchors = {
            role: value.value if isinstance(value, RuntimeBinding) else value
            for role, value in semantic_anchors.items()
            if (
                (not isinstance(value, RuntimeBinding)
                 or value.status is BindingStatus.GROUNDED)
                and (value.value if isinstance(value, RuntimeBinding) else value)
                not in (None, "")
            )
        }
        output_identity = list(
            atomic.validator_spec.get("output_identity") or []
        )
        input_roles = {item.name for item in atomic.inputs}
        output_roles = {item.name for item in atomic.outputs}
        for item in output_identity:
            output_role = str(item.get("output_role", ""))
            input_role = str(item.get("input_role", ""))
            if output_role not in output_roles or input_role not in input_roles:
                return AtomicEffectResolution(
                    False,
                    failure_code="atomic_output_identity_invalid",
                    message="Atomic output identity references an unknown boundary role",
                )
        resolution = validator_channel.resolve_atomic_effect({
            "atomic_ref": str(atomic.ref),
            "occurrence_id": occurrence.occurrence_id,
            "effects": list(atomic.effects),
            "known_bindings": plain,
            "semantic_anchors": anchors,
            "input_specs": list(atomic.inputs),
            "output_specs": list(atomic.outputs),
            "output_identity": output_identity,
            "preferred_values": list(preferred_values),
            "current_revision": current_revision,
        })
        if not resolution.passed:
            return resolution
        if not resolution.witness_refs:
            return AtomicEffectResolution(
                False,
                resolved_bindings=dict(resolution.resolved_bindings),
                output_candidates=dict(resolution.output_candidates),
                checks=dict(resolution.checks),
                failure_code="atomic_effect_witness_missing",
                message="Passed Atomic effect resolution requires validator witnesses",
            )

        merged = {**plain, **resolution.resolved_bindings}
        outputs = dict(resolution.output_candidates)
        for item in output_identity:
            output_role = str(item.get("output_role", ""))
            input_role = str(item.get("input_role", ""))
            if output_role and input_role in merged:
                if (
                    output_role in outputs
                    and outputs[output_role] != merged[input_role]
                ):
                    return AtomicEffectResolution(
                        False,
                        resolved_bindings=dict(resolution.resolved_bindings),
                        output_candidates=outputs,
                        witness_refs=list(resolution.witness_refs),
                        checks=dict(resolution.checks),
                        failure_code="atomic_output_identity_mismatch",
                        message="Validator output conflicts with explicit output identity",
                    )
                outputs[output_role] = merged[input_role]
        # Contextual outputs without an identity mapping may still be supplied
        # directly by a validator resolver.  No role-name substring heuristic
        # is used here.
        final = self.validate(
            atomic,
            occurrence,
            merged,
            validator_channel,
            outputs,
        )
        checks = {**resolution.checks, **final.checks}
        if not final.passed:
            return AtomicEffectResolution(
                False,
                resolved_bindings=dict(resolution.resolved_bindings),
                output_candidates=outputs,
                witness_refs=list(resolution.witness_refs),
                checks=checks,
                failure_code=(
                    final.failure_codes[0]
                    if final.failure_codes
                    else "atomic_effect_violation"
                ),
                message=(
                    final.messages[0]
                    if final.messages
                    else "Resolved effect failed final Atomic validation"
                ),
            )
        return AtomicEffectResolution(
            True,
            resolved_bindings=dict(resolution.resolved_bindings),
            output_candidates=outputs,
            witness_refs=list(dict.fromkeys([
                *resolution.witness_refs,
                *final.witness_refs,
            ])),
            checks=checks,
        )

    def already_satisfied(
        self, atomic: AbstractAtomicSkill, occurrence: RuntimeOccurrence,
        bindings: dict[str, RuntimeBinding | Any], validator_channel: Any,
    ) -> ValidationResult:
        return self.validate(atomic, occurrence, bindings, validator_channel, {
            spec.name: (bindings[spec.name].value if isinstance(bindings.get(spec.name), RuntimeBinding) else bindings.get(spec.name))
            for spec in atomic.outputs if spec.name in bindings
        })
