"""Atomic effect, witness identity, and output validation."""

from __future__ import annotations

from typing import Any

from ..core.bindings import RuntimeBinding
from ..core.contracts import AbstractAtomicSkill
from ..core.results import RuntimeOccurrence, ValidationResult


class AtomicValidator:
    def validate(
        self, atomic: AbstractAtomicSkill, occurrence: RuntimeOccurrence,
        bindings: dict[str, RuntimeBinding | Any], validator_channel: Any,
        output_candidates: dict[str, Any] | None = None,
    ) -> ValidationResult:
        plain = {role: value.value if isinstance(value, RuntimeBinding) else value for role, value in bindings.items()}
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

    def already_satisfied(
        self, atomic: AbstractAtomicSkill, occurrence: RuntimeOccurrence,
        bindings: dict[str, RuntimeBinding | Any], validator_channel: Any,
    ) -> ValidationResult:
        return self.validate(atomic, occurrence, bindings, validator_channel, {
            spec.name: (bindings[spec.name].value if isinstance(bindings.get(spec.name), RuntimeBinding) else bindings.get(spec.name))
            for spec in atomic.outputs if spec.name in bindings
        })
