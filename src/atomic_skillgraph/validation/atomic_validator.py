"""Atomic effect, witness identity, and output validation."""

from __future__ import annotations

from typing import Any, Mapping

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

    @staticmethod
    def _output_derivations(atomic: AbstractAtomicSkill) -> dict[str, dict[str, Any]]:
        raw = dict(atomic.validator_spec.get("output_derivations") or {})
        if raw:
            return {str(role): dict(value) for role, value in raw.items()}
        migrated: dict[str, dict[str, Any]] = {}
        for item in list(atomic.validator_spec.get("output_identity") or []):
            output_role = str(item.get("output_role", ""))
            input_role = str(item.get("input_role", ""))
            if output_role and input_role:
                migrated[output_role] = {
                    "kind": "input_identity",
                    "input_role": input_role,
                }
        return migrated

    def validate_execution_result(
        self,
        atomic: AbstractAtomicSkill,
        occurrence: RuntimeOccurrence,
        bindings: dict[str, Any],
        tool_output_candidates: dict[str, Any],
        validator_channel: Any,
        *,
        current_revision: int,
        authoritative_evidence_facts: list[dict[str, Any]] | None = None,
    ) -> ValidationResult:
        """Validate generated outputs against Harness effect witnesses.

        Tool RETURN values are candidates only.  ``resolve_atomic_effect`` is
        the semantic authority that derives fresh outputs from current facts.
        """

        plain = self._plain(bindings)
        derivations = self._output_derivations(atomic)
        output_identity = [
            {
                "output_role": str(role),
                "input_role": str(derivation.get("input_role", "")),
            }
            for role, derivation in derivations.items()
            if derivation.get("kind") == "input_identity"
        ]
        output_roles = {item.name for item in atomic.outputs}
        candidate_outputs = {
            str(role): value
            for role, value in dict(tool_output_candidates or {}).items()
            if role in output_roles
        }
        for role, derivation in derivations.items():
            if derivation.get("kind") == "input_identity":
                input_role = str(derivation.get("input_role", ""))
                if role in candidate_outputs and repr(
                    candidate_outputs[role]
                ) != repr(plain.get(input_role)):
                    return ValidationResult(
                        "atomic", False,
                        {"output_derivation_consistent": False},
                        ["atomic_output_identity_mismatch"],
                        ["Tool output conflicts with input_identity derivation"],
                    )
                candidate_outputs[role] = plain.get(input_role)
        try:
            resolution = validator_channel.resolve_atomic_effect({
                "atomic_ref": str(atomic.ref),
                "occurrence_id": occurrence.occurrence_id,
                "effects": list(atomic.effects),
                "known_bindings": dict(plain),
                "semantic_anchors": {},
                "input_specs": list(atomic.inputs),
                "output_specs": list(atomic.outputs),
                "output_identity": output_identity,
                "preferred_values": [],
                "preferred_bindings": {},
                "authoritative_evidence_facts": list(
                    authoritative_evidence_facts or []
                ),
                "current_revision": int(current_revision),
            })
        except (KeyError, TypeError, ValueError) as exc:
            return ValidationResult(
                "atomic", False,
                {"effect_resolution_available": False},
                ["atomic_effect_violation"],
                [str(exc)],
            )
        if not resolution.passed:
            return ValidationResult(
                "atomic", False,
                dict(resolution.checks),
                ["atomic_effect_violation"],
                [resolution.message],
                witness_refs=list(resolution.witness_refs),
            )
        authoritative_outputs = {
            str(role): value
            for role, value in dict(resolution.output_candidates).items()
            if role in output_roles
        }
        # Fresh outputs can use a role name that differs from the predicate
        # argument name (for example ``found_entity`` feeding
        # ``entity.discovered_at(entity=...)``).  The Harness resolves that
        # binding while matching the declared Effect, so it is authoritative
        # even when the predicate's argument key cannot be projected into
        # ``output_candidates`` by name.
        for role, derivation in derivations.items():
            if (
                derivation.get("kind") == "effect_witness"
                and role in output_roles
                and role in resolution.resolved_bindings
            ):
                authoritative_outputs[role] = (
                    resolution.resolved_bindings[role]
                )
        for role, value in candidate_outputs.items():
            if role in authoritative_outputs and repr(
                authoritative_outputs[role]
            ) != repr(value):
                return ValidationResult(
                    "atomic", False,
                    {"tool_return": value, "effect_witness": authoritative_outputs[role]},
                    ["atomic_output_effect_witness_mismatch"],
                    ["Tool RETURN conflicts with authoritative Effect witness"],
                    witness_refs=list(resolution.witness_refs),
                )
        merged_outputs = {**authoritative_outputs, **candidate_outputs}
        merged_bindings = {
            **plain,
            **{
                role: value
                for role, value in resolution.resolved_bindings.items()
                if role not in plain
            },
        }
        final = self.validate(
            atomic, occurrence, merged_bindings, validator_channel,
            merged_outputs,
        )
        return ValidationResult(
            final.level,
            final.passed,
            {
                **resolution.checks,
                **final.checks,
                "generated_outputs_validated": final.passed,
            },
            final.failure_codes,
            final.messages,
            witness_refs=list(dict.fromkeys([
                *resolution.witness_refs,
                *final.witness_refs,
            ])),
            before_ref=final.before_ref,
            after_ref=final.after_ref,
        )

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
        preferred_bindings: Mapping[str, Any] | None = None,
        current_revision: int,
        authoritative_evidence_facts: list[dict[str, Any]] | None = None,
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
        input_roles = {item.name for item in atomic.inputs}
        output_roles = {item.name for item in atomic.outputs}
        derivations = self._output_derivations(atomic)
        output_identity = [
            {
                "output_role": str(role),
                "input_role": str(derivation.get("input_role", "")),
            }
            for role, derivation in derivations.items()
            if derivation.get("kind") == "input_identity"
        ]
        claims = {
            str(role): value
            for role, value in dict(preferred_bindings or {}).items()
        }
        unknown_claims = sorted(set(claims) - input_roles)
        if unknown_claims:
            return AtomicEffectResolution(
                False,
                failure_code="atomic_preferred_binding_role_invalid",
                message=(
                    "Agent-preferred bindings may reference only Atomic input "
                    f"roles; unknown roles: {unknown_claims!r}"
                ),
            )
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
            # Agent preference only: the Harness must match this claim against
            # current accepted-action-derived facts and must never turn it into
            # a synthetic fact or a replacement for a hard semantic anchor.
            "preferred_bindings": claims,
            "authoritative_evidence_facts": list(
                authoritative_evidence_facts or []
            ),
            "current_revision": current_revision,
        })
        if not resolution.passed:
            return resolution

        # Harness resolvers expose raw Effect source-role assignments.  In
        # v3.2 those source roles may name either an immutable Atomic input or
        # a fresh Atomic output.  This validator is the single adapter that
        # partitions the raw result into the two Runtime-facing channels.
        raw_resolved = {
            str(role): value
            for role, value in dict(resolution.resolved_bindings).items()
        }
        raw_outputs = {
            str(role): value
            for role, value in dict(resolution.output_candidates).items()
        }
        unknown_resolved_roles = sorted(
            set(raw_resolved) - input_roles - output_roles
        )
        if unknown_resolved_roles:
            return AtomicEffectResolution(
                False,
                witness_refs=list(resolution.witness_refs),
                checks=dict(resolution.checks),
                failure_code="atomic_effect_resolved_role_invalid",
                message=(
                    "Atomic effect resolver returned roles outside the "
                    "declared input/output boundary: "
                    f"{unknown_resolved_roles!r}"
                ),
            )
        unknown_output_roles = sorted(set(raw_outputs) - output_roles)
        if unknown_output_roles:
            return AtomicEffectResolution(
                False,
                witness_refs=list(resolution.witness_refs),
                checks=dict(resolution.checks),
                failure_code="atomic_effect_output_role_invalid",
                message=(
                    "Atomic effect resolver returned output candidates outside "
                    f"the declared output boundary: {unknown_output_roles!r}"
                ),
            )

        resolved_inputs = {
            role: value
            for role, value in raw_resolved.items()
            if role in input_roles
        }
        outputs = {
            role: value
            for role, value in raw_outputs.items()
            if role in output_roles
        }
        if not resolution.witness_refs:
            return AtomicEffectResolution(
                False,
                resolved_bindings=resolved_inputs,
                output_candidates=outputs,
                checks=dict(resolution.checks),
                failure_code="atomic_effect_witness_missing",
                message="Passed Atomic effect resolution requires validator witnesses",
            )

        merged_inputs = {**plain, **resolved_inputs}
        for output_role, derivation in derivations.items():
            kind = str(derivation.get("kind", ""))
            if kind == "input_identity":
                input_role = str(derivation.get("input_role", ""))
                if input_role not in merged_inputs:
                    continue
                if (
                    output_role in outputs
                    and outputs[output_role] != merged_inputs[input_role]
                ):
                    return AtomicEffectResolution(
                        False,
                        resolved_bindings=resolved_inputs,
                        output_candidates=outputs,
                        witness_refs=list(resolution.witness_refs),
                        checks=dict(resolution.checks),
                        failure_code="atomic_output_identity_mismatch",
                        message="Validator output conflicts with explicit output identity",
                    )
                outputs[output_role] = merged_inputs[input_role]
            elif kind == "effect_witness" and output_role in raw_resolved:
                witnessed_value = raw_resolved[output_role]
                if (
                    output_role in outputs
                    and outputs[output_role] != witnessed_value
                ):
                    return AtomicEffectResolution(
                        False,
                        resolved_bindings=resolved_inputs,
                        output_candidates=outputs,
                        witness_refs=list(resolution.witness_refs),
                        checks=dict(resolution.checks),
                        failure_code="atomic_output_effect_witness_mismatch",
                        message=(
                            "Validator output conflicts with authoritative "
                            "Effect witness"
                        ),
                    )
                outputs[output_role] = witnessed_value

        # Fresh outputs can be source roles in the Effect itself, so final
        # validation must see them even though they never enter input state.
        final_bindings = {**plain, **resolved_inputs, **outputs}
        final = self.validate(
            atomic,
            occurrence,
            final_bindings,
            validator_channel,
            outputs,
        )
        checks = {**resolution.checks, **final.checks}
        if not final.passed:
            return AtomicEffectResolution(
                False,
                resolved_bindings=resolved_inputs,
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
            resolved_bindings=resolved_inputs,
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
