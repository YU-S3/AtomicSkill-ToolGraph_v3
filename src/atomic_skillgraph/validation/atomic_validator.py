"""Atomic effect, witness identity, and output validation."""

from __future__ import annotations

from typing import Any, Mapping

from ..core.bindings import BindingResolution, BindingStatus, BindingSource, RuntimeBinding, resolution_satisfies
from ..core.serialization import json_values_equal
from ..core.semantic_types import semantic_types_compatible
from dataclasses import replace
import copy
from ..core.contracts import AbstractAtomicSkill
from ..core.results import AtomicEffectResolution, RuntimeOccurrence, ValidationResult
from ..tooling.proposal import validate_output_semantic_constraints
from ..tooling.entry_contract import parameter_schema
from ..agents.protocol import validate_schema_instance


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
        semantic_compatible: Any = None,
    ) -> ValidationResult:
        """Check submitted values against joint Harness witnesses; never choose outputs."""

        plain = self._plain(bindings)
        for spec in atomic.inputs:
            source = bindings.get(spec.name)
            if isinstance(source, RuntimeBinding) and not semantic_types_compatible(source.semantic_type, spec.semantic_type):
                return ValidationResult.fail('atomic', 'atomic_input_type_mismatch', spec.name)
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
        candidate_outputs = dict(tool_output_candidates or {})
        if set(candidate_outputs) - output_roles:
            return ValidationResult("atomic", False, {}, ["unknown_outputs"], ["Undeclared output role"])
        for role, derivation in derivations.items():
            if derivation.get("kind") == "input_identity" and role not in candidate_outputs:
                source = str(derivation.get("input_role", ""))
                if source in plain:
                    candidate_outputs[role] = plain[source]
        missing = [p.name for p in atomic.outputs if p.required and p.name not in candidate_outputs]
        if missing:
            return ValidationResult("atomic", False, {}, ["missing_outputs"], [repr(missing)])
        try:
            validate_schema_instance(candidate_outputs, parameter_schema(atomic.outputs))
        except (ValueError, TypeError) as exc:
            return ValidationResult("atomic", False, {}, ["atomic_output_schema_invalid"], [str(exc)])
        constraints = atomic.validator_spec.get("output_semantic_constraints", {})
        preferred_bindings = {}
        if constraints:
            try:
                validate_output_semantic_constraints(atomic.inputs, atomic.outputs, constraints)
            except ValueError as exc:
                return ValidationResult("atomic", False, {}, ["atomic_output_semantic_constraint_invalid"], [str(exc)])
            input_specs = {p.name: p for p in atomic.inputs}
            for role, constraint in constraints.items():
                source = constraint["compatible_with_input"]
                if (role not in candidate_outputs or source not in plain or not callable(semantic_compatible)
                    or not semantic_compatible(role=source, concrete_value=candidate_outputs[role],
                        semantic_anchor=plain[source], semantic_type=input_specs[source].semantic_type)):
                    return ValidationResult("atomic", False, {"output_semantic_compatibility": False},
                        ["atomic_output_semantic_mismatch"], [f"Output {role} does not match its declared input anchor"])
            preferred_bindings = {p.name: candidate_outputs[p.name] for p in atomic.outputs
                if p.required_resolution in {"concrete", "relation_verified"}
                and p.name in candidate_outputs and p.name not in plain}
        for role, derivation in derivations.items():
            if derivation.get("kind") == "input_identity":
                input_role = str(derivation.get("input_role", ""))
                if role in candidate_outputs and not json_values_equal(candidate_outputs[role], plain.get(input_role)):
                    return ValidationResult(
                        "atomic", False,
                        {"output_derivation_consistent": False},
                        ["atomic_output_identity_mismatch"],
                        ["Tool output conflicts with input_identity derivation"],
                    )
                candidate_outputs[role] = plain.get(input_role)
        # Pin fresh output variables. Declared input-identity aliases are
        # checked above and pin their source input, not an invented effect
        # variable with the output's name.
        preferred_bindings = {
            (str(derivations[role]["input_role"])
             if derivations.get(role, {}).get("kind") == "input_identity" else role): value
            for role, value in candidate_outputs.items()
        }
        missing_inputs = [p.name for p in atomic.inputs if p.required and p.name not in plain]
        if missing_inputs:
            return ValidationResult("atomic", False, {}, ["missing_inputs"], [repr(missing_inputs)])
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
                "preferred_bindings": preferred_bindings,
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
        if resolution.passed and not resolution.witness_refs:
            return ValidationResult("atomic", False, {}, ["atomic_effect_witness_missing"],
                                    ["Effect validation requires authoritative witness refs"])
        invalid_inputs = set(resolution.resolved_bindings) - set(plain) - output_roles
        invalid_outputs = set(resolution.output_candidates) - output_roles
        if resolution.passed and invalid_inputs:
            return ValidationResult("atomic", False, {}, ["atomic_effect_resolved_role_invalid"],
                [f"Undeclared resolved roles: {sorted(invalid_inputs)}"])
        if resolution.passed and invalid_outputs:
            return ValidationResult("atomic", False, {}, ["atomic_effect_output_role_invalid"],
                [f"Undeclared output roles: {sorted(invalid_outputs)}"])
        if resolution.passed and any(
            role in resolution.resolved_bindings and not json_values_equal(resolution.resolved_bindings[role], value)
            for role, value in resolution.output_candidates.items()
        ):
            return ValidationResult("atomic", False, {}, ["atomic_output_effect_witness_mismatch"],
                ["Witness assignments disagree on a submitted output"])
        if resolution.passed and (
            set(resolution.output_candidates) - set(candidate_outputs)
            or set(resolution.resolved_bindings) - set(plain) - set(candidate_outputs)
        ):
            return ValidationResult("atomic", False, {}, ["missing_outputs"],
                ["Effect evidence cannot supply an unsubmitted boundary value"])
        if not resolution.passed:
            return ValidationResult(
                "atomic", False,
                dict(resolution.checks),
                [resolution.failure_code or "atomic_effect_violation"],
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
        input_specs = {item.name: item for item in atomic.inputs}
        for role, value in resolution.resolved_bindings.items():
            original = bindings.get(role)
            spec = input_specs.get(role)
            if role in plain and value != plain[role] and not resolution.witness_refs:
                return ValidationResult(
                    "atomic", False, {"input_assignment_witnessed": False},
                    ["atomic_effect_input_witness_missing"],
                    ["Changing a validation-local input assignment requires an authoritative witness"],
                )
            locked = (
                isinstance(original, RuntimeBinding)
                and original.resolution in {BindingResolution.CONCRETE, BindingResolution.RELATION_VERIFIED}
            ) or (spec is not None and spec.required_resolution != "semantic")
            if locked and role in plain and value != plain[role]:
                return ValidationResult(
                    "atomic", False, {"concrete_input_preserved": False},
                    ["atomic_effect_input_binding_conflict"],
                    ["Effect witness conflicts with an already concrete input"],
                    witness_refs=list(resolution.witness_refs),
                )
        # A semantic input remains immutable Tool input. Its concrete witness
        # is a local assignment for final validation only, already constrained
        # by the original known bindings in the Harness resolver request.
        merged_bindings = {
            **plain,
            **{
                role: value
                for role, value in resolution.resolved_bindings.items()
                if role in input_specs or role in output_roles
            },
        }
        final = self.validate(
            atomic, occurrence, merged_bindings, validator_channel,
            merged_outputs,
        )
        result = ValidationResult(
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
        if not result.passed:
            return result
        # Certify each value from its own original input or joint effect
        # assignment, not from the whole conjunction's passed flag.
        certified = {}
        for spec in atomic.inputs:
            role = spec.name
            if role not in plain:
                continue
            original = bindings.get(role)
            if isinstance(original, RuntimeBinding) and original.status is BindingStatus.GROUNDED:
                certified[role] = replace(copy.deepcopy(original), role=role, semantic_type=spec.semantic_type)
            else:
                witnessed = role in resolution.resolved_bindings and json_values_equal(resolution.resolved_bindings[role], plain[role])
                certified[role] = RuntimeBinding(role, copy.deepcopy(plain[role]), spec.semantic_type,
                    BindingSource.HARNESS_EVIDENCE, BindingStatus.GROUNDED,
                    BindingResolution.CONCRETE if witnessed else BindingResolution.SEMANTIC,
                    list(result.witness_refs) if witnessed else [], current_revision)
        outputs = {}
        for spec in atomic.outputs:
            role = spec.name
            if role not in merged_outputs:
                continue
            derivation = derivations.get(role, {})
            if derivation.get('kind') == 'input_identity':
                source = certified.get(str(derivation.get('input_role', '')))
                if source is None or not json_values_equal(source.value, merged_outputs[role]):
                    return ValidationResult.fail('atomic', 'atomic_output_identity_mismatch', role)
                binding = replace(copy.deepcopy(source), role=role, semantic_type=spec.semantic_type,
                    source=BindingSource.TOOL_OUTPUT)
            else:
                witnessed = role in authoritative_outputs and json_values_equal(authoritative_outputs[role], merged_outputs[role])
                if not witnessed:
                    return ValidationResult.fail('atomic', 'atomic_output_witness_missing', role)
                binding = RuntimeBinding(role, copy.deepcopy(merged_outputs[role]), spec.semantic_type,
                    BindingSource.TOOL_OUTPUT, BindingStatus.GROUNDED,
                    BindingResolution.RELATION_VERIFIED if spec.required_resolution == 'relation_verified' else BindingResolution.CONCRETE,
                    list(result.witness_refs), current_revision)
            if not resolution_satisfies(binding.resolution, spec.required_resolution):
                return ValidationResult.fail('atomic', 'atomic_output_resolution_insufficient', role)
            outputs[role] = binding
        result.validated_output_bindings = outputs
        result.certified_input_bindings = certified
        return result

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
        self, atomic, occurrence, bindings, validator_channel, *,
        semantic_anchors, preferred_values, preferred_bindings=None,
        candidate_outputs=None, current_revision, authoritative_evidence_facts=None,
        semantic_compatible=None,
    ) -> AtomicEffectResolution:
        """Validate only supplied inputs and outputs; never solve missing roles."""
        input_roles = {p.name for p in atomic.inputs}
        plain = {role: value.value if isinstance(value, RuntimeBinding) else value
                 for role, value in bindings.items()
                 if role in input_roles and (
                     not isinstance(value, RuntimeBinding) or value.status is BindingStatus.GROUNDED)}
        claims = dict(preferred_bindings or {})
        if set(claims) - {p.name for p in atomic.inputs}:
            return AtomicEffectResolution(False, failure_code="atomic_preferred_binding_role_invalid",
                                          message="Undeclared input role")
        for role, value in claims.items():
            anchor = semantic_anchors.get(role)
            if anchor is not None:
                original = anchor.value if isinstance(anchor, RuntimeBinding) else anchor
                spec = next(p for p in atomic.inputs if p.name == role)
                compatible = (semantic_compatible(role=role, concrete_value=value,
                    semantic_anchor=original, semantic_type=spec.semantic_type)
                    if callable(semantic_compatible) else value == original)
                if not compatible:
                    return AtomicEffectResolution(False, failure_code="runtime_semantic_anchor_mismatch",
                                                  message=f"Candidate conflicts with formal input: {role}")
        plain.update(claims)
        outputs = dict(candidate_outputs or {})
        for role, derivation in self._output_derivations(atomic).items():
            source = derivation.get("input_role")
            if derivation.get("kind") == "input_identity" and role not in outputs and source in plain:
                outputs[role] = plain[source]
        typed_inputs = {role: (bindings[role] if isinstance(bindings.get(role), RuntimeBinding)
            and role not in claims else value) for role, value in plain.items()}
        result = self.validate_execution_result(
            atomic, occurrence, typed_inputs, outputs, validator_channel,
            current_revision=current_revision,
            authoritative_evidence_facts=authoritative_evidence_facts,
            semantic_compatible=semantic_compatible,
        )
        return AtomicEffectResolution(
            result.passed, resolved_bindings=plain if result.passed else {},
            output_candidates=outputs if result.passed else {},
            witness_refs=list(result.witness_refs), checks=dict(result.checks),
            failure_code="" if result.passed else (result.failure_codes or ["atomic_effect_violation"])[0],
            message="" if result.passed else "; ".join(result.messages),
            validated_output_bindings=result.validated_output_bindings if result.passed else {},
            certified_input_bindings=result.certified_input_bindings if result.passed else {},
        )

    def already_satisfied(
        self, atomic: AbstractAtomicSkill, occurrence: RuntimeOccurrence,
        bindings: dict[str, RuntimeBinding | Any], validator_channel: Any,
    ) -> ValidationResult:
        return self.validate(atomic, occurrence, bindings, validator_channel, {
            spec.name: (bindings[spec.name].value if isinstance(bindings.get(spec.name), RuntimeBinding) else bindings.get(spec.name))
            for spec in atomic.outputs if spec.name in bindings
        })
