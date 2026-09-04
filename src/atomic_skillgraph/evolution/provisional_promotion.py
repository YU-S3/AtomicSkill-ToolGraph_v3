"""Strict-success promotion of isolated Provisional Atomic trials.

Promotion is deliberately a compiler, not a copy operation: it rebuilds the
Atomic/Implementation/Tool candidate from the current successful task's
action span, validates that span, performs fresh source replay/admission, and
requires exact equality with the provisional contract signature.  It never
constructs a Composite.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable

from ..core.bindings import BindingExpression, BindingExprKind
from ..core.contracts import ParameterSpec, SemanticPredicate
from ..core.status import SkillStatus, ToolStatus
from ..knowledge.failure_knowledge_store import ProvisionalStatus
from ..runtime.cold_start_executor import ProvisionalTrialResult
from .atomicizer import AtomicOccurrenceProposal
from .contract_canonicalizer import (
    AtomicContractCanonicalizer,
    atomic_contract_signature,
)
from .portability import resolve_capability_label
from .tool_compiler import CompiledKnowledge, rewrite_capability_labels


@dataclass(frozen=True)
class PreparedPromotion:
    provisional_ref: str
    contract_signature: str
    compiled: CompiledKnowledge
    source_replay: dict[str, Any]
    action_span: tuple[int, int]
    task_id: str
    trace_id: str

    @property
    def candidate_refs(self) -> tuple[str, str, str]:
        return (
            str(self.compiled.atomic.ref),
            str(self.compiled.implementation.ref),
            str(self.compiled.tool.ref),
        )


@dataclass(frozen=True)
class PromotionRejection:
    provisional_ref: str
    code: str
    detail: str


def _expression_value(raw: Any, bindings: dict[str, Any]) -> Any:
    if isinstance(raw, BindingExpression):
        expression = raw
    elif isinstance(raw, dict) and "kind" in raw:
        expression = BindingExpression.from_dict(raw)
    elif isinstance(raw, str) and raw.startswith("$"):
        return bindings.get(raw[1:])
    else:
        return raw
    if expression.kind is BindingExprKind.CONSTANT:
        return expression.constant
    return bindings.get(expression.source_role)


def _predicate(raw: Any, bindings: dict[str, Any]) -> SemanticPredicate:
    if isinstance(raw, SemanticPredicate):
        value = raw
    else:
        value = SemanticPredicate(
            str(raw["predicate"]),
            dict(raw.get("args") or {}),
            int(raw.get("cardinality", 1)),
            str(raw.get("distinct_by", "")),
            raw.get("effect_domain", "world"),
        )
    return SemanticPredicate(
        value.predicate,
        {
            str(role): _expression_value(item, bindings)
            for role, item in value.args.items()
        },
        value.cardinality,
        value.distinct_by,
        value.effect_domain,
    )


def _parameter(raw: Any) -> ParameterSpec:
    return raw if isinstance(raw, ParameterSpec) else ParameterSpec(**dict(raw))


class ProvisionalPromotionCompiler:
    """Prepare replay-admitted Verified candidates from successful trial spans."""

    def __init__(
        self,
        *,
        normalizer: Any,
        atomicizer: Any,
        tool_compiler: Any,
        admission: Any,
        harness: Any,
        canonicalizer: AtomicContractCanonicalizer | None = None,
    ) -> None:
        self.normalizer = normalizer
        self.atomicizer = atomicizer
        self.tool_compiler = tool_compiler
        self.admission = admission
        self.harness = harness
        self.canonicalizer = canonicalizer or AtomicContractCanonicalizer()
        self.last_rejections: list[PromotionRejection] = []

    def prepare(
        self,
        trace: Any,
        successful_trials: list[ProvisionalTrialResult],
        *,
        provisional_lookup: Callable[[str], Any],
        task: Any,
    ) -> list[PreparedPromotion]:
        self.last_rejections = []
        if not bool(getattr(trace, "strict_task_success", False)):
            return []
        normalized = self.normalizer.build(trace)
        prepared: list[PreparedPromotion] = []
        seen: set[str] = set()
        for trial in successful_trials:
            if not trial.local_effect_passed:
                continue
            if trial.provisional_ref in seen:
                self._reject(
                    trial.provisional_ref,
                    "provisional_promotion_admission_failed",
                    "one strict-success task may promote a provisional ref once",
                )
                continue
            seen.add(trial.provisional_ref)
            provisional = provisional_lookup(trial.provisional_ref)
            status = ProvisionalStatus(provisional.status)
            if status not in {
                ProvisionalStatus.TRIAL_READY,
                ProvisionalStatus.TRIAL_SUPPORTED,
            }:
                self._reject(
                    trial.provisional_ref,
                    "provisional_promotion_admission_failed",
                    f"provisional status is not promotable: {status.value}",
                )
                continue
            try:
                proposal = self._proposal(provisional, trial)
                occurrence = self.atomicizer.validate_and_canonicalize(
                    [proposal], normalized,
                )[0]
                # Atomicizer infers semantic types from human-readable role
                # aliases.  Failure-side contracts already use alpha-neutral
                # names (input_000, ...), so retain the code-validated span
                # while restoring the authoritative provisional role specs.
                contract = dict(provisional.atomic_contract)
                occurrence = replace(
                    occurrence,
                    input_specs=[
                        _parameter(item)
                        for item in contract.get("inputs", ())
                    ],
                    output_specs=[
                        _parameter(item)
                        for item in contract.get("outputs", ())
                    ],
                )
                raw = self.tool_compiler.compile([occurrence])[0]
                bundle = self.canonicalizer.canonicalize(
                    raw.atomic, raw.tool, raw.implementation,
                )
                assert bundle.tool is not None and bundle.implementation is not None
                occurrence = self.canonicalizer.rewrite_canonical_occurrence(
                    occurrence, bundle, atomic_ref=bundle.atomic.ref,
                )
                compiled = CompiledKnowledge(
                    occurrence, bundle.atomic, bundle.tool, bundle.implementation,
                )
                compiled = rewrite_capability_labels(
                    compiled,
                    resolve_capability_label(occurrence, bundle.atomic),
                )
            except ValueError as exc:
                self._reject(
                    trial.provisional_ref,
                    "provisional_promotion_admission_failed",
                    str(exc),
                )
                continue

            signature = atomic_contract_signature(compiled.atomic)
            if signature != str(provisional.contract_signature):
                self._reject(
                    trial.provisional_ref,
                    "provisional_promotion_contract_mismatch",
                    (
                        f"expected {provisional.contract_signature}, "
                        f"compiled {signature}"
                    ),
                )
                continue

            admitted_tool = self.admission.admit_tool(
                compiled.tool,
                replay=lambda tool, case: bool(
                    self.harness.replay_tool(task, tool, case)
                ),
            )
            if admitted_tool.status is not ToolStatus.CANDIDATE:
                failures = list(
                    admitted_tool.metadata.get("admission_failure") or []
                )
                replay_failed = any("replay" in value for value in failures)
                self._reject(
                    trial.provisional_ref,
                    (
                        "provisional_promotion_replay_failed"
                        if replay_failed
                        else "provisional_promotion_admission_failed"
                    ),
                    ",".join(map(str, failures)) or "Tool admission rejected",
                )
                continue
            admitted_implementation = self.admission.admit_implementation(
                compiled.implementation,
                admitted_tool,
                atomic=compiled.atomic,
                harness=self.harness,
            )
            if admitted_implementation.status is not SkillStatus.CANDIDATE:
                self._reject(
                    trial.provisional_ref,
                    "provisional_promotion_admission_failed",
                    ",".join(map(
                        str,
                        admitted_implementation.quality.get(
                            "admission_failure", ()
                        ),
                    )) or "Implementation admission rejected",
                )
                continue

            metadata = {
                "origin": "promoted_from_provisional",
                "provisional_ref": trial.provisional_ref,
                "promotion_trace_id": str(trace.trace_id),
            }
            compiled = CompiledKnowledge(
                compiled.occurrence,
                replace(
                    compiled.atomic,
                    metadata={**compiled.atomic.metadata, **metadata},
                ),
                replace(
                    admitted_tool,
                    metadata={**admitted_tool.metadata, **metadata},
                ),
                replace(
                    admitted_implementation,
                    metadata={**admitted_implementation.metadata, **metadata},
                ),
            )
            prepared.append(PreparedPromotion(
                provisional_ref=trial.provisional_ref,
                contract_signature=signature,
                compiled=compiled,
                source_replay={
                    "passed": True,
                    "source_trace_id": str(trace.trace_id),
                    "event_range": list(trial.action_span),
                },
                action_span=tuple(map(int, trial.action_span)),
                task_id=str(trace.task.task_id),
                trace_id=str(trace.trace_id),
            ))
        return prepared

    @staticmethod
    def _proposal(
        provisional: Any,
        trial: ProvisionalTrialResult,
    ) -> AtomicOccurrenceProposal:
        start, end = map(int, trial.action_span)
        if start < 0 or end <= start:
            raise ValueError("promotion requires a non-empty current trial action span")
        contract = dict(provisional.atomic_contract)
        inputs = [_parameter(item) for item in contract.get("inputs", ())]
        outputs = [_parameter(item) for item in contract.get("outputs", ())]
        bindings = dict(trial.resolved_bindings)
        input_roles: dict[str, Any] = {}
        for spec in inputs:
            value = bindings.get(spec.name)
            if value is None:
                raise ValueError(
                    f"promotion trial lacks resolved input role: {spec.name}"
                )
            input_roles[spec.name] = value

        identity = {
            str(item.get("output_role", "")): str(item.get("input_role", ""))
            for item in dict(contract.get("validator_spec") or {}).get(
                "output_identity", ()
            )
            if isinstance(item, dict)
        }
        output_roles: dict[str, Any] = {}
        for spec in outputs:
            value = bindings.get(spec.name)
            if value is None and identity.get(spec.name):
                value = input_roles.get(identity[spec.name])
            if value is None:
                raise ValueError(
                    f"promotion trial lacks resolved output identity: {spec.name}"
                )
            output_roles[spec.name] = value
        all_bindings = {**input_roles, **output_roles}
        return AtomicOccurrenceProposal(
            phase_id=f"promotion::{trial.step_id}",
            intent=str(provisional.canonical_intent),
            event_start=start,
            event_end=end - 1,
            input_roles=input_roles,
            output_roles=output_roles,
            preconditions=[
                _predicate(item, all_bindings)
                for item in contract.get("preconditions", ())
            ],
            effects=[
                _predicate(item, all_bindings)
                for item in contract.get("effects", ())
            ],
            rationale=(
                "revalidate the current strict-success provisional trial span"
            ),
        )

    def _reject(self, ref: str, code: str, detail: str) -> None:
        self.last_rejections.append(PromotionRejection(ref, code, detail))


def commit_prepared_promotion(
    prepared: PreparedPromotion,
    *,
    store: Any,
    verified_refs: Iterable[str],
) -> Any:
    """Complete the lifecycle transition after normal registry persistence.

    The registry remains responsible for aligning/storing the three candidates.
    This function intentionally rejects Composite refs and lets every storage
    exception escape so the surrounding task checkpoint can roll back.
    """

    refs = tuple(map(str, verified_refs))
    skill_refs = [value for value in refs if value.startswith("skill://")]
    tool_refs = [value for value in refs if value.startswith("tool://")]
    valid_kinds = (
        len(refs) == 3
        and len(skill_refs) == 2
        and len(tool_refs) == 1
        and any("atomic_" in value.casefold() for value in skill_refs)
        and any("impl_" in value.casefold() for value in skill_refs)
        and not any("composite" in value.casefold() for value in refs)
    )
    if not valid_kinds:
        raise ValueError(
            "provisional promotion may record only Atomic/Implementation/Tool refs"
        )
    current_signature = atomic_contract_signature(prepared.compiled.atomic)
    if current_signature != prepared.contract_signature:
        raise ValueError("provisional_promotion_contract_mismatch")
    return store.promote_provisional(
        prepared.provisional_ref,
        refs,
        task_id=prepared.task_id,
        trace_id=prepared.trace_id,
        metadata={
            "origin": "promoted_from_provisional",
            "action_span": list(prepared.action_span),
            "source_replay": copy.deepcopy(prepared.source_replay),
        },
    )


__all__ = [
    "PreparedPromotion", "PromotionRejection",
    "ProvisionalPromotionCompiler", "commit_prepared_promotion",
]
