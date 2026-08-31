"""Code authority for F1 plan alignment and F2 failure assets."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Callable

from ..core.contracts import SemanticPredicate
from ..core.errors import PlannerProposalError
from ..core.refs import content_hash
from ..core.results import ValidationResult
from ..core.serialization import to_primitive
from ..core.status import ToolStatus
from ..knowledge.failure_knowledge_store import (
    FailureExperience,
    FailureExperienceStatus,
    ProvisionalAtomicRecord,
    ProvisionalStatus,
    provisional_ref_for,
)
from ..planner.cold_start_retriever import task_cluster_signature
from ..planner.multiplicity import requirement_instance_shape_id
from .atomicizer import AtomicOccurrenceProposal
from .contract_canonicalizer import (
    AtomicContractCanonicalizer,
    atomic_contract_signature,
)
from .failure_extractor_session import (
    FailureAtomicProposal,
    FailureExtractionProposal,
    FailureExtractorSession,
    FailurePlanAlignment,
    PlanStepAlignment,
)
from .portability import resolve_capability_label
from .tool_compiler import CompiledKnowledge, rewrite_capability_labels


def _authoritative_witnesses(trace: Any) -> set[str]:
    return {
        str(ref)
        for validation in trace.validations
        for ref in dict(validation.result).get("witness_refs", ())
    }


def _portable(value: Any) -> bool:
    if isinstance(value, dict):
        forbidden = {
            "actions", "action_list", "source_actions", "concrete_bindings",
            "source_task_id", "source_trace_id",
            "action", "source_action_strings", "admissible_commands",
            "raw_action", "observation", "game_file", "env_index",
            "tool", "tool_ref", "tool_body",
            "implementation", "implementation_ref", "composite",
            "composite_ref", "primitive_steps", "steps", "replay_script",
        }
        return not (forbidden & set(map(str, value))) and all(
            _portable(item) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return all(_portable(item) for item in value)
    if isinstance(value, str):
        return not bool(re.search(r"(?:_|\s)\d+\b", value))
    return True


@dataclass
class FailureExtractionValidation:
    proposal: FailureExtractionProposal
    result: ValidationResult
    source_replays: dict[str, dict[str, Any]]
    provisional_rejections: list[dict[str, Any]]
    failure_experience_accepted: bool


@dataclass(frozen=True)
class FailureExtractionEligibility:
    """Deterministic gate for the failure-side extractor.

    The caller must provide the *validated* ColdStartPlan bit.  Merely having
    a model proposal is intentionally insufficient.
    """

    cold_start_enabled: bool
    valid_cold_start_plan: bool
    strict_task_success: bool
    infrastructure_failure: bool
    runtime_mode: str

    @property
    def passed(self) -> bool:
        mode = str(getattr(self.runtime_mode, "value", self.runtime_mode))
        return (
            self.cold_start_enabled
            and self.valid_cold_start_plan
            and not self.strict_task_success
            and not self.infrastructure_failure
            and mode.casefold() == "online"
        )


@dataclass
class PreparedFailureExtraction:
    """Validated failure assets staged before any durable mutation."""

    alignment: FailurePlanAlignment | None
    f1_validation: ValidationResult | None
    f2_validation: FailureExtractionValidation | None
    provisional_records: list[ProvisionalAtomicRecord]
    failure_experience: FailureExperience | None
    rejection: dict[str, Any]

    @property
    def accepted(self) -> bool:
        return (
            not self.rejection
            and (
                bool(self.provisional_records)
                or self.failure_experience is not None
            )
        )

    def commit(self, store: Any) -> tuple[list[str], list[str]]:
        """Persist a validated stage; storage/program errors deliberately escape."""

        if not self.accepted:
            return [], []
        provisional_refs = [
            str(store.upsert_provisional(record).provisional_ref)
            for record in self.provisional_records
        ]
        experience_ids: list[str] = []
        if self.failure_experience is not None:
            experience_ids.append(str(
                store.upsert_failure_experience(
                    self.failure_experience,
                ).experience_id
            ))
        return provisional_refs, experience_ids


def _predicate(value: dict[str, Any]) -> SemanticPredicate:
    return SemanticPredicate(
        str(value["predicate"]),
        dict(value.get("args") or {}),
        int(value.get("cardinality", 1)),
        str(value.get("distinct_by", "")),
    )


def _atomic_occurrence_proposal(item: FailureAtomicProposal) -> AtomicOccurrenceProposal:
    value = item.atomic_proposal
    start = int(value["event_start"])
    end_exclusive = int(value["event_end"])
    if end_exclusive <= start:
        raise ValueError("failure Atomic event_end must be exclusive")
    return AtomicOccurrenceProposal(
        phase_id=str(value["phase_id"]),
        intent=str(value["intent"]),
        event_start=start,
        event_end=end_exclusive - 1,
        input_roles=dict(value["input_roles"]),
        output_roles=dict(value["output_roles"]),
        preconditions=[_predicate(raw) for raw in value["preconditions"]],
        effects=[_predicate(raw) for raw in value["effects"]],
        rationale=str(value["rationale"]),
    )


class FailureAtomicSourceReplay:
    """Compile and freshly replay an F2 Atomic without registering it.

    The transient Tool is used only as a deterministic replay vehicle.  It is
    never returned to the success registry, which preserves failure-side
    isolation.
    """

    def __init__(
        self,
        *,
        trace: Any,
        task: Any,
        normalizer: Any,
        atomicizer: Any,
        tool_compiler: Any,
        admission: Any,
        harness: Any,
        canonicalizer: AtomicContractCanonicalizer | None = None,
    ) -> None:
        self.trace = trace
        self.task = task
        self.normalized = normalizer.build(trace)
        self.atomicizer = atomicizer
        self.tool_compiler = tool_compiler
        self.admission = admission
        self.harness = harness
        self.canonicalizer = canonicalizer or AtomicContractCanonicalizer()
        self._compiled: dict[str, CompiledKnowledge] = {}
        self._replays: dict[str, dict[str, Any]] = {}

    def __call__(self, item: FailureAtomicProposal) -> dict[str, Any]:
        phase_id = str(item.atomic_proposal.get("phase_id", ""))
        if phase_id in self._replays:
            return copy.deepcopy(self._replays[phase_id])
        try:
            proposal = _atomic_occurrence_proposal(item)
            occurrence = self.atomicizer.validate_and_canonicalize(
                [proposal], self.normalized,
            )[0]
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
            label = resolve_capability_label(occurrence, bundle.atomic)
            compiled = rewrite_capability_labels(compiled, label)
        except ValueError as exc:
            result = {
                "passed": False,
                "failure_code": "failure_extractor_atomic_invalid",
                "detail": str(exc),
            }
            self._replays[phase_id] = result
            return copy.deepcopy(result)

        admitted = self.admission.admit_tool(
            compiled.tool,
            replay=lambda tool, case: bool(
                self.harness.replay_tool(self.task, tool, case)
            ),
        )
        passed = admitted.status is ToolStatus.CANDIDATE
        result = {
            "passed": passed,
            "failure_code": "" if passed else "provisional_source_replay_failed",
            "source_trace_id": str(self.trace.trace_id),
            "event_range": [
                compiled.occurrence.event_start,
                compiled.occurrence.event_end + 1,
            ],
            "admission_failures": list(
                admitted.metadata.get("admission_failure") or []
            ),
        }
        if passed:
            self._compiled[phase_id] = CompiledKnowledge(
                compiled.occurrence,
                compiled.atomic,
                admitted,
                compiled.implementation,
            )
        self._replays[phase_id] = result
        return copy.deepcopy(result)

    def compiled_for(self, item: FailureAtomicProposal) -> CompiledKnowledge:
        phase_id = str(item.atomic_proposal.get("phase_id", ""))
        if phase_id not in self._compiled:
            raise KeyError(f"failure Atomic was not replay-admitted: {phase_id}")
        return self._compiled[phase_id]


class FailureAssetRecordBuilder:
    """Convert code-validated F2 output into isolated store records."""

    def __init__(
        self,
        source_replay: FailureAtomicSourceReplay,
        *,
        task_contract: Any,
        requirement_expansion: Any,
        cold_start_plan: Any,
        trace: Any,
        harness_profile: str,
    ) -> None:
        self.source_replay = source_replay
        self.task_contract = task_contract
        self.requirement_expansion = requirement_expansion
        self.cold_start_plan = cold_start_plan
        self.trace = trace
        self.harness_profile = str(harness_profile)

    @staticmethod
    def _atomic_contract(compiled: CompiledKnowledge) -> dict[str, Any]:
        atomic = compiled.atomic
        return {
            "summary": str(atomic.summary),
            "inputs": to_primitive(atomic.inputs),
            "outputs": to_primitive(atomic.outputs),
            "preconditions": to_primitive(atomic.preconditions),
            "effects": to_primitive(atomic.effects),
            "validator_spec": to_primitive(atomic.validator_spec),
        }

    def build(
        self,
        alignment: FailurePlanAlignment,
        validation: FailureExtractionValidation,
    ) -> tuple[list[ProvisionalAtomicRecord], FailureExperience | None]:
        records: list[ProvisionalAtomicRecord] = []
        for item in validation.proposal.provisional_atomics:
            compiled = self.source_replay.compiled_for(item)
            signature = atomic_contract_signature(compiled.atomic)
            replay = copy.deepcopy(validation.source_replays[
                str(item.atomic_proposal["phase_id"])
            ])
            records.append(ProvisionalAtomicRecord(
                provisional_ref=provisional_ref_for(signature),
                contract_signature=signature,
                canonical_intent=str(
                    compiled.atomic.metadata.get("canonical_intent")
                    or compiled.occurrence.intent
                ),
                atomic_contract=self._atomic_contract(compiled),
                seeded_guideline={
                    "intent": "establish the declared local Effect",
                    "parameter_flow": "preserve declared role identity",
                },
                harness_profile=self.harness_profile,
                source_trace_id=str(self.trace.trace_id),
                source_task_id=str(self.trace.task.task_id),
                source_span={
                    "event_start": compiled.occurrence.event_start,
                    "event_end": compiled.occurrence.event_end + 1,
                    "witness_refs": list(compiled.occurrence.validation_refs),
                },
                source_replay={
                    "passed": True,
                    "event_range": list(replay.get("event_range") or []),
                    "admission_failures": [],
                },
                aligned_plan_step_ids=tuple(item.aligned_plan_step_ids),
                progress_relation=item.progress_relation,
                status=ProvisionalStatus.TRIAL_READY,
                metadata={
                    "origin": "failure_extractor_f2",
                    "portable_contract_valid": True,
                    "canonical_label_source": compiled.atomic.metadata.get(
                        "canonical_label_source", ""
                    ),
                },
            ))

        if not validation.failure_experience_accepted:
            return records, None

        instances_by_id = {
            str(item.instance_id): item
            for item in getattr(self.requirement_expansion, "instances", ())
        }
        requirement_ids = tuple(
            instances_by_id
        )
        remaining = tuple(map(str, alignment.remaining_requirement_instance_ids))
        if not requirement_ids or not remaining:
            raise ValueError(
                "validated failure extraction requires non-empty requirement instances"
            )
        missing_instance_ids = set(remaining) - set(instances_by_id)
        if missing_instance_ids:
            raise ValueError(
                "validated failure extraction references unknown RequirementInstances: "
                + ", ".join(sorted(missing_instance_ids))
            )
        shape_id_by_instance = {
            instance_id: requirement_instance_shape_id(
                instance,
                self.requirement_expansion,
                self.task_contract,
            )
            for instance_id, instance in instances_by_id.items()
        }
        remaining_shape_ids = [
            shape_id_by_instance[instance_id]
            for instance_id in remaining
        ]
        divergence = dict(alignment.first_unrecovered_divergence)
        kind = str(divergence.get("kind") or "other")
        plan_steps = {
            str(item.step_id): item
            for item in getattr(self.cold_start_plan, "steps", ())
        }
        failed_step = plan_steps.get(str(divergence.get("step_id") or ""))
        failed_step_instance_ids = tuple(map(
            str,
            getattr(failed_step, "requirement_instance_ids", ()),
        ))
        unknown_failed_ids = set(failed_step_instance_ids) - set(instances_by_id)
        if unknown_failed_ids:
            raise ValueError(
                "failed cold-start step references unknown RequirementInstances: "
                + ", ".join(sorted(unknown_failed_ids))
            )
        failed_step_shape_ids = sorted(
            shape_id_by_instance[instance_id]
            for instance_id in failed_step_instance_ids
        )
        validated_prefix_shape_ids: list[str] = []
        for step_id in validation.proposal.validated_plan_prefix:
            step = plan_steps.get(str(step_id))
            if step is None:
                raise ValueError(
                    f"validated failure prefix references unknown plan step: {step_id}"
                )
            for instance_id in map(
                str, getattr(step, "requirement_instance_ids", ()),
            ):
                if instance_id not in shape_id_by_instance:
                    raise ValueError(
                        "validated failure prefix references unknown "
                        f"RequirementInstance: {instance_id}"
                    )
                validated_prefix_shape_ids.append(
                    shape_id_by_instance[instance_id]
                )
        divergence_payload = {
            "kind": kind,
            "remaining_requirement_shape_ids": remaining_shape_ids,
            "failed_step_requirement_shape_ids": failed_step_shape_ids,
            "repeat_index_shape": [
                int(instances_by_id[instance_id].repeat_index)
                for instance_id in remaining
                if instances_by_id[instance_id].repeat_block_id
            ],
        }
        cluster = task_cluster_signature(
            self.task_contract,
            self.harness_profile,
            requirement_expansion=self.requirement_expansion,
        )
        divergence_signature = content_hash(divergence_payload)
        experience_id = "failure_exp_" + content_hash({
            "cluster": cluster,
            "divergence": divergence_signature,
        })[:24]
        raw_codes = validation.proposal.negative_method_suffix.get(
            "avoid_pattern_codes", ()
        )
        codes = tuple(map(str, raw_codes)) if isinstance(
            raw_codes, (list, tuple)
        ) else ()
        experience = FailureExperience(
            experience_id=experience_id,
            cluster_signature=cluster,
            divergence_signature=divergence_signature,
            harness_profile=self.harness_profile,
            requirement_instance_ids=requirement_ids,
            validated_prefix_step_ids=tuple(
                validation.proposal.validated_plan_prefix
            ),
            first_unrecovered_divergence={
                "kind": kind,
                "failed_plan_step_template": (
                    "the next unresolved requirement template"
                ),
            },
            remaining_requirement_instance_ids=remaining,
            negative_suffix_summary=copy.deepcopy(
                validation.proposal.negative_method_suffix
            ),
            avoid_pattern_codes=codes or (kind,),
            provisional_atomic_refs=tuple(
                record.provisional_ref for record in records
            ),
            status=FailureExperienceStatus.OBSERVED,
            support_trace_ids=(str(self.trace.trace_id),),
            metadata={
                "origin": "failure_extractor_f2",
                "source_task_id": str(self.trace.task.task_id),
                "semantic_shape_version": 1,
                "requirement_shape_ids": [
                    shape_id_by_instance[instance_id]
                    for instance_id in requirement_ids
                ],
                "validated_prefix_shape_ids": validated_prefix_shape_ids,
                "remaining_requirement_shape_ids": remaining_shape_ids,
                "failed_step_requirement_shape_ids": failed_step_shape_ids,
            },
        )
        return records, experience


class FailureExtractionCoordinator:
    """Run F1/F2 in one session and stage knowledge only after both validate."""

    def __init__(
        self,
        alignment_validator: "FailurePlanAlignmentValidator" | None = None,
        asset_validator: "FailureAssetValidator" | None = None,
    ) -> None:
        self.alignment_validator = alignment_validator or FailurePlanAlignmentValidator()
        self.asset_validator = asset_validator or FailureAssetValidator()

    def prepare(
        self,
        *,
        eligibility: FailureExtractionEligibility,
        extractor: FailureExtractorSession,
        task_contract: Any,
        requirement_expansion: Any,
        cold_start_plan: Any,
        trace: Any,
        task_progress: Any,
        failures: Any,
        candidate_contracts: Any,
        source_replay: Callable[[FailureAtomicProposal], dict[str, Any] | bool],
        record_builder: FailureAssetRecordBuilder,
        portability_check: Callable[[FailureAtomicProposal], bool] | None = None,
    ) -> PreparedFailureExtraction:
        if not eligibility.passed:
            return PreparedFailureExtraction(
                None, None, None, [], None,
                {"code": "failure_extractor_not_eligible"},
            )
        try:
            proposed_alignment = extractor.align(
                task_contract=task_contract,
                requirement_expansion=requirement_expansion,
                cold_start_plan=cold_start_plan,
                trace_events=trace,
                task_progress=task_progress,
                failures=failures,
                candidate_contracts=candidate_contracts,
            )
        except PlannerProposalError as exc:
            return PreparedFailureExtraction(
                None, None, None, [], None,
                {"code": exc.code, "message": str(exc), "stage": "f1"},
            )
        alignment, f1_result = self.alignment_validator.validate(
            proposed_alignment,
            cold_start_plan=cold_start_plan,
            trace=trace,
        )
        if alignment is None or not f1_result.passed:
            return PreparedFailureExtraction(
                None, f1_result, None, [], None,
                {
                    "code": "failure_extractor_alignment_invalid",
                    "stage": "f1_validation",
                    "failure_codes": list(f1_result.failure_codes),
                },
            )
        try:
            proposal = extractor.extract(
                validated_alignment=alignment,
                authoritative_trace=trace,
                task_contract=task_contract,
            )
        except PlannerProposalError as exc:
            return PreparedFailureExtraction(
                alignment, f1_result, None, [], None,
                {"code": exc.code, "message": str(exc), "stage": "f2"},
            )
        f2_result = self.asset_validator.validate(
            proposal,
            alignment=alignment,
            trace=trace,
            source_replay=source_replay,
            portability_check=portability_check,
        )
        if not f2_result.result.passed:
            return PreparedFailureExtraction(
                alignment, f1_result, f2_result, [], None,
                {
                    "code": (
                        f2_result.result.failure_codes[0]
                        if f2_result.result.failure_codes
                        else "failure_extractor_atomic_invalid"
                    ),
                    "stage": "f2_validation",
                    "failure_codes": list(f2_result.result.failure_codes),
                },
            )
        try:
            provisional, experience = record_builder.build(alignment, f2_result)
        except ValueError as exc:
            # Record construction still validates model-authored F2 content
            # (portable labels, remaining suffixes, contract identity).  Such
            # rejection is not a database/program failure and must leave the
            # failed task Trace intact with zero failure-side writes.
            return PreparedFailureExtraction(
                alignment, f1_result, f2_result, [], None,
                {
                    "code": "failure_extractor_atomic_invalid",
                    "stage": "f2_record_validation",
                    "message": str(exc),
                },
            )
        return PreparedFailureExtraction(
            alignment, f1_result, f2_result,
            provisional, experience, {},
        )


class FailurePlanAlignmentValidator:
    def validate(
        self,
        alignment: FailurePlanAlignment,
        *,
        cold_start_plan: Any,
        trace: Any,
    ) -> tuple[FailurePlanAlignment | None, ValidationResult]:
        sequence = list(cold_start_plan.control_sequence)
        position = {step_id: index for index, step_id in enumerate(sequence)}
        plan_steps = {
            str(step.step_id): step
            for step in getattr(cold_start_plan, "steps", ())
        }
        action_count = len(trace.environment_actions)
        witnesses = _authoritative_witnesses(trace)
        by_step: dict[str, PlanStepAlignment] = {}
        valid_ranges = True
        chronological = True
        last_end = 0
        for item in alignment.step_alignments:
            if item.step_id not in position or item.step_id in by_step:
                continue
            has_range = item.event_start is not None or item.event_end is not None
            range_valid = (
                not has_range
                or item.event_start is not None
                and item.event_end is not None
                and 0 <= item.event_start < item.event_end <= action_count
            )
            if not range_valid:
                valid_ranges = False
                continue
            if item.event_start is not None:
                chronological &= item.event_start >= last_end
                last_end = int(item.event_end or last_end)
            if item.status in {"achieved", "diverged_then_recovered"}:
                if not item.effect_witness_refs or not set(item.effect_witness_refs).issubset(witnesses):
                    continue
            by_step[item.step_id] = item

        claimed_prefix = list(alignment.matched_prefix_step_ids)
        prefix_valid = (
            claimed_prefix == sequence[:len(claimed_prefix)]
            and all(
                step_id in by_step
                and by_step[step_id].status in {"achieved", "diverged_then_recovered"}
                for step_id in claimed_prefix
            )
        )
        divergence = dict(alignment.first_unrecovered_divergence)
        divergence_step = str(divergence.get("step_id", ""))
        divergence_event = divergence.get("event_index")
        prefix_end = max((
            by_step[step_id].event_end or 0 for step_id in claimed_prefix
        ), default=0)
        divergence_after_prefix = (
            (not divergence_step or position.get(divergence_step, len(sequence)) >= len(claimed_prefix))
            and (divergence_event is None or int(divergence_event) >= prefix_end)
        )

        remaining_step_ids = sequence[len(claimed_prefix):]
        plan_steps_complete = all(
            step_id in plan_steps for step_id in remaining_step_ids
        )
        authoritative_remaining = (
            [
                str(instance_id)
                for step_id in remaining_step_ids
                for instance_id in plan_steps[step_id].requirement_instance_ids
            ]
            if plan_steps_complete else []
        )
        remaining_valid = (
            plan_steps_complete
            and list(alignment.remaining_requirement_instance_ids)
            == authoritative_remaining
        )

        valid_spans: list[dict[str, Any]] = []
        for span in alignment.candidate_progress_spans:
            start, end = int(span["event_start"]), int(span["event_end"])
            if not (0 <= start < end <= action_count):
                continue
            actions = trace.environment_actions[start:end]
            if not any(
                action.accepted and int(action.new_revision) > int(action.revision)
                for action in actions
            ):
                continue
            refs = set(map(str, span.get("effect_witness_refs", ())))
            if not refs or not refs.issubset(witnesses):
                continue
            if str(span.get("step_id", "")) not in position:
                continue
            valid_spans.append(copy.deepcopy(span))

        rejected_span_count = (
            len(alignment.candidate_progress_spans) - len(valid_spans)
        )
        checks = {
            "step_ids_and_ranges_valid": valid_ranges and bool(by_step),
            "event_ranges_chronological": chronological,
            "matched_prefix_authoritative": prefix_valid,
            "first_divergence_after_prefix": divergence_after_prefix,
            "remaining_requirement_instances_authoritative": remaining_valid,
        }
        passed = all(checks.values())
        result = ValidationResult(
            level="failure_extractor_f1",
            passed=passed,
            checks=checks,
            failure_codes=[] if passed else ["failure_extractor_alignment_invalid"],
            messages=(
                [f"candidate_progress_span_rejected:{rejected_span_count}"]
                if rejected_span_count else []
            ),
        )
        if not passed:
            return None, result
        cleaned = FailurePlanAlignment(
            alignment_id=alignment.alignment_id,
            step_alignments=[by_step[step_id] for step_id in sequence if step_id in by_step],
            matched_prefix_step_ids=claimed_prefix,
            first_unrecovered_divergence=divergence,
            remaining_requirement_instance_ids=authoritative_remaining,
            candidate_progress_spans=valid_spans,
        )
        return cleaned, result


class FailureAssetValidator:
    def validate(
        self,
        proposal: FailureExtractionProposal,
        *,
        alignment: FailurePlanAlignment,
        trace: Any,
        source_replay: Callable[[FailureAtomicProposal], dict[str, Any] | bool],
        portability_check: Callable[[FailureAtomicProposal], bool] | None = None,
    ) -> FailureExtractionValidation:
        spans = {
            (
                str(value["step_id"]), int(value["event_start"]),
                int(value["event_end"]),
            )
            for value in alignment.candidate_progress_spans
        }
        accepted: list[FailureAtomicProposal] = []
        replay_results: dict[str, dict[str, Any]] = {}
        provisional_rejections: list[dict[str, Any]] = []
        seen_phases: set[str] = set()
        used_ranges: set[tuple[int, int]] = set()
        experience_checks = {
            "portable_failure_summary": (
                _portable(proposal.negative_method_suffix)
                and _portable(proposal.reusable_failure_summary)
                and bool(proposal.negative_method_suffix)
                and bool(proposal.reusable_failure_summary)
            ),
            "validated_prefix_unchanged": (
                proposal.validated_plan_prefix
                == alignment.matched_prefix_step_ids
            ),
        }
        failure_experience_accepted = all(experience_checks.values())
        for item in proposal.provisional_atomics:
            atomic = item.atomic_proposal
            phase_id = str(atomic.get("phase_id", ""))
            event_range = (
                int(atomic.get("event_start", -1)),
                int(atomic.get("event_end", -1)),
            )
            unique_atomic = (
                bool(phase_id)
                and phase_id not in seen_phases
                and event_range not in used_ranges
            )
            seen_phases.add(phase_id)
            used_ranges.add(event_range)
            span_key_matches = any(
                step_id in item.aligned_plan_step_ids
                and start == int(atomic.get("event_start", -1))
                and end == int(atomic.get("event_end", -1))
                for step_id, start, end in spans
            )
            valid = (
                unique_atomic
                and item.progress_relation != "no_progress"
                and bool(item.aligned_plan_step_ids)
                and span_key_matches
                and bool(atomic.get("effects"))
                and bool(atomic.get("input_roles"))
                and bool(atomic.get("output_roles"))
                and set(atomic.get("output_roles", {}).values()).issubset(
                    set(atomic.get("input_roles", {}).values())
                )
            )
            if not valid:
                provisional_rejections.append({
                    "phase_id": phase_id,
                    "code": "failure_extractor_atomic_invalid",
                })
                continue
            replay = source_replay(item)
            replay_payload = replay if isinstance(replay, dict) else {"passed": bool(replay)}
            replay_passed = replay_payload.get("passed") is True
            if not replay_passed:
                provisional_rejections.append({
                    "phase_id": phase_id,
                    "code": "provisional_source_replay_failed",
                })
                continue
            portable = True if portability_check is None else bool(portability_check(item))
            if not portable:
                provisional_rejections.append({
                    "phase_id": phase_id,
                    "code": "failure_experience_portability_rejected",
                })
                continue
            accepted.append(item)
            replay_results[phase_id] = replay_payload

        any_failure_asset_admitted = (
            failure_experience_accepted or bool(accepted)
        )
        checks = {
            **experience_checks,
            "provisional_candidates_filtered": True,
            "at_least_one_failure_asset_admitted": any_failure_asset_admitted,
            "no_composite_or_tool_output": True,
        }
        cleaned = FailureExtractionProposal(
            provisional_atomics=accepted,
            validated_plan_prefix=list(proposal.validated_plan_prefix),
            negative_method_suffix=copy.deepcopy(proposal.negative_method_suffix),
            reusable_failure_summary=copy.deepcopy(proposal.reusable_failure_summary),
        )
        return FailureExtractionValidation(
            proposal=cleaned,
            result=ValidationResult(
                level="failure_extractor_f2",
                passed=any_failure_asset_admitted,
                checks=checks,
                failure_codes=(
                    []
                    if any_failure_asset_admitted
                    else ["failure_extractor_atomic_invalid"]
                ),
            ),
            source_replays=replay_results,
            provisional_rejections=provisional_rejections,
            failure_experience_accepted=failure_experience_accepted,
        )


__all__ = [
    "FailureAssetRecordBuilder", "FailureAssetValidator",
    "FailureAtomicSourceReplay", "FailureExtractionCoordinator",
    "FailureExtractionEligibility", "FailureExtractionValidation",
    "FailurePlanAlignmentValidator", "PreparedFailureExtraction",
]
