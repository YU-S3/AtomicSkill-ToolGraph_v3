"""Bounded, code-authoritative views for failure-side extraction.

The failure extractor is a semantic consumer, not a Trace persistence or
transport consumer.  This module therefore projects only the public evidence
needed by F1/F2.  It deliberately never parses an observation, reconstructs a
benchmark workflow, or exposes provider/session/debug state.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from ..core.serialization import to_primitive


DEFAULT_PUBLIC_OBSERVATION_CHAR_LIMIT = 4096
_TRUNCATION_MARKER = "\n[truncated]"


@dataclass(frozen=True)
class FailureExecutionEvent:
    event_index: int
    revision_before: int
    revision_after: int
    occurrence_id: str
    step_id: str
    origin: str
    action_type: str
    arguments: dict[str, Any]
    accepted: bool
    done: bool
    won: bool
    bounded_public_observation: str
    validation_witness_refs: tuple[str, ...]
    task_progress_before_digest: str
    task_progress_after_digest: str


@dataclass(frozen=True)
class FailureTaskProgressDelta:
    revision: int
    source: str
    validator_revision: int
    progress_digest: str
    targets: tuple[dict[str, Any], ...]
    unsatisfied_identity_constraint_count: int


@dataclass(frozen=True)
class FailureAlignmentView:
    trace_id: str
    task_id: str
    task_contract: dict[str, Any]
    requirement_expansion: dict[str, Any]
    cold_start_plan: dict[str, Any]
    plan_steps: tuple[dict[str, Any], ...]
    execution_events: tuple[FailureExecutionEvent, ...]
    task_progress_deltas: tuple[FailureTaskProgressDelta, ...]
    failures: tuple[dict[str, Any], ...]
    candidate_contract_views: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class FailureCandidateProgressSpan:
    step_id: str
    event_start: int
    event_end: int
    accepted_events: tuple[FailureExecutionEvent, ...]
    authoritative_positive_effects: tuple[dict[str, Any], ...]
    witness_refs: tuple[str, ...]
    progress_before: dict[str, Any]
    progress_after: dict[str, Any]


@dataclass(frozen=True)
class FailureAssetExtractionView:
    trace_id: str
    task_id: str
    validated_alignment: dict[str, Any]
    task_contract: dict[str, Any]
    validated_prefix: tuple[str, ...]
    candidate_progress_spans: tuple[FailureCandidateProgressSpan, ...]
    first_unrecovered_divergence: dict[str, Any]
    remaining_requirement_instances: tuple[str, ...]


def _mapping(value: Any) -> dict[str, Any]:
    primitive = to_primitive(value)
    return copy.deepcopy(primitive) if isinstance(primitive, dict) else {}


def _sequence(value: Any) -> list[Any]:
    primitive = to_primitive(value)
    if isinstance(primitive, (list, tuple)):
        return list(primitive)
    return []


def _bounded_public_observation(value: Any, limit: int) -> str:
    """Normalize transport newlines and apply a deterministic prefix bound.

    No semantic rewriting is permitted here.  In particular, empty-container
    observations stay empty-container observations; the projection never turns
    them into conclusions about a target or recommendations for the next step.
    """

    observation = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    if len(observation) <= limit:
        return observation
    if limit <= len(_TRUNCATION_MARKER):
        return observation[:limit]
    return observation[: limit - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER


def _progress_delta(value: Any) -> FailureTaskProgressDelta:
    record = _mapping(value)
    snapshot = _mapping(record.get("snapshot"))
    targets = []
    for raw in _sequence(snapshot.get("targets")):
        target = _mapping(raw)
        # Concrete witnesses, shared values, and distinct entity identities are
        # intentionally excluded.  F1 receives only validator-backed progress
        # shape and counts.
        targets.append({
            "constraint_id": str(target.get("constraint_id", "")),
            "predicate": str(target.get("predicate", "")),
            "required_count": int(target.get("required_count", 0)),
            "satisfied_count": int(target.get("satisfied_count", 0)),
            "remaining_count": int(target.get("remaining_count", 0)),
            "distinct_by": str(target.get("distinct_by", "")),
        })
    return FailureTaskProgressDelta(
        revision=int(record.get("revision", 0)),
        source=str(record.get("source", "")),
        validator_revision=int(snapshot.get("revision", 0)),
        progress_digest=str(snapshot.get("progress_digest", "")),
        targets=tuple(targets),
        unsatisfied_identity_constraint_count=len(
            _sequence(snapshot.get("unsatisfied_identity_constraints"))
        ),
    )


def _progress_at_revision(
    progress: Iterable[FailureTaskProgressDelta], revision: int,
) -> FailureTaskProgressDelta | None:
    eligible = [item for item in progress if item.validator_revision <= revision]
    return max(eligible, key=lambda item: item.validator_revision, default=None)


def _progress_projection(
    value: FailureTaskProgressDelta | None,
) -> dict[str, Any]:
    if value is None:
        return {
            "progress_digest": "",
            "targets": [],
            "unsatisfied_identity_constraint_count": 0,
        }
    return {
        "progress_digest": value.progress_digest,
        "targets": copy.deepcopy(list(value.targets)),
        "unsatisfied_identity_constraint_count": (
            value.unsatisfied_identity_constraint_count
        ),
    }


def _progressed_contract_effects(
    before_targets: Mapping[str, dict[str, Any]],
    after_targets: Mapping[str, dict[str, Any]],
    contract_effects: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map validator progress to TaskContract effects without predicate fan-out.

    The default TaskProgress constraint id carries the exact target ordinal.
    An adapter-supplied constraint id does not, so it is safe to project an
    effect only when its predicate identifies exactly one contract target.
    """

    selected: list[dict[str, Any]] = []
    for constraint_id, after in after_targets.items():
        before = before_targets.get(constraint_id, {})
        if int(after.get("satisfied_count", 0)) <= int(
            before.get("satisfied_count", 0)
        ):
            continue
        predicate = str(after.get("predicate", ""))
        exact: dict[str, Any] | None = None
        parts = str(constraint_id).split("::", 2)
        claims_target_ordinal = len(parts) == 3 and parts[0] == "target"
        if claims_target_ordinal:
            try:
                ordinal = int(parts[1])
            except ValueError:
                ordinal = -1
            if 0 <= ordinal < len(contract_effects):
                candidate = contract_effects[ordinal]
                if str(candidate.get("predicate", "")).casefold() == predicate.casefold():
                    exact = candidate
        if exact is None and not claims_target_ordinal:
            matches = [
                effect for effect in contract_effects
                if str(effect.get("predicate", "")).casefold()
                == predicate.casefold()
            ]
            if len(matches) == 1:
                exact = matches[0]
        if exact is not None and exact not in selected:
            selected.append(_mapping(exact))
    return selected


def _plan_projection(cold_start_plan: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = _mapping(cold_start_plan)
    # Accept either a plan proposal or the Trace wrapper that contains one.
    if isinstance(raw.get("proposal"), dict):
        raw = _mapping(raw["proposal"])
    steps = [_mapping(item) for item in _sequence(raw.pop("steps", []))]
    return raw, steps


class FailureExtractionViewBuilder:
    """Project strict F1/F2 inputs without interpreting environment text."""

    def __init__(
        self,
        *,
        public_observation_char_limit: int = DEFAULT_PUBLIC_OBSERVATION_CHAR_LIMIT,
    ) -> None:
        if public_observation_char_limit < 0:
            raise ValueError("public observation character limit must be non-negative")
        self.public_observation_char_limit = int(public_observation_char_limit)

    def build_alignment(
        self,
        *,
        trace: Any,
        task_contract: Any,
        requirement_expansion: Any,
        cold_start_plan: Any,
        candidate_contract_views: Any,
    ) -> FailureAlignmentView:
        plan, plan_steps = _plan_projection(cold_start_plan)
        progress = tuple(
            _progress_delta(item)
            for item in list(getattr(trace, "task_progress_records", ()))
        )
        spans = {
            str(_mapping(item).get("span_id", "")): _mapping(item)
            for item in list(getattr(trace, "runtime_spans", ()))
        }
        node_steps = {
            str(_mapping(item).get("occurrence_id", "")): str(
                _mapping(item).get("step_id", "")
            )
            for item in list(getattr(trace, "node_records", ()))
        }
        cold_step_ranges = [
            _mapping(item) for item in list(getattr(trace, "cold_start_steps", ()))
        ]
        trace_cold_plan = _mapping(getattr(trace, "cold_start_plan", None))
        first_unresolved_step = str(
            trace_cold_plan.get("first_unresolved_step_id", "")
        )

        validations = [_mapping(item) for item in getattr(trace, "validations", ())]
        events: list[FailureExecutionEvent] = []
        for event_index, raw in enumerate(getattr(trace, "environment_actions", ())):
            action = _mapping(raw)
            span = spans.get(str(action.get("span_id", "")), {})
            occurrence_id = str(span.get("occurrence_id", ""))
            origin = str(span.get("kind", ""))
            step_id = node_steps.get(occurrence_id, "")
            if not step_id:
                step_id = next((
                    str(item.get("step_id", ""))
                    for item in cold_step_ranges
                    if int(item.get("action_start", -1)) <= event_index
                    < int(item.get("action_end", -1))
                ), "")
            if (
                not step_id
                and origin == "cold_start_dynamic_continuation"
            ):
                step_id = first_unresolved_step

            revision_before = int(action.get("revision", 0))
            revision_after = int(action.get("new_revision", revision_before))
            before = _progress_at_revision(progress, revision_before)
            after = _progress_at_revision(progress, revision_after)
            witness_refs = sorted({
                str(ref)
                for validation in validations
                if int(validation.get("revision", -1)) == revision_after
                and str(validation.get("occurrence_id", "")) == occurrence_id
                for ref in _sequence(_mapping(validation.get("result")).get(
                    "witness_refs", ()
                ))
                if str(ref)
            })
            events.append(FailureExecutionEvent(
                event_index=event_index,
                revision_before=revision_before,
                revision_after=revision_after,
                occurrence_id=occurrence_id,
                step_id=step_id,
                origin=origin,
                action_type=str(action.get("action_type", "")),
                arguments=_mapping(action.get("arguments")),
                accepted=action.get("accepted") is True,
                done=action.get("done") is True,
                won=action.get("won") is True,
                bounded_public_observation=_bounded_public_observation(
                    action.get("observation", ""),
                    self.public_observation_char_limit,
                ),
                validation_witness_refs=tuple(witness_refs),
                task_progress_before_digest=(
                    before.progress_digest if before is not None else ""
                ),
                task_progress_after_digest=(
                    after.progress_digest if after is not None else ""
                ),
            ))

        task = _mapping(getattr(trace, "task", {}))
        return FailureAlignmentView(
            trace_id=str(getattr(trace, "trace_id", "")),
            task_id=str(task.get("task_id", "")),
            task_contract=_mapping(task_contract),
            requirement_expansion=_mapping(requirement_expansion),
            cold_start_plan=plan,
            plan_steps=tuple(plan_steps),
            execution_events=tuple(events),
            task_progress_deltas=progress,
            failures=tuple(
                _mapping(item) for item in getattr(trace, "failures", ())
            ),
            candidate_contract_views=tuple(
                _mapping(item) for item in _sequence(candidate_contract_views)
            ),
        )

    def build_assets(
        self,
        *,
        trace: Any,
        alignment_view: FailureAlignmentView,
        validated_alignment: Any,
        task_contract: Any,
    ) -> FailureAssetExtractionView:
        alignment = _mapping(validated_alignment)
        events_by_index = {
            event.event_index: event for event in alignment_view.execution_events
        }
        progress_by_digest = {
            item.progress_digest: item
            for item in alignment_view.task_progress_deltas
            if item.progress_digest
        }
        plan_steps = {
            str(item.get("step_id", "")): item
            for item in alignment_view.plan_steps
        }
        validations = [_mapping(item) for item in getattr(trace, "validations", ())]
        contract = _mapping(task_contract)
        contract_effects = [
            _mapping(item) for item in _sequence(contract.get("target_effects"))
        ]

        spans: list[FailureCandidateProgressSpan] = []
        for raw_span in _sequence(alignment.get("candidate_progress_spans")):
            span = _mapping(raw_span)
            start = int(span.get("event_start", -1))
            end = int(span.get("event_end", -1))
            selected = [
                events_by_index[index]
                for index in range(start, end)
                if index in events_by_index
            ]
            accepted = tuple(event for event in selected if event.accepted)
            witness_refs = tuple(dict.fromkeys(
                str(ref) for ref in _sequence(span.get("effect_witness_refs"))
                if str(ref)
            ))

            before_delta = (
                progress_by_digest.get(selected[0].task_progress_before_digest)
                if selected else None
            )
            after_delta = (
                progress_by_digest.get(selected[-1].task_progress_after_digest)
                if selected else None
            )
            before_targets = {
                str(item.get("constraint_id", "")): item
                for item in (before_delta.targets if before_delta else ())
            }
            after_targets = {
                str(item.get("constraint_id", "")): item
                for item in (after_delta.targets if after_delta else ())
            }
            selected_occurrences = {
                event.occurrence_id for event in selected if event.occurrence_id
            }
            selected_revisions = {
                event.revision_after for event in selected
            }
            validated_effects: list[dict[str, Any]] = []
            matching_positive_validation = False
            wanted_witnesses = set(witness_refs)
            for validation in validations:
                result = _mapping(validation.get("result"))
                refs = {
                    str(ref) for ref in _sequence(result.get("witness_refs"))
                }
                structurally_matches = (
                    validation.get("revision") in selected_revisions
                    and (
                        not selected_occurrences
                        or str(validation.get("occurrence_id", ""))
                        in selected_occurrences
                    )
                )
                if (
                    result.get("passed") is True
                    and structurally_matches
                    and (not wanted_witnesses or bool(refs & wanted_witnesses))
                ):
                    matching_positive_validation = True
                    # Future validators may publish their exact validated
                    # effects.  Copy those fields if present; never derive
                    # effects by parsing an action or observation here.
                    for key in (
                        "authoritative_positive_effects", "validated_effects",
                    ):
                        validated_effects.extend(
                            _mapping(item) for item in _sequence(result.get(key))
                        )

            if matching_positive_validation:
                validated_effects.extend(
                    _mapping(item)
                    for item in _sequence(
                        plan_steps.get(str(span.get("step_id", "")), {}).get(
                            "expected_effects", ()
                        )
                    )
                )
            validated_effects.extend(_progressed_contract_effects(
                before_targets,
                after_targets,
                contract_effects,
            ))
            unique_effects: list[dict[str, Any]] = []
            for effect in validated_effects:
                if effect not in unique_effects:
                    unique_effects.append(effect)

            spans.append(FailureCandidateProgressSpan(
                step_id=str(span.get("step_id", "")),
                event_start=start,
                event_end=end,
                accepted_events=accepted,
                authoritative_positive_effects=tuple(unique_effects),
                witness_refs=witness_refs,
                progress_before=_progress_projection(before_delta),
                progress_after=_progress_projection(after_delta),
            ))

        task = _mapping(getattr(trace, "task", {}))
        return FailureAssetExtractionView(
            trace_id=str(getattr(trace, "trace_id", "")),
            task_id=str(task.get("task_id", "")),
            validated_alignment=alignment,
            task_contract=contract,
            validated_prefix=tuple(map(
                str, _sequence(alignment.get("matched_prefix_step_ids"))
            )),
            candidate_progress_spans=tuple(spans),
            first_unrecovered_divergence=_mapping(
                alignment.get("first_unrecovered_divergence")
            ),
            remaining_requirement_instances=tuple(map(
                str,
                _sequence(alignment.get("remaining_requirement_instance_ids")),
            )),
        )


__all__ = [
    "DEFAULT_PUBLIC_OBSERVATION_CHAR_LIMIT",
    "FailureAlignmentView",
    "FailureAssetExtractionView",
    "FailureCandidateProgressSpan",
    "FailureExecutionEvent",
    "FailureExtractionViewBuilder",
    "FailureTaskProgressDelta",
]
