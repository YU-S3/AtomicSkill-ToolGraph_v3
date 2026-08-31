"""Isolated, non-executable failure-side knowledge for the v3.1 method patch.

The success bank remains the only executable Skill/Tool registry.  This store
owns Provisional Atomic contracts, negative Failure Experiences, and their
append-only cold-start evidence.  It deliberately returns sanitized views to
Planner/Runtime and never materializes an Implementation, Tool, or Composite.
"""

from __future__ import annotations

import copy
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..core.refs import content_hash
from ..core.serialization import atomic_write_json, read_json, to_primitive
from .database import SCHEMA_VERSION, STATE_PATCH_LEVEL, StateDatabase


class ProvisionalStatus(str, Enum):
    DISCOVERED = "discovered"
    LOCAL_VALIDATED = "local_validated"
    TRIAL_READY = "trial_ready"
    TRIAL_SUPPORTED = "trial_supported"
    PROMOTED = "promoted"
    SUPPRESSED = "suppressed"


class FailureExperienceStatus(str, Enum):
    OBSERVED = "observed"
    CONFIRMED = "confirmed"
    RESOLVED = "resolved"
    ARCHIVED = "archived"


@dataclass(frozen=True)
class ProvisionalAtomicRecord:
    provisional_ref: str
    contract_signature: str
    canonical_intent: str
    atomic_contract: dict[str, Any]
    seeded_guideline: dict[str, Any]
    harness_profile: str

    source_trace_id: str
    source_task_id: str
    source_span: dict[str, Any]
    source_replay: dict[str, Any]
    aligned_plan_step_ids: tuple[str, ...]
    progress_relation: str

    status: ProvisionalStatus
    promoted_verified_refs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ProvisionalStatus(self.status))
        object.__setattr__(
            self,
            "aligned_plan_step_ids",
            tuple(map(str, self.aligned_plan_step_ids)),
        )
        object.__setattr__(
            self,
            "promoted_verified_refs",
            tuple(map(str, self.promoted_verified_refs)),
        )
        _require_nonempty(
            provisional_ref=self.provisional_ref,
            contract_signature=self.contract_signature,
            canonical_intent=self.canonical_intent,
            harness_profile=self.harness_profile,
            source_trace_id=self.source_trace_id,
            source_task_id=self.source_task_id,
        )
        expected_ref = provisional_ref_for(self.contract_signature)
        if self.provisional_ref != expected_ref:
            raise ValueError(
                "provisional ref must be derived from the canonical contract "
                f"signature: expected {expected_ref!r}"
            )
        if not re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", self.canonical_intent):
            raise ValueError("provisional canonical_intent must be portable lower_snake_case")
        if not isinstance(self.atomic_contract, dict) or not self.atomic_contract:
            raise ValueError("provisional Atomic requires a non-empty atomic_contract")
        effects = self.atomic_contract.get("effects")
        if not isinstance(effects, list) or not effects:
            raise ValueError("provisional Atomic contract requires at least one Effect")
        if not isinstance(self.seeded_guideline, dict):
            raise TypeError("seeded_guideline must be an object")
        if not isinstance(self.source_span, dict) or not isinstance(self.source_replay, dict):
            raise TypeError("source_span and source_replay must be objects")
        if not self.aligned_plan_step_ids:
            raise ValueError("provisional Atomic requires at least one aligned plan step")
        if self.progress_relation not in {
            "partial_target_effect",
            "consumed_prerequisite",
        }:
            raise ValueError(
                "persisted provisional Atomic progress_relation must describe real progress"
            )
        if _contains_executable_failure_asset(self.atomic_contract):
            raise ValueError(
                "provisional Atomic payload may not contain an Implementation, Tool, "
                "Composite, or executable source action script"
            )
        if _contains_executable_failure_asset(self.seeded_guideline):
            raise ValueError(
                "provisional seeded guideline may not contain a Tool body or source action script"
            )
        if (
            _contains_executable_failure_asset(self.source_span)
            or _contains_executable_failure_asset(self.source_replay)
        ):
            raise ValueError(
                "provisional source evidence may record hashes/witnesses, not an action script"
            )
        if self.status is ProvisionalStatus.PROMOTED and not self.promoted_verified_refs:
            raise ValueError("promoted provisional Atomic requires promoted_verified_refs")
        if (
            self.status is not ProvisionalStatus.PROMOTED
            and self.promoted_verified_refs
        ):
            raise ValueError("only a promoted provisional Atomic may name Verified refs")


@dataclass(frozen=True)
class FailureExperience:
    experience_id: str
    cluster_signature: str
    divergence_signature: str
    harness_profile: str

    requirement_instance_ids: tuple[str, ...]
    validated_prefix_step_ids: tuple[str, ...]
    first_unrecovered_divergence: dict[str, Any]
    remaining_requirement_instance_ids: tuple[str, ...]
    negative_suffix_summary: dict[str, Any]
    avoid_pattern_codes: tuple[str, ...]
    provisional_atomic_refs: tuple[str, ...]

    status: FailureExperienceStatus
    support_trace_ids: tuple[str, ...]
    resolved_by_trace_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", FailureExperienceStatus(self.status))
        for name in (
            "requirement_instance_ids",
            "validated_prefix_step_ids",
            "remaining_requirement_instance_ids",
            "avoid_pattern_codes",
            "provisional_atomic_refs",
            "support_trace_ids",
            "resolved_by_trace_ids",
        ):
            object.__setattr__(self, name, tuple(map(str, getattr(self, name))))
        _require_nonempty(
            experience_id=self.experience_id,
            cluster_signature=self.cluster_signature,
            divergence_signature=self.divergence_signature,
            harness_profile=self.harness_profile,
        )
        if not isinstance(self.first_unrecovered_divergence, dict):
            raise TypeError("first_unrecovered_divergence must be an object")
        if not isinstance(self.negative_suffix_summary, dict):
            raise TypeError("negative_suffix_summary must be an object")
        if not self.requirement_instance_ids:
            raise ValueError("Failure Experience requires RequirementInstance coverage")
        if not self.remaining_requirement_instance_ids:
            raise ValueError("Failure Experience requires a remaining requirement suffix")
        if not self.avoid_pattern_codes:
            raise ValueError("Failure Experience requires at least one avoid pattern code")
        if not self.support_trace_ids:
            raise ValueError("Failure Experience requires at least one support Trace")
        if _failure_summary_requires_sanitization(
            self.first_unrecovered_divergence
        ) or _failure_summary_requires_sanitization(self.negative_suffix_summary):
            raise ValueError(
                "Failure Experience contains concrete source terms or action material"
            )


@dataclass(frozen=True)
class ProvisionalAtomicCandidate:
    """Source-free C0/Runtime view of a Provisional Atomic."""

    provisional_ref: str
    contract_signature: str
    canonical_intent: str
    atomic_contract: dict[str, Any]
    seeded_guideline: dict[str, Any]
    harness_profile: str
    status: ProvisionalStatus
    independent_source_replay_support: int
    independent_local_trial_successes: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FailureExperienceView:
    """Explicitly non-executable, source-free negative-example view."""

    experience_id: str
    cluster_signature: str
    divergence_signature: str
    harness_profile: str
    requirement_instance_ids: tuple[str, ...]
    validated_prefix_step_ids: tuple[str, ...]
    first_unrecovered_divergence: dict[str, Any]
    remaining_requirement_instance_ids: tuple[str, ...]
    negative_suffix_summary: dict[str, Any]
    avoid_pattern_codes: tuple[str, ...]
    provisional_atomic_refs: tuple[str, ...]
    status: FailureExperienceStatus
    support_count: int
    resolved_count: int
    executable: bool = False
    warning: str = "FAILED HISTORICAL METHOD — NOT AN EXECUTABLE OR SUCCESSFUL PLAN"


_PROVISIONAL_TRANSITIONS: dict[ProvisionalStatus, frozenset[ProvisionalStatus]] = {
    ProvisionalStatus.DISCOVERED: frozenset({ProvisionalStatus.LOCAL_VALIDATED}),
    ProvisionalStatus.LOCAL_VALIDATED: frozenset({ProvisionalStatus.TRIAL_READY}),
    ProvisionalStatus.TRIAL_READY: frozenset({
        ProvisionalStatus.TRIAL_SUPPORTED,
        ProvisionalStatus.PROMOTED,
        ProvisionalStatus.SUPPRESSED,
    }),
    ProvisionalStatus.TRIAL_SUPPORTED: frozenset({
        ProvisionalStatus.PROMOTED,
        ProvisionalStatus.SUPPRESSED,
    }),
    ProvisionalStatus.PROMOTED: frozenset(),
    ProvisionalStatus.SUPPRESSED: frozenset(),
}

_EXPERIENCE_TRANSITIONS: dict[
    FailureExperienceStatus, frozenset[FailureExperienceStatus]
] = {
    FailureExperienceStatus.OBSERVED: frozenset({
        FailureExperienceStatus.CONFIRMED,
        FailureExperienceStatus.RESOLVED,
        FailureExperienceStatus.ARCHIVED,
    }),
    FailureExperienceStatus.CONFIRMED: frozenset({
        FailureExperienceStatus.RESOLVED,
        FailureExperienceStatus.ARCHIVED,
    }),
    FailureExperienceStatus.RESOLVED: frozenset({FailureExperienceStatus.ARCHIVED}),
    FailureExperienceStatus.ARCHIVED: frozenset(),
}

_FORBIDDEN_EXECUTABLE_KEYS = frozenset({
    "implementation",
    "implementation_ref",
    "implementation_refs",
    "tool",
    "tool_ref",
    "tool_refs",
    "tool_body",
    "composite",
    "composite_ref",
    "source_actions",
    "source_action_strings",
    "action_list",
    "replay_script",
    "primitive_steps",
    "steps",
    "action_sequence",
})

_SANITIZED_DROP_KEYS = frozenset({
    "concrete_bindings",
    "source_actions",
    "source_action_strings",
    "action_list",
    "raw_action",
    "admissible_commands",
    "observation",
    "game_file",
    "env_index",
    "source_task",
    "source_task_id",
    "source_trace",
    "source_trace_id",
    "source_span",
    "source_replay",
    "action",
    "tool_body",
    "replay_script",
})

_CONCRETE_INSTANCE = re.compile(
    r"(?i)\b[a-z][a-z0-9.-]*(?:(?:_|\s+)\d+)\b"
)
_UNATTRIBUTED_TASK = "__unattributed__"


def provisional_ref_for(contract_signature: str) -> str:
    signature = str(contract_signature).strip()
    if not signature or not re.fullmatch(r"[A-Za-z0-9._+-]+", signature):
        raise ValueError("canonical contract signature must be a stable identifier")
    return f"provisional://atomic_{signature}@1.0.0"


def _require_nonempty(**values: str) -> None:
    empty = [name for name, value in values.items() if not str(value).strip()]
    if empty:
        raise ValueError("non-empty fields required: " + ", ".join(sorted(empty)))


def _contains_executable_failure_asset(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() in _FORBIDDEN_EXECUTABLE_KEYS:
                return True
            if _contains_executable_failure_asset(item):
                return True
    elif isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_executable_failure_asset(item) for item in value)
    return False


def _replay_passed(value: Mapping[str, Any]) -> bool:
    return value.get("passed") is True or value.get("source_replay_passed") is True


def _sanitized(value: Any) -> Any:
    """Remove source material and redact common concrete instance tokens."""

    if isinstance(value, Mapping):
        return {
            str(key): _sanitized(item)
            for key, item in value.items()
            if str(key).casefold() not in _SANITIZED_DROP_KEYS
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_sanitized(item) for item in value]
    if isinstance(value, str):
        return _CONCRETE_INSTANCE.sub("<redacted_instance>", value)
    return copy.deepcopy(value)


def _failure_summary_requires_sanitization(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).casefold() in _SANITIZED_DROP_KEYS
            or _failure_summary_requires_sanitization(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_failure_summary_requires_sanitization(item) for item in value)
    return isinstance(value, str) and bool(_CONCRETE_INSTANCE.search(value))


def _json(value: Any) -> str:
    return json.dumps(
        to_primitive(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


class FailureKnowledgeStore:
    """Crash-safe failure-side payload store and lifecycle index."""

    def __init__(
        self,
        data_dir: str | Path,
        database: StateDatabase,
        *,
        experience_confirm_independent_tasks: int = 2,
    ) -> None:
        if (
            isinstance(experience_confirm_independent_tasks, bool)
            or int(experience_confirm_independent_tasks) < 2
        ):
            raise ValueError(
                "Failure Experience confirmation requires at least two independent tasks"
            )
        self.data_dir = Path(data_dir)
        self.root = self.data_dir / "failure_knowledge"
        self.provisional_root = self.root / "provisional"
        self.experience_root = self.root / "experiences"
        self.database = database
        self.experience_confirm_independent_tasks = int(
            experience_confirm_independent_tasks
        )
        self._failure_side_read_count = 0
        if not database.readonly:
            self.provisional_root.mkdir(parents=True, exist_ok=True)
            self.experience_root.mkdir(parents=True, exist_ok=True)

    @property
    def failure_side_read_count(self) -> int:
        return self._failure_side_read_count

    def reset_read_count(self) -> None:
        self._failure_side_read_count = 0

    # -- Provisional Atomic -------------------------------------------------

    def upsert_provisional(
        self,
        record: ProvisionalAtomicRecord,
    ) -> ProvisionalAtomicRecord:
        """Align an equivalent local Effect to one stable provisional ref."""

        record = self._copy_provisional(record)
        if record.status in {
            ProvisionalStatus.TRIAL_SUPPORTED,
            ProvisionalStatus.PROMOTED,
            ProvisionalStatus.SUPPRESSED,
        }:
            raise ValueError(
                "new provisional knowledge must enter no later than TRIAL_READY"
            )
        existing_row = self.database.execute(
            "SELECT * FROM provisional_artifacts WHERE provisional_ref=?",
            (record.provisional_ref,),
        ).fetchone()
        if existing_row is not None:
            existing = self._provisional_from_row(existing_row)
            self._require_equivalent_provisional(existing, record)
            self._record_provisional_source_support(record)
            return self._get_provisional(record.provisional_ref)

        payload = {
            "schema_version": SCHEMA_VERSION,
            "state_patch_level": STATE_PATCH_LEVEL,
            "kind": "provisional_atomic",
            "record": to_primitive(record),
        }
        digest = content_hash(payload)
        target = self.provisional_root / f"{digest}.json"
        existed = target.exists()
        if existed:
            if read_json(target) != payload:
                raise RuntimeError("failure knowledge content hash collision")
        else:
            atomic_write_json(target, payload)
        now = time.time()
        try:
            with self.database.transaction() as connection:
                connection.execute(
                    "INSERT INTO provisional_artifacts("
                    "provisional_ref,contract_signature,canonical_intent,status,"
                    "harness_profile,content_hash,file_path,source_trace_id,"
                    "source_task_id,promoted_refs_json,schema_version,created_at,updated_at"
                    ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        record.provisional_ref,
                        record.contract_signature,
                        record.canonical_intent,
                        record.status.value,
                        record.harness_profile,
                        digest,
                        str(target.resolve()),
                        record.source_trace_id,
                        record.source_task_id,
                        _json(record.promoted_verified_refs),
                        SCHEMA_VERSION,
                        now,
                        now,
                    ),
                )
                self._insert_evidence(
                    connection,
                    task_id=record.source_task_id,
                    trace_id=record.source_trace_id,
                    subject_ref=record.provisional_ref,
                    subject_kind="provisional_atomic",
                    event_type="discovered",
                    sequence_no=0,
                    metadata={
                        "source_span_hash": content_hash(record.source_span),
                        "aligned_plan_step_ids": list(record.aligned_plan_step_ids),
                        "progress_relation": record.progress_relation,
                    },
                )
                if record.status in {
                    ProvisionalStatus.LOCAL_VALIDATED,
                    ProvisionalStatus.TRIAL_READY,
                }:
                    self._insert_evidence(
                        connection,
                        task_id=record.source_task_id,
                        trace_id=record.source_trace_id,
                        subject_ref=record.provisional_ref,
                        subject_kind="provisional_atomic",
                        event_type="local_validated",
                        sequence_no=0,
                        metadata={},
                    )
                if record.status is ProvisionalStatus.TRIAL_READY:
                    if not _replay_passed(record.source_replay):
                        raise ValueError(
                            "TRIAL_READY provisional Atomic requires passed source replay"
                        )
                    self._insert_evidence(
                        connection,
                        task_id=record.source_task_id,
                        trace_id=record.source_trace_id,
                        subject_ref=record.provisional_ref,
                        subject_kind="provisional_atomic",
                        event_type="source_replay_passed",
                        sequence_no=0,
                        metadata={
                            "source_replay_hash": content_hash(record.source_replay),
                        },
                    )
        except Exception:
            if not existed:
                target.unlink(missing_ok=True)
            raise
        return self._get_provisional(record.provisional_ref)

    put_provisional = upsert_provisional

    def get_provisional(self, ref: str) -> ProvisionalAtomicRecord:
        self._failure_side_read_count += 1
        return self._get_provisional(ref)

    def _get_provisional(self, ref: str) -> ProvisionalAtomicRecord:
        row = self.database.execute(
            "SELECT * FROM provisional_artifacts WHERE provisional_ref=?",
            (str(ref),),
        ).fetchone()
        if row is None:
            raise KeyError(str(ref))
        return self._provisional_from_row(row)

    def list_provisionals(
        self,
        statuses: Iterable[ProvisionalStatus | str] | None = None,
        *,
        contract_signature: str = "",
        harness_profile: str = "",
    ) -> list[ProvisionalAtomicRecord]:
        self._failure_side_read_count += 1
        clauses: list[str] = []
        parameters: list[Any] = []
        if statuses is not None:
            allowed = sorted({ProvisionalStatus(item).value for item in statuses})
            if not allowed:
                return []
            clauses.append("status IN (" + ",".join("?" for _ in allowed) + ")")
            parameters.extend(allowed)
        if contract_signature:
            clauses.append("contract_signature=?")
            parameters.append(str(contract_signature))
        if harness_profile:
            clauses.append("harness_profile=?")
            parameters.append(str(harness_profile))
        query = "SELECT * FROM provisional_artifacts"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY provisional_ref"
        return [
            self._provisional_from_row(row)
            for row in self.database.rows(query, tuple(parameters))
        ]

    def provisional_candidate_view(
        self,
        record_or_ref: ProvisionalAtomicRecord | str,
    ) -> ProvisionalAtomicCandidate:
        record = (
            record_or_ref
            if isinstance(record_or_ref, ProvisionalAtomicRecord)
            else self.get_provisional(record_or_ref)
        )
        replay_support = self._independent_event_task_count(
            record.provisional_ref, "source_replay_passed"
        )
        local_successes = self._independent_event_task_count(
            record.provisional_ref, "trial_local_success"
        )
        return ProvisionalAtomicCandidate(
            record.provisional_ref,
            record.contract_signature,
            record.canonical_intent,
            copy.deepcopy(record.atomic_contract),
            copy.deepcopy(record.seeded_guideline),
            record.harness_profile,
            record.status,
            replay_support,
            local_successes,
            {
                "source_replay_support": replay_support,
                "local_trial_successes": local_successes,
            },
        )

    def update_provisional_status(
        self,
        ref: str,
        status: ProvisionalStatus | str,
        *,
        task_id: str = "",
        trace_id: str = "",
        promoted_verified_refs: Iterable[str] = (),
        event_type: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> ProvisionalAtomicRecord:
        target = ProvisionalStatus(status)
        current = self._get_provisional(ref)
        promoted = tuple(map(str, promoted_verified_refs))
        if target is ProvisionalStatus.PROMOTED and not promoted:
            promoted = current.promoted_verified_refs
        if target is ProvisionalStatus.PROMOTED and not promoted:
            raise ValueError("PROMOTED status requires Verified Atomic/Implementation/Tool refs")
        if target is not ProvisionalStatus.PROMOTED and promoted:
            raise ValueError("promoted refs are valid only for PROMOTED status")
        if target is current.status:
            if target is ProvisionalStatus.PROMOTED and (
                tuple(current.promoted_verified_refs) != promoted
            ):
                raise ValueError("promoted Verified refs are immutable")
            return current
        if target not in _PROVISIONAL_TRANSITIONS[current.status]:
            raise ValueError(
                f"invalid provisional status transition: {current.status.value} -> {target.value}"
            )
        if target is ProvisionalStatus.TRIAL_READY and not _replay_passed(
            current.source_replay
        ):
            raise ValueError("TRIAL_READY requires passed source replay")
        effective_task = str(task_id or current.source_task_id)
        effective_trace = str(trace_id or current.source_trace_id)
        default_event = {
            ProvisionalStatus.LOCAL_VALIDATED: "local_validated",
            ProvisionalStatus.TRIAL_READY: "source_replay_passed",
            ProvisionalStatus.TRIAL_SUPPORTED: "trial_supported",
            ProvisionalStatus.PROMOTED: "promoted",
            ProvisionalStatus.SUPPRESSED: "suppressed",
        }[target]
        with self.database.transaction() as connection:
            self._insert_evidence(
                connection,
                task_id=effective_task,
                trace_id=effective_trace,
                subject_ref=current.provisional_ref,
                subject_kind="provisional_atomic",
                event_type=event_type or default_event,
                sequence_no=0,
                metadata=dict(metadata or {}),
            )
            connection.execute(
                "UPDATE provisional_artifacts SET status=?,promoted_refs_json=?,updated_at=? "
                "WHERE provisional_ref=?",
                (
                    target.value,
                    _json(promoted),
                    time.time(),
                    current.provisional_ref,
                ),
            )
        return self._get_provisional(ref)

    def record_provisional_trial(
        self,
        ref: str,
        *,
        task_id: str,
        trace_id: str,
        started: bool,
        local_effect_passed: bool,
        strict_task_success: bool = False,
        infrastructure_failure: bool = False,
        provider_or_protocol_failure: bool = False,
        suppress_after: int = 3,
        metadata: Mapping[str, Any] | None = None,
    ) -> ProvisionalAtomicRecord:
        """Record one real Seeded trial and enforce support/suppression isolation."""

        if isinstance(suppress_after, bool) or int(suppress_after) < 1:
            raise ValueError("provisional suppression threshold must be positive")
        current = self._get_provisional(ref)
        if current.status not in {
            ProvisionalStatus.TRIAL_READY,
            ProvisionalStatus.TRIAL_SUPPORTED,
        }:
            raise ValueError("only trial-ready/supported provisional Atomic may be tried")
        neutral = (
            not started
            or infrastructure_failure
            or provider_or_protocol_failure
        )
        event_type = (
            "trial_neutral"
            if neutral
            else (
                "trial_local_success"
                if local_effect_passed
                else "trial_local_failure"
            )
        )
        target_status = current.status
        with self.database.transaction() as connection:
            self._insert_evidence(
                connection,
                task_id=task_id,
                trace_id=trace_id,
                subject_ref=ref,
                subject_kind="provisional_atomic",
                event_type=event_type,
                sequence_no=0,
                metadata={
                    **dict(metadata or {}),
                    "started": bool(started),
                    "strict_task_success": bool(strict_task_success),
                    "infrastructure_failure": bool(infrastructure_failure),
                    "provider_or_protocol_failure": bool(
                        provider_or_protocol_failure
                    ),
                },
            )
            if not neutral and local_effect_passed and not strict_task_success:
                target_status = ProvisionalStatus.TRIAL_SUPPORTED
            elif not neutral and not local_effect_passed:
                failures = self._consecutive_independent_trial_failures(
                    connection, ref
                )
                if failures >= int(suppress_after):
                    target_status = ProvisionalStatus.SUPPRESSED
            if target_status is not current.status:
                if target_status not in _PROVISIONAL_TRANSITIONS[current.status]:
                    raise ValueError(
                        f"invalid trial lifecycle transition: {current.status.value} "
                        f"-> {target_status.value}"
                    )
                connection.execute(
                    "UPDATE provisional_artifacts SET status=?,updated_at=? "
                    "WHERE provisional_ref=?",
                    (target_status.value, time.time(), ref),
                )
                self._insert_evidence(
                    connection,
                    task_id=task_id,
                    trace_id=trace_id,
                    subject_ref=ref,
                    subject_kind="provisional_atomic",
                    event_type=(
                        "trial_supported"
                        if target_status is ProvisionalStatus.TRIAL_SUPPORTED
                        else "suppressed"
                    ),
                    sequence_no=0,
                    metadata={},
                )
        return self._get_provisional(ref)

    def promote_provisional(
        self,
        ref: str,
        verified_refs: Iterable[str],
        *,
        task_id: str,
        trace_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> ProvisionalAtomicRecord:
        return self.update_provisional_status(
            ref,
            ProvisionalStatus.PROMOTED,
            task_id=task_id,
            trace_id=trace_id,
            promoted_verified_refs=tuple(verified_refs),
            event_type="promoted",
            metadata=metadata,
        )

    # -- Failure Experiences ----------------------------------------------

    def upsert_failure_experience(
        self,
        experience: FailureExperience,
    ) -> FailureExperience:
        experience = self._copy_experience(experience)
        if experience.status is not FailureExperienceStatus.OBSERVED:
            raise ValueError("new Failure Experience must enter as OBSERVED")
        existing_row = self.database.execute(
            "SELECT * FROM failure_experiences "
            "WHERE cluster_signature=? AND divergence_signature=?",
            (experience.cluster_signature, experience.divergence_signature),
        ).fetchone()
        task_ids = self._experience_support_task_ids(experience)
        if existing_row is not None:
            existing = self._experience_from_row(existing_row)
            if existing.harness_profile != experience.harness_profile:
                raise ValueError("equivalent Failure Experience changed harness profile")
            for index, trace_id in enumerate(experience.support_trace_ids):
                self.record_failure_experience_support(
                    existing.experience_id,
                    task_id=task_ids[index],
                    trace_id=trace_id,
                    metadata={},
                )
            return self._get_failure_experience(existing.experience_id)

        payload = {
            "schema_version": SCHEMA_VERSION,
            "state_patch_level": STATE_PATCH_LEVEL,
            "kind": "failure_experience",
            "record": to_primitive(experience),
        }
        digest = content_hash(payload)
        target = self.experience_root / f"{digest}.json"
        existed = target.exists()
        if existed:
            if read_json(target) != payload:
                raise RuntimeError("failure knowledge content hash collision")
        else:
            atomic_write_json(target, payload)
        now = time.time()
        try:
            with self.database.transaction() as connection:
                connection.execute(
                    "INSERT INTO failure_experiences("
                    "experience_id,cluster_signature,divergence_signature,status,"
                    "harness_profile,content_hash,file_path,support_count,resolved_count,"
                    "schema_version,created_at,updated_at"
                    ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        experience.experience_id,
                        experience.cluster_signature,
                        experience.divergence_signature,
                        FailureExperienceStatus.OBSERVED.value,
                        experience.harness_profile,
                        digest,
                        str(target.resolve()),
                        0,
                        0,
                        SCHEMA_VERSION,
                        now,
                        now,
                    ),
                )
                for index, trace_id in enumerate(experience.support_trace_ids):
                    self._insert_evidence(
                        connection,
                        task_id=task_ids[index],
                        trace_id=trace_id,
                        subject_ref=experience.experience_id,
                        subject_kind="failure_experience",
                        event_type="observed_support",
                        sequence_no=0,
                        metadata={},
                    )
                self._refresh_experience_counts(connection, experience.experience_id)
        except Exception:
            if not existed:
                target.unlink(missing_ok=True)
            raise
        return self._get_failure_experience(experience.experience_id)

    put_failure_experience = upsert_failure_experience

    def get_failure_experience(self, experience_id: str) -> FailureExperience:
        self._failure_side_read_count += 1
        return self._get_failure_experience(experience_id)

    def _get_failure_experience(self, experience_id: str) -> FailureExperience:
        row = self.database.execute(
            "SELECT * FROM failure_experiences WHERE experience_id=?",
            (str(experience_id),),
        ).fetchone()
        if row is None:
            raise KeyError(str(experience_id))
        return self._experience_from_row(row)

    def list_failure_experiences(
        self,
        statuses: Iterable[FailureExperienceStatus | str] | None = None,
        *,
        cluster_signature: str = "",
        harness_profile: str = "",
    ) -> list[FailureExperience]:
        self._failure_side_read_count += 1
        clauses: list[str] = []
        parameters: list[Any] = []
        if statuses is not None:
            allowed = sorted({FailureExperienceStatus(item).value for item in statuses})
            if not allowed:
                return []
            clauses.append("status IN (" + ",".join("?" for _ in allowed) + ")")
            parameters.extend(allowed)
        if cluster_signature:
            clauses.append("cluster_signature=?")
            parameters.append(str(cluster_signature))
        if harness_profile:
            clauses.append("harness_profile=?")
            parameters.append(str(harness_profile))
        query = "SELECT * FROM failure_experiences"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY experience_id"
        return [
            self._experience_from_row(row)
            for row in self.database.rows(query, tuple(parameters))
        ]

    def failure_experience_view(
        self,
        experience_or_id: FailureExperience | str,
    ) -> FailureExperienceView:
        experience = (
            experience_or_id
            if isinstance(experience_or_id, FailureExperience)
            else self.get_failure_experience(experience_or_id)
        )
        row = self.database.execute(
            "SELECT support_count,resolved_count FROM failure_experiences "
            "WHERE experience_id=?",
            (experience.experience_id,),
        ).fetchone()
        if row is None:
            raise KeyError(experience.experience_id)
        return FailureExperienceView(
            experience.experience_id,
            experience.cluster_signature,
            experience.divergence_signature,
            experience.harness_profile,
            tuple(experience.requirement_instance_ids),
            tuple(experience.validated_prefix_step_ids),
            _sanitized(experience.first_unrecovered_divergence),
            tuple(experience.remaining_requirement_instance_ids),
            _sanitized(experience.negative_suffix_summary),
            tuple(experience.avoid_pattern_codes),
            tuple(experience.provisional_atomic_refs),
            experience.status,
            int(row["support_count"]),
            int(row["resolved_count"]),
        )

    def record_failure_experience_support(
        self,
        experience_id: str,
        *,
        task_id: str,
        trace_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> FailureExperience:
        current = self._get_failure_experience(experience_id)
        if current.status not in {
            FailureExperienceStatus.OBSERVED,
            FailureExperienceStatus.CONFIRMED,
        }:
            return current
        with self.database.transaction() as connection:
            self._insert_evidence(
                connection,
                task_id=task_id or _UNATTRIBUTED_TASK,
                trace_id=trace_id,
                subject_ref=experience_id,
                subject_kind="failure_experience",
                event_type="observed_support",
                sequence_no=0,
                metadata=dict(metadata or {}),
            )
            self._refresh_experience_counts(connection, experience_id)
        return self._get_failure_experience(experience_id)

    def resolve_failure_experience(
        self,
        experience_id: str,
        *,
        task_id: str,
        trace_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> FailureExperience:
        current = self._get_failure_experience(experience_id)
        if current.status is FailureExperienceStatus.ARCHIVED:
            raise ValueError("archived Failure Experience cannot be resolved")
        with self.database.transaction() as connection:
            self._insert_evidence(
                connection,
                task_id=task_id,
                trace_id=trace_id,
                subject_ref=experience_id,
                subject_kind="failure_experience",
                event_type="resolved",
                sequence_no=0,
                metadata=dict(metadata or {}),
            )
            self._refresh_experience_counts(
                connection,
                experience_id,
                forced_status=FailureExperienceStatus.RESOLVED,
            )
        return self._get_failure_experience(experience_id)

    def update_failure_experience_status(
        self,
        experience_id: str,
        status: FailureExperienceStatus | str,
        *,
        task_id: str,
        trace_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> FailureExperience:
        target = FailureExperienceStatus(status)
        current = self._get_failure_experience(experience_id)
        if target is current.status:
            return current
        if target not in _EXPERIENCE_TRANSITIONS[current.status]:
            raise ValueError(
                f"invalid Failure Experience transition: {current.status.value} "
                f"-> {target.value}"
            )
        with self.database.transaction() as connection:
            self._insert_evidence(
                connection,
                task_id=task_id,
                trace_id=trace_id,
                subject_ref=experience_id,
                subject_kind="failure_experience",
                event_type=target.value,
                sequence_no=0,
                metadata=dict(metadata or {}),
            )
            connection.execute(
                "UPDATE failure_experiences SET status=?,updated_at=? "
                "WHERE experience_id=?",
                (target.value, time.time(), experience_id),
            )
        return self._get_failure_experience(experience_id)

    # -- Evidence and integrity -------------------------------------------

    def record_evidence(
        self,
        *,
        task_id: str,
        trace_id: str,
        subject_ref: str,
        subject_kind: str,
        event_type: str,
        sequence_no: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        with self.database.transaction() as connection:
            return self._insert_evidence(
                connection,
                task_id=task_id,
                trace_id=trace_id,
                subject_ref=subject_ref,
                subject_kind=subject_kind,
                event_type=event_type,
                sequence_no=sequence_no,
                metadata=dict(metadata or {}),
            )

    def verify_all(self) -> None:
        """Verify indexed payloads without counting a semantic failure-side read."""

        indexed: set[Path] = set()
        for table in ("provisional_artifacts", "failure_experiences"):
            for row in self.database.rows(
                f"SELECT content_hash,file_path,schema_version FROM {table}"
            ):
                if int(row["schema_version"]) != SCHEMA_VERSION:
                    raise RuntimeError("failure knowledge schema version mismatch")
                path = Path(str(row["file_path"])).resolve()
                try:
                    path.relative_to(self.root.resolve())
                except ValueError as exc:
                    raise RuntimeError(
                        "failure knowledge index points outside failure_knowledge"
                    ) from exc
                if not path.is_file():
                    raise RuntimeError(f"failure knowledge payload missing: {path}")
                payload = read_json(path)
                if content_hash(payload) != str(row["content_hash"]):
                    raise RuntimeError(f"failure knowledge payload hash mismatch: {path}")
                if (
                    int(payload.get("schema_version", 0)) != SCHEMA_VERSION
                    or str(payload.get("state_patch_level", ""))
                    != STATE_PATCH_LEVEL
                ):
                    raise RuntimeError("failure knowledge payload patch mismatch")
                indexed.add(path)
        if self.root.is_dir():
            disk = {
                path.resolve()
                for path in self.root.rglob("*.json")
                if path.is_file()
            }
            orphaned = sorted(str(path) for path in disk - indexed)
            if orphaned:
                raise RuntimeError(
                    "unindexed failure knowledge payloads: " + ", ".join(orphaned)
                )

    # -- Internal helpers --------------------------------------------------

    @staticmethod
    def _copy_provisional(record: ProvisionalAtomicRecord) -> ProvisionalAtomicRecord:
        return ProvisionalAtomicRecord(**copy.deepcopy(to_primitive(record)))

    @staticmethod
    def _copy_experience(experience: FailureExperience) -> FailureExperience:
        return FailureExperience(**copy.deepcopy(to_primitive(experience)))

    def _payload_for_row(self, row: Any, expected_kind: str) -> dict[str, Any]:
        path = Path(str(row["file_path"])).resolve()
        try:
            path.relative_to(self.root.resolve())
        except ValueError as exc:
            raise RuntimeError(
                "failure knowledge index points outside failure_knowledge"
            ) from exc
        if not path.is_file():
            raise RuntimeError(f"failure knowledge payload missing: {path}")
        payload = read_json(path)
        if content_hash(payload) != str(row["content_hash"]):
            raise RuntimeError(f"failure knowledge payload hash mismatch: {path}")
        if payload.get("kind") != expected_kind:
            raise RuntimeError("failure knowledge payload kind mismatch")
        if (
            int(payload.get("schema_version", 0)) != SCHEMA_VERSION
            or str(payload.get("state_patch_level", "")) != STATE_PATCH_LEVEL
        ):
            raise RuntimeError("failure knowledge payload patch mismatch")
        raw = payload.get("record")
        if not isinstance(raw, dict):
            raise RuntimeError("failure knowledge payload lacks a record")
        return copy.deepcopy(raw)

    def _provisional_from_row(self, row: Any) -> ProvisionalAtomicRecord:
        raw = self._payload_for_row(row, "provisional_atomic")
        raw["status"] = str(row["status"])
        raw["promoted_verified_refs"] = tuple(
            map(str, json.loads(str(row["promoted_refs_json"])))
        )
        raw["metadata"] = {
            **dict(raw.get("metadata") or {}),
            "independent_source_replay_support": self._independent_event_task_count(
                str(row["provisional_ref"]), "source_replay_passed"
            ),
            "independent_local_trial_success": self._independent_event_task_count(
                str(row["provisional_ref"]), "trial_local_success"
            ),
        }
        return ProvisionalAtomicRecord(**raw)

    def _experience_from_row(self, row: Any) -> FailureExperience:
        raw = self._payload_for_row(row, "failure_experience")
        raw["experience_id"] = str(row["experience_id"])
        raw["status"] = str(row["status"])
        support_rows = self.database.rows(
            "SELECT trace_id FROM cold_start_evidence "
            "WHERE subject_ref=? AND event_type='observed_support' "
            "ORDER BY rowid",
            (str(row["experience_id"]),),
        )
        resolved_rows = self.database.rows(
            "SELECT trace_id FROM cold_start_evidence "
            "WHERE subject_ref=? AND event_type='resolved' ORDER BY rowid",
            (str(row["experience_id"]),),
        )
        raw["support_trace_ids"] = tuple(dict.fromkeys(
            str(item["trace_id"]) for item in support_rows
        )) or tuple(map(str, raw.get("support_trace_ids") or ()))
        raw["resolved_by_trace_ids"] = tuple(dict.fromkeys(
            str(item["trace_id"]) for item in resolved_rows
        ))
        return FailureExperience(**raw)

    @staticmethod
    def _require_equivalent_provisional(
        existing: ProvisionalAtomicRecord,
        proposed: ProvisionalAtomicRecord,
    ) -> None:
        if (
            existing.contract_signature != proposed.contract_signature
            or existing.harness_profile != proposed.harness_profile
            or content_hash(existing.atomic_contract)
            != content_hash(proposed.atomic_contract)
        ):
            raise ValueError(
                "stable provisional ref conflicts with a non-equivalent Atomic contract"
            )

    def _record_provisional_source_support(
        self,
        record: ProvisionalAtomicRecord,
    ) -> None:
        with self.database.transaction() as connection:
            self._insert_evidence(
                connection,
                task_id=record.source_task_id,
                trace_id=record.source_trace_id,
                subject_ref=record.provisional_ref,
                subject_kind="provisional_atomic",
                event_type="discovered",
                sequence_no=0,
                metadata={
                    "source_span_hash": content_hash(record.source_span),
                    "aligned_plan_step_ids": list(record.aligned_plan_step_ids),
                    "progress_relation": record.progress_relation,
                },
            )
            if _replay_passed(record.source_replay):
                self._insert_evidence(
                    connection,
                    task_id=record.source_task_id,
                    trace_id=record.source_trace_id,
                    subject_ref=record.provisional_ref,
                    subject_kind="provisional_atomic",
                    event_type="source_replay_passed",
                    sequence_no=0,
                    metadata={
                        "source_replay_hash": content_hash(record.source_replay),
                    },
                )

    def _insert_evidence(
        self,
        connection: Any,
        *,
        task_id: str,
        trace_id: str,
        subject_ref: str,
        subject_kind: str,
        event_type: str,
        sequence_no: int,
        metadata: Mapping[str, Any],
    ) -> str:
        _require_nonempty(
            task_id=task_id,
            trace_id=trace_id,
            subject_ref=subject_ref,
            subject_kind=subject_kind,
            event_type=event_type,
        )
        if isinstance(sequence_no, bool) or int(sequence_no) < 0:
            raise ValueError("cold-start evidence sequence_no must be non-negative")
        encoded = _json(dict(metadata))
        identity = {
            "task_id": str(task_id),
            "trace_id": str(trace_id),
            "subject_ref": str(subject_ref),
            "subject_kind": str(subject_kind),
            "event_type": str(event_type),
            "sequence_no": int(sequence_no),
            "metadata_json": encoded,
        }
        event_id = "cold_evidence_" + content_hash(identity)[:24]
        existing = connection.execute(
            "SELECT * FROM cold_start_evidence WHERE trace_id=? AND subject_ref=? "
            "AND event_type=? AND sequence_no=?",
            (str(trace_id), str(subject_ref), str(event_type), int(sequence_no)),
        ).fetchone()
        if existing is not None:
            comparable = {
                key: existing[key]
                for key in identity
            }
            if comparable != identity:
                raise ValueError("cold-start evidence identity is immutable")
            return str(existing["event_id"])
        connection.execute(
            "INSERT INTO cold_start_evidence("
            "event_id,task_id,trace_id,subject_ref,subject_kind,event_type,"
            "sequence_no,metadata_json) VALUES(?,?,?,?,?,?,?,?)",
            (
                event_id,
                identity["task_id"],
                identity["trace_id"],
                identity["subject_ref"],
                identity["subject_kind"],
                identity["event_type"],
                identity["sequence_no"],
                identity["metadata_json"],
            ),
        )
        return event_id

    def _independent_event_task_count(self, subject_ref: str, event_type: str) -> int:
        row = self.database.execute(
            "SELECT COUNT(DISTINCT task_id) AS count FROM cold_start_evidence "
            "WHERE subject_ref=? AND event_type=? AND task_id<>?",
            (subject_ref, event_type, _UNATTRIBUTED_TASK),
        ).fetchone()
        return 0 if row is None else int(row["count"])

    @staticmethod
    def _consecutive_independent_trial_failures(
        connection: Any,
        subject_ref: str,
    ) -> int:
        rows = connection.execute(
            "SELECT task_id,event_type FROM cold_start_evidence "
            "WHERE subject_ref=? AND event_type IN "
            "('trial_local_success','trial_local_failure') ORDER BY rowid DESC",
            (subject_ref,),
        ).fetchall()
        tasks: set[str] = set()
        for row in rows:
            if str(row["event_type"]) == "trial_local_success":
                break
            task_id = str(row["task_id"])
            if task_id != _UNATTRIBUTED_TASK:
                tasks.add(task_id)
        return len(tasks)

    def _experience_support_task_ids(
        self,
        experience: FailureExperience,
    ) -> list[str]:
        metadata = dict(experience.metadata or {})
        raw = metadata.get("support_task_ids")
        if isinstance(raw, (list, tuple)) and len(raw) == len(
            experience.support_trace_ids
        ):
            return [str(item) or _UNATTRIBUTED_TASK for item in raw]
        source_task = str(
            metadata.get("source_task_id") or metadata.get("task_id") or ""
        )
        return [source_task or _UNATTRIBUTED_TASK] * len(
            experience.support_trace_ids
        )

    def _refresh_experience_counts(
        self,
        connection: Any,
        experience_id: str,
        *,
        forced_status: FailureExperienceStatus | None = None,
    ) -> None:
        support = connection.execute(
            "SELECT COUNT(DISTINCT task_id) AS count FROM cold_start_evidence "
            "WHERE subject_ref=? AND event_type='observed_support' AND task_id<>?",
            (experience_id, _UNATTRIBUTED_TASK),
        ).fetchone()
        resolved = connection.execute(
            "SELECT COUNT(DISTINCT task_id) AS count FROM cold_start_evidence "
            "WHERE subject_ref=? AND event_type='resolved' AND task_id<>?",
            (experience_id, _UNATTRIBUTED_TASK),
        ).fetchone()
        row = connection.execute(
            "SELECT status FROM failure_experiences WHERE experience_id=?",
            (experience_id,),
        ).fetchone()
        if row is None:
            raise KeyError(experience_id)
        current = FailureExperienceStatus(row["status"])
        support_count = 0 if support is None else int(support["count"])
        resolved_count = 0 if resolved is None else int(resolved["count"])
        target = forced_status or current
        if (
            forced_status is None
            and current is FailureExperienceStatus.OBSERVED
            and support_count >= self.experience_confirm_independent_tasks
        ):
            target = FailureExperienceStatus.CONFIRMED
        if target is not current and target not in _EXPERIENCE_TRANSITIONS[current]:
            raise ValueError(
                f"invalid Failure Experience transition: {current.value} -> {target.value}"
            )
        connection.execute(
            "UPDATE failure_experiences SET status=?,support_count=?,resolved_count=?,"
            "updated_at=? WHERE experience_id=?",
            (
                target.value,
                support_count,
                resolved_count,
                time.time(),
                experience_id,
            ),
        )


__all__ = [
    "FailureExperience",
    "FailureExperienceStatus",
    "FailureExperienceView",
    "FailureKnowledgeStore",
    "ProvisionalAtomicCandidate",
    "ProvisionalAtomicRecord",
    "ProvisionalStatus",
    "provisional_ref_for",
]
