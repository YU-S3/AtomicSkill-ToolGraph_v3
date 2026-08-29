"""Independent lifecycle policies for Atomic, Implementation, Tool, Composite."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

from ..core.status import (
    RuntimeMode,
    SkillStatus,
    ToolStatus,
    skill_status_usable,
    tool_status_usable,
)
from ..knowledge.database import StateDatabase
from .projections import ArtifactStats, LifecycleProjection


@dataclass(frozen=True)
class LifecycleThresholds:
    atomic_active_independent_support: int = 2
    implementation_active_direct_successes: int = 2
    tool_active_started_successes: int = 2
    tool_candidate_max_intrinsic_failures: int = 1
    tool_preferred_min_started: int = 5
    tool_preferred_reliability_lower_bound: float = 0.50
    tool_preferred_wilson_z: float = 1.96
    composite_active_self_sufficient_successes: int = 2

    atomic_suppress_consecutive_failures: int = 3
    implementation_suppress_consecutive_failures: int = 3
    tool_suppress_consecutive_failures: int = 3
    composite_suppress_consecutive_failures: int = 2

    def __post_init__(self) -> None:
        integer_names = (
            "atomic_active_independent_support",
            "implementation_active_direct_successes",
            "tool_active_started_successes",
            "tool_preferred_min_started",
            "composite_active_self_sufficient_successes",
            "atomic_suppress_consecutive_failures",
            "implementation_suppress_consecutive_failures",
            "tool_suppress_consecutive_failures",
            "composite_suppress_consecutive_failures",
        )
        for name in integer_names:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.tool_candidate_max_intrinsic_failures < 0:
            raise ValueError("tool_candidate_max_intrinsic_failures must be non-negative")
        if not 0.0 <= self.tool_preferred_reliability_lower_bound <= 1.0:
            raise ValueError("tool_preferred_reliability_lower_bound must be in [0, 1]")
        if self.tool_preferred_wilson_z <= 0:
            raise ValueError("tool_preferred_wilson_z must be positive")


@dataclass(frozen=True)
class LifecycleDecision:
    artifact_ref: str
    artifact_kind: str
    current_status: str
    next_status: str
    reason: str

    @property
    def changed(self) -> bool:
        return self.current_status != self.next_status


class LifecyclePolicy:
    """Pure lifecycle decisions; all thresholds are configurable experiment policy."""

    def __init__(self, thresholds: LifecycleThresholds | None = None) -> None:
        self.thresholds = thresholds or LifecycleThresholds()

    def review_atomic(
        self, ref: str, status: SkillStatus | str, stats: ArtifactStats
    ) -> LifecycleDecision:
        status = SkillStatus(status)
        if status is SkillStatus.RETIRED:
            return _keep(ref, "atomic", status, "retired_is_terminal")
        if status is SkillStatus.SUPPRESSED:
            if stats.stable_replacement:
                return _move(ref, "atomic", status, SkillStatus.RETIRED, "stable_replacement")
            return _keep(ref, "atomic", status, "awaiting_reliable_replacement")
        if status is SkillStatus.ACTIVE:
            if stats.stable_replacement:
                return _move(ref, "atomic", status, SkillStatus.SUPPRESSED, "superseded")
            if (
                stats.consecutive_intrinsic_failures
                >= self.thresholds.atomic_suppress_consecutive_failures
            ):
                return _move(
                    ref,
                    "atomic",
                    status,
                    SkillStatus.SUPPRESSED,
                    "repeated_semantic_failure_after_lower_layers_excluded",
                )
            return _keep(ref, "atomic", status, "active_evidence_stable")
        if status is SkillStatus.CANDIDATE:
            if (
                stats.independent_task_count
                >= self.thresholds.atomic_active_independent_support
            ):
                return _move(
                    ref, "atomic", status, SkillStatus.ACTIVE, "independent_canonical_support"
                )
            return _keep(ref, "atomic", status, "needs_independent_canonical_support")
        if status in {SkillStatus.DRAFT, SkillStatus.SHADOW} and stats.validated_count:
            return _move(
                ref, "atomic", status, SkillStatus.CANDIDATE, "deterministic_occurrence_validated"
            )
        return _keep(ref, "atomic", status, "not_yet_deterministically_validated")

    def review_implementation(
        self, ref: str, status: SkillStatus | str, stats: ArtifactStats
    ) -> LifecycleDecision:
        status = SkillStatus(status)
        if status is SkillStatus.RETIRED:
            return _keep(ref, "implementation", status, "retired_is_terminal")
        if status is SkillStatus.SUPPRESSED:
            if stats.stable_replacement:
                return _move(
                    ref, "implementation", status, SkillStatus.RETIRED, "stable_replacement"
                )
            return _keep(ref, "implementation", status, "awaiting_reliable_replacement")
        if status is SkillStatus.ACTIVE:
            if stats.stable_replacement:
                return _move(ref, "implementation", status, SkillStatus.SUPPRESSED, "superseded")
            if (
                stats.consecutive_intrinsic_failures
                >= self.thresholds.implementation_suppress_consecutive_failures
            ):
                return _move(
                    ref,
                    "implementation",
                    status,
                    SkillStatus.SUPPRESSED,
                    "repeated_intrinsic_mapping_constraint_or_policy_failure",
                )
            return _keep(ref, "implementation", status, "active_evidence_stable")
        if status is SkillStatus.CANDIDATE:
            if (
                stats.independent_direct_success_count
                >= self.thresholds.implementation_active_direct_successes
            ):
                return _move(
                    ref,
                    "implementation",
                    status,
                    SkillStatus.ACTIVE,
                    "independent_started_direct_successes",
                )
            return _keep(ref, "implementation", status, "needs_started_direct_successes")
        if status in {SkillStatus.DRAFT, SkillStatus.SHADOW} and stats.validated_count:
            return _move(
                ref,
                "implementation",
                status,
                SkillStatus.CANDIDATE,
                "static_mapping_constraint_compatibility_validated",
            )
        return _keep(ref, "implementation", status, "static_closure_not_validated")

    def review_tool(
        self, ref: str, status: ToolStatus | str, stats: ArtifactStats
    ) -> LifecycleDecision:
        status = ToolStatus(status)
        if status is ToolStatus.RETIRED:
            return _keep(ref, "tool", status, "retired_is_terminal")
        if status is ToolStatus.SUPPRESSED:
            if stats.stable_replacement:
                return _move(ref, "tool", status, ToolStatus.RETIRED, "stable_replacement")
            return _keep(ref, "tool", status, "awaiting_reliable_replacement")
        if status in {ToolStatus.ACTIVE, ToolStatus.PREFERRED}:
            if stats.stable_replacement:
                return _move(ref, "tool", status, ToolStatus.SUPPRESSED, "superseded")
            if (
                stats.consecutive_intrinsic_failures
                >= self.thresholds.tool_suppress_consecutive_failures
            ):
                return _move(
                    ref,
                    "tool",
                    status,
                    ToolStatus.SUPPRESSED,
                    "repeated_started_intrinsic_tool_failure",
                )
            if status is ToolStatus.ACTIVE and self._preferred_tool(stats):
                return _move(
                    ref,
                    "tool",
                    status,
                    ToolStatus.PREFERRED,
                    "reliable_and_cost_utility_preferred",
                )
            return _keep(ref, "tool", status, "active_evidence_stable")
        if status is ToolStatus.CANDIDATE:
            if (
                stats.validated_count > 0
                and stats.independent_direct_success_count
                >= self.thresholds.tool_active_started_successes
                and stats.intrinsic_failure_count
                <= self.thresholds.tool_candidate_max_intrinsic_failures
            ):
                return _move(
                    ref,
                    "tool",
                    status,
                    ToolStatus.ACTIVE,
                    "admitted_with_independent_started_successes",
                )
            return _keep(ref, "tool", status, "needs_admission_or_started_successes")
        if status in {ToolStatus.ADMISSION_PENDING, ToolStatus.SHADOW}:
            if stats.validated_count:
                return _move(ref, "tool", status, ToolStatus.CANDIDATE, "admission_passed")
            return _keep(ref, "tool", status, "admission_not_passed")
        if status is ToolStatus.DRAFT:
            if stats.validated_count:
                return _move(ref, "tool", status, ToolStatus.CANDIDATE, "admission_passed")
            if stats.proposed_count:
                return _move(
                    ref, "tool", status, ToolStatus.ADMISSION_PENDING, "admission_requested"
                )
        return _keep(ref, "tool", status, "not_submitted_for_admission")

    def _preferred_tool(self, stats: ArtifactStats) -> bool:
        return (
            stats.started_count >= self.thresholds.tool_preferred_min_started
            and stats.reliability_lower_bound(z=self.thresholds.tool_preferred_wilson_z)
            >= self.thresholds.tool_preferred_reliability_lower_bound
            and stats.preferred_utility_evidence_count > 0
        )

    def review_composite(
        self, ref: str, status: SkillStatus | str, stats: ArtifactStats
    ) -> LifecycleDecision:
        status = SkillStatus(status)
        if status is SkillStatus.RETIRED:
            return _keep(ref, "composite", status, "retired_is_terminal")
        if status is SkillStatus.SUPPRESSED:
            if stats.stable_replacement:
                return _move(ref, "composite", status, SkillStatus.RETIRED, "stable_replacement")
            return _keep(ref, "composite", status, "awaiting_reliable_replacement")
        if status is SkillStatus.ACTIVE:
            if stats.stable_replacement:
                return _move(ref, "composite", status, SkillStatus.SUPPRESSED, "superseded")
            if (
                stats.consecutive_intrinsic_failures
                >= self.thresholds.composite_suppress_consecutive_failures
            ):
                return _move(
                    ref,
                    "composite",
                    status,
                    SkillStatus.SUPPRESSED,
                    "repeated_task_rescue_or_structural_mismatch",
                )
            return _keep(ref, "composite", status, "active_evidence_stable")
        if status is SkillStatus.CANDIDATE:
            if (
                stats.independent_self_sufficient_success_count
                >= self.thresholds.composite_active_self_sufficient_successes
            ):
                return _move(
                    ref,
                    "composite",
                    status,
                    SkillStatus.ACTIVE,
                    "independent_graph_self_sufficient_successes",
                )
            return _keep(ref, "composite", status, "needs_self_sufficient_successes")
        if status in {SkillStatus.DRAFT, SkillStatus.SHADOW} and stats.validated_count:
            return _move(
                ref,
                "composite",
                status,
                SkillStatus.CANDIDATE,
                "canonical_workflow_contract_validated",
            )
        return _keep(ref, "composite", status, "canonical_workflow_not_validated")

    def review(
        self, ref: str, artifact_kind: str, status: str, stats: ArtifactStats
    ) -> LifecycleDecision:
        methods = {
            "atomic": self.review_atomic,
            "implementation": self.review_implementation,
            "tool": self.review_tool,
            "composite": self.review_composite,
        }
        try:
            return methods[artifact_kind](ref, status, stats)
        except KeyError as exc:
            raise ValueError(f"unsupported lifecycle artifact kind: {artifact_kind!r}") from exc


@dataclass(frozen=True)
class LifecycleReviewResult:
    reviewed_count: int
    decisions: tuple[LifecycleDecision, ...]

    @property
    def changed_count(self) -> int:
        return sum(decision.changed for decision in self.decisions)


class LifecycleController:
    """Apply pure policy decisions to registry status and recommended pointers."""

    def __init__(
        self,
        database: StateDatabase,
        projection: LifecycleProjection,
        policy: LifecyclePolicy | None = None,
    ) -> None:
        self.database = database
        self.projection = projection
        self.policy = policy or LifecyclePolicy()

    def review(self, artifact_refs: Iterable[str] | None = None) -> LifecycleReviewResult:
        if self.database.readonly:
            raise RuntimeError("frozen lifecycle registry is read-only")
        requested = None if artifact_refs is None else tuple(dict.fromkeys(artifact_refs))
        if requested is None:
            rows = self.database.rows(
                "SELECT artifact_ref,artifact_kind,logical_id,status FROM artifact_index "
                "ORDER BY artifact_ref"
            )
        else:
            rows = []
            for artifact_ref in requested:
                row = self.database.execute(
                    "SELECT artifact_ref,artifact_kind,logical_id,status FROM artifact_index "
                    "WHERE artifact_ref=?",
                    (artifact_ref,),
                ).fetchone()
                if row is None:
                    raise KeyError(artifact_ref)
                rows.append(row)

        decisions: list[LifecycleDecision] = []
        logical_ids: set[str] = set()
        for row in rows:
            stats = self.projection.stats(row["artifact_ref"], row["artifact_kind"])
            decision = self.policy.review(
                str(row["artifact_ref"]),
                str(row["artifact_kind"]),
                str(row["status"]),
                stats,
            )
            decisions.append(decision)
            if decision.changed:
                logical_ids.add(str(row["logical_id"]))

        if logical_ids:
            with self.database.transaction() as connection:
                for decision in decisions:
                    if not decision.changed:
                        continue
                    cursor = connection.execute(
                        "UPDATE artifact_index SET status=? WHERE artifact_ref=? AND status=?",
                        (decision.next_status, decision.artifact_ref, decision.current_status),
                    )
                    if cursor.rowcount != 1:
                        raise RuntimeError(
                            f"concurrent lifecycle status change for {decision.artifact_ref}"
                        )
                for logical_id in sorted(logical_ids):
                    self._refresh_recommended(connection, logical_id)

        return LifecycleReviewResult(len(rows), tuple(decisions))

    @staticmethod
    def _refresh_recommended(connection: object, logical_id: str) -> None:
        rows = list(
            connection.execute(
                "SELECT artifact_ref,artifact_kind,version,status FROM artifact_index "
                "WHERE logical_id=?",
                (logical_id,),
            )
        )
        eligible = [
            row
            for row in rows
            if (
                row["artifact_kind"] == "tool"
                and row["status"] in {ToolStatus.ACTIVE.value, ToolStatus.PREFERRED.value}
            )
            or (
                row["artifact_kind"] != "tool" and row["status"] == SkillStatus.ACTIVE.value
            )
        ]
        if not eligible:
            connection.execute(
                "DELETE FROM recommended_pointers WHERE logical_id=?", (logical_id,)
            )
            return
        eligible.sort(
            key=lambda row: (
                1 if row["status"] == ToolStatus.PREFERRED.value else 0,
                _version_key(str(row["version"])),
                str(row["artifact_ref"]),
            ),
            reverse=True,
        )
        connection.execute(
            "INSERT INTO recommended_pointers(logical_id,artifact_ref) VALUES(?,?) "
            "ON CONFLICT(logical_id) DO UPDATE SET artifact_ref=excluded.artifact_ref",
            (logical_id, str(eligible[0]["artifact_ref"])),
        )


@dataclass(frozen=True)
class CandidateUsePolicy:
    """Controlled online Candidate exploration with a replay-stable quota."""

    exploration_quota: float = 0.15
    seed: int | str = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.exploration_quota <= 1.0:
            raise ValueError("exploration_quota must be in [0, 1]")

    def allows(
        self,
        *,
        artifact_ref: str,
        artifact_kind: str,
        status: SkillStatus | ToolStatus | str,
        mode: RuntimeMode | str,
        task_id: str,
        reliable_active_available: bool = True,
        clearly_better_match: bool = False,
        explicit_exploration: bool = False,
    ) -> bool:
        mode = RuntimeMode(mode)
        if not status_usable(artifact_kind, status, mode):
            return False
        value = str(getattr(status, "value", status))
        if value != "candidate":
            return True
        if mode is RuntimeMode.FROZEN:
            return False
        if not reliable_active_available or clearly_better_match or explicit_exploration:
            return True
        digest = hashlib.sha256(
            f"{self.seed}\x1f{task_id}\x1f{artifact_ref}".encode("utf-8")
        ).digest()
        sample = int.from_bytes(digest[:8], "big") / float(1 << 64)
        return sample < self.exploration_quota


def status_usable(
    artifact_kind: str,
    status: SkillStatus | ToolStatus | str,
    mode: RuntimeMode | str,
) -> bool:
    if artifact_kind == "tool":
        return tool_status_usable(status, mode)
    if artifact_kind in {"atomic", "implementation", "composite"}:
        return skill_status_usable(status, mode)
    raise ValueError(f"unsupported artifact kind: {artifact_kind!r}")


def _keep(ref: str, kind: str, status: object, reason: str) -> LifecycleDecision:
    value = str(getattr(status, "value", status))
    return LifecycleDecision(ref, kind, value, value, reason)


def _move(ref: str, kind: str, current: object, target: object, reason: str) -> LifecycleDecision:
    return LifecycleDecision(
        ref,
        kind,
        str(getattr(current, "value", current)),
        str(getattr(target, "value", target)),
        reason,
    )


def _version_key(version: str) -> tuple[int, int, int, int, str]:
    try:
        major, minor, patch = (int(piece) for piece in version.split("."))
        return (1, major, minor, patch, version)
    except (TypeError, ValueError):
        return (0, 0, 0, 0, version)
