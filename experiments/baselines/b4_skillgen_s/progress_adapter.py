"""Train-only SkillGen subgoal-progress supervision authority.

Pinned SkillGen indexes label rows by the final ``family/trial`` pair in
``infos['extra.gamefile']`` and advances progress whenever an observation
matches a label-owned regular expression.  This module preserves exactly that
observable contract, but validates the whole Train manifest before a provider
call is allowed.  In particular, the shipped upstream ``all.jsonl`` covers
valid_unseen rather than the common Train manifest and is therefore not a
legal substitute for Train supervision.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from experiments.baselines.common.manifest import ManifestTask, TaskManifestSet


LABEL_SCHEMA = "skillgen-alfworld-subgoal-label-v1"
REQUIRED_SOURCE_SPLIT = "train"


class SkillGenLabelAuthorityError(ValueError):
    """The supplied supervision cannot legally cover the Train manifest."""

    failure_kind = "protocol_failure"

    def __init__(self, message: str, *, report: "LabelCoverageReport | None" = None) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class SkillGenLabel:
    task_key: str
    source_split: str
    goal: str
    subgoals: tuple[str, ...]
    difficulty: str
    source_line: int


@dataclass(frozen=True)
class LabelCoverageReport:
    schema_version: int
    label_schema: str
    label_path: str
    label_sha256: str
    manifest_id: str
    manifest_digest: str
    required_source_split: str
    manifest_task_count: int
    label_row_count: int
    covered_task_count: int
    missing_task_ids: tuple[str, ...]
    wrong_split_task_ids: tuple[str, ...]
    duplicate_task_keys: tuple[str, ...]
    invalid_rows: tuple[str, ...]
    unused_label_count: int
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def task_key_from_gamefile(value: str | Path) -> str:
    """Return the exact key used by pinned ``AlfWorld.reset``.

    The upstream implementation uses ``'/'.join(gamefile.split('/')[-3:-1])``.
    Normalizing separators first makes the same rule work for controller-side
    Windows paths without changing the actual key.
    """

    parts = PurePosixPath(str(value).replace("\\", "/")).parts
    if len(parts) < 3 or parts[-1] != "game.tw-pddl":
        raise ValueError(f"not an ALFWorld gamefile path: {value!r}")
    return "/".join(parts[-3:-1])


def _label_key(row: dict[str, Any]) -> str:
    additional = row.get("additional_info")
    if not isinstance(additional, dict):
        raise ValueError("additional_info must be a mapping")
    key = str(additional.get("description", "")).strip().replace("\\", "/")
    if len(PurePosixPath(key).parts) != 2:
        raise ValueError("additional_info.description must be '<family>/<trial>'")
    return key


def load_skillgen_labels(path: str | Path) -> tuple[dict[str, SkillGenLabel], dict[str, Any]]:
    """Load upstream-format labels without accepting ambiguous ownership."""

    label_path = Path(path).expanduser().resolve(strict=True)
    labels: dict[str, SkillGenLabel] = {}
    duplicate_keys: set[str] = set()
    invalid_rows: list[str] = []
    row_count = 0
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1,
    ):
        if not line.strip():
            continue
        row_count += 1
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("row root must be a mapping")
            if str(payload.get("task", "")).strip().casefold() != "alfworld":
                raise ValueError("task must be 'alfworld'")
            key = _label_key(payload)
            # Pinned SkillGen labels do not carry a split field.  Ownership is
            # established by exact coverage of the immutable Train manifest.
            # An optional explicit field is accepted only as an additional
            # guard and must never contradict that authority.
            source_split = str(payload.get("source_split", "")).strip()
            raw_subgoals = payload.get("subgoals")
            if not isinstance(raw_subgoals, list) or not raw_subgoals:
                raise ValueError("subgoals must be a non-empty list")
            subgoals = tuple(str(item) for item in raw_subgoals)
            if any(not item for item in subgoals):
                raise ValueError("subgoal regexes must be non-empty strings")
            for pattern in subgoals:
                re.compile(pattern)
            if key in labels:
                duplicate_keys.add(key)
                continue
            labels[key] = SkillGenLabel(
                task_key=key,
                source_split=source_split,
                goal=str(payload.get("goal", "")),
                subgoals=subgoals,
                difficulty=str(payload.get("difficulty", "")),
                source_line=line_number,
            )
        except (ValueError, TypeError, json.JSONDecodeError, re.error) as exc:
            invalid_rows.append(f"line {line_number}: {type(exc).__name__}: {exc}")
    return labels, {
        "path": str(label_path),
        "sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
        "row_count": row_count,
        "duplicate_keys": tuple(sorted(duplicate_keys)),
        "invalid_rows": tuple(invalid_rows),
    }


def analyze_label_coverage(
    manifest: TaskManifestSet,
    label_path: str | Path,
) -> tuple[LabelCoverageReport, dict[str, SkillGenLabel]]:
    """Compare one explicit label file with every task in one Train manifest."""

    labels, metadata = load_skillgen_labels(label_path)
    missing: list[str] = []
    wrong_split: list[str] = []
    used: set[str] = set()
    for task in manifest.tasks:
        key = task_key_from_gamefile(task.gamefile_rel)
        label = labels.get(key)
        if label is None:
            missing.append(task.task_id)
            continue
        used.add(key)
        if (
            task.source_split != REQUIRED_SOURCE_SPLIT
            or label.source_split not in {"", REQUIRED_SOURCE_SPLIT}
        ):
            wrong_split.append(task.task_id)
            continue
        labels[key] = replace(label, source_split=REQUIRED_SOURCE_SPLIT)
    covered = len(manifest.tasks) - len(missing) - len(wrong_split)
    duplicates = tuple(metadata["duplicate_keys"])
    invalid = tuple(metadata["invalid_rows"])
    passed = (
        manifest.source_split == REQUIRED_SOURCE_SPLIT
        and covered == len(manifest.tasks)
        and not duplicates
        and not invalid
    )
    report = LabelCoverageReport(
        schema_version=1,
        label_schema=LABEL_SCHEMA,
        label_path=str(metadata["path"]),
        label_sha256=str(metadata["sha256"]),
        manifest_id=manifest.manifest_id,
        manifest_digest=manifest.digest,
        required_source_split=REQUIRED_SOURCE_SPLIT,
        manifest_task_count=len(manifest.tasks),
        label_row_count=int(metadata["row_count"]),
        covered_task_count=covered,
        missing_task_ids=tuple(missing),
        wrong_split_task_ids=tuple(wrong_split),
        duplicate_task_keys=duplicates,
        invalid_rows=invalid,
        unused_label_count=len(set(labels) - used),
        passed=passed,
    )
    return report, labels


def require_complete_train_coverage(
    manifest: TaskManifestSet,
    label_path: str | Path,
) -> tuple[LabelCoverageReport, dict[str, SkillGenLabel]]:
    report, labels = analyze_label_coverage(manifest, label_path)
    if not report.passed:
        problems: list[str] = []
        if manifest.source_split != REQUIRED_SOURCE_SPLIT:
            problems.append(
                f"manifest source_split={manifest.source_split!r}, expected 'train'"
            )
        if report.missing_task_ids:
            problems.append(
                f"missing={len(report.missing_task_ids)}/{report.manifest_task_count}"
            )
        if report.wrong_split_task_ids:
            problems.append(f"wrong_split={len(report.wrong_split_task_ids)}")
        if report.duplicate_task_keys:
            problems.append(f"duplicates={len(report.duplicate_task_keys)}")
        if report.invalid_rows:
            problems.append(f"invalid_rows={len(report.invalid_rows)}")
        raise SkillGenLabelAuthorityError(
            "SkillGen-S Train label coverage failed: " + ", ".join(problems),
            report=report,
        )
    return report, labels


class SubgoalProgressTracker:
    """Pinned SkillGen progress semantics for one labeled Train task."""

    def __init__(self, label: SkillGenLabel) -> None:
        if label.source_split != REQUIRED_SOURCE_SPLIT:
            raise SkillGenLabelAuthorityError(
                f"label {label.task_key!r} belongs to {label.source_split!r}, not Train"
            )
        self.label = label
        # Pinned SkillGen reserves one final slot for terminal completion.
        self.finished_sub_goal = [0 for _ in range(len(label.subgoals) + 1)]
        self.is_done = False

    def observe(self, observation: str, *, done: bool) -> float:
        self.is_done = bool(done)
        for index, pattern in enumerate(self.label.subgoals):
            if re.search(pattern, str(observation)):
                self.finished_sub_goal[index] = 1
        if self.is_done:
            return 1.0
        return sum(self.finished_sub_goal) / len(self.finished_sub_goal)


def labels_for_manifest(
    manifest: TaskManifestSet,
    labels: dict[str, SkillGenLabel],
) -> dict[str, SkillGenLabel]:
    """Bind already-validated labels to manifest task ids."""

    result: dict[str, SkillGenLabel] = {}
    for task in manifest.tasks:
        key = task_key_from_gamefile(task.gamefile_rel)
        try:
            result[task.task_id] = labels[key]
        except KeyError as exc:
            raise SkillGenLabelAuthorityError(
                f"label coverage changed after preflight for task {task.task_id!r}"
            ) from exc
    return result
