"""Formal read-only ALFWorld frozen evaluation and source-train replay."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from atomic_skillgraph.knowledge.database import StateDatabase
from atomic_skillgraph.system import AtomicSkillGraphSystem, load_config

from .protocol import (
    ALFWORLD_FORMAL_TASK_TYPES,
    AttemptTraceLedger,
    ManifestStore,
    ProtocolError,
    RunManifest,
    RunState,
    TaskManifest,
    audit_failed_attempt,
    artifact_audit_snapshot,
    artifact_growth_audit,
    ensure_task_manifest,
    formal_reasoning_effort_audit,
    hash_code,
    hash_config,
    hash_knowledge,
    load_task_report_traces,
    require_active_composite_frozen_closure,
    require_r9_formal_freeze_audit,
    task_signature,
    validate_deepseek_formal_llm,
    validate_distinct_formal_tasks,
    write_run_observability,
)
from .report import (
    validate_frozen_v31_guards,
    validate_formal_usage,
    validate_usage_event_persistence,
    write_reports,
)
from .reference_manifest import (
    ReferenceManifest,
    load_formal_reference_manifest,
    select_reference_tasks,
    validate_reference_disjoint,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
_LEGACY_FROZEN_EVAL_RUN_NAMES = frozenset({
    "alfworld_frozen_eval_60",
    "alfworld_frozen_eval_60_r6",
    "alfworld_frozen_eval_60_b6a82ed",
})
_R7_FROZEN_EVAL_RUN_SEEDS = {
    "alfworld_frozen_eval_134_r7_seed42": 42,
    "alfworld_frozen_eval_134_r7_seed43": 43,
    "alfworld_frozen_eval_134_r7_seed44": 44,
}
_R8_FROZEN_EVAL_RUN_SEEDS = {
    "alfworld_frozen_eval_134_r8_seed42": 42,
    "alfworld_frozen_eval_134_r8_seed43": 43,
    "alfworld_frozen_eval_134_r8_seed44": 44,
}
_R9_FROZEN_EVAL_RUN_SEEDS = {
    "alfworld_frozen_eval_134_r9_seed42": 42,
    "alfworld_frozen_eval_134_r9_seed43": 43,
    "alfworld_frozen_eval_134_r9_seed44": 44,
}
_R7_TRAIN_REFERENCE_ID = "train_120"
_R7_TRAIN_REFERENCE_PATH = Path("data/baseline_manifests/train_120.json")
_R7_TEST_REFERENCE_ID = "test_ood_full_134"
_R7_TEST_REFERENCE_PATH = Path("data/baseline_manifests/test_ood_full_134.json")


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _is_source_train_replay(config: dict[str, Any]) -> bool:
    experiment = dict(config.get("experiment") or {})
    return experiment.get("phase") == "frozen_train_replay"


def _frozen_protocol(config: dict[str, Any]) -> tuple[str, int, int, int]:
    experiment = dict(config.get("experiment") or {})
    name = str(experiment.get("name", ""))
    if _is_source_train_replay(config):
        if name != "alfworld_frozen_train30_replay_b6a82ed":
            raise ProtocolError(
                "experiment.name does not identify the allowed source-train replay"
            )
        return "legacy_train30_replay", 42, 5, 30
    if name in _LEGACY_FROZEN_EVAL_RUN_NAMES:
        return "legacy_frozen60", 42, 10, 60
    if name in _R7_FROZEN_EVAL_RUN_SEEDS:
        return "r7_frozen134", _R7_FROZEN_EVAL_RUN_SEEDS[name], 0, 134
    if name in _R8_FROZEN_EVAL_RUN_SEEDS:
        return "r8_frozen134", _R8_FROZEN_EVAL_RUN_SEEDS[name], 0, 134
    if name in _R9_FROZEN_EVAL_RUN_SEEDS:
        return "r9_frozen134", _R9_FROZEN_EVAL_RUN_SEEDS[name], 0, 134
    raise ProtocolError(
        "experiment.name does not identify an allowed formal frozen protocol"
    )


def _r7_reference_manifests(
    config: dict[str, Any],
) -> tuple[ReferenceManifest, ReferenceManifest] | None:
    protocol, _, _, _ = _frozen_protocol(config)
    if protocol not in {"r7_frozen134", "r8_frozen134", "r9_frozen134"}:
        return None
    selection = dict((config.get("harness") or {}).get("task_selection") or {})
    expected_train_path = _path(_R7_TRAIN_REFERENCE_PATH)
    expected_test_path = _path(_R7_TEST_REFERENCE_PATH)
    train_path = _path(selection.get("train_reference_manifest_path", ""))
    test_path = _path(selection.get("reference_manifest_path", ""))
    if selection.get("train_reference_manifest_id") != _R7_TRAIN_REFERENCE_ID:
        raise ProtocolError(
            "fixed Frozen-134 train_reference_manifest_id must be 'train_120'"
        )
    if selection.get("reference_manifest_id") != _R7_TEST_REFERENCE_ID:
        raise ProtocolError(
            "fixed Frozen-134 reference_manifest_id must be 'test_ood_full_134'"
        )
    if train_path != expected_train_path:
        raise ProtocolError(
            "fixed Frozen-134 train_reference_manifest_path must identify the frozen train_120 manifest"
        )
    if test_path != expected_test_path:
        raise ProtocolError(
            "fixed Frozen-134 reference_manifest_path must identify the frozen test_ood_full_134 manifest"
        )
    train = load_formal_reference_manifest(
        train_path, manifest_id=_R7_TRAIN_REFERENCE_ID,
    )
    test = load_formal_reference_manifest(
        test_path, manifest_id=_R7_TEST_REFERENCE_ID,
    )
    validate_reference_disjoint(train, test)
    return train, test


def _selection(config: dict[str, Any]) -> tuple[list[str], int, int]:
    selection = dict((config.get("harness") or {}).get("task_selection") or {})
    labels = [str(item) for item in selection.get("task_types", [])]
    per_type = int(selection.get("tasks_per_type", 0))
    total = int(selection.get("total_tasks", 0))
    if tuple(labels) != ALFWORLD_FORMAL_TASK_TYPES:
        raise ProtocolError(
            "formal frozen eval requires the six ALFWorld task types in frozen order"
        )
    protocol, _, expected_per_type, expected_total = _frozen_protocol(config)
    if per_type != expected_per_type or total != expected_total:
        label = {
            "legacy_train30_replay": "source-train replay",
            "legacy_frozen60": "held-out eval",
            "r7_frozen134": "R7 fixed-manifest held-out eval",
            "r8_frozen134": "R8 fixed-manifest held-out eval",
            "r9_frozen134": "R9 fixed-manifest held-out eval",
        }[protocol]
        raise ProtocolError(
            f"formal frozen {label} has invalid task count: expected "
            f"tasks_per_type={expected_per_type}, total_tasks={expected_total}"
        )
    if selection.get("require_exact_count") is not True:
        raise ProtocolError("formal frozen selection must require the exact count")
    if protocol not in {"r7_frozen134", "r8_frozen134", "r9_frozen134"} and total != len(labels) * per_type:
        raise ProtocolError("legacy formal frozen selection must use the exact balanced count")
    require_disjoint = selection.get("require_disjoint_from_train_manifest")
    if protocol == "legacy_train30_replay":
        if require_disjoint is not False:
            raise ProtocolError("frozen source-train replay must use the train manifest")
    elif require_disjoint is not True:
        raise ProtocolError("frozen held-out manifest must be disjoint from train")
    return labels, per_type, total


def _validate_formal_config(config: dict[str, Any], output_dir: Path) -> None:
    validate_deepseek_formal_llm(config)
    experiment = dict(config.get("experiment") or {})
    harness = dict(config.get("harness") or {})
    selection = dict(harness.get("task_selection") or {})
    lifecycle = dict(config.get("lifecycle") or {})
    planner = dict(config.get("planner") or {})
    cold_start = dict(config.get("cold_start") or {})
    source_train_replay = _is_source_train_replay(config)
    protocol, expected_seed, _, _ = _frozen_protocol(config)
    allowed_run_names = (
        {"alfworld_frozen_train30_replay_b6a82ed"}
        if source_train_replay
        else (
            _LEGACY_FROZEN_EVAL_RUN_NAMES
            | frozenset(_R7_FROZEN_EVAL_RUN_SEEDS)
            | frozenset(_R8_FROZEN_EVAL_RUN_SEEDS)
            | frozenset(_R9_FROZEN_EVAL_RUN_SEEDS)
        )
    )
    expected_split = "train" if source_train_replay else "eval_out_of_distribution"
    expected = {
        "method_patch": (config.get("method_patch"), "3.2"),
        "planner.max_repeat_count": (planner.get("max_repeat_count"), 4),
        "planner.max_runtime_occurrences": (
            planner.get("max_runtime_occurrences"), 16
        ),
        "planner.cold_start_c1_repair_limit": (
            planner.get("cold_start_c1_repair_limit"), 1
        ),
        "cold_start": (cold_start, {"enabled": False}),
        "experiment.condition": (experiment.get("condition"), "full"),
        "experiment.freeze_skills": (experiment.get("freeze_skills"), True),
        "experiment.seed": (experiment.get("seed"), expected_seed),
        "experiment.require_knowledge_digest_unchanged": (
            experiment.get("require_knowledge_digest_unchanged"), True
        ),
        "experiment.allow_eval_traces_and_metrics": (
            experiment.get("allow_eval_traces_and_metrics"), True
        ),
        "experiment.allow_long_term_knowledge_writes": (
            experiment.get("allow_long_term_knowledge_writes"), False
        ),
        "experiment.resume_completed_task_boundary_only": (
            experiment.get("resume_completed_task_boundary_only"), True
        ),
        "harness.adapter": (harness.get("adapter"), "alfworld_v3"),
        "harness.alfworld_data_env": (harness.get("alfworld_data_env"), "ALFWORLD_DATA"),
        "harness.split": (harness.get("split"), expected_split),
        "harness.max_steps": (harness.get("max_steps"), 100),
        "harness.task_selection.policy": (
            selection.get("policy"),
            "fixed_manifest"
            if protocol in {"r7_frozen134", "r8_frozen134", "r9_frozen134"}
            else "balanced_fixed_manifest",
        ),
    }
    mismatches = [
        f"{name}: expected {wanted!r}, got {actual!r}"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if experiment.get("name") not in allowed_run_names:
        mismatches.append(
            "experiment.name must identify the selected formal frozen protocol"
        )
    frozen_dir = _path(experiment.get("source_frozen_snapshot_dir", ""))
    if _path(config.get("data_dir", "")) != frozen_dir:
        mismatches.append("data_dir must equal source_frozen_snapshot_dir")
    if _path(config.get("trace_data_dir", output_dir)) != output_dir:
        mismatches.append("trace_data_dir must equal experiment.output_dir")
    if output_dir.name != str(experiment.get("name", "")):
        mismatches.append("output_dir basename must equal experiment.name")
    if _path(experiment.get("task_manifest_path", "")) != output_dir / "task_manifest.json":
        mismatches.append("task_manifest_path must be <output_dir>/task_manifest.json")
    train_dir = _path(experiment.get("source_train_run_dir", ""))
    if frozen_dir != train_dir / "frozen" / "data_v3":
        mismatches.append("source_frozen_snapshot_dir must be <source_train_run_dir>/frozen/data_v3")
    if output_dir == frozen_dir or frozen_dir in output_dir.parents:
        mismatches.append("eval output_dir must be outside the read-only frozen snapshot")
    require_source_code_match = experiment.get("require_source_code_match", True)
    if not isinstance(require_source_code_match, bool):
        mismatches.append("experiment.require_source_code_match must be boolean")
    run_name = str(experiment.get("name", ""))
    source_revision = experiment.get("source_git_revision")
    if protocol in {"r7_frozen134", "r8_frozen134", "r9_frozen134"}:
        revision = (
            "r9" if protocol == "r9_frozen134"
            else "r8" if protocol == "r8_frozen134"
            else "r7"
        )
        expected_train_name = (
            f"alfworld_train_full_120_{revision}_seed{expected_seed}"
        )
        if train_dir.name != expected_train_name:
            mismatches.append(
                f"{revision.upper()} Frozen-134 seed {expected_seed} must use source {expected_train_name}"
            )
        if require_source_code_match is not True:
            mismatches.append(
                f"{revision.upper()} Frozen-134 must require source code match"
            )
        if source_revision not in (None, ""):
            mismatches.append(
                f"{revision.upper()} Frozen-134 must not override source revision"
            )
        if lifecycle.get("candidate_exploration_seed") != expected_seed:
            mismatches.append(
                "lifecycle.candidate_exploration_seed must equal experiment.seed"
            )
        reference_paths = {
            "reference_manifest_path": _R7_TEST_REFERENCE_PATH,
            "train_reference_manifest_path": _R7_TRAIN_REFERENCE_PATH,
        }
        for field, relative in reference_paths.items():
            if _path(selection.get(field, "")) != _path(relative):
                mismatches.append(
                    f"{revision.upper()} frozen eval {field} differs from frozen protocol"
                )
        if selection.get("reference_manifest_id") != _R7_TEST_REFERENCE_ID:
            mismatches.append(
                f"{revision.upper()} frozen eval reference_manifest_id must be 'test_ood_full_134'"
            )
        if selection.get("train_reference_manifest_id") != _R7_TRAIN_REFERENCE_ID:
            mismatches.append(
                f"{revision.upper()} frozen eval train_reference_manifest_id must be 'train_120'"
            )
        if protocol == "r8_frozen134":
            zero_success_limit = lifecycle.get(
                "composite_candidate_zero_success_trial_limit"
            )
            if (
                isinstance(zero_success_limit, bool)
                or not isinstance(zero_success_limit, int)
                or zero_success_limit != 3
            ):
                mismatches.append(
                    "R8 lifecycle.composite_candidate_zero_success_trial_limit "
                    "must be integer 3"
                )
        if protocol == "r9_frozen134":
            required_lifecycle = {
                "composite_active_deployment_successes": 2,
                "composite_candidate_zero_success_trial_limit": 3,
                "composite_candidate_activation_trial_limit": 5,
                "composite_active_consecutive_deployment_unsuccessful_limit": 3,
            }
            for field, wanted in required_lifecycle.items():
                value = lifecycle.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value != wanted:
                    mismatches.append(
                        f"R9 lifecycle.{field} must be integer {wanted}"
                    )
            if planner.get("literal_authorities") != {}:
                mismatches.append(
                    "R9 formal ALFWorld planner.literal_authorities must be empty"
                )
    elif source_train_replay:
        expected_revision = experiment.get("source_git_revision")
        if expected_revision != "b6a82ed47a2685e69a1fa052f70cd269f63e63c0":
            mismatches.append(
                "frozen source-train replay must pin source_git_revision to b6a82ed"
            )
    elif run_name == "alfworld_frozen_eval_60_r6":
        if train_dir.name != "alfworld_train_full_30_r6":
            mismatches.append(
                "R6 Frozen-60 must use the alfworld_train_full_30_r6 source"
            )
        if require_source_code_match is not True:
            mismatches.append("R6 Frozen-60 must require source code match")
        if source_revision not in (None, ""):
            mismatches.append("R6 Frozen-60 must not override source revision")
    elif run_name == "alfworld_frozen_eval_60_b6a82ed":
        if train_dir.name != "alfworld_train_full_30_v32":
            mismatches.append(
                "b6a82ed Frozen-60 must use the preserved Full-30 source"
            )
        if source_revision != "b6a82ed47a2685e69a1fa052f70cd269f63e63c0":
            mismatches.append("b6a82ed Frozen-60 must pin source_git_revision")
        if require_source_code_match is not False:
            mismatches.append(
                "b6a82ed Frozen-60 must record cross-revision evaluation"
            )
    elif not source_train_replay:
        if train_dir.name != "alfworld_train_full_30_v32":
            mismatches.append(
                "standard Frozen-60 must use the standard Full-30 source"
            )
        if require_source_code_match is not True:
            mismatches.append("standard Frozen-60 must require source code match")
        if source_revision not in (None, ""):
            mismatches.append("standard Frozen-60 must not override source revision")
    max_task_attempts = experiment.get("max_task_attempts")
    if (
        isinstance(max_task_attempts, bool)
        or not isinstance(max_task_attempts, int)
        or max_task_attempts <= 0
    ):
        mismatches.append("experiment.max_task_attempts must be a positive integer")
    if mismatches:
        raise ProtocolError("formal frozen config mismatch: " + "; ".join(mismatches))
    if protocol in {"r7_frozen134", "r8_frozen134", "r9_frozen134"}:
        _r7_reference_manifests(config)


def _verify_source_train(
    *,
    train_run_dir: Path,
    train_manifest: RunManifest,
    freeze_manifest: dict[str, Any],
    frozen_digest: str,
    current_code_digest: str,
    current_llm_hash: str,
    require_source_code_match: bool = True,
    expected_source_git_revision: str = "",
    reference_train_manifest: ReferenceManifest | None = None,
    expected_experiment_seed: int = 42,
    require_frozen_closure: bool = False,
    require_r9_freeze_audit: bool = False,
) -> None:
    """Bind one frozen bank to its completed immutable formal train source."""
    if train_manifest.phase != "train":
        raise ProtocolError("source manifest is not a train run")
    if train_run_dir.name != train_manifest.run_id:
        raise ProtocolError("source_train_run_dir basename differs from source run_id")
    metadata = train_manifest.metadata
    expected_per_type = 20 if reference_train_manifest is not None else 5
    expected_total = 120 if reference_train_manifest is not None else 30
    if (
        metadata.get("condition") != "full"
        or int(metadata.get("tasks_per_type", 0)) != expected_per_type
        or int(metadata.get("total_tasks", 0)) != expected_total
        or len(train_manifest.tasks) != expected_total
    ):
        raise ProtocolError(
            f"source manifest is not the formal full 6×{expected_per_type} train run"
        )
    if len({item.task_signature for item in train_manifest.tasks}) != expected_total:
        raise ProtocolError("source train manifest contains duplicate task signatures")
    if reference_train_manifest is not None:
        expected_reference_metadata = {
            "reference_manifest_id": reference_train_manifest.manifest_id,
            "reference_manifest_digest": reference_train_manifest.digest,
            "reference_manifest_seed": reference_train_manifest.seed,
            "reference_manifest_task_count": len(reference_train_manifest.tasks),
            "reference_manifest_family_counts": {
                label: sum(
                    task.task_type == label
                    for task in reference_train_manifest.tasks
                )
                for label in ALFWORLD_FORMAL_TASK_TYPES
            },
            "seed": expected_experiment_seed,
            "final_batch_maintenance_milestone": "formal_full_120_final_batch",
        }
        for field, expected in expected_reference_metadata.items():
            if metadata.get(field) != expected:
                raise ProtocolError(
                    f"source fixed Full-120 train metadata {field} differs from frozen protocol"
                )
        observed_identity: list[tuple[Any, ...]] = []
        for item in train_manifest.tasks:
            try:
                item_metadata = json.loads(item.metadata_json)
            except json.JSONDecodeError as exc:
                raise ProtocolError(
                    f"source train task metadata is invalid JSON: {item.task_id}"
                ) from exc
            game_file = str(item_metadata.get("game_file", "")).replace("\\", "/")
            observed_identity.append((
                item.task_id,
                item.task_signature,
                str(item_metadata.get("task_type", "")),
                item_metadata.get("env_index"),
                game_file,
                item.split,
            ))
        expected_identity = [
            (
                task.task_id,
                task.task_signature,
                task.task_type,
                task.env_index,
                task.gamefile_rel,
                "train",
            )
            for task in reference_train_manifest.tasks
        ]
        for index, (observed, expected) in enumerate(
            zip(observed_identity, expected_identity)
        ):
            if observed[:4] != expected[:4] or observed[5] != expected[5]:
                raise ProtocolError(
                    f"source fixed Full-120 task identity differs from reference at index {index}"
                )
            if not observed[4].endswith("/" + expected[4]) and observed[4] != expected[4]:
                raise ProtocolError(
                    f"source fixed Full-120 game file differs from reference at index {index}"
                )
    expected_provenance = {
        "source_run_id": train_manifest.run_id,
        "source_run_manifest_hash": train_manifest.manifest_hash,
        "source_config_hash": train_manifest.config_hash,
        "source_code_commit": train_manifest.code_commit,
        "source_task_manifest_hash": train_manifest.task_manifest_hash,
        "source_initial_knowledge_digest": train_manifest.knowledge_digest,
        "source_final_knowledge_digest": frozen_digest,
        "source_llm_config_hash": str(metadata.get("llm_config_hash", "")),
    }
    if require_frozen_closure:
        freeze_provenance = freeze_manifest.get("provenance")
        if not isinstance(freeze_provenance, dict):
            raise ProtocolError("R8 frozen snapshot provenance must be an object")
        closure_audit = freeze_provenance.get(
            "active_composite_frozen_closure_audit"
        )
        if not isinstance(closure_audit, dict):
            raise ProtocolError(
                "R8 frozen snapshot lacks its Active Composite closure audit"
            )
        if closure_audit.get(
            "active_composite_frozen_closure_passed"
        ) is not True:
            raise ProtocolError(
                "R8 frozen snapshot declares failed Active Composite closure"
            )
        expected_provenance.update({
            "active_composite_frozen_closure_passed": True,
            "active_composite_frozen_closure_audit": closure_audit,
        })
    if require_r9_freeze_audit:
        freeze_provenance = freeze_manifest.get("provenance")
        if not isinstance(freeze_provenance, dict):
            raise ProtocolError("R9 frozen snapshot provenance must be an object")
        formal_audit = freeze_provenance.get("r9_formal_freeze_audit")
        if not isinstance(formal_audit, dict):
            raise ProtocolError("R9 frozen snapshot lacks its formal freeze audit")
        if formal_audit.get("r9_formal_freeze_audit_passed") is not True:
            raise ProtocolError("R9 frozen snapshot declares failed formal freeze audit")
        expected_provenance.update({
            "r9_formal_freeze_audit_passed": True,
            "r9_formal_freeze_audit": formal_audit,
        })
    if freeze_manifest.get("provenance") != expected_provenance:
        raise ProtocolError("frozen snapshot provenance does not match source train manifest")
    if expected_source_git_revision and str(
        metadata.get("git_revision", "")
    ) != expected_source_git_revision:
        raise ProtocolError("source train git revision differs from the configured revision")
    if require_source_code_match and train_manifest.code_commit != current_code_digest:
        raise ProtocolError("frozen evaluation code differs from the source train code")
    if str(metadata.get("llm_config_hash", "")) != current_llm_hash:
        raise ProtocolError("frozen evaluation LLM configuration differs from source train")

    source_data_dir = train_run_dir / "data_v3"
    source_database_path = source_data_dir / "state.sqlite3"
    if not source_database_path.is_file():
        raise ProtocolError(f"source train state database missing: {source_database_path}")
    with StateDatabase(source_database_path, readonly=True) as source_database:
        run_row = source_database.execute(
            "SELECT * FROM run_manifests WHERE run_id=?", (train_manifest.run_id,)
        ).fetchone()
        if run_row is None or str(run_row["state"]) != RunState.COMPLETED.value:
            raise ProtocolError("source train run is not durably completed")
        run_identity = {
            "phase": train_manifest.phase,
            "config_hash": train_manifest.config_hash,
            "task_manifest_hash": train_manifest.task_manifest_hash,
            "code_commit": train_manifest.code_commit,
        }
        if any(str(run_row[key]) != value for key, value in run_identity.items()):
            raise ProtocolError("source train run ledger identity differs from immutable manifest")
        rows = source_database.rows(
            "SELECT * FROM run_tasks "
            "WHERE run_id=? ORDER BY rowid",
            (train_manifest.run_id,),
        )
        if len(rows) != len(train_manifest.tasks) or any(
            str(row["state"]) != "completed" or not str(row["trace_id"])
            for row in rows
        ):
            raise ProtocolError("source train task ledger is incomplete")
        expected_before = train_manifest.knowledge_digest
        for task, row in zip(train_manifest.tasks, rows):
            identity = {
                "task_id": task.task_id,
                "task_signature": task.task_signature,
                "config_hash": train_manifest.config_hash,
                "code_commit": train_manifest.code_commit,
                "knowledge_milestone": task.knowledge_milestone,
            }
            if any(str(row[key]) != value for key, value in identity.items()):
                raise ProtocolError(
                    f"source train task ledger identity mismatch at {task.task_id}"
                )
            result = json.loads(str(row["result_json"]))
            if str(result.get("knowledge_digest_before", "")) != expected_before:
                raise ProtocolError(
                    f"source train digest chain is broken before {task.task_id}"
                )
            expected_before = str(result.get("knowledge_digest_after", ""))
            if not expected_before:
                raise ProtocolError(f"source train task {task.task_id} lacks final digest")
            trace_path = train_run_dir / "traces" / f"{row['trace_id']}.json"
            if not trace_path.is_file():
                raise ProtocolError(f"source train trace missing: {trace_path}")
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            trace_task = dict(trace.get("task") or {})
            if (
                str(trace.get("trace_id", "")) != str(row["trace_id"])
                or str(trace_task.get("task_id", "")) != task.task_id
                or str(trace_task.get("task_signature", "")) != task.task_signature
            ):
                raise ProtocolError(f"source train trace identity mismatch: {trace_path}")
        if expected_before != frozen_digest:
            raise ProtocolError("source train final task milestone differs from frozen bank")
        if hash_knowledge(source_data_dir, database=source_database) != frozen_digest:
            raise ProtocolError("source train live bank differs from frozen bank")


def run(config_path: str | Path, *, resume: bool = False) -> int:
    invocation_started_monotonic = time.monotonic()
    invocation_started_at = datetime.now(timezone.utc)
    config_path = _path(config_path)
    config = load_config(config_path)
    experiment = dict(config.get("experiment") or {})
    phase = str(experiment.get("phase", ""))
    if phase not in {"frozen_eval", "frozen_train_replay"} or experiment.get(
        "runtime_mode"
    ) != "frozen":
        raise ProtocolError(
            "frozen runner requires phase=frozen_eval|frozen_train_replay "
            "and runtime_mode=frozen"
        )
    source_train_replay = phase == "frozen_train_replay"
    labels, per_type, expected_total = _selection(config)
    protocol, experiment_seed, _, _ = _frozen_protocol(config)
    output_dir = _path(experiment.get("output_dir", "runs/alfworld_frozen_eval_60"))
    _validate_formal_config(config, output_dir)
    max_task_attempts = int(experiment["max_task_attempts"])
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(experiment.get("name", "alfworld_frozen_eval_60"))
    attempt_ledger = AttemptTraceLedger(
        output_dir / "attempt_history", output_dir / "traces",
    )
    if not resume and attempt_ledger.root.exists():
        raise FileExistsError(attempt_ledger.root)
    recovered_attempts = attempt_ledger.recover_pending(run_id=run_id) if resume else []
    if recovered_attempts:
        print(json.dumps({
            "recovered_attempt_trace_captures": recovered_attempts,
        }, ensure_ascii=False), flush=True)
    unresolved_attempts = attempt_ledger.unresolved(run_id=run_id)
    if unresolved_attempts:
        raise ProtocolError(
            "resume found an attempt with no durable Trace; provider usage is unproven. "
            "Archive this eval run and start fresh without --resume: "
            + ", ".join(str(item["attempt_id"]) for item in unresolved_attempts)
        )

    with AtomicSkillGraphSystem(config) as system:
        if not system.readonly or system.database.readonly is not True:
            raise ProtocolError("frozen system did not open its knowledge database read-only")
        preflight = system.preflight(require_api_key=True, initialize_harness=True)
        if not preflight.get("passed"):
            raise ProtocolError(
                "formal frozen preflight failed: "
                + json.dumps(preflight, ensure_ascii=False, sort_keys=True)
            )
        digest_before = system.knowledge_digest()
        freeze_manifest_path = system.data_dir / "freeze_manifest.json"
        if not freeze_manifest_path.is_file():
            raise ProtocolError(f"frozen snapshot manifest missing: {freeze_manifest_path}")
        freeze_manifest = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
        if freeze_manifest.get("knowledge_digest") != digest_before:
            raise ProtocolError("frozen snapshot digest does not match its manifest")
        if protocol == "r8_frozen134":
            closure_audit = require_active_composite_frozen_closure(
                system.database, system.skills,
            )
            provenance = freeze_manifest.get("provenance")
            if not isinstance(provenance, dict):
                raise ProtocolError("R8 frozen snapshot provenance must be an object")
            if provenance.get(
                "active_composite_frozen_closure_passed"
            ) is not True or provenance.get(
                "active_composite_frozen_closure_audit"
            ) != closure_audit:
                raise ProtocolError(
                    "R8 frozen snapshot Active Composite closure authority mismatch"
                )
        elif protocol == "r9_frozen134":
            formal_audit = require_r9_formal_freeze_audit(
                system.database, system.skills,
            )
            provenance = freeze_manifest.get("provenance")
            if not isinstance(provenance, dict):
                raise ProtocolError("R9 frozen snapshot provenance must be an object")
            if provenance.get(
                "r9_formal_freeze_audit_passed"
            ) is not True or provenance.get(
                "r9_formal_freeze_audit"
            ) != formal_audit:
                raise ProtocolError(
                    "R9 frozen snapshot formal freeze authority mismatch"
                )

        reference_manifests = _r7_reference_manifests(config)
        if reference_manifests is None:
            reference_train_manifest = None
            reference_test_manifest = None
            reference_disjoint_audit = None
            tasks = system.harness.load_balanced_tasks(labels, per_type)
        else:
            reference_train_manifest, reference_test_manifest = reference_manifests
            reference_disjoint_audit = validate_reference_disjoint(
                reference_train_manifest, reference_test_manifest,
            )
            tasks = select_reference_tasks(system.harness, reference_test_manifest)
        if len(tasks) != expected_total:
            raise ProtocolError(
                f"frozen loader returned {len(tasks)} tasks, expected {expected_total}"
            )
        counts = {label: sum(task.task_type == label for task in tasks) for label in labels}
        if reference_test_manifest is None:
            if any(value != per_type for value in counts.values()):
                raise ProtocolError(f"balanced held-out task counts changed: {counts}")
        else:
            expected_counts = {
                label: sum(task.task_type == label for task in reference_test_manifest.tasks)
                for label in labels
            }
            if counts != expected_counts:
                raise ProtocolError(
                    f"fixed Frozen-134 task counts differ from reference manifest: {counts}"
                )
        validate_distinct_formal_tasks(tasks, expected_total=expected_total)
        task_items = tuple(
            TaskManifest(
                index, task.task_id, task_signature(task), f"frozen:{digest_before}",
                task.benchmark, str(system.harness.split),
                json.dumps({
                    "task_type": task.task_type,
                    "env_index": task.context.get("env_index"),
                    "game_file": task.context.get("game_file", ""),
                }, ensure_ascii=False, sort_keys=True),
            )
            for index, task in enumerate(tasks)
        )

        train_run_dir = _path(experiment.get("source_train_run_dir", "runs/alfworld_train_full_30"))
        train_manifest_path = train_run_dir / "run_manifest.json"
        if not train_manifest_path.is_file():
            raise ProtocolError(f"source train manifest missing: {train_manifest_path}")
        train_manifest = RunManifest.from_dict(json.loads(train_manifest_path.read_text(encoding="utf-8")))
        code_digest = hash_code(REPO_ROOT)
        _verify_source_train(
            train_run_dir=train_run_dir,
            train_manifest=train_manifest,
            freeze_manifest=freeze_manifest,
            frozen_digest=digest_before,
            current_code_digest=code_digest,
            current_llm_hash=hash_config(config.get("llm") or {}),
            require_source_code_match=bool(
                experiment.get("require_source_code_match", True)
            ),
            expected_source_git_revision=str(
                experiment.get("source_git_revision", "")
            ),
            reference_train_manifest=reference_train_manifest,
            expected_experiment_seed=experiment_seed,
            require_frozen_closure=(protocol == "r8_frozen134"),
            require_r9_freeze_audit=(protocol == "r9_frozen134"),
        )
        train_signatures = {item.task_signature for item in train_manifest.tasks}
        if source_train_replay:
            replay_identity = [
                (item.task_id, item.task_signature) for item in task_items
            ]
            train_identity = [
                (item.task_id, item.task_signature) for item in train_manifest.tasks
            ]
            if replay_identity != train_identity:
                raise ProtocolError(
                    "frozen source-train replay tasks differ from the immutable train manifest"
                )
        else:
            overlap = train_signatures & {item.task_signature for item in task_items}
            if overlap:
                raise ProtocolError(
                    f"held-out manifest overlaps train signatures ({len(overlap)})"
                )

        state_db = StateDatabase(output_dir / "run_state.sqlite3")
        try:
            config_digest = hash_config(config_path)
            store = ManifestStore(output_dir.parent, state_db)
            if resume:
                manifest = store.validate_resume(
                    run_id,
                    config_hash=config_digest,
                    code_commit=code_digest,
                    knowledge_digest=digest_before,
                    tasks=task_items,
                )
            else:
                manifest = RunManifest.create(
                    run_id=run_id, phase=phase,
                    config_hash=config_digest, code_commit=code_digest,
                    knowledge_digest=digest_before, tasks=task_items,
                    metadata={
                        "condition": "full", "source_train_run": train_manifest.run_id,
                        "run_started_at": invocation_started_at.isoformat(),
                        "environment": {"alfworld_version": "0.4.2"},
                        "task_types": labels, "tasks_per_type": per_type,
                        "total_tasks": expected_total,
                        "source_git_revision": str(
                            train_manifest.metadata.get("git_revision", "")
                        ),
                        "source_code_commit": train_manifest.code_commit,
                        "evaluator_code_commit": code_digest,
                        "source_code_match_required": bool(
                            experiment.get("require_source_code_match", True)
                        ),
                        **(
                            {
                                "seed": experiment_seed,
                                "reference_manifest_id": reference_test_manifest.manifest_id,
                                "reference_manifest_digest": reference_test_manifest.digest,
                                "reference_manifest_seed": reference_test_manifest.seed,
                                "reference_manifest_task_count": len(reference_test_manifest.tasks),
                                "reference_manifest_family_counts": counts,
                                "train_reference_manifest_id": reference_train_manifest.manifest_id,
                                "train_reference_manifest_digest": reference_train_manifest.digest,
                                "train_reference_manifest_seed": reference_train_manifest.seed,
                                "reference_manifest_disjoint_audit": reference_disjoint_audit,
                            }
                            if reference_train_manifest is not None
                            and reference_test_manifest is not None
                            else {}
                        ),
                    },
                )
                store.persist_before_run(manifest)
            run_started_at = str(
                manifest.metadata.get("run_started_at", manifest.created_at)
            )
            ensure_task_manifest(
                _path(experiment.get("task_manifest_path", "")), manifest
            )
            store.mark_run_state(run_id, RunState.RUNNING)
            by_id = {task.task_id: task for task in tasks}
            for item in store.tasks_to_run(manifest):
                artifact_before = artifact_audit_snapshot(system.database)
                try:
                    attempt_sequence = store.mark_task_running(
                        run_id, item.task_id, max_attempts=max_task_attempts,
                    )
                except ProtocolError:
                    store.mark_run_state(run_id, RunState.INFRASTRUCTURE_FAILED)
                    raise
                task_attempt = attempt_ledger.begin(
                    run_id=run_id,
                    task_id=item.task_id,
                    task_signature=item.task_signature,
                    attempt_kind="task",
                    sequence=attempt_sequence,
                )
                try:
                    trace = system.run_task(
                        by_id[item.task_id], attempt_id=task_attempt.attempt_id,
                    )
                    attempt_ledger.capture(task_attempt, reason="run_task_returned")
                    artifact_after = artifact_audit_snapshot(system.database)
                    artifact_growth = artifact_growth_audit(
                        artifact_before, artifact_after,
                    )
                    digest_after = system.knowledge_digest()
                    if digest_after != digest_before:
                        raise ProtocolError("frozen evaluation changed long-term knowledge")
                    result = {
                        "benchmark_success": trace.benchmark_success,
                        "task_contract_success": trace.task_contract_success,
                        "strict_task_success": trace.strict_task_success,
                        "learning_eligible": trace.learning_eligible,
                        "graph_self_sufficient_success": trace.graph_self_sufficient_success,
                        "infrastructure_failure": trace.infrastructure_failure,
                        "knowledge_digest_before": digest_before,
                        "knowledge_digest_after": digest_after,
                        "artifact_growth": artifact_growth,
                        "artifact_lifecycle": artifact_after,
                    }
                    if trace.infrastructure_failure:
                        store.mark_task_failed(
                            run_id, item.task_id, infrastructure=True,
                            trace_id=trace.trace_id, result=result,
                        )
                        store.mark_run_state(run_id, RunState.INFRASTRUCTURE_FAILED)
                        raise RuntimeError(f"infrastructure failure at held-out task {item.task_id}")
                    store.mark_task_completed(
                        run_id, item.task_id, trace_id=trace.trace_id, result=result
                    )
                    print(json.dumps({
                        "task": item.task_id,
                        "official_alfworld_won": trace.benchmark_success,
                        "strict_task_success": trace.strict_task_success,
                        "learning_eligible": trace.learning_eligible,
                        "trace_id": trace.trace_id,
                    }, ensure_ascii=False), flush=True)
                except Exception as primary:
                    def update_failed_state() -> None:
                        row = state_db.execute(
                            "SELECT state FROM run_tasks WHERE run_id=? AND task_id=?",
                            (run_id, item.task_id),
                        ).fetchone()
                        if row is not None and row["state"] == "running":
                            store.mark_task_failed(
                                run_id, item.task_id, infrastructure=True,
                                result={
                                    "error_type": type(primary).__name__,
                                    "error": str(primary),
                                },
                            )
                        store.mark_run_state(run_id, RunState.INFRASTRUCTURE_FAILED)

                    audit_failed_attempt(
                        primary=primary,
                        attempt=task_attempt,
                        attempt_ledger=attempt_ledger,
                        receipt_root=output_dir / "failure_receipts",
                        update_state=update_failed_state,
                        capture_reason="task_exception",
                    )
                    raise

            traces = load_task_report_traces(system.traces, state_db, run_id)
            task_trace_ids = {str(trace.get("trace_id", "")) for trace in traces}
            if "" in task_trace_ids or len(task_trace_ids) != len(traces):
                raise ProtocolError("frozen report contains invalid/duplicate task trace_id")
            attempt_usage_traces = attempt_ledger.auxiliary_traces(
                manifest=manifest,
                excluded_trace_ids=task_trace_ids,
            )
            resource_traces = [*traces, *attempt_usage_traces]
            frozen_v31_guards = validate_frozen_v31_guards(traces)
            validate_formal_usage(resource_traces)
            usage_coverage = validate_usage_event_persistence(
                system.usage.events,
                resource_traces,
            )
            print(json.dumps({
                "usage_trace_coverage": usage_coverage,
                "frozen_v31_guards": frozen_v31_guards,
            }, ensure_ascii=False), flush=True)
            report_stem = (
                "frozen_train30_replay_b6a82ed"
                if source_train_replay
                else (
                    f"frozen_eval_134_{'r9' if protocol == 'r9_frozen134' else 'r8' if protocol == 'r8_frozen134' else 'r7'}_seed{experiment_seed}"
                    if protocol in {"r7_frozen134", "r8_frozen134", "r9_frozen134"}
                    else (
                    "frozen_eval_60"
                    if run_id == "alfworld_frozen_eval_60"
                    else run_id
                    )
                )
            )
            report_title = (
                "AtomicSkillGraph v3 ALFWorld Frozen Train-30 Replay (b6a82ed bank)"
                if source_train_replay
                else (
                    f"AtomicSkillGraph v3 ALFWorld Frozen Held-out-134 "
                    f"{'R9' if protocol == 'r9_frozen134' else 'R8' if protocol == 'r8_frozen134' else 'R7'} Eval "
                    f"(seed {experiment_seed})"
                    if protocol in {"r7_frozen134", "r8_frozen134", "r9_frozen134"}
                    else "AtomicSkillGraph v3 ALFWorld Frozen Held-out Eval"
                )
            )
            write_reports(
                traces, output_dir / "reports", stem=report_stem,
                title=report_title,
                auxiliary_usage_traces=attempt_usage_traces,
                reasoning_effort_audit=formal_reasoning_effort_audit(config),
            )
            if system.knowledge_digest() != digest_before:
                raise ProtocolError("knowledge digest guard failed after report generation")
            run_ended_at = datetime.now(timezone.utc)
            if resume:
                parsed_started_at = datetime.fromisoformat(
                    run_started_at.replace("Z", "+00:00")
                )
                run_elapsed_seconds = max(
                    0.0, (run_ended_at - parsed_started_at).total_seconds()
                )
            else:
                run_elapsed_seconds = time.monotonic() - invocation_started_monotonic
            timing_path = write_run_observability(
                output_dir / "reports" / f"{report_stem}_run.json",
                run_id=run_id,
                run_started_at=run_started_at,
                run_ended_at=run_ended_at.isoformat(),
                run_elapsed_seconds=run_elapsed_seconds,
            )
            store.mark_run_state(run_id, RunState.COMPLETED)
            print(json.dumps({
                "run_id": run_id, "tasks": expected_total,
                "knowledge_digest_before": digest_before,
                "knowledge_digest_after": system.knowledge_digest(),
                "run_started_at": run_started_at,
                "run_ended_at": run_ended_at.isoformat(),
                "run_elapsed_seconds": run_elapsed_seconds,
                "run_observability": str(timing_path),
            }, ensure_ascii=False, indent=2))
        finally:
            state_db.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/alfworld_frozen_eval.yaml",
        help="frozen held-out YAML configuration",
    )
    parser.add_argument("--resume", action="store_true", help="resume at completed-task boundaries")
    args = parser.parse_args(argv)
    return run(args.config, resume=args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
