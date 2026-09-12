"""Read-only GEPA callback instrumentation for formal audit metadata."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from collections import Counter
from pathlib import Path
from typing import Any


class GEPAAuditCallback:
    """Persist compact optimizer events without altering upstream state."""

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists():
            raise FileExistsError(self.output_path)
        self._lock = threading.Lock()
        self._counts: Counter[str] = Counter()
        self._last_budget = 0
        self._last_iteration = 0
        self._accepted_indices: list[int] = []
        self._rejected = 0
        self._full_val_evals = 0
        self._unexpected_merge_events = 0
        self._best_validation_score: float | None = None

    def _record(self, event_name: str, payload: dict[str, Any]) -> None:
        record = {
            "schema_version": 1,
            "event": event_name,
            **_json_safe(payload),
        }
        encoded = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock:
            with self.output_path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            self._counts[event_name] += 1

    def on_optimization_start(self, event: dict[str, Any]) -> None:
        candidate = dict(event["seed_candidate"])
        self._record("optimization_start", {
            "candidate_keys": sorted(candidate),
            "seed_candidate_sha256": _candidate_sha(candidate),
            "trainset_size": int(event["trainset_size"]),
            "valset_size": int(event["valset_size"]),
        })

    def on_optimization_end(self, event: dict[str, Any]) -> None:
        self._last_iteration = max(self._last_iteration, int(event["total_iterations"]))
        self._last_budget = max(self._last_budget, int(event["total_metric_calls"]))
        self._record("optimization_end", {
            "best_candidate_idx": int(event["best_candidate_idx"]),
            "total_iterations": int(event["total_iterations"]),
            "total_metric_calls": int(event["total_metric_calls"]),
        })

    def on_iteration_start(self, event: dict[str, Any]) -> None:
        self._last_iteration = max(self._last_iteration, int(event["iteration"]))
        self._record("iteration_start", {"iteration": int(event["iteration"])})

    def on_iteration_end(self, event: dict[str, Any]) -> None:
        self._record("iteration_end", {
            "iteration": int(event["iteration"]),
            "proposal_accepted": bool(event["proposal_accepted"]),
        })

    def on_candidate_selected(self, event: dict[str, Any]) -> None:
        self._record("candidate_selected", {
            "iteration": int(event["iteration"]),
            "candidate_idx": int(event["candidate_idx"]),
            "candidate_sha256": _candidate_sha(dict(event["candidate"])),
            "score": float(event["score"]),
        })

    def on_minibatch_sampled(self, event: dict[str, Any]) -> None:
        self._record("minibatch_sampled", {
            "iteration": int(event["iteration"]),
            "minibatch_ids": list(event["minibatch_ids"]),
            "trainset_size": int(event["trainset_size"]),
        })

    def on_evaluation_start(self, event: dict[str, Any]) -> None:
        inputs = list(event.get("inputs") or [])
        self._record("evaluation_start", {
            "iteration": int(event["iteration"]),
            "candidate_idx": event.get("candidate_idx"),
            "batch_size": int(event["batch_size"]),
            "capture_traces": bool(event["capture_traces"]),
            "task_ids": [
                str(item.get("task_id", item.get("id", "")))
                if isinstance(item, dict) else ""
                for item in inputs
            ],
            "is_seed_candidate": bool(event["is_seed_candidate"]),
        })

    def on_evaluation_end(self, event: dict[str, Any]) -> None:
        self._record("evaluation_end", {
            "iteration": int(event["iteration"]),
            "candidate_idx": event.get("candidate_idx"),
            "scores": [float(score) for score in event["scores"]],
            "has_trajectories": bool(event["has_trajectories"]),
            "is_seed_candidate": bool(event["is_seed_candidate"]),
        })

    def on_evaluation_skipped(self, event: dict[str, Any]) -> None:
        self._record("evaluation_skipped", {
            "iteration": int(event["iteration"]),
            "candidate_idx": int(event["candidate_idx"]),
            "reason": str(event["reason"]),
            "is_seed_candidate": bool(event["is_seed_candidate"]),
        })

    def on_reflective_dataset_built(self, event: dict[str, Any]) -> None:
        dataset = dict(event["dataset"])
        self._record("reflective_dataset_built", {
            "iteration": int(event["iteration"]),
            "candidate_idx": int(event["candidate_idx"]),
            "components": list(event["components"]),
            "record_counts": {
                str(key): len(value) for key, value in dataset.items()
            },
            "dataset_sha256": hashlib.sha256(
                json.dumps(dataset, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        })

    def on_proposal_start(self, event: dict[str, Any]) -> None:
        self._record("proposal_start", {
            "iteration": int(event["iteration"]),
            "parent_candidate_sha256": _candidate_sha(
                dict(event["parent_candidate"])
            ),
            "components": list(event["components"]),
        })

    def on_proposal_end(self, event: dict[str, Any]) -> None:
        instructions = dict(event["new_instructions"])
        self._record("proposal_end", {
            "iteration": int(event["iteration"]),
            "component_keys": sorted(instructions),
            "new_instruction_sha256": {
                key: hashlib.sha256(str(value).encode("utf-8")).hexdigest()
                for key, value in instructions.items()
            },
            "raw_output_present": {
                key: bool(str(value).strip())
                for key, value in dict(event.get("raw_lm_outputs") or {}).items()
            },
        })

    def on_candidate_accepted(self, event: dict[str, Any]) -> None:
        index = int(event["new_candidate_idx"])
        self._accepted_indices.append(index)
        self._record("candidate_accepted", {
            "iteration": int(event["iteration"]),
            "new_candidate_idx": index,
            "new_score": float(event["new_score"]),
            "parent_ids": list(event["parent_ids"]),
        })

    def on_candidate_rejected(self, event: dict[str, Any]) -> None:
        self._rejected += 1
        self._record("candidate_rejected", {
            "iteration": int(event["iteration"]),
            "old_score": float(event["old_score"]),
            "new_score": float(event["new_score"]),
            "reason": str(event["reason"]),
        })

    def on_pareto_front_updated(self, event: dict[str, Any]) -> None:
        self._record("pareto_front_updated", {
            "iteration": int(event["iteration"]),
            "new_front": list(event["new_front"]),
            "displaced_candidates": list(event["displaced_candidates"]),
        })

    def on_valset_evaluated(self, event: dict[str, Any]) -> None:
        self._full_val_evals += 1
        score = float(event["average_score"])
        if self._best_validation_score is None or score > self._best_validation_score:
            self._best_validation_score = score
        self._record("valset_evaluated", {
            "iteration": int(event["iteration"]),
            "candidate_idx": int(event["candidate_idx"]),
            "candidate_sha256": _candidate_sha(dict(event["candidate"])),
            "average_score": score,
            "num_examples_evaluated": int(event["num_examples_evaluated"]),
            "total_valset_size": int(event["total_valset_size"]),
            "is_best_program": bool(event["is_best_program"]),
        })

    def on_budget_updated(self, event: dict[str, Any]) -> None:
        self._last_budget = max(self._last_budget, int(event["metric_calls_used"]))
        self._record("budget_updated", {
            "iteration": int(event["iteration"]),
            "metric_calls_used": int(event["metric_calls_used"]),
            "metric_calls_delta": int(event["metric_calls_delta"]),
            "metric_calls_remaining": event.get("metric_calls_remaining"),
        })

    def on_state_saved(self, event: dict[str, Any]) -> None:
        self._record("state_saved", {
            "iteration": int(event["iteration"]),
            "run_dir_present": bool(event.get("run_dir")),
        })

    def on_error(self, event: dict[str, Any]) -> None:
        self._record("optimizer_error", {
            "iteration": int(event["iteration"]),
            "error_type": type(event["exception"]).__name__,
            "will_continue": bool(event["will_continue"]),
        })

    def on_merge_attempted(self, event: dict[str, Any]) -> None:
        self._unexpected_merge_events += 1
        self._record("merge_attempted", {"iteration": int(event["iteration"])})

    def on_merge_accepted(self, event: dict[str, Any]) -> None:
        self._unexpected_merge_events += 1
        self._record("merge_accepted", {"iteration": int(event["iteration"])})

    def on_merge_rejected(self, event: dict[str, Any]) -> None:
        self._unexpected_merge_events += 1
        self._record("merge_rejected", {"iteration": int(event["iteration"])})

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "event_counts": dict(sorted(self._counts.items())),
                "iteration_count": self._last_iteration,
                "accepted_candidate_count": len(set(self._accepted_indices)),
                "rejected_candidate_count": self._rejected,
                "num_full_val_evals": self._full_val_evals,
                "reflection_calls": int(self._counts.get("proposal_end", 0)),
                "actual_total_metric_calls": self._last_budget,
                "best_validation_score": self._best_validation_score,
                "unexpected_merge_events": self._unexpected_merge_events,
            }


def _candidate_sha(candidate: dict[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(sorted(candidate.items()), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, set):
        return sorted(_json_safe(child) for child in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
