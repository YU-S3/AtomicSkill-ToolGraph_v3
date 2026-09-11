"""Common post-evaluator: official and strict success, per-family and totals.

The evaluator only reads the immutable worker sidecars and the replay
results of :mod:`.task_authority`; it never re-derives success from method
logs.  Paired transfer metrics against B0 Pure Dynamic are computed once a
B0 result exists and are reported as null until then.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import CommonEpisodeRecord


@dataclass(frozen=True)
class TaskRow:
    method: str
    phase: str
    run_seed: int
    task_id: str
    task_type: str
    manifest_index: int
    official_success: bool
    task_contract_success: bool | None
    strict_success: bool | None
    environment_actions: int
    invalid_actions: int | None
    target_llm_calls: int
    target_prompt_tokens: int
    target_completion_tokens: int
    target_reasoning_tokens: int
    evolution_llm_calls: int
    evolution_prompt_tokens: int
    evolution_completion_tokens: int
    embedding_calls: int
    wall_time_ms: int
    infrastructure_failure: bool
    command_turns: int = 0
    timeout: bool = False
    gamefile: str = ""
    gamefile_hash: str = ""
    artifact_digest_before: str = ""
    artifact_digest_after: str = ""
    method_metrics: dict[str, Any] | None = None

    @classmethod
    def from_episode(cls, episode: CommonEpisodeRecord) -> "TaskRow":
        return cls(
            method=episode.method,
            phase=episode.phase,
            run_seed=episode.run_seed,
            task_id=episode.task_id,
            task_type=episode.task_type,
            manifest_index=episode.manifest_index,
            official_success=episode.official_success,
            task_contract_success=episode.task_contract_success,
            strict_success=episode.strict_success,
            environment_actions=episode.environment_actions,
            invalid_actions=episode.invalid_actions,
            target_llm_calls=episode.target_llm_calls,
            target_prompt_tokens=episode.target_prompt_tokens,
            target_completion_tokens=episode.target_completion_tokens,
            target_reasoning_tokens=episode.target_reasoning_tokens,
            evolution_llm_calls=episode.evolution_llm_calls,
            evolution_prompt_tokens=episode.evolution_prompt_tokens,
            evolution_completion_tokens=episode.evolution_completion_tokens,
            embedding_calls=episode.embedding_calls,
            wall_time_ms=episode.wall_time_ms,
            infrastructure_failure=episode.infrastructure_failure,
            command_turns=episode.command_turns,
            timeout=episode.timeout,
            gamefile=episode.gamefile,
            gamefile_hash=episode.gamefile_hash,
            artifact_digest_before=episode.artifact_digest_before,
            artifact_digest_after=episode.artifact_digest_after,
            method_metrics=dict(episode.method_metrics),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "phase": self.phase,
            "run_seed": self.run_seed,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "manifest_index": self.manifest_index,
            "official_success": self.official_success,
            "task_contract_success": self.task_contract_success,
            "strict_success": self.strict_success,
            "environment_actions": self.environment_actions,
            "invalid_actions": self.invalid_actions,
            "target_llm_calls": self.target_llm_calls,
            "target_prompt_tokens": self.target_prompt_tokens,
            "target_completion_tokens": self.target_completion_tokens,
            "target_reasoning_tokens": self.target_reasoning_tokens,
            "evolution_llm_calls": self.evolution_llm_calls,
            "evolution_prompt_tokens": self.evolution_prompt_tokens,
            "evolution_completion_tokens": self.evolution_completion_tokens,
            "embedding_calls": self.embedding_calls,
            "wall_time_ms": self.wall_time_ms,
            "infrastructure_failure": self.infrastructure_failure,
            "command_turns": self.command_turns,
            "timeout": self.timeout,
            "gamefile": self.gamefile,
            "gamefile_hash": self.gamefile_hash,
            "artifact_digest_before": self.artifact_digest_before,
            "artifact_digest_after": self.artifact_digest_after,
            "method_metrics": dict(self.method_metrics or {}),
        }


def summarize_rows(rows: list[TaskRow], *, task_types: list[str]) -> dict[str, Any]:
    declared_types = tuple(str(name).strip() for name in task_types)
    if not declared_types or any(not name for name in declared_types):
        raise ValueError("task_types must contain non-empty task family names")
    if len(set(declared_types)) != len(declared_types):
        raise ValueError("task_types must not contain duplicate task family names")
    observed_types = {row.task_type for row in rows}
    unknown_types = sorted(observed_types - set(declared_types))
    if unknown_types:
        raise ValueError(f"rows contain undeclared task families: {unknown_types}")

    valid = [row for row in rows if not row.infrastructure_failure]
    infra = len(rows) - len(valid)
    valid_types = {row.task_type for row in valid}
    missing_types = [name for name in declared_types if name not in valid_types]
    if missing_types:
        raise ValueError(
            "cannot compute a complete macro average; task families have no "
            f"scorable rows: {missing_types}"
        )

    def rate(flag: str) -> float | None:
        scored = [row for row in valid if getattr(row, flag) is not None]
        if not scored:
            return None
        return round(sum(bool(getattr(row, flag)) for row in scored) / len(scored), 6)

    family: dict[str, dict[str, Any]] = {}
    for task_type in declared_types:
        family_rows = [row for row in valid if row.task_type == task_type]
        contract_rows = [
            row for row in family_rows if row.task_contract_success is not None
        ]
        strict_rows = [row for row in family_rows if row.strict_success is not None]
        family[task_type] = {
            "tasks": len(family_rows),
            "official_success": sum(row.official_success for row in family_rows),
            "official_rate": (
                round(sum(row.official_success for row in family_rows) / len(family_rows), 6)
                if family_rows else None
            ),
            "task_contract_scored_tasks": len(contract_rows),
            "task_contract_success": sum(
                bool(row.task_contract_success) for row in contract_rows
            ),
            "task_contract_rate": (
                round(
                    sum(bool(row.task_contract_success) for row in contract_rows)
                    / len(contract_rows),
                    6,
                )
                if contract_rows else None
            ),
            "strict_scored_tasks": len(strict_rows),
            "strict_success": sum(bool(row.strict_success) for row in strict_rows),
            "strict_rate": (
                round(
                    sum(bool(row.strict_success) for row in strict_rows)
                    / len(strict_rows),
                    6,
                )
                if strict_rows else None
            ),
        }
    macro_families = [family[name]["official_rate"] for name in declared_types]
    # Every declared family has at least one scorable row by the guard above.
    macro = round(sum(float(value) for value in macro_families) / len(declared_types), 6)

    successes = sum(row.official_success for row in valid)
    environment_actions = [row.environment_actions for row in valid]
    target_calls = [row.target_llm_calls for row in valid]
    total_calls = [
        row.target_llm_calls + row.evolution_llm_calls for row in valid
    ]
    # OpenAI-compatible usage defines completion_tokens as including any
    # reasoning tokens.  Keep reasoning as a reported decomposition, but do
    # not add it a second time to total/cost token counts.
    target_tokens = [
        row.target_prompt_tokens + row.target_completion_tokens
        for row in valid
    ]
    evolution_tokens = [
        row.evolution_prompt_tokens + row.evolution_completion_tokens
        for row in valid
    ]
    total_tokens = [
        target + evolution
        for target, evolution in zip(target_tokens, evolution_tokens)
    ]
    latencies = [row.wall_time_ms for row in valid]
    total_environment_actions = sum(environment_actions)
    total_target_calls = sum(target_calls)
    total_evolution_calls = sum(row.evolution_llm_calls for row in valid)
    total_calls_value = sum(total_calls)
    total_target_tokens = sum(target_tokens)
    total_evolution_tokens = sum(evolution_tokens)
    total_tokens_value = sum(total_tokens)
    scored_tasks = len(valid)
    invalid_action_rows = [
        row.invalid_actions for row in valid if row.invalid_actions is not None
    ]

    return {
        "tasks": scored_tasks,
        "attempted_tasks": len(rows),
        "infrastructure_failed_episodes": infra,
        "official_success": successes,
        "official_rate": rate("official_success"),
        "micro_average_official_rate": rate("official_success"),
        "task_contract_rate": rate("task_contract_success"),
        "strict_rate": rate("strict_success"),
        "macro_family_official_rate": macro,
        "environment_actions": total_environment_actions,
        "actions_per_task": _ratio(total_environment_actions, scored_tasks),
        "actions_per_solved": _ratio(total_environment_actions, successes),
        "p50_actions": _nearest_rank(environment_actions, 0.50),
        "p90_actions": _nearest_rank(environment_actions, 0.90),
        "invalid_actions_scored_tasks": len(invalid_action_rows),
        "invalid_actions": sum(invalid_action_rows),
        "invalid_actions_per_scored_task": _ratio(
            sum(invalid_action_rows), len(invalid_action_rows)
        ),
        "target_llm_calls": total_target_calls,
        "evolution_llm_calls": total_evolution_calls,
        "llm_calls": total_calls_value,
        "target_calls_per_task": _ratio(total_target_calls, scored_tasks),
        "evolution_calls_per_task": _ratio(total_evolution_calls, scored_tasks),
        "calls_per_task": _ratio(total_calls_value, scored_tasks),
        "p50_calls": _nearest_rank(total_calls, 0.50),
        "p90_calls": _nearest_rank(total_calls, 0.90),
        "target_prompt_tokens": sum(row.target_prompt_tokens for row in valid),
        "target_completion_tokens": sum(row.target_completion_tokens for row in valid),
        "target_reasoning_tokens": sum(row.target_reasoning_tokens for row in valid),
        "reasoning_tokens_in_completion": True,
        "target_tokens": total_target_tokens,
        "target_tokens_per_task": _ratio(total_target_tokens, scored_tasks),
        "evolution_prompt_tokens": sum(row.evolution_prompt_tokens for row in valid),
        "evolution_completion_tokens": sum(row.evolution_completion_tokens for row in valid),
        "evolution_tokens": total_evolution_tokens,
        "evolution_tokens_per_task": _ratio(total_evolution_tokens, scored_tasks),
        "llm_tokens": total_tokens_value,
        "tokens_per_task": _ratio(total_tokens_value, scored_tasks),
        "tokens_per_solved": _ratio(total_tokens_value, successes),
        "p50_tokens": _nearest_rank(total_tokens, 0.50),
        "p90_tokens": _nearest_rank(total_tokens, 0.90),
        "embedding_calls": sum(row.embedding_calls for row in valid),
        "wall_time_ms": sum(latencies),
        "latency_per_task_ms": _ratio(sum(latencies), scored_tasks),
        "p50_latency_ms": _nearest_rank(latencies, 0.50),
        "p90_latency_ms": _nearest_rank(latencies, 0.90),
        "family": family,
        # Paired transfer against B0 requires a B0 result on the same manifest.
        "positive_transfer": None,
        "negative_transfer": None,
        "delta_vs_pure_dynamic": None,
        "paired_transfer_status": "unavailable_without_matching_b0_result",
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def _nearest_rank(values: list[int], quantile: float) -> int | None:
    """Return the deterministic nearest-rank sample quantile.

    P50/P90 are reporting statistics rather than interpolated measurements;
    selecting an observed sample keeps integer actions, calls, tokens and
    millisecond latency auditable.
    """

    if not values:
        return None
    ordered = sorted(int(value) for value in values)
    rank = max(1, math.ceil(float(quantile) * len(ordered)))
    return ordered[rank - 1]


def write_rows_jsonl(rows: list[TaskRow], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return path
