"""Common post-evaluator: official and strict success, per-family and totals.

The evaluator only reads the immutable worker sidecars and the replay
results of :mod:`.task_authority`; it never re-derives success from method
logs.  Paired transfer metrics against B0 Pure Dynamic are computed once a
B0 result exists and are reported as null until then.
"""

from __future__ import annotations

import json
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
    evolution_llm_calls: int
    evolution_prompt_tokens: int
    evolution_completion_tokens: int
    embedding_calls: int
    wall_time_ms: int
    infrastructure_failure: bool

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
            evolution_llm_calls=episode.evolution_llm_calls,
            evolution_prompt_tokens=episode.evolution_prompt_tokens,
            evolution_completion_tokens=episode.evolution_completion_tokens,
            embedding_calls=episode.embedding_calls,
            wall_time_ms=episode.wall_time_ms,
            infrastructure_failure=episode.infrastructure_failure,
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
            "evolution_llm_calls": self.evolution_llm_calls,
            "evolution_prompt_tokens": self.evolution_prompt_tokens,
            "evolution_completion_tokens": self.evolution_completion_tokens,
            "embedding_calls": self.embedding_calls,
            "wall_time_ms": self.wall_time_ms,
            "infrastructure_failure": self.infrastructure_failure,
        }


def summarize_rows(rows: list[TaskRow], *, task_types: list[str]) -> dict[str, Any]:
    valid = [row for row in rows if not row.infrastructure_failure]
    infra = len(rows) - len(valid)

    def rate(flag: str) -> float | None:
        scored = [row for row in valid if getattr(row, flag) is not None]
        if not scored:
            return None
        return round(sum(bool(getattr(row, flag)) for row in scored) / len(scored), 6)

    family: dict[str, dict[str, Any]] = {}
    for task_type in sorted({row.task_type for row in valid}):
        family_rows = [row for row in valid if row.task_type == task_type]
        family[task_type] = {
            "tasks": len(family_rows),
            "official_success": sum(row.official_success for row in family_rows),
            "official_rate": (
                round(sum(row.official_success for row in family_rows) / len(family_rows), 6)
                if family_rows else None
            ),
            "strict_success": sum(bool(row.strict_success) for row in family_rows),
            "strict_rate": (
                round(sum(bool(row.strict_success) for row in family_rows) / len(family_rows), 6)
                if family_rows else None
            ),
        }
    macro_families = [family[name]["official_rate"] for name in family.values()]
    macro = (
        round(sum(value for value in macro_families if value is not None) / len(task_types), 6)
        if family else None
    )
    return {
        "tasks": len(valid),
        "infrastructure_failed_episodes": infra,
        "official_success": sum(row.official_success for row in valid),
        "official_rate": rate("official_success"),
        "task_contract_rate": rate("task_contract_success"),
        "strict_rate": rate("strict_success"),
        "macro_family_official_rate": macro,
        "environment_actions": sum(row.environment_actions for row in valid),
        "target_llm_calls": sum(row.target_llm_calls for row in valid),
        "target_prompt_tokens": sum(row.target_prompt_tokens for row in valid),
        "target_completion_tokens": sum(row.target_completion_tokens for row in valid),
        "evolution_llm_calls": sum(row.evolution_llm_calls for row in valid),
        "evolution_prompt_tokens": sum(row.evolution_prompt_tokens for row in valid),
        "evolution_completion_tokens": sum(row.evolution_completion_tokens for row in valid),
        "embedding_calls": sum(row.embedding_calls for row in valid),
        "wall_time_ms": sum(row.wall_time_ms for row in valid),
        "family": family,
        # Paired transfer against B0 requires a B0 result on the same manifest.
        "positive_transfer": None,
        "negative_transfer": None,
    }


def write_rows_jsonl(rows: list[TaskRow], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return path
