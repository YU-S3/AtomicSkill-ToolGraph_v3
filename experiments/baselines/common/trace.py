"""Common sidecar trace writer for baseline episodes.

Every rollout batch appends its ``CommonEpisodeRecord`` rows plus per-episode
``EnvironmentActionEvent`` rows to append-only JSONL files inside the rollout
directory.  These sidecars are the sole source the controller reads for
reporting: no baseline number is ever derived by re-parsing method logs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import CommonEpisodeRecord


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


class CommonSidecarWriter:
    """Append-only sidecar writer rooted at one rollout directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @property
    def episodes_path(self) -> Path:
        return self.root / "common_episodes.jsonl"

    @property
    def events_path(self) -> Path:
        return self.root / "common_environment_actions.jsonl"

    def write_episodes(self, episodes: list[CommonEpisodeRecord]) -> None:
        _append_jsonl(self.episodes_path, [episode.to_dict() for episode in episodes])

    def write_environment_actions(self, rows: list[dict[str, Any]]) -> None:
        _append_jsonl(self.events_path, rows)


def load_episodes(path: str | Path) -> list[CommonEpisodeRecord]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"common episodes sidecar is missing: {path}")
    episodes: list[CommonEpisodeRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"common episodes sidecar is corrupt at line {line_number}") from exc
        episodes.append(CommonEpisodeRecord(**{
            field: payload.get(field)
            for field in (
                "method", "phase", "run_seed", "task_id", "task_type",
                "manifest_index", "gamefile", "gamefile_hash",
                "official_success", "task_contract_success", "strict_success",
                "environment_actions", "invalid_actions", "command_turns",
                "timeout", "target_llm_calls", "target_prompt_tokens",
                "target_completion_tokens", "target_reasoning_tokens",
                "evolution_llm_calls", "evolution_prompt_tokens",
                "evolution_completion_tokens", "embedding_calls",
                "wall_time_ms", "artifact_digest_before",
                "artifact_digest_after", "method_metrics",
                "infrastructure_failure", "infrastructure_error",
            )
        }))
    return episodes
