"""Text-skill episode runner for the SkillOpt baseline.

One episode = one call of the upstream SkillOpt ALFWorld rollout
(``skillopt.envs.alfworld.rollout.run_alfworld_batch``) on the exact
manifest gamefile.  The upstream loop is reused verbatim: text observation
templating, ``<think>/<action>`` protocol, target-model calls through
``chat_target``, and ``infos["won"]`` as the official success authority.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.baselines.common.usage import (
    RoleUsage,
    UsageSnapshot,
    usage_from_skillopt_token_summary,
)


@dataclass
class EpisodeOutcome:
    task: dict[str, Any]
    skillopt_row: dict[str, Any]
    conversation: list[dict[str, Any]]
    target_usage: RoleUsage = field(default_factory=RoleUsage)
    wall_time_ms: int = 0
    infrastructure_failure: bool = False
    infrastructure_error: str = ""


_SPLIT_MODES = {
    "train": ("train", True),
    "valid_seen": ("eval_in_distribution", False),
    "valid_unseen": ("eval_out_of_distribution", False),
}


class SkillOptTextEpisodeRunner:
    """Run one SkillOpt text-skill episode through the upstream rollout.

    ``episode_fn`` is injectable only for deterministic tests; the production
    path is the upstream ``run_alfworld_batch``.
    """

    def __init__(
        self,
        *,
        max_actions: int = 100,
        max_completion_tokens: int = 16384,
        seed: int = 42,
        alfworld_data: str = "",
        episode_fn: Any | None = None,
    ) -> None:
        self.max_actions = max_actions
        self.max_completion_tokens = max_completion_tokens
        self.seed = seed
        self.alfworld_data = alfworld_data or os.environ.get(
            "ALFWORLD_DATA", str(Path.home() / ".cache" / "alfworld"),
        )
        self._episode_fn = episode_fn

    def _run_upstream_episode(
        self,
        task: dict[str, Any],
        skill_content: str,
        out_dir: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        from skillopt.envs.alfworld.rollout import (
            build_alfworld_env,
            run_alfworld_batch,
        )

        source_split = str(task.get("source_split", "train"))
        if source_split not in _SPLIT_MODES:
            raise ValueError(f"unsupported source split: {source_split}")
        eval_dataset, is_train = _SPLIT_MODES[source_split]
        gamefile = str(task.get("gamefile", ""))
        if not gamefile:
            raise ValueError("baseline task is missing its gamefile")
        env_seed = self.seed + int(task.get("env_index", 0))
        env = build_alfworld_env(
            env_num=1,
            eval_dataset=eval_dataset,
            seed=env_seed,
            is_train=is_train,
            specific_gamefiles=[gamefile],
        )
        try:
            rows = run_alfworld_batch(
                env_manager=env,
                skill_content=skill_content,
                max_steps=self.max_actions,
                out_root=out_dir,
                max_api_workers=1,
                max_completion_tokens=self.max_completion_tokens,
                result_ids=[str(task.get("id", "")) or "env_000"],
            )
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        if len(rows) != 1:
            raise RuntimeError(
                f"SkillOpt rollout returned {len(rows)} rows for one episode"
            )
        row = rows[0]
        conversation: list[dict[str, Any]] = []
        conversation_path = Path(out_dir) / "predictions" / str(row.get("id", "")) / "conversation.json"
        if conversation_path.is_file():
            try:
                payload = json.loads(conversation_path.read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    conversation = [dict(item) for item in payload]
            except (OSError, json.JSONDecodeError):
                conversation = []
        return row, conversation

    def run(
        self,
        task: dict[str, Any],
        skill_content: str,
        out_dir: str,
    ) -> EpisodeOutcome:
        """Run one episode.  Target usage is the token-tracker delta across
        this exact episode (one upstream rollout call per episode)."""

        from skillopt.model import get_token_summary

        usage_before = get_token_summary()
        started = time.time()
        try:
            if self._episode_fn is not None:
                row, conversation = self._episode_fn(
                    task, skill_content, out_dir,
                )
            else:
                row, conversation = self._run_upstream_episode(
                    task, skill_content, out_dir,
                )
        except Exception as exc:
            return EpisodeOutcome(
                task=dict(task),
                skillopt_row={"id": str(task.get("id", "")), "hard": 0, "soft": 0.0},
                conversation=[],
                wall_time_ms=int((time.time() - started) * 1000),
                infrastructure_failure=True,
                infrastructure_error=f"{type(exc).__name__}: {exc}",
            )
        usage_after = get_token_summary()
        target_usage = UsageSnapshot.delta(
            usage_from_skillopt_token_summary(usage_after),
            usage_from_skillopt_token_summary(usage_before),
        ).target
        return EpisodeOutcome(
            task=dict(task),
            skillopt_row=dict(row),
            conversation=conversation,
            target_usage=target_usage,
            wall_time_ms=int((time.time() - started) * 1000),
        )
