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

from experiments.baselines.common.formal_validation import verify_observed_gamefile
from experiments.baselines.common.manifest import ManifestTask
from experiments.baselines.common.usage import (
    RoleUsage,
    UsageSnapshot,
    usage_from_skillopt_token_summary,
)
from .provider_observer import ProviderCallExhausted


@dataclass
class EpisodeOutcome:
    task: dict[str, Any]
    skillopt_row: dict[str, Any]
    conversation: list[dict[str, Any]]
    target_usage: RoleUsage = field(default_factory=RoleUsage)
    wall_time_ms: int = 0
    infrastructure_failure: bool = False
    infrastructure_error: str = ""
    actual_gamefile: str = ""


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
        rollout_id: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
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
        base_env = build_alfworld_env(
            env_num=1,
            eval_dataset=eval_dataset,
            seed=env_seed,
            is_train=is_train,
            specific_gamefiles=[gamefile],
        )
        from .provider_observer import active_provider_observer

        observer = active_provider_observer()
        if observer is None:
            raise RuntimeError(
                "provider observer is not installed; provider failures cannot be "
                "isolated from task outcomes"
            )
        env = _ExactManifestEnvironment(
            base_env,
            task=task,
            alfworld_data=self.alfworld_data,
            provider_observer=observer,
        )
        episode_context = observer.episode(
            str(task.get("id", "")), rollout_id=rollout_id,
        )
        try:
            with episode_context:
                rows = run_alfworld_batch(
                    env_manager=env,
                    skill_content=skill_content,
                    max_steps=self.max_actions,
                    out_root=out_dir,
                    max_api_workers=1,
                    max_completion_tokens=self.max_completion_tokens,
                    result_ids=[str(task.get("id", "")) or "env_000"],
                )
                observer.raise_if_active_episode_failed()
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        if len(rows) != 1:
            raise RuntimeError(
                f"SkillOpt rollout returned {len(rows)} rows for one episode"
            )
        row = rows[0]
        if not env.actual_gamefile:
            raise RuntimeError("ALFWorld reset produced no verified gamefile identity")
        conversation_path = Path(out_dir) / "predictions" / str(row.get("id", "")) / "conversation.json"
        if not conversation_path.is_file():
            raise RuntimeError(
                f"SkillOpt rollout produced no conversation trace: {conversation_path}"
            )
        try:
            payload = json.loads(conversation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"SkillOpt conversation trace is unreadable: {conversation_path}"
            ) from exc
        conversation = _validate_episode_payload(
            task=task,
            row=row,
            conversation=payload,
            actual_gamefile=env.actual_gamefile,
        )
        return row, conversation, env.actual_gamefile

    def run(
        self,
        task: dict[str, Any],
        skill_content: str,
        out_dir: str,
        rollout_id: str = "",
    ) -> EpisodeOutcome:
        """Run one episode.  Target usage is the token-tracker delta across
        this exact episode (one upstream rollout call per episode)."""

        started = time.time()
        observer = None
        provider_cursor = 0
        usage_before: dict[str, Any] | None = None
        if self._episode_fn is None:
            from .provider_observer import active_provider_observer

            observer = active_provider_observer()
            if observer is None:
                return EpisodeOutcome(
                    task=dict(task),
                    skillopt_row={
                        "id": str(task.get("id", "")), "hard": 0, "soft": 0.0,
                    },
                    conversation=[],
                    wall_time_ms=int((time.time() - started) * 1000),
                    infrastructure_failure=True,
                    infrastructure_error=(
                        "RuntimeError: provider observer is not installed"
                    ),
                )
            provider_cursor = observer.event_cursor()
        else:
            from skillopt.model import get_token_summary

            try:
                usage_before = get_token_summary()
            except Exception as exc:
                return EpisodeOutcome(
                    task=dict(task),
                    skillopt_row={
                        "id": str(task.get("id", "")), "hard": 0, "soft": 0.0,
                    },
                    conversation=[],
                    wall_time_ms=int((time.time() - started) * 1000),
                    infrastructure_failure=True,
                    infrastructure_error=_safe_error(exc),
                )
        failure: BaseException | None = None
        row: dict[str, Any] = {}
        conversation: list[dict[str, Any]] = []
        actual_gamefile = ""
        try:
            if self._episode_fn is not None:
                fixture_result = self._episode_fn(
                    task, skill_content, out_dir,
                )
                if len(fixture_result) == 2:
                    row, conversation = fixture_result
                    actual_gamefile = str(task.get("gamefile", ""))
                else:
                    row, conversation, actual_gamefile = fixture_result
                conversation = _validate_episode_payload(
                    task=task,
                    row=row,
                    conversation=conversation,
                    actual_gamefile=actual_gamefile,
                )
            else:
                row, conversation, actual_gamefile = self._run_upstream_episode(
                    task, skill_content, out_dir, rollout_id,
                )
        except ProviderCallExhausted as exc:
            # This is our boundary outside pinned SkillOpt.  The sentinel must
            # bypass upstream model fallbacks, then become a durable
            # infrastructure outcome before the batch stops.
            failure = exc
        except Exception as exc:
            failure = exc

        try:
            if observer is not None:
                task_id = str(task.get("id", ""))
                events = [
                    event
                    for event in observer.events_since(
                        provider_cursor, rollout_id=rollout_id,
                    )
                    if str(event.get("episode_task_id", "")) == task_id
                ]
                target_usage = _usage_from_provider_events(
                    events, require_all_succeeded=failure is None,
                )
                if failure is None and target_usage.calls <= 0:
                    raise RuntimeError(
                        f"episode {task_id!r} produced no provider-call evidence"
                    )
            else:
                from skillopt.model import get_token_summary

                usage_after = get_token_summary()
                target_usage = UsageSnapshot.delta(
                    usage_from_skillopt_token_summary(usage_after),
                    usage_from_skillopt_token_summary(usage_before or {}),
                ).target
        except Exception as exc:
            target_usage = RoleUsage()
            if failure is None:
                failure = exc

        if failure is not None:
            return EpisodeOutcome(
                task=dict(task),
                skillopt_row={"id": str(task.get("id", "")), "hard": 0, "soft": 0.0},
                conversation=list(conversation),
                target_usage=target_usage,
                wall_time_ms=int((time.time() - started) * 1000),
                infrastructure_failure=True,
                infrastructure_error=_safe_error(failure),
                actual_gamefile=str(actual_gamefile),
            )
        return EpisodeOutcome(
            task=dict(task),
            skillopt_row=dict(row),
            conversation=conversation,
            target_usage=target_usage,
            wall_time_ms=int((time.time() - started) * 1000),
            actual_gamefile=str(actual_gamefile),
        )


def _usage_from_provider_events(
    events: list[dict[str, Any]],
    *,
    require_all_succeeded: bool,
) -> RoleUsage:
    """Build a concurrency-safe per-episode usage total from audit events."""

    usage = RoleUsage()
    for event in events:
        if str(event.get("role", "")) != "target":
            raise RuntimeError("episode provider evidence contains a non-target call")
        status = str(event.get("status", ""))
        if status != "succeeded":
            if require_all_succeeded:
                raise RuntimeError("episode provider evidence contains a failed call")
            continue
        prompt = int(event.get("prompt_tokens", -1))
        completion = int(event.get("completion_tokens", -1))
        total = int(event.get("total_tokens", -1))
        if prompt < 0 or completion <= 0 or total != prompt + completion:
            raise RuntimeError("episode provider evidence has invalid token usage")
        reasoning_status = str(
            event.get("reasoning_tokens_status", "unavailable")
        )
        raw_reasoning = event.get("reasoning_tokens")
        if reasoning_status == "reported":
            reasoning = int(raw_reasoning)
            if reasoning < 0 or reasoning > completion:
                raise RuntimeError("episode provider evidence has invalid reasoning usage")
        elif reasoning_status == "unavailable" and raw_reasoning in {None, 0}:
            reasoning = 0
        else:
            raise RuntimeError("episode provider evidence has inconsistent reasoning usage")
        usage.calls += 1
        usage.prompt_tokens += prompt
        usage.completion_tokens += completion
        usage.reasoning_tokens += reasoning
    return usage


def _canonical_file(value: str | Path) -> str:
    if not str(value).strip():
        raise RuntimeError("ALFWorld reset returned an empty gamefile")
    try:
        return str(Path(value).expanduser().resolve(strict=True))
    except OSError as exc:
        raise RuntimeError(f"ALFWorld gamefile is not readable: {value}") from exc


def _safe_error(exc: BaseException) -> str:
    """Return useful failure context without persisting credentials."""

    message = str(exc)
    live_key = os.environ.get("MODEL_API_KEY", "").strip()
    if live_key:
        message = message.replace(live_key, "<redacted>")
    return f"{type(exc).__name__}: {message[:500]}"


def _validate_episode_payload(
    *,
    task: dict[str, Any],
    row: dict[str, Any],
    conversation: Any,
    actual_gamefile: str,
) -> list[dict[str, Any]]:
    expected_id = str(task.get("id", ""))
    if not expected_id or str(row.get("id", "")) != expected_id:
        raise RuntimeError(
            f"SkillOpt rollout id mismatch: expected {expected_id!r}, "
            f"got {row.get('id')!r}"
        )
    if not isinstance(row.get("agent_ok", True), bool) or not row.get("agent_ok", True):
        raise RuntimeError("SkillOpt rollout reports that its agent did not run successfully")
    if not isinstance(conversation, list) or any(
        not isinstance(item, dict) for item in conversation
    ):
        raise RuntimeError("SkillOpt conversation trace must be a list of mappings")
    normalized = [dict(item) for item in conversation]
    try:
        reported_turns = int(row.get("n_turns", -1))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("SkillOpt rollout n_turns must be an integer") from exc
    if len(normalized) != reported_turns:
        raise RuntimeError(
            "SkillOpt conversation length does not match n_turns: "
            f"{len(normalized)} != {row.get('n_turns')!r}"
        )
    for index, step in enumerate(normalized):
        if int(step.get("step", -1)) != index:
            raise RuntimeError(
                "SkillOpt conversation step indexes are not contiguous: "
                f"expected {index}, got {step.get('step')!r}"
            )
        action = step.get("action")
        if not isinstance(action, str) or not action.strip():
            raise RuntimeError(
                f"SkillOpt conversation step {index} has no executable action"
            )
    expected_gamefile = str(actual_gamefile).strip()
    reported_gamefile = str(row.get("gamefile", "")).strip()
    if not expected_gamefile or not reported_gamefile:
        raise RuntimeError("SkillOpt rollout is missing verified gamefile evidence")
    if Path(expected_gamefile).is_absolute() or Path(reported_gamefile).is_absolute():
        expected_gamefile = _canonical_file(expected_gamefile)
        reported_gamefile = _canonical_file(reported_gamefile)
    if reported_gamefile != expected_gamefile:
        raise RuntimeError(
            "SkillOpt rollout gamefile differs from the verified reset gamefile: "
            f"{reported_gamefile!r} != {expected_gamefile!r}"
        )
    return normalized


class _ExactManifestEnvironment:
    """Verify ALFWorld reset authority before the first model action."""

    def __init__(
        self,
        environment: Any,
        *,
        task: dict[str, Any],
        alfworld_data: str | Path,
        provider_observer: Any | None,
    ) -> None:
        self._environment = environment
        self._task = dict(task)
        self._alfworld_data = Path(alfworld_data).expanduser().resolve(strict=True)
        self._provider_observer = provider_observer
        self.actual_gamefile = ""

    def reset(self, *args: Any, **kwargs: Any):
        observations, infos = self._environment.reset(*args, **kwargs)
        if not isinstance(infos, list) or len(infos) != 1 or not isinstance(infos[0], dict):
            raise RuntimeError("ALFWorld reset did not return one metadata mapping")
        manifest_task = ManifestTask(
            index=int(self._task.get("manifest_index", -1)),
            task_id=str(self._task.get("task_id", self._task.get("id", ""))),
            task_type=str(self._task.get("task_type", "")),
            source_split=str(self._task.get("source_split", "")),
            env_index=int(self._task.get("env_index", -1)),
            gamefile_rel=str(self._task.get("gamefile", "")),
            gamefile_sha256=str(self._task.get("gamefile_sha256", "")),
            task_signature=str(self._task.get("task_signature", "")),
        )
        self.actual_gamefile = str(verify_observed_gamefile(
            manifest_task,
            str(infos[0].get("extra.gamefile", "")),
            alfworld_data=self._alfworld_data,
        ))
        return observations, infos

    def step(self, *args: Any, **kwargs: Any):
        if self._provider_observer is not None:
            self._provider_observer.raise_if_active_episode_failed()
        return self._environment.step(*args, **kwargs)

    def close(self) -> None:
        close = getattr(self._environment, "close", None)
        if callable(close):
            close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._environment, name)
