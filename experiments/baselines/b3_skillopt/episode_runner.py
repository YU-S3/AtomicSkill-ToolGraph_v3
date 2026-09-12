"""Text-skill episode runner for the SkillOpt baseline.

One episode = one call of the upstream SkillOpt ALFWorld rollout
(``skillopt.envs.alfworld.rollout.run_alfworld_batch``) on the exact
manifest gamefile.  The upstream loop is reused verbatim: text observation
templating, ``<think>/<action>`` protocol, target-model calls through
``chat_target``, and ``infos["won"]`` as the official success authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
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
    failure_kind: str = ""
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
        action_journal_path: Path,
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
            action_journal_path=action_journal_path,
            rollout_id=rollout_id,
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
            alfworld_data=self.alfworld_data,
        )
        journal_conversation, journal_gamefile = _load_action_journal(
            action_journal_path,
            task=task,
            rollout_id=rollout_id,
        )
        if journal_gamefile != env.actual_gamefile:
            raise RuntimeError(
                "environment-action journal names the wrong verified gamefile"
            )
        _validate_journal_conversation(conversation, journal_conversation)
        return row, conversation, env.actual_gamefile

    def run(
        self,
        task: dict[str, Any],
        skill_content: str,
        out_dir: str,
        rollout_id: str = "",
    ) -> EpisodeOutcome:
        """Run one episode and persist usage for this exact execution scope.

        Production usage comes from the episode-scoped provider observer; the
        injected fixture path retains the SkillOpt token-tracker fallback.
        """

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
                    failure_kind="protocol_failure",
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
                    failure_kind="protocol_failure",
                )
        failure: BaseException | None = None
        failure_kind = ""
        row: dict[str, Any] = {}
        conversation: list[dict[str, Any]] = []
        actual_gamefile = ""
        action_journal_path: Path | None = None
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
                    alfworld_data=self.alfworld_data,
                )
            else:
                action_journal_path = _action_journal_path(
                    out_dir,
                    task=task,
                    rollout_id=rollout_id,
                )
                row, conversation, actual_gamefile = self._run_upstream_episode(
                    task,
                    skill_content,
                    out_dir,
                    rollout_id,
                    action_journal_path,
                )
        except ProviderCallExhausted as exc:
            # This is our boundary outside pinned SkillOpt.  The sentinel must
            # bypass upstream model fallbacks, then become a durable
            # failure outcome before the batch stops.  Only the frozen
            # transient-code set is resumable infrastructure; permanent or
            # audit failures remain protocol failures.
            failure = exc
            failure_kind = (
                "infrastructure_failure"
                if exc.infrastructure_failure
                else "protocol_failure"
            )
        except Exception as exc:
            failure = exc
            failure_kind = (
                "infrastructure_failure"
                if isinstance(exc, (ConnectionError, TimeoutError))
                else "protocol_failure"
            )

        if failure is not None and action_journal_path is not None:
            try:
                journal_conversation, journal_gamefile = _load_action_journal(
                    action_journal_path,
                    task=task,
                    rollout_id=rollout_id,
                    missing_ok=True,
                )
                conversation = journal_conversation
                actual_gamefile = journal_gamefile
            except Exception as exc:
                failure = exc
                failure_kind = "protocol_failure"

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
                failure_kind = "protocol_failure"

        if failure is not None:
            return EpisodeOutcome(
                task=dict(task),
                skillopt_row={"id": str(task.get("id", "")), "hard": 0, "soft": 0.0},
                conversation=list(conversation),
                target_usage=target_usage,
                wall_time_ms=int((time.time() - started) * 1000),
                infrastructure_failure=True,
                infrastructure_error=_safe_error(failure),
                failure_kind=failure_kind or "protocol_failure",
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


def _canonical_gamefile(
    value: str | Path,
    *,
    alfworld_data: str | Path,
) -> str:
    """Resolve one gamefile solely under the configured ALFWorld authority."""

    data_root = Path(alfworld_data).expanduser().resolve(strict=True)
    if not data_root.is_dir():
        raise RuntimeError(f"ALFWORLD_DATA is not a directory: {data_root}")
    raw = os.path.expandvars(str(value)).strip()
    if not raw:
        raise RuntimeError("ALFWorld gamefile is empty")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        normalized = raw.replace("\\", "/")
        raw_parts = normalized.split("/")
        portable = PurePosixPath(normalized)
        if (
            portable.is_absolute()
            or not portable.parts
            or any(part in {"", ".", ".."} for part in raw_parts)
        ):
            raise RuntimeError(f"unsafe ALFWorld gamefile path: {raw}")
        candidate = data_root.joinpath(*portable.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"ALFWorld gamefile is not readable: {raw}") from exc
    try:
        resolved.relative_to(data_root)
    except ValueError as exc:
        raise RuntimeError(
            f"ALFWorld gamefile resolves outside ALFWORLD_DATA: {raw}"
        ) from exc
    if not resolved.is_file():
        raise RuntimeError(f"ALFWorld gamefile is not a regular file: {resolved}")
    return str(resolved)


def _safe_error(exc: BaseException) -> str:
    """Return useful failure context without persisting credentials."""

    message = str(exc)
    live_key = os.environ.get("MODEL_API_KEY", "").strip()
    if live_key:
        message = message.replace(live_key, "<redacted>")
    return f"{type(exc).__name__}: {message[:500]}"


def _action_journal_path(
    out_dir: str | Path,
    *,
    task: dict[str, Any],
    rollout_id: str,
) -> Path:
    identity = {
        "rollout_id": str(rollout_id),
        "task_id": str(task.get("task_id", task.get("id", ""))),
        "manifest_index": int(task.get("manifest_index", -1)),
        "gamefile": str(task.get("gamefile", "")),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return Path(out_dir) / "attempt_environment_actions" / f"{digest}.jsonl"


def _write_action_journal_header(
    path: Path,
    *,
    task: dict[str, Any],
    rollout_id: str,
    actual_gamefile: str,
) -> None:
    payload = {
        "schema_version": 1,
        "event": "action_journal_started",
        "rollout_id": str(rollout_id),
        "episode_task_id": str(task.get("task_id", task.get("id", ""))),
        "manifest_index": int(task.get("manifest_index", -1)),
        "gamefile": str(task.get("gamefile", "")),
        "actual_gamefile": str(actual_gamefile),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())


def _append_jsonl_durable(path: Path, payload: dict[str, Any]) -> None:
    if not path.is_file():
        raise RuntimeError(f"environment-action journal is missing: {path}")
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())


def _single_item(value: Any, *, label: str) -> Any:
    if isinstance(value, (str, bytes, dict)):
        raise RuntimeError(f"environment step {label} is not a one-item sequence")
    try:
        if len(value) != 1:
            raise RuntimeError(
                f"environment step {label} must contain exactly one item"
            )
        return value[0]
    except TypeError as exc:
        raise RuntimeError(
            f"environment step {label} is not a one-item sequence"
        ) from exc


def _action_journal_row(
    *,
    task: dict[str, Any],
    rollout_id: str,
    step_index: int,
    call_args: tuple[Any, ...],
    call_kwargs: dict[str, Any],
    result: Any,
) -> dict[str, Any]:
    if call_args:
        action_batch = call_args[0]
    else:
        action_batch = next(
            (
                call_kwargs[name]
                for name in ("text_actions", "actions", "action")
                if name in call_kwargs
            ),
            None,
        )
    action = _single_item(action_batch, label="actions")
    if not isinstance(action, str) or not action.strip():
        raise RuntimeError("environment step produced no auditable action")
    if not isinstance(result, tuple) or len(result) != 4:
        raise RuntimeError("environment step returned an invalid result tuple")
    observations, rewards, dones, _ = result
    if not isinstance(observations, dict):
        raise RuntimeError("environment step observations are not a mapping")
    anchors = observations.get("anchor")
    feedback = _single_item(anchors, label="anchor observations")
    reward = float(_single_item(rewards, label="rewards"))
    done = bool(_single_item(dones, label="dones"))
    task_id = str(task.get("task_id", task.get("id", "")))
    identity = {
        "rollout_id": str(rollout_id),
        "episode_task_id": task_id,
        "step_index": int(step_index),
    }
    return {
        "schema_version": 1,
        "event": "environment_action",
        "action_id": hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
        **identity,
        "manifest_index": int(task.get("manifest_index", -1)),
        "action": action.strip(),
        "env_feedback": str(feedback),
        "reward": reward,
        "done": done,
    }


def _load_action_journal(
    path: Path,
    *,
    task: dict[str, Any],
    rollout_id: str,
    missing_ok: bool = False,
) -> tuple[list[dict[str, Any]], str]:
    if not path.is_file():
        if missing_ok:
            return [], ""
        raise RuntimeError(f"environment-action journal is missing: {path}")
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"environment-action journal is unreadable: {path}") from exc
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError("environment-action journal has no valid header")
    task_id = str(task.get("task_id", task.get("id", "")))
    header = rows[0]
    if (
        header.get("schema_version") != 1
        or header.get("event") != "action_journal_started"
        or header.get("rollout_id") != str(rollout_id)
        or header.get("episode_task_id") != task_id
        or int(header.get("manifest_index", -2))
        != int(task.get("manifest_index", -1))
        or header.get("gamefile") != str(task.get("gamefile", ""))
        or not str(header.get("actual_gamefile", "")).strip()
    ):
        raise RuntimeError("environment-action journal header identity mismatch")
    conversation: list[dict[str, Any]] = []
    seen_action_ids: set[str] = set()
    for step_index, row in enumerate(rows[1:]):
        identity = {
            "rollout_id": str(rollout_id),
            "episode_task_id": task_id,
            "step_index": step_index,
        }
        expected_action_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        action = row.get("action")
        action_id = str(row.get("action_id", ""))
        if (
            row.get("schema_version") != 1
            or row.get("event") != "environment_action"
            or row.get("rollout_id") != str(rollout_id)
            or row.get("episode_task_id") != task_id
            or int(row.get("manifest_index", -2))
            != int(task.get("manifest_index", -1))
            or int(row.get("step_index", -1)) != step_index
            or action_id != expected_action_id
            or action_id in seen_action_ids
            or not isinstance(action, str)
            or not action.strip()
        ):
            raise RuntimeError("environment-action journal row identity mismatch")
        seen_action_ids.add(action_id)
        conversation.append({
            "step": step_index,
            "action": action,
            "env_feedback": str(row.get("env_feedback", "")),
            "reward": float(row.get("reward", 0.0)),
            "done": bool(row.get("done", False)),
        })
    return conversation, str(header["actual_gamefile"])


def _validate_journal_conversation(
    conversation: list[dict[str, Any]],
    journal: list[dict[str, Any]],
) -> None:
    if len(conversation) != len(journal):
        raise RuntimeError(
            "environment-action journal length differs from the conversation trace"
        )
    for index, (step, action) in enumerate(zip(conversation, journal, strict=True)):
        if (
            str(step.get("action", "")).strip().lower()
            != str(action.get("action", ""))
            or str(step.get("env_feedback", ""))
            != str(action.get("env_feedback", ""))
            or float(step.get("reward", 0.0)) != float(action.get("reward", 0.0))
            or bool(step.get("done", False)) != bool(action.get("done", False))
        ):
            raise RuntimeError(
                f"environment-action journal differs at step {index}"
            )


def _validate_episode_payload(
    *,
    task: dict[str, Any],
    row: dict[str, Any],
    conversation: Any,
    actual_gamefile: str,
    alfworld_data: str | Path,
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
    if not str(actual_gamefile).strip() or not str(row.get("gamefile", "")).strip():
        raise RuntimeError("SkillOpt rollout is missing verified gamefile evidence")
    expected_gamefile = _canonical_gamefile(
        actual_gamefile,
        alfworld_data=alfworld_data,
    )
    reported_gamefile = _canonical_gamefile(
        row["gamefile"],
        alfworld_data=alfworld_data,
    )
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
        action_journal_path: str | Path | None = None,
        rollout_id: str = "",
    ) -> None:
        self._environment = environment
        self._task = dict(task)
        self._alfworld_data = Path(alfworld_data).expanduser().resolve(strict=True)
        self._provider_observer = provider_observer
        self._action_journal_path = (
            Path(action_journal_path) if action_journal_path is not None else None
        )
        self._rollout_id = str(rollout_id)
        self._step_index = 0
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
        if self._action_journal_path is not None:
            _write_action_journal_header(
                self._action_journal_path,
                task=self._task,
                rollout_id=self._rollout_id,
                actual_gamefile=self.actual_gamefile,
            )
        return observations, infos

    def step(self, *args: Any, **kwargs: Any):
        if self._provider_observer is not None:
            self._provider_observer.raise_if_active_episode_failed()
        result = self._environment.step(*args, **kwargs)
        if self._action_journal_path is not None:
            row = _action_journal_row(
                task=self._task,
                rollout_id=self._rollout_id,
                step_index=self._step_index,
                call_args=args,
                call_kwargs=kwargs,
                result=result,
            )
            _append_jsonl_durable(self._action_journal_path, row)
            self._step_index += 1
        return result

    def close(self) -> None:
        close = getattr(self._environment, "close", None)
        if callable(close):
            close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._environment, name)
