"""Fail-closed B3 rollout receipts and infrastructure-isolation tests."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("skillopt")

from experiments.baselines.b3_skillopt import provider_observer as provider_observer_module
from experiments.baselines.b3_skillopt.common_alfworld_adapter import (
    CommonALFWorldSkillOptAdapter,
    RolloutInfrastructureError,
    _reconcile_episode_provider_usage,
)
from experiments.baselines.b3_skillopt.episode_runner import (
    SkillOptTextEpisodeRunner,
    _ExactManifestEnvironment,
    _safe_error,
)
from experiments.baselines.b3_skillopt.provider_observer import ProviderCallObserver
from experiments.baselines.b3_skillopt.provider_observer import ProviderCallExhausted
from experiments.baselines.b3_skillopt.provider_observer import ObservedProviderFailure
from experiments.baselines.common.manifest import ManifestTask, TaskManifestSet
from experiments.baselines.common.schema import CommonEpisodeRecord



def _sdk_completion_response(
    content: str | None,
    completion_tokens: int,
    *,
    tool_calls: list[object] | None = None,
) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    usage = SimpleNamespace(
        prompt_tokens=3,
        completion_tokens=completion_tokens,
        total_tokens=3 + completion_tokens,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=usage,
    )


def _stub_observed_sdk_call(monkeypatch, backend, outcomes):
    """Install a one-boundary upstream call whose retry argument is observable."""

    pending = list(outcomes)
    upstream_retry_limits: list[int] = []
    sdk_requests: list[dict] = []

    class Completions:
        def create(self, **kwargs):
            sdk_requests.append(dict(kwargs))
            outcome = pending.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    class Client:
        chat = SimpleNamespace(completions=Completions())

    def original_call(
        messages,
        max_completion_tokens,
        retries,
        stage,
        *,
        role,
        tools=None,
        tool_choice=None,
        return_message=False,
        deployment=None,
        timeout=None,
    ):
        upstream_retry_limits.append(int(retries))
        response = backend._get_client(role).chat.completions.create(
            model="fixture-model",
        )
        message = response.choices[0].message
        result = message if return_message else message.content
        return result, backend.usage_from_openai_usage(response.usage)

    monkeypatch.setattr(backend, "_chat_messages_impl", original_call)
    monkeypatch.setattr(backend, "_get_client", lambda role: Client())
    monkeypatch.setattr(backend, "_asg_provider_observer", None, raising=False)
    return upstream_retry_limits, sdk_requests


def _manifest(tmp_path: Path, name: str, split: str, count: int) -> Path:
    tasks = tuple(
        ManifestTask(
            index=index,
            task_id=f"{name}_{index}",
            task_type="pick_and_place_simple",
            source_split=split,
            env_index=index,
            gamefile_rel=f"json_2.1.1/{split}/fixture_{index}/game.tw-pddl",
            gamefile_sha256=hashlib.sha256(f"game:{name}:{index}".encode()).hexdigest(),
            task_signature=hashlib.sha256(f"task:{name}:{index}".encode()).hexdigest(),
        )
        for index in range(count)
    )
    manifest = TaskManifestSet.create(
        manifest_id=name,
        benchmark="alfworld",
        source_split=split,
        seed=42,
        tasks=tasks,
    )
    return manifest.save(tmp_path / f"{name}.json")


def _successful_episode(task, skill_content, out_dir):
    conversation = [
        {
            "step": 0,
            "action": "look",
            "env_feedback": "room",
            "reward": 1.0,
            "done": True,
        }
    ]
    conversation_path = (
        Path(out_dir) / "predictions" / str(task["id"]) / "conversation.json"
    )
    conversation_path.parent.mkdir(parents=True, exist_ok=True)
    conversation_path.write_text(
        json.dumps(conversation, ensure_ascii=False), encoding="utf-8",
    )
    return {
        "id": str(task["id"]),
        "hard": 1,
        "soft": 1.0,
        "n_turns": 1,
        "fail_reason": "",
        "agent_ok": True,
        "task_type": str(task["task_type"]),
        "gamefile": str(task["gamefile"]),
        "task_description": "fixture",
    }, conversation


def _adapter(
    tmp_path: Path,
    *,
    episode_fn=_successful_episode,
    phase: str = "train",
    identity_extra: dict[str, str] | None = None,
    artifact_digest_override: str | None = None,
    workers: int = 1,
    max_api_workers: int = 1,
) -> CommonALFWorldSkillOptAdapter:
    train_path = _manifest(tmp_path, "train", "train", 2)
    validation_path = _manifest(tmp_path, "validation", "valid_seen", 1)
    train = TaskManifestSet.load(train_path)
    validation = TaskManifestSet.load(validation_path)
    identity = {
        "config_digest": "a" * 64,
        "train_manifest_digest": train.digest,
        "validation_manifest_digest": validation.digest,
    }
    identity.update(identity_extra or {})
    return CommonALFWorldSkillOptAdapter(
        train_manifest_path=train_path,
        validation_manifest_path=validation_path,
        alfworld_data="",
        max_steps=100,
        max_completion_tokens=1024,
        seed=42,
        phase=phase,
        run_id="run_fixture",
        identity=identity,
        artifact_digest_override=artifact_digest_override,
        workers=workers,
        max_api_workers=max_api_workers,
        episode_runner=SkillOptTextEpisodeRunner(
            max_actions=100,
            max_completion_tokens=1024,
            seed=42,
            episode_fn=episode_fn,
        ),
    )


def test_rollout_commits_identity_bound_receipt_last_and_exact_cache_reuses(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path)
    env = adapter.build_train_env(batch_size=2, seed=7)
    out = tmp_path / "rollout"
    skill = "# Full skill bytes\n"

    rows = adapter.rollout(env, skill, str(out))
    receipt = json.loads((out / "rollout_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "completed"
    assert receipt["request"]["run_id"] == "run_fixture"
    assert receipt["request"]["phase"] == "train"
    assert receipt["request"]["batch_seed"] == 7
    assert receipt["request"]["skill_sha256"] == hashlib.sha256(
        skill.encode("utf-8")
    ).hexdigest()
    assert [task["task_id"] for task in receipt["request"]["tasks"]] == [
        row["id"] for row in rows
    ]
    assert set(receipt["files"]) == {
        "results.jsonl",
        "common_episodes.jsonl",
        "common_environment_actions.jsonl",
    }
    assert receipt["counts"] == {
        "results": 2,
        "episodes": 2,
        "actions": 2,
        "provider_calls": 0,
    }

    cached = adapter.rollout(env, skill, str(out))
    assert cached == rows


def test_rollout_parallelizes_independent_episodes_but_commits_manifest_order(
    tmp_path: Path,
) -> None:
    lock = threading.Lock()
    active = 0
    peak = 0
    barrier = threading.Barrier(2)

    def concurrent_episode(task, skill_content, out_dir):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            barrier.wait(timeout=2)
            time.sleep(0.02 if int(task["manifest_index"]) == 0 else 0.005)
            return _successful_episode(task, skill_content, out_dir)
        finally:
            with lock:
                active -= 1

    adapter = _adapter(
        tmp_path,
        episode_fn=concurrent_episode,
        workers=2,
        max_api_workers=2,
    )
    env = adapter.build_train_env(batch_size=2, seed=7)
    rows = adapter.rollout(env, "# Skill\n", str(tmp_path / "parallel"))

    assert peak == 2
    assert [row["id"] for row in rows] == ["train_0", "train_1"]


def test_train_eval_uses_frozen_artifact_digest_but_keeps_full_skill_hash(
    tmp_path: Path,
) -> None:
    frozen_digest = "f" * 64
    adapter = _adapter(
        tmp_path,
        phase="train_eval",
        identity_extra={"frozen_artifact_digest": frozen_digest},
        artifact_digest_override=frozen_digest,
    )
    env = adapter.build_train_evaluation_env(seed=42)
    out = tmp_path / "train_eval"
    skill = "# Frozen best skill\n"

    adapter.rollout(env, skill, str(out))
    receipt = json.loads((out / "rollout_receipt.json").read_text(encoding="utf-8"))
    assert receipt["request"]["artifact_digest"] == frozen_digest
    assert receipt["request"]["skill_sha256"] == hashlib.sha256(
        skill.encode("utf-8")
    ).hexdigest()
    episodes = [
        json.loads(line)
        for line in (out / "common_episodes.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert all(row["artifact_digest_before"] == frozen_digest for row in episodes)
    assert all(row["artifact_digest_after"] == frozen_digest for row in episodes)
    assert all(
        row["method_metrics"]["skill_sha256"]
        == receipt["request"]["skill_sha256"]
        for row in episodes
    )


def test_non_train_eval_rejects_distinct_artifact_digest_override(
    tmp_path: Path,
) -> None:
    adapter = _adapter(
        tmp_path,
        artifact_digest_override="f" * 64,
    )
    env = adapter.build_train_env(batch_size=1, seed=7)
    with pytest.raises(ValueError, match="only distinct"):
        adapter.rollout(env, "# Skill bytes\n", str(tmp_path / "bad_artifact"))


def test_partial_or_stale_rollout_never_becomes_a_cache_hit(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    env = adapter.build_train_env(batch_size=2, seed=7)
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "results.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="partial or stale"):
        adapter.rollout(env, "# Skill\n", str(partial))

    completed = tmp_path / "completed"
    adapter.rollout(env, "# Skill A\n", str(completed))
    with pytest.raises(RuntimeError, match="identity does not match"):
        adapter.rollout(env, "# Skill B\n", str(completed))

    # A fresh adapter session must not silently resume bytes left by an older
    # process, even when run/config/manifest/skill identities are unchanged.
    restarted = _adapter(tmp_path)
    restarted_env = restarted.build_train_env(batch_size=2, seed=7)
    with pytest.raises(RuntimeError, match="identity does not match"):
        restarted.rollout(restarted_env, "# Skill A\n", str(completed))


def test_tampered_completed_sidecar_fails_hash_validation(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    env = adapter.build_train_env(batch_size=2, seed=7)
    out = tmp_path / "rollout"
    adapter.rollout(env, "# Skill\n", str(out))
    with (out / "common_episodes.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        adapter.rollout(env, "# Skill\n", str(out))


def test_infrastructure_failure_is_persisted_then_raised_not_returned(
    tmp_path: Path,
    monkeypatch,
) -> None:
    secret = "fixture-secret-123"
    monkeypatch.setenv("MODEL_API_KEY", secret)

    def fail(task, skill_content, out_dir):
        raise ConnectionError(f"provider rejected {secret}")

    adapter = _adapter(tmp_path, episode_fn=fail)
    env = adapter.build_train_env(batch_size=2, seed=7)
    out = tmp_path / "failed"
    with pytest.raises(RolloutInfrastructureError, match="aborted on task"):
        adapter.rollout(env, "# Skill\n", str(out))

    assert not (out / "results.jsonl").exists()
    assert not (out / "rollout_receipt.json").exists()
    failure_path = out / "rollout_failure.json"
    assert failure_path.is_file()
    content = failure_path.read_text(encoding="utf-8")
    assert secret not in content
    failure = json.loads(content)
    assert failure["status"] == "infrastructure_failure"
    assert failure["failed_episode"]["infrastructure_failure"] is True
    with pytest.raises(RuntimeError, match="infrastructure-failed"):
        adapter.rollout(env, "# Skill\n", str(out))


def test_uncaught_runner_exception_is_also_persisted_before_batch_abort(
    tmp_path: Path,
    monkeypatch,
) -> None:
    adapter = _adapter(tmp_path)
    env = adapter.build_train_env(batch_size=1, seed=7)

    def uncaught(*args, **kwargs):
        raise ImportError("fixture runtime dependency vanished")

    monkeypatch.setattr(adapter._episode_runner, "run", uncaught)
    out = tmp_path / "uncaught"
    with pytest.raises(RolloutInfrastructureError, match="aborted on task"):
        adapter.rollout(env, "# Skill\n", str(out))
    failure = json.loads((out / "rollout_failure.json").read_text(encoding="utf-8"))
    assert failure["failed_episode"]["infrastructure_failure"] is True
    assert "ImportError" in failure["failed_episode"]["infrastructure_error"]
    assert not (out / "results.jsonl").exists()
    assert not (out / "rollout_receipt.json").exists()


def test_upstream_simple_placement_taxonomy_alias_is_normalized(
    tmp_path: Path,
) -> None:
    def upstream_label(task, skill_content, out_dir):
        row, conversation = _successful_episode(task, skill_content, out_dir)
        row["task_type"] = "pick_and_place"
        return row, conversation

    adapter = _adapter(tmp_path, episode_fn=upstream_label)
    env = adapter.build_train_env(batch_size=1, seed=7)
    rows = adapter.rollout(env, "# Skill\n", str(tmp_path / "aliased"))
    assert rows[0]["task_type"] == "pick_and_place_simple"


def test_conversation_action_count_mismatch_aborts_batch(tmp_path: Path) -> None:
    def mismatch(task, skill_content, out_dir):
        row, conversation = _successful_episode(task, skill_content, out_dir)
        row["n_turns"] = 2
        return row, conversation

    adapter = _adapter(tmp_path, episode_fn=mismatch)
    env = adapter.build_train_env(batch_size=1, seed=7)
    out = tmp_path / "mismatch"
    with pytest.raises(RolloutInfrastructureError, match="aborted on task"):
        adapter.rollout(env, "# Skill\n", str(out))
    failure = json.loads((out / "rollout_failure.json").read_text(encoding="utf-8"))
    assert "conversation length" in failure["failed_episode"]["infrastructure_error"]


def test_unknown_split_and_manifest_identity_drift_fail_closed(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    with pytest.raises(ValueError, match="unsupported split"):
        adapter.build_eval_env(env_num=1, split="typo_split", seed=42)

    item = dict(adapter.dataloader.train_items[0])
    item["gamefile_sha256"] = "0" * 64
    env = type("Batch", (), {"tasks": [item], "seed": 42})()
    with pytest.raises(ValueError, match="identity differs from manifest"):
        adapter.rollout(env, "# Skill\n", str(tmp_path / "bad_identity"))


def test_exact_environment_uses_shared_reset_gamefile_and_hash_authority(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "json_2.1.1" / "train" / "fixture" / "game.tw-pddl"
    expected.parent.mkdir(parents=True)
    expected.write_text("expected", encoding="utf-8")
    wrong = tmp_path / "json_2.1.1" / "train" / "wrong" / "game.tw-pddl"
    wrong.parent.mkdir(parents=True)
    wrong.write_text("wrong", encoding="utf-8")
    task = {
        "id": "task_0",
        "task_id": "task_0",
        "task_type": "pick_and_place_simple",
        "source_split": "train",
        "env_index": 0,
        "manifest_index": 0,
        "gamefile": expected.relative_to(tmp_path).as_posix(),
        "gamefile_sha256": hashlib.sha256(expected.read_bytes()).hexdigest(),
        "task_signature": "f" * 64,
    }

    class FakeEnvironment:
        def __init__(self, gamefile: Path) -> None:
            self.gamefile = gamefile

        def reset(self, *args, **kwargs):
            return {"text": ["room"]}, [{"extra.gamefile": str(self.gamefile)}]

    wrapped = _ExactManifestEnvironment(
        FakeEnvironment(expected),
        task=task,
        alfworld_data=tmp_path,
        provider_observer=None,
    )
    wrapped.reset({})
    assert wrapped.actual_gamefile == str(expected.resolve())

    wrapped = _ExactManifestEnvironment(
        FakeEnvironment(wrong),
        task=task,
        alfworld_data=tmp_path,
        provider_observer=None,
    )
    with pytest.raises(ValueError, match="wrong gamefile"):
        wrapped.reset({})


def test_provider_observer_records_rollout_identity_and_cursor(tmp_path: Path) -> None:
    observer = ProviderCallObserver(
        output_path=tmp_path / "provider_calls.jsonl",
        method="b3_skillopt",
        phase="train",
        model="fixture-model",
        reasoning_effort="high",
        run_id="run_fixture",
    )
    cursor = observer.event_cursor()
    with observer.episode("task_1", rollout_id="rollout_1"):
        observer._record(
            call_id="call_1",
            role="target",
            stage="rollout",
            status="succeeded",
            prompt_tokens=2,
            completion_tokens=1,
            total_tokens=3,
        )
    events = observer.events_since(cursor, rollout_id="rollout_1")
    assert len(events) == 1
    assert events[0]["episode_task_id"] == "task_1"
    assert events[0]["rollout_id"] == "rollout_1"
    assert events[0]["reasoning_effort"] == "high"
    persisted = json.loads(
        (tmp_path / "provider_calls.jsonl").read_text(encoding="utf-8")
    )
    assert persisted == events[0]

    # Receipts/cache readers must use the fsynced sidecar, not silently trust
    # an in-memory copy after durable evidence has changed.
    with (tmp_path / "provider_calls.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(RuntimeError, match="differs from in-memory"):
        observer.events()


def test_provider_observer_injects_high_into_actual_sdk_boundary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import skillopt.model.openai_compatible_backend as backend

    captured: dict = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _sdk_completion_response("ok", 1)

    class Chat:
        completions = Completions()

    class Client:
        chat = Chat()

    monkeypatch.setattr(backend, "_get_client", lambda role: Client())
    monkeypatch.setattr(backend, "_asg_provider_observer", None, raising=False)
    observer = ProviderCallObserver(
        output_path=tmp_path / "provider_calls.jsonl",
        method="b3_skillopt",
        phase="train",
        model="deepseek-v4-flash",
        reasoning_effort="high",
        run_id="run_fixture",
    )
    observer.install()
    try:
        backend._get_client("target").chat.completions.create(
            model="deepseek-v4-flash"
        )
    finally:
        observer.uninstall()
    assert captured == {
        "model": "deepseek-v4-flash",
        "reasoning_effort": "high",
    }


def test_provider_observer_disables_sdk_retries_on_the_actual_client(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import skillopt.model.openai_compatible_backend as backend

    configured_retries: list[int] = []
    client_lookups: list[str] = []

    class Completions:
        def create(self, **kwargs):
            return _sdk_completion_response("ok", 1)

    class Client:
        def __init__(self, max_retries: int) -> None:
            self.max_retries = max_retries
            self.chat = SimpleNamespace(completions=Completions())

        def with_options(self, *, max_retries: int):
            configured_retries.append(max_retries)
            return Client(max_retries)

    def get_client(role: str):
        client_lookups.append(role)
        return Client(2)

    monkeypatch.setattr(backend, "_get_client", get_client)
    monkeypatch.setattr(backend, "_asg_provider_observer", None, raising=False)
    observer = ProviderCallObserver(
        output_path=tmp_path / "provider_calls.jsonl",
        method="b3_skillopt",
        phase="train",
        model="deepseek-v4-flash",
        reasoning_effort="high",
        run_id="run_fixture",
        expected_sdk_max_retries=0,
    )
    observer.install()
    try:
        first = backend._get_client("target")
        second = backend._get_client("target")
        assert first.max_retries == second.max_retries == 0
        first.chat.completions.create(model="deepseek-v4-flash")
    finally:
        observer.uninstall()

    assert configured_retries == [0]
    assert client_lookups == ["target"]


def test_provider_observer_retries_response_without_usage_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import skillopt.model.openai_compatible_backend as backend

    responses = [
        *[
            _sdk_completion_response("non-empty", 0)
            for _ in range(4)
        ],
        _sdk_completion_response("valid response", 2),
    ]
    attempts: list[dict] = []

    class Completions:
        def create(self, **kwargs):
            attempts.append(dict(kwargs))
            return responses.pop(0)

    class Client:
        chat = SimpleNamespace(completions=Completions())

    monkeypatch.setattr(backend, "_get_client", lambda role: Client())
    monkeypatch.setattr(backend.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(backend, "_asg_provider_observer", None, raising=False)
    observer = ProviderCallObserver(
        output_path=tmp_path / "provider_calls.jsonl",
        method="b3_skillopt",
        phase="train",
        model="deepseek-v4-flash",
        reasoning_effort="high",
        run_id="run_fixture",
    )
    observer.install()
    try:
        with observer.episode("task_1", rollout_id="rollout_1"):
            text, usage = backend._chat_messages_impl(
                [{"role": "user", "content": "question"}],
                32,
                5,
                "rollout",
                role="target",
            )
    finally:
        observer.uninstall()

    assert text == "valid response"
    assert usage["completion_tokens"] == 2
    assert len(attempts) == 5
    assert all(item["reasoning_effort"] == "high" for item in attempts)
    events = observer.events()
    assert len(events) == 1
    assert events[0]["status"] == "succeeded"
    assert events[0]["completion_tokens"] == 2


def test_provider_observer_treats_empty_model_content_as_task_behavior(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import skillopt.model.openai_compatible_backend as backend

    retry_limits, sdk_requests = _stub_observed_sdk_call(
        monkeypatch,
        backend,
        [_sdk_completion_response("", 1)],
    )
    observer = ProviderCallObserver(
        output_path=tmp_path / "provider_calls.jsonl",
        method="b3_skillopt",
        phase="train",
        model="fixture-model",
        reasoning_effort="high",
        run_id="run_fixture",
        application_retry_limit=5,
        retry_delays_seconds=[0, 0, 0, 0],
    )
    observer.install()
    try:
        text, usage = backend._chat_messages_impl(
            [], 32, 5, "rollout", role="target",
        )
    finally:
        observer.uninstall()

    assert text == ""
    assert usage["completion_tokens"] == 1
    assert retry_limits == [1]
    assert len(sdk_requests) == 1
    event = observer.events()[0]
    assert event["status"] == "succeeded"
    assert event["application_attempts"] == 1
    assert event["failure_code_counts"] == {}


def test_provider_observer_explicitly_recovers_two_transient_timeouts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import skillopt.model.openai_compatible_backend as backend

    retry_limits, sdk_requests = _stub_observed_sdk_call(
        monkeypatch,
        backend,
        [
            TimeoutError("first fixture timeout"),
            TimeoutError("second fixture timeout"),
            _sdk_completion_response("recovered", 2),
        ],
    )
    observer = ProviderCallObserver(
        output_path=tmp_path / "provider_calls.jsonl",
        method="b3_skillopt",
        phase="train",
        model="fixture-model",
        reasoning_effort="high",
        run_id="run_fixture",
        application_retry_limit=5,
        retry_delays_seconds=[0, 0, 0, 0],
        expected_sdk_max_retries=None,
    )
    observer.install()
    try:
        with observer.episode("task_1", rollout_id="rollout_1"):
            text, usage = backend._chat_messages_impl(
                [], 32, 3, "rollout", role="target",
            )
            observer.raise_if_active_episode_failed()
    finally:
        observer.uninstall()

    assert text == "recovered"
    assert usage["completion_tokens"] == 2
    assert retry_limits == [1, 1, 1]
    assert len(sdk_requests) == 3
    event = observer.events()[0]
    assert event["status"] == "succeeded"
    assert event["application_attempts"] == 3
    assert event["sdk_boundary_attempts"] == 3
    assert event["requested_retry_limit"] == 3
    assert event["retry_limit"] == 3
    assert event["failure_code_counts"] == {"timeout": 2}
    assert event["last_failure_code"] == "timeout"
    assert event["recovered"] is True


def test_provider_observer_does_not_retry_http_401(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import skillopt.model.openai_compatible_backend as backend

    class UnauthorizedError(RuntimeError):
        status_code = 401
        request_id = "req_fixture_401"

    retry_limits, sdk_requests = _stub_observed_sdk_call(
        monkeypatch,
        backend,
        [UnauthorizedError("secret response body must not be persisted")],
    )
    observer = ProviderCallObserver(
        output_path=tmp_path / "provider_calls.jsonl",
        method="b3_skillopt",
        phase="train",
        model="fixture-model",
        reasoning_effort="high",
        run_id="run_fixture",
        application_retry_limit=5,
        retry_delays_seconds=[0, 0, 0, 0],
        expected_sdk_max_retries=None,
    )
    observer.install()
    try:
        with observer.episode("task_1", rollout_id="rollout_1"):
            with pytest.raises(ProviderCallExhausted, match="authentication"):
                backend._chat_messages_impl(
                    [], 32, 5, "rollout", role="target",
                )
            with pytest.raises(ObservedProviderFailure, match="authentication"):
                observer.raise_if_active_episode_failed()
    finally:
        observer.uninstall()

    assert retry_limits == [1]
    assert len(sdk_requests) == 1
    event = observer.events()[0]
    assert event["status"] == "failed"
    assert event["application_attempts"] == 1
    assert event["sdk_boundary_attempts"] == 1
    assert event["requested_retry_limit"] == 5
    assert event["retry_limit"] == 5
    assert event["failure_code_counts"] == {"authentication": 1}
    assert event["last_failure_code"] == "authentication"
    assert event["provider_request_ids"] == ["req_fixture_401"]
    assert event["recovered"] is False


def test_provider_observer_exhausts_five_transient_timeouts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import skillopt.model.openai_compatible_backend as backend

    retry_limits, sdk_requests = _stub_observed_sdk_call(
        monkeypatch,
        backend,
        [TimeoutError("fixture timeout") for _ in range(5)],
    )
    observer = ProviderCallObserver(
        output_path=tmp_path / "provider_calls.jsonl",
        method="b3_skillopt",
        phase="train",
        model="fixture-model",
        reasoning_effort="high",
        run_id="run_fixture",
        application_retry_limit=5,
        retry_delays_seconds=[0, 0, 0, 0],
        expected_sdk_max_retries=None,
    )
    observer.install()
    try:
        with observer.episode("task_1", rollout_id="rollout_1"):
            with pytest.raises(ProviderCallExhausted, match="timeout"):
                backend._chat_messages_impl(
                    [], 32, 5, "rollout", role="target",
                )
            with pytest.raises(ObservedProviderFailure, match="attempts=5"):
                observer.raise_if_active_episode_failed()
    finally:
        observer.uninstall()

    assert retry_limits == [1] * 5
    assert len(sdk_requests) == 5
    event = observer.events()[0]
    assert event["status"] == "failed"
    assert event["application_attempts"] == 5
    assert event["sdk_boundary_attempts"] == 5
    assert event["requested_retry_limit"] == 5
    assert event["retry_limit"] == 5
    assert event["failure_code_counts"] == {"timeout": 5}
    assert event["last_failure_code"] == "timeout"
    assert event["recovered"] is False


def test_provider_observer_uses_frozen_retry_schedule_after_upstream_delay(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import skillopt.model.openai_compatible_backend as backend

    _stub_observed_sdk_call(
        monkeypatch,
        backend,
        [TimeoutError("fixture timeout") for _ in range(5)],
    )
    sleeps: list[float] = []
    monkeypatch.setattr(
        provider_observer_module.time,
        "sleep",
        lambda seconds: sleeps.append(float(seconds)),
    )
    observer = ProviderCallObserver(
        output_path=tmp_path / "provider_calls.jsonl",
        method="b3_skillopt",
        phase="train",
        model="fixture-model",
        reasoning_effort="high",
        run_id="run_fixture",
        application_retry_limit=5,
        retry_delays_seconds=[2, 5, 10, 20],
    )
    observer.install()
    try:
        with pytest.raises(ProviderCallExhausted):
            backend._chat_messages_impl([], 32, 5, "rollout", role="target")
    finally:
        observer.uninstall()

    # Pinned upstream already sleeps one second after each failed boundary;
    # the adapter supplies only the remainder of each formal interval.
    assert sleeps == [1.0, 4.0, 9.0, 19.0]
    assert observer.events()[0]["retry_backoff_ms"] == 37_000


def test_provider_exhaustion_cannot_be_swallowed_by_upstream_exception_fallback() -> None:
    assert not issubclass(ProviderCallExhausted, Exception)
    with pytest.raises(ProviderCallExhausted):
        try:
            raise ProviderCallExhausted("fixture")
        except Exception:  # pragma: no cover - must never catch the sentinel
            pytest.fail("infrastructure abort was swallowed as model behavior")


def test_owned_episode_boundary_converts_provider_abort_to_infrastructure_outcome() -> None:
    def aborting_episode(*args, **kwargs):
        raise ProviderCallExhausted(
            "provider call failed (failure_code=timeout, attempts=5)"
        )

    runner = SkillOptTextEpisodeRunner(episode_fn=aborting_episode)
    outcome = runner.run(
        {"id": "task_1", "gamefile": "fixture/game.tw-pddl"},
        "skill",
        ".",
        rollout_id="rollout_fixture",
    )
    assert outcome.infrastructure_failure is True
    assert outcome.skillopt_row == {"id": "task_1", "hard": 0, "soft": 0.0}
    assert "ProviderCallExhausted" in outcome.infrastructure_error
    assert "failure_code=timeout" in outcome.infrastructure_error


def test_provider_observer_propagates_distinct_episode_identity_through_nested_pools(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import skillopt.model.openai_compatible_backend as backend

    def successful_call(*args, **kwargs):
        return "ok", {
            "prompt_tokens": 2,
            "completion_tokens": 1,
            "total_tokens": 3,
            "reasoning_tokens": 0,
            "reasoning_tokens_status": "reported",
        }

    monkeypatch.setattr(backend, "_chat_messages_impl", successful_call)
    monkeypatch.setattr(
        backend,
        "usage_from_openai_usage",
        lambda raw: dict(raw),
    )
    monkeypatch.setattr(backend, "_asg_provider_observer", None, raising=False)
    observer = ProviderCallObserver(
        output_path=tmp_path / "provider_calls.jsonl",
        method="b3_skillopt",
        phase="train",
        model="fixture-model",
        reasoning_effort="high",
        run_id="run_fixture",
    )
    overlap = threading.Barrier(2)
    observer.install()
    try:
        def run_episode(task_id: str) -> None:
            with observer.episode(task_id, rollout_id="rollout_parallel"):
                overlap.wait(timeout=2)
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as inner:
                    result, _ = inner.submit(
                        backend._chat_messages_impl,
                        [], 32, 1, "rollout",
                        role="target",
                    ).result(timeout=2)
                assert result == "ok"
                observer.raise_if_active_episode_failed()

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as outer:
            futures = [outer.submit(run_episode, task_id) for task_id in ("a", "b")]
            for future in futures:
                future.result(timeout=3)
    finally:
        observer.uninstall()

    events = observer.events()
    assert len(events) == 2
    assert {event["episode_task_id"] for event in events} == {"a", "b"}
    assert {event["rollout_id"] for event in events} == {"rollout_parallel"}


def test_episode_usage_reconciles_provider_events_and_retains_reasoning() -> None:
    episode = CommonEpisodeRecord(
        method="b3_skillopt",
        phase="train",
        run_seed=42,
        task_id="task_1",
        task_type="pick_and_place_simple",
        manifest_index=0,
        gamefile="fixture/game.tw-pddl",
        gamefile_hash="a" * 64,
        official_success=True,
        environment_actions=2,
        command_turns=2,
        target_llm_calls=2,
        target_prompt_tokens=11,
        target_completion_tokens=9,
    )
    events = [
        {
            "status": "succeeded",
            "role": "target",
            "episode_task_id": "task_1",
            "prompt_tokens": 5,
            "completion_tokens": 4,
            "total_tokens": 9,
            "reasoning_tokens": 1,
            "reasoning_tokens_status": "reported",
        },
        {
            "status": "succeeded",
            "role": "target",
            "episode_task_id": "task_1",
            "prompt_tokens": 6,
            "completion_tokens": 5,
            "total_tokens": 11,
            "reasoning_tokens": 2,
            "reasoning_tokens_status": "reported",
        },
    ]
    _reconcile_episode_provider_usage(
        [episode], events, required=True, validate_persisted_reasoning=False,
    )
    assert episode.target_reasoning_tokens == 3
    assert episode.method_metrics["target_reasoning_tokens_status"] == "reported"

    _reconcile_episode_provider_usage(
        [episode], events, required=True, validate_persisted_reasoning=True,
    )
    episode.target_prompt_tokens += 1
    with pytest.raises(RuntimeError, match="does not reconcile"):
        _reconcile_episode_provider_usage(
            [episode], events, required=True, validate_persisted_reasoning=True,
        )


def test_provider_observer_marks_target_failure_before_upstream_can_swallow_it(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import skillopt.model.openai_compatible_backend as backend

    def failed_call(*args, **kwargs):
        raise TimeoutError("fixture timeout")

    monkeypatch.setattr(backend, "_chat_messages_impl", failed_call)
    monkeypatch.setattr(
        backend,
        "usage_from_openai_usage",
        lambda raw: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )
    monkeypatch.setattr(backend, "_asg_provider_observer", None, raising=False)
    observer = ProviderCallObserver(
        output_path=tmp_path / "provider_calls.jsonl",
        method="b3_skillopt",
        phase="train",
        model="fixture-model",
        reasoning_effort="high",
        run_id="run_fixture",
    )
    observer.install()
    try:
        with observer.episode("task_1", rollout_id="rollout_1"):
            with pytest.raises(ProviderCallExhausted, match="timeout"):
                backend._chat_messages_impl(
                    [], 32, 1, "rollout", role="target",
                )
            with pytest.raises(ObservedProviderFailure, match="task_1"):
                observer.raise_if_active_episode_failed()
    finally:
        observer.uninstall()
    events = observer.events()
    assert len(events) == 1
    assert events[0]["status"] == "failed"
    assert events[0]["error_type"] == "ProviderCallExhausted"


def test_safe_error_with_empty_api_key_does_not_corrupt_message(
    monkeypatch,
) -> None:
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    assert _safe_error(RuntimeError("plain failure")) == "RuntimeError: plain failure"
