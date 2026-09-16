"""Regression coverage for real zero-output usage and audited continuation."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from experiments.baselines.b3_skillopt.episode_runner import _usage_from_provider_events
from experiments.baselines.b3_skillopt.provider_observer import ProviderCallObserver
from experiments.baselines.b3_skillopt.worker import _provider_usage
from experiments.baselines.b3_skillopt.common_alfworld_adapter import _reconcile_episode_provider_usage
from experiments.baselines.b5_gepa import controller, recover_campaign as recovery
from experiments.baselines.tests.test_reasoning_budget import reply


@pytest.fixture
def empty_event(tmp_path, monkeypatch):
    import skillopt.model.openai_compatible_backend as backend
    monkeypatch.setattr(backend, "_get_client", lambda role: NS(chat=NS(completions=NS(
        create=lambda **kw: reply("", 0, 0, "stop")))))
    monkeypatch.setattr(backend.TARGET_CONFIG, "deployment", "deepseek-v4-flash")
    observer = ProviderCallObserver(output_path=tmp_path/"calls.jsonl", method="b5_gepa",
        phase="train", model="deepseek-v4-flash", reasoning_effort="high", run_id="fixture",
        run_seed=43, application_retry_limit=5)
    observer.install()
    try:
        result, _ = backend._chat_messages_impl([], 512, 5, "rollout", role="target")
        assert result == ""  # The executor's existing fallback remains authoritative.
        event = observer.events()[0]
        event["episode_task_id"] = "fixture_task"
        assert event["completion_tokens"] == 0 and event["total_tokens"] == 100
        yield event
    finally:
        observer.uninstall()


def phase_usage(event):
    wire = NS(method="b5_gepa", phase="train", run_id="fixture", run_seed=43,
              model=dict(model="deepseek-v4-flash", reasoning_effort="high"), campaign=None)
    return _provider_usage(NS(events=lambda:[event]), wire=wire, wall_time_ms=1)


def test_known_empty_target_is_charged_through_episode_cache_reconcile_and_phase(empty_event, tmp_path):
    e = empty_event
    usage = _usage_from_provider_events([e], require_all_succeeded=True)
    assert (usage.calls, usage.prompt_tokens, usage.completion_tokens) == (1, 100, 0)
    assert phase_usage(e).target == usage
    episode = NS(task_id="fixture_task", target_llm_calls=1, target_prompt_tokens=100,
        target_completion_tokens=0, target_reasoning_tokens=0, environment_actions=1,
        method_metrics={"target_reasoning_tokens_status":"reported"})
    _reconcile_episode_provider_usage([episode], [e], required=True, validate_persisted_reasoning=True)
    cache = ProviderCallObserver(output_path=tmp_path/"cache.jsonl", method="b5_gepa", phase="train",
        model="deepseek-v4-flash", reasoning_effort="high", run_id="fixture", run_seed=43)
    imported = cache.import_cached_events([e], rollout_id="cached", task_id="fixture_task")
    assert len(imported) == 1 and imported[0]["cache_reused"] is True
    assert _usage_from_provider_events(imported, require_all_succeeded=True) == usage


@pytest.mark.parametrize("change", [dict(completion_tokens=-1), dict(prompt_tokens=-1),
    dict(total_tokens=101), dict(prompt_tokens=0,total_tokens=0),
    dict(reasoning_tokens=1), dict(completion_tokens=None)])
def test_missing_negative_inconsistent_or_zero_total_usage_still_rejected(empty_event, change):
    event = {**empty_event, **change}
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        _usage_from_provider_events([event], require_all_succeeded=True)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        phase_usage(event)


def test_empty_optimizer_not_accepted_as_target(empty_event):
    with pytest.raises(ValueError, match="invalid token usage"):
        phase_usage({**empty_event, "role":"optimizer"})


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_failed_prefix_reconstructed_without_rewriting_old_evidence(empty_event, tmp_path):
    root = tmp_path/"old"
    manifest = dict(method="b5_gepa", phase="train", run_id="fixture", run_seed=43,
        started_at_unix=10, model_identity=dict(model="deepseek-v4-flash", reasoning_effort="high"))
    failure = dict(passed=False, failure_kind="protocol_failure", failed_at_unix=12,
        error="GEPA train worker failed: RuntimeError: episode provider evidence has invalid token usage")
    write(root/"run_manifest.json", manifest)
    write(root/"failure.json", failure)
    write(root/"train/evaluations/eval/process_0/provider_calls.jsonl", empty_event)
    before = recovery.digest_directory(root)
    usage, refs, ids = recovery.load_failed_usage(root)
    assert usage.target.calls == 1 and usage.target.completion_tokens == 0
    assert usage.wall_time_ms == 2000 and ids == [empty_event["call_id"]]
    assert len(refs) == 1 and recovery.digest_directory(root) == before
    failure["error"] = "an unrelated protocol failure"
    write(root/"failure.json", failure)
    with pytest.raises(ValueError, match="requires the evidenced"):
        recovery.load_failed_usage(root)


def test_resume_preserves_checkpoint_and_requires_matching_noncode_identity(tmp_path, monkeypatch):
    source, destination = tmp_path/"source", tmp_path/"new"
    identity = dict(controller_code_digest="new", model_identity_digest="same", config_digest="same")
    ctx = NS(identity=identity, external_commit="upstream", model_config=NS(to_wire=lambda:{"model":"same"}),
             campaign={"campaign_lock_digest":"same"}, run_seed=43)
    manifest = dict(method="b5_gepa", phase="train", run_seed=43, run_id="prior",
        identity={**identity,"controller_code_digest":"old"}, controller_git={"commit":"oldcommit"},
        external_commit="upstream", model_identity={"model":"same"}, campaign=ctx.campaign)
    write(source/"run_manifest.json", manifest)
    write(source/"failure.json", dict(passed=False, failure_kind="protocol_failure"))
    state = source/"train/gepa_state/gepa_state.bin"
    state.parent.mkdir(parents=True)
    state.write_bytes(b"immutable checkpoint fixture")
    monkeypatch.setattr(controller, "_controller_git_state", lambda:{"commit":"newcommit"})
    receipt = dict(source_run=str(source.resolve()), original_code_digest="old", original_commit="oldcommit")
    with pytest.raises(ValueError, match="infrastructure"):
        controller._prepare_resume(source_run=source, destination_run=destination, expected_ctx=ctx)
    result = controller._prepare_resume(source_run=source, destination_run=destination,
                                        expected_ctx=ctx, recovery=receipt)
    assert result["source_failure_kind"] == "protocol_failure"
    assert (destination/"train/gepa_state/gepa_state.bin").read_bytes() == state.read_bytes()
    ctx.identity = {**identity, "model_identity_digest":"changed"}
    with pytest.raises(ValueError, match="identity mismatch"):
        controller._prepare_resume(source_run=source, destination_run=tmp_path/"another",
                                   expected_ctx=ctx, recovery=receipt)


def test_evidence_tampering_rejected(tmp_path):
    path = tmp_path/"evidence.json"
    write(path, {"passed":True})
    ref = recovery.reference(path)
    write(path, {"passed":False})
    with pytest.raises(ValueError, match="changed"):
        recovery.verify_reference(ref)


def test_failed_recovery_cannot_publish_three_seed_completion(tmp_path, monkeypatch):
    output = tmp_path/"result"
    output.mkdir()
    monkeypatch.setattr(recovery, "_resource_summary", lambda _: {})
    monkeypatch.setattr(recovery, "_campaign_cost_summary", lambda _: {})
    monkeypatch.setattr(recovery.campaign, "_campaign_actual_usage", lambda _: {})
    monkeypatch.setattr(recovery, "build_campaign_report", lambda *a, **kw: pytest.fail("Incomplete report"))
    lanes = [dict(seed=s, passed=s!=43, effectiveness={}) for s in (42,43,44)]
    result = recovery.publish(output, lanes, {})
    assert result["passed"] is False and result["completed_seeds"] == [42,44]
    assert not (output/"completion.json").exists() and not (output/"paper_report.json").exists()


def test_recovery_dispatches_only_missing_lane_and_charges_failed_prefix(tmp_path, monkeypatch):
    from contextlib import nullcontext
    source, output = tmp_path/"old", tmp_path/"recovery"
    source.mkdir(); output.mkdir()
    failed = source/"seed_43/train_attempt_1"
    old_lanes = [dict(seed=s, passed=s!=43, train_root=f"train{s}", test_root=f"test{s}") for s in (42,43,44)]
    write(source/"campaign_report.json", dict(lanes=old_lanes))
    write(source/"campaign_lock.json", dict(manifests={r:{"path":r} for r in ("train","validation","test")},
        config_path="config", worker_python="python"))
    receipt = dict(seed=43, source_run=str(failed), reconstructed_usage={"path":str(output/"prefix.json")})
    monkeypatch.setattr(recovery, "_method_campaign_lease", lambda *a,**kw:nullcontext())
    monkeypatch.setattr(recovery, "validate_recovery", lambda *a:receipt)
    calls = []
    def run_lane(spec, **kwargs):
        calls.append(kwargs)
        return dict(seed=43, passed=True, train_root="newtrain", test_root="newtest")
    monkeypatch.setattr(recovery.campaign, "_run_lane", run_lane)
    monkeypatch.setattr(recovery.campaign, "_validate_train", lambda *a:NS())
    monkeypatch.setattr(recovery.campaign, "_validate_test", lambda *a,**kw:{})
    monkeypatch.setattr(recovery, "publish", lambda out, lanes, r:lanes)
    result = recovery.run(source, output, tmp_path/"smoke")
    assert [c["seed"] for c in calls] == [43]
    assert calls[0]["initial_resume_source"] == failed
    assert calls[0]["initial_usage_path"] == output/"prefix.json"
    assert result[0] == old_lanes[0] and result[2] == old_lanes[2]
