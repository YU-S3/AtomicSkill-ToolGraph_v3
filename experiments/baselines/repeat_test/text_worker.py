"""Fresh-process text evaluation on copied, unchanged historical frozen assets."""
import argparse
from dataclasses import asdict
import hashlib
import sys
import time
from pathlib import Path

from experiments.baselines.b0_dynamic.driver import task_example, file_hash, usage_totals, completed_outcome
from experiments.baselines.b4_embodiskill.state import read_json, read_jsonl, write_json
from experiments.baselines.common.artifact_digest import digest_directory
from experiments.baselines.common.freeze import FrozenArtifact, assert_frozen_unchanged
from experiments.baselines.common.manifest import ManifestTask
from experiments.baselines.common.schema import CommonEpisodeRecord
from experiments.baselines.common.source_identity import sanitize_error_text
from .authority import SKILL_FILES


def skill_from_frozen(frozen, authority):
    if frozen.method_id != authority["method"] or frozen.digest != authority["frozen_digest"]:
        raise ValueError("Repeat frozen identity differs from original seed42")
    assert_frozen_unchanged(frozen)
    name = SKILL_FILES[authority["method"]]
    if name is None:
        if read_json(frozen.root/"empty_descriptor.json").get("persistent_skill") is not None:
            raise ValueError("B0 cannot gain a skill")
        return None
    text = (frozen.root/name).read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("Frozen skill is empty")
    return text


def run_episode_job(job):
    # Spawn creates fresh model/environment state for EVERY task (including
    # after resume); posthoc facts are never fed to the text executor.
    METHOD = job["authority"]["method"]
    root = Path(job["episode_dir"])
    root.mkdir(parents=True, exist_ok=True)
    entry = ManifestTask(**job["task"])
    example = task_example(entry)
    frozen = FrozenArtifact.load(job["frozen"])
    assert_frozen_unchanged(frozen)
    try:
        skill_text = skill_from_frozen(frozen, job["authority"])
        outcome = completed_outcome(root, example)
        if outcome is None:
            sys.path.insert(0, job["source"])
            import skillopt
            Path(skillopt.__file__).resolve().relative_to(Path(job["source"]).resolve())
            from experiments.baselines.b3_skillopt.worker import _configure_model
            from experiments.baselines.b3_skillopt.provider_observer import install_provider_observer, uninstall_provider_observer
            from experiments.baselines.common.model_config import ModelConfig
            from experiments.baselines.common.provider_gate import CampaignProviderGate
            from experiments.baselines.common.text_skill_executor import TextSkillALFWorldExecutor
            model = ModelConfig.from_mapping(job["authority"]["model"])
            _configure_model(model, sdk_max_retries=0)
            attempts = root / "attempts"
            attempts.mkdir(exist_ok=True)
            attempt = attempts / f"attempt_{len(list(attempts.iterdir()))+1:04d}"
            attempt.mkdir()
            gate = CampaignProviderGate(gate_dir=Path(job["gate"]), campaign_id=job["campaign_id"],
                                        max_inflight=job["cap"])
            observer = install_provider_observer(output_path=attempt / "provider_calls.jsonl",
                method=METHOD, phase=job["phase"], model=model.model, reasoning_effort=model.reasoning_effort,
                run_id=job["campaign_id"], run_seed=job["seed"], campaign_gate=gate,
                application_retry_limit=job["authority"]["config"]["provider_transport"]["application_retry_limit"],
                retry_delays_seconds=job["authority"]["config"]["provider_transport"]["retry_delays_seconds"],
                deterministic_jitter_ratio=job["authority"]["config"]["provider_transport"].get("deterministic_jitter_ratio",.10), expected_sdk_max_retries=0)
            try:
                runner = TextSkillALFWorldExecutor(max_actions=100,
                    max_completion_tokens=job["authority"]["method_output_token_hint"],
                    seed=job["seed"], alfworld_data=job["alfworld_data"])
                outcome = asdict(runner.run_episode(task=example, skill_text=skill_text, run_seed=job["seed"],
                    phase=job["phase"], output_dir=attempt, rollout_id=attempt.name))
                observer.events()
                write_json(attempt / "outcome.json", outcome)
            finally:
                uninstall_provider_observer(observer)
            if outcome["infrastructure_failure"]:
                failure = dict(passed=False, failure_kind=outcome["failure_kind"],
                               error=outcome["infrastructure_error"], task_id=entry.task_id)
                write_json(attempt / "rollout_failure.json", failure)
                return failure
            # Commit before replay: a replay-only failure never repeats API work.
            write_json(root / "episode_checkpoint.json", dict(attempt=str(attempt.relative_to(root)),
                       digest=digest_directory(attempt)))
        events = [e for path in sorted((root / "attempts").glob("*/provider_calls.jsonl"))
                  for e in read_jsonl(path)]
        usage = usage_totals(events)
        if usage["logical_calls"] <= 0:
            raise ValueError("Missing provider evidence")
        actions = [step["action"] for step in outcome["conversation"]]
        if len(actions) > 100:
            raise ValueError("Action ceiling exceeded")
        official = bool(outcome["skillopt_row"]["hard"])
        from experiments.baselines.common.task_authority import StrictTaskEvaluator
        replay_started = time.monotonic()
        strict = StrictTaskEvaluator(job["alfworld_data"]).evaluate(entry, actions, official_success=official)
        if strict.replayed_terminal_won != official:
            raise ValueError("Saved won and replay won differ")
        record = CommonEpisodeRecord(method=METHOD, phase=job["phase"], run_seed=job["seed"],
            task_id=entry.task_id, task_type=entry.task_type, manifest_index=entry.index,
            gamefile=entry.gamefile_rel, gamefile_hash=entry.gamefile_sha256,
            official_success=official, contract_consistency=strict.task_contract_success,
            common_strict_success=strict.strict_success, environment_actions=len(actions),
            invalid_actions=strict.invalid_actions, command_turns=len(actions),
            timeout=not official and len(actions)==100,
            termination_reason="won" if official else "action_budget" if len(actions)==100 else "done",
            target_llm_calls=usage["logical_calls"],
            target_prompt_tokens=usage["prompt_tokens_known_subtotal"],
            target_completion_tokens=usage["completion_tokens_known_subtotal"],
            target_reasoning_tokens=usage["reasoning_tokens_known_subtotal"],
            target_visible_completion_tokens=usage["visible_completion_tokens"],
            evolution_visible_completion_tokens=0,
            wall_time_ms=sum(read_json(p)["wall_time_ms"] for p in (root/"attempts").glob("*/outcome.json")),
            artifact_digest_before=frozen.digest, artifact_digest_after=frozen.digest,
            method_metrics=dict(skill_text_sha256=hashlib.sha256(skill_text.encode()).hexdigest() if skill_text is not None else None,
                repeat_id=job["repeat"], frozen_policy=job["authority"]["frozen_policy"],
                persistent_experience=False, usage={"target":usage},
                observed_gamefile=outcome["actual_gamefile"],
                model_action_parse_failures=sum(bool(e.get("model_action_parse_failure")) for e in events),
                all_attempt_environment_actions=sum(e.get("event")=="environment_action"
                    for p in (root/"attempts").glob("*/attempt_environment_actions/*.jsonl") for e in read_jsonl(p)),
                replay_wall_time_ms=int((time.monotonic()-replay_started)*1000)))
        assert_frozen_unchanged(frozen)
        write_json(root / "record.json", record.to_dict())
        write_json(root / "completion.json", dict(passed=True,
            evidence_digest=digest_directory(root / "attempts"), record_hash=file_hash(root/"record.json")))
        return dict(passed=True, seed=job["seed"], task_id=entry.task_id,
                    official_success=official, episode_dir=str(root))
    except Exception as exc:
        failure = dict(passed=False, task_id=entry.task_id,
            failure_kind=getattr(exc, "failure_kind", getattr(exc, "code", "protocol_failure")),
            error_type=type(exc).__name__, error=sanitize_error_text(exc))
        write_json(root / f"report_failure_{time.time_ns()}.json", failure)
        return failure



def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--job",required=True)
    args=parser.parse_args()
    job=read_json(args.job)
    result=run_episode_job(job)
    write_json(Path(job["episode_dir"])/"worker_result.json",result)
    return 0 if result["passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
