"""Pure Dynamic driver and isolated, durably checkpointed episode worker."""
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import sys
import time

from experiments.baselines.b4_embodiskill.state import read_json, read_jsonl, write_json
from experiments.baselines.common.artifact_digest import digest_directory
from experiments.baselines.common.freeze import FrozenArtifact, freeze_files, assert_frozen_unchanged
from experiments.baselines.common.manifest import ManifestTask
from experiments.baselines.common.schema import CommonEpisodeRecord
from experiments.baselines.common.source_identity import sanitize_error_text

METHOD = "b0_dynamic"


def task_example(task):
    return dict(id=task.task_id, task_id=task.task_id, task_type=task.task_type,
                source_split=task.source_split, env_index=task.env_index,
                manifest_index=task.index, gamefile=task.gamefile_rel,
                gamefile_sha256=task.gamefile_sha256, task_signature=task.task_signature)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def usage_totals(events):
    physical = [p for event in events for p in event.get("physical_attempt_usage", [])]
    if events and (not physical or any(e.get("role") != "target" for e in events)):
        raise ValueError("B0 requires target-only physical provider evidence")
    totals = dict(logical_calls=len(events), physical_attempts=len(physical))
    for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "visible_completion_tokens"):
        values = [p.get(key) for p in physical]
        subtotal = sum(v for v in values if isinstance(v, int) and v >= 0)
        totals[key] = subtotal if all(isinstance(v, int) and v >= 0 for v in values) else None
        totals[key + "_known_subtotal"] = subtotal
    return totals


class PureDynamicDriver:
    method_id = METHOD

    def train(self, ctx=None, train_manifest=None, validation_manifest=None):
        if train_manifest is not None or validation_manifest is not None:
            raise ValueError("B0 does not train or consume Validation")
        return dict(episodes=0, llm_calls=0, tokens=0, persistent_skill=None,
                    persistent_experience=None)

    def freeze(self, lane):
        lane = Path(lane)
        if (lane / "frozen").exists():
            frozen = FrozenArtifact.load(lane / "frozen")
            if read_json(frozen.root / "empty_descriptor.json") != self.train():
                raise ValueError("B0 frozen artifact contains nonempty knowledge")
            if sorted(p.name for p in frozen.root.iterdir()) != ["empty_descriptor.json"]:
                raise ValueError("Unexpected B0 knowledge file")
            return frozen
        descriptor = lane / "empty_descriptor.json"
        write_json(descriptor, self.train())
        return freeze_files(method_id=METHOD, source_files={descriptor.name: descriptor},
            destination=lane / "frozen", source_train_manifest_hash="",
            source_validation_manifest_hash=None, metadata=dict(training_required=False))

    def evaluate(self, job):
        return run_episode_job(job)


def completed_outcome(root, task):
    """Reuse only a committed complete episode; never reuse partial responses."""
    receipt_path = root / "episode_checkpoint.json"
    if not receipt_path.exists():
        return None
    receipt = read_json(receipt_path)
    attempt = (root / receipt["attempt"]).resolve()
    attempt.relative_to(root.resolve())
    if digest_directory(attempt) != receipt["digest"]:
        raise ValueError("Completed episode evidence changed")
    outcome = read_json(attempt / "outcome.json")
    if outcome["task"] != task or outcome["infrastructure_failure"]:
        raise ValueError("Invalid committed episode identity/outcome")
    return outcome


def run_episode_job(job):
    # Spawn creates fresh model/environment state for EVERY task (including
    # after resume); posthoc facts are never fed to the text executor.
    root = Path(job["episode_dir"])
    root.mkdir(parents=True, exist_ok=True)
    entry = ManifestTask(**job["task"])
    example = task_example(entry)
    frozen = FrozenArtifact.load(job["frozen"])
    assert_frozen_unchanged(frozen)
    started = time.monotonic()
    try:
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
            model = ModelConfig.from_mapping(job["config"]["model"])
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
                application_retry_limit=5, retry_delays_seconds=[2,5,10,20],
                deterministic_jitter_ratio=.10, expected_sdk_max_retries=0)
            try:
                runner = TextSkillALFWorldExecutor(max_actions=100,
                    max_completion_tokens=job["config"]["method_output_token_hint"],
                    seed=job["seed"], alfworld_data=job["alfworld_data"])
                outcome = asdict(runner.run_episode(task=example, skill_text=None, run_seed=job["seed"],
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
            wall_time_ms=sum(read_json(p)["wall_time_ms"] for p in (root/"attempts").glob("*/outcome.json")),
            artifact_digest_before=frozen.digest, artifact_digest_after=frozen.digest,
            method_metrics=dict(skill_text=None, persistent_experience=False, usage={"target":usage},
                observed_gamefile=outcome["actual_gamefile"],
                model_action_parse_failures=sum(bool(e.get("model_action_parse_failure")) for e in events),
                all_attempt_environment_actions=sum(len(read_json(p)["conversation"]) for p in (root/"attempts").glob("*/outcome.json")),
                replay_wall_time_ms=int((time.monotonic()-started)*1000)))
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
