"""Static pinned SkillOpt seed; reuse the B0 execution and accounting protocol."""
from dataclasses import asdict
from pathlib import Path
import sys
import time

from experiments.baselines.b4_embodiskill.state import read_json, read_jsonl, write_json
from experiments.baselines.common.artifact_digest import digest_directory
from experiments.baselines.common.freeze import FrozenArtifact, freeze_files, assert_frozen_unchanged
from experiments.baselines.common.manifest import ManifestTask
from experiments.baselines.common.schema import CommonEpisodeRecord
from experiments.baselines.common.source_identity import sanitize_error_text
from experiments.baselines.b0_dynamic.driver import (
    task_example, file_hash, usage_totals, completed_outcome,
)

METHOD = "b1_static_skill"
INITIAL_REL = "skillopt/envs/alfworld/skills/initial.md"
REPO = Path(__file__).resolve().parents[3]


def initial_authority(source):
    import yaml
    lock = yaml.safe_load((REPO / "experiments/baselines/baseline_lock.yaml").read_text())
    path = Path(source).resolve() / INITIAL_REL
    expected = lock["skillopt"]["key_files"][INITIAL_REL]
    if file_hash(path) != expected:
        raise ValueError("B1 initial.md differs from the pinned SkillOpt seed")
    return dict(source_path=str(path), source_relative_path=INITIAL_REL,
                external_commit=lock["skillopt"]["commit"], initial_skill_sha256=expected)


def frozen_skill(frozen, authority):
    assert_frozen_unchanged(frozen)
    if (frozen.method_id != METHOD or frozen.source_train_manifest_hash != ""
            or frozen.source_validation_manifest_hash is not None):
        raise ValueError("B1 frozen method or no-training provenance differs")
    if any(frozen.metadata.get(k) != v for k, v in authority.items()):
        raise ValueError("B1 frozen initial skill provenance differs")
    if (frozen.metadata.get("training_required") is not False
            or sorted(p.name for p in frozen.root.iterdir()) != ["initial.md"]
            or file_hash(frozen.root / "initial.md") != authority["initial_skill_sha256"]):
        raise ValueError("B1 frozen skill is not exactly the pinned initial.md")
    return (frozen.root / "initial.md").read_bytes().decode("utf-8")


class StaticSkillDriver:
    method_id = METHOD

    def train(self, ctx=None, train_manifest=None, validation_manifest=None):
        if train_manifest is not None or validation_manifest is not None:
            raise ValueError("B1 does not train or consume Validation")
        return dict(episodes=0, llm_calls=0, tokens=0, persistent_skill="static_initial",
                    persistent_experience=None)

    def freeze(self, lane, source):
        lane = Path(lane)
        authority = initial_authority(source)
        if (lane / "frozen").exists():
            frozen = FrozenArtifact.load(lane / "frozen")
        else:
            frozen = freeze_files(method_id=METHOD,
                source_files={"initial.md": Path(authority["source_path"])},
                destination=lane / "frozen", source_train_manifest_hash="",
                source_validation_manifest_hash=None,
                metadata=dict(training_required=False, **authority))
        frozen_skill(frozen, authority)
        return frozen

    def evaluate(self, job):
        return run_episode_job(job)


def run_episode_job(job):
    # Spawn creates fresh model/environment state for EVERY task (including
    # after resume); posthoc facts are never fed to the text executor.
    root = Path(job["episode_dir"])
    root.mkdir(parents=True, exist_ok=True)
    entry = ManifestTask(**job["task"])
    example = task_example(entry)
    frozen = FrozenArtifact.load(job["frozen"])
    assert_frozen_unchanged(frozen)
    try:
        skill_text = frozen_skill(frozen, job["initial_skill"])
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
            method_metrics=dict(skill_text="frozen/initial.md", initial_skill_sha256=job["initial_skill"]["initial_skill_sha256"],
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
