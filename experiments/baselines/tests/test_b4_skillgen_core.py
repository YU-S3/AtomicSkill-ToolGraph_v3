"""B4 SkillGen-S train supervision, sampling, extraction, and retrieval tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from experiments.baselines.b4_skillgen_s.corpus_builder import (
    build_checkpoint_inventory,
    build_sampling_jobs,
    save_job_result,
    stable_sample_seed,
    validate_completed_corpus,
)
from experiments.baselines.b4_skillgen_s.driver import SkillGenBaselineDriver
from experiments.baselines.b4_skillgen_s import campaign as campaign_module
from experiments.baselines.b4_skillgen_s import controller as controller_module
from experiments.baselines.b4_skillgen_s import corpus_builder as corpus_builder_module
from experiments.baselines.b4_skillgen_s import driver as driver_module
from experiments.baselines.b4_skillgen_s import provider_probe as provider_probe_module
from experiments.baselines.b4_skillgen_s import worker as worker_module
from experiments.baselines.b4_skillgen_s.campaign import (
    CampaignSpec,
    _campaign_lease,
    _phase_command,
)
from experiments.baselines.b4_skillgen_s.extraction_adapter import extract_skill_library
from experiments.baselines.b4_skillgen_s.inference_adapter import (
    AuditedDeepSeekClient,
    ProviderExecutionError,
    ProviderResult,
    make_frozen_inference_prompt,
    run_skillgen_episode,
)
from experiments.baselines.b4_skillgen_s.progress_adapter import (
    SkillGenLabel,
    analyze_label_coverage,
    require_complete_train_coverage,
)
from experiments.baselines.b4_skillgen_s.worker import _load_resume_rows, _ordered_map
from experiments.baselines.common.driver import RunContext
from experiments.baselines.common.freeze import freeze_files
from experiments.baselines.common.manifest import ManifestTask, TaskManifestSet, sha256_json
from experiments.baselines.common.model_config import ModelConfig
from experiments.baselines.common.schema import CommonEpisodeRecord
from experiments.baselines.common.subprocess_worker import WorkerWire
from experiments.baselines.common.usage import RoleUsage


def _manifest(keys: list[str], *, split: str = "train") -> TaskManifestSet:
    tasks = tuple(
        ManifestTask(
            index=index,
            task_id=f"task_{index}",
            task_type=key.split("-", 1)[0],
            source_split=split,
            env_index=index,
            gamefile_rel=f"json_2.1.1/{split}/{key}/game.tw-pddl",
            gamefile_sha256=hashlib.sha256(key.encode()).hexdigest(),
            task_signature=hashlib.sha256(f"signature:{key}".encode()).hexdigest(),
        )
        for index, key in enumerate(keys)
    )
    return TaskManifestSet.create(
        manifest_id=f"{split}_{len(tasks)}",
        benchmark="alfworld",
        source_split=split,
        seed=42,
        tasks=tasks,
    )


def _write_labels(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def _label_row(key: str, **extra: object) -> dict:
    return {
        "task": "alfworld",
        "goal": "put the apple on the table",
        "subgoals": ["You pick up the apple", "You put the apple"],
        "difficulty": "hard",
        "additional_info": {"description": key},
        **extra,
    }


def test_native_upstream_label_schema_is_bound_by_exact_train_coverage(
    tmp_path: Path,
) -> None:
    keys = [
        "pick_and_place_simple-Apple-None-Table-1/trial_train_a",
        "pick_heat_then_place_in_recep-Apple-None-Table-2/trial_train_b",
    ]
    manifest = _manifest(keys)
    labels = _write_labels(tmp_path / "labels.jsonl", [_label_row(key) for key in keys])

    report, loaded = require_complete_train_coverage(manifest, labels)

    assert report.passed is True
    assert report.covered_task_count == 2
    assert all(loaded[key].source_split == "train" for key in keys)


def test_shipped_valid_unseen_labels_fail_closed_for_train6() -> None:
    repo = Path(__file__).resolve().parents[3]
    manifest = TaskManifestSet.load(repo / "data/baseline_manifests/train_6_smoke.json")
    labels = repo / ".external/skillgen/data/alfworld/all.jsonl"

    report, _ = analyze_label_coverage(manifest, labels)

    assert report.passed is False
    assert report.covered_task_count == 0
    assert len(report.missing_task_ids) == 6


@pytest.mark.parametrize(
    "rows,problem",
    [
        (
            lambda key: [_label_row(key, source_split="valid_unseen")],
            "wrong_split_task_ids",
        ),
        (lambda key: [_label_row(key), _label_row(key)], "duplicate_task_keys"),
        (lambda key: [{**_label_row(key), "subgoals": ["["]}], "invalid_rows"),
    ],
)
def test_ambiguous_or_invalid_supervision_is_rejected(
    tmp_path: Path,
    rows,
    problem: str,
) -> None:
    key = "pick_and_place_simple-Apple-None-Table-1/trial_train_a"
    manifest = _manifest([key])
    labels = _write_labels(tmp_path / "labels.jsonl", rows(key))
    report, _ = analyze_label_coverage(manifest, labels)
    assert report.passed is False
    assert getattr(report, problem)


def test_sampling_jobs_are_six_per_task_stable_and_barrier_ordered() -> None:
    keys = [
        "pick_and_place_simple-Apple-None-Table-1/trial_a",
        "pick_cool_then_place_in_recep-Tomato-None-Fridge-2/trial_b",
    ]
    manifest = _manifest(keys)
    labels = {
        task.task_id: SkillGenLabel(
            task_key=key,
            source_split="train",
            goal="goal",
            subgoals=("done",),
            difficulty="hard",
            source_line=index + 1,
        )
        for index, (task, key) in enumerate(zip(manifest.tasks, keys))
    }
    # build_sampling_jobs consumes task-key indexed labels, matching upstream.
    labels = {label.task_key: label for label in labels.values()}
    jobs = build_sampling_jobs(manifest, labels, run_seed=43)
    assert len(jobs) == 12
    assert [job.sample_idx for job in jobs[:6]] == list(range(6))
    assert jobs[0].sample_seed == stable_sample_seed(43, "task_0", 0)
    assert jobs == build_sampling_jobs(manifest, labels, run_seed=43)
    rows = [
        {
            "job_index": job.job_index,
            "task_id": job.task.task_id,
            "manifest_index": job.manifest_index,
            "sample_idx": job.sample_idx,
            "sample_seed": job.sample_seed,
            "goal": "goal",
            "trajectory": [["OBSERVATION", "initial"]],
            "grounding": [],
            "progress": [],
            "progress_rate": 0.0,
            "official_success": False,
            "actions": [],
            "provider_calls": [],
        }
        for job in reversed(jobs)
    ]
    ordered = validate_completed_corpus(jobs, rows)
    assert [row["job_index"] for row in ordered] == list(range(12))
    rows[0]["failure_kind"] = "infrastructure_failure"
    with pytest.raises(RuntimeError, match="cannot cross extraction barrier"):
        validate_completed_corpus(jobs, rows)


class _FakeEncoder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, values, *, convert_to_numpy=True):
        self.calls.append(list(values))
        return [[float(index + 1), 0.5] for index, _ in enumerate(values)]


class _FakeGraph:
    nodes = {"INIT_STATE", "go to table", "FINAL_STATE"}
    edges = [("INIT_STATE", "go to table"), ("go to table", "FINAL_STATE")]

    def __contains__(self, item: str) -> bool:
        return item in self.nodes


class _FakeTaskGraph:
    constructed: list[tuple[str, str]] = []

    def __init__(self, save_dir: str, task_name: str) -> None:
        self.constructed.append((save_dir, task_name))
        self.graph = _FakeGraph()
        self.edge_progress = {
            ("INIT_STATE", "go to table"): [0.5],
            ("go to table", "FINAL_STATE"): [0.5],
        }
        self.received = None

    def add_sample_trajectory(self, trajectories) -> None:
        self.received = trajectories


class _FakeEstimator:
    calls: list[tuple[float, float, float, int]] = []

    def __init__(self, graph, gamma: float, lambda_: float, alpha: float) -> None:
        self.values = (gamma, lambda_, alpha)

    def compute_q_values(self, num_iterations: int) -> None:
        self.calls.append((*self.values, num_iterations))

    def combined_ranking_step_wise(self):
        return {"INIT_STATE": [("go to table", 0.8)]}


def test_extraction_calls_upstream_graph_td_and_embedding_core(tmp_path: Path) -> None:
    encoder = _FakeEncoder()
    valid = {
        "unique_id": 0,
        "task_name": "pick_and_place_simple",
        "sample_name": "sample",
        "goal": "put apple",
        "init_obs": "room",
        "actions": ["go to table"],
        "observations": ["at table"],
        "progress": [0.5],
        "progress_rate": 0.5,
    }

    class Utils:
        @staticmethod
        def keep_one_of_duplicates_dicts(rows):
            return rows

        @staticmethod
        def collect_valid_traj(rows, dataset_name=None):
            assert dataset_name == "alfworld"
            return [dict(valid)]

        @staticmethod
        def process_trajectories(rows):
            assert rows == [valid]
            return [{"trajs": [("INIT_OBS", "INIT_STATE", 0.0)]}]

    class Embed:
        @staticmethod
        def add_current_node_embeddings(rows, model):
            vectors = model.encode(
                [row["current_node"] for row in rows], convert_to_numpy=True
            )
            for row, vector in zip(rows, vectors):
                row["embedding"] = list(vector)
            return rows

    upstream = SimpleNamespace(
        extraction_utils=Utils(),
        domain_graph=SimpleNamespace(
            TaskGraph=_FakeTaskGraph,
            ActionContributionEstimator=_FakeEstimator,
            np=SimpleNamespace(random=SimpleNamespace(seed=lambda value: None)),
        ),
        embed_skills=Embed(),
    )
    corpus = [{
        "task_id": "task_0",
        "task_key": "pick_and_place_simple-Apple-None-Table-1/trial_a",
        "task_type": "pick_and_place_simple",
        "goal": "put apple",
        "progress": [[0, 0.5]],
        "trajectory": [
            ["OBSERVATION", "room"], ["ACTION", "go to table"],
            ["OBSERVATION", "at table"],
        ],
        "grounding": [0],
        "command_turns": 1,
        "grounding_rate": 1.0,
        "progress_rate": 0.5,
    }]

    result = extract_skill_library(
        corpus,
        output_dir=tmp_path / "library",
        external_root=Path(__file__).resolve().parents[3] / ".external/skillgen",
        encoder=encoder,
        upstream=upstream,
    )

    assert _FakeEstimator.calls[-1] == (0.95, 0.9, 0.05, 500)
    assert result.metrics["embedding_calls"] == 2
    assert result.metrics["skill_count"] == 1
    assert result.metrics["golden_segment_count"] == 1
    assert sorted(result.files) == [
        "artifact_manifest.json",
        "domain_graphs/pick_and_place_simple.json",
        "golden_segments/pick_and_place_simple.txt",
        "step_skill_embeddings/pick_and_place_simple.jsonl",
        "step_skills/pick_and_place_simple.jsonl",
        "train_metadata_embeddings.jsonl",
    ]


def test_frozen_prompt_uses_top_one_and_counts_encoder_calls(tmp_path: Path) -> None:
    root = tmp_path / "library"
    (root / "step_skill_embeddings").mkdir(parents=True)
    (root / "golden_segments").mkdir()
    (root / "train_metadata_embeddings.jsonl").write_text(
        json.dumps({
            "task_id": "train_0", "domain": "pick_and_place_simple",
            "embedding": [1.0, 0.5],
        }) + "\n",
        encoding="utf-8",
    )
    (root / "step_skill_embeddings/pick_and_place_simple.jsonl").write_text(
        json.dumps({"current_node": "INIT_STATE", "children": [], "embedding": [1.0, 0.5]}) + "\n",
        encoding="utf-8",
    )
    (root / "golden_segments/pick_and_place_simple.txt").write_text(
        "Example 1:\nGoal: put apple", encoding="utf-8"
    )
    encoder = _FakeEncoder()
    observed: dict[str, int] = {}

    class Retrieval:
        @staticmethod
        def retrieve(data, action, model, top_k_actions, top_k_skills):
            observed.update(actions=top_k_actions, skills=top_k_skills)
            model.encode([action], convert_to_numpy=True)
            return {"matched_nodes": ["INIT_STATE"], "skills": []}

    prompt = SimpleNamespace(
        skill_template="{instructions}\n{goal}\n{skills}\n{trajectories}\n{history}",
        prompt_dict={"simple_instruction": "solve", "system_msg": "system"},
    )
    messages, evidence = make_frozen_inference_prompt(
        prompt_core=prompt,
        retrieval_core=Retrieval(),
        frozen_root=root,
        category="pick_and_place_simple",
        goal="put apple",
        history=[["OBSERVATION", "room"]],
        encoder=encoder,
    )
    assert observed == {"actions": 1, "skills": 1}
    assert evidence["embedding_calls"] == 2
    assert evidence["golden_segment_used"] is True
    assert messages[0] == {"role": "system", "content": "system"}

    _, cached_evidence = make_frozen_inference_prompt(
        prompt_core=prompt,
        retrieval_core=Retrieval(),
        frozen_root=root,
        category="pick_and_place_simple",
        goal="put apple",
        history=[["OBSERVATION", "room"], ["ACTION", "look"], ["OBSERVATION", "room"]],
        encoder=encoder,
        retrieved_domains=evidence["retrieved_domains"],
    )
    assert cached_evidence["embedding_calls"] == 1
    assert encoder.calls == [
        ["Goal: put apple Domain: pick_and_place_simple."],
        ["INIT_STATE"],
        ["look"],
    ]


def test_mock_sampling_episode_preserves_progress_and_official_won() -> None:
    task = _manifest([
        "pick_and_place_simple-Apple-None-Table-1/trial_train_a"
    ]).tasks[0]
    label = SkillGenLabel(
        task_key="pick_and_place_simple-Apple-None-Table-1/trial_train_a",
        source_split="train",
        goal="put apple",
        subgoals=("picked apple",),
        difficulty="hard",
        source_line=1,
    )

    class Provider:
        def complete(self, messages, **kwargs):
            return ProviderResult(
                text="look",
                usage=RoleUsage(calls=1, prompt_tokens=4, completion_tokens=1),
                event={
                    "status": "succeeded", "stage": "sampling",
                    "prompt_tokens": 4, "completion_tokens": 1,
                    "reasoning_tokens": 0,
                },
            )

    class Env:
        won = False
        actual_gamefile = "fixture"

        def __init__(self, **kwargs):
            pass

        def reset(self):
            return "room", "put apple", label.task_key

        def get_action_space(self):
            return ["look"]

        def step(self, action):
            self.won = True
            return {
                "action": action, "observation": "picked apple", "done": True,
                "won": True, "score": 1.0, "environment_action_executed": True,
            }

        def close(self):
            pass

    prompt = SimpleNamespace(
        sampling_make_prompt=lambda **kwargs: [{"role": "user", "content": "go"}],
        extract_action=lambda text: text,
        prompt_dict={},
    )
    row = run_skillgen_episode(
        task=task,
        external_root=".",
        alfworld_data=".",
        model={},
        run_id="run",
        run_seed=42,
        phase="train",
        max_steps=10,
        max_length=64,
        temperature=1.0,
        top_p=0.95,
        do_sample=True,
        sample_seed=7,
        sample_idx=0,
        label=label,
        prompt_core=prompt,
        provider=Provider(),
        env_factory=Env,
    )
    assert row["official_success"] is True
    assert row["progress_rate"] == 1.0
    assert row["progress"] == [[0, 1.0]]
    assert row["environment_actions"] == 1
    assert row["target_usage"]["calls"] == 1


def test_mid_episode_provider_failure_preserves_paid_calls_and_actions() -> None:
    task = _manifest([
        "pick_and_place_simple-Apple-None-Table-1/trial_train_a"
    ]).tasks[0]
    label = SkillGenLabel(
        task_key="pick_and_place_simple-Apple-None-Table-1/trial_train_a",
        source_split="train",
        goal="put apple",
        subgoals=("picked apple",),
        difficulty="hard",
        source_line=1,
    )

    class Provider:
        calls = 0

        def complete(self, messages, **kwargs):
            self.calls += 1
            event = {
                "call_id": f"call_{self.calls}",
                "status": "succeeded" if self.calls == 1 else "failed",
                "application_attempts": 1,
                "prompt_tokens": 4 if self.calls == 1 else 0,
                "completion_tokens": 1 if self.calls == 1 else 0,
            }
            if self.calls == 2:
                raise ProviderExecutionError(
                    "provider interrupted",
                    failure_kind="infrastructure_failure",
                    event=event,
                )
            return ProviderResult(
                text="look",
                usage=RoleUsage(calls=1, prompt_tokens=4, completion_tokens=1),
                event=event,
            )

    class Env:
        won = False
        actual_gamefile = "fixture"

        def __init__(self, **kwargs):
            pass

        def reset(self):
            return "room", "put apple", label.task_key

        def get_action_space(self):
            return ["look"]

        def step(self, action):
            return {
                "action": action,
                "observation": "still working",
                "done": False,
                "won": False,
                "score": 0.0,
                "environment_action_executed": True,
            }

        def close(self):
            pass

    prompt = SimpleNamespace(
        sampling_make_prompt=lambda **kwargs: [{"role": "user", "content": "go"}],
        extract_action=lambda text: text,
        prompt_dict={},
    )
    with pytest.raises(ProviderExecutionError) as raised:
        run_skillgen_episode(
            task=task,
            external_root=".",
            alfworld_data=".",
            model={},
            run_id="run",
            run_seed=42,
            phase="train",
            max_steps=10,
            max_length=64,
            temperature=1.0,
            top_p=0.95,
            do_sample=True,
            sample_seed=7,
            sample_idx=0,
            label=label,
            prompt_core=prompt,
            provider=Provider(),
            env_factory=Env,
        )

    assert [event["call_id"] for event in raised.value.event["provider_calls"]] == [
        "call_1",
        "call_2",
    ]
    assert len(raised.value.event["action_events"]) == 1
    assert raised.value.event["action_events"][0]["action"] == "look"


def _wire(
    tmp_path: Path,
    *,
    phase: str,
    identity: dict[str, str],
    manifest_path: str | None,
    test_manifest_path: str | None = None,
    supervision_path: str | None = None,
    frozen_artifact_path: str | None = None,
    resume: dict | None = None,
) -> WorkerWire:
    return WorkerWire(
        method="b4_skillgen_s",
        phase=phase,
        manifest_path=manifest_path,
        validation_manifest_path=None,
        test_manifest_path=test_manifest_path,
        config_path=str(tmp_path / "config_resolved.json"),
        output_dir=str(tmp_path / "run"),
        run_seed=42,
        model={"model": "deepseek-v4-flash", "reasoning_effort": "high"},
        run_id="b4_fixture",
        result_path=str(tmp_path / "run" / phase / "worker_result.json"),
        identity=identity,
        frozen_artifact_path=frozen_artifact_path,
        external_method_root=str(tmp_path / "skillgen"),
        supervision_path=supervision_path,
        resume=resume,
    )


def test_b4_worker_wire_separates_train_supervision_and_test(tmp_path: Path) -> None:
    base = {
        "config_digest": "a" * 64,
        "train_manifest_digest": "b" * 64,
        "external_source_digest": "c" * 64,
    }
    smoke = _wire(
        tmp_path,
        phase="smoke",
        identity={
            **base,
            "supervision_digest": "d" * 64,
        },
        manifest_path="train.json",
        supervision_path="labels.jsonl",
    )
    assert smoke.manifest_path == "train.json"
    assert smoke.test_manifest_path is None
    with pytest.raises(ValueError, match="forbids.*test_manifest_path"):
        _wire(
            tmp_path,
            phase="smoke",
            identity={
                **base,
                "supervision_digest": "d" * 64,
            },
            manifest_path="train.json",
            test_manifest_path="test.json",
            supervision_path="labels.jsonl",
        )

    smoke_test = _wire(
        tmp_path,
        phase="smoke_test",
        identity={
            **base,
            "test_manifest_digest": "e" * 64,
            "evaluation_manifest_digest": "e" * 64,
            "frozen_artifact_digest": "f" * 64,
        },
        manifest_path=None,
        test_manifest_path="test.json",
        frozen_artifact_path="frozen",
    )
    assert smoke_test.manifest_path is None
    assert smoke_test.supervision_path is None

    test = _wire(
        tmp_path,
        phase="test",
        identity={
            **base,
            "test_manifest_digest": "e" * 64,
            "evaluation_manifest_digest": "e" * 64,
            "frozen_artifact_digest": "f" * 64,
        },
        manifest_path=None,
        test_manifest_path="test.json",
        frozen_artifact_path="frozen",
    )
    assert test.manifest_path is None
    assert test.supervision_path is None


def test_b4_worker_requires_spawn_start_method(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[3]
    common = yaml.safe_load(
        (repo / "configs/baselines/common.yaml").read_text(encoding="utf-8")
    )
    method = yaml.safe_load(
        (repo / "configs/baselines/b4_skillgen_s_smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    config = {**common, **method}
    output = tmp_path / "run"
    output.mkdir()
    config_path = output / "config_resolved.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    model = ModelConfig.from_mapping(dict(config["model"]))
    wire = _wire(
        tmp_path,
        phase="smoke",
        identity={
            "config_digest": sha256_json(config),
            "train_manifest_digest": "b" * 64,
            "external_source_digest": "c" * 64,
            "supervision_digest": "d" * 64,
        },
        manifest_path="train.json",
        supervision_path="labels.jsonl",
    )
    object.__setattr__(wire, "config_path", str(config_path))
    object.__setattr__(wire, "model", model.to_wire())

    assert worker_module._validate_config(wire, config) == model

    config["parallel"]["mp_start_method"] = "fork"
    wire.identity["config_digest"] = sha256_json(config)
    with pytest.raises(ValueError, match="parallel.mp_start_method"):
        worker_module._validate_config(wire, config)


def test_sampling_map_persists_each_completion_before_reassembly() -> None:
    durable: list[int] = []
    rows = _ordered_map(
        [{"index": 2}, {"index": 0}, {"index": 1}],
        lambda payload: {"job_index": payload["index"]},
        workers=1,
        index_field="job_index",
        on_result=lambda row: durable.append(int(row["job_index"])),
    )
    assert durable == [2, 0, 1]
    assert [row["job_index"] for row in rows] == [0, 1, 2]


def _resume_checkpoint_fixture(
    tmp_path: Path,
) -> tuple[WorkerWire, tuple, Path, Path]:
    key = "pick_and_place_simple-Apple-None-Table-1/trial_train_a"
    manifest = _manifest([key])
    label = SkillGenLabel(
        task_key=key,
        source_split="train",
        goal="goal",
        subgoals=("done",),
        difficulty="hard",
        source_line=1,
    )
    jobs = build_sampling_jobs(manifest, {key: label}, run_seed=42)
    gamefile = tmp_path / jobs[0].task.gamefile_rel
    gamefile.parent.mkdir(parents=True)
    gamefile.write_text(key, encoding="utf-8")
    source = tmp_path / "source"
    jobs_dir = source / "train" / "sampling" / "jobs"
    jobs_dir.mkdir(parents=True)
    identity = {
        "config_digest": "a" * 64,
        "train_manifest_digest": "b" * 64,
        "test_manifest_digest": "c" * 64,
        "external_source_digest": "d" * 64,
        "controller_code_digest": "e" * 64,
        "model_identity_digest": "f" * 64,
        "alfworld_data_digest": "1" * 64,
        "formal_config_digest": "2" * 64,
        "supervision_digest": "3" * 64,
    }
    run_manifest = {
        "method": "b4_skillgen_s",
        "phase": "train",
        "run_id": "b4_source_fixture",
        "run_seed": 42,
        "identity": identity,
        "campaign": {"campaign_lock_digest": "4" * 64},
    }
    raw_manifest = (json.dumps(run_manifest) + "\n").encode()
    (source / "run_manifest.json").write_bytes(raw_manifest)
    raw_state = (json.dumps({
        "schema_version": 1,
        "run_id": "b4_source_fixture",
        "state": "failed",
        "phase": "train",
        "failure_kind": "infrastructure_failure",
    }) + "\n").encode()
    (source / "run_state.json").write_bytes(raw_state)
    response_text = "look"
    logical_task_id = f"{jobs[0].task.task_id}::sample_{jobs[0].sample_idx}"
    success = {
        "job_index": 0,
        "task_id": jobs[0].task.task_id,
        "task_key": key,
        "task_type": jobs[0].task.task_type,
        "manifest_index": jobs[0].manifest_index,
        "sample_idx": jobs[0].sample_idx,
        "sample_seed": jobs[0].sample_seed,
        "goal": "goal",
        "trajectory": [
            ["OBSERVATION", "start"],
            ["ACTION", "look"],
            ["OBSERVATION", "done"],
        ],
        "command_turns": 1,
        "grounding": [0],
        "grounding_rate": 1.0,
        "progress": [[0, 1.0]],
        "progress_rate": 1.0,
        "official_success": False,
        "termination_reason": "environment_done_without_win",
        "environment_done": True,
        "actions": [{
            "episode_task_id": logical_task_id,
            "step_index": 0,
            "command_turn_index": 0,
            "action": "look",
            "env_feedback": "done",
            "reward": 0.0,
            "done": True,
            "won": False,
            "admissible_before": True,
        }],
        "environment_actions": 1,
        "provider_calls": [{
            "schema_version": 1,
            "event": "provider_call",
            "method": "b4_skillgen_s",
            "phase": "train",
            "run_id": "b4_source_fixture",
            "run_seed": 42,
            "episode_task_id": logical_task_id,
            "call_id": "skillgen_" + hashlib.sha256(
                f"b4_source_fixture\0{logical_task_id}\0sampling\0{0}\0{'5' * 64}".encode()
            ).hexdigest()[:24],
            "role": "target",
            "stage": "sampling",
            "model": "deepseek-v4-flash",
            "reasoning_effort": "high",
            "messages_sha256": "5" * 64,
            "max_tokens": 64,
            "temperature": 1.0,
            "top_p": 0.95,
            "sample_seed": jobs[0].sample_seed,
            "do_sample": True,
            "requested_retry_limit": 5,
            "status": "succeeded",
            "application_attempts": 1,
            "sdk_boundary_attempts": 1,
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "reasoning_tokens": 1,
            "total_tokens": 10,
            "response_text": response_text,
            "response_sha256": hashlib.sha256(response_text.encode()).hexdigest(),
        }],
        "target_usage": {
            "calls": 1,
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "reasoning_tokens": 1,
        },
        "retrieval": [],
        "embedding_calls": 0,
        "actual_gamefile": str(gamefile.resolve()),
        "gamefile": jobs[0].task.gamefile_rel,
        "gamefile_hash": jobs[0].task.gamefile_sha256,
        "wall_time_ms": 5,
        "failure_kind": "",
    }
    (jobs_dir / "0000.json").write_text(json.dumps(success), encoding="utf-8")
    (jobs_dir / "0001.json").write_text(
        json.dumps({
            "job_index": 1,
            "task_id": jobs[1].task.task_id,
            "manifest_index": jobs[1].manifest_index,
            "sample_idx": jobs[1].sample_idx,
            "sample_seed": jobs[1].sample_seed,
            "failure_kind": "infrastructure_failure",
            "provider_calls": [],
            "actions": [],
            "wall_time_ms": 1,
        }),
        encoding="utf-8",
    )
    train_wire_identity = {
        key: value for key, value in identity.items() if key != "test_manifest_digest"
    }
    wire = _wire(
        tmp_path,
        phase="train",
        identity=train_wire_identity,
        manifest_path="train.json",
        supervision_path="labels.jsonl",
        resume={
            "schema_version": 2,
            "boundary": "sampling_job",
            "source_run": str(source),
            "run_manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
            "run_state_sha256": hashlib.sha256(raw_state).hexdigest(),
            "checkpoint_files_observed": 2,
            "checkpoint_inventory": build_checkpoint_inventory(jobs_dir),
        },
    )
    object.__setattr__(wire, "campaign", {
        "campaign_id": "campaign",
        "campaign_lock_path": str(tmp_path / "lock.json"),
        "campaign_lock_digest": "4" * 64,
        "provider_gate_dir": str(tmp_path / "gate"),
        "campaign_provider_max_inflight": 16,
    })

    return wire, jobs, tmp_path, source


def test_resume_imports_only_identity_bound_successful_sampling_jobs(
    tmp_path: Path,
) -> None:
    wire, jobs, data_root, _ = _resume_checkpoint_fixture(tmp_path)

    reused, evidence = _load_resume_rows(
        wire, jobs, data_root, extract_action=lambda text: text,
    )

    assert list(reused) == [0]
    assert evidence["reused_sampling_jobs"] == 1
    assert evidence["checkpoint_inventory_digest"] == wire.resume[
        "checkpoint_inventory"
    ]["digest"]


def test_controller_resume_descriptor_binds_state_and_every_checkpoint(
    tmp_path: Path,
) -> None:
    _, _, _, source = _resume_checkpoint_fixture(tmp_path)
    manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))

    descriptor = controller_module._load_resume_descriptor(
        source,
        seed=42,
        expected_identity=dict(manifest["identity"]),
        expected_campaign={"campaign_lock_digest": "4" * 64},
    )

    assert descriptor["schema_version"] == 2
    assert descriptor["checkpoint_files_observed"] == 2
    assert [
        row["name"] for row in descriptor["checkpoint_inventory"]["files"]
    ] == ["0000.json", "0001.json"]
    assert len(descriptor["run_state_sha256"]) == 64


def test_b4_campaign_descriptor_requires_spawn_start_method(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ModelConfig.from_mapping({
        "provider": "openai_compatible",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
        "api_key_env": "MODEL_API_KEY",
        "reasoning_effort": "high",
    })
    config = {
        "parallel": {
            "seed_lanes": 1,
            "campaign_provider_max_inflight": 16,
            "mp_start_method": "spawn",
        }
    }
    lock = {
        "skillgen": {
            "commit": "a" * 40,
            "runtime_tree": {"sha256": "b" * 64},
        }
    }
    campaign_root = tmp_path / "campaign"
    campaign_root.mkdir()
    path = campaign_root / "campaign_lock.json"
    payload = {
        "schema_version": 1,
        "method": "b4_skillgen_s",
        "seeds": [42, 43, 44],
        "train_manifest_digest": "1" * 64,
        "validation_manifest_digest": None,
        "test_manifest_digest": "2" * 64,
        "controller_commit": "3" * 40,
        "controller_code_digest": "4" * 64,
        "external_method_commit": "a" * 40,
        "external_runtime_tree_digest": "b" * 64,
        "supervision_digest": "5" * 64,
        "model": "deepseek-v4-flash",
        "reasoning_effort": "high",
        "formal_config_digest": controller_module._formal_config_digest(config),
        "campaign_provider_max_inflight": 16,
        "seed_lanes": 1,
        "parallel": dict(config["parallel"]),
        "provider_gate_dir": str(campaign_root / "provider_gate"),
        "campaign_id": "campaign",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        controller_module,
        "_validate_campaign_probe_receipt",
        lambda *args, **kwargs: None,
    )
    train = SimpleNamespace(digest="1" * 64)
    test = SimpleNamespace(digest="2" * 64)

    descriptor = controller_module._load_campaign_descriptor(
        path,
        config=config,
        lock=lock,
        model=model,
        train=train,
        test=test,
        supervision_digest="5" * 64,
        seed=42,
        git_state={"commit": "3" * 40},
        code_digest="4" * 64,
    )
    assert descriptor["campaign_provider_max_inflight"] == 16

    payload["parallel"]["mp_start_method"] = "fork"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="mp_start_method"):
        controller_module._load_campaign_descriptor(
            path,
            config=config,
            lock=lock,
            model=model,
            train=train,
            test=test,
            supervision_digest="5" * 64,
            seed=42,
            git_state={"commit": "3" * 40},
            code_digest="4" * 64,
        )


def test_resume_rejects_checkpoint_tampering_after_inventory(tmp_path: Path) -> None:
    wire, jobs, data_root, source = _resume_checkpoint_fixture(tmp_path)
    path = source / "train" / "sampling" / "jobs" / "0000.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    row["environment_actions"] = 99
    path.write_text(json.dumps(row), encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint digest changed"):
        _load_resume_rows(wire, jobs, data_root, extract_action=lambda text: text)


@pytest.mark.parametrize(
    "tamper,error",
    [
        (
            lambda row: row["provider_calls"][0].__setitem__(
                "model", "different-model"
            ),
            "provider identity differs",
        ),
        (
            lambda row: (
                row.__setitem__("progress", []),
                row.__setitem__("progress_rate", 0.0),
            ),
            "supervised progress differs",
        ),
        (
            lambda row: (
                row.__setitem__("grounding", []),
                row.__setitem__("grounding_rate", 0.0),
            ),
            "grounding evidence differs",
        ),
        (
            lambda row: row["provider_calls"][0].__setitem__(
                "call_id", "skillgen_" + "a" * 24
            ),
            "provider call evidence differs",
        ),
        (
            lambda row: (
                row["provider_calls"][0].__setitem__("response_text", "inventory"),
                row["provider_calls"][0].__setitem__(
                    "response_sha256", hashlib.sha256(b"inventory").hexdigest()
                ),
            ),
            "response/action evidence differs",
        ),
    ],
)
def test_resume_rejects_semantic_tampering_even_with_new_inventory(
    tmp_path: Path,
    tamper,
    error: str,
) -> None:
    wire, jobs, data_root, source = _resume_checkpoint_fixture(tmp_path)
    jobs_dir = source / "train" / "sampling" / "jobs"
    path = jobs_dir / "0000.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    tamper(row)
    path.write_text(json.dumps(row), encoding="utf-8")
    descriptor = {**dict(wire.resume or {})}
    descriptor["checkpoint_inventory"] = build_checkpoint_inventory(jobs_dir)
    object.__setattr__(wire, "resume", descriptor)

    with pytest.raises(ValueError, match=error):
        _load_resume_rows(wire, jobs, data_root, extract_action=lambda text: text)


def test_sampling_checkpoint_write_is_fsynced_and_never_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fsync_calls: list[int] = []
    monkeypatch.setattr(
        corpus_builder_module.os,
        "fsync",
        lambda descriptor: fsync_calls.append(int(descriptor)),
    )
    target = tmp_path / "jobs" / "0000.json"
    save_job_result(target, {"job_index": 0})

    assert json.loads(target.read_text(encoding="utf-8")) == {"job_index": 0}
    assert fsync_calls
    assert not list(target.parent.glob("*.tmp"))
    with pytest.raises(FileExistsError):
        save_job_result(target, {"job_index": 0, "changed": True})


def test_sampling_checkpoint_publish_does_not_overwrite_a_racing_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "jobs" / "0000.json"
    original_link = corpus_builder_module.os.link

    def racing_link(source, destination, **kwargs):
        Path(destination).write_text('{"owner": "other"}\n', encoding="utf-8")
        return original_link(source, destination, **kwargs)

    monkeypatch.setattr(corpus_builder_module.os, "link", racing_link)

    with pytest.raises(FileExistsError):
        save_job_result(target, {"owner": "ours"})
    assert json.loads(target.read_text(encoding="utf-8")) == {"owner": "other"}
    assert not list(target.parent.glob("*.tmp"))


class _SDK:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create),
        )

    def create(self, **kwargs):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _HTTPError(RuntimeError):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}")


def _provider_response() -> SimpleNamespace:
    return SimpleNamespace(
        _request_id="request-ok",
        choices=[SimpleNamespace(message=SimpleNamespace(content="look"))],
        usage=SimpleNamespace(
            prompt_tokens=3,
            completion_tokens=2,
            total_tokens=5,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=1),
        ),
    )


def test_provider_transient_retry_and_permanent_failure_are_distinct() -> None:
    model = {
        "provider": "openai_compatible",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
        "reasoning_effort": "high",
        "api_key_env": "MODEL_API_KEY",
    }
    recovered = AuditedDeepSeekClient(
        model=model,
        run_id="run",
        run_seed=42,
        phase="train",
        retry_delays=(0, 0, 0, 0),
        jitter_ratio=0,
        sdk_client=_SDK([_HTTPError(429), _provider_response()]),
    ).complete(
        [{"role": "user", "content": "act"}],
        task_id="task",
        stage="sampling",
        call_index=0,
        max_tokens=64,
        temperature=1.0,
        top_p=0.95,
        seed=7,
        do_sample=True,
    )
    assert recovered.event["application_attempts"] == 2
    assert recovered.event["failure_codes"] == ["http_429"]
    assert recovered.event["recovered"] is True

    permanent = AuditedDeepSeekClient(
        model=model,
        run_id="run",
        run_seed=42,
        phase="train",
        retry_delays=(0, 0, 0, 0),
        jitter_ratio=0,
        sdk_client=_SDK([_HTTPError(400)]),
    )
    with pytest.raises(ProviderExecutionError) as raised:
        permanent.complete(
            [{"role": "user", "content": "act"}],
            task_id="task",
            stage="sampling",
            call_index=0,
            max_tokens=64,
            temperature=1.0,
            top_p=0.95,
            seed=7,
            do_sample=True,
        )
    assert raised.value.failure_kind == "protocol_failure"
    assert raised.value.event["application_attempts"] == 1


def test_driver_constructs_distinct_smoke_and_smoke_test_authorities(
    tmp_path: Path,
) -> None:
    labels = tmp_path / "labels.jsonl"
    labels.write_text("{}\n", encoding="utf-8")
    driver = SkillGenBaselineDriver(
        config={},
        repo_root=tmp_path,
        external_root=tmp_path / "skillgen",
        lock={"skillgen": {"runtime_tree": {"sha256": "c" * 64}}},
        worker_python=tmp_path / "python",
        supervision_path=labels,
    )
    model = ModelConfig(
        provider="openai_compatible",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
        api_key_env="MODEL_API_KEY",
        reasoning_effort="high",
    )
    identity = {
        "config_digest": "a" * 64,
        "train_manifest_digest": "b" * 64,
        "test_manifest_digest": "d" * 64,
        "external_source_digest": "c" * 64,
    }
    ctx = RunContext(
        campaign_id="fixture",
        method_id="b4_skillgen_s",
        run_seed=42,
        output_dir=tmp_path / "run",
        repo_root=tmp_path,
        external_repo=tmp_path / "skillgen",
        external_commit="commit",
        model_config=model,
        max_environment_actions=100,
        alfworld_data=tmp_path,
        config_hash="a" * 64,
        code_hash="e" * 64,
        run_id="fixture",
        resolved_config_path=tmp_path / "run/config_resolved.json",
        identity=identity,
        train_manifest_path=tmp_path / "train.json",
        test_manifest_path=tmp_path / "test.json",
    )

    train_wire = driver._wire(ctx, phase="smoke", include_supervision=True)
    assert train_wire.manifest_path == str(tmp_path / "train.json")
    assert train_wire.test_manifest_path is None
    assert "test_manifest_digest" not in train_wire.identity
    assert train_wire.supervision_path == str(labels)

    frozen_identity = {
        **identity,
        "evaluation_manifest_digest": "d" * 64,
        "frozen_artifact_digest": "f" * 64,
        "supervision_digest": hashlib.sha256(labels.read_bytes()).hexdigest(),
    }
    frozen_ctx = RunContext(**{**ctx.__dict__, "identity": frozen_identity})
    test_wire = driver._wire(
        frozen_ctx,
        phase="smoke_test",
        frozen_dir=tmp_path / "smoke_frozen",
    )
    assert test_wire.manifest_path is None
    assert test_wire.test_manifest_path == str(tmp_path / "test.json")
    assert test_wire.supervision_path is None
    assert "supervision_digest" not in test_wire.identity


def test_b4_native_provider_probe_persists_complete_concurrent_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, **kwargs) -> None:
            self.run_id = kwargs["run_id"]
            self.run_seed = kwargs["run_seed"]
            self.phase = kwargs["phase"]

        def complete(self, messages, **kwargs):
            return ProviderResult(
                text="OK",
                usage=RoleUsage(calls=1, prompt_tokens=2, completion_tokens=1),
                event={
                    "schema_version": 1,
                    "method": "b4_skillgen_s",
                    "phase": self.phase,
                    "run_id": self.run_id,
                    "run_seed": self.run_seed,
                    "stage": kwargs["stage"],
                    "call_id": f"call_{kwargs['call_index']}",
                    "status": "succeeded",
                    "application_attempts": 1,
                    "sdk_boundary_attempts": 1,
                    "failure_codes": [],
                    "provider_slot_ids": [0],
                    "provider_queue_wait_ms": 0,
                    "retry_backoff_ms": 0,
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
                    "response_text": "OK",
                },
            )

    monkeypatch.setattr(provider_probe_module, "AuditedDeepSeekClient", FakeClient)
    report = provider_probe_module.run_probe(
        output_dir=tmp_path / "probe",
        gate_dir=tmp_path / "gate",
        campaign_id="campaign",
        run_id="probe_run",
        run_seed=42,
        model={
            "provider": "openai_compatible",
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-v4-flash",
            "api_key_env": "MODEL_API_KEY",
            "reasoning_effort": "high",
        },
        concurrency=2,
        requests=4,
        max_completion_tokens=256,
        max_inflight=2,
        retry_delays=(0, 0, 0, 0),
        jitter_ratio=0,
    )
    assert report["passed"] is True
    assert report["logical_calls_recorded"] == 4
    calls = (tmp_path / "probe/provider_calls.jsonl").read_text(encoding="utf-8")
    assert "response_text" not in calls


def test_b4_campaign_commands_keep_train_and_test_authority_separate(
    tmp_path: Path,
) -> None:
    spec = CampaignSpec(
        train_manifest=tmp_path / "train.json",
        test_manifest=tmp_path / "test.json",
        supervision=tmp_path / "labels.jsonl",
        config=tmp_path / "config.yaml",
        output_dir=tmp_path / "campaign",
        python=tmp_path / "python",
        repo_root=tmp_path,
    )
    lock = tmp_path / "campaign_lock.json"
    train = _phase_command(
        spec,
        phase="train",
        seed=42,
        output_dir=tmp_path / "seed_42/train",
        campaign_lock=lock,
        resume_source_run=tmp_path / "prior",
    )
    test = _phase_command(
        spec,
        phase="test",
        seed=42,
        output_dir=tmp_path / "seed_42/test",
        campaign_lock=lock,
        source_run=tmp_path / "seed_42/train",
    )
    assert "--supervision" in train
    assert "--resume-source-run" in train
    assert "--validation-manifest" not in train
    assert "--supervision" not in test
    assert "--resume-source-run" not in test
    assert "--source-run" in test


def test_b4_campaign_lease_does_not_relabel_body_io_errors(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="body failure"):
        with _campaign_lease(tmp_path, owner="fixture"):
            raise OSError("body failure")


def test_b4_campaign_runs_seed_lanes_serially(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = CampaignSpec(
        train_manifest=tmp_path / "train.json",
        test_manifest=tmp_path / "test.json",
        supervision=tmp_path / "labels.jsonl",
        config=tmp_path / "config.yaml",
        output_dir=tmp_path / "campaign",
        python=tmp_path / "python",
        repo_root=tmp_path,
    )
    events: list[tuple[str, int]] = []
    active = 0

    def run_lane(_spec, *, seed, **kwargs):
        nonlocal active
        active += 1
        assert active == 1
        events.append(("start", seed))
        events.append(("finish", seed))
        active -= 1
        return {"passed": True, "seed": seed}

    monkeypatch.setattr(campaign_module, "_run_lane", run_lane)
    lanes = campaign_module._run_seed_lanes(
        spec,
        campaign_lock=tmp_path / "campaign_lock.json",
        lock_digest="a" * 64,
        runner=lambda *args, **kwargs: 0,
    )

    assert [row["seed"] for row in lanes] == [42, 43, 44]
    assert events == [
        ("start", 42), ("finish", 42),
        ("start", 43), ("finish", 43),
        ("start", 44), ("finish", 44),
    ]


def test_b4_campaign_retries_infrastructure_with_sampling_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = CampaignSpec(
        train_manifest=tmp_path / "train.json",
        test_manifest=tmp_path / "test.json",
        supervision=tmp_path / "labels.jsonl",
        config=tmp_path / "config.yaml",
        output_dir=tmp_path / "campaign",
        python=tmp_path / "python",
        repo_root=tmp_path,
    )
    commands: list[list[str]] = []
    outcomes = iter([1, 0, 0])

    def runner(command, **kwargs):
        commands.append(list(command))
        return next(outcomes)

    monkeypatch.setattr(campaign_module, "_assert_lock", lambda *args: None)
    monkeypatch.setattr(
        campaign_module,
        "_failure",
        lambda root, code: {
            "returncode": code,
            "failure_kind": "infrastructure_failure",
            "error_type": "TimeoutError",
            "error": "timeout",
            "interrupted_with_sampling_checkpoints": True,
        },
    )
    monkeypatch.setattr(
        campaign_module,
        "_failed_checkpoint_overhead",
        lambda root: {
            "failed_sampling_jobs": 1,
            "logical_calls": 2,
            "successful_logical_calls": 1,
            "failed_logical_calls": 1,
            "application_attempts": 5,
            "provider_retries": 3,
            "environment_actions": 2,
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "reasoning_tokens": 1,
            "wall_time_ms_episode_sum": 9,
            "measurement_complete": True,
            "evidence": [{"path": "failed.json", "sha256": "a" * 64}],
        },
    )
    monkeypatch.setattr(
        campaign_module,
        "_sampling_resume_source",
        lambda root: root,
    )
    monkeypatch.setattr(
        campaign_module,
        "_validate_train",
        lambda root, seed: (
            SimpleNamespace(digest="f" * 64),
            {"training_cost": {
                "unique_train_tasks": 1,
                "sampling_episodes": {
                    "episodes": 6,
                    "environment_actions": 10,
                    "target_llm_calls": 6,
                    "target_prompt_tokens": 60,
                    "target_completion_tokens": 30,
                    "target_reasoning_tokens": 12,
                    "wall_time_ms_episode_sum": 100,
                },
                "usage": {
                    "target": {
                        "calls": 6,
                        "prompt_tokens": 60,
                        "completion_tokens": 30,
                        "reasoning_tokens": 12,
                    },
                    "evolution": {
                        "calls": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "reasoning_tokens": 0,
                    },
                    "embedding_calls": 4,
                    "per_stage": {"sampling": {
                        "calls": 6,
                        "prompt_tokens": 60,
                        "completion_tokens": 30,
                        "reasoning_tokens": 12,
                    }},
                },
                "provider_retries": 1,
                    "provider_evidence": {
                        "logical_calls": 6,
                        "application_attempts": 7,
                    },
                    "method_metrics": {
                        "sampling_episode_count": 6,
                        "valid_trajectory_count": 5,
                        "graph_nodes": 4,
                        "graph_edges": 3,
                        "skill_count": 2,
                        "golden_segment_count": 1,
                    },
                    "api_cost": None,
                    "api_cost_unpriced": True,
                }},
        ),
    )
    monkeypatch.setattr(
        campaign_module,
        "_validate_test",
        lambda *args, **kwargs: {
            "effectiveness": {
                "official_rate": 0.5,
                "task_contract_rate": 0.5,
                "strict_rate": 0.5,
            },
            "test_cost": {"retrieval_count": 1, "retrieved_skill_count": 1},
        },
    )

    result = campaign_module._run_lane(
        spec,
        seed=42,
        lane_root=tmp_path / "lane",
        campaign_lock=tmp_path / "campaign_lock.json",
        lock_digest="a" * 64,
        runner=runner,
    )

    assert result["passed"] is True
    assert len(result["train_attempts"]) == 2
    assert "--resume-source-run" not in commands[0]
    assert commands[1][commands[1].index("--resume-source-run") + 1] == str(
        tmp_path / "lane" / "train"
    )
    assert "--supervision" not in commands[2]
    assert "--source-run" in commands[2]
    assert result["committed_training_cost"]["usage"]["target"]["calls"] == 6
    assert result["failed_attempt_paid_cost"]["application_attempts"] == 5
    authoritative = result["training_cost"]
    assert authoritative["usage"]["target"] == {
        "calls": 7,
        "prompt_tokens": 67,
        "completion_tokens": 33,
        "reasoning_tokens": 13,
    }
    assert authoritative["sampling_episodes"]["environment_actions"] == 12
    assert authoritative["sampling_episodes"]["episodes"] == 6
    assert authoritative["sampling_episodes"]["failed_sampling_jobs"] == 1
    assert authoritative["logical_provider_calls"] == 8
    assert authoritative["application_provider_attempts"] == 12
    assert authoritative["provider_retries"] == 4

    campaign_cost = campaign_module._campaign_training_cost_accounting([result])
    totals = campaign_cost["authoritative_paid_training_totals"]
    assert totals["usage"]["target"]["calls"] == 7
    assert totals["sampling_episodes"]["environment_actions"] == 12
    assert campaign_cost["failed_attempt_paid_cost_totals"][
        "application_attempts"
    ] == 5


def _cost_checkpoint_row(
    job_index: int,
    *,
    run_id: str = "attempt_1",
    failure_kind: str = "",
) -> dict:
    manifest_index, sample_idx = divmod(job_index, 6)
    task_id = f"task_{manifest_index}"
    logical_task_id = f"{task_id}::sample_{sample_idx}"
    return {
        "job_index": job_index,
        "task_id": task_id,
        "manifest_index": manifest_index,
        "sample_idx": sample_idx,
        "sample_seed": 77 + job_index,
        "failure_kind": failure_kind,
        "actions": [{
            "episode_task_id": logical_task_id,
            "step_index": 0,
            "command_turn_index": 0,
            "action": "look",
            "env_feedback": "room",
            "reward": 0.0,
            "done": False,
            "won": False,
            "admissible_before": True,
        }],
        "environment_actions": 1,
        "wall_time_ms": 17,
        "provider_calls": [{
            "call_id": f"call_{run_id}_{job_index}",
            "status": "succeeded",
            "method": "b4_skillgen_s",
            "phase": "train",
            "run_id": run_id,
            "run_seed": 42,
            "episode_task_id": logical_task_id,
            "role": "target",
            "stage": "sampling",
            "model": "deepseek-v4-flash",
            "reasoning_effort": "high",
            "sample_seed": 77 + job_index,
            "requested_retry_limit": 5,
            "application_attempts": 1,
            "sdk_boundary_attempts": 1,
            "prompt_tokens": 11,
            "completion_tokens": 4,
            "reasoning_tokens": 2,
            "total_tokens": 15,
        }],
    }


def _write_cost_attempt(
    root: Path,
    rows: list[dict],
    *,
    state: str,
) -> None:
    jobs = root / "train" / "sampling" / "jobs"
    jobs.mkdir(parents=True)
    for row in rows:
        (jobs / f"{int(row['job_index']):04d}.json").write_text(
            json.dumps(row), encoding="utf-8"
        )
    (root / "run_state.json").write_text(
        json.dumps({"state": state, "failure_kind": "infrastructure_failure"}),
        encoding="utf-8",
    )


def test_terminal_extraction_failure_charges_all_successful_sampling_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = CampaignSpec(
        train_manifest=tmp_path / "train.json",
        test_manifest=tmp_path / "test.json",
        supervision=tmp_path / "labels.jsonl",
        config=tmp_path / "config.yaml",
        output_dir=tmp_path / "campaign",
        python=tmp_path / "python",
        repo_root=tmp_path,
    )

    def runner(command, **kwargs):
        output = Path(command[command.index("--output-dir") + 1])
        _write_cost_attempt(
            output,
            [_cost_checkpoint_row(0), _cost_checkpoint_row(1)],
            state="completed",
        )
        return 0

    monkeypatch.setattr(campaign_module, "_assert_lock", lambda *args: None)
    monkeypatch.setattr(
        campaign_module,
        "_validate_train",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("extraction failed")),
    )

    result = campaign_module._run_lane(
        spec,
        seed=42,
        lane_root=tmp_path / "lane",
        campaign_lock=tmp_path / "campaign_lock.json",
        lock_digest="a" * 64,
        runner=runner,
    )

    assert result["passed"] is False
    assert result["phase"] == "controller"
    assert result["train_attempts"][0]["status"] == "completed"
    paid = result["failed_attempt_paid_cost"]
    assert paid["measurement_complete"] is True
    assert paid["charged_sampling_jobs"] == 2
    assert paid["successful_uncommitted_sampling_jobs"] == 2
    assert paid["logical_calls"] == 2
    assert paid["application_attempts"] == 2
    assert paid["environment_actions"] == 2
    assert paid["prompt_tokens"] == 22
    assert paid["completion_tokens"] == 8


def test_copied_successful_checkpoints_are_charged_once_across_attempts(
    tmp_path: Path,
) -> None:
    row = _cost_checkpoint_row(0)
    roots = [tmp_path / "attempt_1", tmp_path / "attempt_2"]
    for root in roots:
        _write_cost_attempt(root, [row], state="failed")
    attempts = [
        {
            "attempt": index,
            "status": "failed",
            "output_dir": str(root),
            "failed_job_overhead": campaign_module._failed_checkpoint_overhead(root),
        }
        for index, root in enumerate(roots, start=1)
    ]

    paid = campaign_module._aggregate_failed_attempt_paid_cost(
        attempts, include_successful_checkpoints=True
    )

    assert paid["charged_sampling_jobs"] == 1
    assert paid["successful_uncommitted_sampling_jobs"] == 1
    assert paid["logical_calls"] == 1
    assert paid["application_attempts"] == 1
    assert paid["environment_actions"] == 1


@pytest.mark.parametrize("field,value", [("prompt_tokens", -1), ("sdk_boundary_attempts", 2)])
def test_checkpoint_cost_rejects_invalid_provider_usage(
    tmp_path: Path,
    field: str,
    value: int,
) -> None:
    row = _cost_checkpoint_row(0, failure_kind="infrastructure_failure")
    row["provider_calls"][0][field] = value
    root = tmp_path / "attempt"
    _write_cost_attempt(root, [row], state="failed")

    with pytest.raises(ValueError, match="provider usage|invalid prompt_tokens"):
        campaign_module._failed_checkpoint_overhead(root)


def test_failed_checkpoint_paid_cost_uses_persisted_provider_and_action_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "attempt"
    jobs = root / "train" / "sampling" / "jobs"
    jobs.mkdir(parents=True)
    (root / "run_state.json").write_text(
        json.dumps({"state": "failed", "failure_kind": "infrastructure_failure"}),
        encoding="utf-8",
    )
    (jobs / "0000.json").write_text(json.dumps({
        "job_index": 0,
        "task_id": "task_0",
        "manifest_index": 0,
        "sample_idx": 0,
        "sample_seed": 77,
        "failure_kind": "infrastructure_failure",
        "actions": [
            {
                "episode_task_id": "task_0::sample_0",
                "step_index": 0,
                "command_turn_index": 0,
                "action": "look",
                "env_feedback": "room",
                "reward": 0.0,
                "done": False,
                "won": False,
                "admissible_before": True,
            },
            {
                "episode_task_id": "task_0::sample_0",
                "step_index": 1,
                "command_turn_index": 1,
                "action": "take apple",
                "env_feedback": "taken",
                "reward": 0.0,
                "done": False,
                "won": False,
                "admissible_before": True,
            },
        ],
        "wall_time_ms": 17,
        "provider_calls": [
            {
                "call_id": "call_success",
                "status": "succeeded",
                "method": "b4_skillgen_s",
                "phase": "train",
                "run_id": "attempt_1",
                "run_seed": 42,
                "episode_task_id": "task_0::sample_0",
                "role": "target",
                "stage": "sampling",
                "model": "deepseek-v4-flash",
                "reasoning_effort": "high",
                "sample_seed": 77,
                "requested_retry_limit": 5,
                "application_attempts": 1,
                "sdk_boundary_attempts": 1,
                "prompt_tokens": 11,
                "completion_tokens": 4,
                "reasoning_tokens": 2,
                "total_tokens": 15,
            },
            {
                "call_id": "call_failed",
                "status": "failed",
                "method": "b4_skillgen_s",
                "phase": "train",
                "run_id": "attempt_1",
                "run_seed": 42,
                "episode_task_id": "task_0::sample_0",
                "role": "target",
                "stage": "sampling",
                "model": "deepseek-v4-flash",
                "reasoning_effort": "high",
                "sample_seed": 77,
                "requested_retry_limit": 5,
                "application_attempts": 5,
                "sdk_boundary_attempts": 5,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": None,
                "total_tokens": 0,
            },
        ],
    }), encoding="utf-8")

    cost = campaign_module._failed_checkpoint_overhead(root)

    assert cost["failed_sampling_jobs"] == 1
    assert cost["environment_actions"] == 2
    assert cost["logical_calls"] == 2
    assert cost["successful_logical_calls"] == 1
    assert cost["failed_logical_calls"] == 1
    assert cost["application_attempts"] == 6
    assert cost["provider_retries"] == 4
    assert cost["prompt_tokens"] == 11
    assert cost["completion_tokens"] == 4
    assert cost["reasoning_tokens"] == 2
    assert cost["wall_time_ms_episode_sum"] == 17
    assert cost["measurement_complete"] is True
    assert len(cost["evidence"]) == 1


def test_b4_campaign_uses_fresh_boundary_without_sampling_checkpoint(
    tmp_path: Path,
) -> None:
    interrupted = tmp_path / "interrupted"
    jobs = interrupted / "train" / "sampling" / "jobs"
    jobs.mkdir(parents=True)

    assert campaign_module._sampling_resume_source(interrupted) is None

    (jobs / "0000.json").write_text("{}\n", encoding="utf-8")
    assert campaign_module._sampling_resume_source(interrupted) == interrupted


def test_b4_campaign_does_not_retry_protocol_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = CampaignSpec(
        train_manifest=tmp_path / "train.json",
        test_manifest=tmp_path / "test.json",
        supervision=tmp_path / "labels.jsonl",
        config=tmp_path / "config.yaml",
        output_dir=tmp_path / "campaign",
        python=tmp_path / "python",
        repo_root=tmp_path,
    )
    calls = 0

    def runner(command, **kwargs):
        nonlocal calls
        calls += 1
        return 1

    monkeypatch.setattr(campaign_module, "_assert_lock", lambda *args: None)
    monkeypatch.setattr(
        campaign_module,
        "_failure",
        lambda root, code: {
            "returncode": code,
            "failure_kind": "protocol_failure",
            "error_type": "ValueError",
            "error": "invalid output",
            "interrupted_with_sampling_checkpoints": False,
        },
    )
    monkeypatch.setattr(
        campaign_module,
        "_failed_checkpoint_overhead",
        lambda root: {},
    )

    result = campaign_module._run_lane(
        spec,
        seed=42,
        lane_root=tmp_path / "lane",
        campaign_lock=tmp_path / "campaign_lock.json",
        lock_digest="a" * 64,
        runner=runner,
    )

    assert result["passed"] is False
    assert result["failure_kind"] == "protocol_failure"
    assert calls == 1
    assert len(result["train_attempts"]) == 1


def test_b4_run_manifest_exposes_runtime_and_protocol_authorities(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    model = ModelConfig.from_mapping({
        "provider": "openai_compatible",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
        "api_key_env": "MODEL_API_KEY",
        "reasoning_effort": "high",
    })
    ctx = SimpleNamespace(
        output_dir=output,
        run_id="b4_manifest_test",
        run_seed=42,
        code_hash="b" * 64,
        identity={"config_digest": "c" * 64},
        campaign={"campaign_provider_max_inflight": 16},
        resume=None,
        model_config=model,
        max_environment_actions=100,
        train_manifest_path=tmp_path / "train.json",
        test_manifest_path=tmp_path / "test.json",
    )
    train = SimpleNamespace(digest="1" * 64, tasks=[1] * 120)
    test = SimpleNamespace(digest="2" * 64, tasks=[1] * 134)
    runtime = {
        "python_major_minor": "3.9",
        "alfworld_distribution_version": "0.4.2",
        "skillopt_distribution_version": None,
        "atomic_skillgraph_distribution_version": None,
    }
    controller_module._write_run_identity(
        ctx=ctx,
        lock={
            "skillgen": {
                "repo": "https://github.com/ruomengd/SkillGen",
                "commit": "8" * 40,
                "runtime_tree": {"sha256": "9" * 64},
            }
        },
        config={
            "experiment_kind": "formal",
            "sampling": {},
            "extraction": {},
            "inference": {},
            "parallel": {
                "seed_lanes": 1,
                "episode_workers_per_seed": 16,
                "test_workers_per_seed": 16,
                "campaign_provider_max_inflight": 16,
            },
        },
        phase="train",
        train=train,
        test=test,
        receipts={"train": {}, "test": {}},
        git_state={"commit": "7" * 40, "dirty": False},
        data_signature={"digest": "d" * 64},
        python_runtime=runtime,
    )

    manifest = json.loads(
        (output / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["controller_commit"] == "7" * 40
    assert manifest["model"] == "deepseek-v4-flash"
    assert manifest["model_identity"] == model.to_wire()
    assert manifest["alfworld_package_version"] == "0.4.2"
    assert manifest["episode_workers"] == 16
    assert manifest["provider_max_inflight"] == 16
    assert manifest["method_specific_prior"] is False
    assert manifest["extra_train_supervision"] is True
    assert manifest["weak_supervision"] == "subgoal_progress"
    assert manifest["python_runtime"] == runtime

    frozen = SimpleNamespace(
        digest="f" * 64,
        source_train_manifest_hash=train.digest,
        source_validation_manifest_hash=None,
    )
    controller_module._bind_frozen_digest(ctx, frozen)
    controller_module._bind_frozen_digest(ctx, frozen)
    manifest = json.loads(
        (output / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["frozen_artifact_digest"] == frozen.digest
    assert manifest["identity"]["frozen_artifact_digest"] == frozen.digest
    assert ctx.identity["frozen_artifact_digest"] == frozen.digest


def _reporting_lane(seed: int) -> dict:
    offset = seed - 42
    rate = 0.4 + 0.1 * offset
    family = {
        name: {
            "tasks": 20 + index,
            "official_success": 8 + offset,
            "official_rate": rate,
            "task_contract_scored_tasks": 20 + index,
            "task_contract_success": 7 + offset,
            "task_contract_rate": rate - 0.05,
            "strict_scored_tasks": 20 + index,
            "strict_success": 7 + offset,
            "strict_rate": rate - 0.05,
        }
        for index, name in enumerate(campaign_module.ALFWORLD_FORMAL_TASK_TYPES)
    }
    target_calls = 720 + offset
    authoritative = {
        "measurement_complete": True,
        "unique_train_tasks": 120,
        "sampling_episodes": {
            "episodes": 720,
            "attempted_sampling_jobs": 720 + offset,
            "environment_actions": 3600 + offset,
            "wall_time_ms_episode_sum": 10000 + offset,
        },
        "usage": {
            "target": {
                "calls": target_calls,
                "prompt_tokens": 7200 + offset,
                "completion_tokens": 1800 + offset,
                "reasoning_tokens": 900 + offset,
            },
            "evolution": {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
            },
            "embedding_calls": 12 + offset,
        },
        "provider_retries": offset,
        "logical_provider_calls": target_calls,
        "application_provider_attempts": target_calls + offset,
        "api_cost": 10.0 + offset,
        "api_cost_unpriced": False,
    }
    return {
        "seed": seed,
        "passed": True,
        "train_root": f"/train/{seed}",
        "test_root": f"/test/{seed}",
        "frozen_digest": str(offset + 1) * 64,
        "authoritative_training_cost": authoritative,
        "training_wall_time_ms": 20000 + offset,
        "test_wall_time_ms": 30000 + offset,
        "effectiveness": {
            "tasks": 134,
            "attempted_tasks": 134,
            "official_success": 50 + offset,
            "official_rate": rate,
            "micro_average_official_rate": rate,
            "task_contract_rate": rate - 0.05,
            "strict_rate": rate - 0.05,
            "macro_family_official_rate": rate,
            "environment_actions": 1000 + offset,
            "actions_per_task": (1000 + offset) / 134,
            "actions_per_solved": (1000 + offset) / (50 + offset),
            "p50_actions": 7 + offset,
            "p90_actions": 15 + offset,
            "target_llm_calls": 200 + offset,
            "evolution_llm_calls": 0,
            "llm_calls": 200 + offset,
            "calls_per_task": (200 + offset) / 134,
            "p50_calls": 1 + offset,
            "p90_calls": 3 + offset,
            "p50_tokens": 100 + offset,
            "p90_tokens": 300 + offset,
            "latency_per_task_ms": 200.0 + offset,
            "p50_latency_ms": 100 + offset,
            "p90_latency_ms": 500 + offset,
            "family": family,
        },
        "test_cost": {
            "episodes": {
                "episodes": 134,
                "environment_actions": 1000 + offset,
                "wall_time_ms_episode_sum": 26000 + offset,
            },
            "usage": {
                "target": {
                    "calls": 200 + offset,
                    "prompt_tokens": 4000 + offset,
                    "completion_tokens": 1000 + offset,
                    "reasoning_tokens": 500 + offset,
                },
                "evolution": {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "reasoning_tokens": 0,
                },
                "embedding_calls": 268 + offset,
                "api_cost": 2.0 + offset,
                "api_cost_unpriced": False,
            },
            "provider_retries": offset,
        },
        "method_specific_metrics": {
            "sampling_episode_count": 720,
            "valid_trajectory_count": 600 + offset,
            "graph_nodes": 40 + offset,
            "graph_edges": 30 + offset,
            "skill_count": 20 + offset,
            "golden_segment_count": 10 + offset,
            "retrieval_count": 134,
            "retrieved_skill_count": 268 + offset,
        },
    }


def test_b4_campaign_summary_exports_frozen_reporting_contract() -> None:
    lanes = [_reporting_lane(seed) for seed in (42, 43, 44)]

    summary = campaign_module._campaign_summary(lanes)

    assert summary["paper_eligible"] is True
    official = summary["effectiveness"]["mean_std"]["official_rate"]
    assert official == {
        "mean": pytest.approx(0.5),
        "std": pytest.approx(0.1),
        "std_ddof": 1,
        "values_by_seed": {"42": 0.4, "43": 0.5, "44": pytest.approx(0.6)},
    }
    family = campaign_module.ALFWORLD_FORMAL_TASK_TYPES[0]
    assert summary["effectiveness"]["per_family_mean_std"][family][
        "official_rate"
    ]["std_ddof"] == 1
    training = summary["training_cost"]
    assert training["totals"]["unique_train_tasks"] == 360
    assert training["totals"]["train_episode_count"] == 2163
    assert training["mean_std"]["target_llm_calls"]["values_by_seed"] == {
        "42": 720,
        "43": 721,
        "44": 722,
    }
    testing = summary["test_cost"]
    assert testing["totals"]["test_episode_count"] == 402
    assert testing["totals"]["api_cost"] == pytest.approx(9.0)
    method = summary["method_specific_metrics"]
    assert set(method["by_seed"]["42"]) == set(
        campaign_module._SKILLGEN_METRIC_FIELDS
    )
    assert method["mean_std"]["valid_trajectory_count"]["mean"] == 601.0
    assert summary["statistical_definitions"]["std_ddof"] == 1
    assert "not added twice" in summary["statistical_definitions"]["tokens"]


def test_b4_campaign_summary_rejects_incomplete_seed_set() -> None:
    with pytest.raises(ValueError, match="exactly seeds 42, 43, and 44"):
        campaign_module._campaign_summary([
            _reporting_lane(42),
            _reporting_lane(43),
        ])


def test_b4_campaign_persists_complete_summary_and_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = CampaignSpec(
        train_manifest=tmp_path / "train.json",
        test_manifest=tmp_path / "test.json",
        supervision=tmp_path / "labels.jsonl",
        config=tmp_path / "config.yaml",
        output_dir=tmp_path / "campaign",
        python=tmp_path / "python",
        repo_root=tmp_path,
    )
    source = {"commit": "a" * 40, "dirty": False, "code_digest": "b" * 64}
    monkeypatch.setattr(
        campaign_module,
        "build_campaign_lock",
        lambda *args, **kwargs: {"campaign_id": "b4_reporting_test"},
    )
    monkeypatch.setattr(
        campaign_module,
        "_run_probe_with_fallback",
        lambda spec, payload, *, runner: (spec, dict(payload), {"passed": True}),
    )
    monkeypatch.setattr(
        campaign_module,
        "_run_seed_lanes",
        lambda *args, **kwargs: [_reporting_lane(seed) for seed in (42, 43, 44)],
    )

    report = campaign_module.run_campaign(
        spec,
        runner=lambda *args, **kwargs: 0,
        source_inspector=lambda root: dict(source),
    )

    assert report["passed"] is True
    assert report["paper_eligible"] is True
    assert report["test_metrics_mean_std"]["official_rate"]["std_ddof"] == 1
    assert set(report["test_per_family_mean_std"]) == set(
        campaign_module.ALFWORLD_FORMAL_TASK_TYPES
    )
    assert report["training_cost_summary"]["totals"]["unique_train_tasks"] == 360
    assert report["test_cost_summary"]["totals"]["test_episode_count"] == 402
    assert report["method_specific_metrics"]["totals"][
        "sampling_episode_count"
    ] == 2160
    summary_path = spec.output_dir / "campaign_summary.json"
    assert summary_path.is_file()
    persisted = json.loads(summary_path.read_text(encoding="utf-8"))
    assert persisted["statistical_definitions"]["std_ddof"] == 1
    assert (spec.output_dir / "campaign_report.json").is_file()
    assert (spec.output_dir / "completion.json").is_file()


def test_b4_posthoc_evaluation_uses_common_success_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(["pick_and_place_simple-trial"])
    task = manifest.tasks[0]
    episode = CommonEpisodeRecord(
        method="b4_skillgen_s",
        phase="test",
        run_seed=42,
        task_id=task.task_id,
        task_type=task.task_type,
        manifest_index=task.index,
        gamefile=task.gamefile_rel,
        gamefile_hash=task.gamefile_sha256,
        official_success=True,
    )

    class Evaluator:
        def __init__(self, root: Path) -> None:
            assert root == tmp_path

        def evaluate(self, *args, **kwargs):
            return SimpleNamespace(
                task_contract_success=False,
                strict_success=False,
                invalid_actions=2,
            )

    monkeypatch.setattr(driver_module, "StrictTaskEvaluator", Evaluator)
    monkeypatch.setattr(
        driver_module,
        "_load_actions",
        lambda *args, **kwargs: {task.task_id: ["look"]},
    )

    result = driver_module._apply_strict_evaluation(
        [episode],
        manifest,
        SimpleNamespace(alfworld_data=tmp_path),
        phase_root=tmp_path / "phase",
    )

    assert result == [episode]
    assert episode.contract_consistency is False
    assert episode.common_strict_success is False
    assert episode.task_contract_success is False
    assert episode.strict_success is False
    assert episode.invalid_actions == 2


def test_b4_campaign_train_validation_requires_manifest_frozen_binding(
    tmp_path: Path,
) -> None:
    root = tmp_path / "train_run"
    asset = tmp_path / "graph.json"
    asset.write_text("{}\n", encoding="utf-8")
    frozen = freeze_files(
        method_id="b4_skillgen_s",
        source_files={"graph.json": asset},
        destination=root / "frozen",
        source_train_manifest_hash="a" * 64,
        source_validation_manifest_hash=None,
        metadata={"run_seed": 42},
    )
    identity = {"frozen_artifact_digest": frozen.digest}
    root.mkdir(exist_ok=True)
    (root / "completion.json").write_text(
        json.dumps({"passed": True, "phase": "train", "identity": identity}),
        encoding="utf-8",
    )
    (root / "report.json").write_text(
        json.dumps({
            "passed": True,
            "method": "b4_skillgen_s",
            "frozen": {"digest": frozen.digest},
        }),
        encoding="utf-8",
    )
    manifest_path = root / "run_manifest.json"
    manifest_path.write_text(
        json.dumps({
            "run_seed": 42,
            "frozen_artifact_digest": frozen.digest,
            "identity": identity,
        }),
        encoding="utf-8",
    )

    loaded, _ = campaign_module._validate_train(root, 42)
    assert loaded.digest == frozen.digest

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frozen_artifact_digest"] = "b" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="frozen artifact identity"):
        campaign_module._validate_train(root, 42)
