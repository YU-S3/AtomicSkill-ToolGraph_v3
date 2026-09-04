"""Deterministic B3 SkillOpt trainer chain test.

Runs the REAL upstream ReflACT trainer pipeline — rollout → reflect →
aggregate → select → update → gate → best_skill — with two deterministic
seams only:

1. the episode runner is a scripted fixture (no ALFWorld simulator, no
   network target model);
2. the upstream ``openai_compatible`` backend points at an in-process fake
   ``/chat/completions`` server that returns the scripted analyst patch.

This proves the common-manifest adapter satisfies the upstream EnvAdapter
contract and that the frozen paper-aligned config drives the full chain.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from experiments.baselines.b3_skillopt.common_alfworld_adapter import (
    CommonALFWorldSkillOptAdapter,
)
from experiments.baselines.b3_skillopt.episode_runner import SkillOptTextEpisodeRunner
from experiments.baselines.common.manifest import ManifestTask, TaskManifestSet

skillopt = pytest.importorskip("skillopt")

_APPENDED_GUIDANCE = "When the target object is on a table, take it from there first."

_SCRIPTED_ANALYST = {
    "patch": {
        "reasoning": "scripted deterministic analyst response",
        "edits": [{"op": "append", "content": _APPENDED_GUIDANCE}],
    },
}


class _FakeOpenAICompatibleHandler(BaseHTTPRequestHandler):
    server_version = "fake_openai_compatible"

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        request = json.loads(body.decode("utf-8"))
        self.server.requests.append(request)
        payload = {
            "id": "fake",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(_SCRIPTED_ANALYST),
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            },
        }
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *args) -> None:  # silence request logging
        pass


class _FakeServer:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.httpd = HTTPServer(("127.0.0.1", 0), _FakeOpenAICompatibleHandler)
        self.httpd.requests = self.requests
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def _fake_episode_fn(task, skill_content, out_dir):
    """Scripted text-skill episode: fails until the analyst guidance lands."""

    hard = 1 if _APPENDED_GUIDANCE in str(skill_content) else 0
    out = Path(out_dir)
    (out / "predictions" / str(task["id"])).mkdir(parents=True, exist_ok=True)
    conversation = [
        {
            "step": 0,
            "action": "go to table 1",
            "reasoning": "fixture",
            "model_response": "<think>fixture</think><action>go to table 1</action>",
            "env_feedback": "You arrive at the table.",
            "reward": 0.0,
            "done": False,
        },
        {
            "step": 1,
            "action": "take object 1 from table 1",
            "reasoning": "fixture",
            "model_response": "<think>fixture</think><action>take object 1 from table 1</action>",
            "env_feedback": "You pick up object 1." if hard else "Nothing happens.",
            "reward": float(hard),
            "done": True,
        },
    ]
    (out / "predictions" / str(task["id"]) / "conversation.json").write_text(
        json.dumps(conversation, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return {
        "id": str(task["id"]),
        "hard": hard,
        "soft": float(hard),
        "n_turns": len(conversation),
        "fail_reason": "" if hard else "Timeout after 100 steps",
        "agent_ok": True,
        "task_type": str(task["task_type"]),
        "gamefile": str(task["gamefile"]),
        "task_description": "fixture task",
    }, conversation


def _manifest(tmp_path: Path, name: str, count: int) -> TaskManifestSet:
    import hashlib

    tasks = tuple(
        ManifestTask(
            index=index,
            task_id=f"{name}_{index}",
            task_type="pick_and_place_simple",
            source_split="train" if name == "train" else "valid_seen",
            env_index=index,
            gamefile_rel=(
                f"json_2.1.1/train/game_{index}.tw-pddl"
                if name == "train"
                else f"json_2.1.1/valid_seen/game_{index}.tw-pddl"
            ),
            gamefile_sha256="a" * 64,
            task_signature=hashlib.sha256(f"{name}:{index}".encode()).hexdigest(),
        )
        for index in range(count)
    )
    manifest = TaskManifestSet.create(
        manifest_id=name,
        benchmark="alfworld",
        source_split="train" if name == "train" else "valid_seen",
        seed=42,
        tasks=tasks,
    )
    return manifest.save(tmp_path / f"{name}.json")


def test_reference_guard_returns_empty(tmp_path) -> None:
    adapter = _adapter(tmp_path)
    assert adapter.build_reference_text({}) == ""
    assert adapter.get_reference_metadata({}) == {"fields": [], "preview": ""}


def test_full_trainer_chain(tmp_path) -> None:
    server = _FakeServer()
    try:
        import skillopt.model as skillopt_model

        skillopt_model.set_backend("openai_compatible")
        skillopt_model.configure_openai_compatible(
            base_url=server.base_url,
            api_key="fixture-key-not-a-secret",
            model="deepseek-v4-flash",
            max_tokens=16384,
            timeout_seconds=30,
        )
        skill_init = tmp_path / "initial.md"
        skill_init.write_text("# Fixture Skill\nInitial skill text.\n", encoding="utf-8")
        out_root = tmp_path / "train_out"
        adapter = CommonALFWorldSkillOptAdapter(
            train_manifest_path=_manifest(tmp_path, "train", 2),
            validation_manifest_path=_manifest(tmp_path, "validation", 1),
            alfworld_data="",
            max_steps=100,
            minibatch_size=2,
            edit_budget=4,
            seed=42,
            phase="train",
            episode_runner=SkillOptTextEpisodeRunner(
                max_actions=100, seed=42, episode_fn=_fake_episode_fn,
            ),
        )
        cfg = {
            "out_root": str(out_root),
            "env": "alfworld",
            "model_backend": "openai_compatible",
            "optimizer_backend": "openai_compatible",
            "target_backend": "openai_compatible",
            "optimizer_model": "deepseek-v4-flash",
            "target_model": "deepseek-v4-flash",
            "reasoning_effort": "",
            "skill_init": str(skill_init),
            "batch_size": 2,
            "num_epochs": 1,
            "accumulation": 1,
            "seed": 42,
            "merge_batch_size": 2,
            "analyst_workers": 1,
            "max_analyst_rounds": 3,
            "failure_only": False,
            "minibatch_size": 2,
            "edit_budget": 4,
            "min_edit_budget": 2,
            "lr_scheduler": "constant",
            "lr_control_mode": "fixed",
            "skill_update_mode": "patch",
            "longitudinal_pair_policy": "mixed",
            "use_slow_update": False,
            "slow_update_samples": 20,
            "slow_update_gate_with_selection": False,
            "use_meta_skill": False,
            "use_skill_aware_reflection": False,
            "skill_aware_appendix_source": "both",
            "skill_aware_consolidate_threshold": 0,
            "use_gate": True,
            "gate_metric": "hard",
            "gate_mixed_weight": 0.5,
            "use_semantic_density": False,
            "semantic_density_weight": 0.05,
            "leading_words": None,
            "sel_env_num": 1,
            "test_env_num": 0,
            "eval_test": False,
            "rewrite_reasoning_effort": "",
            "rewrite_max_completion_tokens": 64000,
            "train_size": 2,
        }
        from skillopt.engine.trainer import ReflACTTrainer

        summary = ReflACTTrainer(cfg, adapter).train()
    finally:
        server.close()

    # 1. The chain executed: rollout, analyst, apply, selection gate.
    assert summary["total_steps"] == 1
    assert summary["best_selection_hard"] >= 1.0
    assert summary["total_accepts"] >= 1
    # 2. best_skill.md carries the applied edit.
    best_skill = (out_root / "best_skill.md").read_text(encoding="utf-8")
    assert _APPENDED_GUIDANCE in best_skill
    # 3. History records the accept and its selection score.
    history = json.loads((out_root / "history.json").read_text(encoding="utf-8"))
    assert history[0]["action"] in {"accept", "accept_new_best"}
    assert history[0]["selection_hard"] >= 1.0
    # 4. The only upstream LLM stage exercised is the failure analyst.
    stages = {request.get("model") for request in server.requests}
    assert stages == {"deepseek-v4-flash"}
    from skillopt.model import get_token_summary

    token_summary = get_token_summary()
    assert token_summary.get("analyst", {}).get("calls", 0) == 1
    # 5. Sidecar episodes exist for train + validation phases.
    episodes_by_phase: dict[str, int] = {}
    for sidecar in sorted(out_root.rglob("common_episodes.jsonl")):
        for line in sidecar.read_text(encoding="utf-8").splitlines():
            episode = json.loads(line)
            episodes_by_phase[episode["phase"]] = episodes_by_phase.get(episode["phase"], 0) + 1
    assert episodes_by_phase.get("train") == 2
    assert episodes_by_phase.get("validation") == 2  # baseline + candidate gate eval
    # 6. No api key ever landed on disk.
    content = "".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in out_root.rglob("*") if path.is_file()
    )
    assert "fixture-key-not-a-secret" not in content


def test_loader_and_adapter_contract(tmp_path) -> None:
    train_manifest = TaskManifestSet.load(_manifest(tmp_path, "train", 2))
    validation_manifest = TaskManifestSet.load(_manifest(tmp_path, "validation", 1))
    adapter = _adapter(tmp_path)
    loader = adapter.get_dataloader()
    assert loader.get_train_size() == 2
    assert len(train_manifest.tasks) == 2
    assert len(validation_manifest.tasks) == 1
    batch = loader.build_train_batch(batch_size=2, seed=42)
    assert batch.batch_size == 2
    env = adapter.build_env_from_batch(batch)
    assert len(env) == 2
    rows = adapter.rollout(env, "# Skill\n", str(tmp_path / "rollout"))
    assert len(rows) == 2
    results_path = tmp_path / "rollout" / "results.jsonl"
    assert results_path.is_file()
    # resume semantics: second call returns the persisted rows
    again = adapter.rollout(env, "# Skill\n", str(tmp_path / "rollout"))
    assert [row["id"] for row in again] == [row["id"] for row in rows]


def _adapter(tmp_path: Path) -> CommonALFWorldSkillOptAdapter:
    return CommonALFWorldSkillOptAdapter(
        train_manifest_path=_manifest(tmp_path, "train", 2),
        validation_manifest_path=_manifest(tmp_path, "validation", 1),
        alfworld_data="",
        max_steps=100,
        seed=42,
        phase="train",
        episode_runner=SkillOptTextEpisodeRunner(
            max_actions=100, seed=42, episode_fn=_fake_episode_fn,
        ),
    )
