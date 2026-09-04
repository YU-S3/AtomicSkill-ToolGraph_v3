"""SkillOpt baseline worker.

Runs in the dedicated per-method venv (``.venv_b3_skillopt``).  The upstream
SkillOpt trainer / reflection / optimizer / gate run unchanged; this worker
only:

- configures the upstream ``openai_compatible`` backend with the frozen
  common model (base_url/model from the wire, key from the environment);
- verifies the pinned ``initial.md`` seed against ``baseline_lock.yaml``;
- instantiates the common-manifest ``EnvAdapter``;
- runs the upstream trainer (train phase) or the frozen held-out rollout
  (test phase), and mirrors everything into Common sidecars.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import yaml

from experiments.baselines.bootstrap_external import load_lock
from experiments.baselines.common.artifact_digest import digest_directory
from experiments.baselines.common.manifest import TaskManifestSet
from experiments.baselines.common.model_config import ModelConfig
from experiments.baselines.common.subprocess_worker import WorkerWire
from experiments.baselines.common.usage import (
    UsageSnapshot,
    usage_from_skillopt_token_summary,
)

from .common_alfworld_adapter import CommonALFWorldSkillOptAdapter

REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKER_RESULT_NAME = "worker_result.json"


def _load_method_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("b3_skillopt config root must be a mapping")
    return payload


def _require_file(path: str | Path, what: str) -> Path:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{what} is missing: {resolved}")
    return resolved


def _verify_skill_init(
    *,
    external_root: str | Path,
    skill_init_rel: str | None,
    destination_dir: Path,
    lock: dict[str, Any],
) -> str:
    """Copy the pinned upstream ALFWorld seed into the run and verify its hash."""

    relative = str(skill_init_rel or "skillopt/envs/alfworld/skills/initial.md")
    source = Path(external_root) / relative
    _require_file(source, "upstream SkillOpt initial.md")
    expected = str((lock.get("skillopt") or {}).get("key_files", {}).get(
        relative, "",
    ))
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if not expected or actual != expected:
        raise RuntimeError(
            f"upstream initial.md hash mismatch: expected {expected or '<missing>'}, "
            f"got {actual}"
        )
    destination_dir.mkdir(parents=True, exist_ok=True)
    target = destination_dir / "initial.md"
    shutil.copyfile(source, target)
    return str(target)


def _flat_train_cfg(
    *,
    config: dict[str, Any],
    model: ModelConfig,
    run_seed: int,
    out_root: Path,
    skill_init_path: str,
    train_size: int,
    selection_size: int,
) -> dict[str, Any]:
    """Flat ReflACTTrainer config with the paper-aligned values (§16.5)."""

    train = dict(config.get("train") or {})
    gradient = dict(config.get("gradient") or {})
    optimizer = dict(config.get("optimizer") or {})
    evaluation = dict(config.get("evaluation") or {})
    env = dict(config.get("env") or {})
    cfg = {
        "out_root": str(out_root),
        "env": "alfworld",
        "model_backend": "openai_compatible",
        "optimizer_backend": "openai_compatible",
        "target_backend": "openai_compatible",
        "optimizer_model": model.model,
        "target_model": model.model,
        "reasoning_effort": "",
        "skill_init": skill_init_path,
        "batch_size": int(train.get("batch_size", 40)),
        "num_epochs": int(train["num_epochs"]),
        "accumulation": int(train.get("accumulation", 1)),
        "seed": int(run_seed),
        "merge_batch_size": int(gradient["merge_batch_size"]),
        "analyst_workers": int(gradient.get("analyst_workers", 4)),
        "max_analyst_rounds": 3,
        "failure_only": False,
        "minibatch_size": int(gradient["minibatch_size"]),
        "edit_budget": int(optimizer["learning_rate"]),
        "min_edit_budget": 2,
        "lr_scheduler": "cosine",
        "lr_control_mode": "fixed",
        "skill_update_mode": "patch",
        "longitudinal_pair_policy": "mixed",
        "use_slow_update": bool(optimizer.get("use_slow_update", True)),
        "slow_update_samples": 20,
        "slow_update_gate_with_selection": bool(
            optimizer.get("slow_update_gate_with_selection", False)
        ),
        "use_meta_skill": bool(optimizer.get("use_meta_skill", True)),
        "use_skill_aware_reflection": bool(
            optimizer.get("use_skill_aware_reflection", False)
        ),
        "skill_aware_appendix_source": "both",
        "skill_aware_consolidate_threshold": 0,
        "use_gate": bool(evaluation.get("use_gate", True)),
        "gate_metric": "hard",
        "gate_mixed_weight": 0.5,
        "use_semantic_density": False,
        "semantic_density_weight": 0.05,
        "leading_words": None,
        "sel_env_num": int(selection_size),
        "test_env_num": 0,
        "eval_test": False,
        "rewrite_reasoning_effort": "",
        "rewrite_max_completion_tokens": 64000,
        "train_size": int(train_size),
        "max_steps": int(env.get("max_steps", 100)),
        "workers": int(env.get("workers", 1)),
        "max_api_workers": int(env.get("max_api_workers", 1)),
        "max_completion_tokens": int(env.get("max_completion_tokens", 16384)),
    }
    return cfg


def _configure_model(model: ModelConfig) -> None:
    import skillopt.model as skillopt_model

    skillopt_model.set_backend("openai_compatible")
    skillopt_model.configure_openai_compatible(
        base_url=model.base_url,
        api_key=model.require_api_key(),
        model=model.model,
        max_tokens=32768,
        timeout_seconds=300,
    )


def _collect_episodes(root: Path) -> list[Any]:
    from experiments.baselines.common.trace import load_episodes

    episodes = []
    for sidecar in sorted(root.rglob("common_episodes.jsonl")):
        episodes.extend(load_episodes(sidecar))
    return episodes


def _write_result(output_dir: Path, result: dict[str, Any], *, passed: bool) -> None:
    result.setdefault("passed", bool(passed))
    (output_dir / _WORKER_RESULT_NAME).write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_train(
    *,
    wire: WorkerWire,
    model: ModelConfig,
    config: dict[str, Any],
    lock: dict[str, Any],
) -> int:
    _require_file(wire.manifest_path or "", "train manifest")
    _require_file(wire.validation_manifest_path or "", "validation manifest")
    train_manifest = TaskManifestSet.load(wire.manifest_path)
    validation_manifest = TaskManifestSet.load(wire.validation_manifest_path)
    output_dir = Path(wire.output_dir)
    train_out = output_dir / "train"
    train_out.mkdir(parents=True, exist_ok=True)
    skill_init_path = _verify_skill_init(
        external_root=wire.external_skillopt_root or "",
        skill_init_rel=wire.skill_init_rel,
        destination_dir=train_out / "skill_init",
        lock=lock,
    )
    adapter = CommonALFWorldSkillOptAdapter(
        train_manifest_path=wire.manifest_path,
        validation_manifest_path=wire.validation_manifest_path,
        test_manifest_path=wire.test_manifest_path,
        alfworld_data=os.environ.get("ALFWORLD_DATA", ""),
        max_steps=int(dict(config.get("env") or {}).get("max_steps", 100)),
        workers=int(dict(config.get("env") or {}).get("workers", 1)),
        max_api_workers=int(dict(config.get("env") or {}).get("max_api_workers", 1)),
        analyst_workers=int(dict(config.get("gradient") or {}).get("analyst_workers", 4)),
        failure_only=False,
        minibatch_size=int(dict(config.get("gradient") or {}).get("minibatch_size", 8)),
        edit_budget=int(dict(config.get("optimizer") or {}).get("learning_rate", 4)),
        max_completion_tokens=int(
            dict(config.get("env") or {}).get("max_completion_tokens", 16384)
        ),
        seed=int(wire.run_seed),
        phase="train",
    )
    cfg = _flat_train_cfg(
        config=config,
        model=model,
        run_seed=int(wire.run_seed),
        out_root=train_out,
        skill_init_path=skill_init_path,
        train_size=len(train_manifest.tasks),
        selection_size=len(validation_manifest.tasks),
    )
    from skillopt.engine.trainer import ReflACTTrainer

    trainer = ReflACTTrainer(cfg, adapter)
    started = time.time()
    summary = trainer.train()
    from skillopt.model import get_token_summary

    usage = usage_from_skillopt_token_summary(get_token_summary())
    usage.wall_time_ms = int((time.time() - started) * 1000)
    usage.save(train_out / "usage.json")
    episodes = _collect_episodes(train_out)
    best_skill_path = train_out / "best_skill.md"
    _require_file(best_skill_path, "trained best_skill.md")
    result = {
        "phase": "train",
        "output_dir": str(output_dir),
        "train_summary": {
            key: summary.get(key)
            for key in (
                "best_step", "best_selection_hard", "final_selection_hard",
                "total_steps", "total_accepts", "total_rejects", "total_skips",
                "baseline_selection_hard", "total_wall_time_s",
            )
        },
        "usage": usage.to_dict(),
        "episodes": {
            "total": len(episodes),
            "train": sum(episode.phase == "train" for episode in episodes),
            "validation": sum(episode.phase == "validation" for episode in episodes),
        },
        "best_skill_path": str(best_skill_path),
    }
    _write_result(output_dir, result, passed=True)
    return 0


def _run_test(
    *,
    wire: WorkerWire,
    model: ModelConfig,
    config: dict[str, Any],
) -> int:
    _require_file(wire.test_manifest_path or "", "test manifest")
    frozen_dir = Path(wire.frozen_artifact_path or "")
    frozen_digest_manifest = frozen_dir / "digest.json"
    artifact_root = frozen_dir / "artifact"
    _require_file(frozen_digest_manifest, "frozen digest.json")
    _require_file(artifact_root / "best_skill.md", "frozen best_skill.md")
    test_manifest = TaskManifestSet.load(wire.test_manifest_path)
    best_skill = (artifact_root / "best_skill.md").read_text(encoding="utf-8")
    digest_before = digest_directory(artifact_root)
    adapter = CommonALFWorldSkillOptAdapter(
        train_manifest_path=wire.manifest_path or "",
        validation_manifest_path=wire.validation_manifest_path or "",
        test_manifest_path=wire.test_manifest_path,
        alfworld_data=os.environ.get("ALFWORLD_DATA", ""),
        max_steps=int(dict(config.get("env") or {}).get("max_steps", 100)),
        workers=int(dict(config.get("env") or {}).get("workers", 1)),
        max_api_workers=int(dict(config.get("env") or {}).get("max_api_workers", 1)),
        analyst_workers=int(dict(config.get("gradient") or {}).get("analyst_workers", 4)),
        failure_only=False,
        minibatch_size=int(dict(config.get("gradient") or {}).get("minibatch_size", 8)),
        edit_budget=int(dict(config.get("optimizer") or {}).get("learning_rate", 4)),
        max_completion_tokens=int(
            dict(config.get("env") or {}).get("max_completion_tokens", 16384)
        ),
        seed=int(wire.run_seed),
        phase="test",
    )
    output_dir = Path(wire.output_dir)
    test_out = output_dir / "test"
    test_out.mkdir(parents=True, exist_ok=True)
    env = adapter.build_eval_env(
        env_num=len(test_manifest.tasks),
        split="valid_unseen",
        seed=int(wire.run_seed),
    )
    started = time.time()
    rows = adapter.rollout(env, best_skill, str(test_out / "rollout"))
    digest_after = digest_directory(artifact_root)
    if digest_before != digest_after:
        _write_result(output_dir, {
            "phase": "test",
            "error": "frozen artifact changed during the held-out phase",
            "frozen_digest_before": digest_before,
            "frozen_digest_after": digest_after,
        }, passed=False)
        return 1
    from skillopt.model import get_token_summary

    usage = usage_from_skillopt_token_summary(get_token_summary())
    usage.wall_time_ms = int((time.time() - started) * 1000)
    usage.save(test_out / "usage.json")
    episodes = _collect_episodes(test_out)
    result = {
        "phase": "test",
        "output_dir": str(output_dir),
        "episodes": len(episodes),
        "rows": len(rows),
        "usage": usage.to_dict(),
        "frozen_digest_before": digest_before,
        "frozen_digest_after": digest_after,
        "frozen_unchanged": True,
    }
    _write_result(output_dir, result, passed=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wire", required=True, help="worker wire JSON path")
    args = parser.parse_args(argv)
    payload: dict[str, Any] | None = None
    try:
        payload = json.loads(Path(args.wire).read_text(encoding="utf-8"))
        wire = WorkerWire.from_dict(payload)
        model = ModelConfig.from_mapping(wire.model)
        model.validate_formal_identity()
        _configure_model(model)
        config = _load_method_config(wire.config_path)
        lock = load_lock(REPO_ROOT / "experiments" / "baselines" / "baseline_lock.yaml")
        if wire.phase == "train":
            return _run_train(wire=wire, model=model, config=config, lock=lock)
        if wire.phase == "test":
            return _run_test(wire=wire, model=model, config=config)
        raise ValueError(f"unsupported worker phase: {wire.phase}")
    except Exception as exc:
        output_dir = Path(str((payload or {}).get("output_dir") or "."))
        _write_result(output_dir, {
            "phase": str((payload or {}).get("phase", "unknown")),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }, passed=False)
        print(json.dumps({
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
