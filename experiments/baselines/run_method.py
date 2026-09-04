"""Run one baseline method phase (train / test) — design doc section 36.

Examples::

    python -m experiments.baselines.run_method \\
        --method b3_skillopt --phase train \\
        --train-manifest data/baseline_manifests/train_30.json \\
        --validation-manifest data/baseline_manifests/validation_30.json

    python -m experiments.baselines.run_method \\
        --method b3_skillopt --phase test \\
        --train-manifest data/baseline_manifests/train_30.json \\
        --validation-manifest data/baseline_manifests/validation_30.json \\
        --test-manifest data/baseline_manifests/test_ood_60.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from experiments.protocol import hash_code, hash_config, sha256_json

from .b3_skillopt.driver import load_lock_and_driver
from .common.driver import RunContext
from .common.freeze import FrozenArtifact
from .common.integrity import (
    assert_no_secrets_on_disk,
    run_manifest_payload,
    validate_episode_usage,
)
from .common.manifest import TaskManifestSet, verify_disjoint
from .common.model_config import ModelConfig
from .common.post_evaluator import TaskRow, summarize_rows, write_rows_jsonl

REPO_ROOT = Path(__file__).resolve().parents[2]


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _alfworld_data_signature(data_root: Path) -> str:
    logic = data_root / "logic"
    required = {
        "alfred.pddl": logic / "alfred.pddl",
        "alfred.twl2": logic / "alfred.twl2",
    }
    hashes: dict[str, str] = {}
    for name, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"ALFWorld logic file is missing: {path}")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    dataset_root = data_root / "json_2.1.1"
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"ALFWorld dataset root is missing: {dataset_root}")
    gamefiles = sorted(
        path.relative_to(dataset_root).as_posix()
        for path in dataset_root.rglob("game.tw-pddl")
    )
    hashes["dataset_gamefile_manifest_sha256"] = hashlib.sha256(
        "\n".join(gamefiles).encode("utf-8")
    ).hexdigest()
    hashes["dataset_gamefile_count"] = str(len(gamefiles))
    return json.dumps(hashes, sort_keys=True, separators=(",", ":"))


def _load_manifest(path: str | Path) -> TaskManifestSet:
    return TaskManifestSet.load(_path(path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, help="method id, e.g. b3_skillopt")
    parser.add_argument("--phase", required=True, choices=["train", "test"])
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--test-manifest", default=None)
    parser.add_argument("--config", default="configs/baselines/b3_skillopt.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--frozen-artifact", default=None)
    args = parser.parse_args(argv)
    if args.method != "b3_skillopt":
        raise SystemExit("only the b3_skillopt method is implemented in this round")
    try:
        lock, driver = load_lock_and_driver(repo_root=REPO_ROOT)
        config = driver.config
        model = ModelConfig.from_mapping(config["model"])
        model.validate_formal_identity()
        model.require_api_key()
        alfworld_data = Path(os.environ.get("ALFWORLD_DATA", "")).expanduser()
        if not alfworld_data.is_dir():
            raise FileNotFoundError(
                f"ALFWORLD_DATA directory does not exist: {alfworld_data}"
            )
        campaign_id = str(config.get("campaign_id", "pilot"))
        run_seed = int(config.get("run_seed", 42))
        output_dir = _path(args.output_dir) if args.output_dir else (
            REPO_ROOT / "runs" / "baselines" / campaign_id / args.method / str(run_seed)
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        train_manifest = _load_manifest(args.train_manifest)
        validation_manifest = _load_manifest(args.validation_manifest)
        test_manifest = (
            _load_manifest(args.test_manifest) if args.test_manifest else None
        )
        sets = [train_manifest, validation_manifest]
        if test_manifest is not None:
            sets.append(test_manifest)
        verify_disjoint(*sets)

        max_actions = int(config.get("max_environment_actions", 100))
        ctx = RunContext(
            campaign_id=campaign_id,
            method_id=args.method,
            run_seed=run_seed,
            output_dir=output_dir,
            repo_root=REPO_ROOT,
            external_repo=REPO_ROOT / ".external" / "skillopt",
            external_commit=str(lock["skillopt"]["commit"]),
            model_config=model,
            max_environment_actions=max_actions,
            alfworld_data=alfworld_data,
            config_hash=sha256_json([
                hash_config(REPO_ROOT / "configs" / "baselines" / "common.yaml"),
                hash_config(_path(args.config)),
            ]),
            code_hash=hash_code(REPO_ROOT),
            train_manifest_path=_path(args.train_manifest),
            validation_manifest_path=_path(args.validation_manifest),
            test_manifest_path=_path(args.test_manifest) if args.test_manifest else None,
        )
        driver.preflight(ctx)

        provenance = run_manifest_payload(
            method=args.method,
            external_repo=str(lock["skillopt"]["repo"]),
            external_commit=str(lock["skillopt"]["commit"]),
            ours_controller_commit=ctx.code_hash,
            train_manifest_hash=train_manifest.digest,
            validation_manifest_hash=validation_manifest.digest,
            test_manifest_hash=test_manifest.digest if test_manifest else None,
            alfworld_version=_installed_alfworld_version(),
            alfworld_data_signature=_alfworld_data_signature(alfworld_data),
            model=model.model,
            provider=model.provider,
            method_specific_decoding={
                "train": dict(config.get("train") or {}),
                "gradient": dict(config.get("gradient") or {}),
                "optimizer": dict(config.get("optimizer") or {}),
                "evaluation": dict(config.get("evaluation") or {}),
                "env": dict(config.get("env") or {}),
            },
            max_environment_actions=max_actions,
            run_seed=run_seed,
            phase=args.phase,
            output_dir=str(output_dir),
        )
        if args.phase == "train":
            # Train creates the run directory identity; Test only appends its
            # own phase provenance and must never overwrite train records.
            _write_json(output_dir / "run_manifest.json", provenance)
            _write_json(output_dir / "config_resolved.json", {
                "model": model.to_wire(),
                "max_environment_actions": max_actions,
                "campaign_id": campaign_id,
                "run_seed": run_seed,
                "train": config.get("train"),
                "gradient": config.get("gradient"),
                "optimizer": config.get("optimizer"),
                "evaluation": config.get("evaluation"),
                "env": config.get("env"),
            })
            _write_json(output_dir / "source_lock.json", {
                "schema_version": 1,
                "ours": lock.get("ours"),
                "skillopt": {
                    key: value
                    for key, value in lock["skillopt"].items()
                    if key != "key_files"
                },
            })
            _write_json(output_dir / "task_manifest.json", {
                "train": {
                    "path": str(_path(args.train_manifest)),
                    "manifest_id": train_manifest.manifest_id,
                    "digest": train_manifest.digest,
                    "tasks": len(train_manifest.tasks),
                },
                "validation": {
                    "path": str(_path(args.validation_manifest)),
                    "manifest_id": validation_manifest.manifest_id,
                    "digest": validation_manifest.digest,
                    "tasks": len(validation_manifest.tasks),
                },
                "test": (
                    {
                        "path": str(_path(args.test_manifest)),
                        "manifest_id": test_manifest.manifest_id,
                        "digest": test_manifest.digest,
                        "tasks": len(test_manifest.tasks),
                    }
                    if test_manifest is not None else None
                ),
            })
        else:
            _write_json(output_dir / "run_manifest_test.json", provenance)

        if args.phase == "train":
            train_result = driver.train(ctx, train_manifest, validation_manifest)
            frozen = driver.freeze(ctx, train_result)
            print(json.dumps({
                "passed": True,
                "method": args.method,
                "phase": "train",
                "output_dir": str(output_dir),
                "frozen": {
                    "root": str(frozen.root),
                    "digest": frozen.digest,
                },
                "train_episodes": len(train_result.episodes),
                "validation_episodes": len(train_result.validation_episodes),
                "usage": train_result.usage.to_dict(),
                "method_metrics": train_result.method_metrics,
            }, ensure_ascii=False, indent=2))
            return 0

        # phase == test
        frozen_dir = (
            _path(args.frozen_artifact) if args.frozen_artifact else output_dir / "frozen"
        )
        frozen = FrozenArtifact.load(frozen_dir)
        if frozen.source_train_manifest_hash != train_manifest.digest:
            raise ValueError("frozen artifact does not match the given Train manifest")
        if (
            frozen.source_validation_manifest_hash is not None
            and frozen.source_validation_manifest_hash != validation_manifest.digest
        ):
            raise ValueError("frozen artifact does not match the given Validation manifest")
        assert test_manifest is not None
        episodes = driver.evaluate(
            ctx, frozen, test_manifest, frozen_dir=frozen_dir,
        )
        validate_episode_usage(episodes)
        assert_no_secrets_on_disk(output_dir, api_key_env=model.api_key_env)
        rows = [TaskRow.from_episode(episode) for episode in episodes]
        write_rows_jsonl(rows, output_dir / "test" / "task_rows.jsonl")
        from experiments.protocol import ALFWORLD_FORMAL_TASK_TYPES

        summary = summarize_rows(rows, task_types=list(ALFWORLD_FORMAL_TASK_TYPES))
        _write_json(output_dir / "test" / "summary.json", {
            "method": args.method,
            "phase": "test",
            "frozen_digest": frozen.digest,
            **summary,
        })
        print(json.dumps({
            "passed": True,
            "method": args.method,
            "phase": "test",
            "output_dir": str(output_dir),
            "frozen_digest": frozen.digest,
            "summary": summary,
        }, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({
            "passed": False,
            "method": args.method,
            "phase": args.phase,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }, ensure_ascii=False, indent=2))
        return 1


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _installed_alfworld_version() -> str:
    try:
        import importlib.metadata as importlib_metadata

        return str(importlib_metadata.version("alfworld"))
    except Exception:
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
