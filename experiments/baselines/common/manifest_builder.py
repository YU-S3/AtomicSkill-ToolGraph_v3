"""Build the common Train / Validation / Test manifests from ALFWorld data.

The train selection reuses the exact deterministic ``load_balanced_tasks``
algorithm of the main experiment (first-N-per-type in the unfiltered scan
order, sorted by global env index), so ``train_30`` is the same physical
30-task batch the main Full-30 experiment trains on.  ``train_30 ⊂
train_120 ⊂ train_300`` holds by construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from atomic_skillgraph.harness.alfworld import AlfWorldAdapter
from experiments.protocol import (
    ALFWORLD_FORMAL_TASK_TYPES,
    task_signature,
)

from .manifest import (
    ManifestTask,
    TaskManifestSet,
    verify_disjoint,
    verify_nesting,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

_BENCHMARK = "alfworld"

# (manifest_id, adapter split, source split label, per-type count or None for full)
_BUILD_PLAN = [
    ("train_30", "train", "train", 5),
    ("train_120", "train", "train", 20),
    ("train_300", "train", "train", 50),
    ("validation_30", "eval_in_distribution", "valid_seen", 5),
    ("test_ood_60", "eval_out_of_distribution", "valid_unseen", 10),
    ("test_ood_full_134", "eval_out_of_distribution", "valid_unseen", None),
]

_TRAIN_TASKS = 3553
_VALID_UNSEEN_TASKS = 134
_SEED = 42


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _gamefile_sha256(alfworld_data: Path, gamefile_rel: str) -> str:
    return hashlib.sha256(
        (alfworld_data / gamefile_rel).read_bytes()
    ).hexdigest()


def _entry(
    task: Any,
    *,
    index: int,
    source_split: str,
    alfworld_data: Path,
) -> ManifestTask:
    game_file = str(task.context.get("game_file", "")).replace("\\", "/")
    data_prefix = str(alfworld_data).replace("\\", "/").rstrip("/") + "/"
    if not game_file.startswith(data_prefix):
        raise ValueError(f"gamefile outside ALFWORLD_DATA: {game_file}")
    gamefile_rel = game_file[len(data_prefix):]
    return ManifestTask(
        index=index,
        task_id=str(task.task_id),
        task_type=str(task.task_type),
        source_split=source_split,
        env_index=int(task.context.get("env_index", 0)),
        gamefile_rel=gamefile_rel,
        gamefile_sha256=_gamefile_sha256(alfworld_data, gamefile_rel),
        task_signature=task_signature(task),
    )


def build_manifests(
    *, alfworld_data: str | Path, output_dir: str | Path, seed: int = _SEED,
) -> dict[str, TaskManifestSet]:
    data = _path(alfworld_data)
    output = _path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, TaskManifestSet] = {}
    for manifest_id, adapter_split, source_split, per_type in _BUILD_PLAN:
        adapter = AlfWorldAdapter(split=adapter_split, alfworld_data=str(data))
        if per_type is None:
            tasks = adapter.load_tasks(limit=0)
            if len(tasks) != _VALID_UNSEEN_TASKS:
                raise ValueError(
                    f"{manifest_id}: expected {_VALID_UNSEEN_TASKS} valid_unseen tasks, "
                    f"got {len(tasks)}"
                )
        else:
            tasks = adapter.load_balanced_tasks(ALFWORLD_FORMAL_TASK_TYPES, per_type)
        entries = tuple(
            _entry(task, index=index, source_split=source_split, alfworld_data=data)
            for index, task in enumerate(tasks)
        )
        manifest = TaskManifestSet.create(
            manifest_id=manifest_id,
            benchmark=_BENCHMARK,
            source_split=source_split,
            seed=seed,
            tasks=entries,
        )
        manifest.save(output / f"{manifest_id}.json")
        manifests[manifest_id] = manifest
    verify_disjoint(
        manifests["train_300"], manifests["validation_30"], manifests["test_ood_full_134"],
    )
    verify_nesting(manifests["train_30"], manifests["train_120"])
    verify_nesting(manifests["train_120"], manifests["train_300"])
    return manifests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alfworld-data", default=os.environ.get("ALFWORLD_DATA", ""),
        help="ALFWorld data root containing json_2.1.1/ (default: $ALFWORLD_DATA)",
    )
    parser.add_argument(
        "--output", default=str(REPO_ROOT / "data" / "baseline_manifests"),
        help="output directory for manifest JSON files",
    )
    parser.add_argument("--seed", type=int, default=_SEED)
    args = parser.parse_args(argv)
    if not args.alfworld_data:
        raise SystemExit("--alfworld-data (or ALFWORLD_DATA) is required")
    try:
        manifests = build_manifests(
            alfworld_data=args.alfworld_data, output_dir=args.output, seed=args.seed,
        )
    except Exception as exc:
        print(json.dumps({
            "passed": False,
            "gate": "baseline_manifest_build",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps({
        "passed": True,
        "gate": "baseline_manifest_build",
        "output_dir": str(_path(args.output)),
        "manifests": {
            name: {
                "tasks": len(manifest.tasks),
                "digest": manifest.digest,
                "source_split": manifest.source_split,
            }
            for name, manifest in manifests.items()
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
