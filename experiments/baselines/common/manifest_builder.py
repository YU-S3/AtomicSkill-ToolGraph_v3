"""Build immutable ALFWorld manifests for the external-baseline protocol.

Formal Train and Validation selection follows the frozen protocol exactly:
enumerate the authoritative TextWorld gamefiles, canonical-sort within each
family, shuffle each family once with selection seed 42, take a family prefix,
then store the selected tasks in canonical path order. The smaller pilot and
smoke sets are prefixes of the same family permutations. Frozen Test contains
all 134 ``valid_unseen`` games in canonical path order and is never sampled.

Only physical task identity is materialized. The builder never resets a game
and never reads expert actions, PDDL state, human plans, or hidden references.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

from atomic_skillgraph.harness.alfworld import AlfWorldAdapter
from .formal_validation import ALFWORLD_FORMAL_TASK_TYPES

from .manifest import (
    ManifestTask,
    TaskManifestSet,
    verify_disjoint,
    verify_nesting,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

_BENCHMARK = "alfworld"
_SELECTION_SEED = 42

# (manifest_id, adapter split, source split label, per-family count or None)
# Legacy pilot manifests remain because earlier local comparison runs refer to
# their digests. Formal B3 consumes Train120, Validation24, and Test134 only.
_BUILD_PLAN = (
    ("train_6_smoke", "train", "train", 1),
    ("train_30", "train", "train", 5),
    ("train_120", "train", "train", 20),
    ("train_300", "train", "train", 50),
    ("validation_6_smoke", "eval_in_distribution", "valid_seen", 1),
    ("validation_6", "eval_in_distribution", "valid_seen", 1),
    ("validation_12", "eval_in_distribution", "valid_seen", 2),
    ("validation_24", "eval_in_distribution", "valid_seen", 4),
    ("validation_30", "eval_in_distribution", "valid_seen", 5),
    ("test_6_smoke", "eval_out_of_distribution", "valid_unseen", 1),
    ("test_ood_60", "eval_out_of_distribution", "valid_unseen", 10),
    ("test_ood_full_134", "eval_out_of_distribution", "valid_unseen", None),
)

_EXPECTED_FORMAL_GAMEFILES = {
    "train": 3553,
    "valid_seen": 140,
    "valid_unseen": 134,
}


@dataclass(frozen=True)
class _GamefileIdentity:
    adapter_split: str
    source_split: str
    env_index: int
    gamefile_rel: str
    gamefile_sha256: str
    task_type: str
    task_signature: str


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _physical_task_signature(
    *, source_split: str, task_type: str, gamefile_rel: str, gamefile_sha256: str,
) -> str:
    """Hash only public physical identity, with no task solution material."""

    payload = "\x1f".join((
        _BENCHMARK,
        source_split,
        task_type,
        gamefile_rel,
        gamefile_sha256,
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _task_type_from_gamefile(gamefile_rel: str, *, source_split: str) -> str:
    parts = gamefile_rel.split("/")
    if len(parts) < 4 or parts[:2] != ["json_2.1.1", source_split]:
        raise ValueError(
            f"gamefile is outside json_2.1.1/{source_split}: {gamefile_rel}"
        )
    family_directory = parts[2]
    matches = [
        task_type
        for task_type in ALFWORLD_FORMAL_TASK_TYPES
        if family_directory == task_type
        or family_directory.startswith(task_type + "-")
    ]
    if len(matches) != 1:
        raise ValueError(
            f"gamefile has unsupported or ambiguous ALFWorld family: {gamefile_rel}"
        )
    return matches[0]


def _close_adapter(adapter: AlfWorldAdapter) -> None:
    env = getattr(adapter, "_env", None)
    close = getattr(env, "close", None)
    if callable(close):
        close()


def _enumerate_gamefiles(
    *,
    adapter_split: str,
    source_split: str,
    alfworld_data: Path,
) -> list[_GamefileIdentity]:
    """Read ALFWorld's authoritative gamefile/index mapping without reset."""

    adapter = AlfWorldAdapter(
        split=adapter_split,
        alfworld_data=str(alfworld_data),
    )
    try:
        adapter.initialize()
        tw_env = getattr(adapter, "_tw_env", None)
        raw_files = (
            getattr(tw_env, "gamefiles", None)
            or getattr(tw_env, "game_files", None)
        )
        if not raw_files:
            raise RuntimeError(
                f"ALFWorld exposed no gamefiles for split {adapter_split!r}"
            )
        identities: list[_GamefileIdentity] = []
        prefix = ("json_2.1.1", source_split)
        for env_index, raw_file in enumerate(raw_files):
            gamefile = Path(str(raw_file)).expanduser().resolve(strict=True)
            try:
                relative = gamefile.relative_to(alfworld_data)
            except ValueError as exc:
                raise ValueError(
                    f"ALFWorld gamefile resolves outside ALFWORLD_DATA: {gamefile}"
                ) from exc
            if tuple(relative.parts[:2]) != prefix:
                # The upstream train environment also exposes non-train rows.
                continue
            gamefile_rel = relative.as_posix()
            task_type = _task_type_from_gamefile(
                gamefile_rel,
                source_split=source_split,
            )
            content_sha = _sha256_file(gamefile)
            identities.append(_GamefileIdentity(
                adapter_split=adapter_split,
                source_split=source_split,
                env_index=env_index,
                gamefile_rel=gamefile_rel,
                gamefile_sha256=content_sha,
                task_type=task_type,
                task_signature=_physical_task_signature(
                    source_split=source_split,
                    task_type=task_type,
                    gamefile_rel=gamefile_rel,
                    gamefile_sha256=content_sha,
                ),
            ))
    finally:
        _close_adapter(adapter)

    expected = _EXPECTED_FORMAL_GAMEFILES[source_split]
    if len(identities) != expected:
        raise ValueError(
            f"{source_split}: expected {expected} formal TextWorld gamefiles, "
            f"got {len(identities)}"
        )
    paths = [item.gamefile_rel for item in identities]
    if len(set(paths)) != len(paths):
        raise ValueError(f"{source_split}: ALFWorld exposed duplicate gamefiles")
    observed_families = {item.task_type for item in identities}
    expected_families = set(ALFWORLD_FORMAL_TASK_TYPES)
    if observed_families != expected_families:
        raise ValueError(
            f"{source_split}: formal family set mismatch; "
            f"missing={sorted(expected_families - observed_families)}, "
            f"unexpected={sorted(observed_families - expected_families)}"
        )
    return identities


def _family_permutations(
    tasks: list[_GamefileIdentity], *, selection_seed: int,
) -> dict[str, list[_GamefileIdentity]]:
    """Canonical-sort then seed-shuffle each family exactly once."""

    rng = random.Random(int(selection_seed))
    result: dict[str, list[_GamefileIdentity]] = {}
    for task_type in ALFWORLD_FORMAL_TASK_TYPES:
        family = sorted(
            (task for task in tasks if task.task_type == task_type),
            key=lambda task: task.gamefile_rel,
        )
        if not family:
            raise ValueError(f"no ALFWorld games found for family {task_type!r}")
        rng.shuffle(family)
        result[task_type] = family
    return result


def _select_balanced(
    permutations: dict[str, list[_GamefileIdentity]], *, per_type: int,
) -> list[_GamefileIdentity]:
    selected: list[_GamefileIdentity] = []
    for task_type in ALFWORLD_FORMAL_TASK_TYPES:
        family = permutations[task_type]
        if len(family) < per_type:
            raise ValueError(
                f"insufficient {task_type} games: need {per_type}, got {len(family)}"
            )
        selected.extend(family[:per_type])
    # Selection is random but storage/evaluation identity is canonical.
    return sorted(selected, key=lambda task: task.gamefile_rel)


def _manifest_entry(task: _GamefileIdentity, *, index: int) -> ManifestTask:
    return ManifestTask(
        index=index,
        task_id=(
            f"alfworld_{task.adapter_split}_{task.env_index}_{task.task_type}"
        ),
        task_type=task.task_type,
        source_split=task.source_split,
        env_index=task.env_index,
        gamefile_rel=task.gamefile_rel,
        gamefile_sha256=task.gamefile_sha256,
        task_signature=task.task_signature,
    )


def build_manifests(
    *,
    alfworld_data: str | Path,
    output_dir: str | Path,
    seed: int = _SELECTION_SEED,
) -> dict[str, TaskManifestSet]:
    if int(seed) != _SELECTION_SEED:
        raise ValueError(
            f"formal selection seed is frozen at {_SELECTION_SEED}, got {seed}"
        )
    data = _path(alfworld_data)
    output = _path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    tasks_by_split: dict[str, list[_GamefileIdentity]] = {}
    family_orders: dict[str, dict[str, list[_GamefileIdentity]]] = {}
    for adapter_split, source_split in dict.fromkeys(
        (item[1], item[2]) for item in _BUILD_PLAN
    ):
        tasks = _enumerate_gamefiles(
            adapter_split=adapter_split,
            source_split=source_split,
            alfworld_data=data,
        )
        tasks_by_split[adapter_split] = tasks
        family_orders[adapter_split] = _family_permutations(
            tasks,
            selection_seed=seed,
        )

    manifests: dict[str, TaskManifestSet] = {}
    for manifest_id, adapter_split, source_split, per_type in _BUILD_PLAN:
        if per_type is None:
            selected = sorted(
                tasks_by_split[adapter_split],
                key=lambda task: task.gamefile_rel,
            )
        else:
            selected = _select_balanced(
                family_orders[adapter_split],
                per_type=int(per_type),
            )
        entries = tuple(
            _manifest_entry(task, index=index)
            for index, task in enumerate(selected)
        )
        manifest = TaskManifestSet.create(
            manifest_id=manifest_id,
            benchmark=_BENCHMARK,
            source_split=source_split,
            seed=int(seed),
            tasks=entries,
        )
        manifest.save(output / f"{manifest_id}.json")
        manifests[manifest_id] = manifest

    verify_disjoint(
        manifests["train_300"],
        manifests["validation_30"],
        manifests["test_ood_full_134"],
    )
    verify_nesting(manifests["train_6_smoke"], manifests["train_30"])
    verify_nesting(manifests["train_30"], manifests["train_120"])
    verify_nesting(manifests["train_120"], manifests["train_300"])
    verify_nesting(
        manifests["validation_6_smoke"], manifests["validation_12"]
    )
    verify_nesting(manifests["validation_6"], manifests["validation_12"])
    verify_nesting(manifests["validation_12"], manifests["validation_24"])
    verify_nesting(manifests["validation_24"], manifests["validation_30"])
    verify_nesting(manifests["test_6_smoke"], manifests["test_ood_60"])
    return manifests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alfworld-data",
        default=os.environ.get("ALFWORLD_DATA", ""),
        help="ALFWorld data root containing json_2.1.1/ (default: $ALFWORLD_DATA)",
    )
    parser.add_argument(
        "--output",
        "--out",
        dest="output",
        default=str(REPO_ROOT / "data" / "baseline_manifests"),
        help="output directory for manifest JSON files",
    )
    parser.add_argument(
        "--seed",
        "--selection-seed",
        dest="seed",
        type=int,
        default=_SELECTION_SEED,
    )
    args = parser.parse_args(argv)
    if not args.alfworld_data:
        raise SystemExit("--alfworld-data (or ALFWORLD_DATA) is required")
    try:
        manifests = build_manifests(
            alfworld_data=args.alfworld_data,
            output_dir=args.output,
            seed=args.seed,
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
        "selection_seed": args.seed,
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
