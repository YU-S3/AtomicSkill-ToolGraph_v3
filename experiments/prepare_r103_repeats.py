#!/usr/bin/env python3
"""Prepare three isolated Test134 configs for one frozen R10.3 bank.

No training, provider calls, bank writes, protocol bypass, or runner changes.
Requires PyYAML (the project's normal config dependency). Run this helper from
outside the tracked source tree, or from runs/ (excluded by hash_code).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required; use the project's validated Python environment.") from exc

ALLOWED_CHANGES = {
    ("trace_data_dir",),
    ("experiment", "output_dir"),
    ("experiment", "task_manifest_path"),
}


def resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def changed_paths(a: Any, b: Any, prefix: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    if isinstance(a, dict) and isinstance(b, dict):
        changes: set[tuple[str, ...]] = set()
        for key in a.keys() | b.keys():
            if key not in a or key not in b:
                changes.add(prefix + (str(key),))
            else:
                changes |= changed_paths(a[key], b[key], prefix + (str(key),))
        return changes
    return set() if type(a) is type(b) and a == b else {prefix}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def make_configs(template: dict[str, Any], seed: int, group_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    if type(seed) is not int or seed not in (42, 43, 44):
        raise ValueError("seed must be one of the existing training seeds: 42, 43, 44")
    exp = template.get("experiment", {})
    expected_name = f"alfworld_frozen_eval_134_r103_seed{seed}"
    expected = {
        "name": expected_name, "phase": "frozen_eval", "condition": "full",
        "runtime_mode": "frozen", "freeze_skills": True, "seed": seed,
        "require_source_code_match": True, "require_knowledge_digest_unchanged": True,
        "allow_eval_traces_and_metrics": True, "allow_long_term_knowledge_writes": False,
        "resume_completed_task_boundary_only": True,
    }
    for key, value in expected.items():
        if type(exp.get(key)) is not type(value) or exp.get(key) != value:
            raise ValueError(f"unexpected template experiment.{key}")
    if template.get("repair_revision") != "R10.3":
        raise ValueError("template must use the existing R10.3 protocol")
    if template.get("r103_interventions") or template.get("r103_learning_intervention", "Full") != "Full":
        raise ValueError("diagnostic masks must not be used for the Full repeat group")
    if template.get("lifecycle", {}).get("candidate_exploration_seed") != seed:
        raise ValueError("do not change the training seed / candidate seed between repeats")
    items = []
    for repeat in (1, 2, 3):
        out = group_root / f"rep{repeat:02d}" / expected_name
        cfg = copy.deepcopy(template)
        cfg["trace_data_dir"] = str(out)
        cfg["experiment"]["output_dir"] = str(out)
        cfg["experiment"]["task_manifest_path"] = str(out / "task_manifest.json")
        if changed_paths(template, cfg) != ALLOWED_CHANGES:
            raise ValueError("repeat generation changed fields other than the three output paths")
        items.append((group_root / "configs" / f"seed{seed}_rep{repeat:02d}.yaml", cfg))
    return items


def prepare(repo: Path, seed: int, group_root: Path, expected_code: str | None = None) -> dict[str, Any]:
    repo, group_root = repo.resolve(), group_root.resolve()
    runs = (repo / "runs").resolve()
    if not group_root.is_relative_to(runs):
        raise ValueError("group-root must be under this checkout's runs/; do not add configs to hashed source after training")
    if group_root.exists():
        raise FileExistsError(f"refusing to overwrite an existing repeat group: {group_root}")
    template_path = repo / "configs" / f"alfworld_frozen_eval_134_r103_seed{seed}.yaml"
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    if not isinstance(template, dict):
        raise ValueError("template must be a mapping")
    exp = template.get("experiment", {})
    train = resolve(repo, exp["source_train_run_dir"])
    bank = resolve(repo, exp["source_frozen_snapshot_dir"])
    if bank != train / "frozen" / "data_v3" or resolve(repo, template["data_dir"]) != bank:
        raise ValueError("data_dir and the original train/frozen source are not aligned")
    if train.name != f"alfworld_train_full_120_r103_seed{seed}":
        raise ValueError("wrong source training seed/name")
    if group_root == bank or group_root.is_relative_to(bank) or bank.is_relative_to(group_root):
        raise ValueError("repeat output group must not contain or be contained by the frozen bank")
    # Read-only identity capture. The actual runner additionally validates the
    # completed training ledger, actual code hash, snapshot digest and closure.
    train_manifest_path = train / "run_manifest.json"
    freeze_manifest_path = bank / "freeze_manifest.json"
    train_manifest = json.loads(train_manifest_path.read_text(encoding="utf-8"))
    frozen = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
    if train_manifest.get("run_id") != train.name or train_manifest.get("phase") != "train":
        raise ValueError("invalid source train manifest")
    if len(train_manifest.get("tasks", [])) != 120:
        raise ValueError("source must be the complete fixed Train120 manifest, not dev16 or a partial bank")
    code_hash = train_manifest.get("code_commit")
    if not isinstance(code_hash, str) or not code_hash or not frozen.get("knowledge_digest"):
        raise ValueError("source code identity / frozen knowledge digest missing")
    if expected_code is not None and code_hash != expected_code:
        raise ValueError("source execution code hash differs from the explicitly supplied value")
    prov = frozen.get("provenance", {})
    if prov.get("source_run_id") != train.name or prov.get("source_code_commit") != code_hash:
        raise ValueError("freeze manifest is not bound to the named training source")
    configs = make_configs(template, seed, group_root)
    plan = {
        "schema": "r103.fixed-bank-repeat-plan.v1", "training_seed": seed,
        "repeat_count": 3, "repeat_indices": [1, 2, 3], "primary_repeat": 1,
        "source_train_run_dir": str(train), "source_frozen_snapshot_dir": str(bank),
        "source_execution_code_hash": code_hash,
        "source_run_manifest_file_sha256": hashlib.sha256(train_manifest_path.read_bytes()).hexdigest(),
        "freeze_manifest_file_sha256": hashlib.sha256(freeze_manifest_path.read_bytes()).hexdigest(),
        "frozen_knowledge_digest": frozen["knowledge_digest"],
        "template_file_sha256": hashlib.sha256(template_path.read_bytes()).hexdigest(),
        "template_canonical_config_hash": canonical_hash(template),
        "changed_fields_only": [".".join(p) for p in sorted(ALLOWED_CHANGES)],
        "interpretation": "3 independent fresh invocations of the same fixed bank; not 3 training seeds and not best-of-3",
        "validation_note": "Preparation is not formal release; original runner source/code/readonly guards remain authoritative.",
        "repeats": [],
    }
    (group_root / "configs").mkdir(parents=True, exist_ok=False)
    for index, (path, cfg) in enumerate(configs, 1):
        text = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
        with path.open("x", encoding="utf-8", newline="\n") as f:
            f.write(text)
        plan["repeats"].append({"repeat_index": index, "config_path": str(path),
            "config_canonical_sha256": canonical_hash(cfg),
            "config_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "output_dir": cfg["experiment"]["output_dir"],
            "experiment_name": cfg["experiment"]["name"]})
    with (group_root / "repetition_manifest.json").open("x", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--group-root", type=Path,
        help="default: <repo>/runs/r103_fixed_bank_seed<seed>_test3")
    parser.add_argument("--expected-code-hash", default=None,
        help="optional exact source execution hash; not a Git commit id")
    args = parser.parse_args()
    root = args.repo.resolve()
    group = args.group_root or root / "runs" / f"r103_fixed_bank_seed{args.seed}_test3"
    try:
        plan = prepare(root, args.seed, group, args.expected_code_hash)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        print(f"Preparation refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
