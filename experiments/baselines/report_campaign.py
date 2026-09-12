"""Recompute cross-method paper statistics from completed test artifacts.

The reporter is deliberately read-only with respect to every input run.  It
loads persisted post-hoc episode records, recomputes all effectiveness and
efficiency summaries, and labels comparisons unavailable when their required
authority (B0, Ours, paired rows, or API pricing) is absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common.formal_validation import ALFWORLD_FORMAL_TASK_TYPES
from .common.post_evaluator import TaskRow, load_rows_jsonl, summarize_rows
from .common.trace import load_episodes


_DEFAULT_SEEDS = (42, 43, 44)
_MEAN_STD_METRICS = (
    "official_success_rate",
    "macro_family_official_rate",
    "contract_consistency_rate",
    "contract_consistent_success_rate",
    "actions_per_task",
    "actions_per_solved",
    "calls_per_task",
    "tokens_per_task",
    "tokens_per_solved",
    "p50_actions",
    "p90_actions",
    "p50_calls",
    "p90_calls",
    "p50_tokens",
    "p90_tokens",
    "latency_per_task_ms",
    "p50_latency_ms",
    "p90_latency_ms",
    "cost_per_task",
    "cost_per_solved",
)


@dataclass(frozen=True)
class _SeedArtifact:
    method: str
    seed: int
    root: Path
    rows: tuple[TaskRow, ...]
    api_cost: float | None
    api_cost_unpriced: bool
    source_hashes: Mapping[str, str]


def build_campaign_report(
    method_runs: Mapping[str, Sequence[str | Path] | str | Path],
    *,
    expected_seeds: Sequence[int] = _DEFAULT_SEEDS,
    task_types: Sequence[str] = ALFWORLD_FORMAL_TASK_TYPES,
    b0_method: str = "b0_pure_dynamic",
    ours_method: str = "ours",
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260912,
) -> dict[str, Any]:
    """Build a deterministic paper report without trusting run summaries."""

    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    seeds = tuple(int(seed) for seed in expected_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("expected_seeds must be non-empty and unique")
    families = tuple(str(value) for value in task_types)
    if not method_runs:
        raise ValueError("at least one method run is required")

    artifacts: dict[str, list[_SeedArtifact]] = {}
    for raw_method, roots in method_runs.items():
        method = str(raw_method).strip()
        if not method:
            raise ValueError("method names must be non-empty")
        if method in artifacts:
            raise ValueError(f"duplicate method: {method}")
        normalized_roots = [roots] if isinstance(roots, (str, Path)) else roots
        expanded = _expand_completed_roots(normalized_roots)
        loaded = [_load_seed_artifact(method, root) for root in expanded]
        by_seed = {item.seed: item for item in loaded}
        if len(by_seed) != len(loaded):
            raise ValueError(f"method {method!r} contains duplicate seed artifacts")
        if set(by_seed) != set(seeds):
            raise ValueError(
                f"method {method!r} completed seeds {sorted(by_seed)} do not equal "
                f"expected seeds {sorted(seeds)}"
            )
        artifacts[method] = [by_seed[seed] for seed in seeds]

    method_reports = {
        method: _summarize_method(
            method,
            runs,
            task_types=families,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        for method, runs in artifacts.items()
    }

    if b0_method in artifacts:
        baseline_rows = _paired_rows(artifacts[b0_method])
        for method, runs in artifacts.items():
            if method == b0_method:
                method_reports[method]["transfer_vs_b0"] = {
                    "status": "reference_method",
                    "reference_method": b0_method,
                }
                continue
            method_reports[method]["transfer_vs_b0"] = _transfer_report(
                baseline_rows,
                _paired_rows(runs),
                reference_method=b0_method,
                method=method,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
    else:
        for method in method_reports:
            method_reports[method]["transfer_vs_b0"] = {
                "status": "unavailable_without_b0_artifacts",
                "reference_method": b0_method,
            }

    pairwise = _ours_pairwise(
        artifacts,
        ours_method=ours_method,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    paired_rows = [
        {
            "method": method,
            **row.to_dict(),
            "pair_key": [row.task_id, row.run_seed],
        }
        for method in sorted(artifacts)
        for run in artifacts[method]
        for row in run.rows
    ]
    return {
        "schema_version": 1,
        "passed": True,
        "source": "read_only_completed_test_episode_artifacts",
        "expected_seeds": list(seeds),
        "task_types": list(families),
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "confidence_level": 0.95,
            "unit": "paired_task_seed",
        },
        "methods": method_reports,
        "paired_task_rows": paired_rows,
        "paired_task_key": ["task_id", "run_seed"],
        "ours_pairwise": pairwise,
    }


def _expand_completed_roots(roots: Sequence[str | Path]) -> list[Path]:
    expanded: list[Path] = []
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve(strict=True)
        campaign_path = root / "campaign_report.json"
        if not campaign_path.is_file():
            expanded.append(root)
            continue
        report = _read_json(campaign_path)
        if report.get("passed") is not True:
            raise ValueError(f"campaign is not completed successfully: {root}")
        lanes = list(report.get("lanes") or [])
        if not lanes:
            raise ValueError(f"campaign has no completed lanes: {root}")
        for lane in lanes:
            if not isinstance(lane, dict) or lane.get("passed") is not True:
                raise ValueError(f"campaign contains an incomplete lane: {root}")
            raw_test_root = lane.get("test_root")
            if not isinstance(raw_test_root, str) or not raw_test_root.strip():
                raise ValueError(f"campaign lane has no test_root: {root}")
            test_root = Path(raw_test_root).expanduser()
            if not test_root.is_absolute():
                test_root = root / test_root
            expanded.append(test_root.resolve(strict=True))
    if not expanded:
        raise ValueError("method run list must not be empty")
    return expanded


def _load_seed_artifact(method: str, root: Path) -> _SeedArtifact:
    completion_path = root / "completion.json"
    report_path = root / "test_report.json"
    completion = _read_json(completion_path)
    report = _read_json(report_path)
    if completion.get("passed") is not True or completion.get("phase") != "test":
        raise ValueError(f"test artifact is not completed successfully: {root}")
    if report.get("passed") is not True or report.get("phase", "test") != "test":
        raise ValueError(f"test report is not completed successfully: {root}")
    completion_report = completion.get("report")
    if completion_report is not None and completion_report != "test_report.json":
        raise ValueError(f"test completion names an unexpected report: {root}")
    completion_run_id = str(completion.get("run_id") or "")
    report_run_id = str(report.get("run_id") or "")
    if completion_run_id and report_run_id and completion_run_id != report_run_id:
        raise ValueError(f"test completion/report run_id mismatch: {root}")
    reported_method = str(report.get("method", ""))
    if reported_method != method:
        raise ValueError(
            f"artifact method {reported_method!r} does not match requested {method!r}"
        )

    evaluated_path = root / "test" / "evaluated_common_episodes.jsonl"
    task_rows_path = root / "test" / "task_rows.jsonl"
    source_hashes = {
        "completion.json": _sha256_file(completion_path),
        "test_report.json": _sha256_file(report_path),
    }
    if evaluated_path.is_file():
        rows = [TaskRow.from_episode(item) for item in load_episodes(evaluated_path)]
        source_hashes["test/evaluated_common_episodes.jsonl"] = _sha256_file(
            evaluated_path
        )
        if task_rows_path.is_file():
            persisted_rows = load_rows_jsonl(task_rows_path)
            _assert_same_rows(rows, persisted_rows, root=root)
            source_hashes["test/task_rows.jsonl"] = _sha256_file(task_rows_path)
    elif task_rows_path.is_file():
        # Compatibility for completed pilot artifacts.  New formal runs write
        # evaluated_common_episodes.jsonl and are cross-checked above.
        rows = load_rows_jsonl(task_rows_path)
        source_hashes["test/task_rows.jsonl"] = _sha256_file(task_rows_path)
    else:
        raise FileNotFoundError(f"test episode records are missing: {root}")
    if not rows:
        raise ValueError(f"test episode records are empty: {root}")

    seed_values = {int(row.run_seed) for row in rows}
    if len(seed_values) != 1:
        raise ValueError(f"test artifact mixes run seeds: {root}")
    seed = next(iter(seed_values))
    seen: set[tuple[str, int]] = set()
    for row in rows:
        key = (row.task_id, row.run_seed)
        if key in seen:
            raise ValueError(f"duplicate paired task row {key!r}: {root}")
        seen.add(key)
        if row.method != method or row.phase != "test":
            raise ValueError(f"task row identity disagrees with test artifact: {root}")
        if row.infrastructure_failure:
            raise ValueError(f"completed test contains infrastructure failure: {root}")
        if not isinstance(row.official_success, bool):
            raise ValueError(f"task row official_success is not boolean: {root}")
        if not isinstance(row.contract_consistency, bool):
            raise ValueError(f"task row lacks contract_consistency: {root}")
        if not isinstance(row.common_strict_success, bool):
            raise ValueError(f"task row lacks common_strict_success: {root}")
        counters = {
            "environment_actions": row.environment_actions,
            "target_llm_calls": row.target_llm_calls,
            "target_prompt_tokens": row.target_prompt_tokens,
            "target_completion_tokens": row.target_completion_tokens,
            "target_reasoning_tokens": row.target_reasoning_tokens,
            "evolution_llm_calls": row.evolution_llm_calls,
            "evolution_prompt_tokens": row.evolution_prompt_tokens,
            "evolution_completion_tokens": row.evolution_completion_tokens,
            "evolution_reasoning_tokens": row.evolution_reasoning_tokens,
            "embedding_calls": row.embedding_calls,
            "wall_time_ms": row.wall_time_ms,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counters.values()
        ):
            raise ValueError(f"task row has an invalid cost counter: {root}")
        if (
            isinstance(row.invalid_actions, bool)
            or not isinstance(row.invalid_actions, int)
            or row.invalid_actions < 0
            or row.invalid_actions > row.environment_actions
        ):
            raise ValueError(f"task row has invalid action accounting: {root}")

    test_cost = dict(report.get("test_cost") or {})
    unpriced = bool(test_cost.get("api_cost_unpriced", True))
    raw_cost = test_cost.get("api_cost")
    if unpriced:
        if raw_cost is not None:
            raise ValueError(f"unpriced test report declares api_cost: {root}")
        api_cost = None
    else:
        if isinstance(raw_cost, bool) or not isinstance(raw_cost, (int, float)):
            raise ValueError(f"priced test report lacks numeric api_cost: {root}")
        api_cost = float(raw_cost)
        if not math.isfinite(api_cost) or api_cost < 0:
            raise ValueError(f"test report api_cost is invalid: {root}")
    return _SeedArtifact(
        method=method,
        seed=seed,
        root=root,
        rows=tuple(rows),
        api_cost=api_cost,
        api_cost_unpriced=unpriced,
        source_hashes=source_hashes,
    )


def _summarize_method(
    method: str,
    runs: Sequence[_SeedArtifact],
    *,
    task_types: Sequence[str],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    seed_summaries: dict[str, dict[str, Any]] = {}
    all_rows: list[TaskRow] = []
    all_priced = all(not run.api_cost_unpriced for run in runs)
    total_cost = sum(float(run.api_cost or 0.0) for run in runs) if all_priced else None
    for run in runs:
        summary = summarize_rows(
            list(run.rows),
            task_types=list(task_types),
            api_cost=run.api_cost,
            api_cost_unpriced=run.api_cost_unpriced,
        )
        seed_summaries[str(run.seed)] = summary
        all_rows.extend(run.rows)
    pooled = summarize_rows(
        all_rows,
        task_types=list(task_types),
        api_cost=total_cost,
        api_cost_unpriced=not all_priced,
    )
    mean_std = {
        metric: _mean_std_by_seed(seed_summaries, metric)
        for metric in _MEAN_STD_METRICS
    }
    family_mean_std = {
        family: _mean_std_values(
            {
                seed: summary["family"][family]["official_rate"]
                for seed, summary in seed_summaries.items()
            }
        )
        for family in task_types
    }

    row_metrics = {
        "official_success_rate": [float(row.official_success) for row in all_rows],
        "actions_per_task": [float(row.environment_actions) for row in all_rows],
        "calls_per_task": [
            float(row.target_llm_calls + row.evolution_llm_calls) for row in all_rows
        ],
        "tokens_per_task": [
            float(
                row.target_prompt_tokens
                + row.target_completion_tokens
                + row.evolution_prompt_tokens
                + row.evolution_completion_tokens
            )
            for row in all_rows
        ],
    }
    bootstrap = {
        metric: _bootstrap_mean(
            values,
            samples=bootstrap_samples,
            seed=_derived_seed(bootstrap_seed, method, metric),
            unit="task_seed",
        )
        for metric, values in row_metrics.items()
    }
    if all_priced:
        cost_values = [
            seed_summaries[str(run.seed)]["cost_per_task"] for run in runs
        ]
        solved_cost_values = [
            seed_summaries[str(run.seed)]["cost_per_solved"] for run in runs
        ]
        bootstrap["cost_per_task"] = _bootstrap_mean(
            [float(value) for value in cost_values if value is not None],
            samples=bootstrap_samples,
            seed=_derived_seed(bootstrap_seed, method, "cost_per_task"),
            unit="seed",
        )
        bootstrap["cost_per_solved"] = (
            _bootstrap_mean(
                [float(value) for value in solved_cost_values if value is not None],
                samples=bootstrap_samples,
                seed=_derived_seed(bootstrap_seed, method, "cost_per_solved"),
                unit="seed",
            )
            if all(value is not None for value in solved_cost_values)
            else {"status": "unavailable_without_solved_tasks_in_every_seed"}
        )
    else:
        bootstrap["cost_per_task"] = {
            "status": "unavailable_without_pricing_authority"
        }
        bootstrap["cost_per_solved"] = {
            "status": "unavailable_without_pricing_authority"
        }
    return {
        "method": method,
        "completed_seeds": [run.seed for run in runs],
        "artifact_roots": {str(run.seed): str(run.root) for run in runs},
        "source_hashes": {
            str(run.seed): dict(run.source_hashes) for run in runs
        },
        "seed_summaries": seed_summaries,
        "mean_std": mean_std,
        "family_official_success_rate_mean_std": family_mean_std,
        "pooled": pooled,
        "bootstrap_95_ci": bootstrap,
    }


def _mean_std_by_seed(
    summaries: Mapping[str, Mapping[str, Any]], metric: str
) -> dict[str, Any]:
    return _mean_std_values({seed: summary.get(metric) for seed, summary in summaries.items()})


def _mean_std_values(values_by_seed: Mapping[str, Any]) -> dict[str, Any]:
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values_by_seed.values()
    ):
        return {
            "status": "unavailable_in_one_or_more_seeds",
            "mean": None,
            "std": None,
            "values_by_seed": dict(values_by_seed),
        }
    values = [float(value) for value in values_by_seed.values()]
    return {
        "status": "available",
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "std_ddof": 1 if len(values) > 1 else 0,
        "values_by_seed": dict(values_by_seed),
    }


def _paired_rows(runs: Sequence[_SeedArtifact]) -> dict[tuple[str, int], TaskRow]:
    result: dict[tuple[str, int], TaskRow] = {}
    for run in runs:
        for row in run.rows:
            key = (row.task_id, row.run_seed)
            if key in result:
                raise ValueError(f"duplicate paired row: {key!r}")
            result[key] = row
    return result


def _require_matching_pairs(
    left: Mapping[tuple[str, int], TaskRow],
    right: Mapping[tuple[str, int], TaskRow],
    *,
    comparison: str,
) -> list[tuple[str, int]]:
    if set(left) != set(right):
        missing_left = sorted(set(right) - set(left))[:5]
        missing_right = sorted(set(left) - set(right))[:5]
        raise ValueError(
            f"{comparison} paired task keys differ; missing_left={missing_left}, "
            f"missing_right={missing_right}"
        )
    keys = sorted(left, key=lambda key: (key[1], key[0]))
    for key in keys:
        if (
            left[key].task_type != right[key].task_type
            or left[key].manifest_index != right[key].manifest_index
            or left[key].gamefile_hash != right[key].gamefile_hash
        ):
            raise ValueError(f"{comparison} task authority differs for {key!r}")
    return keys


def _transfer_report(
    baseline: Mapping[tuple[str, int], TaskRow],
    candidate: Mapping[tuple[str, int], TaskRow],
    *,
    reference_method: str,
    method: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    keys = _require_matching_pairs(
        baseline, candidate, comparison=f"{method} vs {reference_method}"
    )
    positive = sum(
        not baseline[key].official_success and candidate[key].official_success
        for key in keys
    )
    negative = sum(
        baseline[key].official_success and not candidate[key].official_success
        for key in keys
    )
    base_failures = sum(not baseline[key].official_success for key in keys)
    base_successes = len(keys) - base_failures
    differences = [
        float(candidate[key].official_success) - float(baseline[key].official_success)
        for key in keys
    ]
    return {
        "status": "available",
        "reference_method": reference_method,
        "paired_tasks": len(keys),
        "positive_transfer": _ratio(positive, len(keys)),
        "negative_transfer": _ratio(negative, len(keys)),
        "positive_transfer_count": positive,
        "negative_transfer_count": negative,
        "positive_transfer_given_b0_failure": _ratio(positive, base_failures),
        "negative_transfer_given_b0_success": _ratio(negative, base_successes),
        "official_success_rate_delta": statistics.fmean(differences),
        "paired_bootstrap_95_ci": _bootstrap_mean(
            differences,
            samples=bootstrap_samples,
            seed=_derived_seed(bootstrap_seed, method, reference_method, "delta"),
            unit="paired_task_seed",
        ),
    }


def _ours_pairwise(
    artifacts: Mapping[str, Sequence[_SeedArtifact]],
    *,
    ours_method: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    del bootstrap_samples, bootstrap_seed  # Delta CI is already in transfer reports.
    if ours_method not in artifacts:
        return {
            "status": "unavailable_without_ours_artifacts",
            "ours_method": ours_method,
            "comparisons": {},
            "holm": {"status": "unavailable_without_ours_artifacts"},
        }
    ours = _paired_rows(artifacts[ours_method])
    comparisons: dict[str, dict[str, Any]] = {}
    raw_p: dict[str, float] = {}
    for baseline_method in sorted(artifacts):
        if baseline_method == ours_method:
            continue
        baseline = _paired_rows(artifacts[baseline_method])
        keys = _require_matching_pairs(
            ours, baseline, comparison=f"{ours_method} vs {baseline_method}"
        )
        ours_only = sum(
            ours[key].official_success and not baseline[key].official_success
            for key in keys
        )
        baseline_only = sum(
            not ours[key].official_success and baseline[key].official_success
            for key in keys
        )
        p_value = _mcnemar_exact_two_sided(ours_only, baseline_only)
        raw_p[baseline_method] = p_value
        comparisons[baseline_method] = {
            "status": "available",
            "paired_tasks": len(keys),
            "ours_success_baseline_failure": ours_only,
            "ours_failure_baseline_success": baseline_only,
            "discordant_pairs": ours_only + baseline_only,
            "mcnemar_exact_two_sided_p": p_value,
        }
    if not comparisons:
        return {
            "status": "unavailable_without_baseline_artifacts",
            "ours_method": ours_method,
            "comparisons": {},
            "holm": {"status": "unavailable_without_baseline_artifacts"},
        }
    adjusted = _holm_adjust(raw_p)
    for method, value in adjusted.items():
        comparisons[method]["holm_adjusted_p"] = value
        comparisons[method]["reject_at_0_05"] = value <= 0.05
    return {
        "status": "available",
        "ours_method": ours_method,
        "comparisons": comparisons,
        "holm": {
            "status": "available",
            "alpha": 0.05,
            "number_of_comparisons": len(comparisons),
            "method": "Holm-Bonferroni",
        },
    }


def _mcnemar_exact_two_sided(left_only: int, right_only: int) -> float:
    discordant = int(left_only) + int(right_only)
    if discordant == 0:
        return 1.0
    tail = min(int(left_only), int(right_only))
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (
        2 ** discordant
    )
    return min(1.0, 2.0 * probability)


def _holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * float(p_value)))
        adjusted[name] = running
    return adjusted


def _bootstrap_mean(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
    unit: str,
) -> dict[str, Any]:
    if not values:
        return {"status": "unavailable_without_values"}
    numbers = [float(value) for value in values]
    if any(not math.isfinite(value) for value in numbers):
        raise ValueError("bootstrap values must be finite")
    rng = random.Random(seed)
    size = len(numbers)
    estimates = [
        statistics.fmean(numbers[rng.randrange(size)] for _ in range(size))
        for _ in range(samples)
    ]
    estimates.sort()
    return {
        "status": "available",
        "estimate": statistics.fmean(numbers),
        "lower": _percentile_nearest_rank(estimates, 0.025),
        "upper": _percentile_nearest_rank(estimates, 0.975),
        "samples": samples,
        "unit": unit,
    }


def _percentile_nearest_rank(values: Sequence[float], quantile: float) -> float:
    rank = max(1, math.ceil(float(quantile) * len(values)))
    return float(values[rank - 1])


def _derived_seed(seed: int, *labels: str) -> int:
    payload = ":".join((str(seed), *labels)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(int(numerator) / int(denominator), 6)


def _assert_same_rows(
    evaluated: Sequence[TaskRow], persisted: Sequence[TaskRow], *, root: Path
) -> None:
    if len(evaluated) != len(persisted):
        raise ValueError(f"evaluated episodes and task rows differ in length: {root}")
    for left, right in zip(evaluated, persisted, strict=True):
        if left.to_dict() != right.to_dict():
            raise ValueError(
                f"evaluated episodes disagree with persisted task rows: {root}"
            )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON artifact is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_method_run(value: str) -> tuple[str, Path]:
    method, separator, raw_path = value.partition("=")
    if not separator or not method.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("expected METHOD=COMPLETED_RUN_OR_CAMPAIGN")
    return method.strip(), Path(raw_path).expanduser()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method-run",
        action="append",
        type=_parse_method_run,
        required=True,
        help="repeat METHOD=PATH; PATH may be one test run or a campaign root",
    )
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=list(_DEFAULT_SEEDS))
    parser.add_argument("--b0-method", default="b0_pure_dynamic")
    parser.add_argument("--ours-method", default="ours")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260912)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    grouped: dict[str, list[Path]] = {}
    for method, path in args.method_run:
        grouped.setdefault(method, []).append(path)
    report = build_campaign_report(
        grouped,
        expected_seeds=args.expected_seeds,
        b0_method=args.b0_method,
        ours_method=args.ours_method,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    output = Path(args.output).expanduser().resolve()
    _write_json_atomic(output, report)
    print(json.dumps({"passed": True, "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
