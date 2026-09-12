"""Read-only cross-method campaign reporting from persisted episode rows."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.baselines.common.post_evaluator import (
    TaskRow,
    write_evaluated_episodes_jsonl,
    write_rows_jsonl,
)
from experiments.baselines.common.schema import CommonEpisodeRecord
from experiments.baselines.report_campaign import build_campaign_report


_FAMILIES = [f"family_{index}" for index in range(6)]


def _write_test_run(
    root: Path,
    *,
    method: str,
    seed: int,
    successes: list[bool],
    api_cost: float | None = None,
) -> Path:
    (root / "test").mkdir(parents=True)
    episodes: list[CommonEpisodeRecord] = []
    for index, success in enumerate(successes):
        episode = CommonEpisodeRecord(
            method=method,
            phase="test",
            run_seed=seed,
            task_id=f"task_{index}",
            task_type=_FAMILIES[index],
            manifest_index=index,
            gamefile=f"game_{index}.tw-pddl",
            gamefile_hash=f"{index + 1:064x}",
            official_success=success,
            environment_actions=index + 1,
            invalid_actions=0,
            target_llm_calls=1,
            target_prompt_tokens=10,
            target_completion_tokens=2,
            wall_time_ms=100,
        )
        episode.set_posthoc_outcome(contract_consistency=True)
        episodes.append(episode)
    write_evaluated_episodes_jsonl(
        episodes, root / "test" / "evaluated_common_episodes.jsonl"
    )
    write_rows_jsonl(
        [TaskRow.from_episode(episode) for episode in episodes],
        root / "test" / "task_rows.jsonl",
    )
    (root / "completion.json").write_text(
        json.dumps({"passed": True, "phase": "test"}), encoding="utf-8"
    )
    (root / "test_report.json").write_text(
        json.dumps(
            {
                "passed": True,
                "phase": "test",
                "method": method,
                "test_cost": {
                    "api_cost": api_cost,
                    "api_cost_unpriced": api_cost is None,
                },
            }
        ),
        encoding="utf-8",
    )
    return root


def _runs(tmp_path: Path, method: str, successes: list[bool]) -> list[Path]:
    return [
        _write_test_run(
            tmp_path / method / f"seed_{seed}",
            method=method,
            seed=seed,
            successes=successes,
        )
        for seed in (42, 43, 44)
    ]


def test_report_recomputes_deterministic_paired_statistics(tmp_path: Path) -> None:
    b0 = _runs(tmp_path, "b0_pure_dynamic", [False, True, False, True, False, True])
    ours = _runs(tmp_path, "ours", [True, True, False, False, False, True])
    method_runs = {"b0_pure_dynamic": b0, "ours": ours}

    first = build_campaign_report(
        method_runs,
        expected_seeds=(42, 43, 44),
        task_types=_FAMILIES,
        bootstrap_samples=200,
        bootstrap_seed=7,
    )
    second = build_campaign_report(
        method_runs,
        expected_seeds=(42, 43, 44),
        task_types=_FAMILIES,
        bootstrap_samples=200,
        bootstrap_seed=7,
    )

    assert first == second
    ours_report = first["methods"]["ours"]
    assert ours_report["pooled"]["official_success_rate"] == 0.5
    assert ours_report["mean_std"]["official_success_rate"] == {
        "status": "available",
        "mean": 0.5,
        "std": 0.0,
        "std_ddof": 1,
        "values_by_seed": {"42": 0.5, "43": 0.5, "44": 0.5},
    }
    transfer = ours_report["transfer_vs_b0"]
    assert transfer["positive_transfer_count"] == 3
    assert transfer["negative_transfer_count"] == 3
    assert transfer["positive_transfer"] == pytest.approx(1 / 6, abs=1e-6)
    assert transfer["negative_transfer"] == pytest.approx(1 / 6, abs=1e-6)
    comparison = first["ours_pairwise"]["comparisons"]["b0_pure_dynamic"]
    assert comparison["discordant_pairs"] == 6
    assert comparison["mcnemar_exact_two_sided_p"] == 1.0
    assert comparison["holm_adjusted_p"] == 1.0
    assert len(first["paired_task_rows"]) == 36
    assert ours_report["pooled"]["cost_per_task"] is None
    assert (
        ours_report["bootstrap_95_ci"]["cost_per_task"]["status"]
        == "unavailable_without_pricing_authority"
    )


def test_missing_pairing_authorities_are_explicit(tmp_path: Path) -> None:
    runs = _runs(tmp_path, "b5_gepa", [True, False, False, True, False, True])
    report = build_campaign_report(
        {"b5_gepa": runs},
        expected_seeds=(42, 43, 44),
        task_types=_FAMILIES,
        bootstrap_samples=20,
    )

    assert (
        report["methods"]["b5_gepa"]["transfer_vs_b0"]["status"]
        == "unavailable_without_b0_artifacts"
    )
    assert report["ours_pairwise"]["status"] == "unavailable_without_ours_artifacts"


def test_report_rejects_task_row_drift(tmp_path: Path) -> None:
    runs = _runs(tmp_path, "ours", [True, False, False, True, False, True])
    rows_path = runs[0] / "test" / "task_rows.jsonl"
    payloads = [json.loads(line) for line in rows_path.read_text().splitlines()]
    payloads[0]["official_success"] = False
    rows_path.write_text(
        "".join(json.dumps(payload) + "\n" for payload in payloads),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid|disagree"):
        build_campaign_report(
            {"ours": runs},
            expected_seeds=(42, 43, 44),
            task_types=_FAMILIES,
            bootstrap_samples=20,
        )
