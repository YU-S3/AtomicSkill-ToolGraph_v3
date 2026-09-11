"""Common result aggregation, completeness, and cost-statistic tests."""

from __future__ import annotations

import pytest

from experiments.baselines.common.post_evaluator import TaskRow, summarize_rows


_FAMILIES = [f"family_{index}" for index in range(1, 7)]


def _row(
    index: int,
    *,
    task_type: str | None = None,
    infrastructure_failure: bool = False,
) -> TaskRow:
    success = index % 2 == 1
    return TaskRow(
        method="b3_skillopt",
        phase="final_train_eval",
        run_seed=42,
        task_id=f"task_{index}",
        task_type=task_type or _FAMILIES[index - 1],
        manifest_index=index - 1,
        official_success=success,
        task_contract_success=success,
        strict_success=success,
        environment_actions=index,
        invalid_actions=0,
        target_llm_calls=index,
        target_prompt_tokens=index * 10,
        target_completion_tokens=index,
        target_reasoning_tokens=1,
        evolution_llm_calls=1,
        evolution_prompt_tokens=2,
        evolution_completion_tokens=3,
        embedding_calls=0,
        wall_time_ms=index * 100,
        infrastructure_failure=infrastructure_failure,
    )


def test_six_family_macro_and_cost_distributions_are_complete() -> None:
    summary = summarize_rows(
        [_row(index) for index in range(1, 7)],
        task_types=list(_FAMILIES),
    )

    assert summary["tasks"] == 6
    assert summary["attempted_tasks"] == 6
    assert summary["official_success"] == 3
    assert summary["official_rate"] == 0.5
    assert summary["macro_family_official_rate"] == 0.5
    assert list(summary["family"]) == _FAMILIES

    assert summary["environment_actions"] == 21
    assert summary["actions_per_task"] == 3.5
    assert summary["actions_per_solved"] == 7.0
    assert summary["p50_actions"] == 3
    assert summary["p90_actions"] == 6
    assert summary["invalid_actions_scored_tasks"] == 6
    assert summary["invalid_actions"] == 0
    assert summary["invalid_actions_per_scored_task"] == 0.0

    assert summary["target_llm_calls"] == 21
    assert summary["evolution_llm_calls"] == 6
    assert summary["llm_calls"] == 27
    assert summary["target_calls_per_task"] == 3.5
    assert summary["evolution_calls_per_task"] == 1.0
    assert summary["calls_per_task"] == 4.5
    assert summary["p50_calls"] == 4
    assert summary["p90_calls"] == 7

    assert summary["target_reasoning_tokens"] == 6
    assert summary["target_tokens"] == 231
    assert summary["target_tokens_per_task"] == 38.5
    assert summary["reasoning_tokens_in_completion"] is True
    assert summary["evolution_tokens"] == 30
    assert summary["evolution_tokens_per_task"] == 5.0
    assert summary["llm_tokens"] == 261
    assert summary["tokens_per_task"] == 43.5
    assert summary["tokens_per_solved"] == 87.0
    assert summary["p50_tokens"] == 38
    assert summary["p90_tokens"] == 71

    assert summary["wall_time_ms"] == 2100
    assert summary["latency_per_task_ms"] == 350.0
    assert summary["p50_latency_ms"] == 300
    assert summary["p90_latency_ms"] == 600


def test_macro_is_independent_of_row_order() -> None:
    rows = [_row(index) for index in range(1, 7)]
    forward = summarize_rows(rows, task_types=list(_FAMILIES))
    reverse = summarize_rows(list(reversed(rows)), task_types=list(_FAMILIES))
    assert reverse["macro_family_official_rate"] == forward[
        "macro_family_official_rate"
    ]
    assert reverse["family"] == forward["family"]


def test_missing_scorable_family_fails_closed() -> None:
    with pytest.raises(ValueError, match="no scorable rows"):
        summarize_rows(
            [_row(index) for index in range(1, 6)],
            task_types=list(_FAMILIES),
        )


def test_unknown_family_fails_closed() -> None:
    rows = [_row(index) for index in range(1, 7)]
    rows[-1] = _row(6, task_type="not_declared")
    with pytest.raises(ValueError, match="undeclared task families"):
        summarize_rows(rows, task_types=list(_FAMILIES))


def test_infrastructure_rows_are_reported_but_not_scored() -> None:
    rows = [_row(index) for index in range(1, 7)]
    rows.append(_row(7, task_type=_FAMILIES[0], infrastructure_failure=True))
    summary = summarize_rows(rows, task_types=list(_FAMILIES))
    assert summary["attempted_tasks"] == 7
    assert summary["tasks"] == 6
    assert summary["infrastructure_failed_episodes"] == 1
    assert summary["official_rate"] == 0.5


def test_family_strict_rate_does_not_turn_missing_evidence_into_failure() -> None:
    rows = [_row(index) for index in range(1, 7)]
    first = rows[0]
    rows[0] = TaskRow(
        **{
            **first.to_dict(),
            "task_contract_success": None,
            "strict_success": None,
        }
    )
    summary = summarize_rows(rows, task_types=list(_FAMILIES))
    assert summary["family"][_FAMILIES[0]]["strict_scored_tasks"] == 0
    assert summary["family"][_FAMILIES[0]]["strict_rate"] is None
