from __future__ import annotations

from types import SimpleNamespace

from experiments.run_v3_smoke import _run_real_smoke_curriculum


class _Rows:
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[dict[str, str]]:
        return list(self._rows)


class _Database:
    def __init__(self) -> None:
        self.rows: list[dict[str, str]] = []

    def execute(self, _query: str) -> _Rows:
        return _Rows(sorted(self.rows, key=lambda item: item["artifact_ref"]))


class _Skills:
    def __init__(self) -> None:
        self.composites: dict[str, object] = {}

    def get_composite(self, ref: str) -> object:
        return self.composites[ref]


class _System:
    def __init__(self, additions: dict[str, list[tuple[dict[str, str], object | None]]]) -> None:
        self.database = _Database()
        self.skills = _Skills()
        self.additions = additions
        self.called: list[str] = []

    def run_task(self, task: object) -> object:
        task_id = str(getattr(task, "task_id"))
        self.called.append(task_id)
        for row, composite in self.additions.get(task_id, []):
            self.database.rows.append(dict(row))
            if composite is not None:
                self.skills.composites[row["artifact_ref"]] = composite
        return SimpleNamespace(
            trace_id=f"trace_{task_id}",
            benchmark_success=True,
            infrastructure_failure=False,
            extraction_policy={"should_extract": True},
            metadata={"extraction": {"prepared": True}},
            failures=[],
            node_records=[],
            implementation_invocations=[],
            tool_executions=[],
            runtime_plan={"source": "full_dynamic"},
            binding_changes=[],
            graph_self_sufficient_success=False,
            task_rescue_required=False,
        )


def _tasks() -> list[object]:
    return [
        SimpleNamespace(task_id=task_id)
        for task_id in (
            "cold-1", "cold-2", "cold-3",
            "warm-1", "warm-2", "multi-1",
        )
    ]


def _row(ref: str, kind: str, status: str = "candidate") -> dict[str, str]:
    return {"artifact_ref": ref, "artifact_kind": kind, "status": status}


def _composite(occurrences: int, data_edges: int) -> object:
    return SimpleNamespace(
        occurrences=[object() for _ in range(occurrences)],
        data_edges=[object() for _ in range(data_edges)],
    )


def _four_layer_candidates(prefix: str) -> list[tuple[dict[str, str], object | None]]:
    return [
        (_row(f"atomic:{prefix}@1.0.0", "atomic"), None),
        (_row(f"implementation:{prefix}@1.0.0", "implementation"), None),
        (_row(f"tool:{prefix}@1.0.0", "tool"), None),
        (
            _row(f"composite:{prefix}@1.0.0", "composite"),
            _composite(1, 0),
        ),
    ]


def test_stage_a_stops_immediately_when_first_cold_task_learns_no_candidate(
    capsys,
) -> None:
    system = _System({})

    result = _run_real_smoke_curriculum(system, _tasks())

    assert result.error_code == "missing_four_layer_candidates_after_cold_task"
    assert system.called == ["cold-1"]
    assert len(result.cold_traces) == 1
    assert result.warm_traces == []
    assert result.multi_traces == []
    assert result.diagnostics[0]["stage"] == "A"
    assert result.diagnostics[0]["new_artifact_refs"] == []
    assert "missing_four_layer_candidates_after_cold_task" in capsys.readouterr().out


def test_stage_b_requires_candidate_and_stage_c_rejects_non_dataflow_assets(
    capsys,
) -> None:
    system = _System({
        "cold-1": _four_layer_candidates("a"),
        # Neither a single-occurrence Composite nor a two-occurrence Composite
        # without an explicit DataFlow edge satisfies the Stage C asset gate.
        "warm-1": [(
            _row("composite:single@1.0.0", "composite"),
            _composite(1, 1),
        )],
        "warm-2": [(
            _row("composite:no-flow@1.0.0", "composite"),
            _composite(2, 0),
        )],
    })

    result = _run_real_smoke_curriculum(system, _tasks())

    assert len(result.candidate_refs_after_cold) == 4
    assert system.called == ["cold-1", "cold-2", "cold-3", "warm-1", "warm-2"]
    assert len(result.warm_traces) == 2
    assert result.multi_traces == []
    assert result.error_code == "no_learned_dataflow_asset"
    assert result.learned_dataflow_assets == []
    assert "no_learned_dataflow_asset" in capsys.readouterr().out


def test_stage_c_runs_only_with_learned_two_occurrence_dataflow_asset() -> None:
    learned = "composite:learned-flow@1.0.0"
    system = _System({
        "cold-1": _four_layer_candidates("i"),
        "warm-2": [(_row(learned, "composite"), _composite(2, 1))],
    })

    result = _run_real_smoke_curriculum(system, _tasks())

    assert result.error_code == ""
    assert system.called == [
        "cold-1", "cold-2", "cold-3", "warm-1", "warm-2", "multi-1",
    ]
    assert len(result.multi_traces) == 1
    assert result.learned_dataflow_assets == [{
        "artifact_ref": learned,
        "status": "candidate",
        "occurrence_count": 2,
        "data_edge_count": 1,
    }]
    assert [
        item["stage"] for item in result.diagnostics
        if item["event"] == "cold_task_extraction"
    ] == ["A", "A", "A"]


def test_stage_c_does_not_count_a_preexisting_dataflow_asset_as_learned() -> None:
    seeded = "composite:preexisting@1.0.0"
    system = _System({"cold-1": _four_layer_candidates("new")})
    system.database.rows.append(_row(seeded, "composite"))
    system.skills.composites[seeded] = _composite(2, 1)

    result = _run_real_smoke_curriculum(system, _tasks())

    assert result.error_code == "no_learned_dataflow_asset"
    assert result.learned_dataflow_assets == []
    assert system.called == ["cold-1", "cold-2", "cold-3", "warm-1", "warm-2"]
