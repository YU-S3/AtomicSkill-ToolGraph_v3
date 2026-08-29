from __future__ import annotations

from pathlib import Path

import pytest

from atomic_skillgraph.agents.provider import _parse_usage
from atomic_skillgraph.agents.usage import AgentBudget, BudgetTracker, LLMUsage
from atomic_skillgraph.core.contracts import ToolAsset
from atomic_skillgraph.core.errors import ArtifactIntegrityError, BudgetExhausted, FailureLayer
from atomic_skillgraph.core.refs import ToolRef
from atomic_skillgraph.core.status import ToolStatus
from atomic_skillgraph.system import AtomicSkillGraphSystem, load_config
from atomic_skillgraph.runtime.budget import RuntimeBudget
from experiments.protocol import ProtocolError, hash_code
from experiments.report import validate_formal_usage
from experiments.run_v3_frozen_eval import _validate_formal_config as _validate_frozen_config
from experiments.run_v3_train import _validate_formal_config


def _config() -> dict:
    return {
        "schema_version": 3,
        "llm": {
            "provider": "openai_compatible",
            "base_url": "https://example.test/v1",
            "model": "model",
            "api_key_env": "MODEL_API_KEY",
        },
        "experiment": {"condition": "full"},
    }


def _system_config(data_dir: Path) -> dict:
    config = _config()
    config["data_dir"] = str(data_dir)
    config["experiment"].update({
        "runtime_mode": "online",
        "freeze_skills": False,
        "output_dir": str(data_dir.parent),
    })
    return config


@pytest.mark.parametrize(
    "name",
    ["x-api-key", "subscription-key", "access-token", "Authorization", "cookie"],
)
def test_config_rejects_every_inline_auth_alias(name: str) -> None:
    config = _config()
    config["llm"][name] = "must-not-be-stored"
    with pytest.raises(ValueError, match="api_key_env only"):
        load_config(config)


def test_provider_usage_is_required_but_reasoning_may_be_unavailable() -> None:
    with pytest.raises(ValueError, match="missing required provider usage"):
        _parse_usage({"prompt_tokens": 2, "completion_tokens": 1})
    usage, metadata = _parse_usage({
        "prompt_tokens": 2,
        "completion_tokens": 1,
        "total_tokens": 3,
    })
    assert usage["reasoning_tokens"] is None
    assert metadata["usage_status"] == "reported"
    assert metadata["reasoning_tokens_status"] == "unavailable"


def test_reasoning_metering_never_changes_visible_budget_control_flow() -> None:
    budget = AgentBudget(2, 100, "runtime_node_token_budget_exhausted", 5)
    first = BudgetTracker(budget)
    second = BudgetTracker(budget)
    first.consume(LLMUsage(1, 5, 6, reasoning_tokens=0, call_count=1))
    second.consume(LLMUsage(1, 5, 6, reasoning_tokens=5, call_count=1))
    assert first.snapshot() == second.snapshot()


def test_task_level_dynamic_keeps_global_usage_but_leaves_node_quota() -> None:
    budget = RuntimeBudget(global_action_budget=3, node_action_budget=1)
    budget.begin_node("occ_last")
    budget.consume_action()
    with pytest.raises(BudgetExhausted) as node_failure:
        budget.consume_action()
    assert getattr(node_failure.value, "code", "") == "runtime_node_action_budget_exhausted"

    budget.end_node()
    assert budget.snapshot()["node_budget_active"] is False
    budget.consume_action()
    budget.consume_action()
    with pytest.raises(BudgetExhausted) as global_failure:
        budget.consume_action()
    assert getattr(global_failure.value, "code", "") == "episode_action_budget_exhausted"

    tracker = BudgetTracker(AgentBudget(0, 1, "runtime_node_token_budget_exhausted"))
    with pytest.raises(BudgetExhausted) as token_failure:
        tracker.check_before_call()
    assert token_failure.value.code == "runtime_node_token_budget_exhausted"
    assert token_failure.value.layer is FailureLayer.RUNTIME_AGENT


@pytest.mark.parametrize(
    ("online", "last_maintenance", "interval", "expected"),
    [
        (3, 0, 5, ""),
        (4, 0, 5, "online_success_5"),
        (9, 5, 5, "online_success_10"),
        (8, 5, 5, ""),
    ],
)
def test_periodic_expectation_uses_pre_task_authoritative_counters(
    online: int, last_maintenance: int, interval: int, expected: str,
) -> None:
    system = object.__new__(AtomicSkillGraphSystem)
    system._online_successes = online
    system._last_maintenance_success_count = last_maintenance
    system.maintenance_interval = interval
    assert system.expected_periodic_maintenance_milestone_after_success() == expected


def test_code_hash_ignores_mutable_run_outputs(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    before = hash_code(tmp_path)
    (tmp_path / "runs" / "formal" / "reports").mkdir(parents=True)
    (tmp_path / "runs" / "formal" / "run_manifest.json").write_text(
        '{"mutable":true}', encoding="utf-8"
    )
    (tmp_path / "runs" / "formal" / "reports" / "result.json").write_text(
        '{"tasks":30}', encoding="utf-8"
    )
    (tmp_path / "src" / "atomic_skillgraph.egg-info").mkdir()
    (tmp_path / "src" / "atomic_skillgraph.egg-info" / "SOURCES.txt").write_text(
        "generated packaging metadata\n", encoding="utf-8"
    )
    (tmp_path / "build" / "lib").mkdir(parents=True)
    (tmp_path / "build" / "lib" / "module.py").write_text(
        "GENERATED = True\n", encoding="utf-8"
    )
    assert hash_code(tmp_path) == before


@pytest.mark.parametrize(
    ("sql", "parameters"),
    [
        (
            "INSERT INTO artifact_index VALUES(?,?,?,?,?,?,?,?)",
            ("atomic:probe@1", "atomic", "probe", "1", "hash", "candidate", "missing", 3),
        ),
        (
            "INSERT INTO recommended_pointers VALUES(?,?)",
            ("probe", "atomic:probe@1"),
        ),
        (
            "INSERT INTO graph_edges VALUES(?,?,?,?,?)",
            ("edge", "atomic:a@1", "atomic:b@1", "contains", "{}"),
        ),
        (
            "INSERT INTO evidence_events VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "event", 3, "task", "trace", "occ", "attempt", 0,
                "atomic:probe@1", "atomic", "selected", "", 1.0, "{}",
            ),
        ),
        (
            "INSERT INTO lifecycle_projection VALUES(?,?,?)",
            ("atomic:probe@1", "{}", 0),
        ),
        (
            "INSERT INTO projection_checkpoints VALUES(?,?)",
            ("lifecycle", 0),
        ),
        (
            "INSERT INTO metadata VALUES(?,?)",
            ("online_success_count", "0"),
        ),
    ],
)
def test_fresh_bank_gate_covers_every_long_term_table(
    tmp_path: Path,
    sql: str,
    parameters: tuple[object, ...],
) -> None:
    with AtomicSkillGraphSystem(_system_config(tmp_path / "data_v3")) as system:
        assert system.is_empty_knowledge_bank() is True
        system.database.execute(sql, parameters)
        system.database.connection.commit()
        assert system.is_empty_knowledge_bank() is False
        checks = system.preflight(
            require_api_key=False,
            initialize_harness=False,
            require_empty_bank=True,
        )
        assert checks["empty_bank"] is False
        assert checks["bank_protocol"] is False
        assert checks["passed"] is False


def test_fresh_bank_gate_rejects_unindexed_artifact_file(tmp_path: Path) -> None:
    with AtomicSkillGraphSystem(_system_config(tmp_path / "data_v3")) as system:
        assert system.is_empty_knowledge_bank() is True
        orphan = system.artifacts.root / "atomic" / "orphan.json"
        orphan.write_text("{}", encoding="utf-8")
        assert system.is_empty_knowledge_bank() is False


def test_tool_body_is_covered_by_immutable_artifact_verification(tmp_path: Path) -> None:
    data_dir = tmp_path / "data_v3"
    with AtomicSkillGraphSystem(_system_config(data_dir)) as system:
        tool = ToolAsset(
            ToolRef("body_probe", "1.0.0"), "probe", {}, {}, "python",
            {"filename": "tool.py"}, [], {}, {}, {}, ToolStatus.DRAFT,
        )
        system.tools.register(tool, body="VALUE = 1\n")
        system.artifacts.verify_all()
        body_path = Path(system.tools.body_path(tool.ref) or "")
        body_path.write_text("VALUE = 2\n", encoding="utf-8")
        with pytest.raises(ArtifactIntegrityError, match="body hash mismatch"):
            system.artifacts.verify_all()
    with pytest.raises(ArtifactIntegrityError, match="body hash mismatch"):
        AtomicSkillGraphSystem(_system_config(data_dir))


def test_existing_incomplete_database_schema_fails_closed_at_startup(tmp_path: Path) -> None:
    data_dir = tmp_path / "data_v3"
    with AtomicSkillGraphSystem(_system_config(data_dir)) as system:
        system.database.execute("DROP TABLE graph_edges")
        system.database.connection.commit()
        checks = system.preflight(
            require_api_key=False,
            initialize_harness=False,
            require_empty_bank=False,
        )
        assert checks["database_schema"] is False
        assert checks["passed"] is False

    with pytest.raises(RuntimeError, match="missing required v3 tables"):
        AtomicSkillGraphSystem(_system_config(data_dir))


def test_formal_train_max_task_attempts_config_is_strict_positive_integer() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs" / "alfworld_train_full_30.yaml"
    output_dir = repo_root / "runs" / "alfworld_train_full_30"
    config = load_config(config_path)
    assert config["experiment"]["max_task_attempts"] == 3
    _validate_formal_config(config, output_dir)

    for invalid in (None, 0, -1, True, 1.5, "3"):
        changed = load_config(config_path)
        changed["experiment"]["max_task_attempts"] = invalid
        with pytest.raises(ProtocolError, match="max_task_attempts"):
            _validate_formal_config(changed, output_dir)

    changed = load_config(config_path)
    changed["trace_data_dir"] = str(output_dir / "different-trace-root")
    with pytest.raises(ProtocolError, match="trace_data_dir"):
        _validate_formal_config(changed, output_dir)


def test_formal_frozen_max_task_attempts_config_is_strict_positive_integer() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs" / "alfworld_frozen_eval.yaml"
    output_dir = repo_root / "runs" / "alfworld_frozen_eval_60"
    config = load_config(config_path)
    assert config["experiment"]["max_task_attempts"] == 3
    _validate_frozen_config(config, output_dir)

    for invalid in (None, 0, -1, True, 1.5, "3"):
        changed = load_config(config_path)
        changed["experiment"]["max_task_attempts"] = invalid
        with pytest.raises(ProtocolError, match="max_task_attempts"):
            _validate_frozen_config(changed, output_dir)

    changed = load_config(config_path)
    changed["trace_data_dir"] = str(output_dir / "different-trace-root")
    with pytest.raises(ProtocolError, match="trace_data_dir"):
        _validate_frozen_config(changed, output_dir)


def test_formal_usage_allows_true_zero_llm_and_rejects_partial_metering() -> None:
    zero_llm = {
        "trace_id": "zero",
        "agent_turns": [],
        "llm_usage": [],
        "metadata": {"usage_reconciliation": {"episode_total_tokens": 0}},
    }
    assert validate_formal_usage([zero_llm])["token_mismatch"] == 0

    partial = {
        "trace_id": "partial",
        "agent_turns": [{"turn_index": 0}],
        "llm_usage": [{
            "bucket": "planner_p1",
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
            "call_count": 1,
            "latency_ms": 0,
            "provider_metadata": {"usage_status": "partial"},
        }],
    }
    with pytest.raises(ValueError, match="unavailable/partial"):
        validate_formal_usage([partial])
