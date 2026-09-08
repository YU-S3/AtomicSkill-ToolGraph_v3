from __future__ import annotations

import copy
from pathlib import Path

import pytest

from atomic_skillgraph.agents.usage import UsageLedger
from atomic_skillgraph.system import AtomicSkillGraphSystem, load_config
from atomic_skillgraph.traces.schema import TaskRecord, TraceRecord
from experiments.protocol import (
    ProtocolError,
    formal_reasoning_effort_audit,
    hash_config,
    validate_deepseek_formal_llm,
)


ROOT = Path(__file__).resolve().parents[2]
R61_CONFIGS = (
    "alfworld_train_full_30_r6.yaml",
    "alfworld_frozen_eval_60_r6.yaml",
)
FORMAL_STAGES = (
    "planner",
    "runtime",
    "extractor",
    "tool_builder",
    "evolution_repair",
)


@pytest.mark.parametrize("config_name", R61_CONFIGS)
def test_r61_formal_stages_are_explicit_high_without_budget_drift(
    config_name: str,
) -> None:
    config = load_config(ROOT / "configs" / config_name)

    validate_deepseek_formal_llm(config)
    assert formal_reasoning_effort_audit(config) == {
        "configured_reasoning_effort": "high",
        "effective_reasoning_effort": "high",
        "reasoning_effort_source": "explicit_request",
    }
    for stage in FORMAL_STAGES:
        assert config["llm"][stage]["reasoning_effort"] == "high"

    assert config["llm"]["runtime"]["max_total_tokens_per_node"] == 100000
    assert config["llm"]["runtime"]["max_total_tokens_per_task"] == 300000
    assert config["llm"]["extractor"]["max_turns"] == 3
    assert config["llm"]["extractor"]["max_total_tokens_per_task"] == 262144
    assert config["runtime"]["global_action_budget"] == 100
    assert config["runtime"]["node_action_budget"] == 35


def test_r61_train_and_frozen_eval_share_the_exact_llm_protocol() -> None:
    train = load_config(ROOT / "configs" / R61_CONFIGS[0])
    frozen = load_config(ROOT / "configs" / R61_CONFIGS[1])

    assert train["llm"] == frozen["llm"]
    assert hash_config(train["llm"]) == hash_config(frozen["llm"])


@pytest.mark.parametrize("stage", FORMAL_STAGES)
def test_r61_formal_validator_rejects_any_non_high_stage(stage: str) -> None:
    config = load_config(ROOT / "configs" / R61_CONFIGS[0])
    invalid = copy.deepcopy(config)
    invalid["llm"][stage]["reasoning_effort"] = "low"

    with pytest.raises(
        ProtocolError,
        match=rf"llm\.{stage}\.reasoning_effort",
    ):
        validate_deepseek_formal_llm(invalid)


def test_r61_formal_validator_covers_tool_builder_and_extractor_turn_cap() -> None:
    config = load_config(ROOT / "configs" / R61_CONFIGS[0])

    invalid_builder = copy.deepcopy(config)
    invalid_builder["llm"]["tool_builder"]["max_turns"] = 2
    with pytest.raises(ProtocolError, match=r"llm\.tool_builder\.max_turns"):
        validate_deepseek_formal_llm(invalid_builder)

    invalid_extractor = copy.deepcopy(config)
    invalid_extractor["llm"]["extractor"]["max_turns"] = 2
    with pytest.raises(ProtocolError, match=r"llm\.extractor\.max_turns"):
        validate_deepseek_formal_llm(invalid_extractor)


def test_r61_runtime_session_trace_snapshots_are_explicit_high() -> None:
    config = load_config(ROOT / "configs" / R61_CONFIGS[0])
    system = object.__new__(AtomicSkillGraphSystem)
    system.config = config
    system._provider_override = None
    system._provider_cache = {}
    system._observed_sessions = []
    system._current_task_id = "r61-runtime-snapshot"
    system._runtime_turn_caps = (35, 100)
    system.usage = UsageLedger()

    for index, stage in enumerate(
        ("runtime_preparation", "runtime_seeded", "runtime_dynamic")
    ):
        session = system._runtime_session(stage, f"occ-{index}")
        provider = session.snapshot()["provider"]
        assert provider["model"] == "deepseek-v4-flash"
        assert provider["reasoning_effort"] == "high"

    trace = TraceRecord.create(
        TaskRecord(
            task_id="r61-runtime-snapshot",
            benchmark="alfworld",
            goal="fixture",
            task_type="fixture",
            task_signature="fixture-signature",
        ),
        {},
        {},
        {},
    )
    system._attach_external_sessions(trace, system._observed_sessions)

    by_type = {record.session_type: record.snapshot for record in trace.agent_sessions}
    assert set(by_type) == {
        "RuntimePreparationSession",
        "SeededSession",
        "DynamicTaskSession",
    }
    for snapshot in by_type.values():
        assert snapshot["provider"]["model"] == "deepseek-v4-flash"
        assert snapshot["provider"]["reasoning_effort"] == "high"
