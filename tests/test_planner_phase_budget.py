"""Regression for the Planner's shared-dialogue, phase-local budget boundary."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from atomic_skillgraph.core.bindings import BindingExpression, BindingExprKind
from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    ParameterSpec,
    SemanticPredicate,
)
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.status import SkillStatus
from atomic_skillgraph.system import AtomicSkillGraphSystem, load_config
from experiments.fakes import (
    FakeHarness,
    FakeReply,
    ScriptedAgentProvider,
    fake_task,
)
from experiments.protocol import ProtocolError, validate_deepseek_formal_llm


ROOT = Path(__file__).resolve().parents[1]


def test_p1_p1r_p2_p2r_can_exceed_120k_in_total_without_starving_p2r(
    tmp_path: Path,
) -> None:
    """The interrupted Full-30 shape must finish as an Atomic composition."""

    ref = SkillRef("atomic_take", "1.0.0")
    invalid_requirement = {
        "requirements": [{
            "requirement_id": "req_hold",
            "intent": "observe target",
            "desired_effects": [{
                "predicate": "object.observed",
                "args": {"object": "$item"},
            }],
            "expected_inputs": [{"name": "item", "semantic_type": "entity"}],
            "expected_outputs": [],
            "precondition_hints": [],
            "semantic_variants": ["observe item"],
            "required": True,
            "rationale": "Exercise the single bounded P1R stage.",
        }],
        "repeat_blocks": [],
    }
    valid_requirement = {
        "requirements": [{
            "requirement_id": "req_hold",
            "intent": "hold target",
            "desired_effects": [{
                "predicate": "agent.holds",
                "args": {"object": "$item"},
            }],
            "expected_inputs": [{"name": "item", "semantic_type": "entity"}],
            "expected_outputs": [],
            "precondition_hints": [],
            "semantic_variants": ["take item"],
            "required": True,
            "rationale": "Match the verified reusable take capability.",
        }],
        "repeat_blocks": [],
    }

    def workflow(source_role: str) -> dict[str, object]:
        return {
            "steps": [{
                "step_id": "take",
                "occurrence_id": "take_occ",
                "node_ref": str(ref),
                "requirement_instance_ids": ["single::req_hold"],
                "repeat_role_bindings": {},
                "binding_specs": {
                    "item": {
                        "kind": "skill_input",
                        "source_role": source_role,
                    },
                },
            }],
            "control_sequence": ["take"],
            "data_edges": [],
            "dependency_edges": [],
            "requirement_coverage": {"single::req_hold": ["take"]},
        }

    provider = ScriptedAgentProvider([
        FakeReply.structured(
            invalid_requirement, prompt_tokens=29990, completion_tokens=10,
        ),
        FakeReply.structured(
            valid_requirement, prompt_tokens=29990, completion_tokens=10,
        ),
        FakeReply.structured(
            workflow("wrong"), prompt_tokens=39990, completion_tokens=10,
        ),
        FakeReply.structured(
            workflow("item"), prompt_tokens=39990, completion_tokens=10,
        ),
    ])
    atomic = AbstractAtomicSkill(
        ref,
        "take item",
        [ParameterSpec("item", "entity")],
        [],
        [],
        [SemanticPredicate(
            "agent.holds",
            {
                "object": BindingExpression(
                    BindingExprKind.SKILL_INPUT,
                    source_role="item",
                ),
            },
        )],
        {},
        [],
        {},
        {"harness_profiles": ["fake_v3"]},
        SkillStatus.ACTIVE,
    )

    config = {
        "schema_version": 3,
        "method_patch": "3.2",
        "data_dir": str(tmp_path / "data_v3"),
        "trace_data_dir": str(tmp_path / "trace_data"),
        "llm": {
            "provider": "openai_compatible",
            "base_url": "https://example.test",
            "model": "deterministic-phase-budget",
            "api_key_env": "TEST_DEEPSEEK_KEY",
            "planner": {
                "max_turns": 4,
                "max_total_tokens_per_phase": 120000,
            },
        },
        "planner": {
            "max_repeat_count": 4,
            "max_runtime_occurrences": 16,
        },
        "cold_start": {"enabled": False},
        "runtime": {
            "global_action_budget": 100,
            "node_action_budget": 35,
        },
        "experiment": {
            "condition": "full",
            "runtime_mode": "online",
            "freeze_skills": False,
            "allow_long_term_knowledge_writes": True,
            "output_dir": str(tmp_path / "run"),
        },
    }

    with AtomicSkillGraphSystem(
        config,
        harness=FakeHarness(),
        provider={"planner": provider, "default": provider},
    ) as system:
        system.skills.register_atomic(atomic)
        task = fake_task("phase-budget-gate", "apple_1")
        plan = system.planner.build_plan(
            task,
            system.harness,
            initial_observation="initial",
        )

        assert plan.source == "atomic_composition"
        assert plan.planner_audit["workflow_p2r"]
        assert system.usage.total().total_tokens == 140000
        assert {
            key: value["total_tokens"]
            for key, value in system.usage.by_bucket().items()
        } == {
            "planner_p1": 30000,
            "planner_p1_repair": 30000,
            "planner_p2": 40000,
            "planner_p2_repair": 40000,
        }
        assert len(system._observed_sessions) == 1
        planner_snapshot = system._observed_sessions[0].session.snapshot()
        assert [
            (item["usage_bucket"], item["used_total_tokens"])
            for item in planner_snapshot["budget_phases"]
        ] == [
            ("planner_p1", 30000),
            ("planner_p1_repair", 30000),
            ("planner_p2", 40000),
            ("planner_p2_repair", 40000),
        ]
        assert planner_snapshot["budget_scope"] == "usage_bucket"
        assert planner_snapshot["semantic_budget"]["max_turns"] == 4
        assert all(
            item["max_turns"] == 2
            and item["max_total_tokens"] == 120000
            for item in planner_snapshot["budget_phases"]
        )
        assert len(provider.requests) == 4

        cold_snapshot = system._cold_start_session(task, None).snapshot()
        assert cold_snapshot["budget_scope"] == "usage_bucket"
        assert cold_snapshot["budget"]["max_turns"] == 2
        assert cold_snapshot["budget"]["max_total_tokens"] == 120000
        assert cold_snapshot["semantic_budget"]["max_turns"] == 2


@pytest.mark.parametrize(
    "config_name",
    ("alfworld_train_full_30_r6.yaml", "alfworld_frozen_eval_60_r6.yaml"),
)
def test_r6_formal_configs_freeze_planner_budget_per_phase(
    config_name: str,
) -> None:
    config = load_config(ROOT / "configs" / config_name)
    planner = config["llm"]["planner"]

    assert planner["max_total_tokens_per_phase"] == 120000
    assert "max_total_tokens_per_task" not in planner
    validate_deepseek_formal_llm(config)

    invalid = copy.deepcopy(config)
    invalid["llm"]["planner"]["max_total_tokens_per_task"] = 120000
    with pytest.raises(
        ProtocolError,
        match="llm.planner.max_total_tokens_per_task is forbidden",
    ):
        validate_deepseek_formal_llm(invalid)
