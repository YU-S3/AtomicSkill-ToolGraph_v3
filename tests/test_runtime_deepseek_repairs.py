from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from atomic_skillgraph.core.bindings import (
    BindingExpression,
    BindingExprKind,
    BindingResolution,
    BindingSource,
    BindingStatus,
    RuntimeBinding,
)
from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    ContractSource,
    ImplementationAtom,
    ParameterSpec,
    SemanticPredicate,
    TaskContract,
)
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.results import (
    ImplementationInvocationSpec,
    RuntimeLinearPlan,
    RuntimeOccurrence,
)
from atomic_skillgraph.core.status import SkillStatus
from atomic_skillgraph.harness.action_catalog import HarnessActionCatalog
from atomic_skillgraph.harness.alfworld import (
    AlfWorldAdapter,
    AlfWorldValidatorChannel,
    parse_alfworld_action,
    semantic_value_compatible,
)
from atomic_skillgraph.harness.protocol import HarnessActionSpec
from atomic_skillgraph.planner.validator import validate_runtime_plan
from atomic_skillgraph.runtime.binding_store import RuntimeBindingStore
from atomic_skillgraph.runtime.budget import (
    RuntimeBudget,
    required_runtime_turn_caps,
    validate_runtime_turn_caps,
)
from atomic_skillgraph.runtime.invocation_compiler import (
    CompiledInvocation,
    InvocationCompiler,
)
from atomic_skillgraph.runtime.loop_guard import ActionLoopGuard
from atomic_skillgraph.runtime.node_executor import NodeExecutor
from atomic_skillgraph.runtime.evidence_store import GroundingEvidenceStore
from atomic_skillgraph.validation.engine import ValidationEngine
from atomic_skillgraph.validation.task_validator import TaskValidator
from experiments.fakes import FakeHarness, fake_task


def test_runtime_turn_cap_covers_action_budget() -> None:
    assert required_runtime_turn_caps(
        global_action_budget=100,
        node_action_budget=35,
        learned_toolcall_repair_limit=2,
        protocol_repair_limit=1,
    ) == (41, 104)
    assert validate_runtime_turn_caps(
        global_action_budget=100,
        node_action_budget=35,
        learned_toolcall_repair_limit=2,
        protocol_repair_limit=1,
        max_turns_per_node=41,
        max_turns_per_task=104,
    ) == (41, 104)
    with pytest.raises(ValueError, match="max_turns_per_node"):
        validate_runtime_turn_caps(
            global_action_budget=100,
            node_action_budget=35,
            learned_toolcall_repair_limit=2,
            max_turns_per_node=12,
            max_turns_per_task=104,
        )


def test_action_loop_guard_blocks_without_environment_result() -> None:
    catalog = [{"action_type": "LOOK", "arguments": {}}]
    guard = ActionLoopGuard()
    assert not guard.inspect(
        action_type="LOOK", arguments={}, observation="same", catalog=catalog,
    ).blocked
    assert not guard.inspect(
        action_type="LOOK", arguments={}, observation="same", catalog=catalog,
    ).blocked
    third = guard.inspect(
        action_type="LOOK", arguments={}, observation="same", catalog=catalog,
    )
    assert third.blocked
    assert third.reason == "same_action_same_state_three_times"
    payload = third.tool_result()
    assert payload["error"] == "loop_blocked"
    assert payload["environment_called"] is False
    assert payload["action_budget_consumed"] is False
    fourth = guard.inspect(
        action_type="LOOK", arguments={}, observation="same", catalog=catalog,
    )
    assert fourth.blocked and fourth.fallback_required

    alternating = ActionLoopGuard()
    decisions = [
        alternating.inspect(
            action_type=name, arguments={}, observation="same", catalog=catalog,
        )
        for name in ["A", "B", "A", "B", "A", "B"]
    ]
    assert decisions[-1].blocked
    assert decisions[-1].reason == "two_action_cycle_three_rounds"


def test_action_ids_are_revision_scoped_and_stale_ids_are_rejected() -> None:
    catalog = HarnessActionCatalog(parse_alfworld_action)
    first = catalog.replace(["look"], revision=0)[0]
    assert first.action_id == "r000_a001"
    second = catalog.replace(["look"], revision=1)[0]
    assert second.action_id == "r001_a001"
    assert first.action_id != second.action_id
    assert catalog.tool_schema()["properties"]["action_id"]["enum"] == [
        "r001_a001"
    ]
    with pytest.raises(KeyError, match="unknown action_id"):
        catalog.get(first.action_id, revision=1)
    with pytest.raises(KeyError, match="stale action catalog revision"):
        catalog.get(second.action_id, revision=0)


@pytest.mark.parametrize(
    ("concrete", "anchor", "expected"),
    [
        ("apple_2", "apple", True),
        ("mug_1", "apple", False),
        ("fridge_1", "fridge", True),
        ("cabinet_2", "fridge", False),
    ],
)
def test_alfworld_semantic_value_compatibility(
    concrete: str, anchor: str, expected: bool,
) -> None:
    assert semantic_value_compatible(
        role="object",
        concrete_value=concrete,
        semantic_anchor=anchor,
        semantic_type="entity",
    ) is expected


def _compiled_invocation() -> tuple[CompiledInvocation, RuntimeOccurrence]:
    atomic_ref = SkillRef("place_semantic_object", "1.0.0")
    implementation_ref = SkillRef("place_semantic_object_impl", "1.0.0")
    atomic = AbstractAtomicSkill(
        atomic_ref,
        "place the task object",
        [ParameterSpec("object", "entity", required_resolution="concrete")],
        [],
        [],
        [SemanticPredicate("object.at_location", {"object": "$object"})],
        {},
        [],
        {},
        {},
        SkillStatus.ACTIVE,
    )
    implementation = ImplementationAtom(
        implementation_ref,
        atomic_ref,
        [],
        [],
        {"mode": "serial"},
        {"harness_profiles": ["alfworld_v3"]},
        {},
        SkillStatus.ACTIVE,
    )
    spec = ImplementationInvocationSpec(
        "invoke_impl_place_semantic_object",
        implementation_ref,
        atomic_ref,
        "place",
        {
            "type": "object",
            "required": ["object"],
            "additionalProperties": False,
            "properties": {"object": {"type": "string"}},
        },
        [],
        [],
        {"mode": "serial"},
    )
    occurrence = RuntimeOccurrence(
        "s1", "occ1", atomic_ref, [], {
            "object": BindingExpression(
                BindingExprKind.SKILL_INPUT, source_role="object",
            ),
        }, [implementation_ref], atomic.effects,
    )
    return CompiledInvocation(spec, atomic, implementation, []), occurrence


def test_compiled_invocation_uses_short_opaque_native_name() -> None:
    atomic_ref = SkillRef("navigate_source", "1.0.0")
    implementation_ref = SkillRef(
        "impl_navigate_to_the_source_location_of_the_target_1c1b67f5732e",
        "1.0.0",
    )
    atomic = AbstractAtomicSkill(
        atomic_ref, "navigate", [], [], [], [], {}, [], {}, {}, SkillStatus.ACTIVE,
    )
    implementation = ImplementationAtom(
        implementation_ref,
        atomic_ref,
        [],
        [],
        {"mode": "serial"},
        {"harness_profiles": ["alfworld_v3"]},
        {},
        SkillStatus.ACTIVE,
    )

    spec = InvocationCompiler(
        SimpleNamespace(), SimpleNamespace(), SimpleNamespace()
    ).compile(atomic, implementation, [], {})

    assert re.fullmatch(r"invoke_impl_[0-9a-f]{16}", spec.name)
    assert len(spec.name) == 28
    assert implementation_ref.logical_id not in spec.name


def test_wrong_semantic_family_rejected_before_tool_start() -> None:
    compiled, occurrence = _compiled_invocation()

    class Skills:
        def get_implementation(self, _ref):
            return compiled.implementation

    compiler = InvocationCompiler(Skills(), SimpleNamespace(), AlfWorldAdapter())
    bindings = RuntimeBindingStore()
    bindings.bind_task_value("object", "apple", "entity", 0)
    bindings.resolve_occurrence_specs(occurrence, 0)
    evidence = GroundingEvidenceStore()
    evidence.replace_action_catalog(
        [
            HarnessActionSpec(
                "r000_a001", 0, "TAKE", {"object": "mug_1"},
                "take mug 1", "take mug 1", {},
            ),
            HarnessActionSpec(
                "r000_a002", 0, "TAKE", {"object": "apple_2"},
                "take apple 2", "take apple 2", {},
            ),
        ],
        0,
    )
    rejected = compiler.preflight(
        compiled,
        call_name=compiled.spec.name,
        call_id="call_wrong",
        arguments={"object": "mug_1"},
        occurrence=occurrence,
        binding_store=bindings,
        evidence_store=evidence,
        revision=0,
    )
    assert rejected.passed is False
    assert rejected.failure_code == "runtime_semantic_anchor_mismatch"
    assert bindings.snapshot_for_node(occurrence)["object"].value == "apple"

    accepted = compiler.preflight(
        compiled,
        call_name=compiled.spec.name,
        call_id="call_right",
        arguments={"object": "apple_2"},
        occurrence=occurrence,
        binding_store=bindings,
        evidence_store=evidence,
        revision=0,
    )
    assert accepted.passed is True
    assert accepted.normalized_arguments["object"] == "apple_2"


def test_unanchored_destination_does_not_use_global_task_destination() -> None:
    compiled, occurrence = _compiled_invocation()
    compiled.atomic.inputs[0].name = "destination"
    compiled.spec.input_schema["required"] = ["destination"]
    compiled.spec.input_schema["properties"] = {
        "destination": {"type": "string"},
    }
    occurrence.binding_specs = {}

    class Skills:
        def get_implementation(self, _ref):
            return compiled.implementation

    compiler = InvocationCompiler(Skills(), SimpleNamespace(), AlfWorldAdapter())
    bindings = RuntimeBindingStore()
    bindings.bind_task_value("destination", "sidetable", "entity", 0)
    evidence = GroundingEvidenceStore()
    evidence.replace_action_catalog([
        HarnessActionSpec(
            "r000_a001", 0, "GO_TO", {"destination": "cabinet_3"},
            "go to cabinet 3", "go to cabinet 3", {},
        ),
    ], 0)

    result = compiler.preflight(
        compiled,
        call_name=compiled.spec.name,
        call_id="call_source",
        arguments={"destination": "cabinet_3"},
        occurrence=occurrence,
        binding_store=bindings,
        evidence_store=evidence,
        revision=0,
    )

    assert result.passed is True
    assert bindings.semantic_anchor_for(occurrence, "destination") is None
    assert result.normalized_arguments["destination"] == "cabinet_3"


def test_runtime_prompt_binding_categories_check_resolution() -> None:
    store = RuntimeBindingStore()
    store.bind_task_value("object", "apple", "entity", 0)
    store.bind_task_value("destination", "fridge", "entity", 0)
    occurrence = RuntimeOccurrence(
        "s1", "occ1", SkillRef("anchor_projection", "1.0.0"), [],
        {
            "object": BindingExpression(
                BindingExprKind.SKILL_INPUT, source_role="object",
            ),
        }, [], [],
    )
    store.resolve_occurrence_specs(occurrence, 0)
    store.commit_grounded("occ1", {
        "held_object": RuntimeBinding(
            "held_object", "apple_2", "entity",
            BindingSource.HARNESS_EVIDENCE,
            BindingStatus.GROUNDED,
            BindingResolution.RELATION_VERIFIED,
            ["fixture"],
            0,
        ),
    })
    projection = store.runtime_prompt_projection(
        "occ1",
        [
            ParameterSpec("object", "entity", required_resolution="concrete"),
            ParameterSpec("destination", "entity", required_resolution="concrete"),
            ParameterSpec(
                "held_object", "entity",
                required_resolution="relation_verified",
            ),
        ],
    )
    assert projection == {
        "task_semantic_context": {"object": "apple", "destination": "fridge"},
        "occurrence_semantic_anchors": {"object": "apple"},
        "execution_ready_bindings": {"held_object": "apple_2"},
        "missing_or_insufficient_bindings": ["object", "destination"],
    }


def test_environment_action_does_not_auto_commit_binding() -> None:
    harness = FakeHarness()
    task = fake_task("no-auto-commit", "apple_1")
    reset = harness.reset(task)
    changes = []
    bindings = RuntimeBindingStore(on_change=changes.append)
    bindings.seed_task_bindings(
        task,
        TaskContract(source=ContractSource.ADAPTER_DERIVED),
        reset.new_revision,
    )
    initial_change_count = len(changes)
    trace = SimpleNamespace(
        environment_actions=[], native_tool_calls=[], agent_turns=[],
    )
    ctx = SimpleNamespace(
        action_catalog=list(reset.catalog),
        world_revision=reset.new_revision,
        observation=reset.observation,
        budget=RuntimeBudget(global_action_budget=5, node_action_budget=5),
        harness=harness,
        trace_builder=SimpleNamespace(trace=trace),
        binding_store=bindings,
    )

    def update_after_action(result, _history):
        ctx.action_catalog = list(result.catalog)
        ctx.world_revision = result.new_revision
        ctx.observation = result.observation

    ctx.update_after_action = update_after_action
    executor = NodeExecutor(
        SimpleNamespace(), ValidationEngine(), lambda *_args: None,
    )
    action = reset.catalog[0]
    executor._execute_environment_call(
        SimpleNamespace(
            call_id="call_env",
            name="environment_action",
            arguments={"action_id": action.action_id},
        ),
        SimpleNamespace(session_id="session"),
        SimpleNamespace(occurrence_id="occ1"),
        ctx,
        span_id="span",
        origin="runtime_preparation",
        loop_guard=ActionLoopGuard(),
    )
    assert len(changes) == initial_change_count
    assert bindings.snapshot_for_node("occ1") == {}
    assert bindings.runtime_prompt_projection("occ1", [
        ParameterSpec("item", "entity", required_resolution="concrete"),
    ])["task_semantic_context"] == {"item": "apple_1"}
    assert ctx.budget.used_global_actions == 1
    assert trace.environment_actions[0].action_id == "r000_a001"
    assert trace.environment_actions[0].revision == 0
    assert trace.environment_actions[0].action_type == "TAKE"
    assert trace.environment_actions[0].arguments == {"item": "apple_1"}


def test_won_does_not_bypass_task_contract() -> None:
    channel = AlfWorldValidatorChannel()
    contract = TaskContract(
        [SemanticPredicate(
            "object.at_location",
            {"object": "apple", "location": "fridge"},
        )],
        source=ContractSource.ADAPTER_DERIVED,
    )
    channel.won = True
    terminal = TaskValidator().terminal(contract, channel, benchmark_won=True)
    assert terminal.passed is False
    assert terminal.failure_codes == ["benchmark_goal_contract_mismatch"]

    channel.record(
        HarnessActionSpec(
            "r000_a001", 0, "PUT",
            {"object": "apple_2", "destination": "fridge_1"},
            "put apple 2 in/on fridge 1",
            "put apple 2 in/on fridge 1",
            {},
        ),
        accepted=True,
        revision=1,
        done=True,
        won=True,
    )
    assert TaskValidator().terminal(contract, channel, benchmark_won=True).passed


def test_linear_sequence_needs_no_fake_semantic_edge() -> None:
    occurrences = [
        RuntimeOccurrence(
            f"s{index}", f"occ{index}", SkillRef(f"atomic_{index}", "1.0.0"),
            [], {}, [], [],
        )
        for index in range(1, 4)
    ]
    plan = RuntimeLinearPlan(
        "task", "atomic_composition", None,
        occurrences, ["s1", "s2", "s3"], [], [], TaskContract(), {},
    )
    result = validate_runtime_plan(plan)
    assert result.passed
    assert "no_disconnected_occurrence" not in result.checks
