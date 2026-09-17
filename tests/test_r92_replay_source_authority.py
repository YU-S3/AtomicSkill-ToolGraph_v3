from __future__ import annotations
from fixtures.r102 import compile_fixture

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from atomic_skillgraph.core.contracts import (
    ParameterSpec,
    SemanticPredicate,
    ToolAsset,
)
from atomic_skillgraph.core.refs import SkillRef, ToolRef
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.core.status import RuntimeMode, SkillStatus, ToolStatus
from atomic_skillgraph.evolution.admission import Admission
from atomic_skillgraph.evolution.aligner import Aligner, _tool_signature
from atomic_skillgraph.evolution.atomicizer import CanonicalAtomicOccurrence
from atomic_skillgraph.evolution.replay import (
    ReplayCaseResult,
    ReplaySourceAuthority,
    ReplaySourceAuthorityError,
)
from atomic_skillgraph.evolution.tool_compiler import (
    ToolCompiler,
    build_occurrence_replay_case,
)
from atomic_skillgraph.governance.credit import CreditAssigner
from atomic_skillgraph.governance.ledger import EvidenceLedger
from atomic_skillgraph.harness.protocol import HarnessTask
from atomic_skillgraph.knowledge.artifact_store import ArtifactStore
from atomic_skillgraph.knowledge.database import StateDatabase
from atomic_skillgraph.knowledge.skill_registry import SkillRegistry
from atomic_skillgraph.knowledge.tool_registry import ToolRegistry
from atomic_skillgraph.system import AtomicSkillGraphSystem
from atomic_skillgraph.validation.engine import ValidationEngine
from experiments.fakes import FakeHarness, fake_task


_C10_SOURCE_TASK_MISMATCH_FIXTURE = json.loads(
    (
        Path(__file__).with_name("fixtures")
        / "r92_source_task_mismatch_cases.json"
    ).read_text(encoding="utf-8")
)


def _source(task_id: str, trace_id: str, role: str, value: str) -> CanonicalAtomicOccurrence:
    signature = f"signature-{task_id}"
    return CanonicalAtomicOccurrence(
        occurrence_id=f"occ-{task_id}",
        phase_id=f"phase-{task_id}",
        intent="pick_up_object",
        event_start=0,
        event_end=1,
        input_bindings={role: value},
        output_bindings={f"held_{role}": value},
        input_specs=[ParameterSpec(role, "entity")],
        output_specs=[ParameterSpec(f"held_{role}", "entity")],
        preconditions=[],
        effects=[SemanticPredicate("agent.holds", {"object": f"${role}"})],
        action_events=[{
            "event_id": f"event-{task_id}",
            "action_type": "TAKE",
            "arguments": {"item": value},
            "accepted": True,
        }],
        prefix_events=[],
        source_task={
            "task_id": task_id,
            "task_signature": signature,
            "goal": f"take {value}",
            "benchmark": "fake",
            "task_type": "pick",
            "context": {"env_index": 1, "game_file": f"/{task_id}.game"},
            "metadata": {
                "task_signature": signature,
                "env_index": 1,
                "game_file": f"/{task_id}.game",
                "split": "train",
            },
        },
        source_trace_id=trace_id,
        proposed_ref=SkillRef(f"atomic-{task_id}", "1.0.0"),
    )


def _task(task_id: str, value: str) -> HarnessTask:
    signature = f"signature-{task_id}"
    return HarnessTask(
        task_id,
        f"take {value}",
        "fake",
        "pick",
        context={"env_index": 1, "game_file": f"/{task_id}.game"},
        metadata={"task_signature": signature},
    )


def _trace_task(task_id: str, value: str, *, split: str = "train") -> dict:
    signature = f"signature-{task_id}"
    return {
        "task_id": task_id,
        "task_signature": signature,
        "goal": f"take {value}",
        "benchmark": "fake",
        "task_type": "pick",
        "metadata": {
            "task_signature": signature,
            "env_index": 1,
            "game_file": f"/{task_id}.game",
            "split": split,
        },
    }


class _TraceStore:
    def __init__(self, payloads: dict[str, dict]) -> None:
        self.payloads = payloads

    def exists(self, trace_id: str) -> bool:
        return trace_id in self.payloads

    def load_payload(self, trace_id: str) -> dict:
        return copy.deepcopy(self.payloads[trace_id])


def test_c01_exact_reuse_builds_current_canonical_case_without_mutating_old_tool(
    tmp_path,
) -> None:
    database = StateDatabase(tmp_path / "state.sqlite3")
    artifacts = ArtifactStore(tmp_path, database)
    skills = SkillRegistry(artifacts, database)
    tools = ToolRegistry(artifacts, database)
    old_occurrence = _source("task-a", "trace-a", "object", "apple_1")
    old = compile_fixture([old_occurrence])[0]
    old_atomic = replace(old.atomic, status=SkillStatus.ACTIVE)
    old_tool = replace(old.tool, status=ToolStatus.ACTIVE)
    old_implementation = replace(
        old.implementation,
        abstract_ref=old_atomic.ref,
        status=SkillStatus.ACTIVE,
    )
    skills.register_atomic(old_atomic)
    tools.register(old_tool)
    skills.register_implementation(old_implementation)
    old_payload = copy.deepcopy(to_primitive(tools.get(old_tool.ref)))
    old_signature = _tool_signature(old_tool)

    new_occurrence = _source("task-b", "trace-b", "item", "mug_2")
    new_atomic = compile_fixture([new_occurrence])[0].atomic
    system = AtomicSkillGraphSystem.__new__(AtomicSkillGraphSystem)
    system.skills = skills
    system.tools = tools
    system.aligner = Aligner(skills, tools)
    system.mode = RuntimeMode.ONLINE

    reused = system._existing_executable_reuse(
        new_occurrence,
        new_atomic,
        source_task=_task("task-b", "mug_2"),
    )

    assert reused is not None and reused.tool is not None
    assert reused.tool is not old_tool
    assert _tool_signature(reused.tool) == old_signature
    assert reused.tool.tests[0]["source_task"]["task_id"] == "task-b"
    assert reused.tool.tests[0]["trace_id"] == "trace-b"
    assert reused.tool.tests[0]["bindings"] == {"object": "mug_2"}
    assert to_primitive(tools.get(old_tool.ref)) == old_payload
    database.close()


def test_c02_each_case_resolves_to_its_own_trace_and_manifest_task(tmp_path) -> None:
    task_a = _task("task-a", "apple_1")
    task_b = _task("task-b", "mug_2")
    trace_a = SimpleNamespace(
        trace_id="trace-a",
        task=SimpleNamespace(**_trace_task("task-a", "apple_1")),
    )
    trace_b = SimpleNamespace(
        trace_id="trace-b",
        task=SimpleNamespace(**_trace_task("task-b", "mug_2")),
    )
    manifest_path = tmp_path / "task_manifest.json"
    manifest_path.write_text(json.dumps({"tasks": [
        {
            "task_id": task_id,
            "task_signature": f"signature-{task_id}",
            "benchmark": "fake",
            "split": "train",
            "metadata": {
                "task_type": "pick",
                "env_index": 1,
                "game_file": f"/{task_id}.game",
            },
        }
        for task_id in ("task-a", "task-b")
    ]}), encoding="utf-8")
    authority = ReplaySourceAuthority(
        _TraceStore({"trace-a": {"task": _trace_task("task-a", "apple_1")}}),
        allowed_split="train",
        task_manifest_path=manifest_path,
    )
    compiler = ToolCompiler()
    case_a = compile_fixture([_source("task-a", "trace-a", "object", "apple_1")])[0].tool.tests[0]
    case_b = compile_fixture([_source("task-b", "trace-b", "item", "mug_2")])[0].tool.tests[0]

    resolved_a = authority.resolve(
        case_a, current_task=task_b, current_trace=trace_b,
    )
    resolved_b = authority.resolve(
        case_b, current_task=task_b, current_trace=trace_b,
    )

    assert resolved_a.task_id == task_a.task_id
    assert resolved_b.task_id == task_b.task_id
    assert resolved_a.context["game_file"] == "/task-a.game"
    assert resolved_b.context["game_file"] == "/task-b.game"


def test_formal_manifest_supplies_split_when_trace_task_metadata_does_not(
    tmp_path,
) -> None:
    case = compile_fixture([
        _source("task-a", "trace-a", "object", "apple_1")
    ])[0].tool.tests[0]
    case["source_task"]["metadata"].pop("split", None)
    trace_task = _trace_task("task-a", "apple_1")
    trace_task["metadata"] = {"task_signature": "signature-task-a"}
    manifest_path = tmp_path / "task_manifest.json"
    manifest_path.write_text(json.dumps({"tasks": [{
        "task_id": "task-a",
        "task_signature": "signature-task-a",
        "benchmark": "fake",
        "split": "train",
        "metadata": {
            "task_type": "pick",
            "env_index": 1,
            "game_file": "/task-a.game",
        },
    }]}), encoding="utf-8")
    authority = ReplaySourceAuthority(
        _TraceStore({"trace-a": {"task": trace_task}}),
        allowed_split="train",
        task_manifest_path=manifest_path,
    )

    resolved = authority.resolve(case)

    assert resolved.task_id == "task-a"
    assert resolved.context["env_index"] == 1
    assert resolved.context["game_file"] == "/task-a.game"
    assert resolved.metadata["split"] == "train"


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (lambda case: case["source_task"].update(task_signature="wrong"),
         "replay_source_task_conflict"),
        (lambda case: case.update(trace_id="missing-trace"),
         "replay_source_trace_missing"),
    ],
)
def test_c04_c09_source_identity_is_fail_closed(mutation, expected_code) -> None:
    case = compile_fixture([
        _source("task-a", "trace-a", "object", "apple_1")
    ])[0].tool.tests[0]
    mutation(case)
    authority = ReplaySourceAuthority(
        _TraceStore({"trace-a": {"task": _trace_task("task-a", "apple_1")}}),
        allowed_split="train",
    )

    with pytest.raises(ReplaySourceAuthorityError) as caught:
        authority.resolve(case)

    assert caught.value.code == expected_code
    assert caught.value.result.stage == "source_resolution"
    assert caught.value.result.passed is False


def test_c09_disallowed_source_split_does_not_fall_back_to_current_task() -> None:
    case = compile_fixture([
        _source("task-a", "trace-a", "object", "apple_1")
    ])[0].tool.tests[0]
    authority = ReplaySourceAuthority(
        _TraceStore({
            "trace-a": {"task": _trace_task("task-a", "apple_1", split="valid_unseen")},
        }),
        allowed_split="train",
    )
    case["source_task"]["metadata"]["split"] = "valid_unseen"

    with pytest.raises(ReplaySourceAuthorityError) as caught:
        authority.resolve(case)

    assert caught.value.code == "replay_source_split_disallowed"
    assert caught.value.result.resolved_task_id == "task-a"


def test_c08_terminal_prefix_does_not_admit_the_unfinished_tool() -> None:
    task = fake_task("terminal-prefix", "apple_1")
    tool = ToolAsset(
        ref=ToolRef("terminal_prefix_probe", "1.0.0"),
        summary="take the requested item",
        signature={
            "type": "object",
            "properties": {"item": {"type": "string"}},
            "required": ["item"],
            "additionalProperties": False,
        },
        interface={"entry_contract": {"conditions": [], "grounding_constraints": []},
            "output_schema": {
                "type": "object",
                "properties": {"held_object": {"type": "string"}},
                "required": ["held_object"],
                "additionalProperties": False,
            },
        },
        artifact_kind="tool_ir_v1",
        artifact={
            "schema_version": 1,
            "max_actions": 1,
            "program": [
                {
                    "node_id": "take",
                    "op": "ACTION",
                    "action_type": "TAKE",
                    "argument_mapping": {
                        "item": {
                            "kind": "skill_input",
                            "source_role": "item",
                        },
                    },
                    "expected_effects": [{
                        "predicate": "agent.holds",
                        "args": {"object": "$item"},
                        "effect_domain": "world",
                    }],
                },
                {
                    "node_id": "return",
                    "op": "RETURN",
                    "output_sources": {
                        "held_object": {
                            "source": "tool_input",
                            "field": "item",
                        },
                    },
                },
            ],
            "final_effects": [{
                "predicate": "agent.holds",
                "args": {"object": "$item"},
                "effect_domain": "world",
            }],
            "evidence_outputs": [],
            "output_mapping": {
                "held_object": {
                    "kind": "skill_input",
                    "source_role": "item",
                },
            },
        },
        tests=[],
        safety={
            "reviewed": True,
            "allowed_action_types": ["TAKE"],
            "zero_llm": True,
        },
        provenance={},
        metadata={},
        status=ToolStatus.ADMISSION_PENDING,
    )
    case = {
        "kind": "tool_proposal_replay",
        "case_id": "case-terminal-prefix",
        "trace_id": "trace-terminal-prefix",
        "source_task": {
            "task_id": task.task_id,
            "task_signature": task.metadata["task_signature"],
            "goal": task.goal,
            "benchmark": task.benchmark,
            "task_type": task.task_type,
            "context": copy.deepcopy(task.context),
            "metadata": copy.deepcopy(task.metadata),
        },
        "bindings": {"item": "apple_1"},
        "prefix": [{
            "action_type": "TAKE",
            "arguments": {"item": "apple_1"},
        }],
        "effects": copy.deepcopy(tool.artifact["final_effects"]),
    }
    system = AtomicSkillGraphSystem.__new__(AtomicSkillGraphSystem)
    system.harness = FakeHarness()
    system.validation = ValidationEngine()
    system.config = {}

    result = system._replay_tool_candidate_result(
        task,
        tool,
        case,
        requested_task_id="current-task",
    )

    assert result.stage == "prefix"
    assert result.passed is False
    assert result.failure_code == "replay_terminal_prefix"
    assert result.terminal_interrupted is True
    assert result.started is False
    assert result.executed_action_count == 0
    assert system.harness.current_task == task
    admitted = Admission(ValidationEngine().tool).admit_tool(
        replace(tool, tests=[case]),
        replay=lambda _tool, _case: result,
    )
    assert admitted.status is ToolStatus.SHADOW
    assert admitted.metadata["admission_failure"] == [
        "tool_ir_replay_failed"
    ]
    assert admitted.metadata["replay_results"][0][
        "failure_code"
    ] == "replay_terminal_prefix"


@pytest.mark.parametrize("error_type", [TypeError, RuntimeError])
def test_harness_replay_programming_errors_propagate(
    monkeypatch,
    error_type,
) -> None:
    occurrence = _source("task-a", "trace-a", "object", "apple_1")
    compiled = compile_fixture([occurrence])[0]
    opaque_tool = replace(compiled.tool, artifact_kind="opaque_tool")
    task = fake_task("task-a", "apple_1")
    system = AtomicSkillGraphSystem.__new__(AtomicSkillGraphSystem)
    system.harness = FakeHarness()
    system.config = {}

    def fail_replay(*_args, **_kwargs):
        raise error_type("unexpected harness replay failure")

    monkeypatch.setattr(system.harness, "replay_tool", fail_replay)

    with pytest.raises(error_type, match="unexpected harness replay failure"):
        system._replay_tool_candidate_result(
            task,
            opaque_tool,
            compiled.tool.tests[0],
            requested_task_id="current-task",
        )


@pytest.mark.parametrize("error_type", [TypeError, RuntimeError])
def test_tool_runner_programming_errors_propagate(
    monkeypatch,
    error_type,
) -> None:
    from atomic_skillgraph.runtime.tool_runner import ToolRunner

    occurrence = _source("task-a", "trace-a", "object", "apple_1")
    compiled = compile_fixture([occurrence])[0]
    tool_ir = replace(compiled.tool, artifact_kind="tool_ir_v1")
    task = fake_task("task-a", "apple_1")
    system = AtomicSkillGraphSystem.__new__(AtomicSkillGraphSystem)
    system.harness = FakeHarness()
    system.validation = ValidationEngine()
    system.config = {}

    def fail_replay(*_args, **_kwargs):
        raise error_type("unexpected ToolRunner failure")

    monkeypatch.setattr(ToolRunner, "run", fail_replay)

    with pytest.raises(error_type, match="unexpected ToolRunner failure"):
        system._replay_tool_candidate_result(
            task,
            tool_ir,
            compiled.tool.tests[0],
            requested_task_id="current-task",
        )


def test_c10_fixture_reproduces_the_frozen_r9_mismatch_population() -> None:
    fixture = _C10_SOURCE_TASK_MISMATCH_FIXTURE

    assert fixture["source_run"] == "alfworld_train_full_120_r9_seed42"
    assert fixture["selection"]["shadow_tool_count"] == 75
    assert fixture["selection"]["source_task_mismatch_count"] == 59
    assert len(fixture["cases"]) == 59
    assert len({case["tool_ref"] for case in fixture["cases"]}) == 59
    assert all(
        case["source_task_id"] != case["proposed_task_id"]
        for case in fixture["cases"]
    )


@pytest.mark.parametrize(
    "mismatch_case",
    _C10_SOURCE_TASK_MISMATCH_FIXTURE["cases"],
    ids=lambda case: str(case["tool_ref"]),
)
def test_c10_source_authority_routes_without_claiming_replay_success(
    mismatch_case,
) -> None:
    source_task_id = str(mismatch_case["source_task_id"])
    source_task = copy.deepcopy(
        _C10_SOURCE_TASK_MISMATCH_FIXTURE["source_tasks"][source_task_id]
    )
    source_trace_id = str(mismatch_case["source_trace_id"])
    authority = ReplaySourceAuthority(
        _TraceStore({source_trace_id: {"task": source_task}}),
        allowed_split="train",
    )
    current_task = HarnessTask(
        task_id=str(mismatch_case["proposed_task_id"]),
        goal="historical proposed-task audit placeholder",
        benchmark="alfworld",
        task_type="source_routing_audit",
    )
    replay_case = {
        "kind": "tool_proposal_replay",
        "trace_id": source_trace_id,
        "occurrence_id": str(mismatch_case["occurrence_id"]),
        "source_task": source_task,
    }

    routed = authority.resolve(replay_case, current_task=current_task)

    assert routed.task_id == source_task_id
    assert routed.task_id != current_task.task_id
    assert routed.context == source_task["context"]
    assert isinstance(routed, HarnessTask)
    assert not hasattr(routed, "passed")


def test_c07_admission_consumes_typed_passed_and_rejects_truthy_objects() -> None:
    case = {
        "kind": "source_replay",
        "trace_id": "trace-a",
        "source_task": {"task_id": "task-a"},
    }
    typed = ReplayCaseResult(
        case_id="case-a",
        source_trace_id="trace-a",
        source_task_id="task-a",
        requested_task_id="task-b",
        resolved_task_id="task-a",
        stage="final_validation",
        passed=True,
        started=True,
        executed_action_count=1,
        completed=True,
        atomic_effect_passed=True,
        output_validation_passed=True,
    )
    passed, details = Admission._run_replays(
        SimpleNamespace(), [case], lambda _tool, _case: typed,
    )
    assert passed == [True]
    assert details[0]["resolved_task_id"] == "task-a"
    assert details[0]["executed_action_count"] == 1

    failed, invalid = Admission._run_replays(
        SimpleNamespace(), [case], lambda _tool, _case: SimpleNamespace(passed=True),
    )
    assert failed == [False]
    assert invalid[0]["failure_code"] == "replay_result_type_invalid"
    with pytest.raises(TypeError):
        bool(typed)


def test_shared_case_builder_rejects_noncanonical_binding_roles() -> None:
    occurrence = _source("task-a", "trace-a", "item", "apple_1")
    atomic = compile_fixture([occurrence])[0].atomic
    original_occurrence = to_primitive(occurrence)
    mismatched = replace(
        occurrence,
        input_bindings={"unmapped": "apple_1"},
    )

    with pytest.raises(ValueError, match="canonical replay bindings"):
        build_occurrence_replay_case(
            mismatched, atomic, source_task=occurrence.source_task,
        )

    assert to_primitive(occurrence) == original_occurrence


def test_c06_apply_evolution_replays_each_case_once_and_deduplicates_version(
    tmp_path,
) -> None:
    database = StateDatabase(tmp_path / "state.sqlite3")
    artifacts = ArtifactStore(tmp_path, database)
    skills = SkillRegistry(artifacts, database)
    tools = ToolRegistry(artifacts, database)
    compiler = ToolCompiler()

    source_a = _source("task-a", "trace-a", "object", "apple_1")
    compiled_a = compile_fixture([source_a])[0]
    atomic_a = replace(compiled_a.atomic, status=SkillStatus.ACTIVE)
    tool_a = replace(compiled_a.tool, status=ToolStatus.ACTIVE)
    implementation_a = replace(
        compiled_a.implementation,
        abstract_ref=atomic_a.ref,
        status=SkillStatus.ACTIVE,
    )
    skills.register_atomic(atomic_a)
    tools.register(tool_a)
    skills.register_implementation(implementation_a)

    system = AtomicSkillGraphSystem.__new__(AtomicSkillGraphSystem)
    system.skills = skills
    system.tools = tools
    system.aligner = Aligner(skills, tools)
    system.admission = Admission(ValidationEngine().tool)
    system.harness = FakeHarness()
    system.mode = RuntimeMode.ONLINE
    system.credit = CreditAssigner()
    ledger = EvidenceLedger(database)
    system.ledger = ledger
    system.readonly = False
    append_results = []
    system._commit_evidence = lambda events: append_results.append(
        ledger.append_transaction(events)
    )
    system._add_structural_edge = lambda *_args, **_kwargs: None
    repair_proposals = []
    system.repair_store = SimpleNamespace(save=repair_proposals.append)

    source_b = _source("task-b", "trace-b", "item", "mug_2")
    atomic_b = compile_fixture([source_b])[0].atomic
    task_b = _task("task-b", "mug_2")
    compiled_b = system._existing_executable_reuse(
        source_b,
        atomic_b,
        source_task=task_b,
    )
    assert compiled_b is not None

    tasks = {
        "task-a": _task("task-a", "apple_1"),
        "task-b": task_b,
    }
    system._replay_source_authority = lambda: SimpleNamespace(
        resolve=lambda case, **_kwargs: tasks[
            case["source_task"]["task_id"]
        ]
    )
    physical_replays: list[str] = []
    failing_tasks: set[str] = set()

    def physical_replay(task, _tool, case, *, requested_task_id):
        physical_replays.append(task.task_id)
        passed = task.task_id not in failing_tasks
        return ReplayCaseResult(
            case_id=str(case["case_id"]),
            source_trace_id=str(case["trace_id"]),
            source_task_id=str(case["source_task"]["task_id"]),
            requested_task_id=str(requested_task_id),
            resolved_task_id=str(task.task_id),
            stage="final_validation",
            passed=passed,
            failure_code="" if passed else "tool_ir_replay_atomic_effect_failed",
            started=True,
            executed_action_count=1,
            completed=True,
            atomic_effect_passed=passed,
            output_validation_passed=passed,
        )

    system._replay_tool_candidate_result = physical_replay

    def apply_once(compiled=compiled_b, active_task=task_b):
        trace = SimpleNamespace(
            metadata={},
            evidence_event_refs=[],
            trace_id=str(compiled.occurrence.source_trace_id),
            task=SimpleNamespace(task_id=active_task.task_id),
            environment_actions=[],
        )
        prepared = SimpleNamespace(
            compiled=[compiled],
            composite=None,
            source_composite_ref="",
        )
        applied = system._apply_evolution(prepared, trace, active_task)
        system._commit_replay_certificates(trace)
        return applied, trace

    first, first_trace = apply_once()

    assert physical_replays == ["task-b"]
    first_results = first_trace.metadata["tool_replay_results"]
    assert [item["resolved_task_id"] for item in first_results] == [
        "task-b",
    ]
    assert len(first_results) == 1
    assert len(tools.list_refs()) == 1
    first_tool_ref = first["tool_refs"][0]
    assert append_results[-1].inserted_count > 0
    assert first_tool_ref == tool_a.ref
    assert [
        item["source_task"]["task_id"]
        for item in tools.get(first_tool_ref).tests
    ] == ["task-a"]
    assert {item["source_task"]["task_id"] for item in
            tools.tools_with_replay_evidence()[0].tests} == {"task-a", "task-b"}

    duplicate, duplicate_trace = apply_once()

    assert physical_replays == ["task-b"]
    assert duplicate_trace.metadata.get("tool_replay_results", []) == []
    assert duplicate["tool_refs"] == [first_tool_ref]
    assert len(tools.list_refs()) == 1
    assert append_results[-1].inserted_count == 0
    assert append_results[-1].duplicate_count > 0

    source_c = _source("task-c", "trace-c", "item", "plate_3")
    task_c = _task("task-c", "plate_3")
    tasks["task-c"] = task_c
    atomic_c = compile_fixture([source_c])[0].atomic
    compiled_c = system._existing_executable_reuse(
        source_c,
        atomic_c,
        source_task=task_c,
    )
    assert compiled_c is not None
    failing_tasks.add("task-c")

    rejected, rejected_trace = apply_once(compiled_c, task_c)

    assert physical_replays[-1] == "task-c"
    assert len(rejected_trace.metadata["tool_replay_results"]) == 1
    assert rejected_trace.metadata["tool_replay_results"][0]["passed"] is False
    assert rejected["tool_refs"] == [first_tool_ref]
    assert len(tools.list_refs()) == 1
    assert repair_proposals == []
    rejected_event = ledger.records_after(0)[-1].event
    assert rejected_event.event.value == "replay_rejected"
    assert rejected_event.artifact_ref == str(first_tool_ref)
    database.close()
