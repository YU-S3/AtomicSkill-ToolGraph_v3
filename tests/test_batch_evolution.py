from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from atomic_skillgraph.agents import AgentProviderError, StructuredSubmissionClient
from atomic_skillgraph.core.bindings import (
    BindingExprKind, BindingExpression, GroundingConstraint,
    GroundingConstraintKind,
)
from atomic_skillgraph.core.contracts import (
    CompositeOccurrence, CompositeSkill, SemanticPredicate,
)
from atomic_skillgraph.core.edges import GraphEdge, GraphEdgeType, GlobalRelationType
from atomic_skillgraph.core.errors import (
    AgentProtocolError, BudgetExhausted, FailureEnvelope, FailureLayer,
)
from atomic_skillgraph.core.refs import SkillRef, ToolRef, content_hash
from atomic_skillgraph.core.results import ValidationResult
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.core.status import SkillStatus, ToolStatus
from atomic_skillgraph.evolution.admission import Admission
from atomic_skillgraph.evolution.aligner import Aligner
from atomic_skillgraph.evolution.maintenance import (
    BatchMaintenanceResult, EvolutionMaintenance,
)
from atomic_skillgraph.evolution.extractor_session import ExtractionContentError
from atomic_skillgraph.evolution.repair import RepairProposal, RepairStore
from atomic_skillgraph.evolution.repair_session import (
    EvolutionToolCandidateProposal,
    EvolutionToolEditProposal,
)
from atomic_skillgraph.evolution.tool_compiler import ToolCompiler
from atomic_skillgraph.evolution.trace_replay import (
    TraceRepairExecutor,
    build_trace_repair_evidence,
)
from atomic_skillgraph.evolution.typed_repair_session import TypedRepairReview
from atomic_skillgraph.evolution.typed_repairs import RepairEvidence
from atomic_skillgraph.knowledge.artifact_store import ArtifactStore
from atomic_skillgraph.knowledge.database import StateDatabase
from atomic_skillgraph.knowledge.skill_registry import SkillRegistry
from atomic_skillgraph.knowledge.tool_registry import ToolRegistry
from atomic_skillgraph.governance.ledger import (
    EvidenceEvent, EvidenceEventType, EvidenceLedger,
)
from atomic_skillgraph.harness.protocol import HarnessTask
from atomic_skillgraph.planner.related_composite import RelatedCompositeHintFinder
from atomic_skillgraph.system import AtomicSkillGraphSystem
from atomic_skillgraph.traces.schema import (
    ImplementationInvocationRecord, TaskRecord, TraceRecord,
)
from atomic_skillgraph.validation.engine import ValidationEngine
from experiments.fakes import FakeHarness, FakeReply, ScriptedAgentProvider, fake_task

from test_evolution_guards import _take_canonical
from test_composite_repairs import _composite as _repair_composite


def _replay_case(trace_id: str, effect: str = "agent.holds") -> dict:
    return {
        "kind": "source_replay",
        "trace_id": trace_id,
        "bindings": {"item": "apple_1"},
        "source_task": {
            "task_id": f"task_{trace_id}",
            "goal": "take apple",
            "benchmark": "fake",
            "task_type": "pick",
            "context": {},
            "metadata": {},
        },
        "prefix": [],
        "effects": [to_primitive(SemanticPredicate(effect, {"object": "apple_1"}))],
    }


def _system_config(data_dir) -> dict:
    return {
        "schema_version": 3,
        "data_dir": str(data_dir),
        "llm": {
            "provider": "openai_compatible",
            "base_url": "https://example.test/v1",
            "model": "model",
            "api_key_env": "MODEL_API_KEY",
        },
        "experiment": {
            "condition": "full",
            "runtime_mode": "online",
            "freeze_skills": False,
            "output_dir": str(data_dir.parent),
        },
    }


def _candidate(
    *,
    steps: list[dict],
    cases: tuple[dict, ...],
    step_indexes: tuple[int, ...],
    suffix: str,
) -> EvolutionToolCandidateProposal:
    return EvolutionToolCandidateProposal(
        "split candidate",
        {"type": "object", "properties": {}, "required": []},
        {
            "output_schema": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            }
        },
        "primitive_ir",
        {"steps": steps, "output_mapping": {}},
        {"reviewed": True},
        cases,
        step_indexes,
        suffix,
    )


def test_tool_split_requires_disjoint_step_and_effect_partitions() -> None:
    steps = [
        {"action_type": "TAKE", "argument_mapping": {}},
        {"action_type": "PUT", "argument_mapping": {}},
    ]
    effects = [
        to_primitive(SemanticPredicate("agent.holds", {"object": "apple_1"})),
        to_primitive(SemanticPredicate(
            "object.at_location", {"object": "apple_1", "location": "bowl_1"},
        )),
    ]
    review = {
        "review_id": "review_split",
        "eligible_operations": ["split"],
        "target_refs": ["tool://compound@1.0.0"],
        "tools": [{
            "ref": "tool://compound@1.0.0",
            "artifact": {"steps": steps, "output_mapping": {}},
        }],
        "source_cases": [{
            "case_id": "case_source",
            "target_refs": ["tool://compound@1.0.0"],
            "case": {**_replay_case("source"), "effects": effects},
        }],
    }
    valid = EvolutionToolEditProposal(
        "review_split",
        "split",
        ("tool://compound@1.0.0",),
        (
            _candidate(
                steps=[steps[0]],
                cases=({"case_id": "case_source", "effect_indexes": [0]},),
                step_indexes=(0,),
                suffix="take",
            ),
            _candidate(
                steps=[steps[1]],
                cases=({"case_id": "case_source", "effect_indexes": [1]},),
                step_indexes=(1,),
                suffix="put",
            ),
        ),
        "two independently replayable primitive boundaries",
    )
    EvolutionMaintenance._validate_agent_proposal(valid, review)

    overlapping_effect = replace(
        valid,
        candidates=(
            valid.candidates[0],
            replace(
                valid.candidates[1],
                source_cases=({
                    "case_id": "case_source",
                    "effect_indexes": [0, 1],
                },),
            ),
        ),
    )
    with pytest.raises(ValueError, match="effect partitions overlap"):
        EvolutionMaintenance._validate_agent_proposal(overlapping_effect, review)

    overlapping_step = replace(
        valid,
        candidates=(
            valid.candidates[0],
            replace(
                valid.candidates[1],
                artifact={"steps": [steps[0]], "output_mapping": {}},
                source_step_indexes=(0,),
            ),
        ),
    )
    with pytest.raises(ValueError, match="step spans must be disjoint"):
        EvolutionMaintenance._validate_agent_proposal(overlapping_step, review)


def _cluster(
    task_id: str,
    trace_id: str,
    attempt_id: str,
    *,
    failure_code: str = "tool_primitive_rejected",
    started: bool = True,
    intrinsic: bool = True,
) -> dict:
    input_types = {"item": "string"}
    harness_context = {"profile": "fake_v3", "task_type": "pick", "split": "train"}
    parameter_constraints: list[dict] = []
    failure_replay = _replay_case(trace_id)
    failure_replay["failure_replay"] = True
    result = {
        "failure_code": failure_code,
        "task_id": task_id,
        "trace_id": trace_id,
        "attempt_id": attempt_id,
        "started": started,
        "intrinsic_tool_failure": intrinsic,
        "input_semantic_types": input_types,
        "harness_context": harness_context,
        "parameter_constraints": parameter_constraints,
        "failure_replay_case": failure_replay,
    }
    result["cluster_key"] = content_hash({
        "failure_code": failure_code,
        "input_semantic_types": input_types,
        "harness_context": harness_context,
        "parameter_constraints": parameter_constraints,
    })
    return result


def _save_failure(
    store: RepairStore,
    target_ref: str,
    failure_id: str,
    cluster: dict,
) -> None:
    store.save(RepairProposal.create(
        target_ref,
        "tool",
        "add_tool_test",
        {
            "diagnostic": {"failure_clusters": [cluster]},
            "requires_concrete_patch": True,
        },
        [failure_id],
    ))


def test_tool_specialization_requires_independent_homogeneous_intrinsic_cluster(
    tmp_path,
) -> None:
    database = StateDatabase(tmp_path / "state.sqlite3")
    artifacts = ArtifactStore(tmp_path, database)
    tools = ToolRegistry(artifacts, database)
    tool = replace(
        ToolCompiler().compile([_take_canonical()])[0].tool,
        status=ToolStatus.CANDIDATE,
    )
    tools.register(tool)
    store = RepairStore(database)
    maintenance = EvolutionMaintenance(store)

    # Agent/preflight parameter failures never become Tool specialization evidence.
    agent_failure = FailureEnvelope(
        "agent", FailureLayer.RUNTIME_AGENT, "runtime_agent_schema_error",
        "task_a", "trace_a", "occ", "attempt_a", True, [str(tool.ref)],
    )
    preflight_failure = FailureEnvelope(
        "preflight", FailureLayer.TOOL, "tool_preflight_rejected",
        "task_b", "trace_b", "occ", "attempt_b", False, [str(tool.ref)],
    )
    assert maintenance._tool_failure_cluster(
        agent_failure, trace=None, tools=tools, skills=None, harness_profile="fake_v3",
    ) is None
    assert maintenance._tool_failure_cluster(
        preflight_failure, trace=None, tools=tools, skills=None, harness_profile="fake_v3",
    ) is None

    repeated = _cluster("task_1", "trace_1", "attempt_1")
    _save_failure(store, str(tool.ref), "failure_1", repeated)
    _save_failure(store, str(tool.ref), "failure_2", repeated)
    assert not any(item["kind"] == "specialize" for item in maintenance.build_batch_reviews(tools))

    heterogeneous = _cluster(
        "task_2", "trace_2", "attempt_2", failure_code="tool_execution_error",
    )
    _save_failure(store, str(tool.ref), "failure_3", heterogeneous)
    assert not any(item["kind"] == "specialize" for item in maintenance.build_batch_reviews(tools))

    independent = _cluster("task_2", "trace_2", "attempt_2")
    _save_failure(store, str(tool.ref), "failure_4", independent)
    reviews = [
        item for item in maintenance.build_batch_reviews(tools)
        if item["kind"] == "specialize"
    ]
    assert len(reviews) == 1
    assert reviews[0]["stable_failure_cluster"]["task_ids"] == ["task_1", "task_2"]
    failure_cases = [
        item for item in reviews[0]["source_cases"]
        if item["case"].get("failure_replay")
    ]
    assert len(failure_cases) == 2
    database.close()


def test_batch_duplicate_detection_replays_admits_new_version_and_empties_queue(
    tmp_path,
) -> None:
    database = StateDatabase(tmp_path / "state.sqlite3")
    artifacts = ArtifactStore(tmp_path, database)
    skills = SkillRegistry(artifacts, database)
    tools = ToolRegistry(artifacts, database)
    primary = replace(
        ToolCompiler().compile([_take_canonical()])[0].tool,
        status=ToolStatus.CANDIDATE,
    )
    alias = replace(
        primary,
        ref=ToolRef("tool_take_alias", "1.0.0"),
        status=ToolStatus.CANDIDATE,
    )
    tools.register(primary)
    tools.register(alias)
    store = RepairStore(database)
    result = EvolutionMaintenance(store).run_batch(
        maintenance_trace_id="trace_maintenance",
        reviews=[],
        agent_proposals=[],
        tools=tools,
        skills=skills,
        admission=Admission(ValidationEngine().tool),
        projection=SimpleNamespace(),
        traces=SimpleNamespace(),
        planner_validator=SimpleNamespace(),
        harness_profile="fake_v3",
        replay_tool=lambda _tool, _case: True,
        replay_composite=lambda _composite, _case: True,
    )
    assert result.pending_count == 0
    assert len(result.admitted_assets) == 1
    merged_ref, kind = result.admitted_assets[0]
    assert kind == "tool"
    expected_logical_id = min(primary.ref.tool_id, alias.ref.tool_id)
    assert merged_ref == f"tool://{expected_logical_id}@1.0.1"
    assert tools.get(primary.ref).status is ToolStatus.CANDIDATE
    assert tools.get(alias.ref).status is ToolStatus.CANDIDATE
    assert tools.get(merged_ref).status is ToolStatus.CANDIDATE
    assert store.pending() == []
    assert any(item.status == "admitted" for item in store.history())
    database.close()


def test_tool_add_replay_creates_immutable_candidate_and_replays_full_union(
    tmp_path,
) -> None:
    database = StateDatabase(tmp_path / "state.sqlite3")
    artifacts = ArtifactStore(tmp_path, database)
    skills = SkillRegistry(artifacts, database)
    tools = ToolRegistry(artifacts, database)
    base = replace(
        ToolCompiler().compile([_take_canonical()])[0].tool,
        tests=[_replay_case("old")],
        status=ToolStatus.CANDIDATE,
    )
    tools.register(base)
    candidate = replace(
        base,
        tests=[_replay_case("new")],
        status=ToolStatus.CANDIDATE,
    )
    replayed: list[str] = []
    result = Aligner(skills, tools).align_tool_with_replays(
        candidate,
        admission=Admission(ValidationEngine().tool),
        replay=lambda _tool, case: not replayed.append(str(case["trace_id"])),
    )
    assert result.operation == "add_replay"
    assert result.admitted is True
    assert str(result.ref).endswith("@1.0.1")
    assert replayed == ["old", "new"]
    assert [case["trace_id"] for case in tools.get(base.ref).tests] == ["old"]
    assert [case["trace_id"] for case in tools.get(result.ref).tests] == ["old", "new"]

    rejected = Aligner(skills, tools).align_tool_with_replays(
        replace(candidate, tests=[_replay_case("rejected")]),
        admission=Admission(ValidationEngine().tool),
        replay=lambda _tool, case: case["trace_id"] != "rejected",
    )
    assert rejected.admitted is False
    assert rejected.ref == result.ref
    assert len(tools.list_refs()) == 2
    database.close()


def test_shadow_tool_discovery_is_not_reported_or_credited_as_admitted(
    tmp_path,
) -> None:
    database = StateDatabase(tmp_path / "state.sqlite3")
    artifacts = ArtifactStore(tmp_path, database)
    skills = SkillRegistry(artifacts, database)
    tools = ToolRegistry(artifacts, database)
    rejected = replace(
        ToolCompiler().compile([_take_canonical()])[0].tool,
        status=ToolStatus.SHADOW,
        metadata={"admission_failure": ["source_replay_failed"]},
    )
    result = Aligner(skills, tools).align_tool_with_replays(
        rejected,
        admission=Admission(ValidationEngine().tool),
        replay=lambda _tool, _case: True,
    )
    assert result.operation == "discover"
    assert result.admitted is False
    assert result.admission_failures == ("source_replay_failed",)
    assert tools.get(result.ref).status is ToolStatus.SHADOW
    database.close()


def test_composite_redundancy_requires_fresh_replay_not_stored_success(
    tmp_path,
) -> None:
    database = StateDatabase(tmp_path / "state.sqlite3")
    store = RepairStore(database)
    source = replace(_repair_composite(), status=SkillStatus.ACTIVE)
    removed = source.occurrences[0]
    remaining = source.occurrences[1:]
    implementations = {
        occurrence.occurrence_id: SimpleNamespace(
            ref=SkillRef(f"impl_{occurrence.occurrence_id}", "1.0.0"),
            abstract_ref=occurrence.node_ref,
        )
        for occurrence in remaining
    }

    class Skills:
        registered = []

        def composites(self):
            return [source]

        def get_implementation(self, ref):
            for implementation in implementations.values():
                if str(implementation.ref) == str(ref):
                    return implementation
            raise KeyError(str(ref))

        def list_refs(self, kind):
            return [source.ref] if kind == "composite" else []

        def register_composite(self, candidate):
            self.registered.append(candidate)

    payloads = {}
    events = []
    for index in (1, 2):
        task_id = f"task_redundancy_{index}"
        trace_id = f"trace_redundancy_{index}"
        payloads[trace_id] = {
            "trace_id": trace_id,
            # These historical booleans deliberately claim success; the
            # replay callback below is authoritative and rejects the edit.
            "benchmark_success": True,
            "graph_self_sufficient_success": True,
            "task_rescue_required": False,
            "task": {
                "task_id": task_id, "goal": "goal", "benchmark": "fake",
                "task_type": "redundancy", "metadata": {},
            },
            "runtime_plan": {
                "source_composite_ref": str(source.ref),
            },
            "implementation_invocations": [
                {
                    "occurrence_id": occurrence.occurrence_id,
                    "implementation_ref": str(
                        implementations[occurrence.occurrence_id].ref
                    ),
                    "arguments": {"item": "apple_1"},
                }
                for occurrence in remaining
            ],
        }
        events.append(EvidenceEvent.create(
            task_id=task_id,
            trace_id=trace_id,
            occurrence_id=removed.occurrence_id,
            attempt_id=f"attempt_{index}",
            sequence_no=0,
            artifact_ref=str(source.ref),
            artifact_kind="composite",
            event=EvidenceEventType.GOAL_TERMINAL_SKIPPED,
        ))
    EvidenceLedger(database).append_transaction(events)

    class Traces:
        def load_payload(self, trace_id):
            return payloads[trace_id]

    projection = SimpleNamespace(stats=lambda *_args: SimpleNamespace(
        occurrence_stats={
            removed.occurrence_id: {
                "selected": 2, "skipped_goal_terminal": 2,
            }
        },
    ))
    replayed = []
    results = EvolutionMaintenance(store)._review_composite_redundancy(
        maintenance_trace_id="trace_maintenance",
        skills=Skills(),
        projection=projection,
        traces=Traces(),
        planner_validator=SimpleNamespace(
            validate=lambda *_args, **_kwargs: ValidationResult.ok("planner"),
        ),
        harness_profile="fake_v3",
        replay_composite=lambda _candidate, case: not replayed.append(
            str(case["source_task"]["task_id"])
        ) and False,
    )
    assert len(results) == 1
    assert results[0][0].status == "rejected"
    assert replayed == ["task_redundancy_1", "task_redundancy_2"]
    assert Skills.registered == []
    database.close()


def test_periodic_maintenance_retains_cross_cycle_evidence_until_final_close(
    tmp_path,
) -> None:
    database = StateDatabase(tmp_path / "state.sqlite3")
    store = RepairStore(database)
    maintenance = EvolutionMaintenance(store)

    class EmptyTools:
        def tools(self):
            return []

    class EmptySkills:
        def composites(self):
            return []

    def run(*, finalize_pending=False):
        return maintenance.run_batch(
            maintenance_trace_id="trace_maintenance",
            reviews=[], agent_proposals=[],
            tools=EmptyTools(), skills=EmptySkills(),
            admission=SimpleNamespace(), projection=SimpleNamespace(),
            traces=SimpleNamespace(), planner_validator=SimpleNamespace(),
            harness_profile="fake_v3",
            replay_tool=lambda _tool, _case: True,
            replay_composite=lambda _composite, _case: True,
            finalize_pending=finalize_pending,
        )

    first = RepairProposal.create(
        "skill://atomic_waiting@1.0.0", "atomic", "revise_atomic_contract",
        {"requires_concrete_patch": True, "evidence_ids": ["evidence_1"]},
        ["failure_1"],
    )
    store.save(first)
    assert run().pending_proposal_ids == (first.proposal_id,)
    second = RepairProposal.create(
        first.target_ref, "atomic", "revise_atomic_contract",
        {"requires_concrete_patch": True, "evidence_ids": ["evidence_2"]},
        ["failure_2"],
    )
    store.save(second)
    periodic = run()
    assert set(periodic.pending_proposal_ids) == {
        first.proposal_id, second.proposal_id,
    }
    final = run(finalize_pending=True)
    assert final.pending_count == 0
    assert set(final.rejected_proposal_ids) == {
        first.proposal_id, second.proposal_id,
    }
    assert {
        tuple(item.source_failure_ids) for item in store.history()
    } >= {("failure_1",), ("failure_2",)}
    database.close()


def test_composite_insight_requires_multi_trace_evidence_and_reaches_planner_hint(
    tmp_path,
) -> None:
    database = StateDatabase(tmp_path / "state.sqlite3")
    source = replace(_repair_composite(), status=SkillStatus.ACTIVE, insight={})
    implementations = {
            occurrence.occurrence_id: SimpleNamespace(
                ref=SkillRef(f"impl_{occurrence.occurrence_id}", "1.0.0"),
                abstract_ref=occurrence.node_ref,
            )
            for occurrence in source.occurrences
        }
    unique_impl = SimpleNamespace(
        ref=SkillRef("impl_unique", "1.0.0"),
        abstract_ref=source.occurrences[0].node_ref,
    )

    class Skills:
        def __init__(self):
            self.registered = []

        def composites(self, *, mode=None):
            return [source, *self.registered]

        def get_implementation(self, ref):
            for item in implementations.values():
                if str(item.ref) == str(ref):
                    return item
            if str(unique_impl.ref) == str(ref):
                return unique_impl
            raise KeyError(str(ref))

        def list_refs(self, kind):
            return [item.ref for item in self.composites()] if kind == "composite" else []

        def register_composite(self, candidate):
            self.registered.append(candidate)

    skills = Skills()
    payloads = {}

    def add_support(
        name: str,
        *,
        common_success: bool = False,
        common_failure: bool = False,
        unique: bool = False,
    ) -> None:
        task_id = f"task_insight_{name}"
        trace_id = f"trace_insight_{name}"
        occurrences = list(source.occurrences)
        sequence = [item.step_id for item in occurrences]
        redundant = occurrences[-1]
        binding = {
            "occurrence_id": occurrences[0].occurrence_id,
            "role": "item" if not unique else "unique_role",
            "previous": None,
            "current": {
                "role": "item" if not unique else "unique_role",
                # Concrete values differ and must never enter insight.
                "value": f"apple_{name}",
                "source": "task" if not unique else "agent_proposed",
                "resolution": "semantic" if not unique else "concrete",
            },
        }
        failures = []
        if common_failure:
            failures.append({
                "layer": "composite",
                "code": "composite_self_sufficiency_failure",
                "occurrence_id": occurrences[0].occurrence_id,
            })
        if unique:
            failures.append({
                "layer": "task_contract",
                "code": "task_contract_mismatch",
                "occurrence_id": occurrences[-1].occurrence_id,
            })
            sequence = list(reversed(sequence))
            redundant = occurrences[0]
        payloads[trace_id] = {
            "trace_id": trace_id,
            "benchmark_success": bool(common_success or unique),
            "graph_self_sufficient_success": bool(common_success or unique),
            "task": {
                "task_id": task_id, "goal": "goal", "benchmark": "fake",
                "task_type": "insight", "metadata": {},
            },
            "runtime_plan": {
                "source_composite_ref": str(source.ref),
                "control_sequence": sequence,
            },
            "binding_changes": [binding],
            "failures": failures,
            "node_records": [
                {
                    "step_id": occurrence.step_id,
                    "occurrence_id": occurrence.occurrence_id,
                    "status": (
                        "skipped_goal_terminal"
                        if occurrence.occurrence_id == redundant.occurrence_id
                        else "direct_autonomous_success"
                    ),
                }
                for occurrence in occurrences
            ],
            "implementation_invocations": [
                {
                    "occurrence_id": occurrence.occurrence_id,
                    "implementation_ref": str(
                        unique_impl.ref
                        if unique and occurrence is occurrences[0]
                        else implementations[occurrence.occurrence_id].ref
                    ),
                    "arguments": {"item": "apple_1"},
                    "result": {
                        "started": True,
                        "completed": True,
                        "atomic_effect_passed": True,
                    },
                }
                for occurrence in occurrences
            ],
        }
        events = []
        if common_success or unique:
            events.extend([
                EvidenceEvent.create(
                    task_id=task_id, trace_id=trace_id,
                    occurrence_id="composite",
                    attempt_id=f"attempt_insight_{name}_success",
                    sequence_no=0, artifact_ref=str(source.ref),
                    artifact_kind="composite",
                    event=EvidenceEventType.SELF_SUFFICIENT_SUCCESS,
                ),
                EvidenceEvent.create(
                    task_id=task_id, trace_id=trace_id,
                    occurrence_id=redundant.occurrence_id,
                    attempt_id=f"attempt_insight_{name}_skip",
                    sequence_no=1, artifact_ref=str(source.ref),
                    artifact_kind="composite",
                    event=EvidenceEventType.GOAL_TERMINAL_SKIPPED,
                ),
            ])
        if common_failure or unique:
            events.append(EvidenceEvent.create(
                task_id=task_id, trace_id=trace_id,
                occurrence_id="composite",
                attempt_id=f"attempt_insight_{name}_failure",
                sequence_no=2, artifact_ref=str(source.ref),
                artifact_kind="composite",
                event=(
                    EvidenceEventType.TASK_RESCUE_REQUIRED
                    if common_failure else EvidenceEventType.CONTRACT_MISMATCH
                ),
            ))
        EvidenceLedger(database).append_transaction(events)

    class Traces:
        def load_payload(self, trace_id):
            return payloads[trace_id]

    maintenance = EvolutionMaintenance(RepairStore(database))
    add_support("single", common_success=True)
    assert maintenance._review_composite_insight(
        maintenance_trace_id="trace_maintenance_single",
        skills=skills, traces=Traces(),
        planner_validator=SimpleNamespace(
            validate=lambda *_args, **_kwargs: ValidationResult.ok("planner"),
        ),
        harness_profile="fake_v3",
        replay_composite=lambda _candidate, _case: True,
    ) == []
    add_support("success_2", common_success=True)
    add_support("failure_1", common_failure=True)
    add_support("failure_2", common_failure=True)
    add_support("unique", unique=True)

    class EmptyTools:
        def tools(self):
            return []

    batch = maintenance.run_batch(
        maintenance_trace_id="trace_maintenance",
        reviews=[], agent_proposals=[], tools=EmptyTools(), skills=skills,
        admission=SimpleNamespace(),
        projection=SimpleNamespace(
            stats=lambda *_args: SimpleNamespace(occurrence_stats={}),
        ),
        traces=Traces(),
        planner_validator=SimpleNamespace(
            validate=lambda *_args, **_kwargs: ValidationResult.ok("planner"),
        ),
        harness_profile="fake_v3",
        replay_tool=lambda _tool, _case: True,
        replay_composite=lambda _candidate, _case: True,
    )
    assert len(batch.admitted_assets) == 1
    admitted_ref = batch.admitted_assets[0][0]
    assert batch.lineage == ({
        "source_ref": admitted_ref,
        "target_ref": str(source.ref),
        "relation": "derived_from",
        "operation": "revise_composite_insight",
        "proposal_id": batch.lineage[0]["proposal_id"],
        "review_id": batch.lineage[0]["review_id"],
    },)
    candidate = skills.registered[0]
    assert str(candidate.ref) == admitted_ref
    assert candidate.metadata["batch_evolution"]["operation"] == "revise_composite_insight"
    aggregate = candidate.insight["evidence_aggregate"]
    categories = {
        "parameter_resolution_strategies",
        "frequent_failure_modes",
        "effective_node_orders",
        "redundant_occurrences",
        "implementation_applicability",
    }
    assert categories <= set(aggregate)
    assert all(aggregate[name] for name in categories)
    for name in categories:
        for fact in aggregate[name]:
            assert len(fact["support_task_ids"]) >= 2
            assert len(fact["support_trace_ids"]) >= 2
            assert len(fact["support_event_ids"]) >= 2
    rendered = repr(aggregate)
    assert "apple_" not in rendered
    assert "unique_role" not in rendered
    assert "agent_proposed" not in rendered
    assert "task_contract_mismatch" not in rendered
    assert str(unique_impl.ref) not in rendered
    assert aggregate["redundant_occurrences"] == [{
        "occurrence_id": source.occurrences[-1].occurrence_id,
        "node_outcome": "skipped_goal_terminal",
        "support_task_ids": ["task_insight_single", "task_insight_success_2"],
        "support_trace_ids": ["trace_insight_single", "trace_insight_success_2"],
        "support_event_ids": sorted(
            aggregate["redundant_occurrences"][0]["support_event_ids"]
        ),
    }]
    assert source.insight == {}
    hints = RelatedCompositeHintFinder(skills).find(
        SimpleNamespace(refs=[str(source.occurrences[0].node_ref)]),
        mode="online",
    )
    assert any(
        hint["composite_ref"] == admitted_ref
        and hint["insight"]["evidence_aggregate"] == aggregate
        for hint in hints
    )
    database.close()


def test_composite_insight_supersedes_only_after_revision_is_active(tmp_path) -> None:
    with AtomicSkillGraphSystem(_system_config(tmp_path / "data_v3")) as system:
        old = replace(_repair_composite(), status=SkillStatus.ACTIVE, insight={})
        new = replace(
            old,
            ref=SkillRef(old.ref.logical_id, "1.0.1"),
            insight={"evidence_aggregate": {"schema": "composite_insight.v2"}},
            status=SkillStatus.CANDIDATE,
        )
        system.skills.register_composite(old)
        system.skills.register_composite(new)
        system._add_structural_edge(
            str(new.ref), str(old.ref), GlobalRelationType.DERIVED_FROM,
            "trace_source", evolution_operation="revise_composite_insight",
            proposal_id="repair_insight",
        )
        assert system._apply_stable_supersedes(
            task_id="maintenance_before", trace_id="trace_before",
        ) == ()
        assert system.skills.get_composite(old.ref).status is SkillStatus.ACTIVE

        system.skills.update_status(new.ref, SkillStatus.ACTIVE)
        emitted = system._apply_stable_supersedes(
            task_id="maintenance_after", trace_id="trace_after",
        )
        assert len(emitted) == 1
        assert emitted[0]["relation"] == "supersedes"
        assert emitted[0]["operation"] == "revise_composite_insight"
        assert system.lifecycle is not None
        system.lifecycle.review([str(old.ref)])
        assert system.skills.get_composite(old.ref).status is SkillStatus.SUPPRESSED


def test_stable_replacement_emits_superseded_credit_and_suppresses_old_version(
    tmp_path,
) -> None:
    with AtomicSkillGraphSystem(_system_config(tmp_path / "data_v3")) as system:
        old = replace(
            ToolCompiler().compile([_take_canonical()])[0].tool,
            status=ToolStatus.ACTIVE,
        )
        new = replace(
            old,
            ref=ToolRef(old.ref.tool_id, "1.0.1"),
            status=ToolStatus.ACTIVE,
        )
        system.tools.register(old)
        system.tools.register(new)
        system._add_structural_edge(
            str(new.ref), str(old.ref), GlobalRelationType.DERIVED_FROM,
            "trace_source", evolution_operation="update", proposal_id="repair_update",
        )
        emitted = system._apply_stable_supersedes(
            task_id="maintenance", trace_id="trace_maintenance",
        )
        assert len(emitted) == 1
        assert emitted[0]["relation"] == "supersedes"
        assert system.lifecycle is not None
        system.lifecycle.review([str(old.ref)])
        assert system.tools.get(old.ref).status is ToolStatus.SUPPRESSED
        events = system.database.rows(
            "SELECT event_type,metadata_json FROM evidence_events WHERE artifact_ref=?",
            (str(old.ref),),
        )
        assert [row["event_type"] for row in events] == ["superseded"]
        assert system._apply_stable_supersedes(
            task_id="maintenance_2", trace_id="trace_maintenance_2",
        ) == ()


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("database fault"),
        AgentProviderError("provider_timeout", "provider timed out"),
        BudgetExhausted(
            "runtime_task_token_budget_exhausted",
            "runtime task budget exhausted",
            layer=FailureLayer.RUNTIME_AGENT,
        ),
    ],
)
def test_non_content_evolution_prepare_error_propagates_for_checkpoint_rollback(
    tmp_path, error,
) -> None:
    with AtomicSkillGraphSystem(_system_config(tmp_path / "data_v3")) as system:
        task = HarnessTask("task", "goal", "fake", "pick")
        trace = TraceRecord.create(
            TaskRecord("task", "fake", "goal", "pick", "sig"),
            {}, {}, {"source": "full_dynamic"},
        )
        trace.benchmark_success = True
        trace.task_contract_success = True
        trace.strict_task_success = True
        trace.learning_eligible = True
        system.orchestrator.run_task = lambda *_args, **_kwargs: trace
        system.failure_processor.localize = lambda _trace: []
        system.extraction_policy.decide = lambda _trace: SimpleNamespace(
            should_extract=True, reasons=["full_dynamic_success"],
        )

        def fail_prepare(*_args, **_kwargs):
            raise error

        system._prepare_evolution = fail_prepare
        with pytest.raises(type(error), match=str(error)):
            system.run_task(task)


def test_extractor_protocol_rejection_discards_evolution_but_preserves_task(
    tmp_path,
) -> None:
    with AtomicSkillGraphSystem(_system_config(tmp_path / "data_v3")) as system:
        task = HarnessTask("task", "goal", "fake", "pick")

        def successful_runtime(
            _task, *, mode, trace_builder, attempt_id="",
        ):
            trace = trace_builder.trace
            trace.runtime_plan = {
                "source": "full_dynamic", "failure_stage": "runtime",
            }
            trace.benchmark_success = True
            trace.task_contract_success = True
            trace.strict_task_success = True
            trace.learning_eligible = True
            return trace_builder.finish()

        system.orchestrator.run_task = successful_runtime
        system.failure_processor.localize = lambda _trace: []
        system.extraction_policy.decide = lambda _trace: SimpleNamespace(
            should_extract=True, reasons=["full_dynamic_success"],
        )

        def reject_extractor(*_args, **_kwargs):
            raise ExtractionContentError(
                "e2",
                "extractor_e2_schema_rejected",
                "malformed E2 native submission",
            )

        system._prepare_evolution = reject_extractor
        result = system.run_task(task)

        assert result.benchmark_success is True
        assert result.strict_task_success is True
        assert result.learning_eligible is True
        assert result.metadata["evolution_branch"] == "success"
        assert result.metadata["extraction"] == {
            "attempted": True,
            "stage": "e2",
            "prepared": False,
            "applied": False,
            "error_type": "ExtractionContentError",
            "error_code": "extractor_e2_schema_rejected",
            "error": "malformed E2 native submission",
        }
        assert "evolution_applied" not in result.metadata
        assert system.skills.list_refs("atomic") == []
        assert system.skills.list_refs("implementation") == []
        assert system.skills.list_refs("composite") == []
        assert system.tools.list_refs() == []
        assert system.graph.edges() == []
        assert system.database.execute(
            "SELECT COUNT(*) FROM evidence_events"
        ).fetchone()[0] == 0
        assert result.resource_usage_complete is True
        assert result.metadata["usage_reconciliation"]["token_mismatch"] == 0
        assert len(list(system.traces.iter_payloads())) == 1


def test_success_extractor_token_exhaustion_is_learning_only(
    tmp_path,
) -> None:
    with AtomicSkillGraphSystem(_system_config(tmp_path / "data_v3")) as system:
        task = HarnessTask("task", "goal", "fake", "pick")

        def successful_runtime(
            _task, *, mode, trace_builder, attempt_id="",
        ):
            trace = trace_builder.trace
            trace.runtime_plan = {
                "source": "full_dynamic", "failure_stage": "runtime",
            }
            trace.benchmark_success = True
            trace.task_contract_success = True
            trace.strict_task_success = True
            trace.learning_eligible = True
            return trace_builder.finish()

        system.orchestrator.run_task = successful_runtime
        system.failure_processor.localize = lambda _trace: []
        system.extraction_policy.decide = lambda _trace: SimpleNamespace(
            should_extract=True, reasons=["full_dynamic_success"],
        )

        def exhaust_extractor(*_args, **_kwargs):
            raise BudgetExhausted(
                "extractor_token_budget_exhausted",
                "extractor budget exceeded by provider call",
                layer=FailureLayer.RUNTIME_AGENT,
            )

        system._prepare_evolution = exhaust_extractor
        result = system.run_task(task)

        assert result.benchmark_success is True
        assert result.strict_task_success is True
        assert result.infrastructure_failure is False
        assert result.metadata["evolution_branch"] == "success"
        assert result.metadata["extraction"] == {
            "attempted": True,
            "stage": "e1",
            "prepared": False,
            "applied": False,
            "error_type": "BudgetExhausted",
            "error_code": "extractor_token_budget_exhausted",
            "error": "extractor budget exceeded by provider call",
        }
        assert "evolution_applied" not in result.metadata


def test_implementation_preflight_failure_builds_replay_but_agent_error_does_not(
    tmp_path,
) -> None:
    database = StateDatabase(tmp_path / "state.sqlite3")
    artifacts = ArtifactStore(tmp_path, database)
    skills = SkillRegistry(artifacts, database)
    compiled = ToolCompiler().compile([_take_canonical()])[0]
    skills.register_atomic(replace(compiled.atomic, status=SkillStatus.CANDIDATE))
    skills.register_implementation(replace(
        compiled.implementation, status=SkillStatus.CANDIDATE,
    ))
    payload = {
        "trace_id": "trace_mapping",
        "task": {
            "task_id": "task_mapping",
            "task_signature": "sig",
            "goal": "hold apple",
            "benchmark": "fake",
            "task_type": "pick",
            "metadata": {"split": "train"},
        },
        "environment_actions": [],
        "runtime_spans": [{
            "span_id": "span_preflight", "action_start": 0, "action_end": 0,
        }],
        "node_records": [{
            "occurrence_id": "occ", "atomic_ref": str(compiled.atomic.ref),
        }],
        "implementation_invocations": [{
            "attempt_id": "attempt", "occurrence_id": "occ",
            "implementation_ref": str(compiled.implementation.ref),
            "arguments": {"item": "apple_1"}, "span_id": "span_preflight",
        }],
    }
    intrinsic = {
        "failure_id": "failure_mapping", "layer": "implementation",
        "code": "implementation_mapping_error", "task_id": "task_mapping",
        "trace_id": "trace_mapping", "occurrence_id": "occ",
        "attempt_id": "attempt", "started": False,
    }
    evidence = build_trace_repair_evidence(
        payload, intrinsic,
        target_layer="implementation",
        target_ref=str(compiled.implementation.ref),
        skills=skills,
        harness_profile="fake_v3",
    )
    assert evidence is not None
    assert evidence.replay_case["occurrence_actions"] == []

    agent_error = {
        **intrinsic,
        "failure_id": "failure_agent",
        "layer": "runtime_agent",
        "code": "runtime_agent_schema_error",
    }
    assert build_trace_repair_evidence(
        payload, agent_error,
        target_layer="implementation",
        target_ref=str(compiled.implementation.ref),
        skills=skills,
        harness_profile="fake_v3",
    ) is None
    database.close()


def test_typed_implementation_replay_checks_live_constraints_and_propagates_infra(
    tmp_path,
) -> None:
    database = StateDatabase(tmp_path / "state.sqlite3")
    artifacts = ArtifactStore(tmp_path, database)
    skills = SkillRegistry(artifacts, database)
    tools = ToolRegistry(artifacts, database)
    compiled = ToolCompiler().compile([_take_canonical()])[0]
    atomic = replace(compiled.atomic, status=SkillStatus.CANDIDATE)
    tool = replace(compiled.tool, status=ToolStatus.CANDIDATE)
    implementation = replace(compiled.implementation, status=SkillStatus.CANDIDATE)
    skills.register_atomic(atomic)
    tools.register(tool)
    harness = FakeHarness()
    executor = TraceRepairExecutor(
        harness=harness,
        skills=skills,
        tools=tools,
        validation=ValidationEngine(),
        admission=Admission(ValidationEngine().tool),
    )
    task = fake_task("repair", "apple_1")
    case = {
        "target_layer": "implementation",
        "source_task": {
            "task_id": task.task_id, "goal": task.goal,
            "benchmark": task.benchmark, "task_type": task.task_type,
            "context": task.context, "metadata": task.metadata,
        },
        "occurrence_id": "occ",
        "bindings": {"item": "apple_1"},
        "prefix": [],
        "occurrence_actions": [],
        "source_attempt_started": False,
    }
    assert executor.replay(implementation, case) is True

    ungrounded = replace(
        implementation,
        grounding_constraints=[GroundingConstraint(
            "bad_relation",
            GroundingConstraintKind.HARNESS_AFFORDANCE,
            action_type="EXAMINE",
            argument_mapping=implementation.grounding_constraints[0].argument_mapping,
            required_resolution="relation_verified",
        )],
    )
    assert executor.replay(ungrounded, case) is False

    class BrokenHarness(FakeHarness):
        def execute_primitive(self, primitive, bindings):
            raise RuntimeError("harness process died")

    broken = TraceRepairExecutor(
        harness=BrokenHarness(),
        skills=skills,
        tools=tools,
        validation=ValidationEngine(),
        admission=Admission(ValidationEngine().tool),
    )
    with pytest.raises(RuntimeError, match="harness process died"):
        broken.replay(implementation, case)
    database.close()


def _register_atomic_merge_source(
    skills, tools, compiled, *, logical_id: str, trace_id: str,
    guideline=None,
) -> tuple[str, dict]:
    atomic = replace(
        compiled.atomic,
        ref=SkillRef(f"atomic_{logical_id}", "1.0.0"),
        guideline=(compiled.atomic.guideline if guideline is None else guideline),
        status=SkillStatus.CANDIDATE,
    )
    case = _replay_case(trace_id)
    tool = replace(
        compiled.tool,
        ref=ToolRef(f"tool_{logical_id}", "1.0.0"),
        tests=[case],
        status=ToolStatus.CANDIDATE,
    )
    binding = replace(compiled.implementation.tool_bindings[0], tool_ref=tool.ref)
    implementation = replace(
        compiled.implementation,
        ref=SkillRef(f"impl_{logical_id}", "1.0.0"),
        abstract_ref=atomic.ref,
        tool_bindings=[binding],
        status=SkillStatus.CANDIDATE,
    )
    skills.register_atomic(atomic)
    tools.register(tool)
    skills.register_implementation(implementation)
    payload = {
        "trace_id": trace_id,
        "benchmark_success": True,
        "task": dict(case["source_task"]),
    }
    return str(atomic.ref), payload


def test_atomic_merge_review_requires_equivalence_and_support_for_every_source(
    tmp_path,
) -> None:
    database = StateDatabase(tmp_path / "state.sqlite3")
    artifacts = ArtifactStore(tmp_path, database)
    skills = SkillRegistry(artifacts, database)
    tools = ToolRegistry(artifacts, database)
    compiled = ToolCompiler().compile([_take_canonical()])[0]
    first, payload1 = _register_atomic_merge_source(
        skills, tools, compiled, logical_id="take_a", trace_id="trace_a",
    )
    second, payload2 = _register_atomic_merge_source(
        skills, tools, compiled, logical_id="take_b", trace_id="trace_b",
    )

    class Traces:
        def __init__(self, payloads):
            self.payloads = payloads

        def load_payload(self, trace_id):
            return self.payloads[trace_id]

    trace_store = Traces({"trace_a": payload1, "trace_b": payload2})
    maintenance = EvolutionMaintenance(RepairStore(database))
    reviews = maintenance.build_typed_reviews(
        skills=skills, tools=tools, traces=trace_store, harness_profile="fake_v3",
    )
    merge = [item for item in reviews if item.eligible_operations == ("merge_atomic",)]
    assert len(merge) == 1
    assert set(merge[0].target_refs) == {first, second}
    assert {
        item.replay_case["target_ref"] for item in merge[0].evidence
    } == {first, second}

    # Removing one source's persisted replay support closes the cohort.
    del trace_store.payloads["trace_b"]
    assert not [
        item for item in maintenance.build_typed_reviews(
            skills=skills, tools=tools, traces=trace_store, harness_profile="fake_v3",
        )
        if item.eligible_operations == ("merge_atomic",)
    ]
    database.close()


def test_atomic_merge_review_ignores_guideline_wording_for_same_contract(tmp_path) -> None:
    database = StateDatabase(tmp_path / "state.sqlite3")
    artifacts = ArtifactStore(tmp_path, database)
    skills = SkillRegistry(artifacts, database)
    tools = ToolRegistry(artifacts, database)
    compiled = ToolCompiler().compile([_take_canonical()])[0]
    first, payload1 = _register_atomic_merge_source(
        skills, tools, compiled, logical_id="take_a", trace_id="trace_a",
    )
    second, payload2 = _register_atomic_merge_source(
        skills, tools, compiled, logical_id="take_b", trace_id="trace_b",
        guideline={"incompatible_boundary": True},
    )

    class Traces:
        def load_payload(self, trace_id):
            return {"trace_a": payload1, "trace_b": payload2}[trace_id]

    reviews = EvolutionMaintenance(RepairStore(database)).build_typed_reviews(
        skills=skills, tools=tools, traces=Traces(), harness_profile="fake_v3",
    )
    merge = [
        item for item in reviews if item.eligible_operations == ("merge_atomic",)
    ]
    assert len(merge) == 1
    assert set(merge[0].target_refs) == {first, second}
    database.close()


def test_run_task_does_not_leak_maintenance_budget_state(tmp_path) -> None:
    with AtomicSkillGraphSystem(
        _system_config(tmp_path / "data_v3"), harness=FakeHarness(),
    ) as system:
        def infrastructure_trace(
            task, *, mode, trace_builder=None, attempt_id="",
        ):
            trace = trace_builder.trace if trace_builder is not None else TraceRecord.create(
                TaskRecord(
                    task.task_id, task.benchmark, task.goal, task.task_type,
                    content_hash(task.goal), task.metadata,
                ),
                {}, {}, {"source": "test", "mode": str(mode)},
            )
            trace.runtime_plan = {
                "source": "test",
                "mode": str(mode),
                "attempt_id": attempt_id,
            }
            trace.infrastructure_failure = True
            return trace.finish()

        system.orchestrator.run_task = infrastructure_trace
        system.run_task(fake_task("one", "apple_1"))
        system.run_task(fake_task("two", "apple_1"))
        assert system._evolution_batch_usage_start is None


def test_maintenance_exception_clears_shared_batch_budget(tmp_path) -> None:
    with AtomicSkillGraphSystem(
        _system_config(tmp_path / "data_v3"), harness=FakeHarness(),
    ) as system:
        def fail_reviews(_tools):
            raise RuntimeError("maintenance database unavailable")

        system.evolution_maintenance.build_batch_reviews = fail_reviews
        with pytest.raises(RuntimeError, match="database unavailable"):
            system.run_maintenance(triggering_task_id="task", milestone="failure")
        assert system._evolution_batch_usage_start is None


def test_evolution_producers_share_one_batch_token_cap(tmp_path) -> None:
    config = _system_config(tmp_path / "data_v3")
    config["llm"]["evolution_repair"] = {
        "max_total_tokens_per_batch": 15,
        "max_turns": 1,
    }
    provider = ScriptedAgentProvider([
        FakeReply.structured(
            {"ok": True}, prompt_tokens=6, completion_tokens=4,
        ),
    ])
    with AtomicSkillGraphSystem(
        config, harness=FakeHarness(), provider=provider,
    ) as system:
        system._evolution_batch_usage_start = len(system.usage.events)
        first = system._evolution_repair_session("maintenance")
        StructuredSubmissionClient().request(
            first,
            prompt="first producer",
            tool_name="submit_batch_probe",
            description="Submit the batch budget probe.",
            schema={
                "type": "object",
                "required": ["ok"],
                "additionalProperties": False,
                "properties": {"ok": {"type": "boolean"}},
            },
        )
        second = system._evolution_repair_session("maintenance")
        assert first.snapshot()["budget"]["max_total_tokens"] == 15
        assert second.snapshot()["budget"]["max_total_tokens"] == 5
        system._evolution_batch_usage_start = None


def test_structured_no_change_is_audited_and_same_review_is_not_repeated(
    tmp_path,
) -> None:
    review = TypedRepairReview(
        review_id="typed_review_stable_no_change",
        target_layer="atomic",
        target_refs=("skill://atomic_waiting@1.0.0",),
        eligible_operations=("revise_atomic_contract",),
        context={"source_proposal_ids": []},
        evidence=tuple(
            RepairEvidence(
                f"evidence_{index}", f"task_{index}", f"trace_{index}",
                "stable_cluster", {"case": index},
                failure_layer="atomic", failure_code="atomic_effect_violation",
            )
            for index in (1, 2)
        ),
        source_failure_ids=("failure_1", "failure_2"),
    )
    provider = ScriptedAgentProvider([FakeReply.structured({
        "decisions": [{
            "review_id": review.review_id,
            "decision": "no_change",
            "operation": "no_change",
            "replacements": [],
            "rationale": "the stable evidence does not justify a safe edit",
        }],
    })])
    with AtomicSkillGraphSystem(
        _system_config(tmp_path / "data_v3"),
        harness=FakeHarness(), provider=provider,
    ) as system:
        system.evolution_maintenance.build_batch_reviews = lambda _tools: []

        def typed_reviews(**_kwargs):
            resolved = {
                item.proposed_patch.get("typed_review_id")
                for item in system.repair_store.history()
            }
            return [] if review.review_id in resolved else [review]

        system.evolution_maintenance.build_typed_reviews = typed_reviews
        system.evolution_maintenance.build_composite_sequence_reviews = (
            lambda **_kwargs: []
        )
        system.evolution_maintenance.run_batch = lambda **kwargs: BatchMaintenanceResult(
            maintenance_trace_id=kwargs["maintenance_trace_id"],
        )
        first = system.run_maintenance(
            triggering_task_id="task_2", milestone="periodic_1",
        )
        assert first.pending_count == 0
        audit = next(
            item for item in system.repair_store.history()
            if item.proposed_patch.get("typed_review_id") == review.review_id
        )
        assert audit.status == "rejected"
        assert audit.proposed_patch["review_outcome"] == "no_change"
        assert audit.proposed_patch["evidence_ids"] == ["evidence_1", "evidence_2"]
        system.run_maintenance(
            triggering_task_id="task_3", milestone="periodic_2",
        )
        assert provider.remaining_replies == 0
        assert len([
            item for item in system.repair_store.history()
            if item.proposed_patch.get("typed_review_id") == review.review_id
        ]) == 1


def test_system_maintenance_revises_composite_sequence_through_fresh_replay(
    tmp_path,
) -> None:
    harness = FakeHarness()
    task1 = fake_task("sequence_1", "apple_1", requires_rescue=True)
    task2 = fake_task("sequence_2", "apple_1", requires_rescue=True)
    compiled = ToolCompiler().compile([_take_canonical()])[0]

    def proposal_reply(request):
        review = request.policy_context["reviews"][0]
        replacement = dict(review["source_composite"])
        replacement["control_sequence"] = ["step_take", "step_examine"]
        return {"decisions": [{
            "review_id": review["review_id"],
            "decision": "propose",
            "replacement": replacement,
            "rationale": "two independent structural failures require the forward order",
        }]}

    provider = ScriptedAgentProvider([FakeReply.structured(proposal_reply)])
    with AtomicSkillGraphSystem(
        _system_config(tmp_path / "data_v3"),
        harness=harness,
        provider=provider,
    ) as system:
        take_atomic = replace(
            compiled.atomic,
            ref=SkillRef("atomic_sequence_take", "1.0.0"),
            guideline={"boundary": "take"},
            status=SkillStatus.CANDIDATE,
        )
        examine_atomic = replace(
            take_atomic,
            ref=SkillRef("atomic_sequence_examine", "1.0.0"),
            effects=[SemanticPredicate(
                "object.observed",
                {"object": BindingExpression(
                    BindingExprKind.SKILL_INPUT, source_role="item",
                )},
            )],
            guideline={"boundary": "examine_after_holding"},
        )
        take_tool = replace(
            compiled.tool,
            ref=ToolRef("tool_sequence_take", "1.0.0"),
            tests=[],
            status=ToolStatus.CANDIDATE,
        )
        examine_tool = replace(
            take_tool,
            ref=ToolRef("tool_sequence_examine", "1.0.0"),
            artifact={
                **take_tool.artifact,
                "steps": [{
                    "action_type": "EXAMINE",
                    "argument_mapping": dict(
                        take_tool.artifact["steps"][0]["argument_mapping"]
                    ),
                }],
            },
            safety={"reviewed": True, "allowed_action_types": ["EXAMINE"]},
        )
        take_binding = replace(
            compiled.implementation.tool_bindings[0], tool_ref=take_tool.ref,
        )
        examine_binding = replace(take_binding, tool_ref=examine_tool.ref)
        take_impl = replace(
            compiled.implementation,
            ref=SkillRef("impl_sequence_take", "1.0.0"),
            abstract_ref=take_atomic.ref,
            tool_bindings=[take_binding],
            status=SkillStatus.CANDIDATE,
        )
        examine_constraints = [
            replace(item, action_type="EXAMINE")
            for item in compiled.implementation.grounding_constraints
        ]
        examine_impl = replace(
            take_impl,
            ref=SkillRef("impl_sequence_examine", "1.0.0"),
            abstract_ref=examine_atomic.ref,
            tool_bindings=[examine_binding],
            grounding_constraints=examine_constraints,
        )
        for atomic in (take_atomic, examine_atomic):
            system.skills.register_atomic(atomic)
        for tool in (take_tool, examine_tool):
            system.tools.register(tool)
        for implementation in (take_impl, examine_impl):
            system.skills.register_implementation(implementation)

        item_constant = BindingExpression(
            BindingExprKind.CONSTANT, constant="apple_1",
        )
        source = CompositeSkill(
            SkillRef("composite_sequence", "1.0.0"),
            "misordered examine then take workflow",
            [
                CompositeOccurrence(
                    "step_examine", "occ_examine", examine_atomic.ref,
                    {"item": item_constant},
                ),
                CompositeOccurrence(
                    "step_take", "occ_take", take_atomic.ref,
                    {"item": item_constant},
                ),
            ],
            ["step_examine", "step_take"],
            [],
            [GraphEdge(
                "requires_take_before_examine",
                GraphEdgeType.REQUIRES_SKILL,
                "step_take", "step_examine",
                origin="extractor_validated",
            )],
            harness.task_contract(task1),
            {}, {}, {"validator_id": "planner"}, {}, SkillStatus.CANDIDATE,
        )
        system.skills.register_composite(source)

        for index, task in enumerate((task1, task2), start=1):
            trace = TraceRecord.create(
                TaskRecord(
                    task.task_id, task.benchmark, task.goal, task.task_type,
                    content_hash(task.goal),
                    {**task.metadata, "context": task.context, "split": "train"},
                ),
                to_primitive(source.goal_contract),
                {},
                {
                    "source": "stored_composite",
                    "source_composite_ref": str(source.ref),
                },
            )
            trace.implementation_invocations = [
                ImplementationInvocationRecord(
                    f"attempt_examine_{index}", "occ_examine",
                    str(examine_impl.ref), {"item": "apple_1"}, {}, {},
                    f"span_examine_{index}",
                ),
                ImplementationInvocationRecord(
                    f"attempt_take_{index}", "occ_take",
                    str(take_impl.ref), {"item": "apple_1"}, {}, {},
                    f"span_take_{index}",
                ),
            ]
            failure = FailureEnvelope(
                f"failure_sequence_{index}", FailureLayer.COMPOSITE,
                "composite_self_sufficiency_failure", task.task_id,
                trace.trace_id, "", f"task_rescue_{index}", False,
                [str(source.ref)], [], True,
            )
            trace.failures = [failure]
            trace.finish()
            system.traces.save_atomic(trace)
            system.repair_store.save(RepairProposal.create(
                str(source.ref), "composite", "insert_missing_occurrence",
                {
                    "required_replay_trace_ids": [trace.trace_id],
                    "requires_concrete_patch": True,
                },
                [failure.failure_id],
            ))

        result = system.run_maintenance(
            triggering_task_id=task2.task_id,
            milestone="sequence_repair",
        )
        assert result.pending_count == 0
        admitted = [
            ref for ref, kind in result.admitted_assets if kind == "composite"
        ]
        assert len(admitted) == 1
        assert system.skills.get_composite(admitted[0]).control_sequence == [
            "step_take", "step_examine",
        ]
        history = system.repair_store.history()
        assert any(
            item.operation == "revise_composite_sequence"
            and item.status == "admitted"
            and item.replay_result.get("source_replays")
            for item in history
        )
