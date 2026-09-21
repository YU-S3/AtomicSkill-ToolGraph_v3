"""Original executed trial facts, immutable source commit and new credit view."""
import copy
from dataclasses import asdict, replace

import pytest

from atomic_skillgraph.knowledge.database import StateDatabase
from atomic_skillgraph.knowledge.execution_observations import ExecutionObservationStore, collect_execution_observations, execution_identity
from atomic_skillgraph.evolution.identity_matching import raw_hash
from atomic_skillgraph.evolution.execution_attribution import prepare_attributions
from atomic_skillgraph.governance.ledger import EvidenceLedger, EvidenceEvent, EvidenceEventType
from atomic_skillgraph.governance.projections import ArtifactStats, ProjectionCorruptionError
from atomic_skillgraph.governance.lifecycle import LifecyclePolicy
from atomic_skillgraph.system import AtomicSkillGraphSystem
from atomic_skillgraph.core.serialization import to_primitive
from fixtures.r921_self_tooling_cases import fixture_config
from test_r10_runtime import staged_observation, CheckpointHarness, StepProvider, route


def executed_source(tmp_path, number):
    system, ctx, _ = staged_observation(tmp_path, f"independent_{number}", repair_revision="R10.3")
    trace = ctx.trace_builder.trace
    trace.metadata["execution_source"] = {"run_id": "source-run", "benchmark": "synthetic",
        "task_id": trace.task.task_id, "task_signature": f"manifest-game-{number}",
        "attempt_id": f"attempt-{number}", "manifest_ordinal": number, "attempt_ordinal": 1,
        "split": "train", "experiment_kind": "formal"}
    observations = collect_execution_observations(trace, system.harness.profile_name)
    assert len(observations) == 1
    assert observations[0][0].outcome == "success", to_primitive(trace)
    return system, trace, observations[0]


def test_trial_collection_uses_actual_native_execution_lineage(tmp_path):
    system, trace, (observation, evidence) = executed_source(tmp_path, 1)
    assert observation.authorizing_native_call_id
    assert evidence["tool_execution"]["result"]["started"]
    assert observation.program_payload_hash == raw_hash(evidence["raw_bundle"]["tool"])
    payload = to_primitive(trace)
    payload["metadata"]["runtime_tool_trials"][evidence["draft_id"]]["authorizing_native_call_id"] = "missing"
    assert collect_execution_observations(payload, system.harness.profile_name) == []
    system.close()


def test_P_unproven_promotion_keeps_original_ineligible_execution_fact(tmp_path):
    system, trace, (observation, evidence) = executed_source(tmp_path, 1)
    try:
        payload = to_primitive(trace)
        trial = payload["metadata"]["runtime_tool_trials"][evidence["draft_id"]]
        trial["promotion_bundle"]["tool"]["artifact"]["max_actions"] += 1
        rows = collect_execution_observations(payload,system.harness.profile_name)
        assert len(rows) == 1
        observed, source = rows[0]
        assert observed.started and observed.outcome == "ineligible"
        assert observed.execution_key == observation.execution_key
        assert source["bundle"] == source["raw_bundle"]
        assert source["source_to_promotion_status"] != "exact"
    finally:
        system.close()


@pytest.mark.parametrize("damage", ["incomplete","outputs","terminal","not_learning_eligible","not_won","outside_canonical"])
def test_P07_P08_P13_positive_credit_requires_all_original_gates(tmp_path,damage):
    system,trace,(observation,evidence) = executed_source(tmp_path,1)
    try:
        original = raw_hash(trace)
        payload = to_primitive(trace)
        trial = payload["metadata"]["runtime_tool_trials"][evidence["draft_id"]]
        if damage == "incomplete": trial["r1"]["tool_completed"] = False
        elif damage == "outputs": trial["r1"]["outputs_valid"] = False
        elif damage == "terminal": trial["r1"]["terminal_interrupted"] = True
        elif damage == "not_learning_eligible": payload["learning_eligible"] = False
        elif damage == "not_won": payload["benchmark_success"] = False
        else: trial["trial_event_start"] = -1
        rows = collect_execution_observations(payload,system.harness.profile_name)
        assert len(rows) == 1
        assert rows[0][0].outcome == "ineligible"
        assert rows[0][0].execution_key == observation.execution_key
        assert raw_hash(trace) == original
    finally:
        system.close()


def test_observation_trace_first_same_ledger_transaction_and_hash_guard(tmp_path):
    origin, trace, (obs, evidence) = executed_source(tmp_path / "source", 1)
    database = StateDatabase(tmp_path / "bank" / "state.sqlite3", r103=True)
    store = ExecutionObservationStore(database, database.path.parent)
    reference = store.stage(obs, evidence)
    assert store.committed() == []
    ledger = EvidenceLedger(database)
    with pytest.raises(RuntimeError):
        ledger.append_transaction([], companion_write=lambda connection: store.commit(connection, reference, ""))
    digest = raw_hash(trace)
    ledger.append_transaction([], companion_write=lambda connection: store.commit(connection, reference, digest))
    ledger.append_transaction([], companion_write=lambda connection: store.commit(connection, reference, digest))
    assert len(store.committed()) == 1
    with pytest.raises(RuntimeError, match="conflicting"):
        ledger.append_transaction([], companion_write=lambda connection: store.commit(connection, reference, "different"))
    assert store.committed()[0]["source_trace_hash"] == digest
    origin.close()
    database.close()


def test_two_committed_real_trials_admit_and_attribute_without_fake_direct(tmp_path):
    first, trace1, obs1 = executed_source(tmp_path / "first", 1)
    second, trace2, obs2 = executed_source(tmp_path / "second", 2)
    config = fixture_config(tmp_path / "bank")
    config["repair_revision"] = "R10.3"
    config["runtime"]["persistent_runtime_support_promotion"] = True
    config["experiment"]["task_manifest_path"] = None
    bank = AtomicSkillGraphSystem(config, harness=CheckpointHarness(route.RouteCase("independent_2")), provider=StepProvider(lambda r,n: None))
    for trace, (observation, evidence) in ((trace1, obs1), (trace2, obs2)):
        reference = bank.runtime_support_store.stage(observation, evidence)
        bank.traces.save_atomic(trace)
        from atomic_skillgraph.knowledge.source_snapshots import publish
        publish(bank.data_dir, bank.traces.load_payload(trace.trace_id))
        bank.ledger.append_transaction([], companion_write=lambda connection: bank.runtime_support_store.commit(connection, reference, raw_hash(trace)))
    # A distinct publication Trace is mandatory; never mutate saved sources.
    publisher = copy.deepcopy(trace2)
    publisher.trace_id = "trace_r103_publisher"
    publisher.metadata.pop("runtime_support_promotions", None)
    events = prepare_attributions(bank, publisher, second.harness._task if hasattr(second.harness, "_task") else None, bank.runtime_support_store.committed())
    attributions = [e for e in events if e.event is EvidenceEventType.EXECUTION_ATTRIBUTED]
    assert len(attributions) == 4, publisher.metadata
    assert not any(e.event in {EvidenceEventType.DIRECT_SUCCESS, EvidenceEventType.EXECUTION_STARTED} for e in events)
    publisher.evidence_event_refs = [e.event_id for e in events]
    from atomic_skillgraph.evolution.replay_publication import resolve
    resolve(bank, publisher)
    bank.traces.save_atomic(publisher)
    bank._commit_replay_certificates(publisher)
    bank._commit_evidence(events)
    bank.lifecycle.review(artifact_refs=sorted({e.artifact_ref for e in events}))
    refs = publisher.metadata["runtime_support_promotions"][0]["refs"]
    assert bank.skills.get_atomic(refs[0]).status.value == "active"
    assert bank.skills.get_implementation(refs[1]).status.value == "active"
    assert bank.tools.get(refs[2]).status.value == "active"
    assert prepare_attributions(bank, publisher, None, bank.runtime_support_store.committed()) == []
    # Exercise the actual freeze/read-only path with execution capsules,
    # attributed Ledger rows, identity proofs and rebuilt lifecycle projections.
    digest = bank.knowledge_digest()
    destination = tmp_path / "frozen"
    bank.freeze(destination)
    frozen_config = copy.deepcopy(config)
    frozen_config.update(data_dir=str(destination),trace_data_dir=str(tmp_path / "frozen_traces"))
    frozen_config["experiment"].update(runtime_mode="frozen",freeze_skills=True,allow_long_term_knowledge_writes=False)
    frozen_config["cold_start"]["enabled"] = False
    with AtomicSkillGraphSystem(frozen_config,readonly=True,harness=CheckpointHarness(route.RouteCase("frozen")),
                               provider=StepProvider(lambda r,n: None)) as frozen:
        assert frozen.knowledge_digest() == digest
        assert frozen.tools.get(refs[2]).status.value == "active"
        assert len(frozen.runtime_support_store.committed()) == 2
    assert bank.knowledge_digest() == digest
    for system in (first, second, bank): system.close()


def event(key, task, order, outcome="complete_success"):
    return EvidenceEvent.create(task_id="publication", trace_id="trace_publish", occurrence_id="source",
        attempt_id=key, sequence_no=order, artifact_ref="tool://a@1.0.0", artifact_kind="tool",
        event=EvidenceEventType.EXECUTION_ATTRIBUTED, metadata={"source_execution_key": key,
            "source_independent_task_key": task, "source_order": [order, 1, 0],
            "source_started": True, "source_completed": outcome == "complete_success", "outcome": outcome})


def test_source_order_not_publication_order_drives_support_and_failure_streak():
    stats = ArtifactStats("tool://a@1.0.0", "tool")
    stats.apply(event("one", "task1", 1), 1)
    assert stats.independent_execution_support_count == 0
    stats.apply(event("same-task-second-trial", "task1", 2), 2)
    assert stats.independent_execution_support_count == 0
    stats.apply(event("failure-later", "task3", 5, "intrinsic_failure"), 3)
    stats.apply(event("earlier-independent", "task2", 3), 4)
    assert stats.independent_execution_support_count == 2
    assert stats.execution_support["consecutive_intrinsic_failures"] == 1
    assert stats.started_count == stats.direct_success_count == 0
    assert stats.execution_support["started_count"] == 4
    assert stats.execution_support["complete_success_count"] == 3
    assert ArtifactStats.from_dict(stats.to_dict()).execution_support == stats.execution_support
    with pytest.raises(ProjectionCorruptionError):
        stats.apply(event("one", "task1", 1, "intrinsic_failure"), 5)


def test_independent_identity_does_not_change_with_attempt_or_path_alias():
    source = {"run_id":"run", "task_id":"display-alias", "task_signature":"actual-game",
        "attempt_id":"attempt1", "benchmark":"domain", "manifest_ordinal":0, "attempt_ordinal":1}
    one = execution_identity(source, "trace_one", "exec", 0)
    two = execution_identity({**source, "task_id":"other-path", "attempt_id":"attempt2"}, "trace_two", "exec", 0)
    assert one["source_independent_task_key"] == two["source_independent_task_key"]
    assert one["source_execution_key"] != two["source_execution_key"]


def test_P03_same_tool_different_implementation_mapping_separates_credit(tmp_path):
    from atomic_skillgraph.core.serialization import dataclass_from_dict
    from atomic_skillgraph.core.contracts import AbstractAtomicSkill, ToolAsset, ImplementationAtom
    from atomic_skillgraph.core.bindings import BindingExprKind, BindingExpression
    system,trace,(observation,evidence) = executed_source(tmp_path,1)
    try:
        source = {"observation":asdict(observation),"evidence":evidence,
                  "source_trace_hash":raw_hash(trace),"reference":{"execution_key":observation.execution_key}}
        bundle = evidence["bundle"]
        atomic = dataclass_from_dict(AbstractAtomicSkill,bundle["atomic"])
        tool = dataclass_from_dict(ToolAsset,bundle["tool"])
        impl = dataclass_from_dict(ImplementationAtom,bundle["implementation"])
        changed = copy.deepcopy(impl)
        role = next(iter(changed.tool_bindings[0].parameter_mapping))
        changed.tool_bindings[0].parameter_mapping[role] = BindingExpression(BindingExprKind.CONSTANT,constant="different")
        args = dict(source=source,target_atomic=atomic,target_tool=tool,target_implementation=changed,
                    publishing_task_id="publisher",publishing_trace_id="publisher")
        assert system.credit.attribute_execution(**args,artifact_kind="tool") is not None
        assert system.credit.attribute_execution(**args,artifact_kind="implementation") is None
        # Exact-program intrinsic failure cannot be laundered by rebinding to a
        # different Atomic. This unit checks proof construction, not source R1.
        source["observation"] = asdict(replace(observation,outcome="failure",intrinsic_failure=True,failure_layer="tool"))
        args["target_atomic"] = replace(atomic,preconditions=[] if atomic.preconditions else [atomic.effects[0]])
        negative = system.credit.attribute_execution(**args,artifact_kind="tool")
        assert negative.metadata["outcome"] == "intrinsic_failure"
        assert [p["layer"] for p in negative.metadata["identity_proofs"]] == ["tool"]
        assert system.credit.attribute_execution(**args,artifact_kind="atomic") is None
    finally:
        system.close()
