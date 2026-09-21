"""Bounded sidecar production validators, independent authorities and commit."""
import copy
from types import SimpleNamespace

import pytest

from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.evolution.extractor_session import ExtractionBatch, ExtractorSession, e1_schema
from atomic_skillgraph.evolution.generalization import prepare_sidecar, apply_sidecar
from atomic_skillgraph.evolution.identity_matching import raw_hash
from atomic_skillgraph.system import AtomicSkillGraphSystem
from test_r1021_boundaries import typed_preparation_example, typed_atomicizer
from test_r1021_final import raw_proposal
from test_r103_execution_support import executed_source
from test_r10_runtime import CheckpointHarness, StepProvider, route
from fixtures.r921_self_tooling_cases import fixture_config


def sources(tmp_path, monkeypatch):
    config = fixture_config(tmp_path / "bank")
    config["repair_revision"] = "R10.3"
    config["runtime"]["persistent_runtime_support_promotion"] = True
    config["experiment"]["task_manifest_path"] = None
    bank = AtomicSkillGraphSystem(config, harness=CheckpointHarness(route.RouteCase("current")),
                                 provider=StepProvider(lambda r, n: None))
    bank.atomicizer = typed_atomicizer()
    records = []
    for number in (1, 2):
        origin, trace, _ = executed_source(tmp_path / str(number), number)
        proposal, normalized = typed_preparation_example()
        for event in normalized["actions"]:
            event["authoritative_negative_effects"] = []
            for fact in event["authoritative_positive_effects"]:
                fact["revision"] = event["after_revision"]
        # Independent source snapshots retain their original authority names;
        # the model has no permission to borrow the other slot's authority.
        normalized["trace_id"] = trace.trace_id
        for authority in normalized["boundary_authorities"]["inputs"]:
            authority["trace_id"] = trace.trace_id
            old = authority["authority_ref"]
            authority["authority_ref"] = f"source{number}:{old}"
            for reference in proposal.input_provenance_refs.values():
                if reference["authority_ref"] == old:
                    reference["authority_ref"] = authority["authority_ref"]
            proposal.local_value_authority_refs = [authority["authority_ref"] if x == old else x
                                                    for x in proposal.local_value_authority_refs]
        canonical, = bank.atomicizer.validate_and_canonicalize([proposal], normalized)
        atomic = bank._canonical_atomic_for_occurrence(canonical)
        records.append((trace, proposal, normalized, canonical, atomic))
        origin.close()
    snapshots = {t.trace_id: n for t, p, n, c, a in records}
    monkeypatch.setattr(bank, "_normalized_learning_source", lambda trace: copy.deepcopy(snapshots[trace.trace_id]))
    historical, p, n, c, a = records[0]
    ref = bank.learning_source_store.stage(historical, n, c, a, bank.harness.profile_name, proposal=p)
    bank.traces.save_atomic(historical)
    from atomic_skillgraph.knowledge.source_snapshots import publish
    publish(bank.data_dir, bank.traces.load_payload(historical.trace_id))
    bank.ledger.append_transaction([], companion_write=lambda connection:
        bank.learning_source_store.commit(connection, ref, raw_hash(bank.traces.load_payload(historical.trace_id))))
    row = dict(bank.database.rows("SELECT * FROM learning_source_index")[0])
    current, current_p, current_n, *_ = records[1]
    group = {"group_id": "offered", "history_reference": row, "history": n}
    submitted = {"group_id": "offered", "rationale": "shared supported acquisition",
        "source_proposals": [{"source_slot": "current", "proposal": raw_proposal(current_p)},
                             {"source_slot": "history", "proposal": raw_proposal(p)}]}
    context = {"normalized": current_n, "groups": [group], "batch": ExtractionBatch([], [submitted])}
    return bank, current, context


def test_G_sidecar_empty_ordinary_validates_both_and_retains_atomic_on_no_tool(tmp_path, monkeypatch):
    bank, trace, context = sources(tmp_path, monkeypatch)
    calls = []
    def builder(*args, **kwargs):
        calls.append(kwargs)
        return None, {"call_count": 1}
    monkeypatch.setattr(bank, "_build_tool_for_occurrence", builder)
    try:
        state = prepare_sidecar(bank, trace, None, context)
        assert len(calls) == 1, state
        assert len(calls[0]["additional_evidence_sources"]) == 1
        assert calls[0]["allow_exact_reuse"] is False
        assert state["item"].tool is None
        assert len(state["sources"]) == 2
        events = apply_sidecar(bank, trace, None, state)
        assert state["attempt"]["result_status"] == "no_tool_atomic_retained"
        assert all(event.artifact_kind == "atomic" for event in events)
        assert len(bank.skills.atomics()) == 1 and not bank.tools.tools()
        bank.ledger.append_transaction([], companion_write=lambda connection:
            bank.learning_source_store.commit_attempt(connection, state["attempt"]))
        again = prepare_sidecar(bank, trace, None, context)
        assert again["attempt"] is None and len(calls) == 1
    finally:
        bank.close()


def test_G_same_manifest_task_different_attempt_is_not_second_source(tmp_path, monkeypatch):
    bank, trace, context = sources(tmp_path, monkeypatch)
    history = bank.learning_source_store.verified(bank, context["groups"][0]["history_reference"])
    trace.metadata["execution_source"]["task_signature"] = history["source_identity"]["task_signature"]
    monkeypatch.setattr(bank, "_build_tool_for_occurrence", lambda *a, **k: pytest.fail("same task reached Builder"))
    try:
        state = prepare_sidecar(bank, trace, None, context)
        assert state["item"] is None and not state["sources"]
        assert "two independent" in state["audit"]["reason"]
    finally:
        bank.close()


def test_G_ordinary_admission_is_not_duplicated_by_sidecar(tmp_path, monkeypatch):
    from atomic_skillgraph.evolution.admission_evidence import additional_admissions
    from atomic_skillgraph.governance.ledger import EvidenceEventType
    bank, trace, context = sources(tmp_path, monkeypatch)
    monkeypatch.setattr(bank, "_build_tool_for_occurrence", lambda *a, **k: (None, {"call_count": 1}))
    try:
        state = prepare_sidecar(bank, trace, None, context)
        ref = bank.aligner.align_atomic(state["item"].atomic)
        ordinary = bank.credit.assign_evolution(trace, [ref], [], [], None)
        # Ordinary proposal was rejected; later lawful sidecar can add exactly
        # the missing VALIDATED fact, never overwrite or duplicate PROPOSED.
        bank._commit_evidence([e for e in ordinary if e.event is EvidenceEventType.PROPOSED])
        events = apply_sidecar(bank, trace, None, state)
        assert [e.event for e in events] == [EvidenceEventType.VALIDATED]
        assert additional_admissions(bank, trace, [ref]) == []
        bank._commit_evidence(events)
        assert additional_admissions(bank, trace, [ref]) == []
        rows = bank.database.rows("SELECT event_type FROM evidence_events WHERE artifact_ref=?", (str(ref),))
        assert sorted(r[0] for r in rows) == ["proposed", "validated"]
    finally:
        bank.close()


def test_R_learning_source_revalidates_after_trace_directory_relocation(tmp_path, monkeypatch):
    from atomic_skillgraph.knowledge.source_snapshots import verify_learning_sources
    bank, trace, context = sources(tmp_path, monkeypatch)
    reference = context["groups"][0]["history_reference"]
    try:
        expected = bank.learning_source_store.verified(bank, reference)
        # Frozen banks contain parent snapshots under artifacts, not traces.
        bank.traces.root = tmp_path / "no_external_traces"
        assert bank.learning_source_store.verified(bank, reference) == expected
        verify_learning_sources(bank)
    finally:
        bank.close()


@pytest.mark.parametrize("mutation", ["borrowed_authority", "duplicate_slot", "different_contract", "unoffered_group"])
def test_G_invalid_sidecar_does_not_call_builder_or_publish_partial_contract(tmp_path, monkeypatch, mutation):
    bank, trace, context = sources(tmp_path, monkeypatch)
    submitted = context["batch"].generalizations[0]
    current, history = submitted["source_proposals"]
    if mutation == "borrowed_authority":
        current["proposal"]["input_provenance_refs"] = history["proposal"]["input_provenance_refs"]
    elif mutation == "duplicate_slot":
        history["source_slot"] = "current"
    elif mutation == "different_contract":
        history["proposal"]["input_specs"][0]["required"] = False
    else:
        submitted["group_id"] = "not_offered"
    monkeypatch.setattr(bank, "_build_tool_for_occurrence", lambda *a, **k: pytest.fail("invalid source reached Builder"))
    try:
        state = prepare_sidecar(bank, trace, None, context)
        assert state["item"] is None, state
        assert not apply_sidecar(bank, trace, None, state)
        assert not bank.skills.atomics()
    finally:
        bank.close()


def test_G_native_batch_is_single_request_no_group_means_no_sidecar():
    from atomic_skillgraph.agents.session import ReplayAgentSession
    from atomic_skillgraph.agents import UsageLedger
    from experiments.fakes import ScriptedAgentProvider, FakeReply
    p, n = typed_preparation_example()
    provider = ScriptedAgentProvider([FakeReply.structured({"occurrences": [raw_proposal(p)], "generalizations": []})])
    extractor = ExtractorSession(ReplayAgentSession(provider, system_prompt="E1", usage_ledger=UsageLedger(), usage_bucket="extractor_e1"))
    batch = extractor.propose_batch(n)
    assert len(provider.requests) == 1 and len(batch.occurrences) == 1
    assert batch.occurrences[0].event_end == p.event_end
    assert not batch.generalizations
    assert e1_schema()["properties"]["generalizations"]["maxItems"] == 0


def test_G_source_capsule_revalidation_and_final_commit_conflict(tmp_path, monkeypatch):
    bank, trace, context = sources(tmp_path, monkeypatch)
    try:
        row = context["groups"][0]["history_reference"]
        assert bank.learning_source_store.verified(bank, row)
        changed = dict(row, source_trace_hash="corrupted")
        with pytest.raises(RuntimeError, match="Trace.*hash"):
            bank.learning_source_store.verified(bank, changed)
        with pytest.raises(RuntimeError, match="conflicting"):
            bank.ledger.append_transaction([], companion_write=lambda connection:
                bank.learning_source_store.commit(connection, row, "corrupted"))
    finally:
        bank.close()
