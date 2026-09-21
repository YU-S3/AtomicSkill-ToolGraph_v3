"""Replay reuse requires BOTH complete program and same-source case proofs."""
import copy
from dataclasses import replace

import pytest

from atomic_skillgraph.knowledge.database import StateDatabase
from atomic_skillgraph.governance.ledger import EvidenceLedger
from atomic_skillgraph.evolution.aligner import _tool_signature
from atomic_skillgraph.evolution.replay import ReplayCaseResult, replay_case_id
from atomic_skillgraph.evolution.replay_certificates import ReplayCertificates
from test_r103_identity import tool


def recorded(tmp_path):
    db = StateDatabase(tmp_path / "state.sqlite3", r103=True)
    ledger = EvidenceLedger(db)
    certs = ReplayCertificates(ledger)
    original = tool()
    case = {"trace_id":"source", "occurrence_id":"owner", "event_range":[0,1],
        "source_task":{"task_id":"task", "task_signature":"game"}, "prefix":[],
        "bindings":{"item":"object_7"}, "effects":[{"predicate":"known", "args":{"object":"$result"}}]}
    result = ReplayCaseResult(replay_case_id(case), "source", "task", "task", "task", "complete", True)
    event = certs.certificate(_tool_signature(original), case, result,
        artifact_ref=str(original.ref), trace_id="publishing", task_id="publishing", tool=original, semantic_profile="typed-v1")
    ledger.append_transaction([event])
    candidate = tool("thing", "answer", "pick", "element")
    alias_case = {**case, "bindings":{"thing":"object_7"},
                  "effects":[{"predicate":"known", "args":{"object":"$answer"}}]}
    return db, certs, candidate, alias_case


def test_D_same_source_alias_case_reuses_without_physical_credit(tmp_path):
    db, certs, candidate, case = recorded(tmp_path)
    try:
        result = certs.lookup(_tool_signature(candidate), case, tool=candidate, semantic_profile="typed-v1")
        assert result is not None and result.passed
        assert not result.started and not result.completed and result.executed_action_count == 0
        assert result.case_id == replay_case_id(case)
        assert certs.last_reuse_proof["executable_proof"]["input_role_map"] == {"thing":"item"}
    finally:
        db.close()


@pytest.mark.parametrize("change", ["profile", "binding", "source", "prefix", "range", "effect", "owner", "program"])
def test_D_case_or_semantic_profile_changes_do_not_reuse(tmp_path, change):
    db, certs, candidate, case = recorded(tmp_path)
    profile = "typed-v1"
    if change == "profile": profile = "typed-v2"
    elif change == "binding": case["bindings"]["thing"] = "object_8"
    elif change == "source": case["source_task"]["task_signature"] = "other-game"
    elif change == "prefix": case["prefix"] = [{"action_type":"LOOK", "arguments":{}}]
    elif change == "range": case["event_range"] = [1,2]
    elif change == "effect": case["effects"][0]["predicate"] = "other"
    elif change == "owner": case["occurrence_id"] = "another-owner"
    else: candidate.artifact["max_actions"] += 1
    try:
        assert certs.lookup(_tool_signature(candidate), case, tool=candidate, semantic_profile=profile) is None
    finally:
        db.close()


def test_I11_forged_equal_signature_is_not_replay_identity_authority(tmp_path):
    db,certs,candidate,case = recorded(tmp_path)
    try:
        original = tool()
        # Exact source case and supplied signature, but actual program differs.
        event = certs.events(_tool_signature(original))[0]
        wrong = copy.deepcopy(original)
        wrong.artifact["max_actions"] += 1
        assert certs.lookup(_tool_signature(original),event.metadata["case"],tool=wrong,
                            semantic_profile="typed-v1") is None
    finally:
        db.close()
