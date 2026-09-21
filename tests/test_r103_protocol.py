"""Fresh-bank metadata, formal resource gates, freeze and source tampering."""
import copy
from pathlib import Path

import pytest

from atomic_skillgraph.system import load_config
from atomic_skillgraph.knowledge.database import StateDatabase
from atomic_skillgraph.knowledge.r103_protocol import METADATA
from experiments.protocol import ProtocolError
from experiments.run_v3_train import _train_protocol, _validate_formal_config as train_guard
from experiments.run_v3_frozen_eval import _frozen_protocol, _validate_formal_config as eval_guard


@pytest.mark.parametrize("seed", [42,43,44])
@pytest.mark.parametrize("phase", ["train_full_120", "frozen_eval_134"])
def test_R_formal_configs_validate_and_reject_interventions(seed, phase):
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / f"alfworld_{phase}_r103_seed{seed}.yaml")
    protocol, guard, expected = ((_train_protocol, train_guard, ("r103_full120",seed,20,120))
        if phase.startswith("train") else (_frozen_protocol, eval_guard, ("r103_frozen134",seed,0,134)))
    assert protocol(config) == expected
    output = root / config["experiment"]["output_dir"]
    guard(config, output)
    config["r103_interventions"] = {"expression": "original"}
    with pytest.raises(ProtocolError, match="interventions"):
        guard(config, output)


def test_R_bank_versions_no_implicit_migration_and_readonly(tmp_path):
    path = tmp_path / "r103" / "state.sqlite3"
    with StateDatabase(path, r103=True) as database:
        metadata = dict(database.rows("SELECT key,value FROM metadata"))
        assert all(metadata[k] == v for k,v in METADATA.items())
    with StateDatabase(path, r103=True, readonly=True) as database:
        assert database.readonly
    old = tmp_path / "old" / "state.sqlite3"
    with StateDatabase(old):
        pass
    original = old.read_bytes()
    with pytest.raises(RuntimeError, match="R10.3"):
        StateDatabase(old, r103=True)
    assert old.read_bytes() == original


def test_R_source_snapshots_are_relocatable_and_integrity_checked(tmp_path):
    import shutil
    from atomic_skillgraph.knowledge.source_snapshots import publish, read
    payload = {"trace_id": "original", "metadata": {"source": "immutable"}}
    digest = publish(tmp_path / "source", payload)
    shutil.copytree(tmp_path / "source", tmp_path / "frozen")
    assert read(tmp_path / "frozen", digest, "original") == payload
    with pytest.raises(RuntimeError, match="identity/hash"):
        read(tmp_path / "frozen", digest, "forged")
    with pytest.raises(RuntimeError, match="hash"):
        read(tmp_path / "frozen", "../escape", "original")


def test_R_inverted_index_schema_and_identity_completeness_fail_closed(tmp_path):
    from atomic_skillgraph.knowledge.artifact_store import ArtifactStore
    from test_r103_identity import atomic
    with StateDatabase(tmp_path / "state.sqlite3", r103=True) as db:
        store = ArtifactStore(tmp_path, db)
        store.put("atomic", atomic())
        with db.transaction() as connection:
            connection.execute("DELETE FROM artifact_identity_index")
        with pytest.raises(RuntimeError, match="missing identity"):
            store.verify_all()
        db.execute("DROP INDEX learning_effect_recall")
        with pytest.raises(RuntimeError, match="index mismatch"):
            from atomic_skillgraph.knowledge.r103_protocol import validate
            validate(db.connection)


def test_R_admission_attempt_companion_rollback_and_immutable_parent(tmp_path):
    from types import SimpleNamespace
    from atomic_skillgraph.knowledge.source_snapshots import publish, commit_admissions, verify_bank
    from atomic_skillgraph.governance.ledger import EvidenceLedger
    trace = SimpleNamespace(trace_id="trace_admit",metadata={"runtime_admission_attempts":[{
        "attempt_key":"pair","source_keys_json":'["exec1", "exec2"]',
        "result_status":"tool_admission_rejected","publishing_trace_id":"trace_admit"}]})
    with StateDatabase(tmp_path / "state.sqlite3",r103=True) as db:
        ledger = EvidenceLedger(db)
        digest = publish(tmp_path,{"trace_id":trace.trace_id,"metadata":trace.metadata})
        def interrupted(connection):
            commit_admissions(connection,trace,digest)
            raise RuntimeError("injected interruption after index write")
        with pytest.raises(RuntimeError,match="injected interruption"):
            ledger.append_transaction([],companion_write=interrupted)
        assert not db.rows("SELECT * FROM runtime_admission_attempts")
        for _ in range(2):
            ledger.append_transaction([],companion_write=lambda c:commit_admissions(c,trace,digest))
        assert len(db.rows("SELECT * FROM runtime_admission_attempts")) == 1
        verify_bank(db,tmp_path)
        trace.metadata["runtime_admission_attempts"][0]["result_status"] = "candidate_admitted"
        with pytest.raises(RuntimeError,match="conflicts"):
            ledger.append_transaction([],companion_write=lambda c:commit_admissions(c,trace,digest))
