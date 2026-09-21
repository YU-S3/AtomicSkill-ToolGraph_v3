"""Recheck deterministic credit certificates at the Ledger trust boundary."""
from ..core.contracts import AbstractAtomicSkill, ToolAsset, ImplementationAtom
from ..core.serialization import dataclass_from_dict
from ..evolution.identity_matching import (IdentityProof, raw_hash, typed_json,
    verify_atomic_proof, verify_tool_proof, verify_implementation_proof)
from ..knowledge.identity_index import _read_artifact, _closure


def verify(event, database):
    metadata = event.metadata
    body = {k:v for k,v in metadata.items() if k != "attribution_certificate_hash"}
    if raw_hash(body) != metadata.get("attribution_certificate_hash"):
        raise ValueError("attribution certificate content hash mismatch")
    root = database.path.parent
    source_class = metadata.get("source_class")
    if source_class == "runtime_online_trial":
        from ..knowledge.execution_observations import ExecutionObservationStore, execution_identity, collect_execution_observations
        from ..knowledge.source_snapshots import read
        reference = metadata["source_observation"]
        row = database.execute("SELECT * FROM runtime_support_observations WHERE execution_key=?", (reference["execution_key"],)).fetchone()
        if row is None or row["source_trace_hash"] != metadata["source_trace_hash"]:
            raise ValueError("attribution source has not committed")
        if any(row[k] != reference[k] for k in ("execution_key", "payload_path", "payload_hash")):
            raise ValueError("attribution source reference mismatch")
        capsule = ExecutionObservationStore(database, root).read_capsule(reference)
        observation, evidence = capsule["observation"], capsule["evidence"]
        parent = read(root, row["source_trace_hash"], row["trace_id"])
        reproduced = [(o,e) for o,e in collect_execution_observations(parent, evidence["harness_profile"])
                      if o.execution_key == observation["execution_key"]]
        if len(reproduced) != 1 or typed_json(reproduced[0]) != typed_json([observation,evidence]):
            raise ValueError("attribution source does not reproduce immutable online facts")
        impl_layer = event.artifact_kind == "implementation"
        identity = execution_identity(evidence["source_identity"], observation["source_trace_id"],
            evidence["implementation_attempt_id"] if impl_layer else observation["source_execution_id"],
            evidence["implementation_event_ordinal"] if impl_layer else observation["source_order"][2])
        if any(metadata.get(k) != v for k,v in identity.items()):
            raise ValueError("attribution execution identity/order mismatch")
        outcome = {"success":"complete_success", "failure":"intrinsic_failure"}.get(observation["outcome"])
        if (outcome != metadata["outcome"] or observation["started"] != metadata.get("source_started")
                or observation["completed"] != metadata.get("source_completed")
                or outcome == "intrinsic_failure" and event.artifact_kind != "tool"):
            raise ValueError("attribution outcome/physical work mismatch")
        source = evidence["bundle"]
    elif source_class == "canonical_learning_source" and event.artifact_kind == "atomic":
        from ..knowledge.learning_sources import LearningSourceStore
        reference = metadata["source_observation"]
        row = database.execute("SELECT * FROM learning_source_index WHERE sample_key=?", (reference["sample_key"],)).fetchone()
        if row is None or row["source_trace_hash"] != metadata["source_trace_hash"]:
            raise ValueError("canonical source has not committed")
        if any(row[k] != reference[k] for k in ("sample_key", "capsule_path", "capsule_hash")):
            raise ValueError("canonical source reference mismatch")
        source = LearningSourceStore(database, root).read(reference)
        expected_key = "canonical_" + raw_hash([row["independent_task_key"], row["canonical_snapshot_hash"], row["sample_key"]])
        if metadata["source_execution_key"] != expected_key or metadata["source_independent_task_key"] != row["independent_task_key"]:
            raise ValueError("canonical source identity mismatch")
    else:
        raise ValueError("unknown attribution evidence class")
    proofs = [IdentityProof(**p) for p in metadata["identity_proofs"]]
    if not proofs or len({p.layer for p in proofs}) != len(proofs):
        raise ValueError("missing or duplicate identity proof layer")
    if not any(p.layer == event.artifact_kind and p.target_ref == event.artifact_ref for p in proofs):
        raise ValueError("proof does not name credited target")
    for proof in proofs:
        cls = {"atomic":AbstractAtomicSkill, "tool":ToolAsset, "implementation":ImplementationAtom}[proof.layer]
        original = dataclass_from_dict(cls, source[proof.layer])
        target = _read_artifact(database, proof.target_ref, cls)
        if proof.layer == "implementation":
            target_dependencies = _closure(database, target)
            tool = dataclass_from_dict(ToolAsset, source["tool"])
            ok = verify_implementation_proof(original, target, proof,
                source_atomic=dataclass_from_dict(AbstractAtomicSkill, source["atomic"]),
                target_atomic=target_dependencies["atomic"], source_tools={str(tool.ref):tool},
                target_tools=target_dependencies["tools"])
        else:
            ok = (verify_atomic_proof if proof.layer == "atomic" else verify_tool_proof)(original, target, proof)
        if not ok:
            raise ValueError("attribution identity proof does not verify")
