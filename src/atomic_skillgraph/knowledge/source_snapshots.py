"""Relocatable immutable parent Traces, staged after Trace publication.

The final parent hash lives in indexes, never inside its own Trace/capsule.
Only these content-addressed copies cross freeze; no external path authority.
"""
from pathlib import Path

from ..core.serialization import atomic_create_json, read_json
from ..evolution.identity_matching import raw_hash, typed_json


def publish(data_dir, payload):
    digest = raw_hash(payload)
    path = Path(data_dir) / "artifacts" / "source_traces" / (digest + ".json")
    if path.exists():
        if typed_json(read_json(path)) != typed_json(payload):
            raise RuntimeError("source Trace content-address collision")
    else:
        atomic_create_json(path, payload)
    return digest


def read(data_dir, digest, trace_id):
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise RuntimeError("invalid immutable parent Trace hash")
    payload = read_json(Path(data_dir) / "artifacts" / "source_traces" / (digest + ".json"))
    if raw_hash(payload) != digest or payload.get("trace_id") != trace_id:
        raise RuntimeError("immutable parent Trace identity/hash mismatch")
    return payload


def load_parent(system, digest, trace_id):
    """Use the original if present; only a missing file allows frozen fallback.

    A corrupt original must not be hidden by a valid secondary copy.
    """
    try:
        payload = system.traces.load_payload(trace_id)
    except FileNotFoundError:
        return read(system.data_dir, digest, trace_id)
    if raw_hash(payload) != digest or payload.get("trace_id") != trace_id:
        raise RuntimeError("immutable source Trace identity/hash mismatch")
    return payload


def verify_learning_sources(system):
    """Full original normalizer/Atomicizer validation, without writes or replay."""
    for row in system.database.rows("SELECT * FROM learning_source_index"):
        system.learning_source_store.verified(system, row)


def verify_bank(database, data_dir):
    """Read-only structural/source integrity used by preflight and freeze."""
    from .execution_observations import ExecutionObservationStore, collect_execution_observations
    from .learning_sources import LearningSourceStore
    from .r103_protocol import METADATA
    tables = {row[0] for row in database.rows("SELECT name FROM sqlite_master WHERE type='table'")}
    if "learning_source_index" not in tables:
        return  # Historical bank reader, not migration.
    execution_store = ExecutionObservationStore(database, data_dir)
    for source in execution_store.committed():
        obs = source["observation"]
        parent = read(data_dir, source["source_trace_hash"], obs["source_trace_id"])
        matches = [(o, e) for o, e in collect_execution_observations(parent, source["evidence"]["harness_profile"])
                   if o.execution_key == obs["execution_key"]]
        if len(matches) != 1 or typed_json(matches[0][0]) != typed_json(obs) or typed_json(matches[0][1]) != typed_json(source["evidence"]):
            raise RuntimeError("frozen observation does not reproduce its parent Trace")
    learning_store = LearningSourceStore(database, data_dir)
    for row in database.rows("SELECT * FROM learning_source_index"):
        payload = learning_store.read(row)
        parent = read(data_dir, row["source_trace_hash"], row["source_trace_id"])
        if (payload["protocol"] != METADATA or raw_hash(payload["normalized"]) != row["canonical_snapshot_hash"]
                or payload["sample_key"] != row["sample_key"]
                or payload["source_identity"] != parent.get("metadata", {}).get("execution_source")
                or parent.get("benchmark_success") is not True or parent.get("learning_eligible") is not True):
            raise RuntimeError("frozen learning source identity/hash mismatch")
        expected = {(row["sample_key"], payload["features"]["profile"], item[1], item[0]) for item in payload["features"]["effects"]}
        actual = {tuple(item) for item in database.rows("SELECT * FROM learning_source_effect_index WHERE sample_key=?", (row["sample_key"],))}
        if expected != actual:
            raise RuntimeError("learning source inverted effect index mismatch")
    for row in database.rows("SELECT * FROM generalization_attempts"):
        record = dict(row)
        digest = record.pop("result_payload_hash")
        if raw_hash(record) != digest:
            raise RuntimeError("generalization attempt integrity mismatch")
    for row in database.rows("SELECT * FROM candidate_route_exposures"):
        parent = read(data_dir, row["source_trace_hash"], row["source_trace_id"])
        from .execution_observations import execution_identity
        source = parent["metadata"]["execution_source"]
        key = execution_identity(source, row["source_trace_id"], "display", 0)["source_independent_task_key"]
        import json
        sessions = sorted({e["session_id"] for e in parent["metadata"].get("candidate_route_exposures", [])
                           if e["route_key"] == row["route_key"]})
        if key != row["independent_task_key"] or sessions != json.loads(row["session_ids_json"]):
            raise RuntimeError("candidate display index differs from published Trace")
    for row in database.rows("SELECT * FROM runtime_admission_attempts"):
        parent = read(data_dir, row["publishing_trace_hash"], row["publishing_trace_id"])
        records = [item for item in parent.get("metadata", {}).get("runtime_admission_attempts", [])
                   if item["attempt_key"] == row["attempt_key"]]
        if len(records) != 1 or raw_hash(records[0]) != row["result_payload_hash"]:
            raise RuntimeError("runtime admission attempt differs from published Trace")
        if any(records[0][k] != row[k] for k in records[0]):
            raise RuntimeError("runtime admission attempt index mismatch")


def commit_admissions(connection, trace, digest):
    for record in trace.metadata.get("runtime_admission_attempts", []):
        if record["result_status"] == "prepared":
            raise RuntimeError("unfinished runtime admission cannot commit")
        values = tuple(record[k] for k in ("attempt_key", "source_keys_json", "result_status", "publishing_trace_id")) + (digest, raw_hash(record))
        previous = connection.execute("SELECT * FROM runtime_admission_attempts WHERE attempt_key=?", (values[0],)).fetchone()
        if previous is not None:
            if tuple(previous) != values:
                raise RuntimeError("runtime admission result conflicts with prior committed attempt")
        else:
            connection.execute("INSERT INTO runtime_admission_attempts VALUES(?,?,?,?,?,?)", values)


def commit_displays(connection, trace, digest):
    from .execution_observations import execution_identity
    import json
    source = trace.metadata.get("execution_source")
    if source is None:
        return
    independent = execution_identity(source, trace.trace_id, "display", 0)["source_independent_task_key"]
    groups = {}
    for row in trace.metadata.get("candidate_route_exposures", []):
        groups.setdefault(row["route_key"], set()).add(row["session_id"])
    for route, sessions in groups.items():
        values = (independent, route, trace.trace_id, digest, json.dumps(sorted(sessions)))
        previous = connection.execute("SELECT * FROM candidate_route_exposures WHERE independent_task_key=? AND route_key=?", values[:2]).fetchone()
        if previous is not None:
            # Count independent tasks, not retries or repeated exposure. Keep
            # the first immutable source; later full exposure remains in Trace.
            if previous["source_trace_id"] == trace.trace_id and tuple(previous) != values:
                raise RuntimeError("committed task display evidence conflicts")
        else:
            connection.execute("INSERT INTO candidate_route_exposures VALUES(?,?,?,?,?)", values)
