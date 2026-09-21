"""R10.3 source-execution observations and immutable source capsules.

This module records facts only. It cannot admit assets or manufacture physical
execution credit. The final Trace hash is stored in the index, never inserted
back into the Trace or its pre-publication capsule.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from ..core.serialization import atomic_create_json, read_json, to_primitive, dataclass_from_dict
from ..evolution.identity_matching import raw_hash, typed_json

OBSERVATION_VERSION = "r103.online.v1"
DDL = """
CREATE TABLE runtime_support_observations (
 execution_key TEXT PRIMARY KEY, observation_version TEXT NOT NULL,
 independent_task_key TEXT NOT NULL, program_equivalence_id TEXT NOT NULL,
 implementation_equivalence_id TEXT NOT NULL, contract_signature TEXT NOT NULL,
 trace_id TEXT NOT NULL, draft_id TEXT NOT NULL, harness_profile TEXT NOT NULL,
 payload_path TEXT NOT NULL, payload_hash TEXT NOT NULL, source_trace_hash TEXT NOT NULL,
 manifest_ordinal INTEGER NOT NULL, attempt_ordinal INTEGER NOT NULL,
 event_ordinal INTEGER NOT NULL, outcome TEXT NOT NULL
);
CREATE INDEX runtime_support_source_order ON runtime_support_observations(manifest_ordinal,attempt_ordinal,event_ordinal);
CREATE INDEX runtime_support_contract ON runtime_support_observations(contract_signature,harness_profile);
CREATE INDEX runtime_support_program_tasks ON runtime_support_observations(program_equivalence_id,independent_task_key);
"""


@dataclass(frozen=True)
class OnlineExecutionObservation:
    observation_version: str
    execution_key: str
    independent_task_key: str
    source_run_id: str
    source_task_id: str
    source_task_signature: str
    source_trace_id: str
    source_attempt_id: str
    source_execution_id: str
    source_order: tuple[int, int, int]
    consumer_scope: str
    local_atomic_ref: str
    local_implementation_ref: str
    local_tool_ref: str
    contract_payload_hash: str
    program_payload_hash: str
    implementation_payload_hash: str
    authorizing_native_call_id: str
    started: bool
    completed: bool
    r1_passed: bool
    atomic_effect_passed: bool
    output_validation_passed: bool
    canonical_retained: bool
    terminal_interrupted: bool
    failure_layer: str
    intrinsic_failure: bool
    parent_completed_after_trial: bool | None
    source_learning_eligible: bool
    source_official_won: bool
    outcome: str

    def __post_init__(self):
        if self.observation_version != OBSERVATION_VERSION:
            raise ValueError("unknown observation protocol")
        required = (self.execution_key, self.independent_task_key, self.source_run_id,
            self.source_task_signature, self.source_trace_id, self.source_attempt_id,
            self.source_execution_id, self.authorizing_native_call_id, self.local_tool_ref,
            self.local_atomic_ref, self.local_implementation_ref, self.contract_payload_hash,
            self.program_payload_hash, self.implementation_payload_hash)
        if not all(isinstance(value, str) and value for value in required):
            raise ValueError("incomplete online execution identity")
        if len(self.source_order) != 3 or any(type(n) is not int or n < 0 for n in self.source_order):
            raise ValueError("source order must use manifest/attempt/event ordinals")
        for name in ("started", "completed", "r1_passed", "atomic_effect_passed", "output_validation_passed",
                     "canonical_retained", "terminal_interrupted", "intrinsic_failure",
                     "source_learning_eligible", "source_official_won"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"observation {name} must be boolean")
        if self.consumer_scope not in {"node", "task"} or self.outcome not in {"success", "failure", "ineligible"}:
            raise ValueError("unknown observation scope/outcome")
        if self.outcome == "success" and not (
            self.started and self.completed and self.r1_passed and self.atomic_effect_passed
            and self.output_validation_passed and self.canonical_retained
            and self.source_learning_eligible and self.source_official_won
            and not self.terminal_interrupted and not self.intrinsic_failure
            and (self.consumer_scope == "task" or self.parent_completed_after_trial is True)):
            raise ValueError("online success lacks full original execution gates")
        if self.outcome == "failure" and not (self.started and self.intrinsic_failure and self.failure_layer == "tool"):
            raise ValueError("negative Tool observation is not a started intrinsic Tool failure")


def execution_identity(source: dict, trace_id: str, execution_id: str, event_ordinal: int) -> dict:
    for key in ("run_id", "task_id", "task_signature", "attempt_id", "benchmark"):
        if not isinstance(source.get(key), str) or not source[key]:
            raise ValueError(f"missing manifest execution authority: {key}")
    order = (source["manifest_ordinal"], source["attempt_ordinal"], event_ordinal)
    if any(type(n) is not int or n < 0 for n in order) or not trace_id or not execution_id:
        raise ValueError("invalid source execution order")
    body = {"run_id": source["run_id"], "task_signature": source["task_signature"],
        "attempt_id": source["attempt_id"], "trace_id": trace_id,
        "actual_execution_id": execution_id, "canonical_attempt_identity": source["attempt_id"]}
    return {"source_execution_key": "execution_" + raw_hash(body),
        "source_independent_task_key": "task_" + raw_hash([source["benchmark"], source["task_signature"]]),
        "source_order": list(order), "source_identity": body}


class ExecutionObservationStore:
    def __init__(self, database: Any, data_dir: str | Path):
        self.database, self.data_dir = database, Path(data_dir)
        exists = database.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='runtime_support_observations'").fetchone()
        if not exists:
            raise RuntimeError("R10.3 observation schema must be created with the fresh bank")
        columns = database.rows('PRAGMA table_info("runtime_support_observations")')
        expected = ("execution_key", "observation_version", "independent_task_key", "program_equivalence_id",
            "implementation_equivalence_id", "contract_signature", "trace_id", "draft_id", "harness_profile",
            "payload_path", "payload_hash", "source_trace_hash", "manifest_ordinal", "attempt_ordinal", "event_ordinal", "outcome")
        if tuple(row["name"] for row in columns) != expected or not columns[0]["pk"]:
            raise RuntimeError("R10.3 requires a fresh execution-observation bank; historical observations are not migrated")

    def stage(self, observation: OnlineExecutionObservation, evidence: dict) -> dict:
        if self.database.readonly:
            raise RuntimeError("Frozen cannot write observation capsules")
        payload = {"observation": asdict(observation), "evidence": evidence}
        digest = raw_hash(payload)
        relative = Path("artifacts") / "execution_sources" / (digest + ".json")
        path = self.data_dir / relative
        if path.exists():
            if typed_json(read_json(path)) != typed_json(payload):
                raise RuntimeError("execution capsule content-address collision")
        else:
            atomic_create_json(path, payload)
        return {"execution_key": observation.execution_key, "payload_path": relative.as_posix(), "payload_hash": digest}

    def read_capsule(self, reference: dict) -> dict:
        path = (self.data_dir / reference["payload_path"]).resolve()
        if not path.is_relative_to(self.data_dir.resolve()):
            raise RuntimeError("execution capsule path escapes knowledge root")
        payload = read_json(path)
        if raw_hash(payload) != reference["payload_hash"]:
            raise RuntimeError("execution capsule hash mismatch")
        OnlineExecutionObservation(**payload["observation"])
        return payload

    def commit(self, connection: Any, reference: dict, source_trace_hash: str) -> None:
        """Caller supplies the original Ledger transaction after Trace publish."""
        if self.database.readonly or not source_trace_hash:
            raise RuntimeError("observation commit requires a published immutable Trace")
        payload = self.read_capsule(reference)
        observation, evidence = payload["observation"], payload["evidence"]
        values = (observation["execution_key"], OBSERVATION_VERSION, observation["independent_task_key"],
            evidence["program_equivalence_id"], evidence["implementation_equivalence_id"], evidence["contract_signature"],
            observation["source_trace_id"], evidence["draft_id"], evidence["harness_profile"], reference["payload_path"],
            reference["payload_hash"], source_trace_hash, *observation["source_order"], observation["outcome"])
        existing = connection.execute("SELECT * FROM runtime_support_observations WHERE execution_key=?", (values[0],)).fetchone()
        if existing is not None:
            if tuple(existing) != values:
                raise RuntimeError("same source execution has conflicting observation content")
            return
        connection.execute("INSERT INTO runtime_support_observations VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", values)

    def committed(self, *, contract_signature: str | None = None) -> list[dict]:
        query = "SELECT * FROM runtime_support_observations"
        parameters = ()
        if contract_signature is not None:
            query += " WHERE contract_signature=?"
            parameters = (contract_signature,)
        result = []
        for row in self.database.rows(query + " ORDER BY manifest_ordinal,attempt_ordinal,event_ordinal,execution_key", parameters):
            if not row["source_trace_hash"]:
                raise RuntimeError("committed observation has no immutable Trace hash")
            payload = self.read_capsule(dict(row))
            payload["source_trace_hash"] = row["source_trace_hash"]
            payload["reference"] = {key: row[key] for key in ("execution_key", "payload_path", "payload_hash")}
            result.append(payload)
        return result


def manifest_source(system: Any, trace: Any) -> dict | None:
    """Resolve source identity from the frozen selection, never goal text/path guesses."""
    experiment = system.config.get("experiment", {})
    path = experiment.get("task_manifest_path")
    if not path:
        return None
    document = read_json(system._resolve_path(path))
    from experiments.protocol import TaskManifest, hash_task_manifest
    if hash_task_manifest([TaskManifest.from_dict(row) for row in document.get("tasks", [])]) != document.get("task_manifest_hash"):
        raise RuntimeError("execution manifest hash mismatch")
    tasks = document.get("tasks", [])
    matches = [row for row in tasks if row.get("task_signature") == trace.task.task_signature]
    if len(matches) != 1 or matches[0].get("task_id") != trace.task.task_id:
        raise RuntimeError("execution source does not uniquely match the committed task manifest")
    task = matches[0]
    run_id = str(experiment.get("name", ""))
    run = system.database.execute("SELECT task_manifest_hash,phase FROM run_manifests WHERE run_id=?", (run_id,)).fetchone()
    if run is None:
        return None
    if run["task_manifest_hash"] != document["task_manifest_hash"] or run["phase"] != "train":
        raise RuntimeError("execution manifest is not the current committed training selection")
    attempt_id = trace.metadata.get("attempt_id", "")
    row = system.database.execute("SELECT attempt_count FROM run_tasks WHERE run_id=? AND task_id=?",
        (run_id, trace.task.task_id)).fetchone()
    if row is None or not attempt_id:
        return None  # A diagnostic without a formal attempt cannot mint online credit.
    source = {"run_id": run_id, "task_id": task["task_id"], "task_signature": task["task_signature"],
        "benchmark": task.get("benchmark", trace.task.benchmark), "manifest_ordinal": task["ordinal"],
        "attempt_id": attempt_id, "attempt_ordinal": int(row["attempt_count"]),
        "task_manifest_hash": document["task_manifest_hash"], "split": task.get("split", ""),
        "experiment_kind": experiment.get("experiment_kind", system.config.get("experiment_kind", "formal")),
        "learning_condition": system.config.get("r103_learning_intervention", "Full")}
    execution_identity(source, trace.trace_id, "identity_check", 0)
    return source


from ..evolution.learning_interventions import full_proof


@full_proof
def collect_execution_observations(trace: Any, harness_profile: str) -> list[tuple[OnlineExecutionObservation, dict]]:
    """Extract only cross-checked physical executions; missing evidence is not PASS."""
    from ..traces.canonical import canonical_trace_records, canonical_action_indices
    from ..core.contracts import AbstractAtomicSkill
    from ..evolution.contract_canonicalizer import atomic_contract_signature
    payload = to_primitive(trace)
    metadata = payload.get("metadata", {})
    source = metadata.get("execution_source")
    from ..evolution.learning_interventions import training_source
    if (not source or payload.get("infrastructure_failure") or source.get("split") != "train"
            or not training_source(source)):
        return []
    retained = {item["attempt_id"] for item in canonical_trace_records(payload, "tool_executions")}
    canonical_actions = set(canonical_action_indices(payload))
    executions = {item["attempt_id"]: (index, item) for index, item in enumerate(payload.get("tool_executions", []))}
    invocations = {item["attempt_id"]: item for item in payload.get("implementation_invocations", [])}
    calls = {item["call_id"]: item for item in payload.get("native_tool_calls", [])}
    trials = metadata.get("runtime_tool_trials", [])
    if isinstance(trials, dict):
        trials = trials.values()
    result = []
    for trial in trials:
        bundle = trial.get("execution_bundle")
        promotion = trial.get("promotion_bundle")
        call_id = trial.get("authorizing_native_call_id")
        call = calls.get(call_id)
        ids = trial.get("tool_execution_ids", [])
        impl_ids = trial.get("implementation_attempt_ids", [])
        if (not bundle or not promotion or len(ids) != 1 or len(impl_ids) != 1
                or ids[0] not in executions or impl_ids[0] not in invocations or call is None):
            continue
        # Native authorization must name this exact draft and owner. The call
        # is recorded after execution but still represents the Agent's request.
        if (call.get("tool_name") != "propose_runtime_automation_atomic"
                or call.get("arguments", {}).get("draft_id") != trial.get("draft_id")
                or call.get("occurrence_id") != trial.get("source_occurrence_id")):
            raise RuntimeError("online trial authorization lineage mismatch")
        ordinal, execution = executions[ids[0]]
        invocation = invocations[impl_ids[0]]
        actual, r1 = execution["result"], trial.get("r1", {})
        from ..core.contracts import ToolAsset, ImplementationAtom
        from ..evolution.identity_matching import match_implementation
        original_a = dataclass_from_dict(AbstractAtomicSkill, bundle["atomic"])
        original_t = dataclass_from_dict(ToolAsset, bundle["tool"])
        original_i = dataclass_from_dict(ImplementationAtom, bundle["implementation"])
        promoted_a = dataclass_from_dict(AbstractAtomicSkill, promotion["atomic"])
        promoted_t = dataclass_from_dict(ToolAsset, promotion["tool"])
        promoted_i = dataclass_from_dict(ImplementationAtom, promotion["implementation"])
        if (execution["tool_ref"] != trial["tool_ref"] or invocation["implementation_ref"] != trial["implementation_ref"]
                or str(original_t.ref) != trial["tool_ref"] or str(original_i.ref) != trial["implementation_ref"]
                or str(original_a.ref) != trial["atomic_ref"]):
            raise RuntimeError("online trial execution ref mismatch")
        source_proof = match_implementation(original_i, promoted_i,
            source_atomic=original_a, target_atomic=promoted_a,
            source_tools={str(original_t.ref): original_t}, target_tools={str(promoted_t.ref): promoted_t})
        if source_proof.status != "exact":
            # A renamed/changed program contract cannot inherit this actual
            # execution. Keep the original fact (especially negative history)
            # rather than dropping it. UNKNOWN never creates positive credit.
            promotion = bundle
        start, end = trial.get("trial_event_start"), trial.get("trial_event_end")
        if type(start) is not int or type(end) is not int:
            continue
        canonical = ids[0] in retained and end >= start and set(range(start, end + 1)) <= canonical_actions
        started = actual.get("started") is True and r1.get("started") is True
        completed = actual.get("completed") is True and r1.get("tool_completed") is True
        effect = actual.get("atomic_effect_passed") is True and r1.get("atomic_effect_passed") is True
        # Empty output interfaces are legal: the original output validator,
        # not truthiness of a result dictionary, is the authority.
        output = r1.get("outputs_valid") is True and isinstance(trial.get("r1_outputs"), dict)
        terminal = actual.get("terminal_interrupted") is not False or r1.get("terminal_interrupted") is not False
        intrinsic = actual.get("intrinsic_failure") is True and r1.get("tool_intrinsic_failure") is True
        layer = str(actual.get("failure_layer", ""))
        passed = r1.get("admission_eligible") is True and r1.get("executed_path_effects_passed") is True
        scope = trial.get("consumer_scope")
        parent = trial.get("parent_completed_after_trial") if scope == "node" else None
        success = (source_proof.status == "exact" and started and completed and effect and output and passed and canonical and not terminal and not intrinsic
            and payload.get("learning_eligible") is True and payload.get("benchmark_success") is True
            and (scope == "task" or parent is True))
        outcome = "success" if success else "failure" if started and intrinsic and layer == "tool" else "ineligible"
        identity = execution_identity(source, payload["trace_id"], ids[0], ordinal)
        observation = OnlineExecutionObservation(OBSERVATION_VERSION, identity["source_execution_key"],
            identity["source_independent_task_key"], source["run_id"], source["task_id"], source["task_signature"],
            payload["trace_id"], source["attempt_id"], ids[0], tuple(identity["source_order"]), scope,
            trial["atomic_ref"], trial["implementation_ref"], trial["tool_ref"], raw_hash(bundle["atomic"]),
            raw_hash(bundle["tool"]), raw_hash(bundle["implementation"]), call_id, started, completed, passed,
            effect, output, canonical, terminal, layer, intrinsic, parent,
            payload.get("learning_eligible") is True, payload.get("benchmark_success") is True, outcome)
        atomic = dataclass_from_dict(AbstractAtomicSkill, promotion["atomic"])
        evidence = {"source_identity": source, "execution_identity": identity, "raw_bundle": bundle,
            "source_to_promotion_proof": to_primitive(source_proof.proof),
            "source_to_promotion_status": source_proof.status,
            "bundle": promotion, "draft_id": trial["draft_id"], "harness_profile": harness_profile,
            "implementation_attempt_id": impl_ids[0],
            "implementation_event_ordinal": next(i for i, row in enumerate(payload.get("implementation_invocations", [])) if row["attempt_id"] == impl_ids[0]),
            "tool_execution": execution, "implementation_invocation": invocation,
            "authorizing_native_call": call, "trial": trial,
            "contract_signature": atomic_contract_signature(atomic),
            # These are conservative retrieval keys, not equivalence authority.
            "program_equivalence_id": "unresolved:" + raw_hash(bundle["tool"]),
            "implementation_equivalence_id": "unresolved:" + raw_hash(bundle["implementation"])}
        result.append((observation, evidence))
    return result


def verify_committed_source(system, source):
    """Re-read the immutable parent and repeat collection, not merely its hash."""
    from .source_snapshots import load_parent
    payload = load_parent(system, source["source_trace_hash"], source["observation"]["source_trace_id"])
    matches = [(observation, evidence) for observation, evidence in collect_execution_observations(
        payload, source["evidence"]["harness_profile"])
        if observation.execution_key == source["observation"]["execution_key"]]
    if len(matches) != 1:
        raise RuntimeError("online source execution no longer resolves uniquely")
    observed, evidence = matches[0]
    if typed_json(asdict(observed)) != typed_json(source["observation"]) or typed_json(evidence) != typed_json(source["evidence"]):
        raise RuntimeError("online source observation does not match its immutable Trace")
    return source
