"""Immutable successful learning sources; retrieval is not equivalence proof."""
import copy
import json
from pathlib import Path

from ..core.serialization import atomic_create_json, read_json, to_primitive
from ..evolution.identity_matching import raw_hash, typed_json
from ..evolution.contract_canonicalizer import atomic_contract_signature
from ..evolution.learning_interventions import full_proof
from .execution_observations import execution_identity
from .r103_protocol import METADATA


def features(atomic, occurrence, profile):
    return {"profile": profile,
        "effects": sorted((p.predicate, p.effect_domain, p.cardinality, p.distinct_by, tuple(sorted(p.args))) for p in atomic.effects),
        "boundary": sorted((direction, p.semantic_type, p.required_resolution, p.required, p.runtime_resolvable)
            for direction, specs in (("input", atomic.inputs), ("output", atomic.outputs)) for p in specs),
        "preconditions": sorted((p.predicate, p.effect_domain, tuple(sorted(p.args))) for p in atomic.preconditions),
        "work": [event["action_type"] for event in occurrence.action_events],
        "derivations": sorted(v.get("kind", "") for v in atomic.validator_spec.get("output_derivations", {}).values())}


def retrieval_rank(feature, normalized, attempted_features, sample_key):
    """Structural lexicographic retrieval only; never a similarity admission."""
    from collections import Counter
    authorities = normalized.get("boundary_authorities", {})
    effects = Counter(typed_json((f["predicate"], f.get("effect_domain", "world"),
        f.get("cardinality", 1), f.get("distinct_by", ""), sorted(f.get("args", {}))))
        for f in authorities.get("effects", []))
    history = Counter(typed_json(item) for item in feature["effects"])
    common = sum((effects & history).values())
    # Derivation shapes are compared to available source authority shapes,
    # not concrete output values or a benchmark-specific type dictionary.
    available = set()
    if authorities.get("inputs"):
        available.add("input_identity")
    if authorities.get("effects"):
        available.add("effect_witness")
    compatible = sum(kind in available for kind in feature["derivations"])
    current_boundary = {(f.get("semantic_type"), f.get("resolution")) for f in authorities.get("inputs", [])}
    history_boundary = {(p[1], p[2]) for p in feature["boundary"] if p[0] == "input"}
    difference = len(history_boundary - current_boundary)
    structural = typed_json({k: feature[k] for k in ("effects", "boundary", "preconditions", "work", "derivations")})
    return (-common, -compatible, difference, structural in attempted_features, sample_key)


class LearningSourceStore:
    def __init__(self, database, data_dir):
        self.database, self.data_dir = database, Path(data_dir)

    def _write(self, payload):
        if self.database.readonly:
            raise RuntimeError("Frozen cannot write learning sources")
        digest = raw_hash(payload)
        relative = Path("artifacts/learning_sources") / (digest + ".json")
        path = self.data_dir / relative
        if path.exists():
            if typed_json(read_json(path)) != typed_json(payload):
                raise RuntimeError("learning source content collision")
        else:
            atomic_create_json(path, payload)
        return {"capsule_path": relative.as_posix(), "capsule_hash": digest}

    def read(self, reference):
        path = (self.data_dir / reference["capsule_path"]).resolve()
        if not path.is_relative_to(self.data_dir.resolve()):
            raise RuntimeError("learning capsule path escapes bank")
        payload = read_json(path)
        if raw_hash(payload) != reference["capsule_hash"]:
            raise RuntimeError("learning source capsule integrity mismatch")
        return payload

    @full_proof
    def stage(self, trace, normalized, occurrence, atomic, profile, *, proposal):
        source = trace.metadata.get("execution_source")
        from ..evolution.learning_interventions import training_source
        if not source or source.get("split") != "train" or not training_source(source) or not trace.learning_eligible or not trace.benchmark_success:
            return None
        identity = execution_identity(source, trace.trace_id, occurrence.occurrence_id, occurrence.event_start)
        snapshot = raw_hash(normalized)
        signature = atomic_contract_signature(atomic)
        sample = "sample_" + raw_hash([identity["source_independent_task_key"], trace.trace_id,
            occurrence.support_event_ids, signature])
        f = features(atomic, occurrence, profile)
        payload = {"protocol": METADATA, "source_identity": source, "normalized": normalized,
            "canonical_snapshot_hash": snapshot, "occurrence": to_primitive(occurrence),
            "atomic": to_primitive(atomic), "proposal": to_primitive(proposal), "features": f, "sample_key": sample}
        reference = self._write(payload)
        return {**reference, "sample_key": sample, "independent_task_key": identity["source_independent_task_key"],
            "source_trace_id": trace.trace_id, "canonical_snapshot_hash": snapshot,
            "source_occurrence_id": occurrence.occurrence_id, "contract_identity_key": signature,
            "feature_payload_json": json.dumps(f, sort_keys=True)}

    def commit(self, connection, reference, trace_hash):
        if not trace_hash:
            raise RuntimeError("learning source requires immutable parent Trace hash")
        self.read(reference)
        columns = ("sample_key", "independent_task_key", "source_trace_id", "canonical_snapshot_hash",
                   "source_occurrence_id", "contract_identity_key", "feature_payload_json", "capsule_path", "capsule_hash")
        previous = connection.execute("SELECT * FROM learning_source_index WHERE sample_key=?", (reference["sample_key"],)).fetchone()
        if previous is not None:
            if any(previous[k] != reference[k] for k in columns) or previous["source_trace_hash"] != trace_hash:
                raise RuntimeError("conflicting learning source identity")
            return
        sequence = connection.execute("SELECT COALESCE(MAX(committed_sequence),0)+1 FROM learning_source_index").fetchone()[0]
        connection.execute("INSERT INTO learning_source_index VALUES(?,?,?,?,?,?,?,?,?,?,?)", (
            reference["sample_key"], reference["independent_task_key"], reference["source_trace_id"], trace_hash,
            reference["canonical_snapshot_hash"], reference["source_occurrence_id"], reference["contract_identity_key"],
            reference["feature_payload_json"], reference["capsule_path"], reference["capsule_hash"], sequence))
        feature = json.loads(reference["feature_payload_json"])
        for predicate, domain in sorted({(item[0], item[1]) for item in feature["effects"]}):
            connection.execute("INSERT INTO learning_source_effect_index VALUES(?,?,?,?)",
                               (reference["sample_key"], feature["profile"], domain, predicate))

    @full_proof
    def verified(self, system, row):
        payload = self.read(row)
        if payload["protocol"] != METADATA:
            raise RuntimeError("learning source protocol mismatch")
        from .source_snapshots import load_parent
        from ..traces.store import TraceStore
        trace = TraceStore.from_payload(load_parent(system, row["source_trace_hash"], row["source_trace_id"]))
        normalized = system._normalized_learning_source(trace)
        if raw_hash(normalized) != row["canonical_snapshot_hash"] or typed_json(normalized) != typed_json(payload["normalized"]):
            raise RuntimeError("learning source no longer matches canonical online facts")
        from ..core.serialization import dataclass_from_dict
        from ..evolution.atomicizer import AtomicOccurrenceProposal
        from ..evolution.identity_matching import match_atomic
        from ..core.contracts import AbstractAtomicSkill
        validated, rejected = system.atomicizer.validate_proposed_subset(
            [dataclass_from_dict(AtomicOccurrenceProposal, payload["proposal"])], normalized)
        if rejected or len(validated) != 1 or match_atomic(system._canonical_atomic_for_occurrence(validated[0]),
            dataclass_from_dict(AbstractAtomicSkill, payload["atomic"])).status != "exact":
            raise RuntimeError("learning source proposal cannot reproduce its attested contract")
        occurrence = validated[0]
        # Atomicizer assigns occurrence IDs by position in its submitted
        # batch. A single-source recheck has position zero, whereas the saved
        # occurrence may be any member of the original accepted batch. Retain
        # its indexed allocation identity, but compare every evidence field.
        source_occurrence_id = row["source_occurrence_id"]
        if not source_occurrence_id or payload["occurrence"]["occurrence_id"] != source_occurrence_id:
            raise RuntimeError("learning source occurrence index identity mismatch")
        occurrence.occurrence_id = source_occurrence_id
        if typed_json(to_primitive(occurrence)) != typed_json(payload["occurrence"]):
            raise RuntimeError("learning source occurrence differs from revalidated evidence")
        source = trace.metadata.get("execution_source", {})
        from ..evolution.learning_interventions import training_source
        if (source != payload["source_identity"] or source.get("split") != "train"
                or not training_source(source) or not trace.learning_eligible or not trace.benchmark_success):
            raise RuntimeError("learning source is not a successful training source")
        identity = execution_identity(source, trace.trace_id, occurrence.occurrence_id, occurrence.event_start)
        signature = atomic_contract_signature(dataclass_from_dict(AbstractAtomicSkill, payload["atomic"]))
        expected = "sample_" + raw_hash([identity["source_independent_task_key"], trace.trace_id,
                                        occurrence.support_event_ids, signature])
        if (expected != row["sample_key"] or expected != payload["sample_key"]
                or identity["source_independent_task_key"] != row["independent_task_key"]
                or signature != row["contract_identity_key"]):
            raise RuntimeError("learning source index identity mismatch")
        return payload

    def recall(self, system, trace, normalized):
        source = trace.metadata.get("execution_source")
        if source is None:
            return [], []
        independent = execution_identity(source, trace.trace_id, "recall", 0)["source_independent_task_key"]
        current_effects = {(f["predicate"], f.get("effect_domain", "world")) for f in normalized.get("boundary_authorities", {}).get("effects", [])}
        if not current_effects:
            return [], []
        candidates = []
        attempted_hashes = {digest for row in self.database.rows("SELECT source_snapshot_hashes_json FROM generalization_attempts")
                            for digest in json.loads(row[0])}
        attempted_features = set()
        recalled = {}
        for predicate, domain in sorted(current_effects):
            for row in self.database.rows("SELECT source.* FROM learning_source_effect_index AS effect "
                    "JOIN learning_source_index AS source USING(sample_key) "
                    "WHERE effect.harness_profile=? AND effect.effect_domain=? AND effect.predicate=? "
                    "AND source.independent_task_key<>?", (system.harness.profile_name, domain, predicate, independent)):
                recalled[row["sample_key"]] = row
        rows = [recalled[key] for key in sorted(recalled)]
        for row in rows:
            feature = json.loads(row["feature_payload_json"])
            if row["canonical_snapshot_hash"] in attempted_hashes:
                attempted_features.add(typed_json({k: feature[k] for k in ("effects", "boundary", "preconditions", "work", "derivations")}))
        for row in rows:
            feature = json.loads(row["feature_payload_json"])
            common = current_effects & {(item[0], item[1]) for item in feature["effects"]}
            if common:
                candidates.append((retrieval_rank(feature, normalized, attempted_features, row["sample_key"]), dict(row), feature))
        candidates.sort(key=lambda item: item[0])
        summaries = [{"sample_key": row["sample_key"], "features": feature} for _,row,feature in candidates[:4]]
        for _,row,feature in candidates[:4]:
            historical = self.verified(system, row)
            # Conservative byte allowance drawn from the *existing* remaining
            # learning budget, not a new pool or a claim of measured tokens.
            # The provider/session still enforces and accounts actual usage.
            input_bytes = len(typed_json(historical["normalized"]).encode("utf-8")) + len(typed_json(normalized).encode("utf-8"))
            if input_bytes > system._shared_tool_builder_tokens("evolution"):
                continue
            group = {"group_id": "group_" + raw_hash([raw_hash(normalized), row["canonical_snapshot_hash"], METADATA])[:24],
                "history": copy.deepcopy(historical["normalized"]), "history_reference": row,
                "current_snapshot_hash": raw_hash(normalized)}
            return summaries, [group]
        return summaries, []

    def commit_attempt(self, connection, attempt):
        payload = {key: value for key,value in attempt.items() if key != "result_payload_hash"}
        if raw_hash(payload) != attempt["result_payload_hash"]:
            raise RuntimeError("generalization attempt result hash mismatch")
        values = tuple(attempt[key] for key in ("attempt_key", "source_pair_key", "protocol_version", "group_id",
            "source_snapshot_hashes_json", "result_status", "result_refs_json", "publishing_trace_id", "result_payload_hash"))
        previous = connection.execute("SELECT * FROM generalization_attempts WHERE attempt_key=?", (attempt["attempt_key"],)).fetchone()
        if previous is not None:
            if tuple(previous) != values:
                raise RuntimeError("same source pair/target/protocol has conflicting results")
            return
        connection.execute("INSERT INTO generalization_attempts VALUES(?,?,?,?,?,?,?,?,?)", values)

    def prepare_attestations(self, system, trace):
        from ..core.serialization import dataclass_from_dict
        from ..core.contracts import AbstractAtomicSkill
        events = []
        for row in self.database.rows("SELECT * FROM learning_source_index ORDER BY committed_sequence,sample_key"):
            source = dict(row)
            payload = self.verified(system, source)
            source_atomic = dataclass_from_dict(AbstractAtomicSkill, payload["atomic"])
            for target in system.skills.atomics():
                immutable = dataclass_from_dict(AbstractAtomicSkill, system.artifacts.get_payload(str(target.ref)))
                event = system.credit.attest_canonical_source(source=source, source_atomic=source_atomic, target_atomic=immutable,
                    publishing_task_id=trace.task.task_id, publishing_trace_id=trace.trace_id)
                if event is None:
                    continue
                previous = self.database.execute("SELECT certificate_hash FROM execution_attribution_index "
                    "WHERE target_ref=? AND source_execution_key=? AND evidence_class=?",
                    (event.artifact_ref, event.metadata["source_execution_key"], event.metadata["source_class"])).fetchone()
                if previous is not None:
                    if previous["certificate_hash"] != event.metadata["attribution_certificate_hash"]:
                        raise RuntimeError("canonical source attribution changed after commit")
                else:
                    events.append(event)
        return events
