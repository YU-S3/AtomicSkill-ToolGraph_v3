"""One version authority for fresh R10.3 banks, proofs and frozen readers."""

METADATA = {
    "r103_protocol_version": "2",
    "identity_version": "r103.identity.v1",
    "execution_support_version": "r103.online.v1",
    "generalization_schema_version": "r103.generalization.v1",
    "runtime_prompt_version": "r103.step.v2",
    "runtime_policy_projection_version": "r103.expression.v1",
    "r103_design_revision": "execution-expression-v1.1",
    "expression_upstream_commit": "760126030eab1d33ec6a6f30988f0f1fb58df3a7",
}

SOURCE_DDL = """
CREATE TABLE runtime_admission_attempts (
 attempt_key TEXT PRIMARY KEY, source_keys_json TEXT NOT NULL, result_status TEXT NOT NULL,
 publishing_trace_id TEXT NOT NULL, publishing_trace_hash TEXT NOT NULL, result_payload_hash TEXT NOT NULL
);
CREATE TABLE learning_source_index (
 sample_key TEXT PRIMARY KEY, independent_task_key TEXT NOT NULL,
 source_trace_id TEXT NOT NULL, source_trace_hash TEXT NOT NULL,
 canonical_snapshot_hash TEXT NOT NULL, source_occurrence_id TEXT NOT NULL,
 contract_identity_key TEXT NOT NULL, feature_payload_json TEXT NOT NULL,
 capsule_path TEXT NOT NULL, capsule_hash TEXT NOT NULL, committed_sequence INTEGER NOT NULL
);
CREATE INDEX learning_source_profile_task ON learning_source_index(json_extract(feature_payload_json,'$.profile'),independent_task_key);
CREATE TABLE learning_source_effect_index (
 sample_key TEXT NOT NULL, harness_profile TEXT NOT NULL, effect_domain TEXT NOT NULL, predicate TEXT NOT NULL,
 PRIMARY KEY(sample_key,harness_profile,effect_domain,predicate),
 FOREIGN KEY(sample_key) REFERENCES learning_source_index(sample_key)
);
CREATE INDEX learning_effect_recall ON learning_source_effect_index(harness_profile,effect_domain,predicate);
CREATE TABLE candidate_route_exposures (
 independent_task_key TEXT NOT NULL, route_key TEXT NOT NULL, source_trace_id TEXT NOT NULL,
 source_trace_hash TEXT NOT NULL, session_ids_json TEXT NOT NULL,
 PRIMARY KEY(independent_task_key,route_key)
);
CREATE TABLE generalization_attempts (
 attempt_key TEXT PRIMARY KEY, source_pair_key TEXT NOT NULL,
 protocol_version TEXT NOT NULL, group_id TEXT NOT NULL,
 source_snapshot_hashes_json TEXT NOT NULL, result_status TEXT NOT NULL,
 result_refs_json TEXT NOT NULL, publishing_trace_id TEXT NOT NULL,
 result_payload_hash TEXT NOT NULL
);
"""

SHAPES = {
    "learning_source_effect_index": (("sample_key", "harness_profile", "effect_domain", "predicate"),
                                     ("sample_key", "harness_profile", "effect_domain", "predicate")),
    "runtime_admission_attempts": (("attempt_key", "source_keys_json", "result_status", "publishing_trace_id", "publishing_trace_hash", "result_payload_hash"), ("attempt_key",)),
    "candidate_route_exposures": (("independent_task_key", "route_key", "source_trace_id", "source_trace_hash", "session_ids_json"),
                                  ("independent_task_key", "route_key")),
    "execution_attribution_index": (("target_ref", "source_execution_key", "evidence_class", "outcome", "certificate_hash", "event_id"),
                                    ("target_ref", "source_execution_key", "evidence_class", "outcome")),
    "runtime_support_observations": (("execution_key", "observation_version", "independent_task_key", "program_equivalence_id",
        "implementation_equivalence_id", "contract_signature", "trace_id", "draft_id", "harness_profile", "payload_path",
        "payload_hash", "source_trace_hash", "manifest_ordinal", "attempt_ordinal", "event_ordinal", "outcome"), ("execution_key",)),
    "learning_source_index": (("sample_key", "independent_task_key", "source_trace_id", "source_trace_hash", "canonical_snapshot_hash",
        "source_occurrence_id", "contract_identity_key", "feature_payload_json", "capsule_path", "capsule_hash", "committed_sequence"), ("sample_key",)),
    "generalization_attempts": (("attempt_key", "source_pair_key", "protocol_version", "group_id", "source_snapshot_hashes_json",
        "result_status", "result_refs_json", "publishing_trace_id", "result_payload_hash"), ("attempt_key",)),
}


def metadata_for(config):
    return dict(METADATA) if config.get("repair_revision") == "R10.3" else {}


def validate_config(config):
    errors = []
    if config.get("repair_revision") != "R10.3":
        errors.append("R10.3 config requires repair_revision=R10.3")
    if config.get("r103_interventions") or config.get("r103_learning_intervention", "Full") != "Full":
        errors.append("diagnostic interventions cannot be labeled formal Full")
    for key, value in METADATA.items():
        if str(config.get(key, "")) != value:
            errors.append(f"R10.3 config requires {key}={value}")
    bounds = {"exact_search_states": 100000, "near_candidates": 4, "generalization_groups": 1,
              "sources_per_group": 2, "sidecar_builder_calls": 1}
    if config.get("r103") != bounds:
        errors.append("R10.3 algorithm bounds differ from the frozen design")
    if config.get("lifecycle", {}).get("candidate_exploration_quota", 0.15) != 0.15:
        errors.append("R10.3 Candidate exploration quota must remain 0.15")
    llm, runtime = config.get("llm", {}), config.get("runtime", {})
    if llm.get("runtime", {}).get("max_total_tokens_per_task") != 600000 or runtime.get("global_action_budget") != 100:
        errors.append("R10.3 online budget must remain 600000 tokens / 100 actions")
    if llm.get("extractor", {}).get("max_total_tokens_per_task") != 262144:
        errors.append("R10.3 shared offline budget must remain 262144")
    if any(llm.get(stage, {}).get("reasoning_effort") != "high" for stage in ("planner", "runtime", "extractor", "tool_builder", "evolution_repair")):
        errors.append("R10.3 all reasoning efforts must remain high")
    return errors


def validate(connection):
    actual = dict(connection.execute("SELECT key,value FROM metadata"))
    if any(actual.get(key) != value for key, value in METADATA.items()):
        raise RuntimeError("R10.3 protocol mismatch: use a fresh bank; no historical write migration")
    for table, (columns, pk) in SHAPES.items():
        rows = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
        if tuple(row[1] for row in rows) != columns or tuple(row[1] for row in sorted(rows, key=lambda r: r[5]) if row[5]) != pk:
            raise RuntimeError(f"R10.3 schema mismatch: {table}")
    indexes = {row[1] for row in connection.execute('PRAGMA index_list("runtime_support_observations")')}
    expected = {"runtime_support_source_order": ("manifest_ordinal", "attempt_ordinal", "event_ordinal"),
                "runtime_support_contract": ("contract_signature", "harness_profile"),
                "runtime_support_program_tasks": ("program_equivalence_id", "independent_task_key")}
    for name, columns in expected.items():
        if name not in indexes or tuple(row[2] for row in connection.execute(f'PRAGMA index_info("{name}")')) != columns:
            raise RuntimeError(f"R10.3 index mismatch: {name}")
    for table, name, columns in (
        ("learning_source_effect_index", "learning_effect_recall", ("harness_profile", "effect_domain", "predicate")),
        ("artifact_identity_index", "identity_bucket", ("identity_version", "artifact_kind", "bucket_key")),
        ("artifact_identity_index", "identity_equivalence", ("identity_version", "equivalence_id")),
    ):
        names = {r[1] for r in connection.execute(f'PRAGMA index_list("{table}")')}
        if name not in names or tuple(r[2] for r in connection.execute(f'PRAGMA index_info("{name}")')) != columns:
            raise RuntimeError(f"R10.3 index mismatch: {name}")
    for table, target, source_col, target_col in (
        ("learning_source_effect_index", "learning_source_index", "sample_key", "sample_key"),
        ("artifact_identity_index", "artifact_index", "artifact_ref", "artifact_ref"),
        ("execution_attribution_index", "evidence_events", "event_id", "event_id"),
    ):
        refs = {(r[2], r[3], r[4]) for r in connection.execute(f'PRAGMA foreign_key_list("{table}")')}
        if (target, source_col, target_col) not in refs:
            raise RuntimeError(f"R10.3 foreign key mismatch: {table}")
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise RuntimeError("R10.3 foreign key integrity failure")


def add_digest_specs(connection, specs):
    """Shared by production, checkpoints and deterministic fake digest readers."""
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table, (_columns, pk) in SHAPES.items():
        if table in tables:
            if table == "runtime_support_observations":
                names = {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}
                pk = ("execution_key",) if "execution_key" in names else ("observation_id",)
            specs[table] = ("*", ",".join(pk))
    if "artifact_identity_index" in tables:
        specs["artifact_identity_index"] = ("*", "artifact_ref,identity_version")
    if "release_deployments" in tables:
        specs["release_deployments"] = ("*", "artifact_ref")
