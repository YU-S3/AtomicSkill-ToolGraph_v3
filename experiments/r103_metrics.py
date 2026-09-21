"""Read-only R10.3 diagnostics. Missing observations are not measured zeros."""
from collections import Counter
from copy import deepcopy
import json
from statistics import mean, median

from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.governance.projections import ArtifactStats


def trace_metrics(trace):
    trace = to_primitive(trace)
    if not isinstance(trace,dict):
        trace = to_primitive(vars(trace))
    meta = trace.get("metadata", {})
    if meta.get("repair_revision") != "R10.3" and "identity_audit" not in meta:
        return None
    audits = [a["expression_audit"] for a in meta.get("runtime_context_projection_audits", [])
              if "expression_audit" in a]
    routes = {}
    for exposure in meta.get("candidate_route_exposures", []):
        ref = exposure["implementation_ref"]
        row = routes.setdefault(ref, {"route_key":exposure["route_key"],"exposed":0,
            "candidate":exposure.get("candidate"),"invocation_records":[]})
        row["exposed"] += 1
    for invocation in trace.get("implementation_invocations", []):
        ref = invocation["implementation_ref"]
        row = routes.setdefault(ref,{"route_key":None,"exposed":None,"candidate":None,"invocation_records":[]})
        row["invocation_records"].append(deepcopy(invocation))
    builds = meta.get("evolution_tool_builds")
    avoided = None if builds is None else sum(
        b.get("outcome") == "exact_reuse" and b.get("builder_entered") is False for b in builds)
    return {"protocol":"r103.report.v1", "identity":deepcopy(meta.get("identity_audit")),
        "exact_reuse_builder_avoided":avoided,"candidate_routes":routes,
        "generalization":deepcopy(meta.get("generalization")),
        "near_recall":deepcopy(meta.get("generalization_recall_summaries")),
        "attribution_prepared_count":meta.get("execution_attribution_prepared_count"),
        "runtime_admission_attempts":deepcopy(meta.get("runtime_admission_attempts")),
        "expression":{"requests_audited":len(audits),
            "schema_dedup_count":sum(sum("input_schema" in p for p in a.get("removed_paths",[])) for a in audits),
            "required_surface_equal":all(a.get("required_surface_equal") is True for a in audits) if audits else None,
            "guideline_hash_unchanged":all(a.get("guidance_input_hash") == a.get("guidance_output_hash") for a in audits) if audits else None,
            "fallback_reasons":dict(Counter(r["reason"] for a in audits for r in a.get("original_fallback_reasons",[]))),
            "audit_records":deepcopy(audits)},
        "counting_notes":"Routes retain raw preflight/start/completion records; exposure is not selection or success. Missing fields remain null."}


def aggregate(rows):
    diagnostics = [r["r103_diagnostics"] for r in rows if r.get("r103_diagnostics") is not None]
    if not diagnostics:
        return None
    layers = {}
    for row in diagnostics:
        for layer, counts in (row.get("identity") or {}).get("identity_exact_different_unknown_by_layer",{}).items():
            target = layers.setdefault(layer,Counter())
            target.update(counts)
    known = [r["exact_reuse_builder_avoided"] for r in diagnostics if r["exact_reuse_builder_avoided"] is not None]
    return {"protocol":"r103.report.v1","traces":len(diagnostics),"identity_calls_by_layer":layers,
        "exact_reuse_builder_avoided_observed":sum(known) if known else None,
        "exact_reuse_traces_without_measurement":len(diagnostics)-len(known),
        "expression_requests_audited":sum(r["expression"]["requests_audited"] for r in diagnostics),
        "expression_schema_dedup_count":sum(r["expression"]["schema_dedup_count"] for r in diagnostics),
        "generalization_status_counts":dict(Counter(r["generalization"]["status"] for r in diagnostics
             if r.get("generalization") and "status" in r["generalization"])),
        "counting_notes":"Matcher invocations include verification. Distinct pairs and route/credit details are per-Trace; do not sum independent task counts across asset layers."}


def bank_metrics(database):
    """Read existing projections; do not review lifecycle or rebuild indexes."""
    return {row["artifact_ref"]: {"artifact_kind":json.loads(row["projection_json"])["artifact_kind"],
            "execution_support":json.loads(row["projection_json"]).get("execution_support"),
            "registered_direct_success_count":ArtifactStats.from_dict(json.loads(row["projection_json"])).direct_success_count}
            for row in database.rows("SELECT artifact_ref,projection_json FROM lifecycle_projection")}


def paired_results(old_rows, new_rows):
    def index(rows):
        result = {r["task_id"]:r for r in rows}
        if len(result) != len(rows):
            raise ValueError("paired comparison contains duplicate task identities")
        return result
    old, new = index(old_rows), index(new_rows)
    if old.keys() != new.keys() or not old:
        raise ValueError("paired comparison requires the same nonempty task set")
    keys = list(old)
    retained = [k for k in keys if old[k]["official_won"] and new[k]["official_won"]]
    return {"tasks":len(keys),
        "newly_solved":[k for k in keys if not old[k]["official_won"] and new[k]["official_won"]],
        "newly_failed":[k for k in keys if old[k]["official_won"] and not new[k]["official_won"]],
        "old_mean_recorded_tokens_all_tasks":mean(old[k]["total_tokens"] for k in keys),
        "new_mean_recorded_tokens_all_tasks":mean(new[k]["total_tokens"] for k in keys),
        "paired_recorded_token_delta_mean":mean(new[k]["total_tokens"]-old[k]["total_tokens"] for k in keys),
        "paired_recorded_token_delta_median":median(new[k]["total_tokens"]-old[k]["total_tokens"] for k in keys),
        "retained_success_tasks":retained,
        "retained_success_recorded_token_delta_mean":mean(new[k]["total_tokens"]-old[k]["total_tokens"] for k in retained) if retained else None,
        "interaction":None,"claim":"Conditional paired diagnostic only; not a formal effect-size claim."}
