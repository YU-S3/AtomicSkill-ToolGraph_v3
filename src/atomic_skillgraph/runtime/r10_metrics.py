"""R10 telemetry derived from immutable sessions and execution records."""
from ..traces.canonical import canonical_action_indices, canonical_metadata_items

COUNTERS = (
    "stored_composite_task_count", "atomic_composition_task_count",
    "runtime_graph_bootstrap_step_count", "runtime_graph_bootstrap_tokens",
    "runtime_step_count", "runtime_preparation_step_count", "runtime_seeded_step_count",
    "runtime_step_prompt_tokens", "runtime_step_reasoning_tokens",
    "composite_auto_gate_attempt_count", "composite_auto_node_success_count",
    "post_bootstrap_llm_free_node_count", "llm_breakpoint_count", "llm_free_environment_action_count",
    "support_agent_selected_count", "runtime_automation_request_count", "runtime_automation_interface_load_count",
    "runtime_automation_proposal_count", "runtime_tool_trial_r1_pass_count",
    "runtime_support_observation_count", "runtime_support_promotion_attempt_count",
    "runtime_support_promotion_success_count", "runtime_support_reuse_count",
    "runtime_step_checkpoint_count", "runtime_rollback_count", "runtime_rollback_replay_action_count",
    "runtime_checkpoint_restore_failure_count", "rolled_back_environment_action_count",
    "canonical_environment_action_count", "p0_complete_eligible_candidate_count",
    "p0_terminal_empirical_excluded_before_ranking_count", "p0_bootstrap_candidate_count",
    "p0_bootstrap_invalid_winner_count",
)


def finalize(trace, config):
    from .r101_metrics import finalize as finalize_r101
    finalize_r101(trace)
    values = trace.metadata.setdefault("r10_metrics", {})
    for name in COUNTERS:
        values.setdefault(name, 0)
    aliases = {"runtime_graph_bootstrap_step_count": "graph_bootstrap_agent_step_count",
               "runtime_preparation_step_count": "runtime_step_preparation_count",
               "runtime_seeded_step_count": "runtime_step_seeded_count",
               "composite_auto_node_success_count": "composite_auto_node_count",
               "llm_breakpoint_count": "composite_breakpoint_count"}
    for name, source in aliases.items():
        values[name] = values.get(source, 0)
    source = trace.runtime_plan.get("source")
    values["stored_composite_task_count"] = int(source == "stored_composite")
    values["atomic_composition_task_count"] = int(source == "atomic_composition")
    values.update(trace.planner_audit.get("p0_metrics", {}))
    values["canonical_environment_action_count"] = len(canonical_action_indices(trace))
    steps = trace.metadata.get("runtime_steps", [])
    bootstrap_occurrences = {item["occurrence_id"] for item in steps if item["bootstrap"]}
    assisted_occurrences = {item["occurrence_id"] for item in steps}
    automatic_ids = {item["attempt_id"] for item in canonical_metadata_items(trace, "graph_entry_invocations")}
    last_invocations = {item.occurrence_id: item for item in trace.implementation_invocations}
    bootstrap_seen = False
    unassisted, terminal_auto = 0, 0
    for node in trace.node_records:
        status = getattr(node.status, "value", node.status)
        result = node.seeded_result or node.direct_result or {}
        if bootstrap_seen and node.occurrence_id not in assisted_occurrences and result.get("atomic_effect_passed"):
            unassisted += 1
        invocation = last_invocations.get(node.occurrence_id)
        if status == "direct_terminal_effect_success" and invocation and invocation.attempt_id in automatic_ids:
            terminal_auto += 1
        if node.occurrence_id in bootstrap_occurrences:
            bootstrap_seen = True
    values["post_bootstrap_llm_free_node_count"] = unassisted
    values["composite_auto_node_success_count"] = values.get("composite_auto_node_count", 0) + terminal_auto
    sessions = {item["session_id"] for item in steps}
    bootstrap = {item["session_id"] for item in steps if item["bootstrap"]}
    usages = [item for item in trace.llm_usage if item.get("session_id") in sessions]
    def tokens(item, key):
        return int(item.get("usage", item).get(key) or 0)
    values["runtime_step_prompt_tokens"] = sum(tokens(item, "prompt_tokens") for item in usages)
    values["runtime_step_reasoning_tokens"] = sum(tokens(item, "reasoning_tokens") for item in usages)
    values["runtime_graph_bootstrap_tokens"] = sum(tokens(item, "total_tokens") for item in usages if item.get("session_id") in bootstrap)
    values["runtime_step_max_semantic_turns"] = max([item["accepted_semantic_turn_count"] for item in steps] or [0])
    limits = config.get("llm", {}).get("runtime", {})
    values.update(configured_max_total_tokens_per_node=limits.get("max_total_tokens_per_node"),
                  configured_max_total_tokens_per_task=limits.get("max_total_tokens_per_task"),
                  actual_node_budget_scope="shared_occurrence",
                  actual_task_budget_scope="shared_task_runtime")
    values["runtime_automation_proposal_count"] = trace.metadata.get("v32_metrics", {}).get("runtime_automation_atomic_proposal_count", 0)
    values["runtime_tool_trial_r1_pass_count"] = trace.metadata.get("v32_metrics", {}).get("runtime_tool_trial_r1_pass_count", 0)


def aggregate(rows):
    rows = list(rows)
    metrics = [row.get("r10_metrics", {}) for row in rows]
    result = {key: sum(int(item.get(key, 0) or 0) for item in metrics) for key in COUNTERS}
    result["runtime_step_max_semantic_turns"] = max([item.get("runtime_step_max_semantic_turns", 0) for item in metrics] or [0])
    solved = sum(bool(row.get("benchmark_success")) for row in rows)
    result["llm_breakpoints_per_solved_task"] = result["llm_breakpoint_count"] / solved if solved else None
    for key in ("configured_max_total_tokens_per_node", "configured_max_total_tokens_per_task",
                "actual_node_budget_scope", "actual_task_budget_scope"):
        values = list(dict.fromkeys(item[key] for item in metrics if key in item))
        result[key] = values[0] if len(values) == 1 else values or None
    return result
