"""Graph self-sufficiency is separate from benchmark success."""

from __future__ import annotations

from typing import Any

from ..core.results import NodeExecutionStatus, RuntimeLinearPlan, ValidationResult


class CompositeValidator:
    def validate_runtime(
        self, plan: RuntimeLinearPlan, node_records: list[Any],
        validated_outputs: dict[str, dict[str, Any]], *, task_rescue_required: bool,
        task_contract_result: ValidationResult,
    ) -> ValidationResult:
        successful = {
            NodeExecutionStatus.ALREADY_SATISFIED, NodeExecutionStatus.DIRECT_AUTONOMOUS_SUCCESS,
            NodeExecutionStatus.DIRECT_AGENT_PREPARED_SUCCESS,
            NodeExecutionStatus.AGENT_COMPLETED_BEFORE_INVOCATION, NodeExecutionStatus.SEEDED_SUCCESS,
            NodeExecutionStatus.SKIPPED_GOAL_TERMINAL,
        }
        checks = {
            "all_occurrences_valid": len(node_records) == len(plan.occurrences) and all(NodeExecutionStatus(item.status) in successful for item in node_records),
            "all_dataflow_realized": all(
                edge.source_role in validated_outputs.get(plan.occurrence(edge.source_step).occurrence_id, {})
                for edge in plan.data_edges
            ),
            "no_task_rescue_required": not task_rescue_required,
            "task_contract_covered_at_graph_boundary": task_contract_result.passed,
        }
        passed = all(checks.values())
        return ValidationResult(
            "composite", passed, checks,
            [] if passed else ["composite_self_sufficiency_failure"],
            [] if passed else ["runtime graph was not self-sufficient"],
        )
