"""Deterministic failure attribution; it never parses free-form log strings."""

from __future__ import annotations

import uuid
from typing import Any

from ..core.errors import FailureEnvelope, FailureLayer


_LAYER_BY_CODE = {
    "runtime_agent_schema_error": FailureLayer.RUNTIME_AGENT,
    "runtime_agent_multiple_tool_calls": FailureLayer.RUNTIME_AGENT,
    "runtime_task_token_budget_exhausted": FailureLayer.RUNTIME_AGENT,
    "runtime_node_token_budget_exhausted": FailureLayer.RUNTIME_AGENT,
    "runtime_plan_conflict": FailureLayer.COMPOSITE,
    "runtime_binding_unresolved": FailureLayer.RUNTIME_BINDING,
    "runtime_binding_not_concrete": FailureLayer.RUNTIME_BINDING,
    "runtime_relation_not_grounded": FailureLayer.RUNTIME_BINDING,
    "stale_grounding_evidence": FailureLayer.RUNTIME_BINDING,
    "implementation_compile_rejected": FailureLayer.IMPLEMENTATION,
    "implementation_mapping_error": FailureLayer.IMPLEMENTATION,
    "implementation_constraint_error": FailureLayer.IMPLEMENTATION,
    "implementation_compatibility_error": FailureLayer.IMPLEMENTATION,
    "tool_primitive_rejected": FailureLayer.TOOL,
    "tool_execution_error": FailureLayer.TOOL,
    "atomic_effect_violation": FailureLayer.ATOMIC,
    "data_flow_error": FailureLayer.DATA_FLOW,
    "composite_self_sufficiency_failure": FailureLayer.COMPOSITE,
    "task_contract_mismatch": FailureLayer.TASK_CONTRACT,
    "benchmark_goal_contract_mismatch": FailureLayer.TASK_CONTRACT,
    "benchmark_failure": FailureLayer.BENCHMARK,
    "llm_error": FailureLayer.INFRASTRUCTURE,
    "infrastructure_failure": FailureLayer.INFRASTRUCTURE,
}


class FailureLocalizer:
    def localize(
        self, *, code: str, task_id: str, trace_id: str, occurrence_id: str,
        attempt_id: str, started: bool, artifact_refs: list[str] | None = None,
        evidence_refs: list[str] | None = None, message: str = "", recoverable: bool = False,
        layer: FailureLayer | None = None,
    ) -> FailureEnvelope:
        return FailureEnvelope(
            failure_id=f"failure_{uuid.uuid4().hex}", layer=layer or _LAYER_BY_CODE.get(code, FailureLayer.INFRASTRUCTURE),
            code=code, task_id=task_id, trace_id=trace_id, occurrence_id=occurrence_id,
            attempt_id=attempt_id, started=started, artifact_refs=list(artifact_refs or []),
            evidence_refs=list(evidence_refs or []), recoverable=recoverable, message=message,
        )
