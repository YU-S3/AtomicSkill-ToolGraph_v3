"""Stable failure codes and structured attribution envelopes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FailureLayer(str, Enum):
    INFRASTRUCTURE = "infrastructure"
    PLANNER_REQUIREMENT = "planner_requirement"
    RETRIEVAL = "retrieval"
    PLANNER_GRAPH = "planner_graph"
    RUNTIME_AGENT = "runtime_agent"
    RUNTIME_BINDING = "runtime_binding"
    IMPLEMENTATION = "implementation"
    TOOL = "tool"
    ATOMIC = "atomic"
    DATA_FLOW = "data_flow"
    COMPOSITE = "composite"
    TASK_CONTRACT = "task_contract"
    BENCHMARK = "benchmark"


ERROR_CODES = frozenset(
    """planner_requirement_invalid planner_requirement_uncovered
    planner_requirement_repair_failed retrieval_miss planner_graph_invalid
    planner_graph_repair_failed runtime_agent_schema_error
    runtime_agent_multiple_tool_calls runtime_binding_unresolved
    runtime_binding_not_concrete runtime_relation_not_grounded
    stale_grounding_evidence implementation_compile_rejected
    implementation_mapping_error implementation_constraint_error
    implementation_compatibility_error implementation_invocation_failed
    tool_preflight_rejected tool_primitive_rejected tool_execution_error
    tool_output_schema_error atomic_effect_violation data_flow_error
    composite_self_sufficiency_failure task_contract_mismatch
    benchmark_goal_contract_mismatch benchmark_failure action_cycle llm_error
    infrastructure_failure planner_token_budget_exhausted
    runtime_node_token_budget_exhausted runtime_node_action_budget_exhausted
    episode_action_budget_exhausted extractor_token_budget_exhausted""".split()
    + """provider_capability_mismatch provider_auth_error provider_rate_limit
    provider_rate_limited provider_timeout provider_transport_error
    provider_invalid_request provider_invalid_response
    provider_reasoning_content_missing provider_usage_missing""".split()
)


@dataclass
class FailureEnvelope:
    failure_id: str
    layer: FailureLayer
    code: str
    task_id: str
    trace_id: str
    occurrence_id: str
    attempt_id: str
    started: bool
    artifact_refs: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    recoverable: bool = False
    message: str = ""

    def __post_init__(self) -> None:
        self.layer = FailureLayer(self.layer)


class AtomicSkillGraphError(RuntimeError):
    def __init__(self, code: str, message: str, *, layer: FailureLayer = FailureLayer.INFRASTRUCTURE):
        super().__init__(message)
        self.code = code
        self.layer = layer


class ArtifactIntegrityError(AtomicSkillGraphError):
    pass


class AgentProtocolError(AtomicSkillGraphError):
    pass


class BudgetExhausted(AtomicSkillGraphError):
    pass
