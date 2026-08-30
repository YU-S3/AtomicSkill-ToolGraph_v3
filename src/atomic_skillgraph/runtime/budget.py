"""Shared action and stage-specific LLM budgets; fallback never resets usage.

Runtime Agent turns are a protocol budget, not a second action budget.  Keep
their lower bounds next to the action budget so callers cannot accidentally
re-introduce a small, hidden turn cap (the historical ``12``-turn failure).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.errors import BudgetExhausted, FailureLayer


PROTOCOL_REPAIR_LIMIT = 1
TURN_COMPLETION_OVERHEAD = 3


def required_runtime_turn_caps(
    *,
    global_action_budget: int,
    node_action_budget: int,
    learned_toolcall_repair_limit: int,
    protocol_repair_limit: int = PROTOCOL_REPAIR_LIMIT,
) -> tuple[int, int]:
    """Return the minimum ``(node, task)`` Agent turn caps.

    A node may spend its environment actions, repair rejected learned calls,
    repair the provider protocol, and still needs a small completion margin.
    A Dynamic/Rescue task has the analogous global-action allowance.
    """

    values = {
        "global_action_budget": global_action_budget,
        "node_action_budget": node_action_budget,
        "learned_toolcall_repair_limit": learned_toolcall_repair_limit,
        "protocol_repair_limit": protocol_repair_limit,
    }
    for name, value in values.items():
        if isinstance(value, bool) or int(value) < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if int(global_action_budget) <= 0 or int(node_action_budget) <= 0:
        raise ValueError("action budgets must be positive")
    node_cap = (
        int(node_action_budget)
        + int(learned_toolcall_repair_limit)
        + int(protocol_repair_limit)
        + TURN_COMPLETION_OVERHEAD
    )
    task_cap = (
        int(global_action_budget)
        + int(protocol_repair_limit)
        + TURN_COMPLETION_OVERHEAD
    )
    return node_cap, task_cap


def validate_runtime_turn_caps(
    *,
    global_action_budget: int,
    node_action_budget: int,
    learned_toolcall_repair_limit: int,
    max_turns_per_node: int,
    max_turns_per_task: int,
    protocol_repair_limit: int = PROTOCOL_REPAIR_LIMIT,
) -> tuple[int, int]:
    """Fail closed when configured caps cannot cover the action budget."""

    required_node, required_task = required_runtime_turn_caps(
        global_action_budget=global_action_budget,
        node_action_budget=node_action_budget,
        learned_toolcall_repair_limit=learned_toolcall_repair_limit,
        protocol_repair_limit=protocol_repair_limit,
    )
    if isinstance(max_turns_per_node, bool) or int(max_turns_per_node) < required_node:
        raise ValueError(
            "max_turns_per_node must cover node_action_budget plus learned/protocol "
            f"repair and completion overhead (minimum {required_node})"
        )
    if isinstance(max_turns_per_task, bool) or int(max_turns_per_task) < required_task:
        raise ValueError(
            "max_turns_per_task must cover global_action_budget plus protocol repair "
            f"and completion overhead (minimum {required_task})"
        )
    return int(max_turns_per_node), int(max_turns_per_task)


@dataclass
class RuntimeBudget:
    global_action_budget: int = 100
    node_action_budget: int = 35
    token_limits: dict[str, int] = field(default_factory=dict)
    turn_limits: dict[str, int] = field(default_factory=dict)
    used_global_actions: int = 0
    used_node_actions: int = 0
    used_tokens: dict[str, int] = field(default_factory=dict)
    used_turns: dict[str, int] = field(default_factory=dict)
    current_occurrence_id: str = ""

    def begin_node(self, occurrence_id: str) -> None:
        if occurrence_id != self.current_occurrence_id:
            self.current_occurrence_id = occurrence_id
            self.used_node_actions = 0

    def end_node(self) -> None:
        """Leave the node quota while preserving the shared global usage."""
        self.current_occurrence_id = ""
        self.used_node_actions = 0

    def consume_action(self, count: int = 1) -> None:
        if self.used_global_actions + count > self.global_action_budget:
            raise BudgetExhausted(
                "episode_action_budget_exhausted", "global environment action budget exhausted",
                layer=FailureLayer.RUNTIME_AGENT,
            )
        if self.current_occurrence_id and self.used_node_actions + count > self.node_action_budget:
            raise BudgetExhausted(
                "runtime_node_action_budget_exhausted", "node action budget exhausted",
                layer=FailureLayer.RUNTIME_AGENT,
            )
        self.used_global_actions += count
        if self.current_occurrence_id:
            self.used_node_actions += count

    def consume_llm(self, bucket: str, total_tokens: int, *, turns: int = 1) -> None:
        tokens = self.used_tokens.get(bucket, 0) + int(total_tokens)
        used_turns = self.used_turns.get(bucket, 0) + int(turns)
        token_limit = self.token_limits.get(bucket)
        turn_limit = self.turn_limits.get(bucket)
        if token_limit is not None and tokens > token_limit:
            prefix = "planner" if bucket.startswith("planner") else "extractor" if bucket.startswith("extractor") else "runtime_node"
            raise BudgetExhausted(f"{prefix}_token_budget_exhausted", f"{bucket} token budget exhausted", layer=FailureLayer.RUNTIME_AGENT)
        if turn_limit is not None and used_turns > turn_limit:
            raise BudgetExhausted("runtime_node_token_budget_exhausted", f"{bucket} turn budget exhausted", layer=FailureLayer.RUNTIME_AGENT)
        self.used_tokens[bucket] = tokens
        self.used_turns[bucket] = used_turns

    @property
    def remaining_global_actions(self) -> int:
        return max(0, self.global_action_budget - self.used_global_actions)

    @property
    def remaining_node_actions(self) -> int:
        return max(0, self.node_action_budget - self.used_node_actions)

    def snapshot(self) -> dict:
        return {
            "global_action_budget": self.global_action_budget,
            "node_action_budget": self.node_action_budget,
            "used_global_actions": self.used_global_actions,
            "used_node_actions": self.used_node_actions,
            "node_budget_active": bool(self.current_occurrence_id),
            "remaining_global_actions": self.remaining_global_actions,
            "remaining_node_actions": self.remaining_node_actions,
            "used_tokens": dict(self.used_tokens), "used_turns": dict(self.used_turns),
        }
