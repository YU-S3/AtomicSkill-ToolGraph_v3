"""Shared action and stage-specific LLM budgets; fallback never resets usage."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.errors import BudgetExhausted, FailureLayer


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
