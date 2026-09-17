"""Task-wide action authority. Tokens are metered only by UsageLedger."""
from dataclasses import dataclass
from ..core.errors import BudgetExhausted, FailureLayer

PROTOCOL_REPAIR_LIMIT = 1
TURN_COMPLETION_OVERHEAD = 3


def required_runtime_turn_caps(*, global_action_budget: int,
                               learned_toolcall_repair_limit: int,
                               protocol_repair_limit: int = PROTOCOL_REPAIR_LIMIT) -> tuple[int, int]:
    values = (global_action_budget, learned_toolcall_repair_limit, protocol_repair_limit)
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values) or global_action_budget == 0:
        raise ValueError('positive task action budget and nonnegative protocol limits required')
    task_cap = global_action_budget + protocol_repair_limit + TURN_COMPLETION_OVERHEAD
    return task_cap + learned_toolcall_repair_limit, task_cap


def validate_runtime_turn_caps(*, global_action_budget: int, learned_toolcall_repair_limit: int,
                               max_turns_per_task: int,
                               protocol_repair_limit: int = PROTOCOL_REPAIR_LIMIT) -> tuple[int, int]:
    step_cap, task_cap = required_runtime_turn_caps(global_action_budget=global_action_budget,
        learned_toolcall_repair_limit=learned_toolcall_repair_limit, protocol_repair_limit=protocol_repair_limit)
    if isinstance(max_turns_per_task, bool) or int(max_turns_per_task) < task_cap:
        raise ValueError(f'max_turns_per_task must cover global_action_budget and protocol overhead (minimum {task_cap})')
    return max(step_cap, int(max_turns_per_task)), int(max_turns_per_task)


@dataclass
class RuntimeBudget:
    global_action_budget: int = 100
    used_global_actions: int = 0
    used_node_actions: int = 0
    current_occurrence_id: str = ''

    def begin_node(self, occurrence_id: str) -> None:
        if occurrence_id != self.current_occurrence_id:
            self.current_occurrence_id = occurrence_id
            self.used_node_actions = 0

    def end_node(self) -> None:
        self.current_occurrence_id = ''
        self.used_node_actions = 0

    def consume_action(self, count: int = 1) -> None:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError('action count must be a nonnegative integer')
        if self.used_global_actions + count > self.global_action_budget:
            raise BudgetExhausted('episode_action_budget_exhausted', 'global environment action budget exhausted',
                layer=FailureLayer.RUNTIME_AGENT)
        self.used_global_actions += count
        if self.current_occurrence_id:
            self.used_node_actions += count

    @property
    def remaining_global_actions(self) -> int:
        return max(0, self.global_action_budget - self.used_global_actions)

    def snapshot(self) -> dict:
        return {'global_action_budget': self.global_action_budget,
            'used_global_actions': self.used_global_actions, 'used_node_actions': self.used_node_actions,
            'remaining_global_actions': self.remaining_global_actions}
