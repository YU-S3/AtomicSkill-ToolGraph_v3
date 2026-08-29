"""TaskContract and official benchmark signals remain independent results."""

from __future__ import annotations

from typing import Any

from ..core.contracts import TaskContract
from ..core.results import ValidationResult


class TaskValidator:
    def validate(self, contract: TaskContract, validator_channel: Any) -> ValidationResult:
        return validator_channel.validate_task_contract(contract)

    def terminal(self, contract: TaskContract, validator_channel: Any, benchmark_won: bool) -> ValidationResult:
        contract_result = self.validate(contract, validator_channel)
        checks = {"benchmark_won": bool(benchmark_won), "task_contract": contract_result.passed}
        if all(checks.values()):
            return ValidationResult("task_terminal", True, checks, witness_refs=contract_result.witness_refs)
        code = "benchmark_goal_contract_mismatch" if benchmark_won else "task_contract_mismatch"
        return ValidationResult("task_terminal", False, checks, [code], ["benchmark and contract must both pass"])
