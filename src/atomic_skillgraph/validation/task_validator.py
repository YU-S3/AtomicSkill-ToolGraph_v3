"""TaskContract and official benchmark signals remain independent results."""

from __future__ import annotations

from typing import Any

from ..core.contracts import TaskContract
from ..core.results import ValidationResult


class TaskValidator:
    def validate(self, contract: TaskContract, validator_channel: Any) -> ValidationResult:
        return validator_channel.validate_task_contract(contract)

    def terminal(self, contract: TaskContract, validator_channel: Any, benchmark_won: bool) -> ValidationResult:
        """Benchmark ``won`` is the sole task-success authority in v3.2.

        TaskContract agreement remains a diagnostic field and never vetoes
        task success.
        """

        contract_result = self.validate(contract, validator_channel)
        checks = {"benchmark_won": bool(benchmark_won), "task_contract": contract_result.passed}
        if benchmark_won:
            return ValidationResult(
                "task_terminal", True, checks, [],
                [] if checks["task_contract"] else ["benchmark_goal_contract_mismatch"],
                witness_refs=contract_result.witness_refs,
            )
        return ValidationResult("task_terminal", False, checks, ["benchmark_failure"], ["benchmark not won"])
