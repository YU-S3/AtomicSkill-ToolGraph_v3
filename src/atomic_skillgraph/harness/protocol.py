"""Policy-facing and validator-only benchmark boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..core.contracts import TaskContract
from ..core.results import PrimitiveToolStep, ValidationResult


@dataclass
class HarnessTask:
    task_id: str
    goal: str
    benchmark: str
    task_type: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HarnessActionSpec:
    action_id: str
    revision: int
    action_type: str
    arguments: dict[str, Any]
    display_text: str
    raw_action: Any
    metadata: dict[str, Any]


@dataclass
class HarnessActionResult:
    accepted: bool
    observation: str
    done: bool
    won: bool
    new_revision: int
    catalog: list[HarnessActionSpec]
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ValidatorChannel(Protocol):
    validation_strength: str

    def snapshot(self) -> dict[str, Any]: ...
    def validate_atomic_effect(self, request: dict[str, Any]) -> ValidationResult: ...
    def validate_task_contract(self, contract: TaskContract) -> ValidationResult: ...


@runtime_checkable
class HarnessAdapter(Protocol):
    profile_name: str

    def reset(self, task: HarnessTask) -> HarnessActionResult: ...
    def action_catalog(self) -> list[HarnessActionSpec]: ...
    def execute_action(self, action_id: str, revision: int) -> HarnessActionResult: ...
    def task_contract(self, task: HarnessTask) -> TaskContract: ...
    def validator_channel(self) -> ValidatorChannel: ...
    def compile_primitive(self, primitive: PrimitiveToolStep, bindings: dict[str, Any]) -> Any: ...
    def execute_primitive(self, primitive: PrimitiveToolStep, bindings: dict[str, Any]) -> HarnessActionResult: ...
    def supports_constraint(self, kind: str, verifier_id: str = "") -> bool: ...
