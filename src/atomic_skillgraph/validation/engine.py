"""Single facade preserving the six distinct validation layers."""

from __future__ import annotations

from .atomic_validator import AtomicValidator
from .composite_validator import CompositeValidator
from .failure_localizer import FailureLocalizer
from .task_validator import TaskValidator
from .tool_validator import ToolValidator


class ValidationEngine:
    def __init__(self) -> None:
        self.tool = ToolValidator()
        self.atomic = AtomicValidator()
        self.composite = CompositeValidator()
        self.task = TaskValidator()
        self.failure_localizer = FailureLocalizer()
