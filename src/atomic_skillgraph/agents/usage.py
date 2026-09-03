"""Auditable per-turn LLM usage buckets and session budget enforcement."""

from __future__ import annotations

import copy
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from ..core.errors import BudgetExhausted, FailureLayer
from .protocol import AgentTurn


class UsageBucket(str, Enum):
    PLANNER_P1 = "planner_p1"
    PLANNER_P1_REPAIR = "planner_p1_repair"
    PLANNER_P2 = "planner_p2"
    PLANNER_P2_REPAIR = "planner_p2_repair"
    COLD_START_C1 = "cold_start_c1"
    COLD_START_C1_REPAIR = "cold_start_c1_repair"
    RUNTIME_PREPARATION = "runtime_preparation"
    RUNTIME_SEEDED = "runtime_seeded"
    RUNTIME_DYNAMIC = "runtime_dynamic"
    RUNTIME_PROVISIONAL_SEEDED = "runtime_provisional_seeded"
    RUNTIME_DYNAMIC_COLD_START_CONTINUATION = "runtime_dynamic_cold_start_continuation"
    EXTRACTOR_E1 = "extractor_e1"
    EXTRACTOR_E2 = "extractor_e2"
    TOOL_BUILDER_RUNTIME = "tool_builder_runtime"
    TOOL_BUILDER_EVOLUTION = "tool_builder_evolution"
    FAILURE_EXTRACTOR_F1 = "failure_extractor_f1"
    FAILURE_EXTRACTOR_F2 = "failure_extractor_f2"
    EVOLUTION_REPAIR = "evolution_repair"
    UNATTRIBUTED = "unattributed"


REAL_USAGE_BUCKETS = tuple(bucket for bucket in UsageBucket if bucket is not UsageBucket.UNATTRIBUTED)


@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int | None = 0
    call_count: int = 0
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        for name in ("prompt_tokens", "completion_tokens", "total_tokens", "call_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.reasoning_tokens is not None:
            if (
                isinstance(self.reasoning_tokens, bool)
                or not isinstance(self.reasoning_tokens, int)
                or self.reasoning_tokens < 0
            ):
                raise ValueError("reasoning_tokens must be a non-negative integer or None")
        if not isinstance(self.latency_ms, (int, float)) or self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")

    @classmethod
    def from_turn(cls, turn: AgentTurn) -> "LLMUsage":
        return cls(
            prompt_tokens=turn.prompt_tokens,
            completion_tokens=turn.completion_tokens,
            total_tokens=turn.total_tokens,
            reasoning_tokens=turn.reasoning_tokens,
            call_count=1,
            latency_ms=float(turn.latency_ms),
        )

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "call_count": self.call_count,
            "latency_ms": round(float(self.latency_ms), 3),
        }


def sum_usage(values: Iterable[LLMUsage]) -> LLMUsage:
    items = list(values)
    reasoning_available = all(item.reasoning_tokens is not None for item in items)
    return LLMUsage(
        prompt_tokens=sum(item.prompt_tokens for item in items),
        completion_tokens=sum(item.completion_tokens for item in items),
        total_tokens=sum(item.total_tokens for item in items),
        reasoning_tokens=(
            sum(int(item.reasoning_tokens or 0) for item in items)
            if reasoning_available
            else None
        ),
        call_count=sum(item.call_count for item in items),
        latency_ms=sum(item.latency_ms for item in items),
    )


@dataclass(frozen=True)
class UsageEvent:
    event_id: str
    session_id: str
    turn_index: int
    bucket: UsageBucket
    usage: LLMUsage
    provider: str = ""
    model: str = ""
    provider_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bucket", UsageBucket(self.bucket))
        if not self.event_id or not self.session_id:
            raise ValueError("usage event requires event_id and session_id")
        if self.turn_index < 0:
            raise ValueError("usage event turn_index must be non-negative")
        if self.usage.call_count != 1:
            raise ValueError("each usage event must describe exactly one provider call")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "bucket": self.bucket.value,
            **self.usage.to_dict(),
            "provider": self.provider,
            "model": self.model,
            "provider_metadata": dict(self.provider_metadata),
        }


class UsageLedger:
    """Append-only in-memory ledger; trace persistence consumes ``snapshot``."""

    def __init__(self) -> None:
        self._events: list[UsageEvent] = []
        self._event_ids: set[str] = set()
        self._lock = threading.RLock()

    @property
    def events(self) -> tuple[UsageEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def append(self, event: UsageEvent) -> UsageEvent:
        with self._lock:
            if event.event_id in self._event_ids:
                raise ValueError(f"duplicate usage event_id: {event.event_id}")
            self._events.append(event)
            self._event_ids.add(event.event_id)
        return event

    def record_turn(
        self,
        *,
        session_id: str,
        turn_index: int,
        bucket: UsageBucket | str,
        turn: AgentTurn,
        event_id: str | None = None,
    ) -> UsageEvent:
        metadata = _metering_metadata(turn.provider_metadata)
        event = UsageEvent(
            event_id=event_id or f"usage_{uuid.uuid4().hex}",
            session_id=session_id,
            turn_index=turn_index,
            bucket=UsageBucket(bucket),
            usage=LLMUsage.from_turn(turn),
            provider=str(metadata.get("provider", "")),
            model=str(metadata.get("model", "")),
            provider_metadata=metadata,
        )
        return self.append(event)

    def total(self, bucket: UsageBucket | str | None = None) -> LLMUsage:
        with self._lock:
            events = list(self._events)
        if bucket is not None:
            normalized = UsageBucket(bucket)
            events = [event for event in events if event.bucket is normalized]
        return sum_usage(event.usage for event in events)

    def by_bucket(self) -> dict[str, dict[str, int | float | None]]:
        return {
            bucket.value: self.total(bucket).to_dict()
            for bucket in UsageBucket
            if any(event.bucket is bucket for event in self.events)
        }

    def reconcile(self, episode_total_tokens: int | None = None) -> dict[str, int]:
        """Report the design invariant: real buckets + unattributed = episode."""
        real_total = sum(self.total(bucket).total_tokens for bucket in REAL_USAGE_BUCKETS)
        unattributed = self.total(UsageBucket.UNATTRIBUTED).total_tokens
        ledger_total = real_total + unattributed
        expected = ledger_total if episode_total_tokens is None else int(episode_total_tokens)
        return {
            "real_bucket_total_tokens": real_total,
            "unattributed_total_tokens": unattributed,
            "episode_total_tokens": expected,
            "token_mismatch": ledger_total - expected,
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "events": [event.to_dict() for event in self.events],
            "by_bucket": self.by_bucket(),
            "episode_total": self.total().to_dict(),
            "reconciliation": self.reconcile(),
        }


@dataclass(frozen=True)
class AgentBudget:
    max_turns: int
    max_total_tokens: int
    exhaustion_code: str

    def __post_init__(self) -> None:
        if self.max_turns < 0 or self.max_total_tokens < 0:
            raise ValueError("agent budgets must be non-negative")
        if not self.exhaustion_code:
            raise ValueError("agent budget requires an exhaustion_code")


class BudgetTracker:
    """Per-session turn/token budget; provider-reported totals are authoritative."""

    def __init__(self, budget: AgentBudget) -> None:
        self.budget = budget
        self.used_turns = 0
        self.used_total_tokens = 0

    def check_before_call(self) -> None:
        if self.used_turns >= self.budget.max_turns:
            self._raise("agent turn budget exhausted")
        if self.used_total_tokens >= self.budget.max_total_tokens:
            self._raise("agent token budget exhausted")

    def consume(self, usage: LLMUsage) -> None:
        self.used_turns += usage.call_count
        self.used_total_tokens += usage.total_tokens
        if self.used_turns > self.budget.max_turns:
            self._raise("agent turn budget exceeded by provider call")
        if self.used_total_tokens > self.budget.max_total_tokens:
            self._raise("agent token budget exceeded by provider call")

    @property
    def remaining_turns(self) -> int:
        return max(0, self.budget.max_turns - self.used_turns)

    @property
    def remaining_total_tokens(self) -> int:
        return max(0, self.budget.max_total_tokens - self.used_total_tokens)

    def snapshot(self) -> dict[str, Any]:
        return {
            "max_turns": self.budget.max_turns,
            "used_turns": self.used_turns,
            "remaining_turns": self.remaining_turns,
            "max_total_tokens": self.budget.max_total_tokens,
            "used_total_tokens": self.used_total_tokens,
            "remaining_total_tokens": self.remaining_total_tokens,
            "exhaustion_code": self.budget.exhaustion_code,
        }

    def _raise(self, message: str) -> None:
        raise BudgetExhausted(
            self.budget.exhaustion_code,
            message,
            # A configured experiment budget is part of Agent control flow,
            # not an API/network/process outage.  Callers retain the specific
            # exhaustion code and must not trigger infrastructure rollback.
            layer=FailureLayer.RUNTIME_AGENT,
        )


def _metering_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep provider audit metadata, but never retain reasoning text/content."""
    allowed_reasoning = {
        "reasoning_tokens_status",
        "reasoning_tokens_source",
        "reasoning_tokens_in_completion",
    }
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        normalized = str(key).lower()
        if "reasoning" in normalized and normalized not in allowed_reasoning:
            continue
        clean[str(key)] = _scrub_nested_reasoning(value)
    return clean


def _scrub_nested_reasoning(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _scrub_nested_reasoning(item)
            for key, item in value.items()
            if "reasoning" not in str(key).lower()
        }
    if isinstance(value, list):
        return [_scrub_nested_reasoning(item) for item in value]
    return copy.deepcopy(value)


__all__ = [
    "AgentBudget",
    "BudgetTracker",
    "LLMUsage",
    "REAL_USAGE_BUCKETS",
    "UsageBucket",
    "UsageEvent",
    "UsageLedger",
    "sum_usage",
]
