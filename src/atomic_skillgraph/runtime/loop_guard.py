"""Small, domain-independent guard against unproductive Agent action loops."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    )


def action_signature(action_type: str, arguments: Mapping[str, Any]) -> str:
    """Return the specified ``action_type + canonical arguments`` signature."""

    return f"{str(action_type).strip().upper()}:{_canonical(dict(arguments))}"


def catalog_state_signature(observation: str, catalog: Iterable[Any]) -> str:
    """Hash policy-visible state while excluding revision-scoped opaque ids."""

    actions: list[dict[str, Any]] = []
    for item in catalog:
        if isinstance(item, Mapping):
            action_type = item.get("action_type", "")
            arguments = item.get("arguments", {})
        else:
            action_type = getattr(item, "action_type", "")
            arguments = getattr(item, "arguments", {})
        actions.append({
            "action_type": str(action_type),
            "arguments": dict(arguments or {}),
        })
    actions.sort(key=_canonical)
    raw = _canonical({"observation": str(observation), "catalog": actions})
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ActionLoopDecision:
    blocked: bool
    reason: str = ""
    action_signature: str = ""
    state_signature: str = ""
    consecutive_loop_blocks: int = 0

    @property
    def fallback_required(self) -> bool:
        return self.consecutive_loop_blocks >= 2

    def tool_result(self) -> dict[str, Any]:
        return {
            "accepted": False,
            "error": "loop_blocked",
            "loop_blocked": True,
            "reason": self.reason,
            "action_signature": self.action_signature,
            "state_signature": self.state_signature,
            "consecutive_loop_blocks": self.consecutive_loop_blocks,
            "fallback_required": self.fallback_required,
            "environment_called": False,
            "action_budget_consumed": False,
        }


class ActionLoopGuard:
    """Track one Agent session; blocked proposals never reach the environment."""

    def __init__(self) -> None:
        self._history: list[tuple[str, str]] = []
        self._locked: set[tuple[str, str]] = set()
        self._consecutive_blocks = 0

    @property
    def consecutive_loop_blocks(self) -> int:
        return self._consecutive_blocks

    def inspect(
        self,
        *,
        action_type: str,
        arguments: Mapping[str, Any],
        observation: str,
        catalog: Iterable[Any],
    ) -> ActionLoopDecision:
        signature = action_signature(action_type, arguments)
        state = catalog_state_signature(observation, catalog)
        event = (signature, state)
        self._history.append(event)

        reason = ""
        if event in self._locked:
            reason = "repeated_blocked_action_in_unchanged_state"
        elif len(self._history) >= 3 and all(item == event for item in self._history[-3:]):
            reason = "same_action_same_state_three_times"
        elif len(self._history) >= 6:
            recent_actions = [item[0] for item in self._history[-6:]]
            first, second = recent_actions[0], recent_actions[1]
            if first != second and recent_actions == [first, second] * 3:
                reason = "two_action_cycle_three_rounds"

        if reason:
            self._locked.add(event)
            self._consecutive_blocks += 1
            return ActionLoopDecision(
                True, reason, signature, state, self._consecutive_blocks,
            )

        self._consecutive_blocks = 0
        return ActionLoopDecision(False, "", signature, state, 0)
