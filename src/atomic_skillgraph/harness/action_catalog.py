"""Revision-scoped action-id catalog and schema compaction."""

from __future__ import annotations

from typing import Any, Callable, Iterable

from .protocol import HarnessActionSpec


Parser = Callable[[Any], tuple[str, dict[str, Any], str, dict[str, Any]]]


class HarnessActionCatalog:
    def __init__(self, parser: Parser, *, revision: int = 0) -> None:
        self.parser = parser
        self.revision = revision
        self._items: list[HarnessActionSpec] = []
        self._by_id: dict[str, HarnessActionSpec] = {}

    def replace(self, raw_actions: Iterable[Any], revision: int) -> list[HarnessActionSpec]:
        self.revision = int(revision)
        items: list[HarnessActionSpec] = []
        seen: set[tuple[str, str]] = set()
        for raw in raw_actions:
            action_type, arguments, display, metadata = self.parser(raw)
            identity = (action_type, repr(sorted(arguments.items())))
            if identity in seen:
                continue
            seen.add(identity)
            index = len(items) + 1
            # Opaque ids are unique across an episode, not merely within one
            # catalog.  This makes a stale id impossible to reinterpret after
            # a transition even when action ordering happens to be unchanged.
            action_id = f"r{self.revision:03d}_a{index:03d}"
            items.append(HarnessActionSpec(
                action_id=action_id, revision=self.revision, action_type=action_type,
                arguments=arguments, display_text=display, raw_action=raw, metadata=metadata,
            ))
        self._items = items
        self._by_id = {item.action_id: item for item in items}
        return self.items()

    def items(self) -> list[HarnessActionSpec]:
        return list(self._items)

    def get(self, action_id: str, revision: int) -> HarnessActionSpec:
        if int(revision) != self.revision:
            raise KeyError(f"stale action catalog revision: got {revision}, current {self.revision}")
        try:
            return self._by_id[action_id]
        except KeyError as exc:
            raise KeyError(f"unknown action_id {action_id!r} at revision {revision}") from exc

    def tool_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["action_id"],
            "additionalProperties": False,
            "properties": {"action_id": {"type": "string", "enum": list(self._by_id)}},
        }

    def compact_text(self) -> str:
        rows = []
        for item in self._items:
            arguments = " | ".join(f"{key}={value}" for key, value in item.arguments.items())
            rows.append(f"{item.action_id} | {item.action_type:<12} | {arguments or item.display_text}")
        return "\n".join(rows)
