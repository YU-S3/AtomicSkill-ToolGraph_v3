"""Usage and cost accounting shared by all baseline methods.

Every generative call of a run is attributed to one of two roles: the
``target`` agent (episode solving) or the ``evolution`` machinery
(reflection / optimizer / retrieval).  Missing usage is never treated as
zero: reporting fails closed when a completed episode has no provider
evidence at all.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RoleUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0

    def add(self, other: "RoleUsage") -> None:
        self.calls += other.calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.reasoning_tokens += other.reasoning_tokens

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RoleUsage":
        return cls(
            calls=int(payload.get("calls", 0)),
            prompt_tokens=int(payload.get("prompt_tokens", 0)),
            completion_tokens=int(payload.get("completion_tokens", 0)),
            reasoning_tokens=int(payload.get("reasoning_tokens", 0)),
        )


@dataclass
class UsageSnapshot:
    target: RoleUsage = field(default_factory=RoleUsage)
    evolution: RoleUsage = field(default_factory=RoleUsage)
    embedding_calls: int = 0
    wall_time_ms: int = 0
    # No pricing table is frozen in the protocol; token counts are the
    # authoritative cost evidence and price application stays in reporting.
    api_cost_unpriced: bool = True
    api_cost: float | None = None
    per_stage: dict[str, RoleUsage] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UsageSnapshot":
        if not isinstance(payload, dict):
            raise ValueError("usage snapshot payload must be a mapping")
        return cls(
            target=RoleUsage.from_dict(dict(payload.get("target") or {})),
            evolution=RoleUsage.from_dict(dict(payload.get("evolution") or {})),
            embedding_calls=int(payload.get("embedding_calls", 0)),
            wall_time_ms=int(payload.get("wall_time_ms", 0)),
            api_cost_unpriced=bool(payload.get("api_cost_unpriced", True)),
            api_cost=payload.get("api_cost"),
            per_stage={
                str(stage): RoleUsage.from_dict(value)
                for stage, value in dict(payload.get("per_stage") or {}).items()
            },
        )

    def add(self, other: "UsageSnapshot") -> None:
        self.target.add(other.target)
        self.evolution.add(other.evolution)
        self.embedding_calls += other.embedding_calls
        self.wall_time_ms += other.wall_time_ms
        for stage, usage in other.per_stage.items():
            bucket = self.per_stage.setdefault(stage, RoleUsage())
            bucket.add(usage)

    @classmethod
    def delta(cls, after: "UsageSnapshot", before: "UsageSnapshot") -> "UsageSnapshot":
        """Per-bucket difference between two tracker snapshots."""

        def subtract(left: RoleUsage, right: RoleUsage) -> RoleUsage:
            return RoleUsage(
                calls=left.calls - right.calls,
                prompt_tokens=left.prompt_tokens - right.prompt_tokens,
                completion_tokens=left.completion_tokens - right.completion_tokens,
                reasoning_tokens=left.reasoning_tokens - right.reasoning_tokens,
            )

        stages = sorted(set(after.per_stage) | set(before.per_stage))
        return cls(
            target=subtract(after.target, before.target),
            evolution=subtract(after.evolution, before.evolution),
            embedding_calls=after.embedding_calls - before.embedding_calls,
            wall_time_ms=max(0, after.wall_time_ms - before.wall_time_ms),
            api_cost_unpriced=True,
            per_stage={
                stage: subtract(
                    after.per_stage.get(stage, RoleUsage()),
                    before.per_stage.get(stage, RoleUsage()),
                )
                for stage in stages
            },
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "UsageSnapshot":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"usage snapshot is unreadable: {path}") from exc
        return cls.from_dict(payload)


def usage_from_skillopt_token_summary(summary: dict[str, Any]) -> UsageSnapshot:
    """Map SkillOpt's per-stage token tracker onto the common role split.

    SkillOpt attributes target calls to stage ``rollout``; every other stage
    (analyst/merge/slow/meta/lr/…) is evolution machinery.  The upstream
    backend does not report reasoning tokens separately, so that counter
    stays at zero and is documented as such.
    """

    target = RoleUsage()
    evolution = RoleUsage()
    per_stage: dict[str, RoleUsage] = {}
    for stage, values in dict(summary or {}).items():
        if stage == "_total" or not isinstance(values, dict):
            continue
        usage = RoleUsage(
            calls=int(values.get("calls", 0)),
            prompt_tokens=int(values.get("prompt_tokens", 0)),
            completion_tokens=int(values.get("completion_tokens", 0)),
        )
        per_stage[stage] = usage
        if stage == "rollout":
            target.add(usage)
        else:
            evolution.add(usage)
    return UsageSnapshot(target=target, evolution=evolution, per_stage=per_stage)
