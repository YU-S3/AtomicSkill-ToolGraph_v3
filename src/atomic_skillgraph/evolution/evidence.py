"""Per-trace aggregation of immutable Evolution asset evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvolutionAssetEvidence:
    artifact_ref: str
    artifact_kind: str
    validated_any: bool = False
    occurrence_ids: set[str] = field(default_factory=set)
    validation_outcomes: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        *,
        occurrence_id: str,
        passed: bool,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.validated_any |= bool(passed)
        self.occurrence_ids.add(str(occurrence_id))
        self.validation_outcomes.append({
            "occurrence_id": str(occurrence_id),
            "passed": bool(passed),
            "reason": str(reason),
            **dict(metadata or {}),
        })

    def metadata(self) -> dict[str, Any]:
        return {
            "occurrence_ids": sorted(self.occurrence_ids),
            "validation_outcomes": sorted(
                (dict(item) for item in self.validation_outcomes),
                key=lambda item: (
                    str(item.get("occurrence_id", "")),
                    str(item.get("reason", "")),
                    repr(sorted(item.items())),
                ),
            ),
        }


class EvolutionEvidenceAccumulator:
    """Aggregate evidence by ``(kind, ref)`` without false overwrites."""

    _KIND_ORDER = {
        "atomic": 0,
        "implementation": 1,
        "tool": 2,
        "composite": 3,
    }

    def __init__(self) -> None:
        self._assets: dict[tuple[str, str], EvolutionAssetEvidence] = {}

    def record(
        self,
        artifact_ref: str,
        artifact_kind: str,
        *,
        occurrence_id: str,
        passed: bool,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> EvolutionAssetEvidence:
        key = (str(artifact_kind), str(artifact_ref))
        state = self._assets.setdefault(
            key,
            EvolutionAssetEvidence(
                artifact_ref=key[1],
                artifact_kind=key[0],
            ),
        )
        state.record(
            occurrence_id=occurrence_id,
            passed=passed,
            reason=reason,
            metadata=metadata,
        )
        return state

    def assets(self) -> list[EvolutionAssetEvidence]:
        return sorted(
            self._assets.values(),
            key=lambda item: (
                self._KIND_ORDER.get(item.artifact_kind, 99),
                item.artifact_kind,
                item.artifact_ref,
            ),
        )

