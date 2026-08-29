"""Related Composite summaries are repair hints only, never auto-executed."""

from __future__ import annotations

from typing import Any

from ..core.status import RuntimeMode
from ..core.serialization import to_primitive
from ..knowledge.skill_registry import SkillRegistry
from .atomic_retriever import AtomicSearchBatch


class RelatedCompositeHintFinder:
    def __init__(self, skills: SkillRegistry) -> None:
        self.skills = skills

    def find(self, search: AtomicSearchBatch, *, mode: RuntimeMode | str) -> list[dict[str, Any]]:
        selected = set(search.refs)
        hints: list[dict[str, Any]] = []
        for composite in self.skills.composites(mode=mode):
            overlap = [item for item in composite.occurrences if str(item.node_ref) in selected]
            if not overlap:
                continue
            hints.append({
                "composite_ref": str(composite.ref), "summary": composite.summary,
                "canonical_sequence": list(composite.control_sequence),
                "related_nodes": [
                    {"step_id": item.step_id, "atomic_ref": str(item.node_ref)} for item in overlap
                ],
                "insight": to_primitive(composite.insight),
                "hint_only": True,
            })
        return hints[:5]
