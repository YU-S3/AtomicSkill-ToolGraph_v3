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
            related_nodes = [
                {"step_id": item.step_id, "atomic_ref": str(item.node_ref)}
                for item in overlap
            ]
            components = []
            effect_predicates: set[str] = set()
            get_atomic = getattr(self.skills, "get_atomic", None)
            if not callable(get_atomic):
                components = []
                effect_predicates = set()
                for item in overlap:
                    components.append({
                        "atomic_ref": str(item.node_ref),
                        "effects": [],
                        "inputs": [],
                        "outputs": [],
                    })
                hints.append({
                    "composite_ref": str(composite.ref),
                    "summary": composite.summary,
                    "canonical_sequence": list(composite.control_sequence),
                    "related_nodes": related_nodes,
                    "components": components,
                    "effect_predicates": sorted(effect_predicates),
                    "insight": to_primitive(composite.insight),
                    "hint_only": True,
                })
                continue
            for item in overlap:
                atomic = get_atomic(item.node_ref)
                effects = [
                    {
                        "predicate": effect.predicate,
                        "args": {
                            key: (value.source_role if hasattr(value, "source_role") else value)
                            for key, value in effect.args.items()
                        },
                        "effect_domain": effect.effect_domain.value,
                    }
                    for effect in atomic.effects
                ]
                effect_predicates.update(effect.predicate.casefold() for effect in atomic.effects)
                components.append({
                    "atomic_ref": str(item.node_ref),
                    "effects": effects,
                    "inputs": [
                        {"name": spec.name, "semantic_type": spec.semantic_type}
                        for spec in atomic.inputs
                    ],
                    "outputs": [
                        {"name": spec.name, "semantic_type": spec.semantic_type}
                        for spec in atomic.outputs
                    ],
                })
            hints.append({
                "composite_ref": str(composite.ref), "summary": composite.summary,
                "canonical_sequence": list(composite.control_sequence),
                "related_nodes": related_nodes,
                "components": components,
                "effect_predicates": sorted(effect_predicates),
                "insight": to_primitive(composite.insight),
                "hint_only": True,
            })
        return hints[:5]
