"""Persistent and runtime graph edge contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GraphEdgeType(str, Enum):
    DATA_FLOW = "data_flow"
    REQUIRES_SKILL = "requires_skill"
    NEXT = "next"


@dataclass(frozen=True)
class GraphEdge:
    edge_id: str
    edge_type: GraphEdgeType
    source_step: str
    target_step: str
    source_role: str = ""
    target_role: str = ""
    origin: str = ""
    existing_edge_id: str = ""
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "edge_type", GraphEdgeType(self.edge_type))


@dataclass(frozen=True)
class ExistingEdgeEvidence:
    edge_id: str
    source_composite_ref: str
    source_step_ref: str
    target_step_ref: str
    edge_type: str
    source_role: str
    target_role: str
    semantic_types: tuple[str, str]
    support_trace_ids: tuple[str, ...]


class GlobalRelationType(str, Enum):
    IMPLEMENTS = "implements"
    CONTAINS = "contains"
    EQUIVALENT = "equivalent"
    SIMILAR = "similar"
    ALTERNATIVE = "alternative"
    CONFLICT = "conflict"
    VERIFIED_DATA_FLOW = "verified_data_flow"
    VERIFIED_DEPENDENCY = "verified_dependency"
    DERIVED_FROM = "derived_from"
    SUPERSEDES = "supersedes"
    SPLIT_FROM = "split_from"
    MERGED_FROM = "merged_from"


@dataclass(frozen=True)
class GlobalGraphEdge:
    edge_id: str
    source_ref: str
    target_ref: str
    relation: GlobalRelationType
    metadata: dict

    def __post_init__(self) -> None:
        object.__setattr__(self, "relation", GlobalRelationType(self.relation))
