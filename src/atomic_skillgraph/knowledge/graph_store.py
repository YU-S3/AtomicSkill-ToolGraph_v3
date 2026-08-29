"""Global semantic/lineage graph and verified reusable edge evidence."""

from __future__ import annotations

import json
from typing import Iterable

from ..core.edges import ExistingEdgeEvidence, GlobalGraphEdge, GlobalRelationType
from ..core.refs import content_hash
from ..core.status import RuntimeMode, SkillStatus
from .database import StateDatabase
from .skill_registry import SkillRegistry


class GraphStore:
    def __init__(self, database: StateDatabase, skills: SkillRegistry) -> None:
        self.database = database
        self.skills = skills

    def add(self, edge: GlobalGraphEdge) -> None:
        if self.database.readonly:
            raise RuntimeError("frozen graph store is read-only")
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO graph_edges(edge_id,source_ref,target_ref,relation,metadata_json) VALUES(?,?,?,?,?)",
                (edge.edge_id, edge.source_ref, edge.target_ref, edge.relation.value, json.dumps(edge.metadata, sort_keys=True)),
            )

    def edges(self, refs: Iterable[str] | None = None) -> list[GlobalGraphEdge]:
        rows = self.database.rows("SELECT * FROM graph_edges ORDER BY edge_id")
        allowed = set(refs or [])
        return [
            GlobalGraphEdge(row["edge_id"], row["source_ref"], row["target_ref"], row["relation"], json.loads(row["metadata_json"]))
            for row in rows if not allowed or row["source_ref"] in allowed or row["target_ref"] in allowed
        ]

    def existing_edges(self, atomic_refs: Iterable[str], *, mode: RuntimeMode | str) -> list[ExistingEdgeEvidence]:
        wanted = set(map(str, atomic_refs))
        evidence: list[ExistingEdgeEvidence] = []
        for composite in self.skills.composites(mode=mode):
            if composite.status is not SkillStatus.ACTIVE:
                continue
            by_step = {item.step_id: item for item in composite.occurrences}
            for edge in composite.data_edges + composite.dependency_edges:
                source = by_step.get(edge.source_step)
                target = by_step.get(edge.target_step)
                if source is None or target is None:
                    continue
                if wanted and (str(source.node_ref) not in wanted or str(target.node_ref) not in wanted):
                    continue
                try:
                    source_atomic = self.skills.get_atomic(source.node_ref)
                    target_atomic = self.skills.get_atomic(target.node_ref)
                except KeyError:
                    # An edge whose endpoint contract no longer exists cannot
                    # be authoritative evidence for Planner reuse.
                    continue
                source_types = {
                    item.name: item.semantic_type for item in source_atomic.outputs
                }
                target_types = {
                    item.name: item.semantic_type for item in target_atomic.inputs
                }
                edge_id = edge.edge_id or content_hash({
                    "composite": str(composite.ref), "source": str(source.node_ref), "target": str(target.node_ref),
                    "type": edge.edge_type.value, "source_role": edge.source_role, "target_role": edge.target_role,
                })[:20]
                evidence.append(ExistingEdgeEvidence(
                    edge_id=edge_id, source_composite_ref=str(composite.ref), source_step_ref=str(source.node_ref),
                    target_step_ref=str(target.node_ref), edge_type=edge.edge_type.value,
                    source_role=edge.source_role, target_role=edge.target_role,
                    semantic_types=(
                        source_types.get(edge.source_role, ""),
                        target_types.get(edge.target_role, ""),
                    ),
                    support_trace_ids=tuple(edge.evidence_refs),
                ))
        return evidence


    def existing_edge_by_id(self, edge_id: str, *, mode: RuntimeMode | str) -> ExistingEdgeEvidence | None:
        for edge in self.existing_edges([], mode=mode):
            if edge.edge_id == edge_id:
                return edge
        return None
