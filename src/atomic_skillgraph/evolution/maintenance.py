"""Extraction policy and replay-gated long-term evolution operations."""

from __future__ import annotations

import copy
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Any, Callable

from ..core.contracts import AbstractAtomicSkill, CompositeSkill, ToolAsset
from ..core.edges import GlobalRelationType
from ..core.errors import FailureEnvelope, FailureLayer
from ..core.refs import SkillRef, ToolRef, bump_version, content_hash
from ..core.results import RuntimeLinearPlan, RuntimeOccurrence
from ..core.serialization import to_primitive
from ..core.status import RuntimeMode, SkillStatus, ToolStatus
from .repair import RepairProposal, RepairStore
from .repair_session import EvolutionToolCandidateProposal, EvolutionToolEditProposal
from .composite_repair_session import CompositeSequenceReview
from .trace_replay import build_trace_repair_evidence
from .typed_repair_session import TypedRepairReview
from .typed_repairs import RepairEvidence


@dataclass
class ExtractionDecision:
    should_extract: bool
    reasons: list[str]


@dataclass
class BatchMaintenanceResult:
    maintenance_trace_id: str
    admitted_assets: tuple[tuple[str, str], ...] = ()
    rejected_proposal_ids: tuple[str, ...] = ()
    pending_proposal_ids: tuple[str, ...] = ()
    reviewed_ids: tuple[str, ...] = ()
    lineage: tuple[dict[str, str], ...] = ()
    lifecycle_result: Any = None

    @property
    def admitted_refs(self) -> tuple[str, ...]:
        return tuple(ref for ref, _ in self.admitted_assets)

    @property
    def pending_count(self) -> int:
        return len(self.pending_proposal_ids)


class ExtractionPolicy:
    def __init__(
        self, *, extract_full_dynamic_success: bool = True,
        extract_task_rescue_success: bool = True,
        extract_novel_seeded_success: bool = True,
        skip_stable_direct_success: bool = True,
    ) -> None:
        self.extract_full_dynamic_success = bool(extract_full_dynamic_success)
        self.extract_task_rescue_success = bool(extract_task_rescue_success)
        self.extract_novel_seeded_success = bool(extract_novel_seeded_success)
        self.skip_stable_direct_success = bool(skip_stable_direct_success)

    def decide(self, trace: Any) -> ExtractionDecision:
        reasons: list[str] = []
        if (
            self.extract_full_dynamic_success
            and trace.runtime_plan.get("source") == "full_dynamic"
            and trace.benchmark_success
        ):
            reasons.append("full_dynamic_success")
        if self.extract_task_rescue_success and trace.task_rescue_required and trace.benchmark_success:
            reasons.append("task_rescue_success")
        if self.extract_novel_seeded_success and any(node.status == "seeded_success" for node in trace.node_records):
            known = {span.occurrence_id for span in trace.runtime_spans if span.kind == "tool"}
            if any(node.status == "seeded_success" and node.occurrence_id not in known for node in trace.node_records):
                reasons.append("novel_seeded_span")
        if any(
            span.learnable
            and (
                (span.kind == "full_dynamic" and self.extract_full_dynamic_success)
                or (span.kind == "task_rescue" and self.extract_task_rescue_success)
            )
            for span in trace.runtime_spans
        ):
            reasons.append("unaligned_runtime_span")
        if (
            not self.skip_stable_direct_success
            and trace.benchmark_success
            and trace.implementation_direct_success
            and not reasons
        ):
            reasons.append("stable_direct_diagnostic")
        return ExtractionDecision(bool(reasons), list(dict.fromkeys(reasons)))

    def should_extract(self, trace: Any) -> bool:
        decision = self.decide(trace)
        trace.extraction_policy = {"should_extract": decision.should_extract, "reasons": decision.reasons}
        return decision.should_extract


class EvolutionMaintenance:
    """Every semantic edit yields a new candidate and remains replay gated."""

    def __init__(self, repair_store: RepairStore) -> None:
        self.repair_store = repair_store

    def prepare_failure_repairs(
        self,
        failures: list[FailureEnvelope],
        *,
        trace: Any | None = None,
        tools: Any | None = None,
        skills: Any | None = None,
        harness_profile: str = "",
    ) -> list[RepairProposal]:
        """Create immutable local proposals without guessing an executable edit.

        The proposals are deliberately not replayed or admitted here.  An
        executable/binding repair must later supply a concrete local patch and
        pass ``RepairStore.replay_and_admit`` before a new Candidate version is
        registered.
        """
        grouped: dict[tuple[str, str, str], list[FailureEnvelope]] = {}
        for failure in failures:
            target, target_layer, operation = self._repair_target(failure)
            if not target:
                continue
            grouped.setdefault((target, target_layer, operation), []).append(failure)

        proposals: list[RepairProposal] = []
        for (target, target_layer, operation), items in sorted(grouped.items()):
            patch = {
                "diagnostic": {
                    "failure_codes": sorted({item.code for item in items}),
                    "occurrence_ids": sorted({item.occurrence_id for item in items if item.occurrence_id}),
                    "attempt_ids": sorted({item.attempt_id for item in items if item.attempt_id}),
                    "started": any(item.started for item in items),
                    "evidence_refs": sorted({ref for item in items for ref in item.evidence_refs}),
                },
                "required_replay_trace_ids": sorted({item.trace_id for item in items}),
                "requires_concrete_patch": True,
            }
            if target_layer == "tool":
                cluster_items = [
                    self._tool_failure_cluster(
                        item,
                        trace=trace,
                        tools=tools,
                        skills=skills,
                        harness_profile=harness_profile,
                    )
                    for item in items
                ]
                patch["diagnostic"]["failure_clusters"] = [
                    item for item in cluster_items if item is not None
                ]
            proposals.append(RepairProposal.create(
                target,
                target_layer,
                operation,
                patch,
                [item.failure_id for item in items],
            ))
        return proposals

    @staticmethod
    def _tool_failure_cluster(
        failure: FailureEnvelope,
        *,
        trace: Any | None,
        tools: Any | None,
        skills: Any | None,
        harness_profile: str,
    ) -> dict[str, Any] | None:
        intrinsic_codes = {
            "tool_primitive_rejected",
            "tool_execution_error",
            "tool_output_schema_error",
        }
        if (
            failure.layer is not FailureLayer.TOOL
            or not failure.started
            or failure.code not in intrinsic_codes
            or trace is None
            or tools is None
        ):
            return None
        execution = next(
            (
                item for item in trace.tool_executions
                if str(item.attempt_id) == failure.attempt_id
                and bool(dict(item.result or {}).get("started"))
            ),
            None,
        )
        tool_ref = next(
            (item for item in failure.artifact_refs if item.startswith("tool://")),
            "",
        )
        if execution is None or not tool_ref or str(execution.tool_ref) != tool_ref:
            return None
        try:
            tool = tools.get(tool_ref)
        except KeyError:
            return None
        input_types = {
            str(role): str((spec or {}).get("type", ""))
            for role, spec in sorted(
                dict(tool.signature.get("properties") or {}).items()
            )
        }
        constraints: list[Any] = []
        replay_case: dict[str, Any] | None = None
        if skills is not None:
            for invocation in trace.implementation_invocations:
                if str(invocation.occurrence_id) != failure.occurrence_id:
                    continue
                try:
                    implementation = skills.get_implementation(
                        str(invocation.implementation_ref)
                    )
                except KeyError:
                    continue
                binding = next(
                    (
                        item for item in implementation.tool_bindings
                        if str(item.tool_ref) == tool_ref
                    ),
                    None,
                )
                if binding is None:
                    continue
                constraints = to_primitive(implementation.grounding_constraints)
                resolved: dict[str, Any] = {}
                for role, expression in binding.parameter_mapping.items():
                    if expression.kind.value == "constant":
                        resolved[role] = expression.constant
                    elif (
                        expression.kind.value == "skill_input"
                        and expression.source_role in invocation.arguments
                    ):
                        resolved[role] = invocation.arguments[expression.source_role]
                required = set(tool.signature.get("required") or [])
                if required - set(resolved):
                    break
                span = next(
                    (
                        item for item in trace.runtime_spans
                        if str(item.span_id) == str(execution.span_id)
                    ),
                    None,
                )
                node = next(
                    (
                        item for item in trace.node_records
                        if str(item.occurrence_id) == failure.occurrence_id
                    ),
                    None,
                )
                if span is None or node is None:
                    break
                try:
                    atomic = skills.get_atomic(str(node.atomic_ref))
                except KeyError:
                    break
                prefix = [
                    {
                        "action_type": str(item.action_type),
                        "arguments": dict(item.arguments),
                    }
                    for item in trace.environment_actions[: int(span.action_start)]
                    if item.accepted
                ]
                replay_case = {
                    "kind": "source_replay",
                    "trace_id": str(trace.trace_id),
                    "failure_attempt_id": failure.attempt_id,
                    "bindings": resolved,
                    "source_task": {
                        "task_id": str(trace.task.task_id),
                        "task_signature": str(trace.task.task_signature),
                        "goal": str(trace.task.goal),
                        "benchmark": str(trace.task.benchmark),
                        "task_type": str(trace.task.task_type),
                        "context": {
                            "env_index": trace.task.metadata.get("env_index"),
                            "game_file": trace.task.metadata.get("game_file", ""),
                        },
                        "metadata": dict(trace.task.metadata),
                    },
                    "prefix": prefix,
                    "effects": to_primitive(atomic.effects),
                    "failure_replay": True,
                }
                break
        if replay_case is None:
            # A stable label without a reproducible failing input is not
            # sufficient authority to synthesize a specialization.
            return None
        harness_context = {
            "profile": str(harness_profile),
            "task_type": str(trace.task.task_type),
            "split": str(trace.task.metadata.get("split", "")),
        }
        cluster = {
            "failure_code": failure.code,
            "task_id": failure.task_id,
            "trace_id": failure.trace_id,
            "attempt_id": failure.attempt_id,
            "started": True,
            "intrinsic_tool_failure": True,
            "input_semantic_types": input_types,
            "harness_context": harness_context,
            "parameter_constraints": constraints,
            "failure_replay_case": replay_case,
        }
        cluster["cluster_key"] = content_hash({
            "failure_code": cluster["failure_code"],
            "input_semantic_types": input_types,
            "harness_context": harness_context,
            "parameter_constraints": constraints,
        })
        return cluster

    def commit_repairs(self, proposals: list[RepairProposal]) -> None:
        for proposal in proposals:
            if proposal.status != "proposed":
                raise ValueError("only immutable proposed repairs may enter the maintenance queue")
            self.repair_store.save(proposal)

    def prepare_validated_composite_repair(
        self,
        target_ref: str,
        candidate: Any,
        failure_ids: list[str],
        *,
        operation: str = "insert_missing_occurrence",
    ) -> RepairProposal:
        """Represent a successful rescue extraction as a concrete local edit."""
        if not target_ref or not target_ref.startswith("skill://"):
            raise ValueError("validated Composite repair requires its source Composite ref")
        if operation not in {
            "insert_missing_occurrence", "revise_composite_sequence",
        }:
            raise ValueError("unsupported validated Composite repair operation")
        return RepairProposal.create(
            target_ref,
            "composite",
            operation,
            {
                "replacement_candidate": to_primitive(candidate),
                "required_replay_trace_ids": sorted({
                    trace_id
                    for trace_id in candidate.metadata.get("source_trace_ids", [])
                    if trace_id
                }),
                "requires_concrete_patch": False,
            },
            list(failure_ids),
        )

    def admit_validated_composite_repair(
        self,
        proposal: RepairProposal,
        *,
        admitted_ref: str,
        validation_passed: bool,
    ) -> RepairProposal:
        """Close replay→validation→admission for an already validated rescue."""
        return self.repair_store.replay_and_admit(
            proposal,
            replay=lambda _proposal: {
                "passed": bool(validation_passed),
                "admitted_ref": admitted_ref,
                "source_replays": list(
                    _proposal.proposed_patch.get("required_replay_trace_ids", [])
                ),
            },
            admit=lambda current, result: bool(
                result.get("passed")
                and result.get("admitted_ref")
                and result.get("admitted_ref") != current.target_ref
            ),
        )

    @staticmethod
    def _repair_target(failure: FailureEnvelope) -> tuple[str, str, str]:
        refs = list(failure.artifact_refs)
        if failure.layer is FailureLayer.IMPLEMENTATION:
            target = next((item for item in refs if item.startswith("skill://impl_")), "")
            target = target or next((item for item in refs if item.startswith("skill://")), "")
            operation = (
                "revise_grounding_constraint"
                if failure.code == "implementation_constraint_error"
                else "revise_implementation_mapping"
            )
            return target, "implementation", operation
        if failure.layer is FailureLayer.TOOL:
            return next((item for item in refs if item.startswith("tool://")), ""), "tool", "add_tool_test"
        if failure.layer is FailureLayer.ATOMIC:
            target = next((item for item in refs if item.startswith("skill://atomic_")), "")
            target = target or next((item for item in refs if item.startswith("skill://")), "")
            return target, "atomic", "revise_atomic_contract"
        if failure.layer in {
            FailureLayer.DATA_FLOW,
            FailureLayer.COMPOSITE,
            FailureLayer.TASK_CONTRACT,
        }:
            target = next((item for item in refs if item.startswith("skill://composite_")), "")
            target = target or next((item for item in refs if item.startswith("skill://")), "")
            return target, "composite", "insert_missing_occurrence"
        return "", "", ""

    def propose_atomic_revision(self, ref: SkillRef, patch: dict[str, Any], failure_ids: list[str]) -> RepairProposal:
        proposal = RepairProposal.create(str(ref), "atomic", "revise_atomic_contract", patch, failure_ids)
        self.repair_store.save(proposal)
        return proposal

    def propose_implementation_revision(self, ref: SkillRef, patch: dict[str, Any], failure_ids: list[str]) -> RepairProposal:
        operation = "revise_grounding_constraint" if "grounding_constraints" in patch else "revise_implementation_mapping"
        proposal = RepairProposal.create(str(ref), "implementation", operation, patch, failure_ids)
        self.repair_store.save(proposal)
        return proposal

    def propose_tool_generalization(self, ref: ToolRef, patch: dict[str, Any], source_case_ids: list[str]) -> RepairProposal:
        patch = {**patch, "required_source_replays": list(source_case_ids), "evolution_operation": "generalize"}
        proposal = RepairProposal.create(str(ref), "tool", "replace_tool_body", patch, [])
        self.repair_store.save(proposal)
        return proposal

    def propose_tool_specialization(self, ref: ToolRef, patch: dict[str, Any], stable_failure_ids: list[str]) -> RepairProposal:
        if len(set(stable_failure_ids)) < 2:
            raise ValueError("Tool specialization requires a stable multi-failure cluster")
        proposal = RepairProposal.create(str(ref), "tool", "specialize_tool", patch, stable_failure_ids)
        self.repair_store.save(proposal)
        return proposal

    def propose_tool_merge(self, refs: list[ToolRef], equivalence: dict[str, Any], replay_ids: list[str]) -> RepairProposal:
        if len(refs) < 2 or not all(equivalence.get(key) for key in ("behavior", "interface", "effect")):
            raise ValueError("Tool merge requires behavior/interface/effect equivalence")
        proposal = RepairProposal.create(str(refs[0]), "tool", "replace_tool_body", {
            "evolution_operation": "merge", "merged_refs": [str(item) for item in refs],
            "required_source_replays": replay_ids, "equivalence": equivalence,
        }, [])
        self.repair_store.save(proposal)
        return proposal

    def propose_tool_split(self, ref: ToolRef, boundaries: list[dict[str, Any]], failure_ids: list[str]) -> RepairProposal:
        if len(boundaries) < 2:
            raise ValueError("Tool split requires at least two independent boundaries")
        proposal = RepairProposal.create(str(ref), "tool", "split_tool", {"boundaries": boundaries}, failure_ids)
        self.repair_store.save(proposal)
        return proposal

    def propose_composite_revision(
        self, ref: SkillRef, *, candidate: CompositeSkill, operation: str,
        evidence_ids: list[str], replay_trace_ids: list[str],
    ) -> RepairProposal:
        if operation not in {"revise_composite_sequence", "remove_redundant_occurrence", "insert_missing_occurrence"}:
            raise ValueError(operation)
        if str(candidate.ref) == str(ref):
            raise ValueError("Composite semantic revision requires a new immutable version")
        proposal = RepairProposal.create(str(ref), "composite", operation, {
            "replacement_candidate": to_primitive(candidate),
            "required_replay_trace_ids": sorted(set(replay_trace_ids)),
            "requires_concrete_patch": False,
        }, evidence_ids)
        self.repair_store.save(proposal)
        return proposal

    def build_batch_reviews(self, tools: Any) -> list[dict[str, Any]]:
        """Build evidence-only cohorts; this method never proposes semantics."""
        usable = [
            item
            for item in tools.tools()
            if item.status in {
                ToolStatus.CANDIDATE,
                ToolStatus.ACTIVE,
                ToolStatus.PREFERRED,
            }
        ]
        reviewed = {
            str(item.proposed_patch.get("review_id"))
            for item in self.repair_store.history()
            if item.proposed_patch.get("review_id")
        }
        reviews: list[dict[str, Any]] = []

        shape_groups: dict[str, list[ToolAsset]] = defaultdict(list)
        for tool in usable:
            shape_groups[_tool_shape(tool)].append(tool)
        for group in shape_groups.values():
            if len(group) < 2 or len({_tool_semantics(item) for item in group}) < 2:
                continue
            context = self._tool_review(
                "generalize",
                sorted(group, key=lambda item: str(item.ref)),
                eligible_operations=["generalize"],
            )
            if context["review_id"] not in reviewed:
                reviews.append(context)

        failures_by_target: dict[str, list[RepairProposal]] = defaultdict(list)
        for proposal in self.repair_store.pending():
            if proposal.target_layer == "tool" and proposal.source_failure_ids:
                failures_by_target[proposal.target_ref].append(proposal)
        by_ref = {str(item.ref): item for item in usable}
        for target_ref, failures in sorted(failures_by_target.items()):
            tool = by_ref.get(target_ref)
            if tool is None:
                continue
            clusters: dict[str, list[tuple[RepairProposal, dict[str, Any]]]] = defaultdict(list)
            for failure_proposal in failures:
                diagnostic = dict(failure_proposal.proposed_patch.get("diagnostic") or {})
                for cluster in diagnostic.get("failure_clusters") or []:
                    key = str(cluster.get("cluster_key", ""))
                    if key:
                        clusters[key].append((failure_proposal, dict(cluster)))
            for cluster_key, entries in sorted(clusters.items()):
                unique_evidence: dict[tuple[str, str, str], dict[str, Any]] = {}
                for _, item in entries:
                    identity = (
                        str(item.get("task_id", "")),
                        str(item.get("trace_id", "")),
                        str(item.get("attempt_id", "")),
                    )
                    if all(identity):
                        unique_evidence.setdefault(identity, item)
                evidence = list(unique_evidence.values())
                tasks = {str(item.get("task_id", "")) for item in evidence}
                traces = {str(item.get("trace_id", "")) for item in evidence}
                attempts = {str(item.get("attempt_id", "")) for item in evidence}
                stable = (
                    len(evidence) >= 2
                    and len(tasks - {""}) >= 2
                    and len(traces - {""}) >= 2
                    and len(attempts - {""}) >= 2
                    and all(
                        item.get("started") is True
                        and item.get("intrinsic_tool_failure") is True
                        and str(item.get("cluster_key", "")) == cluster_key
                        and isinstance(item.get("failure_replay_case"), dict)
                        for item in evidence
                    )
                )
                if not stable:
                    continue
                source_proposals = list(dict.fromkeys(
                    proposal.proposal_id for proposal, _ in entries
                ))
                source_ids = set(source_proposals)
                failure_ids = sorted({
                    failure_id
                    for proposal in failures
                    if proposal.proposal_id in source_ids
                    for failure_id in proposal.source_failure_ids
                })
                context = self._tool_review(
                    "specialize",
                    [tool],
                    eligible_operations=["specialize", "update"],
                    failure_ids=failure_ids,
                    source_proposal_ids=source_proposals,
                    failure_cluster={
                        "cluster_key": cluster_key,
                        "failure_code": evidence[0]["failure_code"],
                        "input_semantic_types": evidence[0]["input_semantic_types"],
                        "harness_context": evidence[0]["harness_context"],
                        "parameter_constraints": evidence[0]["parameter_constraints"],
                        "task_ids": sorted(tasks - {""}),
                        "trace_ids": sorted(traces - {""}),
                        "attempt_ids": sorted(attempts - {""}),
                    },
                    supplemental_cases=[
                        dict(item["failure_replay_case"]) for item in evidence
                    ],
                )
                if context["review_id"] not in reviewed:
                    reviews.append(context)

        for tool in usable:
            steps = list(tool.artifact.get("steps") or [])
            effect_shapes = {
                content_hash(effect)
                for case in _source_replays(tool)
                for effect in case.get("effects", [])
            }
            if len(steps) < 2 or len(effect_shapes) < 2:
                continue
            context = self._tool_review(
                "split",
                [tool],
                eligible_operations=["split"],
            )
            if context["review_id"] not in reviewed:
                reviews.append(context)
        return sorted(reviews, key=lambda item: item["review_id"])

    def build_typed_reviews(
        self,
        *,
        skills: Any,
        tools: Any,
        traces: Any,
        harness_profile: str,
    ) -> list[TypedRepairReview]:
        """Build stable Atomic/Implementation reviews from stored trace facts."""
        grouped: dict[
            tuple[str, str, str],
            list[tuple[RepairProposal, Any]],
        ] = defaultdict(list)
        for proposal in self.repair_store.pending():
            if (
                proposal.target_layer not in {"atomic", "implementation"}
                or proposal.proposed_patch.get("typed_schema")
            ):
                continue
            trace_ids = list(
                proposal.proposed_patch.get("required_replay_trace_ids") or []
            )
            for trace_id in trace_ids:
                payload = traces.load_payload(str(trace_id))
                failures = {
                    str(item.get("failure_id", "")): dict(item)
                    for item in payload.get("failures", [])
                }
                for failure_id in proposal.source_failure_ids:
                    failure = failures.get(str(failure_id))
                    if failure is None:
                        continue
                    evidence = build_trace_repair_evidence(
                        payload,
                        failure,
                        target_layer=proposal.target_layer,
                        target_ref=proposal.target_ref,
                        skills=skills,
                        harness_profile=harness_profile,
                    )
                    if evidence is not None:
                        grouped[
                            (
                                proposal.target_layer,
                                proposal.target_ref,
                                evidence.cluster_key,
                            )
                        ].append((proposal, evidence))

        reviews: list[TypedRepairReview] = []
        resolved_reviews = {
            str(item.proposed_patch.get("typed_review_id", ""))
            for item in self.repair_store.history()
            if item.proposed_patch.get("typed_review_id")
        }
        for (layer, target_ref, cluster_key), entries in sorted(grouped.items()):
            unique: dict[tuple[str, str, str], tuple[RepairProposal, Any]] = {}
            for proposal, evidence in entries:
                unique.setdefault(
                    (evidence.evidence_id, evidence.task_id, evidence.trace_id),
                    (proposal, evidence),
                )
            evidence = [item for _, item in unique.values()]
            if (
                len(evidence) < 2
                or len({item.task_id for item in evidence}) < 2
                or len({item.trace_id for item in evidence}) < 2
                or {item.cluster_key for item in evidence} != {cluster_key}
                or len({item.failure_code for item in evidence}) != 1
            ):
                continue
            try:
                if layer == "atomic":
                    source = skills.get_atomic(target_ref)
                    operations = ["revise_atomic_contract"]
                    if len(source.effects) >= 2:
                        operations.append("split_atomic")
                else:
                    source = skills.get_implementation(target_ref)
                    code = evidence[0].failure_code
                    operations = {
                        "implementation_mapping_error": [
                            "revise_implementation_mapping",
                        ],
                        "implementation_constraint_error": [
                            "revise_grounding_constraint",
                            "specialize_implementation",
                        ],
                        "implementation_compatibility_error": [
                            "specialize_implementation",
                        ],
                        "implementation_invocation_failed": [
                            "revise_implementation_mapping",
                            "revise_grounding_constraint",
                        ],
                    }[code]
            except KeyError:
                continue
            source_ids = sorted({
                proposal.proposal_id for proposal, _ in unique.values()
            })
            failure_ids = sorted({
                failure_id
                for proposal, _ in unique.values()
                for failure_id in proposal.source_failure_ids
            })
            core = {
                "target_layer": layer,
                "target_refs": [target_ref],
                "eligible_operations": operations,
                "cluster_key": cluster_key,
                "evidence_ids": sorted(item.evidence_id for item in evidence),
                "source_proposal_ids": source_ids,
            }
            review_id = "typed_review_" + content_hash(core)[:24]
            if review_id in resolved_reviews:
                continue
            reviews.append(TypedRepairReview(
                review_id=review_id,
                target_layer=layer,
                target_refs=(target_ref,),
                eligible_operations=tuple(operations),
                context={
                    "source_artifact": to_primitive(source),
                    "stable_cluster": {
                        "cluster_key": cluster_key,
                        "failure_code": evidence[0].failure_code,
                        "task_ids": sorted({item.task_id for item in evidence}),
                        "trace_ids": sorted({item.trace_id for item in evidence}),
                    },
                    "source_proposal_ids": source_ids,
                },
                evidence=tuple(evidence),
                source_failure_ids=tuple(failure_ids),
            ))

        # Atomic merge is not a failure repair.  Recall contract-identical
        # cohorts deterministically, then require independently persisted
        # successful source cases for the cohort.  The LLM receives no power
        # to choose cohort members or evidence, and embedding similarity is
        # deliberately absent from this hard merge gate.
        merge_groups: dict[str, list[AbstractAtomicSkill]] = defaultdict(list)
        for atomic in skills.atomics():
            if atomic.status not in {SkillStatus.CANDIDATE, SkillStatus.ACTIVE}:
                continue
            merge_groups[_atomic_alignment_contract(atomic)].append(atomic)
        for cluster_key, cohort in sorted(merge_groups.items()):
            # Different versions of the same logical Atomic are revisions,
            # not a merge cohort.
            logical_ids = {item.ref.logical_id for item in cohort}
            if len(logical_ids) < 2:
                continue
            selected = sorted(
                cohort,
                key=lambda item: (item.ref.logical_id, _version_key(item.ref.version)),
            )
            evidence = _atomic_merge_evidence(
                selected,
                cluster_key=cluster_key,
                skills=skills,
                tools=tools,
                traces=traces,
            )
            if (
                len(evidence) < 2
                or len({item.task_id for item in evidence}) < 2
                or len({item.trace_id for item in evidence}) < 2
                or {
                    str(item.replay_case.get("target_ref") or "")
                    for item in evidence
                } != {str(item.ref) for item in selected}
            ):
                continue
            target_refs = tuple(str(item.ref) for item in selected)
            core = {
                "target_layer": "atomic",
                "target_refs": target_refs,
                "eligible_operations": ["merge_atomic"],
                "cluster_key": cluster_key,
                "evidence_ids": sorted(item.evidence_id for item in evidence),
            }
            review_id = "typed_review_" + content_hash(core)[:24]
            if review_id in resolved_reviews:
                continue
            reviews.append(TypedRepairReview(
                review_id=review_id,
                target_layer="atomic",
                target_refs=target_refs,
                eligible_operations=("merge_atomic",),
                context={
                    "source_artifacts": [to_primitive(item) for item in selected],
                    "alignment_gate": {
                        "contract_key": cluster_key,
                        "effect_compatible": True,
                        "io_compatible": True,
                        "precondition_compatible": True,
                        "validator_compatible": True,
                        "atomic_boundary_compatible": True,
                    },
                    "source_proposal_ids": [],
                },
                evidence=tuple(evidence),
                source_failure_ids=(),
            ))
        return sorted(reviews, key=lambda item: item.review_id)

    def build_composite_sequence_reviews(
        self,
        *,
        skills: Any,
        traces: Any,
        harness_profile: str,
    ) -> list[CompositeSequenceReview]:
        """Build stable structural reviews from matching persisted attempts."""
        grouped: dict[
            tuple[str, str], list[tuple[RepairProposal, RepairEvidence]],
        ] = defaultdict(list)
        for proposal in self.repair_store.pending():
            if (
                proposal.target_layer != "composite"
                or proposal.proposed_patch.get("typed_schema")
                or proposal.operation not in {
                    "insert_missing_occurrence", "revise_composite_sequence",
                }
            ):
                continue
            try:
                source = skills.get_composite(proposal.target_ref)
            except KeyError:
                continue
            for trace_id in list(
                proposal.proposed_patch.get("required_replay_trace_ids") or []
            ):
                try:
                    payload = traces.load_payload(str(trace_id))
                except KeyError:
                    continue
                failure_by_id = {
                    str(item.get("failure_id", "")): dict(item)
                    for item in payload.get("failures", [])
                }
                for failure_id in proposal.source_failure_ids:
                    failure = failure_by_id.get(str(failure_id))
                    if failure is None:
                        continue
                    evidence = _composite_sequence_evidence(
                        payload,
                        failure,
                        source=source,
                        skills=skills,
                        harness_profile=harness_profile,
                    )
                    if evidence is not None:
                        grouped[(str(source.ref), evidence.cluster_key)].append(
                            (proposal, evidence)
                        )

        resolved = {
            str(item.proposed_patch.get("composite_sequence_review_id", ""))
            for item in self.repair_store.history()
            if item.proposed_patch.get("composite_sequence_review_id")
        }
        reviews: list[CompositeSequenceReview] = []
        for (target_ref, cluster_key), entries in sorted(grouped.items()):
            unique: dict[tuple[str, str], tuple[RepairProposal, RepairEvidence]] = {}
            for proposal, evidence in entries:
                unique.setdefault((evidence.task_id, evidence.trace_id), (proposal, evidence))
            evidence = [item for _, item in unique.values()]
            if (
                len(evidence) < 2
                or len({item.task_id for item in evidence}) < 2
                or len({item.trace_id for item in evidence}) < 2
                or {item.cluster_key for item in evidence} != {cluster_key}
                or len({item.failure_code for item in evidence}) != 1
            ):
                continue
            source = skills.get_composite(target_ref)
            source_proposals = sorted({
                proposal.proposal_id for proposal, _ in unique.values()
            })
            failure_ids = sorted({
                failure_id
                for proposal, _ in unique.values()
                for failure_id in proposal.source_failure_ids
            })
            core = {
                "target_ref": target_ref,
                "cluster_key": cluster_key,
                "evidence_ids": sorted(item.evidence_id for item in evidence),
                "source_proposal_ids": source_proposals,
            }
            review_id = "composite_sequence_review_" + content_hash(core)[:24]
            if review_id in resolved:
                continue
            reviews.append(CompositeSequenceReview(
                review_id=review_id,
                target_ref=target_ref,
                source_composite=to_primitive(source),
                structural_context={
                    "failure_code": evidence[0].failure_code,
                    "cluster_key": cluster_key,
                    "harness_profile": harness_profile,
                    "source_proposal_ids": source_proposals,
                    "authoritative_occurrence_order": list(source.control_sequence),
                },
                evidence=tuple(evidence),
                source_failure_ids=tuple(failure_ids),
            ))
        return sorted(reviews, key=lambda item: item.review_id)

    def _tool_review(
        self,
        kind: str,
        tools: list[ToolAsset],
        *,
        eligible_operations: list[str],
        failure_ids: list[str] | None = None,
        source_proposal_ids: list[str] | None = None,
        failure_cluster: dict[str, Any] | None = None,
        supplemental_cases: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        cases = _case_catalog(tools)
        for case in supplemental_cases or []:
            case_id = "case_" + content_hash(case)[:24]
            entry = cases.setdefault(
                case_id,
                {"case": copy.deepcopy(case), "target_refs": set()},
            )
            if content_hash(entry["case"]) != content_hash(case):
                raise ValueError("source replay case hash collision")
            entry["target_refs"].update(str(item.ref) for item in tools)
        core = {
            "kind": kind,
            "eligible_operations": eligible_operations,
            "target_refs": [str(item.ref) for item in tools],
            "failure_ids": sorted(failure_ids or []),
            "source_proposal_ids": sorted(source_proposal_ids or []),
            "stable_failure_cluster": to_primitive(failure_cluster or {}),
            "tools": [
                {
                    "ref": str(item.ref),
                    "summary": item.summary,
                    "signature": to_primitive(item.signature),
                    "interface": to_primitive(item.interface),
                    "artifact_kind": item.artifact_kind,
                    "artifact": to_primitive(item.artifact),
                    "safety": to_primitive(item.safety),
                }
                for item in tools
            ],
            "source_cases": [
                {
                    "case_id": case_id,
                    "target_refs": sorted(entry["target_refs"]),
                    "case": to_primitive(entry["case"]),
                }
                for case_id, entry in sorted(cases.items())
            ],
        }
        return {
            "review_id": "review_" + content_hash(core)[:24],
            **core,
        }

    def run_batch(
        self,
        *,
        maintenance_trace_id: str,
        reviews: list[dict[str, Any]],
        agent_proposals: list[EvolutionToolEditProposal],
        tools: Any,
        skills: Any,
        admission: Any,
        projection: Any,
        traces: Any,
        planner_validator: Any,
        harness_profile: str,
        replay_tool: Callable[[ToolAsset, dict[str, Any]], bool],
        replay_composite: Callable[[CompositeSkill, dict[str, Any]], bool],
        finalize_pending: bool = False,
    ) -> BatchMaintenanceResult:
        """Run deterministic detection and replay-gated batch admissions."""
        review_by_id = {str(item["review_id"]): item for item in reviews}
        if len(review_by_id) != len(reviews):
            raise ValueError("duplicate batch review id")
        response_by_id: dict[str, EvolutionToolEditProposal] = {}
        for proposal in agent_proposals:
            if proposal.review_id in response_by_id:
                raise ValueError("EvolutionRepairSession returned a review twice")
            review = review_by_id.get(proposal.review_id)
            if review is None:
                raise ValueError("EvolutionRepairSession referenced an unknown review")
            self._validate_agent_proposal(proposal, review)
            response_by_id[proposal.review_id] = proposal

        tool_proposals: list[RepairProposal] = []
        proposal_reviews: dict[str, dict[str, Any]] = {}
        reviewed_ids: list[str] = []
        for review_id, review in review_by_id.items():
            response = response_by_id.get(review_id)
            if response is None:
                decision = RepairProposal.create(
                    str(review["target_refs"][0]),
                    "tool",
                    "replace_tool_body",
                    {
                        "review_id": review_id,
                        "review_outcome": "no_change",
                        "requires_concrete_patch": False,
                    },
                    list(review.get("failure_ids") or []),
                )
                decision.status = "rejected"
                decision.replay_result = {
                    "passed": False,
                    "failure_code": "semantic_edit_not_proposed",
                }
                self.repair_store.save(decision)
                reviewed_ids.append(review_id)
                continue
            operation = {
                "generalize": "replace_tool_body",
                "specialize": "specialize_tool",
                "update": "replace_tool_body",
                "merge": "replace_tool_body",
                "split": "split_tool",
            }[response.operation]
            repair = RepairProposal.create(
                response.target_refs[0],
                "tool",
                operation,
                {
                    "review_id": review_id,
                    "evolution_operation": response.operation,
                    "target_refs": list(response.target_refs),
                    "candidate_specs": to_primitive(response.candidates),
                    "source_proposal_ids": list(review.get("source_proposal_ids") or []),
                    "requires_concrete_patch": False,
                    "rationale": response.rationale,
                },
                list(review.get("failure_ids") or []),
            )
            self.repair_store.save(repair)
            tool_proposals.append(repair)
            proposal_reviews[repair.proposal_id] = review
            reviewed_ids.append(review_id)

        for review, response in self._duplicate_merge_edits(tools):
            repair = RepairProposal.create(
                response.target_refs[0],
                "tool",
                "replace_tool_body",
                {
                    "review_id": review["review_id"],
                    "evolution_operation": "merge",
                    "target_refs": list(response.target_refs),
                    "candidate_specs": to_primitive(response.candidates),
                    "source_proposal_ids": [],
                    "requires_concrete_patch": False,
                    "rationale": response.rationale,
                },
                [],
            )
            self.repair_store.save(repair)
            tool_proposals.append(repair)
            proposal_reviews[repair.proposal_id] = review
            reviewed_ids.append(review["review_id"])

        admitted_assets: list[tuple[str, str]] = []
        rejected_ids: list[str] = []
        lineage: list[dict[str, str]] = []
        resolved_source_ids: set[str] = set()
        for proposal in tool_proposals:
            admitted, proposal_lineage = self._process_tool_proposal(
                proposal,
                proposal_reviews[proposal.proposal_id],
                tools=tools,
                admission=admission,
                replay_tool=replay_tool,
            )
            if admitted.status == "admitted":
                admitted_assets.extend(
                    (str(ref), "tool")
                    for ref in admitted.replay_result.get("admitted_refs", [])
                )
                lineage.extend(proposal_lineage)
                resolved_source_ids.update(
                    map(str, proposal.proposed_patch.get("source_proposal_ids") or [])
                )
            else:
                rejected_ids.append(admitted.proposal_id)

        composite_results = self._review_composite_redundancy(
            maintenance_trace_id=maintenance_trace_id,
            skills=skills,
            projection=projection,
            traces=traces,
            planner_validator=planner_validator,
            harness_profile=harness_profile,
            replay_composite=replay_composite,
        )
        for proposal, admitted_ref, source_ref in composite_results:
            reviewed_ids.append(str(proposal.proposed_patch.get("review_id", "")))
            if proposal.status == "admitted" and admitted_ref:
                admitted_assets.append((admitted_ref, "composite"))
                lineage.append({
                    "source_ref": admitted_ref,
                    "target_ref": source_ref,
                    "relation": GlobalRelationType.DERIVED_FROM.value,
                    "operation": "remove_redundant_occurrence",
                    "proposal_id": proposal.proposal_id,
                    "review_id": str(
                        proposal.proposed_patch.get("review_id", "")
                    ),
                })
            else:
                rejected_ids.append(proposal.proposal_id)

        insight_results = self._review_composite_insight(
            maintenance_trace_id=maintenance_trace_id,
            skills=skills,
            traces=traces,
            planner_validator=planner_validator,
            harness_profile=harness_profile,
            replay_composite=replay_composite,
        )
        for proposal, admitted_ref, source_ref in insight_results:
            review_id = str(proposal.proposed_patch.get("insight_review_id", ""))
            if review_id:
                reviewed_ids.append(review_id)
            if proposal.status == "admitted" and admitted_ref:
                admitted_assets.append((admitted_ref, "composite"))
                lineage.append({
                    "source_ref": admitted_ref,
                    "target_ref": source_ref,
                    "relation": GlobalRelationType.DERIVED_FROM.value,
                    "operation": "revise_composite_insight",
                    "proposal_id": proposal.proposal_id,
                    "review_id": review_id,
                })
            else:
                rejected_ids.append(proposal.proposal_id)

        for pending in list(self.repair_store.pending()):
            if pending.proposal_id in resolved_source_ids:
                pending.status = "admitted"
                pending.replay_result = {
                    "passed": True,
                    "resolved_by_batch": True,
                }
                self.repair_store.save(pending)
            elif (
                finalize_pending
                and pending.proposed_patch.get("requires_concrete_patch")
            ):
                pending.status = "rejected"
                pending.replay_result = {
                    "passed": False,
                    "failure_code": "concrete_patch_unavailable",
                }
                self.repair_store.save(pending)
                rejected_ids.append(pending.proposal_id)
            elif finalize_pending and pending.status == "proposed":
                pending.status = "rejected"
                pending.replay_result = {
                    "passed": False,
                    "failure_code": "batch_proposal_not_executable",
                }
                self.repair_store.save(pending)
                rejected_ids.append(pending.proposal_id)

        pending_ids = tuple(
            item.proposal_id for item in self.repair_store.pending()
        )
        return BatchMaintenanceResult(
            maintenance_trace_id=maintenance_trace_id,
            admitted_assets=tuple(dict.fromkeys(admitted_assets)),
            rejected_proposal_ids=tuple(dict.fromkeys(rejected_ids)),
            pending_proposal_ids=pending_ids,
            reviewed_ids=tuple(item for item in dict.fromkeys(reviewed_ids) if item),
            lineage=tuple(lineage),
        )

    @staticmethod
    def _validate_agent_proposal(
        proposal: EvolutionToolEditProposal,
        review: dict[str, Any],
    ) -> None:
        if proposal.operation not in set(review["eligible_operations"]):
            raise ValueError("evolution operation is not eligible for this review")
        if set(proposal.target_refs) != set(map(str, review["target_refs"])):
            raise ValueError("evolution target refs differ from review authority")
        if proposal.operation == "split":
            if len(proposal.candidates) < 2:
                raise ValueError("Tool split requires at least two candidates")
            suffixes = [item.logical_id_suffix for item in proposal.candidates]
            if any(not suffix for suffix in suffixes) or len(suffixes) != len(set(suffixes)):
                raise ValueError("Tool split candidates require unique logical id suffixes")
        elif len(proposal.candidates) != 1:
            raise ValueError("non-split Tool evolution requires exactly one candidate")

        source_tools = {
            str(item["ref"]): dict(item)
            for item in review.get("tools", [])
        }
        if set(source_tools) != set(map(str, review["target_refs"])):
            raise ValueError("batch review lacks authoritative Tool bodies")
        source_action_types = {
            str(step.get("action_type", ""))
            for item in source_tools.values()
            for step in dict(item.get("artifact") or {}).get("steps", [])
        }
        cases = {
            str(item["case_id"]): dict(item["case"])
            for item in review.get("source_cases", [])
        }
        if not cases:
            raise ValueError("Tool evolution requires authoritative replay cases")
        selected: dict[str, set[int]] = defaultdict(set)
        selection_counts: Counter[tuple[str, int]] = Counter()
        split_step_counts: Counter[int] = Counter()
        split_source_steps: list[dict[str, Any]] = []
        if proposal.operation == "split":
            if len(source_tools) != 1:
                raise ValueError("Tool split must have exactly one authoritative source")
            split_source_steps = list(
                dict(next(iter(source_tools.values())).get("artifact") or {}).get(
                    "steps", []
                )
            )
            if len(split_source_steps) < 2:
                raise ValueError("Tool split source has no reusable step boundaries")
        for candidate in proposal.candidates:
            if candidate.artifact_kind != "primitive_ir":
                raise ValueError("only primitive_ir Tool evolution is supported")
            candidate_steps = list(candidate.artifact.get("steps") or [])
            if not candidate_steps:
                raise ValueError("Tool evolution candidate has no executable steps")
            if any(
                not str(step.get("action_type", ""))
                or str(step.get("action_type", "")) not in source_action_types
                for step in candidate_steps
            ):
                raise ValueError("Tool candidate invents an action outside source authority")
            step_indexes = tuple(map(int, candidate.source_step_indexes))
            if len(step_indexes) != len(set(step_indexes)):
                raise ValueError("Tool candidate repeats a source step boundary")
            if proposal.operation == "split":
                if (
                    not step_indexes
                    or any(index < 0 or index >= len(split_source_steps) for index in step_indexes)
                ):
                    raise ValueError("Tool split step selection is outside source authority")
                if list(step_indexes) != sorted(step_indexes):
                    raise ValueError("Tool split step boundary must preserve source order")
                if any(
                    right != left + 1
                    for left, right in zip(step_indexes, step_indexes[1:])
                ):
                    raise ValueError("Tool split boundary must be one contiguous reusable span")
                authoritative_steps = [
                    split_source_steps[index] for index in step_indexes
                ]
                if to_primitive(candidate_steps) != to_primitive(authoritative_steps):
                    raise ValueError("Tool split artifact differs from its source step span")
                split_step_counts.update(step_indexes)
            elif step_indexes != tuple(range(len(candidate_steps))):
                raise ValueError(
                    "non-split candidate must declare every proposed artifact step"
                )
            seen: set[str] = set()
            for choice in candidate.source_cases:
                case_id = str(choice["case_id"])
                if case_id in seen or case_id not in cases:
                    raise ValueError("Tool candidate references an unknown/duplicate replay case")
                seen.add(case_id)
                effect_count = len(cases[case_id].get("effects") or [])
                indexes = set(map(int, choice["effect_indexes"]))
                if not indexes or any(index < 0 or index >= effect_count for index in indexes):
                    raise ValueError("Tool candidate effect selection is outside source authority")
                if proposal.operation != "split" and indexes != set(range(effect_count)):
                    raise ValueError("non-split evolution must replay every source effect")
                selected[case_id].update(indexes)
                selection_counts.update((case_id, index) for index in indexes)

        available_ids = set(cases)
        if proposal.operation in {"generalize", "merge", "specialize", "update"} and set(selected) != available_ids:
            raise ValueError("Tool evolution must replay every authoritative source case")
        if proposal.operation in {"specialize", "update"}:
            cluster = dict(review.get("stable_failure_cluster") or {})
            tasks = set(map(str, cluster.get("task_ids") or [])) - {""}
            traces = set(map(str, cluster.get("trace_ids") or [])) - {""}
            attempts = set(map(str, cluster.get("attempt_ids") or [])) - {""}
            intrinsic_codes = {
                "tool_primitive_rejected",
                "tool_execution_error",
                "tool_output_schema_error",
            }
            expected_key = content_hash({
                "failure_code": str(cluster.get("failure_code", "")),
                "input_semantic_types": dict(
                    cluster.get("input_semantic_types") or {}
                ),
                "harness_context": dict(cluster.get("harness_context") or {}),
                "parameter_constraints": list(
                    cluster.get("parameter_constraints") or []
                ),
            })
            if (
                len(set(review.get("failure_ids") or [])) < 2
                or len(tasks) < 2
                or len(traces) < 2
                or len(attempts) < 2
                or str(cluster.get("failure_code", "")) not in intrinsic_codes
                or str(cluster.get("cluster_key", "")) != expected_key
            ):
                raise ValueError("Tool specialization lacks a stable intrinsic subdomain")
        if proposal.operation == "split":
            if set(split_step_counts) != set(range(len(split_source_steps))) or any(
                count != 1 for count in split_step_counts.values()
            ):
                raise ValueError("Tool split step spans must be disjoint and exhaustive")
            if set(selected) != available_ids:
                raise ValueError("Tool split must cover every source replay case")
            for case_id, case in cases.items():
                if selected[case_id] != set(range(len(case.get("effects") or []))):
                    raise ValueError("Tool split does not preserve every source effect")
                if any(
                    selection_counts[(case_id, index)] != 1
                    for index in range(len(case.get("effects") or []))
                ):
                    raise ValueError("Tool split effect partitions overlap")

    def _duplicate_merge_edits(
        self, tools: Any,
    ) -> list[tuple[dict[str, Any], EvolutionToolEditProposal]]:
        usable = [
            item
            for item in tools.tools()
            if item.status in {
                ToolStatus.CANDIDATE,
                ToolStatus.ACTIVE,
                ToolStatus.PREFERRED,
            }
        ]
        groups: dict[str, list[ToolAsset]] = defaultdict(list)
        for tool in usable:
            groups[_tool_semantics(tool)].append(tool)
        reviewed = {
            str(item.proposed_patch.get("review_id"))
            for item in self.repair_store.history()
            if item.proposed_patch.get("review_id")
        }
        result = []
        for group in groups.values():
            logical_ids = {item.ref.tool_id for item in group}
            if len(group) < 2 or len(logical_ids) < 2:
                continue
            ordered = sorted(group, key=lambda item: str(item.ref))
            review = self._tool_review(
                "exact_duplicate",
                ordered,
                eligible_operations=["merge"],
            )
            if review["review_id"] in reviewed:
                continue
            primary = ordered[0]
            choices = tuple(
                {
                    "case_id": item["case_id"],
                    "effect_indexes": list(range(len(item["case"].get("effects") or []))),
                }
                for item in review["source_cases"]
            )
            candidate = EvolutionToolCandidateProposal(
                primary.summary,
                copy.deepcopy(primary.signature),
                copy.deepcopy(primary.interface),
                primary.artifact_kind,
                copy.deepcopy(primary.artifact),
                copy.deepcopy(primary.safety),
                choices,
                tuple(range(len(primary.artifact.get("steps") or []))),
                "",
            )
            response = EvolutionToolEditProposal(
                review["review_id"],
                "merge",
                tuple(review["target_refs"]),
                (candidate,),
                "exact behavior/interface/effect duplicate with union source replay",
            )
            self._validate_agent_proposal(response, review)
            result.append((review, response))
        return result

    def _process_tool_proposal(
        self,
        proposal: RepairProposal,
        review: dict[str, Any],
        *,
        tools: Any,
        admission: Any,
        replay_tool: Callable[[ToolAsset, dict[str, Any]], bool],
    ) -> tuple[RepairProposal, list[dict[str, str]]]:
        materialized: list[ToolAsset] = []
        admitted_candidates: list[ToolAsset] = []

        def replay(_proposal: RepairProposal) -> dict[str, Any]:
            nonlocal materialized, admitted_candidates
            materialized = self._materialize_tool_candidates(_proposal, review, tools)
            admitted_candidates = [
                admission.admit_tool(
                    candidate,
                    replay=lambda tool, case: bool(replay_tool(tool, case)),
                )
                for candidate in materialized
            ]
            failures = {
                str(item.ref): list(item.metadata.get("admission_failure") or [])
                for item in admitted_candidates
                if item.status is not ToolStatus.CANDIDATE
            }
            return {
                "passed": not failures and bool(admitted_candidates),
                "candidate_refs": [str(item.ref) for item in admitted_candidates],
                "admission_failures": failures,
            }

        def admit(_proposal: RepairProposal, _result: dict[str, Any]) -> bool:
            if any(item.status is not ToolStatus.CANDIDATE for item in admitted_candidates):
                return False
            for candidate in admitted_candidates:
                tools.register(candidate)
            _result["admitted_refs"] = [str(item.ref) for item in admitted_candidates]
            return True

        outcome = self.repair_store.replay_and_admit(proposal, replay, admit)
        if outcome.status != "admitted":
            return outcome, []
        operation = str(proposal.proposed_patch.get("evolution_operation"))
        relation = {
            "merge": GlobalRelationType.MERGED_FROM,
            "split": GlobalRelationType.SPLIT_FROM,
            "generalize": GlobalRelationType.DERIVED_FROM,
            "specialize": GlobalRelationType.DERIVED_FROM,
            "update": GlobalRelationType.DERIVED_FROM,
        }[operation]
        return outcome, [
            {
                "source_ref": str(candidate.ref),
                "target_ref": target_ref,
                "relation": relation.value,
                "operation": operation,
                "proposal_id": proposal.proposal_id,
                "review_id": str(proposal.proposed_patch.get("review_id", "")),
            }
            for candidate in admitted_candidates
            for target_ref in proposal.proposed_patch.get("target_refs", [])
        ]

    def _materialize_tool_candidates(
        self,
        proposal: RepairProposal,
        review: dict[str, Any],
        tools: Any,
    ) -> list[ToolAsset]:
        target_refs = [ToolRef.parse(item) for item in proposal.proposed_patch["target_refs"]]
        source_tools = [tools.get(ref) for ref in target_refs]
        cases = {
            str(item["case_id"]): copy.deepcopy(item["case"])
            for item in review["source_cases"]
        }
        specs = list(proposal.proposed_patch.get("candidate_specs") or [])
        operation = str(proposal.proposed_patch.get("evolution_operation"))
        result: list[ToolAsset] = []
        reserved: set[str] = set()
        for index, spec in enumerate(specs):
            selected_cases = []
            for choice in spec["source_cases"]:
                source = copy.deepcopy(cases[str(choice["case_id"])])
                effects = list(source.get("effects") or [])
                source["effects"] = [
                    effects[int(effect_index)]
                    for effect_index in choice["effect_indexes"]
                ]
                selected_cases.append(source)
            if operation == "split":
                suffix = re.sub(
                    r"[^A-Za-z0-9._-]+",
                    "_",
                    str(spec.get("logical_id_suffix", "")).strip(),
                ).strip("_")
                if not suffix:
                    raise ValueError("Tool split logical id suffix is empty after normalization")
                base = ToolRef(f"{target_refs[0].tool_id}_{suffix}", "1.0.0")
            else:
                base = ToolRef(target_refs[0].tool_id, target_refs[0].version)
            ref = _next_tool_ref(tools, base, reserved)
            reserved.add(str(ref))
            result.append(ToolAsset(
                ref=ref,
                summary=str(spec["summary"]),
                signature=copy.deepcopy(spec["signature"]),
                interface=copy.deepcopy(spec["interface"]),
                artifact_kind=str(spec["artifact_kind"]),
                artifact=copy.deepcopy(spec["artifact"]),
                tests=selected_cases,
                safety=copy.deepcopy(spec["safety"]),
                provenance={
                    "evolution_operation": operation,
                    "source_refs": [str(item.ref) for item in source_tools],
                    "review_id": proposal.proposed_patch["review_id"],
                    "source_step_indexes": list(spec["source_step_indexes"]),
                },
                metadata={
                    "batch_evolution": {
                        "operation": operation,
                        "source_refs": [str(item.ref) for item in source_tools],
                        "review_id": proposal.proposed_patch["review_id"],
                        "source_step_indexes": list(spec["source_step_indexes"]),
                    }
                },
                status=ToolStatus.ADMISSION_PENDING,
            ))
        return result

    def _review_composite_redundancy(
        self,
        *,
        maintenance_trace_id: str,
        skills: Any,
        projection: Any,
        traces: Any,
        planner_validator: Any,
        harness_profile: str,
        replay_composite: Callable[[CompositeSkill, dict[str, Any]], bool],
    ) -> list[tuple[RepairProposal, str, str]]:
        """Remove only occurrences proven terminal-skipped on every selected use."""
        reviewed = {
            str(item.proposed_patch.get("review_id"))
            for item in self.repair_store.history()
            if item.proposed_patch.get("review_id")
        }
        results: list[tuple[RepairProposal, str, str]] = []
        for composite in skills.composites():
            if composite.status is not SkillStatus.ACTIVE or len(composite.occurrences) <= 1:
                continue
            stats = projection.stats(str(composite.ref), "composite")
            for occurrence in composite.occurrences:
                counts = dict(stats.occurrence_stats.get(occurrence.occurrence_id) or {})
                selected = int(counts.get("selected", 0))
                skipped = int(counts.get("skipped_goal_terminal", 0))
                if selected < 2 or skipped != selected:
                    continue
                replay_rows = self.repair_store.database.rows(
                    "SELECT DISTINCT task_id,trace_id FROM evidence_events "
                    "WHERE artifact_ref=? AND occurrence_id=? AND event_type=? "
                    "ORDER BY task_id,trace_id",
                    (
                        str(composite.ref),
                        occurrence.occurrence_id,
                        "goal_terminal_skipped",
                    ),
                )
                if len({str(row["task_id"]) for row in replay_rows}) < 2:
                    continue
                replay_trace_ids = [str(row["trace_id"]) for row in replay_rows]
                review_core = {
                    "source_ref": str(composite.ref),
                    "remove_occurrence_id": occurrence.occurrence_id,
                    "replay_trace_ids": replay_trace_ids,
                }
                review_id = "review_" + content_hash(review_core)[:24]
                if review_id in reviewed:
                    continue

                removed_step = occurrence.step_id
                candidate_occurrences = [
                    item for item in composite.occurrences
                    if item.occurrence_id != occurrence.occurrence_id
                ]
                candidate = replace(
                    composite,
                    ref=_next_skill_ref(skills, composite.ref, "composite"),
                    occurrences=candidate_occurrences,
                    control_sequence=[
                        step for step in composite.control_sequence if step != removed_step
                    ],
                    data_edges=[
                        edge for edge in composite.data_edges
                        if removed_step not in {edge.source_step, edge.target_step}
                    ],
                    dependency_edges=[
                        edge for edge in composite.dependency_edges
                        if removed_step not in {edge.source_step, edge.target_step}
                    ],
                    metadata={
                        **composite.metadata,
                        "batch_evolution": {
                            "operation": "remove_redundant_occurrence",
                            "source_ref": str(composite.ref),
                            "removed_occurrence_id": occurrence.occurrence_id,
                            "review_id": review_id,
                            "source_trace_ids": replay_trace_ids,
                        },
                    },
                    status=SkillStatus.CANDIDATE,
                )
                replay_cases: list[dict[str, Any]] = []
                replay_task_ids: set[str] = set()
                replay_case_trace_ids: set[str] = set()
                for trace_id in replay_trace_ids:
                    payload = traces.load_payload(trace_id)
                    case = _composite_fresh_replay_case(
                        payload,
                        candidate,
                        skills=skills,
                        expected_source_ref=str(composite.ref),
                    )
                    if case is None:
                        continue
                    task_id = str(case["source_task"]["task_id"])
                    if (task_id, trace_id) in {
                        (
                            str(item["source_task"]["task_id"]),
                            str(item["trace_id"]),
                        )
                        for item in replay_cases
                    }:
                        continue
                    case["trace_id"] = trace_id
                    replay_cases.append(case)
                    replay_task_ids.add(task_id)
                    replay_case_trace_ids.add(trace_id)
                if len(replay_task_ids) < 2 or len(replay_case_trace_ids) < 2:
                    continue
                proposal = RepairProposal.create(
                    str(composite.ref),
                    "composite",
                    "remove_redundant_occurrence",
                    {
                        "review_id": review_id,
                        "replacement_candidate": to_primitive(candidate),
                        "removed_occurrence_id": occurrence.occurrence_id,
                        "required_replay_trace_ids": replay_trace_ids,
                        "source_cases": to_primitive(replay_cases),
                        "requires_concrete_patch": False,
                    },
                    [],
                )
                self.repair_store.save(proposal)
                validation = planner_validator.validate(
                    _composite_plan(candidate),
                    mode=RuntimeMode.ONLINE,
                    harness_profile=harness_profile,
                )

                def replay(
                    _proposal: RepairProposal,
                    *,
                    cases: list[dict[str, Any]] = replay_cases,
                    validation_passed: bool = validation.passed,
                ) -> dict[str, Any]:
                    replay_checks = [
                        bool(replay_composite(candidate, dict(case)))
                        for case in cases
                    ]
                    return {
                        "passed": bool(
                            validation_passed
                            and len({
                                str(item["source_task"]["task_id"])
                                for item in cases
                            }) >= 2
                            and len({str(item["trace_id"]) for item in cases}) >= 2
                            and replay_checks
                            and all(replay_checks)
                        ),
                        "planner_validation": to_primitive(validation),
                        "source_replays": [
                            {
                                "trace_id": str(case["trace_id"]),
                                "passed": passed,
                                "fresh_harness": True,
                            }
                            for case, passed in zip(
                                cases, replay_checks, strict=True,
                            )
                        ],
                        "maintenance_trace_id": maintenance_trace_id,
                    }

                admitted_ref = ""

                def admit(_proposal: RepairProposal, result: dict[str, Any]) -> bool:
                    nonlocal admitted_ref
                    skills.register_composite(candidate)
                    admitted_ref = str(candidate.ref)
                    result["admitted_ref"] = admitted_ref
                    return True

                outcome = self.repair_store.replay_and_admit(proposal, replay, admit)
                results.append((outcome, admitted_ref, str(composite.ref)))
        return results

    def _review_composite_insight(
        self,
        *,
        maintenance_trace_id: str,
        skills: Any,
        traces: Any,
        planner_validator: Any,
        harness_profile: str,
        replay_composite: Callable[[CompositeSkill, dict[str, Any]], bool],
    ) -> list[tuple[RepairProposal, str, str]]:
        """Aggregate durable section 25.7 facts into an immutable hint revision.

        The EvidenceLedger is recall/provenance authority and the corresponding
        structured Trace is semantic authority.  A fact is included only when
        the same canonical fact is supported by at least two independent tasks
        and traces.  Binding values and free-form failure text are never copied
        into Planner context.
        """
        reviewed = {
            str(item.proposed_patch.get("insight_review_id", ""))
            for item in self.repair_store.history()
            if item.proposed_patch.get("insight_review_id")
        }
        eligible_events = {
            "self_sufficient_success",
            "task_rescue_required",
            "goal_terminal_skipped",
            "contract_mismatch",
        }
        results: list[tuple[RepairProposal, str, str]] = []
        for composite in skills.composites():
            if composite.status is not SkillStatus.ACTIVE:
                continue
            rows = self.repair_store.database.rows(
                "SELECT event_id,task_id,trace_id,occurrence_id,event_type "
                "FROM evidence_events WHERE artifact_ref=? "
                "ORDER BY event_type,occurrence_id,task_id,trace_id,event_id",
                (str(composite.ref),),
            )
            rows_by_trace: dict[str, list[Any]] = defaultdict(list)
            for row in rows:
                event_type = str(row["event_type"])
                if event_type in eligible_events:
                    rows_by_trace[str(row["trace_id"])].append(row)
            if len(rows_by_trace) < 2:
                continue

            payload_by_trace: dict[str, dict[str, Any]] = {}
            for trace_id, trace_rows in sorted(rows_by_trace.items()):
                payload = traces.load_payload(trace_id)
                payload_trace_id = str(payload.get("trace_id", ""))
                payload_task_id = str(
                    dict(payload.get("task") or {}).get("task_id", "")
                )
                ledger_task_ids = {str(row["task_id"]) for row in trace_rows}
                if (
                    payload_trace_id != trace_id
                    or len(ledger_task_ids) != 1
                    or payload_task_id not in ledger_task_ids
                ):
                    raise ValueError(
                        "Composite insight evidence/Trace identity mismatch"
                    )
                source_ref = str(
                    dict(payload.get("runtime_plan") or {}).get(
                        "source_composite_ref", "",
                    )
                )
                if source_ref != str(composite.ref):
                    raise ValueError(
                        "Composite insight Trace does not name its ledger artifact"
                    )
                payload_by_trace[trace_id] = payload

            aggregate = _stable_composite_insight_aggregate(
                str(composite.ref), rows_by_trace, payload_by_trace,
            )
            if not any(aggregate[name] for name in _COMPOSITE_INSIGHT_CATEGORIES):
                continue
            semantic_aggregate = _composite_insight_semantics(aggregate)
            review_core = {
                "source_ref": str(composite.ref),
                "aggregate": semantic_aggregate,
            }
            review_id = "composite_insight_review_" + content_hash(review_core)[:24]
            if review_id in reviewed:
                continue
            existing_aggregate = dict(
                composite.insight.get("evidence_aggregate") or {}
            )
            if (
                existing_aggregate
                and content_hash(_composite_insight_semantics(existing_aggregate))
                == content_hash(semantic_aggregate)
            ):
                continue
            candidate = replace(
                composite,
                ref=_next_skill_ref(skills, composite.ref, "composite"),
                insight={**composite.insight, "evidence_aggregate": aggregate},
                metadata={
                    **composite.metadata,
                    "batch_evolution": {
                        "operation": "revise_composite_insight",
                        "kind": "composite_insight",
                        "source_ref": str(composite.ref),
                        "review_id": review_id,
                        "maintenance_trace_id": maintenance_trace_id,
                    },
                },
                status=SkillStatus.CANDIDATE,
            )
            replay_cases = []
            seen_task_trace: set[tuple[str, str]] = set()
            supporting_trace_ids = sorted({
                str(trace_id)
                for name in _COMPOSITE_INSIGHT_CATEGORIES
                for fact in aggregate[name]
                for trace_id in fact["support_trace_ids"]
            })
            for trace_id in supporting_trace_ids:
                payload = payload_by_trace[trace_id]
                if not bool(payload.get("benchmark_success", False)):
                    continue
                case = _composite_fresh_replay_case(
                    payload,
                    candidate,
                    skills=skills,
                    expected_source_ref=str(composite.ref),
                )
                if case is None:
                    continue
                key = (str(case["source_task"]["task_id"]), trace_id)
                if key in seen_task_trace:
                    continue
                seen_task_trace.add(key)
                case["trace_id"] = trace_id
                replay_cases.append(case)
            if (
                len({item["source_task"]["task_id"] for item in replay_cases}) < 2
                or len({item["trace_id"] for item in replay_cases}) < 2
            ):
                continue
            proposal = RepairProposal.create(
                str(composite.ref),
                "composite",
                "revise_composite_insight",
                {
                    "insight_review_id": review_id,
                    "replacement_candidate": to_primitive(candidate),
                    "support_event_ids": sorted({
                        str(event_id)
                        for name in _COMPOSITE_INSIGHT_CATEGORIES
                        for fact in aggregate[name]
                        for event_id in fact["support_event_ids"]
                    }),
                    "source_cases": to_primitive(replay_cases),
                    "requires_concrete_patch": False,
                },
                [],
            )
            self.repair_store.save(proposal)
            validation = planner_validator.validate(
                _composite_plan(candidate),
                mode=RuntimeMode.ONLINE,
                harness_profile=harness_profile,
            )

            def replay(
                _proposal: RepairProposal,
                *,
                cases: list[dict[str, Any]] = replay_cases,
                validation_passed: bool = validation.passed,
            ) -> dict[str, Any]:
                checks = [
                    bool(replay_composite(candidate, dict(case)))
                    for case in cases
                ]
                return {
                    "passed": bool(validation_passed and checks and all(checks)),
                    "fresh_harness_replays": [
                        {
                            "trace_id": str(case["trace_id"]),
                            "passed": passed,
                        }
                        for case, passed in zip(cases, checks, strict=True)
                    ],
                    "planner_validation": to_primitive(validation),
                    "maintenance_trace_id": maintenance_trace_id,
                }

            admitted_ref = ""

            def admit(_proposal: RepairProposal, result: dict[str, Any]) -> bool:
                nonlocal admitted_ref
                skills.register_composite(candidate)
                admitted_ref = str(candidate.ref)
                result["admitted_ref"] = admitted_ref
                return True

            outcome = self.repair_store.replay_and_admit(proposal, replay, admit)
            results.append((outcome, admitted_ref, str(composite.ref)))
        return results


_COMPOSITE_INSIGHT_CATEGORIES = (
    "parameter_resolution_strategies",
    "frequent_failure_modes",
    "effective_node_orders",
    "redundant_occurrences",
    "implementation_applicability",
)


def _stable_composite_insight_aggregate(
    source_ref: str,
    rows_by_trace: dict[str, list[Any]],
    payload_by_trace: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Return only facts with >=2 independent task and Trace witnesses."""
    facts: dict[str, dict[str, dict[str, Any]]] = {
        name: {} for name in _COMPOSITE_INSIGHT_CATEGORIES
    }
    for trace_id, payload in sorted(payload_by_trace.items()):
        task_id = str(dict(payload.get("task") or {}).get("task_id", ""))
        trace_rows = rows_by_trace[trace_id]
        for category, fact in _composite_insight_trace_facts(payload):
            relevant_rows = _composite_insight_fact_rows(
                category, fact, trace_rows,
            )
            if not relevant_rows:
                continue
            fact_id = content_hash(fact)
            record = facts[category].setdefault(fact_id, {
                "fact": fact,
                "supports": {},
            })
            support = record["supports"].setdefault(
                (task_id, trace_id), set(),
            )
            support.update(str(row["event_id"]) for row in relevant_rows)

    aggregate: dict[str, Any] = {
        "schema": "composite_insight.v2",
        "source_ref": source_ref,
    }
    for category in _COMPOSITE_INSIGHT_CATEGORIES:
        accepted: list[dict[str, Any]] = []
        for _, record in sorted(facts[category].items()):
            supports = dict(record["supports"])
            task_ids = sorted({task_id for task_id, _ in supports})
            trace_ids = sorted({trace_id for _, trace_id in supports})
            if len(task_ids) < 2 or len(trace_ids) < 2:
                continue
            accepted.append({
                **dict(record["fact"]),
                "support_task_ids": task_ids,
                "support_trace_ids": trace_ids,
                "support_event_ids": sorted({
                    event_id
                    for event_ids in supports.values()
                    for event_id in event_ids
                }),
            })
        aggregate[category] = accepted
    return aggregate


def _composite_insight_trace_facts(
    payload: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    facts: list[tuple[str, dict[str, Any]]] = []
    for change in payload.get("binding_changes") or []:
        change = dict(change)
        binding = dict(change.get("current") or change.get("previous") or {})
        fact = {
            "role": str(change.get("role") or binding.get("role") or ""),
            "source": _enum_text(binding.get("source")),
            "resolution": _enum_text(binding.get("resolution")),
        }
        if all(fact.values()):
            facts.append(("parameter_resolution_strategies", fact))

    for failure in payload.get("failures") or []:
        failure = dict(failure)
        fact = {
            "layer": _enum_text(failure.get("layer")),
            "code": str(failure.get("code", "")),
            "occurrence_id": str(failure.get("occurrence_id", "")),
        }
        if all(fact.values()):
            facts.append(("frequent_failure_modes", fact))

    runtime_plan = dict(payload.get("runtime_plan") or {})
    control_sequence = [
        str(step_id) for step_id in runtime_plan.get("control_sequence") or []
    ]
    node_records = [dict(node) for node in payload.get("node_records") or []]
    nodes_by_step = {
        str(node.get("step_id", "")): node for node in node_records
        if str(node.get("step_id", ""))
    }
    if (
        bool(payload.get("graph_self_sufficient_success", False))
        and control_sequence
        and all(step_id in nodes_by_step for step_id in control_sequence)
    ):
        facts.append(("effective_node_orders", {
            "control_sequence": control_sequence,
            "node_outcomes": [
                {
                    "step_id": step_id,
                    "occurrence_id": str(
                        nodes_by_step[step_id].get("occurrence_id", "")
                    ),
                    "outcome": _enum_text(
                        nodes_by_step[step_id].get("status")
                    ),
                }
                for step_id in control_sequence
            ],
        }))

    for node in node_records:
        if _enum_text(node.get("status")) != "skipped_goal_terminal":
            continue
        occurrence_id = str(node.get("occurrence_id", ""))
        if occurrence_id:
            facts.append(("redundant_occurrences", {
                "occurrence_id": occurrence_id,
                "node_outcome": "skipped_goal_terminal",
            }))

    task_type = str(dict(payload.get("task") or {}).get("task_type", ""))
    for invocation in payload.get("implementation_invocations") or []:
        invocation = dict(invocation)
        result = dict(invocation.get("result") or {})
        fact = {
            "impl_ref": str(invocation.get("implementation_ref", "")),
            "occurrence_id": str(invocation.get("occurrence_id", "")),
            "task_type": task_type,
            "started": bool(result.get("started", False)),
            "success": bool(
                result.get("completed", False)
                and result.get("atomic_effect_passed", False)
            ),
        }
        if fact["impl_ref"] and fact["occurrence_id"] and fact["task_type"]:
            facts.append(("implementation_applicability", fact))
    return facts


def _composite_insight_fact_rows(
    category: str,
    fact: dict[str, Any],
    rows: list[Any],
) -> list[Any]:
    if category == "effective_node_orders":
        allowed = {"self_sufficient_success"}
    elif category == "redundant_occurrences":
        return [
            row for row in rows
            if str(row["event_type"]) == "goal_terminal_skipped"
            and str(row["occurrence_id"]) == str(fact["occurrence_id"])
        ]
    elif category == "frequent_failure_modes":
        allowed = {"task_rescue_required", "contract_mismatch"}
    else:
        allowed = {
            "self_sufficient_success", "task_rescue_required",
            "goal_terminal_skipped", "contract_mismatch",
        }
    return [row for row in rows if str(row["event_type"]) in allowed]


def _composite_insight_semantics(aggregate: dict[str, Any]) -> dict[str, Any]:
    """Strip provenance so one additional Trace cannot mutate stable insight."""
    return {
        "schema": str(aggregate.get("schema", "")),
        **{
            category: [
                {
                    key: value for key, value in dict(fact).items()
                    if not key.startswith("support_")
                }
                for fact in aggregate.get(category) or []
            ]
            for category in _COMPOSITE_INSIGHT_CATEGORIES
        },
    }


def _enum_text(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


def _source_replays(tool: ToolAsset) -> list[dict[str, Any]]:
    return [
        copy.deepcopy(item)
        for item in tool.tests
        if item.get("kind") == "source_replay"
    ]


def _atomic_alignment_contract(atomic: AbstractAtomicSkill) -> str:
    """Code-authoritative §25.1 merge gate (embedding is recall-only)."""
    return content_hash({
        "inputs": atomic.inputs,
        "outputs": atomic.outputs,
        "preconditions": atomic.preconditions,
        "effects": atomic.effects,
        "validator_spec": atomic.validator_spec,
        "failure_modes": atomic.failure_modes,
        "atomic_boundary": atomic.guideline,
    })


def _atomic_merge_evidence(
    cohort: list[AbstractAtomicSkill],
    *,
    cluster_key: str,
    skills: Any,
    tools: Any,
    traces: Any,
) -> list[RepairEvidence]:
    """Return persisted source cases that can be replayed in a fresh episode."""
    evidence: dict[tuple[str, str], RepairEvidence] = {}
    cohort_refs = {str(item.ref) for item in cohort}
    for implementation in skills.implementations():
        if (
            str(implementation.abstract_ref) not in cohort_refs
            or implementation.status not in {SkillStatus.CANDIDATE, SkillStatus.ACTIVE}
            or len(implementation.tool_bindings) != 1
        ):
            continue
        binding = implementation.tool_bindings[0]
        try:
            tool = tools.get(binding.tool_ref)
        except KeyError:
            continue
        if tool.status not in {
            ToolStatus.CANDIDATE, ToolStatus.ACTIVE, ToolStatus.PREFERRED,
        }:
            continue
        for source_case in _source_replays(tool):
            trace_id = str(source_case.get("trace_id", ""))
            task = dict(source_case.get("source_task") or {})
            task_id = str(task.get("task_id", ""))
            bindings = dict(source_case.get("bindings") or {})
            if not trace_id or not task_id or not bindings:
                continue
            try:
                payload = traces.load_payload(trace_id)
            except KeyError:
                continue
            persisted_task = dict(payload.get("task") or {})
            if (
                str(payload.get("trace_id", "")) != trace_id
                or str(persisted_task.get("task_id", "")) != task_id
                or payload.get("benchmark_success") is not True
            ):
                continue
            task = {
                "task_id": task_id,
                "goal": str(task.get("goal") or persisted_task.get("goal") or ""),
                "benchmark": str(
                    task.get("benchmark") or persisted_task.get("benchmark") or ""
                ),
                "task_type": str(
                    task.get("task_type") or persisted_task.get("task_type") or ""
                ),
                "context": dict(task.get("context") or {}),
                "metadata": dict(task.get("metadata") or persisted_task.get("metadata") or {}),
            }
            if not all(task.get(key) for key in ("task_id", "goal", "benchmark", "task_type")):
                continue
            core = {
                "trace_id": trace_id,
                "task_id": task_id,
                "atomic_ref": str(implementation.abstract_ref),
                "implementation_ref": str(implementation.ref),
                "tool_ref": str(tool.ref),
                "bindings": bindings,
                "prefix": list(source_case.get("prefix") or []),
            }
            item = RepairEvidence(
                evidence_id="atomic_merge_support_" + content_hash(core)[:24],
                task_id=task_id,
                trace_id=trace_id,
                cluster_key=cluster_key,
                replay_case={
                    "kind": "atomic_merge_source_replay",
                    "target_layer": "atomic",
                    "target_ref": str(implementation.abstract_ref),
                    "source_task": task,
                    "occurrence_id": str(
                        tool.provenance.get("occurrence_id") or "atomic_merge"
                    ),
                    "bindings": bindings,
                    "prefix": list(source_case.get("prefix") or []),
                    "occurrence_actions": [],
                    "tool_ref": str(tool.ref),
                },
                failure_layer="atomic",
                failure_code="atomic_contract_equivalent_support",
            )
            evidence.setdefault((task_id, trace_id), item)
    return sorted(evidence.values(), key=lambda item: item.evidence_id)


def _composite_sequence_evidence(
    payload: dict[str, Any],
    failure: dict[str, Any],
    *,
    source: CompositeSkill,
    skills: Any,
    harness_profile: str,
) -> RepairEvidence | None:
    layer = str(failure.get("layer", ""))
    code = str(failure.get("code", ""))
    if (
        layer not in {"data_flow", "composite", "task_contract"}
        or code not in {
            "data_flow_error",
            "composite_self_sufficiency_failure",
            "task_contract_mismatch",
            "benchmark_goal_contract_mismatch",
        }
    ):
        return None
    task = dict(payload.get("task") or {})
    task_id = str(failure.get("task_id") or task.get("task_id") or "")
    trace_id = str(failure.get("trace_id") or payload.get("trace_id") or "")
    failure_id = str(failure.get("failure_id") or "")
    if not task_id or not trace_id or not failure_id:
        return None
    if str(payload.get("trace_id") or "") != trace_id:
        return None
    plan = dict(payload.get("runtime_plan") or {})
    if str(plan.get("source_composite_ref") or "") != str(source.ref):
        return None

    invocations = [dict(item) for item in payload.get("implementation_invocations", [])]
    occurrence_cases: dict[str, dict[str, Any]] = {}
    for occurrence in source.occurrences:
        matching = [
            item for item in invocations
            if str(item.get("occurrence_id") or "") == occurrence.occurrence_id
            and isinstance(item.get("arguments"), dict)
            and dict(item.get("arguments") or {})
        ]
        selected = None
        for item in reversed(matching):
            try:
                implementation = skills.get_implementation(
                    str(item.get("implementation_ref") or "")
                )
            except KeyError:
                continue
            if implementation.abstract_ref == occurrence.node_ref:
                selected = item
                break
        if selected is None:
            return None
        occurrence_cases[occurrence.occurrence_id] = {
            "implementation_ref": str(selected["implementation_ref"]),
            "bindings": dict(selected["arguments"]),
        }

    task_metadata = dict(task.get("metadata") or {})
    task_context = dict(task_metadata.get("context") or {})
    if task_metadata.get("env_index") is not None:
        task_context.setdefault("env_index", task_metadata.get("env_index"))
    if task_metadata.get("game_file"):
        task_context.setdefault("game_file", task_metadata.get("game_file"))
    source_task = {
        "task_id": task_id,
        "goal": str(task.get("goal") or ""),
        "benchmark": str(task.get("benchmark") or ""),
        "task_type": str(task.get("task_type") or ""),
        "context": task_context,
        "metadata": task_metadata,
    }
    if not all(source_task.get(key) for key in ("task_id", "goal", "benchmark", "task_type")):
        return None
    cluster_key = content_hash({
        "source_ref": str(source.ref),
        "source_non_order_semantics": {
            "occurrences": source.occurrences,
            "data_edges": source.data_edges,
            "goal_contract": source.goal_contract,
            "validator_spec": source.validator_spec,
        },
        "failure_code": code,
        "harness_context": {
            "profile": harness_profile,
            "task_type": source_task["task_type"],
            "split": str(source_task["metadata"].get("split", "")),
        },
    })
    return RepairEvidence(
        evidence_id=failure_id,
        task_id=task_id,
        trace_id=trace_id,
        cluster_key=cluster_key,
        replay_case={
            "kind": "composite_sequence_trace_replay",
            "target_layer": "composite",
            "target_ref": str(source.ref),
            "source_task": source_task,
            "prefix": [],
            "occurrence_cases": occurrence_cases,
        },
        failure_layer=layer,
        failure_code=code,
    )


def _composite_fresh_replay_case(
    payload: dict[str, Any],
    candidate: CompositeSkill,
    *,
    skills: Any,
    expected_source_ref: str,
) -> dict[str, Any] | None:
    """Reconstruct executable bindings; never consume stored pass booleans."""
    plan = dict(payload.get("runtime_plan") or {})
    if str(plan.get("source_composite_ref") or "") != expected_source_ref:
        return None
    task = dict(payload.get("task") or {})
    task_id = str(task.get("task_id") or "")
    if not task_id:
        return None
    invocations = [dict(item) for item in payload.get("implementation_invocations", [])]
    occurrence_cases: dict[str, dict[str, Any]] = {}
    for occurrence in candidate.occurrences:
        selected = None
        for item in reversed(invocations):
            if (
                str(item.get("occurrence_id") or "") != occurrence.occurrence_id
                or not isinstance(item.get("arguments"), dict)
                or not dict(item.get("arguments") or {})
            ):
                continue
            try:
                implementation = skills.get_implementation(
                    str(item.get("implementation_ref") or "")
                )
            except KeyError:
                continue
            if implementation.abstract_ref == occurrence.node_ref:
                selected = item
                break
        if selected is None:
            return None
        occurrence_cases[occurrence.occurrence_id] = {
            "implementation_ref": str(selected["implementation_ref"]),
            "bindings": dict(selected["arguments"]),
        }
    metadata = dict(task.get("metadata") or {})
    context = dict(metadata.get("context") or {})
    if metadata.get("env_index") is not None:
        context.setdefault("env_index", metadata.get("env_index"))
    if metadata.get("game_file"):
        context.setdefault("game_file", metadata.get("game_file"))
    source_task = {
        "task_id": task_id,
        "goal": str(task.get("goal") or ""),
        "benchmark": str(task.get("benchmark") or ""),
        "task_type": str(task.get("task_type") or ""),
        "context": context,
        "metadata": metadata,
    }
    if not all(source_task.get(key) for key in ("goal", "benchmark", "task_type")):
        return None
    return {
        "kind": "composite_fresh_trace_replay",
        "target_layer": "composite",
        "target_ref": expected_source_ref,
        "source_task": source_task,
        "prefix": [],
        "occurrence_cases": occurrence_cases,
    }


def _case_catalog(tools: list[ToolAsset]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for tool in tools:
        for case in _source_replays(tool):
            case_id = "case_" + content_hash(case)[:24]
            entry = result.setdefault(case_id, {"case": case, "target_refs": set()})
            if content_hash(entry["case"]) != content_hash(case):
                raise ValueError("source replay case hash collision")
            entry["target_refs"].add(str(tool.ref))
    return result


def _tool_semantics(tool: ToolAsset) -> str:
    effect_contracts = sorted({
        content_hash(case.get("effects") or [])
        for case in _source_replays(tool)
    })
    return content_hash({
        "signature": tool.signature,
        "interface": tool.interface,
        "artifact_kind": tool.artifact_kind,
        "artifact": tool.artifact,
        "safety": tool.safety,
        "effects": effect_contracts,
    })


def _tool_shape(tool: ToolAsset) -> str:
    steps = []
    for step in tool.artifact.get("steps") or []:
        mapping = {}
        for role, raw in dict(step.get("argument_mapping") or {}).items():
            value = to_primitive(raw)
            mapping[str(role)] = {
                "kind": value.get("kind") if isinstance(value, dict) else type(value).__name__,
                "source_role": value.get("source_role") if isinstance(value, dict) else "",
            }
        steps.append({"action_type": step.get("action_type"), "mapping": mapping})
    effects = sorted({
        content_hash({
            "predicate": str(effect.get("predicate", "")),
            "argument_roles": sorted(dict(effect.get("args") or {})),
            "cardinality": int(effect.get("cardinality", 1)),
            "distinct_by": str(effect.get("distinct_by", "")),
        })
        for case in _source_replays(tool)
        for effect in map(to_primitive, case.get("effects") or [])
    })
    return content_hash({"steps": steps, "effects": effects})


def _version_key(version: str) -> tuple[int, int, int]:
    try:
        return tuple(int(piece) for piece in version.split("."))  # type: ignore[return-value]
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"semantic version required: {version!r}") from exc


def _next_tool_ref(tools: Any, base: ToolRef, reserved: set[str]) -> ToolRef:
    versions = [
        item.version
        for item in tools.list_refs()
        if item.tool_id == base.tool_id
    ]
    if not versions and str(base) not in reserved:
        return base
    version = bump_version(max([*versions, base.version], key=_version_key))
    candidate = ToolRef(base.tool_id, version)
    while str(candidate) in reserved:
        candidate = ToolRef(candidate.tool_id, bump_version(candidate.version))
    return candidate


def _next_skill_ref(skills: Any, base: SkillRef, kind: str) -> SkillRef:
    versions = [
        item.version
        for item in skills.list_refs(kind)
        if item.logical_id == base.logical_id
    ]
    if not versions:
        return base
    return SkillRef(base.logical_id, bump_version(max(versions, key=_version_key)))


def _composite_plan(composite: CompositeSkill) -> RuntimeLinearPlan:
    occurrences = []
    for item in composite.occurrences:
        occurrences.append(RuntimeOccurrence(
            step_id=item.step_id,
            occurrence_id=item.occurrence_id,
            node_ref=item.node_ref,
            requirement_ids=[],
            binding_specs=dict(item.binding_specs),
            implementation_candidates=[],
            expected_effects=[],
        ))
    return RuntimeLinearPlan(
        task_id="batch_maintenance_replay",
        source="stored_composite",
        source_composite_ref=str(composite.ref),
        occurrences=occurrences,
        control_sequence=list(composite.control_sequence),
        data_edges=list(composite.data_edges),
        dependency_edges=list(composite.dependency_edges),
        task_contract=composite.goal_contract,
        planner_audit={},
    )
