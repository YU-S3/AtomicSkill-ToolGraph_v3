"""The frozen complete-Composite → Atomic composition → Full Dynamic route."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable

from ..core.contracts import PlannerAudit
from ..core.errors import (
    AgentProtocolError,
    AtomicSkillGraphError,
    FailureLayer,
    PlannerCoverageError,
    PlannerGraphValidationError,
    PlannerProposalError,
)
from ..core.results import RuntimeLinearPlan
from ..core.serialization import to_primitive
from ..core.status import RuntimeMode
from ..knowledge.graph_store import GraphStore
from ..knowledge.skill_registry import SkillRegistry
from .atomic_retriever import AtomicRetriever
from .compiler import PlanCompiler
from .composite_retriever import CompositeRetriever
from .related_composite import RelatedCompositeHintFinder
from .requirement_agent import RequirementAgent
from .validator import PlannerValidator
from .workflow_agent import WorkflowAgent


def _is_planner_content_failure(exc: Exception) -> bool:
    """Allow Dynamic fallback only for explicitly attributed Planner content.

    Bare Python exceptions are programming/data-integrity failures, not
    evidence of a capability gap.  They must propagate to the runner instead
    of being silently converted into Full Dynamic.
    """

    return isinstance(
        exc,
        (
            AgentProtocolError,
            PlannerProposalError,
            PlannerCoverageError,
            PlannerGraphValidationError,
        ),
    )


def _planner_failure_reason(exc: Exception, fallback: str) -> str:
    if isinstance(exc, AtomicSkillGraphError) and exc.code:
        return exc.code
    return fallback


def _require_supplied_atomic_refs(
    proposal: Any,
    supplied_refs: set[str],
) -> None:
    unknown = sorted({
        str(step.node_ref)
        for step in proposal.steps
        if str(step.node_ref) not in supplied_refs
    })
    if unknown:
        raise PlannerGraphValidationError(
            "planner_graph_invalid",
            "Planner workflow references Atomic refs not supplied by retrieval: "
            + ", ".join(unknown),
            layer=FailureLayer.PLANNER_GRAPH,
        )


class PlannerPipeline:
    def __init__(
        self, skills: SkillRegistry, graph: GraphStore, session_factory: Callable[[Any, Any], Any],
        *, composite_top_k: int = 5, atomic_top_k: int = 3, max_atomic_top_k: int = 5,
        max_occurrences: int = 16, candidate_policy: Any | None = None,
    ) -> None:
        self.skills, self.graph, self.session_factory = skills, graph, session_factory
        self.composite_retriever = CompositeRetriever(
            skills, top_k=composite_top_k, candidate_policy=candidate_policy
        )
        self.atomic_retriever = AtomicRetriever(
            skills, top_k=atomic_top_k, max_top_k=max_atomic_top_k,
            candidate_policy=candidate_policy,
        )
        self.related = RelatedCompositeHintFinder(skills)
        self.compiler = PlanCompiler(skills)
        self.validator = PlannerValidator(skills, graph, max_occurrences=max_occurrences)

    def build_plan(
        self, task: Any, harness: Any, *, mode: RuntimeMode | str = RuntimeMode.ONLINE,
        initial_observation: str = "",
    ) -> RuntimeLinearPlan:
        mode = RuntimeMode(mode)
        contract = harness.task_contract(task)
        audit = PlannerAudit()

        p0 = self.composite_retriever.retrieve_complete(
            task, contract, mode=mode, harness_profile=harness.profile_name,
        )
        audit.composite_candidates = p0.audit_candidates
        audit.composite_rejections = p0.rejections
        for composite in p0.candidates:
            provisional_audit = to_primitive(audit)
            provisional_audit["selected_composite"] = str(composite.ref)
            provisional_audit["final_outcome"] = "stored_composite"
            plan = self.compiler.from_composite(task, contract, composite, mode=mode, audit=provisional_audit)
            report = self.validator.validate(plan, mode=mode, harness_profile=harness.profile_name)
            if report.passed:
                return plan
            audit.composite_rejections.append({"composite_ref": str(composite.ref), "reasons": report.failure_codes})

        # With no usable Composite and no usable Atomic there is no retrieval
        # evidence for P1/P1R to inspect.  The formal protocol routes directly
        # to Dynamic and records why; later tasks stop taking this shortcut as
        # soon as successful extraction admits online-usable Candidates.
        if (
            not self.skills.list_refs("composite", mode=mode)
            and not self.skills.list_refs("atomic", mode=mode)
        ):
            audit.final_outcome = "full_dynamic"
            audit.fallback_reason = "empty_knowledge_bank"
            return RuntimeLinearPlan.full_dynamic(
                task.task_id,
                contract,
                reason=audit.fallback_reason,
                audit=to_primitive(audit),
            )

        session = self.session_factory(task, contract)
        requirement_agent = RequirementAgent(session)
        workflow_agent = WorkflowAgent(session)
        try:
            requirements = requirement_agent.propose(task, contract, initial_observation, harness.profile_name)
        except Exception as exc:
            if not _is_planner_content_failure(exc):
                raise
            audit.final_outcome = "full_dynamic"
            audit.fallback_reason = _planner_failure_reason(
                exc, f"planner_requirement_invalid:{type(exc).__name__}",
            )
            return RuntimeLinearPlan.full_dynamic(
                task.task_id, contract, reason=audit.fallback_reason,
                audit=to_primitive(audit),
            )
        audit.requirements_p1 = to_primitive(requirements)
        search = self.atomic_retriever.retrieve(
            requirements, mode=mode, harness_profile=harness.profile_name, task_id=task.task_id
        )
        audit.atomic_search_p1 = to_primitive(search.results)
        hints: list[dict[str, Any]] = []
        if not search.full_coverage:
            hints = self.related.find(search, mode=mode)
            audit.related_composite_hints = hints
            try:
                requirements = requirement_agent.repair(task, contract, requirements, search.results, hints)
                audit.requirements_p1r = to_primitive(requirements)
                search = self.atomic_retriever.retrieve(
                    requirements, mode=mode, harness_profile=harness.profile_name, task_id=task.task_id
                )
                audit.atomic_search_p1r = to_primitive(search.results)
            except Exception as exc:
                if not _is_planner_content_failure(exc):
                    raise
                audit.final_outcome = "full_dynamic"
                audit.fallback_reason = _planner_failure_reason(
                    exc, "planner_requirement_repair_failed",
                )
                return RuntimeLinearPlan.full_dynamic(task.task_id, contract, reason=audit.fallback_reason, audit=to_primitive(audit))
        if not search.full_coverage:
            audit.final_outcome = "full_dynamic"
            audit.fallback_reason = "planner_requirement_uncovered"
            return RuntimeLinearPlan.full_dynamic(task.task_id, contract, reason=audit.fallback_reason, audit=to_primitive(audit))

        existing_edges = self.graph.existing_edges(search.refs, mode=mode)
        requirement_candidates = {
            result.requirement.requirement_id: {
                str(candidate.atomic_ref) for candidate in result.candidates
            }
            for result in search.results
        }
        try:
            proposal = workflow_agent.propose(task, contract, requirements, search.candidates, existing_edges, hints)
            audit.workflow_p2 = to_primitive(proposal)
            supplied_refs = {str(ref) for ref in search.refs}
            _require_supplied_atomic_refs(proposal, supplied_refs)
            plan = self.compiler.compile(proposal, task, contract, mode=mode, audit=to_primitive(audit))
            required_ids = [item.requirement_id for item in requirements if item.required]
            report = self.validator.validate(
                plan, mode=mode, required_requirement_ids=required_ids, harness_profile=harness.profile_name,
                requirement_candidates=requirement_candidates,
            )
            audit.validation_p2 = to_primitive(report)
            if not report.passed:
                authoritative = [self.skills.get_atomic(ref) for ref in search.refs]
                proposal = workflow_agent.repair(proposal, report, authoritative, existing_edges)
                audit.workflow_p2r = to_primitive(proposal)
                _require_supplied_atomic_refs(proposal, supplied_refs)
                plan = self.compiler.compile(proposal, task, contract, mode=mode, audit=to_primitive(audit))
                report = self.validator.validate(
                    plan, mode=mode, required_requirement_ids=required_ids, harness_profile=harness.profile_name,
                    requirement_candidates=requirement_candidates,
                )
                audit.validation_p2r = to_primitive(report)
            if not report.passed:
                audit.final_outcome = "full_dynamic"
                audit.fallback_reason = "planner_graph_repair_failed"
                return RuntimeLinearPlan.full_dynamic(task.task_id, contract, reason=audit.fallback_reason, audit=to_primitive(audit))
        except Exception as exc:
            if not _is_planner_content_failure(exc):
                raise
            audit.final_outcome = "full_dynamic"
            audit.fallback_reason = _planner_failure_reason(
                exc, "planner_graph_repair_failed",
            )
            return RuntimeLinearPlan.full_dynamic(task.task_id, contract, reason=audit.fallback_reason, audit=to_primitive(audit))
        audit.final_outcome = "atomic_composition"
        plan.planner_audit = to_primitive(audit) | {
            "requirement_coverage": plan.planner_audit.get("requirement_coverage", {}),
            "sequence_origin": "planner_proposed_sequence",
        }
        return plan
