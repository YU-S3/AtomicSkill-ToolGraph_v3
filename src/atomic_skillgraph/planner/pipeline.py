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
from .cold_start_agent import ColdStartPlanner
from .cold_start_validator import ColdStartPlanValidator
from .compiler import PlanCompiler
from .composite_retriever import CompositeRetriever
from .related_composite import RelatedCompositeHintFinder
from .repairability import RepairabilityGate
from .requirement_agent import RequirementAgent
from .support_retriever import PlannerSupportAtomicRetriever
from .multiplicity import (
    RequirementBundleValidator,
    RequirementMultiplicityCompiler,
    normalize_task_contract,
)
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
        max_occurrences: int = 16, max_repeat_count: int = 4,
        candidate_policy: Any | None = None,
        cold_start_enabled: bool = False,
        provisional_retriever: Any | None = None,
        failure_experience_retriever: Any | None = None,
        cold_start_session_factory: Callable[[Any, Any], Any] | None = None,
        scaffold_max_steps: int = 8,
        cold_start_repair_limit: int = 1,
    ) -> None:
        self.skills, self.graph, self.session_factory = skills, graph, session_factory
        self.composite_retriever = CompositeRetriever(
            skills, top_k=composite_top_k, candidate_policy=candidate_policy
        )
        self.atomic_retriever = AtomicRetriever(
            skills, top_k=atomic_top_k, max_top_k=max_atomic_top_k,
            candidate_policy=candidate_policy,
        )
        self.support_retriever = PlannerSupportAtomicRetriever(
            skills,
            top_k=atomic_top_k,
            candidate_policy=candidate_policy,
        )
        self.related = RelatedCompositeHintFinder(skills)
        self.compiler = PlanCompiler(skills)
        self.validator = PlannerValidator(skills, graph, max_occurrences=max_occurrences)
        self.requirement_validator = RequirementBundleValidator()
        self.repairability_gate = RepairabilityGate()
        self.multiplicity_compiler = RequirementMultiplicityCompiler()
        self.max_occurrences = int(max_occurrences)
        self.max_repeat_count = int(max_repeat_count)
        self.cold_start_enabled = bool(cold_start_enabled)
        self.provisional_retriever = provisional_retriever
        self.failure_experience_retriever = failure_experience_retriever
        self.cold_start_session_factory = cold_start_session_factory or session_factory
        self.cold_start_validator = ColdStartPlanValidator()
        self.scaffold_max_steps = int(scaffold_max_steps)
        if int(cold_start_repair_limit) != 1:
            raise ValueError("v3.1 permits exactly one C1R cold-start repair")

    def build_plan(
        self, task: Any, harness: Any, *, mode: RuntimeMode | str = RuntimeMode.ONLINE,
        initial_observation: str = "",
    ) -> RuntimeLinearPlan:
        mode = RuntimeMode(mode)
        contract = normalize_task_contract(harness.task_contract(task))
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
            audit.composite_rejections.append({
                "composite_ref": str(composite.ref),
                "stage": "plan_validation",
                "reasons": list(report.failure_codes),
                "checks": dict(report.checks),
            })

        terminal_retrieval = self.composite_retriever.retrieve_terminal(
            task, contract, mode=mode, harness_profile=harness.profile_name,
        )
        audit.terminal_empirical_candidates = (
            terminal_retrieval.terminal_empirical_audit
        )
        for composite in terminal_retrieval.terminal_empirical_candidates:
            provisional_audit = to_primitive(audit)
            provisional_audit["selected_composite"] = str(composite.ref)
            provisional_audit["final_outcome"] = "stored_composite"
            provisional_audit["selected_composite_authority"] = {
                "kind": "terminal_empirical",
                "candidate_status": composite.status.value,
                "source_ref": str(composite.ref),
            }
            plan = self.compiler.from_composite(
                task, contract, composite, mode=mode,
                audit=provisional_audit,
            )
            report = self.validator.validate(
                plan, mode=mode, harness_profile=harness.profile_name,
            )
            if report.passed:
                return plan
            audit.composite_rejections.append({
                "composite_ref": str(composite.ref),
                "stage": "terminal_empirical_plan_validation",
                "reasons": list(report.failure_codes),
                "checks": dict(report.checks),
            })

        # Frozen and explicitly cold-start-disabled compatibility runs retain
        # the old empty-bank shortcut.  v3.1 online cold start must still run
        # P1 so C1 and a later Failure Extractor have high-level authority.
        if (
            not self.skills.list_refs("composite", mode=mode)
            and not self.skills.list_refs("atomic", mode=mode)
            and (mode is RuntimeMode.FROZEN or not self.cold_start_enabled)
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
            bundle = requirement_agent.propose(task, contract, initial_observation, harness.profile_name)
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
        audit.requirements_p1 = to_primitive(bundle)
        validation = self.requirement_validator.validate(
            bundle, contract,
            max_repeat_count=self.max_repeat_count,
            max_runtime_occurrences=self.max_occurrences,
        )
        audit.requirement_validation_p1 = to_primitive(validation)
        hints: list[dict[str, Any]] = []
        search = None
        repaired = False
        if validation.passed:
            expansion = self.multiplicity_compiler.expand(bundle, contract)
            search = self.atomic_retriever.retrieve_multiplicity(
                expansion, mode=mode, harness_profile=harness.profile_name,
                task_id=task.task_id,
            )
            audit.atomic_search_p1 = to_primitive(search.template_results)
            if not search.full_coverage:
                hints = self.related.find(search, mode=mode)
                audit.related_composite_hints = hints
        repairability = None
        if not validation.passed:
            repairability = self.repairability_gate.decide(
                bundle, validation, (), (),
            )
            audit.repairability = to_primitive(repairability)
        elif search is not None and not search.full_coverage:
            repairability = self.repairability_gate.decide(
                bundle, validation, search.template_results, hints,
            )
            audit.repairability = to_primitive(repairability)
        if (
            not validation.passed
            and repairability is not None
            and repairability.repairable
        ) or (
            search is not None
            and not search.full_coverage
            and (repairability is None or repairability.repairable)
        ):
            try:
                bundle = requirement_agent.repair(
                    task, contract, bundle,
                    [] if search is None else search.template_results,
                    hints, validation=validation,
                )
                repaired = True
                audit.requirements_p1r = to_primitive(bundle)
            except Exception as exc:
                if not _is_planner_content_failure(exc):
                    raise
                audit.final_outcome = "full_dynamic"
                audit.fallback_reason = _planner_failure_reason(
                    exc, "planner_requirement_repair_failed",
                )
                return RuntimeLinearPlan.full_dynamic(task.task_id, contract, reason=audit.fallback_reason, audit=to_primitive(audit))
        validation = self.requirement_validator.validate(
            bundle, contract,
            max_repeat_count=self.max_repeat_count,
            max_runtime_occurrences=self.max_occurrences,
        )
        audit.requirement_validation_final = to_primitive(validation)
        if not validation.passed:
            audit.final_outcome = "full_dynamic"
            audit.fallback_reason = "planner_requirement_multiplicity_invalid"
            return RuntimeLinearPlan.full_dynamic(task.task_id, contract, reason=audit.fallback_reason, audit=to_primitive(audit))

        expansion = self.multiplicity_compiler.expand(bundle, contract)
        audit.requirement_expansion = to_primitive(expansion)
        search = self.atomic_retriever.retrieve_multiplicity(
            expansion, mode=mode, harness_profile=harness.profile_name,
            task_id=task.task_id,
        )
        if repaired:
            audit.atomic_search_p1r = to_primitive(search.template_results)

        if not search.full_coverage:
            if mode is RuntimeMode.FROZEN or not self.cold_start_enabled:
                audit.final_outcome = "full_dynamic"
                audit.fallback_reason = "planner_requirement_uncovered"
                return RuntimeLinearPlan.full_dynamic(
                    task.task_id, contract, reason=audit.fallback_reason,
                    audit=to_primitive(audit),
                )
            if self.provisional_retriever is None or self.failure_experience_retriever is None:
                raise RuntimeError("online cold start is enabled but failure-side retrievers are not constructed")
            provisional = self.provisional_retriever.retrieve(
                search.missing_instances,
                harness_profile=harness.profile_name,
            )
            experiences = self.failure_experience_retriever.retrieve(
                contract, expansion, harness_profile=harness.profile_name,
            )
            audit.cold_start_retrieval = {
                "missing_instance_ids": [item.instance_id for item in search.missing_instances],
                "provisional_candidates": to_primitive(provisional),
                "failure_experiences": to_primitive(experiences),
            }
            c1_session = self.cold_start_session_factory(task, contract)
            cold_agent = ColdStartPlanner(c1_session)
            verified_candidates = search.instance_candidates
            try:
                cold_proposal = cold_agent.propose(
                    task=task,
                    task_contract=contract,
                    requirement_expansion=expansion,
                    verified_candidates=verified_candidates,
                    provisional_candidates=provisional,
                    failure_experiences=experiences,
                    observation=initial_observation,
                )
                audit.cold_start_plan = to_primitive(cold_proposal)
                verified_refs = {
                    key: {str(item.atomic_ref) for item in values}
                    for key, values in verified_candidates.items()
                }
                provisional_refs = {
                    key: {str(item.provisional_ref) for item in values}
                    for key, values in provisional.items()
                }
                candidate_roles: dict[str, set[str]] = {}
                candidate_required_inputs: dict[str, set[str]] = {}
                candidate_runtime_resolvable_roles: dict[str, set[str]] = {}
                candidate_output_roles: dict[str, set[str]] = {}
                for ref in search.refs:
                    atomic = self.skills.get_atomic(ref)
                    candidate_roles[str(ref)] = {
                        item.name for item in (*atomic.inputs, *atomic.outputs)
                    }
                    candidate_required_inputs[str(ref)] = {
                        item.name for item in atomic.inputs if item.required
                    }
                    candidate_runtime_resolvable_roles[str(ref)] = {
                        item.name
                        for item in atomic.inputs
                        if item.runtime_resolvable
                    }
                    candidate_output_roles[str(ref)] = {
                        item.name for item in atomic.outputs
                    }
                for values in provisional.values():
                    for item in values:
                        inputs = [
                            value
                            for value in item.atomic_contract.get("inputs", ())
                            if isinstance(value, dict)
                        ]
                        outputs = [
                            value
                            for value in item.atomic_contract.get("outputs", ())
                            if isinstance(value, dict)
                        ]
                        candidate_roles[item.provisional_ref] = {
                            str(value.get("name", ""))
                            for value in (*inputs, *outputs)
                            if str(value.get("name", ""))
                        }
                        candidate_required_inputs[item.provisional_ref] = {
                            str(value.get("name", ""))
                            for value in inputs
                            if bool(value.get("required", True))
                            and str(value.get("name", ""))
                        }
                        candidate_runtime_resolvable_roles[
                            item.provisional_ref
                        ] = {
                            str(value.get("name", ""))
                            for value in inputs
                            if bool(value.get("runtime_resolvable", False))
                            and str(value.get("name", ""))
                        }
                        candidate_output_roles[item.provisional_ref] = {
                            str(value.get("name", ""))
                            for value in outputs
                            if str(value.get("name", ""))
                        }
                task_roles = {
                    str(role)
                    for source in (
                        task.context.get("semantic_bindings", {}),
                        task.context.get("goal_roles", {}),
                    )
                    if isinstance(source, dict)
                    for role in source
                }
                cold_validation = self.cold_start_validator.validate(
                    cold_proposal, expansion,
                    verified_candidates=verified_refs,
                    provisional_candidates=provisional_refs,
                    failure_experience_ids={item.experience_id for item in experiences},
                    candidate_roles=candidate_roles,
                    candidate_required_inputs=candidate_required_inputs,
                    candidate_runtime_resolvable_roles=(
                        candidate_runtime_resolvable_roles
                    ),
                    candidate_output_roles=candidate_output_roles,
                    task_roles=task_roles,
                    scaffold_max_steps=self.scaffold_max_steps,
                )
                audit.cold_start_validation = to_primitive(cold_validation)
                if not cold_validation.passed:
                    cold_proposal = cold_agent.repair(
                        cold_proposal, cold_validation,
                        requirement_expansion=expansion,
                        verified_candidates=verified_candidates,
                        provisional_candidates=provisional,
                        failure_experiences=experiences,
                    )
                    audit.cold_start_repair = to_primitive(cold_proposal)
                    cold_validation = self.cold_start_validator.validate(
                        cold_proposal, expansion,
                        verified_candidates=verified_refs,
                        provisional_candidates=provisional_refs,
                        failure_experience_ids={item.experience_id for item in experiences},
                        candidate_roles=candidate_roles,
                        candidate_required_inputs=candidate_required_inputs,
                        candidate_runtime_resolvable_roles=(
                            candidate_runtime_resolvable_roles
                        ),
                        candidate_output_roles=candidate_output_roles,
                        task_roles=task_roles,
                        scaffold_max_steps=self.scaffold_max_steps,
                    )
                    audit.cold_start_repair_validation = to_primitive(cold_validation)
            except Exception as exc:
                if not _is_planner_content_failure(exc):
                    raise
                audit.final_outcome = "full_dynamic"
                audit.fallback_reason = "cold_start_plan_invalid"
                return RuntimeLinearPlan.full_dynamic(
                    task.task_id, contract, reason=audit.fallback_reason,
                    audit=to_primitive(audit),
                )
            if not cold_validation.passed:
                audit.final_outcome = "full_dynamic"
                audit.fallback_reason = "cold_start_plan_invalid"
                return RuntimeLinearPlan.full_dynamic(
                    task.task_id, contract, reason=audit.fallback_reason,
                    audit=to_primitive(audit),
                )
            scaffold = self.cold_start_validator.scaffold(
                cold_proposal,
                candidate_roles=candidate_roles,
                candidate_required_inputs=candidate_required_inputs,
                candidate_runtime_resolvable_roles=(
                    candidate_runtime_resolvable_roles
                ),
                candidate_output_roles=candidate_output_roles,
                task_roles=task_roles,
            )
            if not scaffold.executable_step_ids:
                audit.final_outcome = "full_dynamic"
                audit.fallback_reason = "cold_start_executable_prefix_empty"
                plan = RuntimeLinearPlan.full_dynamic(
                    task.task_id, contract, reason=audit.fallback_reason,
                    audit=to_primitive(audit),
                )
                plan.cold_start_plan = cold_proposal
                plan.cold_start_scaffold = to_primitive(scaffold)
                return plan
            plan = RuntimeLinearPlan.cold_start(
                task.task_id, contract, proposal=cold_proposal,
                scaffold=to_primitive(scaffold), audit=to_primitive(audit),
            )
            plan.repeat_constraints = (
                self.compiler.repeat_compiler.from_requirement_expansion(
                cold_proposal, expansion,
                )
            )
            return plan

        support_candidates = self.support_retriever.retrieve(
            required_instance_candidates=search.instance_candidates,
            mode=mode,
            harness_profile=harness.profile_name,
            task_id=task.task_id,
        )
        audit.support_atomic_candidates = to_primitive(support_candidates)
        audit.planner_support_atomic_candidate_count = len(support_candidates)
        # Preserve the retrieval-produced order while deduplicating only the
        # interface projection.  This does not re-rank, truncate, or otherwise
        # change either required or support candidate pools.
        ordered_supplied_refs: list[str] = []
        seen_supplied_refs: set[str] = set()
        for candidates in search.instance_candidates.values():
            for candidate in candidates:
                ref = str(candidate.atomic_ref)
                if ref not in seen_supplied_refs:
                    ordered_supplied_refs.append(ref)
                    seen_supplied_refs.add(ref)
        for candidate in support_candidates:
            ref = str(candidate.atomic_ref)
            if ref not in seen_supplied_refs:
                ordered_supplied_refs.append(ref)
                seen_supplied_refs.add(ref)
        supplied_refs = set(ordered_supplied_refs)
        authoritative = [
            self.skills.get_atomic(ref) for ref in ordered_supplied_refs
        ]
        existing_edges = self.graph.existing_edges(supplied_refs, mode=mode)
        instance_candidates = {
            instance_id: {str(candidate.atomic_ref) for candidate in candidates}
            for instance_id, candidates in search.instance_candidates.items()
        }
        try:
            proposal = workflow_agent.propose(
                task,
                contract,
                expansion,
                search.candidates,
                existing_edges,
                hints,
                support_candidates=support_candidates,
                authoritative_contracts=authoritative,
            )
            audit.workflow_p2 = to_primitive(proposal)
            _require_supplied_atomic_refs(proposal, supplied_refs)
            plan = self.compiler.compile(
                proposal, task, contract, mode=mode, audit=to_primitive(audit),
                expansion=expansion,
            )
            required_ids = [
                item.instance_id for item in expansion.instances if item.requirement.required
            ]
            report = self.validator.validate(
                plan, mode=mode, required_requirement_ids=required_ids, harness_profile=harness.profile_name,
                expansion=expansion, instance_candidates=instance_candidates,
                support_candidates=support_candidates,
            )
            audit.validation_p2 = to_primitive(report)
            if not report.passed:
                proposal = workflow_agent.repair(
                    proposal,
                    report,
                    authoritative,
                    existing_edges,
                    support_candidates=support_candidates,
                )
                audit.workflow_p2r = to_primitive(proposal)
                _require_supplied_atomic_refs(proposal, supplied_refs)
                plan = self.compiler.compile(
                    proposal, task, contract, mode=mode, audit=to_primitive(audit),
                    expansion=expansion,
                )
                report = self.validator.validate(
                    plan, mode=mode, required_requirement_ids=required_ids, harness_profile=harness.profile_name,
                    expansion=expansion, instance_candidates=instance_candidates,
                    support_candidates=support_candidates,
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
        audit.support_atomic_selected = [
            {
                "step_id": step.step_id,
                "occurrence_id": step.occurrence_id,
                "atomic_ref": str(step.node_ref),
                "data_flow_edges": [
                    to_primitive(edge)
                    for edge in proposal.data_edges
                    if edge.source_step == step.step_id
                ],
            }
            for step in proposal.steps
            if not (step.requirement_instance_ids or step.requirement_ids)
        ]
        audit.planner_support_atomic_selected_count = len(
            audit.support_atomic_selected
        )
        audit.final_outcome = "atomic_composition"
        plan.planner_audit = to_primitive(audit) | {
            "requirement_coverage": plan.planner_audit.get("requirement_coverage", {}),
            "sequence_origin": "planner_proposed_sequence",
        }
        return plan
