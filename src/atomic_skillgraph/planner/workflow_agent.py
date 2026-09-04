"""P2 and the single permitted P2R turn in the same PlannerSession."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from ..agents.structured_submission import (
    PROPOSED_EDGE_SCHEMA,
    PROPOSED_OCCURRENCE_SCHEMA,
    StructuredSubmissionClient,
)
from ..core.bindings import BindingExpression
from ..core.contracts import PlannerWorkflowProposal, ProposedEdge, ProposedOccurrence
from ..core.errors import AgentProtocolError, FailureLayer, PlannerProposalError
from ..core.refs import SkillRef
from ..core.serialization import to_primitive


WORKFLOW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["steps", "control_sequence", "data_edges", "dependency_edges", "requirement_coverage"],
    "additionalProperties": False,
    "properties": {
        "steps": {"type": "array", "minItems": 1, "items": PROPOSED_OCCURRENCE_SCHEMA},
        "control_sequence": {
            "type": "array", "minItems": 1, "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "data_edges": {"type": "array", "items": PROPOSED_EDGE_SCHEMA},
        "dependency_edges": {"type": "array", "items": PROPOSED_EDGE_SCHEMA},
        "requirement_coverage": {
            "type": "object",
            "additionalProperties": {
                "type": "array", "minItems": 1, "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
        },
    },
}


def _json_payload(value: Any) -> str:
    """Encode one deterministic compact Planner policy projection."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _requirement_expansion_projection(value: Any) -> Any:
    """Remove the template copy embedded in every RequirementInstance.

    ``RequirementExpansion`` deliberately carries the complete requirement on
    both its template and every materialized instance for deterministic code
    consumers.  P2 only needs the contract once plus the materialization
    identity of each instance.  Replaying every copy is especially costly for
    repeat blocks and adds no Planner authority.
    """

    templates = getattr(value, "templates", None)
    instances = getattr(value, "instances", None)
    repeat_blocks = getattr(value, "repeat_blocks", None)
    if templates is None or instances is None or repeat_blocks is None:
        return to_primitive(value)
    return {
        "templates": to_primitive(list(templates)),
        "repeat_blocks": to_primitive(list(repeat_blocks)),
        "instances": [
            {
                "instance_id": str(item.instance_id),
                "template_requirement_id": str(item.template_requirement_id),
                "repeat_block_id": str(item.repeat_block_id),
                "repeat_index": int(item.repeat_index),
            }
            for item in instances
        ],
        "instance_ids_by_template": {
            str(key): [str(item) for item in items]
            for key, items in dict(
                getattr(value, "instance_ids_by_template", {}) or {}
            ).items()
        },
    }


def _candidate_projection(value: Any) -> Any:
    """Project the supplied required pool without changing its membership."""

    primitive = to_primitive(value)
    if not isinstance(primitive, Mapping):
        return primitive
    projected: dict[str, Any] = {}
    for instance_id, raw_candidates in primitive.items():
        if not isinstance(raw_candidates, list):
            projected[str(instance_id)] = raw_candidates
            continue
        projected[str(instance_id)] = [
            {
                key: candidate[key]
                for key in ("atomic_ref", "score", "reasons", "contract_match")
                if isinstance(candidate, Mapping) and key in candidate
            }
            if isinstance(candidate, Mapping)
            else candidate
            for candidate in raw_candidates
        ]
    return projected


def _support_candidate_projection(values: Any) -> Any:
    """Bound P2 support context to the frozen formal candidate interface.

    This is a field projection, not another retrieval/ranking pass: candidate
    count, order, score, output/effect summaries, and every formal role mapping
    remain unchanged.
    """

    primitive = to_primitive(values)
    if not isinstance(primitive, list):
        return primitive
    candidate_fields = (
        "atomic_ref",
        "consumer_requirement_instance_id",
        "score",
        "role_mappings",
        "output_roles",
        "effect_predicates",
    )
    mapping_fields = (
        "producer_role",
        "consumer_role",
        "semantic_type",
        "producer_resolution",
        "required_resolution",
        "effect_domain",
        "consumer_atomic_ref",
    )
    projected: list[Any] = []
    for candidate in primitive:
        if not isinstance(candidate, Mapping):
            projected.append(candidate)
            continue
        item = {
            key: candidate[key]
            for key in candidate_fields
            if key in candidate
        }
        mappings = item.get("role_mappings")
        if isinstance(mappings, list):
            item["role_mappings"] = [
                {
                    key: mapping[key]
                    for key in mapping_fields
                    if isinstance(mapping, Mapping) and key in mapping
                }
                if isinstance(mapping, Mapping)
                else mapping
                for mapping in mappings
            ]
        projected.append(item)
    return projected


def _parameter_projection(value: Any) -> dict[str, Any]:
    primitive = to_primitive(value)
    if not isinstance(primitive, Mapping):
        return {}
    return {
        key: primitive[key]
        for key in (
            "name",
            "semantic_type",
            "required",
            "runtime_resolvable",
            "required_resolution",
        )
        if key in primitive
    }


def _authoritative_contract_projection(values: Any) -> Any:
    """Expose bounded Atomic interface authority to P2/P2R.

    The projection deliberately keeps the fields that determine graph
    compatibility and drops lifecycle, implementation, provenance, guideline,
    and validator bulk.  Callers retain candidate membership and ordering in
    the separate required/support candidate payloads; this interface list is
    not a second retrieval or ranking result.
    """

    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Iterable):
        return to_primitive(values)
    projected: list[Any] = []
    for atomic in values:
        if not hasattr(atomic, "ref"):
            projected.append(to_primitive(atomic))
            continue
        projected.append({
            "ref": str(atomic.ref),
            "summary": str(getattr(atomic, "summary", "")),
            "inputs": [
                _parameter_projection(item)
                for item in list(getattr(atomic, "inputs", ()) or ())
            ],
            "outputs": [
                _parameter_projection(item)
                for item in list(getattr(atomic, "outputs", ()) or ())
            ],
            "preconditions": to_primitive(
                list(getattr(atomic, "preconditions", ()) or ())
            ),
            "effects": to_primitive(list(getattr(atomic, "effects", ()) or ())),
        })
    return projected


def _validation_projection(value: Any) -> Any:
    """Keep only deterministic P2 repair findings used to change the graph."""

    if not hasattr(value, "passed"):
        return to_primitive(value)
    return {
        "level": str(getattr(value, "level", "")),
        "passed": bool(value.passed),
        "checks": to_primitive(dict(getattr(value, "checks", {}) or {})),
        "failure_codes": [
            str(item) for item in list(getattr(value, "failure_codes", ()) or ())
        ],
        "messages": [
            str(item) for item in list(getattr(value, "messages", ()) or ())
        ],
    }


def _proposal(payload: dict[str, Any]) -> PlannerWorkflowProposal:
    steps = []
    for item in payload.get("steps", []):
        steps.append(ProposedOccurrence(
            step_id=str(item["step_id"]), occurrence_id=str(item.get("occurrence_id", item["step_id"])),
            node_ref=SkillRef.parse(item["node_ref"] if isinstance(item["node_ref"], str) else SkillRef.from_dict(item["node_ref"])),
            requirement_ids=[str(value) for value in item.get("requirement_instance_ids", [])],
            binding_specs={name: BindingExpression.from_dict(value) for name, value in item.get("binding_specs", {}).items()},
            expected_effects=[],
            requirement_instance_ids=[str(value) for value in item.get("requirement_instance_ids", [])],
            repeat_role_bindings={str(key): str(value) for key, value in item.get("repeat_role_bindings", {}).items()},
        ))
    return PlannerWorkflowProposal(
        steps=steps, control_sequence=[str(item) for item in payload.get("control_sequence", [])],
        data_edges=[ProposedEdge(**item) for item in payload.get("data_edges", [])],
        dependency_edges=[ProposedEdge(**item) for item in payload.get("dependency_edges", [])],
        requirement_coverage={str(key): [str(item) for item in value] for key, value in payload.get("requirement_coverage", {}).items()},
        sequence_origin="planner_proposed_sequence",
    )


class WorkflowAgent:
    def __init__(self, session: Any) -> None:
        self.session = session
        self.submissions = StructuredSubmissionClient()

    def propose(
        self,
        task: Any,
        contract: Any,
        requirements: Any,
        candidates: Any,
        existing_edges: Any,
        hints: Any,
        *,
        support_candidates: Any = (),
        authoritative_contracts: Any = (),
    ) -> PlannerWorkflowProposal:
        compact = getattr(self.session, "compact_completed_structured_phases", None)
        if callable(compact):
            compact()
        if hasattr(self.session, "set_usage_bucket"):
            self.session.set_usage_bucket("planner_p2")
        prompt = (
            "P2 LINEAR WORKFLOW. Select only supplied Atomic candidates. Produce exactly one complete control "
            "sequence. Data/dependency edges may fan in/out but must point forward. Mark every new edge "
            "origin=planner_proposed and every reused edge origin=existing_active with its real id. "
            "Call only the offered submit tool.\n"
            "The same supplied Atomic ref may appear in multiple steps. Every step and\n"
            "occurrence_id must remain unique.\n\n"
            "Cover every RequirementInstance, not only every requirement template.\n"
            "A supplied support Atomic is an ordinary graph node whose "
            "requirement_instance_ids must be empty. Include it only when at "
            "least one supplied formal producer-to-consumer role mapping is "
            "materialized as a forward data_flow edge into a downstream required "
            "occurrence. Do not add unused support nodes or invent support role "
            "mappings.\n"
            "For RepeatBlocks, preserve the declared serial order within each iteration.\n"
            "Use repeat_role_bindings to map block roles to the selected Atomic roles.\n"
            "Do not invent additional repetitions or collapse distinct instances into one\n"
            "occurrence.\n"
            "The Requirements payload carries each complete template contract once; "
            "instances reference it by template_requirement_id.\n"
            "Every listed support role mapping has already passed deterministic type, "
            "resolution, and Effect-domain compatibility checks.\n"
            "Authoritative candidate interfaces contain the exact code-side inputs, "
            "outputs, preconditions, and effects for every supplied required/support "
            "Atomic ref. Use them to determine graph bindings and input/output "
            "compatibility; do not invent roles or predicates. The interface list is "
            "not another ranking and does not change candidate membership/order.\n"
            f"Task: {task.goal}\nContract: {_json_payload(to_primitive(contract))}\n"
            f"Requirements: {_json_payload(_requirement_expansion_projection(requirements))}\n"
            f"Required candidates: {_json_payload(_candidate_projection(candidates))}\n"
            f"Support candidates and formal mappings: "
            f"{_json_payload(_support_candidate_projection(support_candidates))}\n"
            f"Authoritative candidate interfaces: "
            f"{_json_payload(_authoritative_contract_projection(authoritative_contracts))}\n"
            f"Existing edges: {_json_payload(to_primitive(existing_edges))}\n"
            f"Related hints: {_json_payload(to_primitive(hints))}"
        )
        return self._request_workflow(
            prompt,
            error_code="planner_graph_invalid",
            description="Submit one complete linear Planner workflow proposal.",
        )

    def repair(
        self,
        proposal: PlannerWorkflowProposal,
        validation: Any,
        authoritative_contracts: Any,
        existing_edges: Any,
        *,
        support_candidates: Any = (),
    ) -> PlannerWorkflowProposal:
        compact = getattr(self.session, "compact_completed_structured_phases", None)
        if callable(compact):
            compact()
        if hasattr(self.session, "set_usage_bucket"):
            self.session.set_usage_bucket("planner_p2_repair")
        prompt = (
            "P2R GRAPH REPAIR. Return a complete replacement proposal. You may change occurrence selection/order, "
            "planner-proposed edges, delete bad edges, or repeat a supplied Atomic. Do not invent or rewrite skills, "
            "forge existing edges, or add another control path. Call only the offered submit tool.\n"
            "Authoritative contracts are bounded interface projections; omitted lifecycle, "
            "guideline, and provenance fields grant no Planner authority.\n"
            f"Proposal: {_json_payload(to_primitive(proposal))}\n"
            f"Validation errors: {_json_payload(_validation_projection(validation))}\n"
            f"Authoritative contracts: "
            f"{_json_payload(_authoritative_contract_projection(authoritative_contracts))}\n"
            f"Support candidates and formal mappings: "
            f"{_json_payload(_support_candidate_projection(support_candidates))}\n"
            f"Existing edges: {_json_payload(to_primitive(existing_edges))}"
        )
        return self._request_workflow(
            prompt,
            error_code="planner_graph_repair_failed",
            description="Submit one complete replacement Planner workflow proposal.",
        )

    def _request_workflow(
        self,
        prompt: str,
        *,
        error_code: str,
        description: str,
    ) -> PlannerWorkflowProposal:
        try:
            submission = self.submissions.request(
                self.session,
                prompt=prompt,
                tool_name="submit_planner_workflow",
                description=description,
                schema=WORKFLOW_SCHEMA,
            )
        except AgentProtocolError as exc:
            raise PlannerProposalError(
                error_code,
                f"Planner workflow proposal protocol validation failed: {exc}",
                layer=FailureLayer.PLANNER_GRAPH,
            ) from exc
        try:
            return _proposal(submission.value)
        except (KeyError, TypeError, ValueError) as exc:
            raise PlannerProposalError(
                error_code,
                f"Planner workflow proposal semantic validation failed: {exc}",
                layer=FailureLayer.PLANNER_GRAPH,
            ) from exc
