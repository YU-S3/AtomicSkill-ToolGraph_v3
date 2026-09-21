"""Committed online observations -> ordinary admission -> proof-bound credit.

No provider calls, physical replay credit, current-task self-attribution, status
overrides, or benchmark rules. All replay goes through existing admission.
"""
from dataclasses import replace
import json
from itertools import combinations

from ..core.contracts import AbstractAtomicSkill, ImplementationAtom, ToolAsset
from ..core.serialization import dataclass_from_dict
from ..core.status import SkillStatus, ToolStatus
from ..core.edges import GlobalRelationType
from ..knowledge.execution_observations import verify_committed_source
from .identity_matching import match_atomic, match_tool, match_implementation, MAX_SEARCH_STATES, raw_hash
from .replay_certificates import mapped_case_body


def _bundle(source):
    bundle = source["evidence"]["bundle"]
    return (dataclass_from_dict(AbstractAtomicSkill, bundle["atomic"]),
            dataclass_from_dict(ToolAsset, bundle["tool"]),
            dataclass_from_dict(ImplementationAtom, bundle["implementation"]))


def _existing_route(system, atomic, tool, implementation):
    remaining = MAX_SEARCH_STATES
    for candidate in system.skills.implementations():
        target_atomic = system.skills.get_atomic(candidate.abstract_ref)
        target_tools = {str(binding.tool_ref): system.tools.get(binding.tool_ref) for binding in candidate.tool_bindings}
        result = match_implementation(implementation, candidate, source_atomic=atomic, target_atomic=target_atomic,
            source_tools={str(tool.ref): tool}, target_tools=target_tools, max_states=remaining)
        remaining -= result.search_states
        if result.status == "exact":
            return candidate
        if remaining <= 0:
            break
    return None


def prepare_attributions(system, trace, task, sources):
    if system.readonly:
        raise RuntimeError("Frozen cannot attribute training executions")
    # Verify even negative sources before touching the registry or the ledger.
    sources = [verify_committed_source(system, source) for source in sources]
    groups = []
    for source in sources:
        if source["observation"]["outcome"] != "success":
            continue
        atomic, tool, implementation = _bundle(source)
        group = None
        remaining = MAX_SEARCH_STATES
        for items in groups:
            if remaining <= 0:
                break  # Unresolved groups remain separate, without negative credit.
            other_atomic, other_tool, _ = _bundle(items[0])
            contract = match_atomic(atomic, other_atomic, max_states=remaining)
            remaining -= contract.search_states
            if contract.status != "exact":
                continue
            program = match_tool(tool, other_tool, max_states=remaining)
            remaining -= program.search_states
            if program.status == "exact":
                group = items
                break
        if group is None:
            groups.append([source])
        else:
            group.append(source)
    events = []
    for group in groups:
        by_task = {}
        for source in group:
            by_task.setdefault(source["observation"]["independent_task_key"], source)
        if len(by_task) < 2:
            continue
        atomic, tool, implementation = _bundle(next(iter(by_task.values())))
        if _existing_route(system, atomic, tool, implementation) is not None:
            continue
        from ..knowledge.r103_protocol import METADATA
        attempts = trace.metadata.setdefault("runtime_admission_attempts", [])
        selected = None
        for pair in combinations(by_task.values(), 2):
            source_keys = sorted(source["observation"]["execution_key"] for source in pair)
            attempt_key = "admit_" + raw_hash([source_keys, METADATA])
            if (not system.database.execute("SELECT 1 FROM runtime_admission_attempts WHERE attempt_key=?", (attempt_key,)).fetchone()
                    and not any(item["attempt_key"] == attempt_key for item in attempts)):
                selected = pair
                break
        if selected is None:
            continue
        atomic, tool, implementation = _bundle(selected[0])
        attempt = {"attempt_key": attempt_key, "source_keys_json": json.dumps(source_keys),
                   "result_status": "prepared", "publishing_trace_id": trace.trace_id}
        attempts.append(attempt)
        # Program support is independent of Implementation mapping support.
        # Map each Tool's own case with its full program proof; later credit
        # requires a separate joint route proof for each Implementation.
        cases = []
        for source in selected:
            _, source_tool, _ = _bundle(source)
            tool_proof = match_tool(source_tool, tool).proof
            case = source_tool.tests[0]
            mapped = mapped_case_body(case, tool_proof)
            if "case_id" in case:
                mapped["case_id"] = case["case_id"]
            cases.append(mapped)
        tool = replace(tool, tests=cases)
        admitted = system.admission.admit_tool(tool, atomic=atomic, harness=system.harness,
            replay=lambda candidate, case: system._replay_case_with_source_authority(candidate, case,
                current_task=task, current_trace=trace if task is not None else None, audit_trace=trace))
        if admitted.status is not ToolStatus.CANDIDATE:
            attempt["result_status"] = "tool_admission_rejected"
            trace.metadata.setdefault("runtime_support_promotion_rejections", []).append({
                "stage": "tool_admission", "source_execution_keys": [s["observation"]["execution_key"] for s in selected],
                "reasons": admitted.metadata.get("admission_failure", [])})
            continue
        implementation = system.admission.admit_implementation(implementation, admitted, atomic=atomic, harness=system.harness)
        if implementation.status is not SkillStatus.CANDIDATE:
            attempt["result_status"] = "implementation_admission_rejected"
            continue
        atomic_ref = system.aligner.align_atomic(replace(atomic, status=SkillStatus.CANDIDATE))
        alignment = system.aligner.align_tool_with_replays(admitted, admission=system.admission, replay=None)
        if not alignment.admitted:
            attempt["result_status"] = "aligned_executable_unavailable"
            continue
        implementation_ref = system.aligner.align_implementation(implementation, atomic_ref, alignment.ref, source_tool=admitted)
        system._add_structural_edge(str(implementation_ref), str(atomic_ref), GlobalRelationType.IMPLEMENTS, trace.trace_id)
        system._add_structural_edge(str(implementation_ref), str(alignment.ref), GlobalRelationType.CONTAINS, trace.trace_id)
        from .admission_evidence import additional_admissions
        events.extend(additional_admissions(system, trace, [atomic_ref], [implementation_ref], [alignment.ref]))
        attempt["result_status"] = "candidate_admitted"
        trace.metadata.setdefault("runtime_support_promotions", []).append({
            "refs": list(map(str, (atomic_ref, implementation_ref, alignment.ref))),
            "source_execution_keys": [s["observation"]["execution_key"] for s in selected], "status": "candidate"})
    # All exact historical failures enter the same batch BEFORE lifecycle
    # review; a new alias cannot erase them. Suppressed assets stay suppressed.
    pending = set()
    def append_once(event):
        if event is None:
            return
        key = (event.artifact_ref, event.metadata["source_execution_key"], event.metadata["source_class"])
        prior = system.database.execute("SELECT outcome,certificate_hash FROM execution_attribution_index "
            "WHERE target_ref=? AND source_execution_key=? AND evidence_class=?", key).fetchone()
        if prior is not None:
            if (prior["outcome"] != event.metadata["outcome"] or prior["certificate_hash"] != event.metadata["attribution_certificate_hash"]):
                raise RuntimeError("source execution attribution changed after commit")
        elif key not in pending:
            pending.add(key)
            events.append(event)
    for source in sources:
        if source["observation"]["outcome"] not in {"success", "failure"}:
            continue
        if source["observation"]["outcome"] == "failure":
            # Program failures are not contingent on an Atomic/Impl alias.
            # Include bound and standalone Tools, with one canonical proof.
            for tool in system.tools.tools():
                tool = dataclass_from_dict(ToolAsset, system.artifacts.get_payload(str(tool.ref)))
                append_once(system.credit.attribute_execution(source=source, target_atomic=None, target_tool=tool,
                    target_implementation=None, publishing_task_id=trace.task.task_id,
                    publishing_trace_id=trace.trace_id, artifact_kind="tool"))
            continue
        for implementation in system.skills.implementations():
            # Proof hashes bind immutable files, not projected lifecycle status.
            implementation = dataclass_from_dict(ImplementationAtom, system.artifacts.get_payload(str(implementation.ref)))
            atomic = dataclass_from_dict(AbstractAtomicSkill, system.artifacts.get_payload(str(implementation.abstract_ref)))
            for binding in implementation.tool_bindings:
                tool = dataclass_from_dict(ToolAsset, system.artifacts.get_payload(str(binding.tool_ref)))
                for kind in ("atomic", "tool", "implementation"):
                    event = system.credit.attribute_execution(source=source, target_atomic=atomic, target_tool=tool,
                        target_implementation=implementation, publishing_task_id=trace.task.task_id,
                        publishing_trace_id=trace.trace_id, artifact_kind=kind)
                    append_once(event)
    trace.metadata["execution_attribution_prepared_count"] = len(pending)
    return events
