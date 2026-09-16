"""Two independent task observations plus cross-case replay, ordinary admission."""
from __future__ import annotations
import copy
import time
from dataclasses import replace
from typing import Any
from ..core.contracts import AbstractAtomicSkill, ToolAsset, ImplementationAtom
from ..core.refs import content_hash
from ..core.serialization import dataclass_from_dict, to_primitive
from ..core.status import SkillStatus, ToolStatus
from ..core.edges import GlobalRelationType
from ..governance.credit import CreditAttempt, CreditTrace
from .aligner import _tool_signature
from .contract_canonicalizer import atomic_contract_signature


def collect_observations(system: Any, trace: Any) -> list[dict]:
    if (system.readonly or not trace.learning_eligible or trace.infrastructure_failure
            or not trace.benchmark_success or not trace.task_contract_success):
        return []
    observations = []
    seen = set()
    trials = trace.metadata.get("runtime_tool_trials", [])
    if isinstance(trials, dict):
        trials = trials.values()
    for trial in trials:
        if (not trial.get("r1", {}).get("admission_eligible")
                or not trial.get("parent_completed_after_trial")
                or trial.get("terminal_interrupted") or not trial.get("promotion_bundle")):
            continue
        bundle = trial["promotion_bundle"]
        atomic = dataclass_from_dict(AbstractAtomicSkill, bundle["atomic"])
        signature = atomic_contract_signature(atomic)
        if signature in seen:
            continue
        seen.add(signature)
        identity = {"contract_signature": signature, "task_id": trace.task.task_id}
        observations.append({
            **identity, "observation_id": "runtime_support_" + content_hash(identity)[:24],
            "trace_id": trace.trace_id, "draft_id": trial["draft_id"],
            "harness_profile": system.harness.profile_name,
            "bundle": copy.deepcopy(bundle), "tool_proposal": trial.get("tool_proposal", {}),
            "trial": {key: value for key, value in trial.items()
                      if key not in {"promotion_bundle", "tool_proposal"}},
            "created_at": time.time(),
        })
    return observations


def prepare_and_apply(system: Any, trace: Any, task: Any, observations: list[dict]) -> list[Any]:
    if system.readonly:
        return []
    metrics = trace.metadata.setdefault("r10_metrics", {})
    events = []
    for observation in observations:
        group = system.runtime_support_store.observations(observation["contract_signature"])
        by_task = {item["task_id"]: item for item in group}
        by_task.setdefault(observation["task_id"], observation)
        if len(by_task) < 2:
            continue
        sources = list(by_task.values())[-2:]
        programs = {}
        for source in sources:
            tool = dataclass_from_dict(ToolAsset, source["bundle"]["tool"])
            programs.setdefault(_tool_signature(tool), source)
        metrics["runtime_support_promotion_attempt_count"] = metrics.get("runtime_support_promotion_attempt_count", 0) + 1
        cases = [copy.deepcopy(source["bundle"]["tool"]["tests"][0]) for source in sources]
        for source in programs.values():
            atomic = dataclass_from_dict(AbstractAtomicSkill, source["bundle"]["atomic"])
            tool = replace(dataclass_from_dict(ToolAsset, source["bundle"]["tool"]), tests=cases)
            implementation = dataclass_from_dict(ImplementationAtom, source["bundle"]["implementation"])
            admitted = system.admission.admit_tool(
                tool, atomic=atomic, harness=system.harness,
                replay=lambda candidate, case: system._replay_case_with_source_authority(
                    candidate, case, current_task=task, current_trace=trace, audit_trace=trace),
            )
            if admitted.status is not ToolStatus.CANDIDATE:
                trace.metadata.setdefault("runtime_support_promotion_rejections", []).append({
                    "contract_signature": observation["contract_signature"], "stage": "tool_admission",
                    "executable_signature": _tool_signature(tool), "reasons": admitted.metadata.get("admission_failure", []),
                })
                continue
            implementation = system.admission.admit_implementation(
                implementation, admitted, atomic=atomic, harness=system.harness,
            )
            if implementation.status is not SkillStatus.CANDIDATE:
                continue
            atomic_ref = system.aligner.align_atomic(replace(atomic, status=SkillStatus.CANDIDATE,
                metadata={**atomic.metadata, "runtime_support_promotion": True}))
            alignment = system.aligner.align_tool_with_replays(admitted, admission=system.admission, replay=None)
            if not alignment.admitted:
                continue
            implementation_ref = system.aligner.align_implementation(implementation, atomic_ref, alignment.ref)
            system._add_structural_edge(str(implementation_ref), str(atomic_ref), GlobalRelationType.IMPLEMENTS, trace.trace_id)
            system._add_structural_edge(str(implementation_ref), str(alignment.ref), GlobalRelationType.CONTAINS, trace.trace_id)
            refs = [str(atomic_ref), str(implementation_ref), str(alignment.ref)]
            attempts = [CreditAttempt(
                artifact_ref=ref, artifact_kind=kind, occurrence_id="runtime_support_promotion",
                attempt_id=f"runtime_support_promotion:{observation['contract_signature']}:{ref}",
                sequence_no=index, proposed=True, validated=True,
                metadata={"source": "runtime_support_promotion", "source_task_ids": list(by_task)},
            ) for index, (kind, ref) in enumerate(zip(("atomic", "implementation", "tool"), refs))]
            events.extend(system.credit.assign(CreditTrace(trace.task.task_id, trace.trace_id, tuple(attempts))))
            trace.metadata.setdefault("runtime_support_promotions", []).append({
                "contract_signature": observation["contract_signature"], "refs": refs,
                "source_task_ids": [item["task_id"] for item in sources],
                "status": "candidate", "executable_signature": _tool_signature(tool),
            })
            metrics["runtime_support_promotion_success_count"] = metrics.get("runtime_support_promotion_success_count", 0) + 1
            break
    return events
