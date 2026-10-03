"""Bounded E1 sidecar. Two independent sources, original validators and budget."""
import json
from dataclasses import replace

from ..core.serialization import to_primitive
from ..core.status import SkillStatus, ToolStatus
from ..core.edges import GlobalRelationType
from ..core.errors import AgentProtocolError
from ..knowledge.r103_protocol import METADATA
from .extractor_session import parse_occurrence_payload
from .identity_matching import match_atomic, raw_hash
from .contract_canonicalizer import atomic_contract_signature, CanonicalizedAtomicBundle
from .tool_compiler import CompiledKnowledge, build_occurrence_replay_case


def prepare_sidecar(system, trace, task, context):
    groups, proposals, normalized = context["groups"], context["batch"].generalizations, context["normalized"]
    if not proposals:
        return None
    submitted = proposals[0]
    group = next((g for g in groups if g["group_id"] == submitted["group_id"]), None)
    audit = trace.metadata.setdefault("generalization", {"attempted": True, "builder_calls": 0})
    state = {"audit": audit, "item": None, "attempt": None, "sources": []}
    if group is None:
        audit.update(status="rejected", reason="unoffered_group")
        return state
    pair = sorted([raw_hash(normalized), group["history_reference"]["canonical_snapshot_hash"]])
    protocol = system.mechanism_profile['generalization'] if system.mechanism_profile else METADATA["generalization_schema_version"]
    policy = {**METADATA, 'mechanism_effective_profile': system.mechanism_profile} if system.mechanism_profile else METADATA
    attempt = {"attempt_key": "generalize_" + raw_hash([pair, submitted, policy]),
        "source_pair_key": raw_hash(pair), "protocol_version": protocol,
        "group_id": group["group_id"], "source_snapshot_hashes_json": json.dumps(pair),
        "result_status": "rejected", "result_refs_json": "[]", "publishing_trace_id": trace.trace_id}
    state["attempt"] = attempt
    if system.database.execute("SELECT 1 FROM generalization_attempts WHERE attempt_key=?", (attempt["attempt_key"],)).fetchone():
        audit.update(status="already_attempted")
        state["attempt"] = None
        return state
    try:
        if submitted.get('_transport_error'):
            raise ValueError(submitted['_transport_error'])
        historical = system.learning_source_store.verified(system, group["history_reference"])
        if 'candidate_contract' in submitted:
            from .shared_contract import instantiate
            rows = instantiate(submitted, {'current': normalized['trace_id'], 'history': historical['normalized']['trace_id']})
        else:
            rows = submitted["source_proposals"]
        if len(rows) != 2 or sorted(r["source_slot"] for r in rows) != ["current", "history"]:
            raise ValueError("one current and one history source are required")
        parsed = {row["source_slot"]: parse_occurrence_payload(row["proposal"]) for row in rows}
        if group.get('preparation_spans'):
            for slot, span in group['preparation_spans'].items():
                proposed = parsed[slot]
                if (proposed.event_start != span['event_start'] or proposed.event_end + 1 != span['event_end']
                        or not set(span['support_event_ids']) <= set(proposed.support_event_ids)):
                    raise ValueError(slot + ' preparation must preserve the complete offered causal interval')
        from ..knowledge.execution_observations import execution_identity
        current_source = trace.metadata.get("execution_source", {})
        current_task_key = execution_identity(current_source, trace.trace_id, "generalization", 0)["source_independent_task_key"]
        if current_task_key == group["history_reference"]["independent_task_key"]:
            raise ValueError("generalization requires two independent training tasks")
        normals = {"current": normalized, "history": historical["normalized"]}
        canonical, atomics = {}, {}
        for slot in ("current", "history"):
            validated, rejected = system.atomicizer.validate_proposed_subset([parsed[slot]], normals[slot])
            if rejected or len(validated) != 1:
                raise ValueError(f"{slot} source does not validate independently: " + json.dumps(to_primitive(rejected)))
            canonical[slot] = validated[0]
            atomics[slot] = system._canonical_atomic_for_occurrence(validated[0])
            if atomics[slot] is None:
                raise ValueError("source has no lawful output contract")
        proof = match_atomic(atomics["history"], atomics["current"])
        if proof.status != "exact":
            raise ValueError("two source proposals do not establish one identical contract")
        # Canonical target identity, not author wording, determines repeat work.
        attempt["attempt_key"] = "generalize_" + raw_hash([pair, atomic_contract_signature(atomics["current"]), policy])
        if system.database.execute("SELECT 1 FROM generalization_attempts WHERE attempt_key=?", (attempt["attempt_key"],)).fetchone():
            state["attempt"] = None
            audit.update(status="already_attempted")
            return state
        audit.update(source_proof=to_primitive(proof.proof), status="atomic_validated")
        state["item"] = system._stage_atomic_only_occurrence(canonical["current"], atomics["current"])
        history_trace = system.traces.load(group["history_reference"]["source_trace_id"])
        for slot, source_trace in (("current", trace), ("history", history_trace)):
            reference = system.learning_source_store.stage(source_trace, normals[slot], canonical[slot], atomics[slot],
                system.harness.profile_name, proposal=parsed[slot])
            if reference is None:
                raise ValueError("sidecar source lacks committed training identity")
            state["sources"].append((reference, source_trace))
        if 'candidate_contract' in submitted:
            from .shared_contract import add_program_controls
            declarations = submitted['candidate_contract'].get('program_controls', [])
            for instance in submitted['source_instances']:
                slot = instance['source_slot']
                atomics[slot], canonical[slot] = add_program_controls(atomics[slot], canonical[slot], declarations,
                    instance.get('caller_arguments', {}), normals[slot])
            audit['protocol'] = 'skillcompiler.shared-contract.v2'
            audit['observed_occurrences_separate_from_program_controls'] = True
        extra = {"source_slot": "history", "role_mapping_to_primary": to_primitive(proof.proof),
            "source_boundary": system._build_builder_source_context(canonical["history"], normals["history"]),
            "evidence_support": canonical["history"].action_events}
        from ..system import _ToolBuildBudgetExhausted, _ToolBuildContentRejected
        record_start = len(trace.metadata.get("evolution_tool_builds", []))
        try:
            item, metrics = system._build_tool_for_occurrence(canonical["current"], atomics["current"], normalized,
                trace, task, additional_evidence_sources=[extra], allow_exact_reuse=False)
            audit["builder_calls"] = metrics["call_count"]
        except _ToolBuildBudgetExhausted:
            attempt["result_status"] = "generalization_skipped_budget"
            return state
        except _ToolBuildContentRejected:
            attempt["result_status"] = "builder_rejected_atomic_retained"
            return state
        finally:
            records = trace.metadata.get("evolution_tool_builds", [])[record_start:]
            if records:
                audit['builder_record_indices'] = list(range(record_start, record_start + len(records)))
                for record in records:
                    record['source_scope'] = 'generalization_sidecar'
                audit["builder_calls"] = sum(bool(record.get("builder_entered")) for record in records)
        if item is None:
            attempt["result_status"] = "no_tool_atomic_retained"
            return state
        # Canonicalize exactly once, then map each source to that same target.
        bundle = system.aligner.stage_atomic(item.atomic, item.tool, item.implementation)
        current_occurrence = system.aligner.atomic_canonicalizer.rewrite_canonical_occurrence(item.occurrence, bundle, atomic_ref=bundle.atomic.ref)
        hproof = match_atomic(atomics["history"], bundle.atomic)
        if hproof.status != "exact":
            raise ValueError("canonical sidecar target mapping is not proven")
        hb = CanonicalizedAtomicBundle(bundle.atomic, {**hproof.proof.output_role_map, **hproof.proof.input_role_map}, None, None,
            hproof.proof.input_role_map, hproof.proof.output_role_map)
        history_occurrence = system.aligner.atomic_canonicalizer.rewrite_canonical_occurrence(canonical["history"], hb, atomic_ref=bundle.atomic.ref)
        cases = [build_occurrence_replay_case(o, bundle.atomic, source_task=o.source_task, kind="tool_proposal_replay")
                 for o in (current_occurrence, history_occurrence)]
        state["item"] = CompiledKnowledge(current_occurrence, bundle.atomic, replace(bundle.tool, tests=cases), bundle.implementation)
        attempt["result_status"] = "prepared"
    except (KeyError, TypeError, ValueError) as exc:
        audit.update(status="rejected", reason=str(exc))
        # Only two independently validated identical contracts survive. A
        # malformed source never retains a partially validated new Atomic.
        if not audit.get("source_proof"):
            state["item"] = None
            state["sources"] = []
        attempt["result_status"] = "source_rejected" if state["item"] is None else "atomic_only"
        extractor = context.get('extractor')
        if (extractor is not None and group.get('preparation_spans') and state['item'] is None
                and not context.get('content_repair_used')):
            context['content_repair_used'] = True
            audit['initial_rejection'] = str(exc)
            audit['content_repairs'] = 1
            try:
                repaired = extractor.repair_generalization(submitted, str(exc))
            except (KeyError, TypeError, ValueError, AgentProtocolError) as failure:
                audit['repair_rejection'] = str(failure)
            else:
                if repaired:
                    from .extractor_session import ExtractionBatch
                    return prepare_sidecar(system, trace, task, {**context, 'batch': ExtractionBatch([], [repaired])})
    return state


def apply_sidecar(system, trace, task, state):
    if not state or not state["attempt"]:
        return []
    item, attempt = state["item"], state["attempt"]
    refs, events = [], []
    if item is not None:
        atomic_ref = system.aligner.align_atomic(item.atomic)
        refs = [str(atomic_ref)]
        from .admission_evidence import additional_admissions
        events.extend(additional_admissions(system, trace, [atomic_ref]))
        if item.tool is not None:
            admitted = system.admission.admit_tool(item.tool, atomic=item.atomic, harness=system.harness,
                replay=lambda tool, case: system._replay_case_with_source_authority(tool, case,
                    current_task=task, current_trace=trace, audit_trace=trace))
            if admitted.status is ToolStatus.CANDIDATE:
                implementation = system.admission.admit_implementation(item.implementation, admitted, atomic=item.atomic, harness=system.harness)
                alignment = system.aligner.align_tool_with_replays(admitted, admission=system.admission, replay=None)
                if implementation.status is SkillStatus.CANDIDATE and alignment.admitted:
                    impl_ref = system.aligner.align_implementation(implementation, atomic_ref, alignment.ref, source_tool=admitted)
                    system._add_structural_edge(str(impl_ref), str(atomic_ref), GlobalRelationType.IMPLEMENTS, trace.trace_id)
                    system._add_structural_edge(str(impl_ref), str(alignment.ref), GlobalRelationType.CONTAINS, trace.trace_id)
                    refs += list(map(str, (impl_ref, alignment.ref)))
                    events.extend(additional_admissions(system, trace, [], [impl_ref], [alignment.ref]))
                    attempt["result_status"] = "candidate_admitted"
                else:
                    attempt["result_status"] = "implementation_rejected_atomic_retained"
            else:
                attempt["result_status"] = "replay_rejected_atomic_retained"
    attempt["result_refs_json"] = json.dumps(refs)
    attempt["result_payload_hash"] = raw_hash(attempt)
    state["audit"].update(status=attempt["result_status"], refs=refs, attempt_key=attempt["attempt_key"])
    return events
