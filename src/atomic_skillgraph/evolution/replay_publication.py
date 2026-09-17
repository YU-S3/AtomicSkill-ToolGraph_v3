"""Trace-first replay observations, resolved against immutable registered Tools.

Candidate execution is audit evidence, not proof that an asset was admitted.
Resolution happens after all admissions and before Trace persistence. Commit
validates the entire resolved batch again before any ledger append.
"""
from __future__ import annotations

from typing import Any

from ..core.errors import AtomicSkillGraphError, FailureLayer
from ..core.refs import content_hash
from ..core.serialization import to_primitive
from ..governance.ledger import EvidenceEvent
from .aligner import _tool_signature, _version_key
from .replay import ReplayCaseResult
from .replay_certificates import ReplayCertificates, REPLAY_AUTHORITY_VERSION


def _integrity(message: str) -> None:
    raise AtomicSkillGraphError("replay_publication_integrity", message,
                               layer=FailureLayer.INFRASTRUCTURE)


def _recreate(event: EvidenceEvent, ref: str, certificates: ReplayCertificates) -> EvidenceEvent:
    metadata = event.metadata
    case = metadata.get("case")
    if (not isinstance(case, dict) or metadata.get("case_hash") != content_hash(case)
            or metadata.get("replay_authority_version") != REPLAY_AUTHORITY_VERSION
            or event.artifact_kind != "tool"):
        _integrity("replay case hash, kind or authority does not match")
    try:
        result = ReplayCaseResult(**metadata["result"])
        recreated = certificates.certificate(metadata["executable_signature"], case, result,
            artifact_ref=ref, trace_id=event.trace_id, task_id=event.task_id)
    except (KeyError, TypeError, ValueError) as exc:
        _integrity(f"invalid replay observation: {exc}")
    if recreated.metadata != event.metadata or recreated.event != event.event:
        _integrity("replay result and event metadata disagree")
    return recreated


def resolve(system: Any, trace: Any) -> None:
    pending = trace.metadata.get("replay_certificate_events", [])
    if not pending:
        return
    if system.readonly:
        return
    # Keep original identities/results, including refused candidates, immutable
    # in the Trace; only the publication list is rebound to registered assets.
    observations = trace.metadata.setdefault("replay_candidate_observations", list(pending))
    certificates = ReplayCertificates(system.ledger)
    tools = system.tools.tools()
    by_ref = {str(tool.ref): tool for tool in tools}
    published, decisions = [], []
    for raw in observations:
        event = EvidenceEvent(**raw)
        signature = event.metadata.get("executable_signature")
        _recreate(event, event.artifact_ref, certificates)
        existing = by_ref.get(event.artifact_ref)
        if existing is not None and _tool_signature(existing) != signature:
            _integrity(f"registered replay target signature differs: {event.artifact_ref}")
        matches = [tool for tool in tools if _tool_signature(tool) == signature]
        target = existing or (max(matches, key=lambda t: (_version_key(t.ref.version), str(t.ref))) if matches else None)
        decision = {"candidate_ref": event.artifact_ref, "executable_signature": signature,
                    "event_ids": [event.event_id], "canonical_artifact_ref": None,
                    "disposition": "trace_only_rejected", "reason": "no_registered_executable"}
        if target is not None:
            resolved = _recreate(event, str(target.ref), certificates)
            published.append(to_primitive(resolved))
            decision.update(canonical_artifact_ref=str(target.ref), disposition="publish_registered",
                            reason="registered_signature_verified", event_ids=[resolved.event_id])
        decisions.append(decision)
    trace.metadata["replay_publication_decisions"] = decisions
    trace.metadata["replay_certificate_events"] = published
    pending_ids = {item["event_id"] for item in observations}
    trace.evidence_event_refs = list(dict.fromkeys([
        *[ref for ref in trace.evidence_event_refs if ref not in pending_ids],
        *[item["event_id"] for item in published],
    ]))
    trace.metadata.setdefault("r101_metrics", {}).update(
        replay_trace_only_candidate_events=sum(d["disposition"] == "trace_only_rejected" for d in decisions),
        replay_registered_events=len(published))


def validate_batch(system: Any, trace: Any) -> list[EvidenceEvent]:
    certificates = ReplayCertificates(system.ledger)
    events = [EvidenceEvent(**raw) for raw in trace.metadata.get("replay_certificate_events", [])]
    for event in events:
        try:
            tool = system.tools.get(event.artifact_ref)
        except (KeyError, ValueError) as exc:
            _integrity(f"unregistered replay target: {event.artifact_ref}: {exc}")
        if _tool_signature(tool) != event.metadata.get("executable_signature"):
            _integrity(f"registered replay target signature differs: {event.artifact_ref}")
        recreated = _recreate(event, str(tool.ref), certificates)
        if to_primitive(event) != to_primitive(recreated):
            _integrity("replay certificate identity is not canonical")
    return events
