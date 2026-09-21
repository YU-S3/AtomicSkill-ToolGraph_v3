"""Trace-first replay observations, resolved against immutable registered Tools.

Candidate execution is audit evidence, not proof that an asset was admitted.
Resolution happens after all admissions and before Trace persistence. Commit
validates the entire resolved batch again before any ledger append.
"""
from __future__ import annotations

from typing import Any
from dataclasses import replace

from ..core.errors import AtomicSkillGraphError, FailureLayer
from ..core.refs import content_hash
from ..core.serialization import to_primitive, dataclass_from_dict
from ..core.contracts import ToolAsset
from ..governance.ledger import EvidenceEvent
from .aligner import _tool_signature, _version_key
from .replay import ReplayCaseResult
from .replay_certificates import ReplayCertificates, REPLAY_AUTHORITY_VERSION
from .identity_matching import IDENTITY_VERSION, raw_hash, match_tool, IdentityProof, verify_tool_proof


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
        source = _source_tool(event)
        recreated = certificates.certificate(metadata["executable_signature"], case, result,
            artifact_ref=ref, trace_id=event.trace_id, task_id=event.task_id,
            tool=source, identity_binding=metadata.get("identity_binding"),
            semantic_profile=metadata.get("execution_semantic_profile", ""))
    except (KeyError, TypeError, ValueError) as exc:
        _integrity(f"invalid replay observation: {exc}")
    if recreated.metadata != event.metadata or recreated.event != event.event:
        _integrity("replay result and event metadata disagree")
    return recreated


def _source_tool(event: EvidenceEvent) -> ToolAsset | None:
    metadata = event.metadata
    if "source_executable_payload" not in metadata:
        return None  # Historical exact-byte-only certificate fixture.
    try:
        source = dataclass_from_dict(ToolAsset, metadata["source_executable_payload"])
        if (metadata.get("identity_version") != IDENTITY_VERSION
                or raw_hash(source) != metadata.get("source_executable_raw_hash")
                or _tool_signature(source) != metadata.get("executable_signature")):
            _integrity("replay source executable bytes/version do not match")
        return source
    except (KeyError, TypeError, ValueError) as exc:
        _integrity(f"invalid replay source executable: {exc}")


def _matches(event: EvidenceEvent, target: ToolAsset) -> bool:
    source = _source_tool(event)
    if source is None:
        return _tool_signature(target) == event.metadata.get("executable_signature")
    return match_tool(source, target).status == "exact"


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
        if existing is not None and not _matches(event, existing):
            _integrity(f"registered replay target signature differs: {event.artifact_ref}")
        matches = [tool for tool in tools if _matches(event, tool)]
        target = existing or (max(matches, key=lambda t: (_version_key(t.ref.version), str(t.ref))) if matches else None)
        decision = {"candidate_ref": event.artifact_ref, "executable_signature": signature,
                    "event_ids": [event.event_id], "canonical_artifact_ref": None,
                    "disposition": "trace_only_rejected", "reason": "no_registered_executable"}
        if target is not None:
            source = _source_tool(event)
            if source is None:
                resolved = _recreate(event, str(target.ref), certificates)
            else:
                proof = match_tool(source, target).proof
                resolved = certificates.certificate(signature, event.metadata["case"],
                    ReplayCaseResult(**event.metadata["result"]), artifact_ref=str(target.ref),
                    trace_id=event.trace_id, task_id=event.task_id, tool=source,
                    semantic_profile=event.metadata.get("execution_semantic_profile", ""),
                    identity_binding={"target_payload": to_primitive(target), "proof": to_primitive(proof)})
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
        if not _matches(event, tool):
            _integrity(f"registered replay target signature differs: {event.artifact_ref}")
        binding = event.metadata.get("identity_binding")
        if binding is not None:
            snapshot = dataclass_from_dict(ToolAsset, binding["target_payload"])
            # Registry status is a lifecycle projection, not immutable bytes.
            if raw_hash(replace(tool, status=snapshot.status)) != raw_hash(snapshot):
                _integrity("registered executable differs from identity proof target snapshot")
        recreated = _recreate(event, str(tool.ref), certificates)
        if to_primitive(event) != to_primitive(recreated):
            _integrity("replay certificate identity is not canonical")
    return events
