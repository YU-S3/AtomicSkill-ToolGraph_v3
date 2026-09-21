"""Append-only replay evidence, independent of immutable executable versions."""
from __future__ import annotations

from dataclasses import asdict, replace
import copy
from typing import Any, Iterable

from ..core.refs import content_hash
from ..core.contracts import ToolAsset
from ..core.serialization import to_primitive
from ..governance.ledger import EvidenceEvent, EvidenceEventType, EvidenceLedger
from .replay import ReplayCaseResult, replay_case_id


REPLAY_AUTHORITY_VERSION = "tool_replay_r103_identity_v1"
REPLAY_EVENTS = (EvidenceEventType.REPLAY_VALIDATED, EvidenceEventType.REPLAY_REJECTED)


def mapped_case_body(case, proof):
    """Rename declared Tool roles only; source task/prefix/owner stay literal."""
    from .contract_canonicalizer import _rewrite_tool_ir
    body = copy.deepcopy({k: v for k, v in case.items() if k != "case_id"})
    if "bindings" in body:
        if not set(body["bindings"]) <= set(proof.input_role_map):
            raise ValueError("case has inputs outside the proven interface")
        body["bindings"] = {proof.input_role_map[k]: v for k,v in body["bindings"].items()}
    if "effects" in body:
        body["effects"] = _rewrite_tool_ir(body["effects"], proof.input_role_map, proof.output_role_map)
    return body


class ReplayCertificates:
    def __init__(self, ledger: EvidenceLedger, *, authority_version: str = REPLAY_AUTHORITY_VERSION):
        self.ledger = ledger
        self.authority_version = authority_version

    def events(self, signature: str) -> list[EvidenceEvent]:
        rows = self.ledger.database.rows(
            "SELECT * FROM evidence_events WHERE event_type IN (?,?) "
            "AND json_extract(metadata_json, '$.executable_signature')=? ORDER BY rowid",
            (*[event.value for event in REPLAY_EVENTS], signature),
        )
        return [EvidenceEvent.from_row(row) for row in rows]

    def lookup(self, signature: str, case: dict[str, Any], *,
               pending: Iterable[EvidenceEvent] = (), tool: ToolAsset | None = None,
               semantic_profile: str = "") -> ReplayCaseResult | None:
        # A case identifier alone is insufficient: source content must match too.
        self.last_reuse_proof = None
        from .identity_matching import typed_json, match_tool, MAX_SEARCH_STATES, raw_hash
        events = self.events(signature)
        if tool is not None:
            rows = self.ledger.database.rows("SELECT * FROM evidence_events WHERE event_type=? "
                "AND json_extract(metadata_json,'$.execution_semantic_profile')=? ORDER BY rowid",
                (EvidenceEventType.REPLAY_VALIDATED.value, semantic_profile))
            events = [EvidenceEvent.from_row(row) for row in rows]
        remaining = MAX_SEARCH_STATES
        for event in [*events, *pending]:
            metadata = event.metadata
            from ..core.serialization import dataclass_from_dict
            if (event.event is not EvidenceEventType.REPLAY_VALIDATED
                    or metadata.get("execution_semantic_profile", "") != semantic_profile
                    or metadata.get("replay_authority_version") != self.authority_version
                    or metadata.get("case_hash") != content_hash(metadata.get("case"))):
                continue
            matched = False
            if tool is not None:
                raw = metadata.get("source_executable_payload")
                if not isinstance(raw, dict):
                    continue
                original = dataclass_from_dict(ToolAsset, raw)
                if raw_hash(original) != metadata.get("source_executable_raw_hash"):
                    continue
                target_body = {k:v for k,v in metadata["case"].items() if k != "case_id"}
                def compatible(proof):
                    try:
                        return typed_json(mapped_case_body(case, proof)) == typed_json(target_body)
                    except ValueError:
                        return False
                identity = match_tool(tool, original, max_states=remaining, compatible_mapping=compatible)
                remaining -= identity.search_states
                if identity.status != "exact":
                    continue
                matched = True
                self.last_reuse_proof = {"executable_proof": to_primitive(identity.proof),
                    "request_case_hash": content_hash(case), "mapped_case_body_hash": raw_hash(target_body),
                    "certificate_event_id": event.event_id, "execution_semantic_profile": semantic_profile}
            if matched or (tool is None and metadata.get("executable_signature") == signature
                    and metadata.get("replay_case_id") == replay_case_id(case)
                    and metadata.get("case_hash") == content_hash(case)
                    and typed_json(metadata.get("case")) == typed_json(case)
                    and metadata.get("replay_authority_version") == self.authority_version):
                result = ReplayCaseResult(**metadata["result"])
                if result.passed is True:
                    return replace(result, case_id=replay_case_id(case), stage="certificate_reuse", started=False,
                                   executed_action_count=0, completed=False,
                                   terminal_interrupted=False)
        return None

    def certificate(self, signature: str, case: dict[str, Any], result: ReplayCaseResult,
                    *, artifact_ref: str, trace_id: str, task_id: str,
                    tool: ToolAsset | None = None,
                    identity_binding: dict[str, Any] | None = None,
                    semantic_profile: str = "") -> EvidenceEvent:
        if result.case_id != replay_case_id(case):
            raise ValueError("replay result does not name its immutable case")
        case_hash = content_hash(case)
        identity_metadata = {}
        if tool is not None:
            from .identity_matching import IDENTITY_VERSION, raw_hash
            from .aligner import _tool_signature
            if _tool_signature(tool) != signature:
                raise ValueError("replay executable snapshot does not match its raw key")
            identity_metadata = {"identity_version": IDENTITY_VERSION,
                "source_executable_payload": to_primitive(tool), "source_executable_raw_hash": raw_hash(tool)}
        if identity_binding is not None:
            from ..core.serialization import dataclass_from_dict
            from .identity_matching import IdentityProof, verify_tool_proof
            target = dataclass_from_dict(ToolAsset, identity_binding["target_payload"])
            proof = IdentityProof(**identity_binding["proof"])
            if tool is None or str(target.ref) != artifact_ref or not verify_tool_proof(tool, target, proof):
                raise ValueError("replay canonical identity binding is not a full verified proof")
            identity_metadata["identity_binding"] = identity_binding
        return EvidenceEvent.create(
            task_id=task_id, trace_id=trace_id, occurrence_id="replay",
            attempt_id=f"replay:{signature}:{self.authority_version}:{case_hash}:{content_hash(result)}",
            sequence_no=0, artifact_ref=artifact_ref, artifact_kind="tool",
            event=(EvidenceEventType.REPLAY_VALIDATED if result.passed
                   else EvidenceEventType.REPLAY_REJECTED),
            metadata={"replay_case_id": replay_case_id(case), "case_hash": case_hash,
                      "execution_semantic_profile": semantic_profile,
                      "executable_signature": signature,
                      "replay_authority_version": self.authority_version,
                      "source_trace_id": result.source_trace_id,
                      "source_task_id": result.source_task_id,
                      "passed": result.passed, "stage": result.stage,
                      "failure_code": result.failure_code,
                      "executed_action_count": result.executed_action_count,
                      "terminal_interrupted": result.terminal_interrupted,
                      "case": case, "result": asdict(result), **identity_metadata},
        )

    def cases(self, signature: str) -> list[dict[str, Any]]:
        # Include rejection evidence for future repair; never overwrite older cases.
        cases = {}
        for event in self.events(signature):
            case = event.metadata["case"]
            cases.setdefault(content_hash(case), case)
        return list(cases.values())
