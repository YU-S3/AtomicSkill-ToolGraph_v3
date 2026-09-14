"""Append-only replay evidence, independent of immutable executable versions."""
from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any, Iterable

from ..core.refs import content_hash
from ..governance.ledger import EvidenceEvent, EvidenceEventType, EvidenceLedger
from .replay import ReplayCaseResult, replay_case_id


REPLAY_AUTHORITY_VERSION = "tool_replay_r922_v1"
REPLAY_EVENTS = (EvidenceEventType.REPLAY_VALIDATED, EvidenceEventType.REPLAY_REJECTED)


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
               pending: Iterable[EvidenceEvent] = ()) -> ReplayCaseResult | None:
        # A case identifier alone is insufficient: source content must match too.
        for event in [*self.events(signature), *pending]:
            metadata = event.metadata
            if (event.event is EvidenceEventType.REPLAY_VALIDATED
                    and metadata.get("executable_signature") == signature
                    and metadata.get("replay_case_id") == replay_case_id(case)
                    and metadata.get("case_hash") == content_hash(case)
                    and metadata.get("replay_authority_version") == self.authority_version):
                result = ReplayCaseResult(**metadata["result"])
                if result.passed is True:
                    return replace(result, stage="certificate_reuse", started=False,
                                   executed_action_count=0, completed=False,
                                   terminal_interrupted=False)
        return None

    def certificate(self, signature: str, case: dict[str, Any], result: ReplayCaseResult,
                    *, artifact_ref: str, trace_id: str, task_id: str) -> EvidenceEvent:
        if result.case_id != replay_case_id(case):
            raise ValueError("replay result does not name its immutable case")
        case_hash = content_hash(case)
        return EvidenceEvent.create(
            task_id=task_id, trace_id=trace_id, occurrence_id="replay",
            attempt_id=f"replay:{signature}:{self.authority_version}:{case_hash}:{content_hash(result)}",
            sequence_no=0, artifact_ref=artifact_ref, artifact_kind="tool",
            event=(EvidenceEventType.REPLAY_VALIDATED if result.passed
                   else EvidenceEventType.REPLAY_REJECTED),
            metadata={"replay_case_id": replay_case_id(case), "case_hash": case_hash,
                      "executable_signature": signature,
                      "replay_authority_version": self.authority_version,
                      "source_trace_id": result.source_trace_id,
                      "source_task_id": result.source_task_id,
                      "passed": result.passed, "stage": result.stage,
                      "failure_code": result.failure_code,
                      "executed_action_count": result.executed_action_count,
                      "terminal_interrupted": result.terminal_interrupted,
                      "case": case, "result": asdict(result)},
        )

    def cases(self, signature: str) -> list[dict[str, Any]]:
        # Include rejection evidence for future repair; never overwrite older cases.
        cases = {}
        for event in self.events(signature):
            case = event.metadata["case"]
            cases.setdefault(content_hash(case), case)
        return list(cases.values())
