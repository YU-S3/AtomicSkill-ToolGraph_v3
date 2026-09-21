"""One admission fact per task/artifact, across ordinary and sidecar channels.

Ordinary occurrence evidence remains immutable. A later successful sidecar may
add a missing validation, but cannot duplicate or overwrite the earlier proposal.
"""
from ..governance.ledger import EvidenceEventType


def additional_admissions(system, trace, atomic_refs=(), implementation_refs=(), tool_refs=()):
    existing = {(row["artifact_ref"], row["event_type"]) for row in system.database.rows(
        "SELECT artifact_ref,event_type FROM evidence_events WHERE trace_id=? AND "
        "event_type IN (?,?)", (trace.trace_id, EvidenceEventType.PROPOSED.value, EvidenceEventType.VALIDATED.value))}
    staged = trace.metadata.setdefault("r103_admission_facts", [])
    existing.update((row["artifact_ref"], row["event_type"]) for row in staged)
    events = []
    # Each artifact uses its own stable attempt. Ordering of other proposals
    # cannot change the event identity or its content.
    for kind, refs in (("atomic", atomic_refs), ("implementation", implementation_refs), ("tool", tool_refs)):
        for ref in refs:
            args = ([ref] if kind == "atomic" else [], [ref] if kind == "implementation" else [],
                    [ref] if kind == "tool" else [], None)
            for event in system.credit.assign_evolution(trace, *args):
                key = (event.artifact_ref, event.event.value)
                if key not in existing:
                    existing.add(key)
                    staged.append({"artifact_ref": key[0], "event_type": key[1], "event_id": event.event_id})
                    events.append(event)
    return events
