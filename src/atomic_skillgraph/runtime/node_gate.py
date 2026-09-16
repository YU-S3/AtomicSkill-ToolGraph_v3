"""Typed semantic gate after invocation bindings have passed preflight."""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NodeGateDecision:
    ready: bool
    reason: str
    precondition_status: tuple[dict[str, Any], ...]


def precondition_gate(executor, occurrence, compiled, ctx):
    state = executor._activate_occurrence_state(occurrence, compiled.atomic, [compiled], ctx)
    statuses = tuple(state.get("precondition_status", state.get("preconditions", [])))
    ready = len(statuses) == len(compiled.atomic.preconditions) and all(
        item.get("status") == "satisfied" for item in statuses)
    result = NodeGateDecision(ready, "ready" if ready else "runtime_preconditions_unsatisfied", statuses)
    ctx.trace_builder.trace.metadata.setdefault("runtime_node_gates", []).append({
        "occurrence_id": occurrence.occurrence_id, "revision": ctx.world_revision,
        "implementation_ref": str(compiled.implementation.ref),
        "ready": ready, "reason": result.reason, "precondition_status": list(statuses),
    })
    return result
