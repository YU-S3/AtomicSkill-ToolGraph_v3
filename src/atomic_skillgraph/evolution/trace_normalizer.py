"""Project validator-issued transition certificates into the extractor view."""

from __future__ import annotations

from typing import Any

from ..core.serialization import to_primitive


def _project_certificate(record: dict[str, Any]) -> dict[str, Any]:
    certificate = record.get("transition_certificate")
    if not isinstance(certificate, dict):
        raise ValueError(
            f"environment action {record.get('action_id', '')!r} lacks an "
            "ActionTransitionCertificate"
        )
    expected = {
        "action_id": str(record.get("action_id", "")),
        "revision_before": int(record.get("revision", -1)),
        "revision_after": int(record.get("new_revision", -1)),
        "action_type": str(record.get("action_type", "")),
        "arguments": dict(record.get("arguments") or {}),
        "accepted": bool(record.get("accepted")),
    }
    for key, value in expected.items():
        if certificate.get(key) != value:
            raise ValueError(
                f"transition certificate {key} does not match action record: "
                f"{certificate.get(key)!r} != {value!r}"
            )
    required_fields = {
        "before_facts",
        "positive_effects",
        "negative_effects",
        "required_facts",
        "terminal_effects",
        "state_changed",
        "evidence_refs",
    }
    missing = sorted(required_fields - set(certificate))
    if missing:
        raise ValueError(f"transition certificate missing fields: {missing}")
    return certificate


class TraceNormalizer:
    """Expose trace chronology without deriving or guessing semantic facts."""

    def build(self, trace: Any) -> dict[str, Any]:
        actions: list[dict[str, Any]] = []
        for index, raw_record in enumerate(trace.environment_actions):
            record = to_primitive(raw_record)
            certificate = _project_certificate(record)
            actions.append({
                "event_index": index,
                "extractor_event_start": index,
                "extractor_event_end_exclusive": index + 1,
                "action_id": record["action_id"],
                "action_type": record["action_type"],
                "arguments": dict(record.get("arguments") or {}),
                "accepted": bool(record["accepted"]),
                "state_changed": bool(certificate["state_changed"]),
                "before_revision": int(record["revision"]),
                "after_revision": int(record["new_revision"]),
                "done": bool(record["done"]),
                "won": bool(record["won"]),
                "span_id": str(record["span_id"]),
                "transition_certificate": certificate,
            })
        return {
            "trace_id": trace.trace_id,
            "task_goal": trace.task.goal,
            "source_task": {
                "task_id": trace.task.task_id,
                "task_signature": trace.task.task_signature,
                "goal": trace.task.goal,
                "benchmark": trace.task.benchmark,
                "task_type": trace.task.task_type,
                "context": {
                    "env_index": trace.task.metadata.get("env_index"),
                    "game_file": trace.task.metadata.get("game_file", ""),
                },
                "metadata": dict(trace.task.metadata),
            },
            "task_contract": to_primitive(trace.task_contract),
            "benchmark_success": bool(trace.benchmark_success),
            "actions": actions,
            "runtime_spans": [
                to_primitive(item) for item in trace.runtime_spans if item.learnable
            ],
            "validations": [to_primitive(item) for item in trace.validations],
            "node_records": [to_primitive(item) for item in trace.node_records],
            "implementation_invocations": [
                to_primitive(item) for item in trace.implementation_invocations
            ],
            "tool_executions": [to_primitive(item) for item in trace.tool_executions],
        }


__all__ = ["TraceNormalizer"]
