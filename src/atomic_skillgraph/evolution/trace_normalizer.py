"""Build the state-transition-authoritative extractor view of a TraceRecord."""

from __future__ import annotations

from typing import Any

from ..core.serialization import to_primitive


class TraceNormalizer:
    def build(self, trace: Any) -> dict[str, Any]:
        actions = []
        for index, record in enumerate(trace.environment_actions):
            value = to_primitive(record)
            actions.append({
                "event_index": index, "action_id": value["action_id"],
                "action_type": value["action_type"], "arguments": value["arguments"],
                "accepted": value["accepted"], "observation": value["observation"],
                "before_revision": value["revision"], "after_revision": value["new_revision"],
                "done": value["done"], "won": value["won"], "span_id": value["span_id"],
            })
        spans = [to_primitive(item) for item in trace.runtime_spans if item.learnable]
        validations = [to_primitive(item) for item in trace.validations]
        return {
            "trace_id": trace.trace_id, "task_goal": trace.task.goal,
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
            "task_contract": trace.task_contract, "benchmark_success": trace.benchmark_success,
            "actions": actions, "runtime_spans": spans, "validations": validations,
            "node_records": [to_primitive(item) for item in trace.node_records],
            "implementation_invocations": [to_primitive(item) for item in trace.implementation_invocations],
            "tool_executions": [to_primitive(item) for item in trace.tool_executions],
        }
