"""Structured Trace failure localization for the evolution branch.

This module deliberately consumes result fields only.  It never infers a
failure from observations, model prose, or exception-message parsing.
"""

from __future__ import annotations

from typing import Any

from ..core.errors import FailureEnvelope, FailureLayer
from ..validation.failure_localizer import FailureLocalizer


_LAYER_FALLBACK = {
    "tool_preflight_rejected": FailureLayer.TOOL,
    "tool_output_schema_error": FailureLayer.TOOL,
    "composite_self_sufficiency_failure": FailureLayer.COMPOSITE,
    "task_contract_mismatch": FailureLayer.TASK_CONTRACT,
    "benchmark_goal_contract_mismatch": FailureLayer.TASK_CONTRACT,
    "benchmark_failure": FailureLayer.BENCHMARK,
}


def _layer(value: Any, code: str) -> FailureLayer | None:
    if value:
        try:
            return FailureLayer(str(value))
        except ValueError:
            return None
    return _LAYER_FALLBACK.get(code)


class FailureProcessor:
    def __init__(self, localizer: FailureLocalizer) -> None:
        self.localizer = localizer

    def localize(self, trace: Any) -> list[FailureEnvelope]:
        failures = list(trace.failures)
        seen = {
            (
                item.layer.value,
                item.code,
                item.occurrence_id,
                item.attempt_id,
                tuple(item.artifact_refs),
            )
            for item in failures
        }

        def add(
            *,
            code: str,
            occurrence_id: str,
            attempt_id: str,
            started: bool,
            layer: FailureLayer | None = None,
            artifact_refs: list[str] | None = None,
            evidence_refs: list[str] | None = None,
            message: str = "",
            recoverable: bool = False,
        ) -> None:
            if not code:
                return
            resolved_layer = layer or _LAYER_FALLBACK.get(code)
            if resolved_layer is None:
                # FailureLocalizer owns the complete stable-code mapping.
                envelope = self.localizer.localize(
                    code=code,
                    task_id=trace.task.task_id,
                    trace_id=trace.trace_id,
                    occurrence_id=occurrence_id,
                    attempt_id=attempt_id,
                    started=started,
                    artifact_refs=artifact_refs,
                    evidence_refs=evidence_refs,
                    message=message,
                    recoverable=recoverable,
                )
            else:
                envelope = self.localizer.localize(
                    code=code,
                    task_id=trace.task.task_id,
                    trace_id=trace.trace_id,
                    occurrence_id=occurrence_id,
                    attempt_id=attempt_id,
                    started=started,
                    artifact_refs=artifact_refs,
                    evidence_refs=evidence_refs,
                    message=message,
                    recoverable=recoverable,
                    layer=resolved_layer,
                )
            key = (
                envelope.layer.value,
                envelope.code,
                envelope.occurrence_id,
                envelope.attempt_id,
                tuple(envelope.artifact_refs),
            )
            if key not in seen:
                seen.add(key)
                failures.append(envelope)

        for invocation in trace.implementation_invocations:
            preflight = dict(invocation.preflight or {})
            result = dict(invocation.result or {})
            code = str(result.get("failure_code") or preflight.get("failure_code") or "")
            resolved_layer = _layer(
                result.get("failure_layer") or preflight.get("failure_layer"), code,
            )
            refs: list[str] = []
            if resolved_layer in {FailureLayer.IMPLEMENTATION, FailureLayer.TOOL, FailureLayer.ATOMIC}:
                refs.append(str(invocation.implementation_ref))
            atomic_ref = str(result.get("atomic_ref") or "")
            if resolved_layer is FailureLayer.ATOMIC and atomic_ref:
                refs.insert(0, atomic_ref)
            add(
                code=code,
                occurrence_id=str(invocation.occurrence_id),
                attempt_id=str(invocation.attempt_id),
                started=bool(result.get("started")),
                layer=resolved_layer,
                artifact_refs=list(dict.fromkeys(refs)),
                evidence_refs=list(preflight.get("matched_evidence_refs") or []),
                message=str(preflight.get("message") or ""),
                recoverable=not bool(result.get("started")),
            )

        for execution in trace.tool_executions:
            result = dict(execution.result or {})
            code = str(result.get("failure_code") or "")
            add(
                code=code,
                occurrence_id=str(execution.occurrence_id),
                attempt_id=str(execution.attempt_id),
                started=bool(result.get("started")),
                layer=_layer(result.get("failure_layer"), code) or FailureLayer.TOOL,
                artifact_refs=[str(execution.tool_ref)],
                message=str(result.get("failure_message") or ""),
                recoverable=False,
            )

        for node in trace.node_records:
            raw = dict(node.failure or {})
            code = str(raw.get("failure_code") or "")
            add(
                code=code,
                occurrence_id=str(node.occurrence_id),
                attempt_id=f"node:{node.occurrence_id}",
                started=bool(raw.get("direct_started")),
                layer=_layer(raw.get("failure_layer"), code),
                artifact_refs=[str(node.atomic_ref)] if code else [],
                recoverable=False,
            )

        composite_ref = str(trace.runtime_plan.get("source_composite_ref") or "")
        if trace.task_rescue_required and composite_ref:
            add(
                code="composite_self_sufficiency_failure",
                occurrence_id="",
                attempt_id="task_rescue",
                started=False,
                layer=FailureLayer.COMPOSITE,
                artifact_refs=[composite_ref],
                recoverable=True,
            )

        if trace.benchmark_success and not trace.learning_eligible:
            add(
                code="task_contract_mismatch",
                occurrence_id="",
                attempt_id="task_contract",
                started=False,
                layer=FailureLayer.TASK_CONTRACT,
                artifact_refs=[composite_ref] if composite_ref else [],
                recoverable=True,
            )

        dynamic = dict(trace.metadata.get("dynamic_result") or {})
        dynamic_code = str(dynamic.get("failure_code") or "")
        if dynamic_code:
            add(
                code=dynamic_code,
                occurrence_id="",
                attempt_id="full_dynamic",
                started=bool(trace.environment_actions),
                layer=_layer("", dynamic_code),
                recoverable=not trace.infrastructure_failure,
            )
        elif not trace.benchmark_success and not trace.infrastructure_failure and not failures:
            add(
                code="benchmark_failure",
                occurrence_id="",
                attempt_id="benchmark",
                started=bool(trace.environment_actions),
                layer=FailureLayer.BENCHMARK,
                recoverable=False,
            )

        trace.failures = failures
        return failures


__all__ = ["FailureProcessor"]
