"""Transactional world/logic rollback without refunding policy resources."""
from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass
from typing import Any

from ..core.errors import AtomicSkillGraphError, FailureLayer


def metrics(ctx: Any) -> dict[str, Any]:
    return ctx.trace_builder.trace.metadata.setdefault("r10_metrics", {})


def increment(ctx: Any, key: str, amount: int = 1) -> None:
    values = metrics(ctx)
    values[key] = int(values.get(key, 0)) + amount


@dataclass
class RuntimeExecutionCheckpoint:
    checkpoint_id: str
    occurrence_id: str
    world_checkpoint: Any
    logical: dict[str, Any]
    binding_state: dict[str, Any]
    evidence_state: dict[str, Any]
    action_prefix_end: int
    world_revision: int
    timeline_starts: dict[str, int]
    progress_state: dict[str, Any]
    attempt_starts: dict[str, int]


_LOGICAL = (
    "current_step_index", "world_revision", "observation", "action_catalog",
    "action_history", "validated_outputs", "occurrence_evidence",
    "active_occurrence_id", "grounding_state_by_occurrence", "exploration_memory",
)


def capture(ctx: Any, occurrence_id: str) -> RuntimeExecutionCheckpoint:
    checkpoint = ctx.harness.capture_runtime_checkpoint()
    increment(ctx, "runtime_step_checkpoint_count")
    return RuntimeExecutionCheckpoint(
        uuid.uuid4().hex, occurrence_id, checkpoint,
        {k: copy.deepcopy(getattr(ctx, k)) for k in _LOGICAL},
        {k: copy.deepcopy(v) for k, v in vars(ctx.binding_store).items() if not callable(v)},
        {k: copy.deepcopy(v) for k, v in vars(ctx.evidence_store).items() if not callable(v)},
        len(ctx.trace_builder.trace.environment_actions), ctx.world_revision,
        {key: len(value) for key, value in ctx.trace_builder.trace.metadata.items()
         if isinstance(value, list)},
        {key: copy.deepcopy(getattr(ctx.task_progress, key))
         for key in ("_revision", "_last_digest")},
        {key: len(getattr(ctx.trace_builder.trace, key))
         for key in ("implementation_invocations", "tool_executions", "validations", "runtime_spans")},
    )


def restore(ctx: Any, checkpoint: RuntimeExecutionCheckpoint, failure_code: str) -> None:
    if ctx.terminal_latched or ctx.execution_terminal():
        return
    end = len(ctx.trace_builder.trace.environment_actions)
    replay_count = 0
    negative_memory = copy.deepcopy(ctx.exploration_memory.negative_observations)
    from_revision = ctx.world_revision
    discarded_metadata = {
        key: [checkpoint.timeline_starts.get(key, 0), len(value)]
        for key, value in ctx.trace_builder.trace.metadata.items()
        if isinstance(value, list) and key != "runtime_rollbacks"
    }
    try:
        if end > checkpoint.action_prefix_end:
            result = ctx.harness.restore_runtime_checkpoint(checkpoint.world_checkpoint)
            replay_count = int(result.metadata.get("restore_replay_action_count", 0))
            if result.metadata.get("restored_digest") != checkpoint.world_checkpoint.state_digest:
                raise ValueError("restored world digest does not match checkpoint")
        for name, value in checkpoint.logical.items():
            setattr(ctx, name, copy.deepcopy(value))
        # Action ids belong to the restored adapter, not the discarded world.
        ctx.action_catalog = ctx.harness.action_catalog()
        ctx.exploration_memory.negative_observations.update(negative_memory)
        for target, state in ((ctx.binding_store, checkpoint.binding_state),
                              (ctx.evidence_store, checkpoint.evidence_state)):
            for key in list(vars(target)):
                if not callable(getattr(target, key)):
                    delattr(target, key)
            vars(target).update(copy.deepcopy(state))
        for key, value in checkpoint.progress_state.items():
            setattr(ctx.task_progress, key, value)
        ctx.task_progress.validator_channel = ctx.harness.validator_channel()
    except Exception as exc:
        increment(ctx, "runtime_checkpoint_restore_failure_count")
        raise AtomicSkillGraphError(
            "runtime_checkpoint_restore_failed", str(exc), layer=FailureLayer.INFRASTRUCTURE,
        ) from exc
    ctx.trace_builder.trace.metadata.setdefault("runtime_rollbacks", []).append({
        "rollback_id": uuid.uuid4().hex, "checkpoint_id": checkpoint.checkpoint_id,
        "occurrence_id": checkpoint.occurrence_id, "failure_code": failure_code,
        "from_revision": from_revision,
        "restored_revision": ctx.world_revision,
        "discarded_action_start": checkpoint.action_prefix_end, "discarded_action_end": end,
        "restore_replay_action_count": replay_count,
        "before_digest": checkpoint.world_checkpoint.state_digest,
        "restored_digest": checkpoint.world_checkpoint.state_digest,
        "discarded_metadata_ranges": discarded_metadata,
        "discarded_record_ranges": {
            key: [start, len(getattr(ctx.trace_builder.trace, key))]
            for key, start in checkpoint.attempt_starts.items()
        },
        "discarded_attempt_ids": {
            key: [item.attempt_id for item in getattr(ctx.trace_builder.trace, key)[start:]]
            for key, start in checkpoint.attempt_starts.items()
            if key in {"implementation_invocations", "tool_executions"}
        },
    })
    increment(ctx, "runtime_rollback_count")
    increment(ctx, "runtime_rollback_replay_action_count", replay_count)
    increment(ctx, "rolled_back_environment_action_count", end - checkpoint.action_prefix_end)
