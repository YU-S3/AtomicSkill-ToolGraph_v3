"""Canonical action indices retain immutable Trace event identities."""
from typing import Any


def canonical_action_indices(trace: Any) -> list[int]:
    metadata = trace.get("metadata", {}) if isinstance(trace, dict) else trace.metadata
    actions = trace.get("environment_actions", []) if isinstance(trace, dict) else trace.environment_actions
    discarded = set()
    for item in metadata.get("runtime_rollbacks", []):
        discarded.update(range(int(item["discarded_action_start"]), int(item["discarded_action_end"])))
    return [i for i in range(len(actions)) if i not in discarded]


def canonical_environment_actions(trace: Any) -> list[Any]:
    actions = trace.get("environment_actions", []) if isinstance(trace, dict) else trace.environment_actions
    return [actions[i] for i in canonical_action_indices(trace)]


def failure_prefix_action_indices(trace: Any, boundary: int) -> set[int]:
    """State at an historical failure, not the final successful branch.

    Only rollbacks already completed before this invocation affect its
    prefix. A later rollback must not erase the world in which it failed.
    This view is exclusively negative repair/replay evidence.
    """
    metadata = trace.get("metadata", {}) if isinstance(trace, dict) else trace.metadata
    indices = set(range(boundary))
    for rollback in metadata.get("runtime_rollbacks", []):
        start, end = rollback["discarded_action_start"], rollback["discarded_action_end"]
        if end <= boundary:
            indices.difference_update(range(start, end))
    return indices


def canonical_metadata_items(trace: Any, key: str) -> list[Any]:
    metadata = trace.get("metadata", {}) if isinstance(trace, dict) else trace.metadata
    discarded = set()
    for item in metadata.get("runtime_rollbacks", []):
        start, end = item.get("discarded_metadata_ranges", {}).get(key, (0, 0))
        discarded.update(range(start, end))
    return [item for i, item in enumerate(metadata.get(key, [])) if i not in discarded]


def canonical_trace_records(trace: Any, key: str) -> list[Any]:
    metadata = trace.get("metadata", {}) if isinstance(trace, dict) else trace.metadata
    values = trace.get(key, []) if isinstance(trace, dict) else getattr(trace, key)
    discarded = set()
    for rollback in metadata.get("runtime_rollbacks", []):
        start, end = rollback.get("discarded_record_ranges", {}).get(key, (0, 0))
        discarded.update(range(start, end))
    return [item for index, item in enumerate(values) if index not in discarded]
