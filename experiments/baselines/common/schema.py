"""Common episode/result schema shared by every baseline method.

The schema follows the baseline design document, section 11.  Every episode of
every method must produce a ``CommonEpisodeRecord``; method-specific metrics
live in ``method_metrics`` and must never be merged into a synthetic skill
quality score.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CommonEpisodeRecord:
    """One method episode, mirroring the design document field-for-field."""

    method: str
    phase: str
    run_seed: int
    task_id: str
    task_type: str
    manifest_index: int
    gamefile: str
    gamefile_hash: str

    official_success: bool
    task_contract_success: bool | None = None
    strict_success: bool | None = None

    environment_actions: int = 0
    # The SkillOpt upstream trace does not expose per-action validity; the
    # controller's strict post-evaluator fills this in by replaying the
    # action sequence through the Ours harness boundary.
    invalid_actions: int | None = None
    command_turns: int = 0
    timeout: bool = False

    target_llm_calls: int = 0
    target_prompt_tokens: int = 0
    target_completion_tokens: int = 0
    target_reasoning_tokens: int = 0

    evolution_llm_calls: int = 0
    evolution_prompt_tokens: int = 0
    evolution_completion_tokens: int = 0

    embedding_calls: int = 0
    wall_time_ms: int = 0

    artifact_digest_before: str = ""
    artifact_digest_after: str = ""
    method_metrics: dict[str, Any] = field(default_factory=dict)

    infrastructure_failure: bool = False
    infrastructure_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProviderCallEvent:
    """Sidecar provider call (target or evolution role)."""

    episode_task_id: str
    role: str
    stage: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class EnvironmentActionEvent:
    """Sidecar environment action of one episode."""

    episode_task_id: str
    step_index: int
    action: str
    env_feedback: str
    reward: float
    done: bool


@dataclass
class ArtifactWriteEvent:
    """Sidecar persistent-artifact write (e.g. best_skill.md update)."""

    episode_task_id: str
    path: str
    content_sha256: str


_SIDECAR_EVENT_KINDS = ("provider_call", "environment_action", "artifact_write")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
