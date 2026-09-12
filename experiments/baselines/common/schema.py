"""Common episode/result schema shared by every baseline method.

The schema follows the baseline design document, section 12.  Every episode of
every method must produce a ``CommonEpisodeRecord``; method-specific metrics
live in ``method_metrics`` and must never be merged into a synthetic skill
quality score.
"""

from __future__ import annotations

import json
from dataclasses import InitVar, asdict, dataclass, field
from typing import Any


_UNSET = object()


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
    contract_consistency: bool | None = None
    common_strict_success: bool | None = None

    environment_actions: int = 0
    # The SkillOpt upstream trace does not expose per-action validity; the
    # controller's strict post-evaluator fills this in by replaying the
    # action sequence through the Ours harness boundary.
    invalid_actions: int | None = None
    command_turns: int = 0
    timeout: bool = False
    termination_reason: str = ""

    target_llm_calls: int = 0
    target_prompt_tokens: int = 0
    target_completion_tokens: int = 0
    target_reasoning_tokens: int = 0

    evolution_llm_calls: int = 0
    evolution_prompt_tokens: int = 0
    evolution_completion_tokens: int = 0
    evolution_reasoning_tokens: int = 0

    embedding_calls: int = 0
    wall_time_ms: int = 0

    artifact_digest_before: str = ""
    artifact_digest_after: str = ""
    method_metrics: dict[str, Any] = field(default_factory=dict)

    infrastructure_failure: bool = False
    infrastructure_error: str = ""

    # Init-only compatibility arguments used by pre-freeze workers.  Properties
    # with these names are installed below after dataclass has captured the
    # constructor defaults, so there remains only one stored value per signal.
    task_contract_success: InitVar[bool | None | object] = _UNSET
    strict_success: InitVar[bool | None | object] = _UNSET

    def __post_init__(
        self,
        task_contract_success: bool | None | object,
        strict_success: bool | None | object,
    ) -> None:
        if task_contract_success is not _UNSET:
            if (
                task_contract_success is not None
                and self.contract_consistency is not None
                and task_contract_success is not self.contract_consistency
            ):
                raise ValueError(
                    "contract_consistency disagrees with task_contract_success"
                )
            self.contract_consistency = task_contract_success  # type: ignore[assignment]
        if strict_success is not _UNSET:
            if (
                strict_success is not None
                and self.common_strict_success is not None
                and strict_success is not self.common_strict_success
            ):
                raise ValueError(
                    "common_strict_success disagrees with strict_success"
                )
            self.common_strict_success = strict_success  # type: ignore[assignment]
        self.normalize_posthoc_outcome()

    def normalize_posthoc_outcome(self) -> None:
        """Reconcile legacy aliases and enforce the public success contract."""

        _require_optional_bool("contract_consistency", self.contract_consistency)
        _require_optional_bool("common_strict_success", self.common_strict_success)
        supplied_strict = self.common_strict_success
        if self.contract_consistency is None:
            if supplied_strict is not None:
                raise ValueError(
                    "common_strict_success requires contract_consistency evidence"
                )
        elif supplied_strict is not None:
            expected = bool(self.official_success) and self.contract_consistency
            if supplied_strict is not expected:
                raise ValueError(
                    "common_strict_success must equal official_success && "
                    "contract_consistency"
                )

    def set_posthoc_outcome(self, *, contract_consistency: bool) -> None:
        """Attach controller-only TaskContract analysis to this episode."""

        self.contract_consistency = bool(contract_consistency)
        self.common_strict_success = (
            bool(self.official_success) and self.contract_consistency
        )

    def to_dict(self) -> dict[str, Any]:
        self.normalize_posthoc_outcome()
        payload = asdict(self)
        payload["task_contract_success"] = self.contract_consistency
        payload["strict_success"] = self.common_strict_success
        return payload


def _require_optional_bool(name: str, value: bool | None) -> None:
    if value is not None and not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean or null")


def _legacy_contract_get(record: CommonEpisodeRecord) -> bool | None:
    return record.contract_consistency


def _legacy_contract_set(record: CommonEpisodeRecord, value: bool | None) -> None:
    record.contract_consistency = value


def _legacy_strict_get(record: CommonEpisodeRecord) -> bool | None:
    return record.common_strict_success


def _legacy_strict_set(record: CommonEpisodeRecord, value: bool | None) -> None:
    record.common_strict_success = value


# Runtime compatibility for code that reads or assigns the pre-freeze names.
CommonEpisodeRecord.task_contract_success = property(  # type: ignore[assignment]
    _legacy_contract_get, _legacy_contract_set
)
CommonEpisodeRecord.strict_success = property(  # type: ignore[assignment]
    _legacy_strict_get, _legacy_strict_set
)


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


@dataclass
class RetrievalEvent:
    episode_task_id: str
    query: str
    retrieved_ids: list[str]


@dataclass
class KnowledgeUseEvent:
    episode_task_id: str
    artifact_ids: list[str]


@dataclass
class DirectExecutionEvent:
    episode_task_id: str
    component: str
    succeeded: bool


@dataclass
class GraphMutationEvent:
    phase: str
    operation: str
    node_count: int
    edge_count: int


@dataclass
class ValidationEvent:
    candidate_index: int
    metric_calls: int
    aggregate_score: float


_SIDECAR_EVENT_KINDS = (
    "provider_call", "environment_action", "artifact_write", "retrieval",
    "knowledge_use", "direct_execution", "graph_mutation", "validation",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
