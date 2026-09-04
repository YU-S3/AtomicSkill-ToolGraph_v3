"""Composition root for the independent AtomicSkillGraph v3 implementation."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import yaml

from .agents import (
    AgentBudget,
    AgentTurn,
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
    ReplayAgentSession,
    UsageBucket,
    UsageLedger,
    structured_provider_turn_cap,
)
from .core.edges import GlobalGraphEdge, GlobalRelationType
from .core.contracts import AbstractAtomicSkill
from .core.errors import (
    AgentProtocolError,
    AtomicSkillGraphError,
    BudgetExhausted,
    FailureEnvelope,
    FailureLayer,
)
from .core.refs import content_hash
from .core.serialization import atomic_write_json, to_primitive
from .core.status import RuntimeMode, SkillStatus
from .evolution.admission import Admission
from .evolution.aligner import Aligner
from .evolution.atomicizer import Atomicizer
from .evolution.composite_builder import CompositeBuilder
from .evolution.composite_repair_session import CompositeSequenceProposalSession
from .evolution.composite_repairs import CompositeSequenceRepairEngine
from .evolution.extraction_authority import (
    contract_coverage_report,
    extraction_coverage_authority,
)
from .evolution.extractor_session import (
    ExtractionContentError,
    ExtractorSession,
)
from .evolution.evidence import EvolutionEvidenceAccumulator
from .evolution.failure_processor import FailureProcessor
from .evolution.failure_extraction_validator import (
    FailureAssetRecordBuilder,
    FailureAtomicSourceReplay,
    FailureExtractionCoordinator,
    FailureExtractionEligibility,
    PreparedFailureExtraction,
)
from .evolution.failure_extractor_session import (
    FailureExtractorSession,
    FailureExtractorSessionAllocation,
)
from .evolution.gap_diagnosis import GapDiagnoser
from .evolution.maintenance import (
    BatchMaintenanceResult,
    EvolutionMaintenance,
    ExtractionPolicy,
    _composite_plan,
)
from .evolution.portability import (
    occurrence_terms,
    relevant_known_atomic_contracts,
    resolve_capability_label_group,
    source_forbidden_terms,
    validate_portability,
)
from .evolution.repair import RepairProposal, RepairStore
from .evolution.repair_session import EvolutionRepairSession
from .evolution.trace_replay import TraceRepairExecutor
from .evolution.tool_compiler import (
    CompiledKnowledge,
    ToolCompiler,
    rewrite_capability_labels,
)
from .tooling.builder_session import ToolBuilderSession
from .tooling.proposal import ToolProvenance
from .tooling.validator import ToolStaticValidator
from .evolution.trace_normalizer import TraceNormalizer
from .evolution.provisional_promotion import (
    PreparedPromotion,
    ProvisionalPromotionCompiler,
    commit_prepared_promotion,
)
from .evolution.typed_repair_session import TypedRepairProposalSession
from .evolution.typed_repairs import TypedRepairEngine
from .governance import (
    CandidateUsePolicy,
    CreditAssigner,
    CreditOutcome,
    CreditAttempt,
    CreditTrace,
    EvidenceLedger,
    LifecycleController,
    LifecyclePolicy,
    LifecycleProjection,
    LifecycleThresholds,
)
from .harness.alfworld import AlfWorldAdapter, normalize_entity
from .harness.protocol import HarnessTask
from .knowledge import (
    ArtifactStore,
    FailureKnowledgeStore,
    GraphStore,
    SkillRegistry,
    StateDatabase,
    ToolRegistry,
)
from .knowledge.database import STATE_PATCH_LEVEL
from .planner import PlannerPipeline
from .planner.cold_start_agent import cold_start_plan_from_dict
from .planner.cold_start_retriever import (
    FailureExperienceRetriever,
    ProvisionalAtomicRetriever,
)
from .planner.requirement_agent import requirement_bundle_from_dict
from .runtime.cold_start_executor import ProvisionalTrialResult
from .runtime.invocation_compiler import InvocationCompiler
from .runtime.orchestrator import RuntimeOrchestrator, refresh_learning_eligibility
from .runtime.budget import required_runtime_turn_caps, validate_runtime_turn_caps
from .traces import (
    AgentSessionRecord,
    AgentTurnRecord,
    FailureExtractionRecord,
    ProviderRequestRecord,
    TaskRecord,
    TraceBuilder,
    TraceRecord,
    TraceStore,
)
from .validation import ValidationEngine
from .validation.contract_matcher import ExactContractMatcher


_SYSTEM_PROMPTS = {
    "planner": (
        "You are the AtomicSkillGraph v3 planner. Use only supplied contracts, candidates, and "
        "edge evidence. Call the single offered native submission tool exactly once; never execute "
        "an environment action or return a prose/JSON answer outside that ToolCall."
    ),
    "runtime_preparation": (
        "Exactly ONE native ToolCall per turn. You are a runtime preparation agent. Use only "
        "native tools offered in the current turn. Ground missing arguments through current "
        "environment evidence, then invoke at most one learned implementation."
    ),
    "runtime_seeded": (
        "Exactly ONE native ToolCall per turn. You are a fresh seeded runtime agent. Complete "
        "only the supplied Atomic contract using current native actions."
    ),
    "runtime_dynamic": (
        "Exactly ONE native ToolCall per turn. You are a fresh full-dynamic task agent. Solve "
        "the stated task using exactly one currently offered native action per turn."
    ),
    "extractor": (
        "You are the AtomicSkillGraph v3 two-turn extractor. Treat the canonical structured trace as authority; "
        "do not invent actions, effects, occurrences, or existing edges. In each turn call the single "
        "offered native submission tool exactly once; do not return prose or standalone JSON."
    ),
    "evolution_repair": (
        "You are the AtomicSkillGraph v3 batch evolution proposal agent. Use only supplied "
        "structured Tool/replay/failure evidence. Submit through the single offered native tool. You may propose semantic "
        "edits, but code is the sole replay, validation, versioning, and admission authority."
    ),
    "tool_builder": (
        "You are the v3.2 ToolBuilder sub-agent and the only Tool Program author. "
        "Implement exactly the supplied Atomic contract with a bounded declarative "
        "ACTION/IF/FOR_EACH/STOP_WHEN/RETURN Tool IR. Never emit Python, shell, "
        "filesystem, network, task-family, or episode-entity-specific code. "
        "Submit create_tool exactly once, or decision=no_tool when no safe reusable "
        "bounded implementation is justified."
    ),
}


def installed_alfworld_version() -> str:
    """Installed ALFWorld version, resolved without importing alfworld."""

    try:
        import importlib.metadata as importlib_metadata
        return str(importlib_metadata.version("alfworld"))
    except Exception:
        return ""


_LONG_TERM_KNOWLEDGE_TABLES = (
    "artifact_index",
    "recommended_pointers",
    "graph_edges",
    "evidence_events",
    "lifecycle_projection",
    "projection_checkpoints",
    "provisional_artifacts",
    "failure_experiences",
    "cold_start_evidence",
)


def load_config(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Load one v3 YAML configuration without interpolating secrets."""
    if isinstance(source, Mapping):
        config = copy.deepcopy(dict(source))
        config_path: Path | None = None
    else:
        config_path = Path(source).expanduser().resolve()
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("v3 config root must be a mapping")
        config = payload
        config["_config_path"] = str(config_path)
    if int(config.get("schema_version", 0)) != 3:
        raise ValueError("AtomicSkillGraph v3 requires schema_version: 3")
    method_patch = str(config.get("method_patch", "3.1"))
    if method_patch not in {"3.1", "3.2"}:
        raise ValueError("AtomicSkillGraph v3 requires method_patch: \"3.1\" or \"3.2\"")
    config["method_patch"] = method_patch
    llm = dict(config.get("llm") or {})
    if str(llm.get("provider", "openai_compatible")) != "openai_compatible":
        raise ValueError("v3 currently supports provider: openai_compatible only")
    forbidden_auth_keys = {
        "api_key", "apikey", "secret", "client_secret", "auth_token",
        "access_token", "bearer_token", "authorization", "proxy_authorization",
        "password", "cookie", "set_cookie",
    }
    found_auth = []
    stack: list[tuple[str, Any]] = [("llm", llm)]
    while stack:
        prefix, value = stack.pop()
        if not isinstance(value, Mapping):
            continue
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_")
            path = f"{prefix}.{key}"
            looks_like_auth = (
                normalized in forbidden_auth_keys
                or ("api" in normalized and "key" in normalized)
                or normalized.endswith("subscription_key")
                or normalized.endswith("access_token")
            )
            if normalized != "api_key_env" and looks_like_auth:
                found_auth.append(path)
            if isinstance(item, Mapping):
                stack.append((path, item))
    if found_auth:
        raise ValueError(
            "API credentials are forbidden in config; use api_key_env only: "
            + ", ".join(sorted(found_auth))
        )
    if not str(llm.get("api_key_env", "MODEL_API_KEY")).strip():
        raise ValueError("llm.api_key_env must be non-empty")
    condition = str(
        config.get("condition")
        or (config.get("experiment") or {}).get("condition")
        or "full"
    )
    if condition != "full":
        raise ValueError("this experiment implements exactly the full condition; ablations are out of scope")
    benchmark = str((config.get("experiment") or {}).get("benchmark", "alfworld"))
    if benchmark != "alfworld":
        raise ValueError("this v3 experiment is scoped to ALFWorld only")
    planner = dict(config.get("planner") or {})
    if int(planner.get("requirement_repair_limit", 1)) != 1:
        raise ValueError("v3 permits exactly one P1R requirement repair")
    if int(planner.get("graph_repair_limit", 1)) != 1:
        raise ValueError("v3 permits exactly one P2R graph repair")
    if int(planner.get("cold_start_c1_repair_limit", 1)) != 1:
        raise ValueError("v3.1 permits exactly one C1R cold-start repair")
    max_repeat_count = planner.get("max_repeat_count", 4)
    if isinstance(max_repeat_count, bool) or int(max_repeat_count) != 4:
        raise ValueError("v3.1 planner.max_repeat_count must be 4")
    if isinstance(planner.get("max_runtime_occurrences", 16), bool) or int(
        planner.get("max_runtime_occurrences", 16)
    ) != 16:
        raise ValueError("v3.1 planner.max_runtime_occurrences must remain 16")
    cold_start = dict(config.get("cold_start") or {})
    enabled = cold_start.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("cold_start.enabled must be boolean")
    expected_cold_start = {
        "provisional_top_k_per_requirement": 3,
        "failure_experience_top_k": 2,
        "scaffold_max_steps": 8,
        "failure_extractor_enabled": True,
        "source_replay_required": True,
        "provisional_suppress_consecutive_failures": 3,
        "promotion_requires_strict_task_success": True,
        "experience_confirm_independent_tasks": 2,
    }
    if cold_start:
        for key, expected in expected_cold_start.items():
            value = cold_start.get(key, expected)
            if isinstance(expected, bool):
                if value is not expected:
                    raise ValueError(f"cold_start.{key} must be {str(expected).lower()}")
            elif isinstance(value, bool) or int(value) != expected:
                raise ValueError(f"cold_start.{key} must be {expected}")
    runtime_mode = str((config.get("experiment") or {}).get("runtime_mode", "online"))
    if runtime_mode == "frozen" and enabled:
        raise ValueError("frozen config must set cold_start.enabled: false")
    return config


@dataclass
class _ObservedSession:
    session: ReplayAgentSession
    session_type: str
    occurrence_id: str
    task_id: str
    started_at: float
    turns: list[AgentTurn]


class _SessionProxy:
    """Observe Planner/Extractor turns that occur outside Runtime's TraceBuilder."""

    def __init__(self, observed: _ObservedSession) -> None:
        self._observed = observed

    def __getattr__(self, name: str) -> Any:
        return getattr(self._observed.session, name)

    def next_turn(self, *args: Any, **kwargs: Any) -> AgentTurn:
        turn = self._observed.session.next_turn(*args, **kwargs)
        self._observed.turns.append(turn)
        return turn

    def submit_tool_result(self, *args: Any, **kwargs: Any) -> AgentTurn:
        turn = self._observed.session.submit_tool_result(*args, **kwargs)
        self._observed.turns.append(turn)
        return turn


@dataclass
class _PreparedEvolution:
    compiled: list[Any]
    composite: Any | None
    gap_diagnosis: dict[str, Any]
    source_composite_ref: str
    composite_rejection: dict[str, str] | None = None


class AtomicSkillGraphSystem:
    """Wire Planner → Runtime → Validation → Evolution → Governance.

    Runtime paths never import v2 or FlowEvo.  External providers and harnesses
    are injectable solely for deterministic tests.
    """

    def __init__(
        self,
        config: str | Path | Mapping[str, Any],
        *,
        harness: Any | None = None,
        provider: Any | Mapping[str, Any] | None = None,
        readonly: bool | None = None,
    ) -> None:
        self.config = load_config(config)
        experiment_config = dict(self.config.get("experiment") or {})
        configured_frozen = (
            str(experiment_config.get("runtime_mode", "online")) == "frozen"
            or bool(experiment_config.get("freeze_skills", False))
        )
        if readonly is False and configured_frozen:
            raise ValueError("frozen experiment config cannot be opened with readonly=False")
        self.readonly = configured_frozen if readonly is None else bool(readonly)
        if configured_frozen:
            if experiment_config.get("allow_long_term_knowledge_writes") not in {None, False}:
                raise ValueError("frozen config must forbid long-term knowledge writes")
            if experiment_config.get("freeze_skills") is not True:
                raise ValueError("frozen config must set freeze_skills: true")
            if bool((self.config.get("cold_start") or {}).get("enabled", False)):
                raise ValueError("frozen config must set cold_start.enabled: false")
        self.mode = RuntimeMode.FROZEN if self.readonly else RuntimeMode.ONLINE
        config_path = Path(self.config.get("_config_path", Path.cwd() / "config.yaml"))
        self.repo_root = config_path.parent.parent if "_config_path" in self.config else Path.cwd()
        self.data_dir = self._resolve_path(self.config.get("data_dir", "data_v3"))
        experiment_output = (self.config.get("experiment") or {}).get("output_dir")
        default_trace_dir = (
            self._resolve_path(experiment_output) if experiment_output else
            self.data_dir if not self.readonly else self.repo_root / "runs_v3" / "frozen_eval"
        )
        self.trace_data_dir = self._resolve_path(self.config.get("trace_data_dir", default_trace_dir))
        if self.readonly and (
            self.trace_data_dir == self.data_dir
            or self.data_dir in self.trace_data_dir.parents
        ):
            raise ValueError(
                "frozen trace_data_dir must be outside the immutable snapshot root"
            )

        self.database = StateDatabase(self.data_dir / "state.sqlite3", readonly=self.readonly)
        self.artifacts = ArtifactStore(self.data_dir, self.database)
        try:
            self.artifacts.verify_all()
        except Exception:
            self.database.close()
            raise
        self.skills = SkillRegistry(self.artifacts, self.database)
        self.tools = ToolRegistry(self.artifacts, self.database)
        self.graph = GraphStore(self.database, self.skills)
        cold_start_config = dict(self.config.get("cold_start") or {})
        self.cold_start_enabled = bool(
            cold_start_config.get("enabled", False)
            and not self.readonly
        )
        self.failure_knowledge: FailureKnowledgeStore | None = (
            FailureKnowledgeStore(
                self.data_dir,
                self.database,
                experience_confirm_independent_tasks=int(
                    cold_start_config.get(
                        "experience_confirm_independent_tasks", 2,
                    )
                ),
            )
            if self.cold_start_enabled
            else None
        )
        self.provisional_retriever = (
            ProvisionalAtomicRetriever(
                self.failure_knowledge,
                top_k=int(cold_start_config.get(
                    "provisional_top_k_per_requirement", 3,
                )),
            )
            if self.failure_knowledge is not None
            else None
        )
        self.failure_experience_retriever = (
            FailureExperienceRetriever(
                self.failure_knowledge,
                top_k=int(cold_start_config.get("failure_experience_top_k", 2)),
            )
            if self.failure_knowledge is not None
            else None
        )
        self.traces = TraceStore(self.trace_data_dir, readonly=False)
        self.validation = ValidationEngine()
        self.usage = UsageLedger()

        harness_config = dict(self.config.get("harness") or {})
        experiment = experiment_config
        self.harness = harness or AlfWorldAdapter(
            split=str(experiment.get("split", harness_config.get("split", "train"))),
            max_steps=int(harness_config.get("max_steps", 100)),
            alfworld_data=harness_config.get("alfworld_data") or None,
        )
        self._provider_override = provider
        self._provider_cache: dict[str, Any] = {}
        self._observed_sessions: list[_ObservedSession] = []
        self._evolution_batch_usage_start: int | None = None
        self._current_task_id = ""

        lifecycle_config = dict(self.config.get("lifecycle") or {})
        threshold_names = {item.name for item in fields(LifecycleThresholds)}
        thresholds = LifecycleThresholds(**{
            key: value for key, value in lifecycle_config.items() if key in threshold_names
        })
        self.candidate_policy = CandidateUsePolicy(
            exploration_quota=float(lifecycle_config.get("candidate_exploration_quota", 0.15)),
            seed=lifecycle_config.get("candidate_exploration_seed", 0),
        )
        self.maintenance_interval = max(
            1, int(lifecycle_config.get("maintenance_interval_successes", 5))
        )
        success_row = self.database.execute(
            "SELECT value FROM metadata WHERE key='online_success_count'"
        ).fetchone()
        maintenance_row = self.database.execute(
            "SELECT value FROM metadata WHERE key='last_maintenance_success_count'"
        ).fetchone()
        self._online_successes = int(success_row["value"]) if success_row else 0
        self._last_maintenance_success_count = (
            int(maintenance_row["value"]) if maintenance_row else 0
        )
        if not 0 <= self._last_maintenance_success_count <= self._online_successes:
            raise RuntimeError("invalid persisted lifecycle maintenance milestone")

        self.ledger: EvidenceLedger | None = None
        self.projection: LifecycleProjection | None = None
        self.lifecycle: LifecycleController | None = None
        if not self.readonly:
            self.ledger = EvidenceLedger(self.database)
            self.projection = LifecycleProjection(self.database, self.ledger)
            self.lifecycle = LifecycleController(
                self.database, self.projection, LifecyclePolicy(thresholds)
            )

        planner_config = dict(self.config.get("planner") or {})
        self.planner = PlannerPipeline(
            self.skills,
            self.graph,
            self._planner_session,
            composite_top_k=int(planner_config.get("composite_top_k", 5)),
            atomic_top_k=int(planner_config.get("atomic_top_k_per_requirement", 3)),
            max_atomic_top_k=int(planner_config.get("max_atomic_top_k", 5)),
            max_occurrences=int(planner_config.get("max_runtime_occurrences", 16)),
            max_repeat_count=int(planner_config.get("max_repeat_count", 4)),
            candidate_policy=self.candidate_policy,
            cold_start_enabled=self.cold_start_enabled,
            provisional_retriever=self.provisional_retriever,
            failure_experience_retriever=self.failure_experience_retriever,
            cold_start_session_factory=self._cold_start_session,
            scaffold_max_steps=int(cold_start_config.get("scaffold_max_steps", 8)),
            cold_start_repair_limit=int(
                planner_config.get("cold_start_c1_repair_limit", 1)
            ),
        )
        self.invocation_compiler = InvocationCompiler(
            self.skills, self.tools, self.harness, mode=self.mode,
            candidate_policy=self.candidate_policy,
        )
        runtime_config = dict(self.config.get("runtime") or {})
        runtime_llm = self._stage_config("runtime")
        runtime_config.setdefault(
            "learned_toolcall_repair_limit",
            int(runtime_llm.get("learned_toolcall_repair_limit", 2)),
        )
        required_node_turns, required_task_turns = required_runtime_turn_caps(
            global_action_budget=int(runtime_config.get("global_action_budget", 100)),
            node_action_budget=int(runtime_config.get("node_action_budget", 35)),
            learned_toolcall_repair_limit=int(
                runtime_llm.get("learned_toolcall_repair_limit", 2)
            ),
            protocol_repair_limit=int(runtime_llm.get("protocol_repair_limit", 1)),
        )
        self._runtime_turn_caps = validate_runtime_turn_caps(
            global_action_budget=int(runtime_config.get("global_action_budget", 100)),
            node_action_budget=int(runtime_config.get("node_action_budget", 35)),
            learned_toolcall_repair_limit=int(
                runtime_llm.get("learned_toolcall_repair_limit", 2)
            ),
            protocol_repair_limit=int(runtime_llm.get("protocol_repair_limit", 1)),
            max_turns_per_node=int(
                runtime_llm.get("max_turns_per_node", required_node_turns)
            ),
            max_turns_per_task=int(
                runtime_llm.get("max_turns_per_task", required_task_turns)
            ),
        )
        self.orchestrator = RuntimeOrchestrator(
            self.planner,
            self.harness,
            self.invocation_compiler,
            self.validation,
            self._runtime_session,
            runtime_config=runtime_config,
            failure_knowledge=self.failure_knowledge,
        )
        extraction_config = dict(self.config.get("extraction") or {})
        self.extraction_policy = ExtractionPolicy(**{
            key: extraction_config.get(key, default)
            for key, default in {
                "extract_full_dynamic_success": True,
                "extract_task_rescue_success": True,
                "extract_novel_seeded_success": True,
                "skip_stable_direct_success": True,
            }.items()
        })
        self.credit = CreditAssigner()
        self.normalizer = TraceNormalizer()
        self.atomicizer = Atomicizer()
        self.tool_compiler = ToolCompiler()
        self.tool_static_validator = ToolStaticValidator()
        self.admission = Admission(self.validation.tool)
        self.aligner = Aligner(self.skills, self.tools)
        self.orchestrator.attach_runtime_automation(
            tool_builder_factory=self._tool_builder_session,
            tool_compiler=self.tool_compiler,
        )
        self.composite_builder = CompositeBuilder()
        self.failure_processor = FailureProcessor(self.validation.failure_localizer)
        self.gap_diagnoser = GapDiagnoser(self.skills)
        self.failure_extraction_coordinator: FailureExtractionCoordinator | None = (
            FailureExtractionCoordinator()
            if (
                self.cold_start_enabled
                and bool(cold_start_config.get("failure_extractor_enabled", True))
            )
            else None
        )
        self.provisional_promotion_compiler: ProvisionalPromotionCompiler | None = (
            ProvisionalPromotionCompiler(
                normalizer=self.normalizer,
                atomicizer=self.atomicizer,
                tool_compiler=self.tool_compiler,
                admission=self.admission,
                harness=self.harness,
            )
            if self.cold_start_enabled else None
        )
        self.provisional_suppress_consecutive_failures = int(
            cold_start_config.get("provisional_suppress_consecutive_failures", 3)
        )
        self.repair_store = None if self.readonly else RepairStore(self.database)
        self.evolution_maintenance = (
            None if self.readonly else EvolutionMaintenance(self.repair_store)
        )

    def _resolve_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.repo_root / path).resolve()

    def _stage_config(self, stage: str) -> dict[str, Any]:
        llm = dict(self.config.get("llm") or {})
        group = "runtime" if stage.startswith("runtime_") else stage
        return {**llm, **dict(llm.get(group) or {})}

    def _provider(self, stage: str) -> Any:
        override = self._provider_override
        if isinstance(override, Mapping):
            group = "runtime" if stage.startswith("runtime_") else stage
            selected = override.get(stage) or override.get(group) or override.get("default")
            if selected is None:
                raise KeyError(f"no injected provider for stage {stage}")
            return selected
        if override is not None:
            return override
        if stage not in self._provider_cache:
            cfg = self._stage_config(stage)
            protocol = dict(cfg.get("protocol") or {})
            provider_config = OpenAICompatibleConfig(
                base_url=str(cfg.get("base_url", "https://api.deepseek.com")),
                model=str(cfg.get("model", "")),
                api_key_env=str(cfg.get("api_key_env", "MODEL_API_KEY")),
                max_completion_tokens=int(cfg.get("max_completion_tokens", 32768)),
                dialect=str(cfg.get("dialect", "deepseek_v4_chat")),
                thinking_type=str(protocol.get("thinking_type", "enabled")),
                reasoning_effort=str(cfg.get("reasoning_effort", "high")),
                connect_timeout_seconds=float(cfg.get("connect_timeout_seconds", 15)),
                request_timeout_seconds=float(cfg.get("request_timeout_seconds", 120)),
                max_retries=int(cfg.get("max_retries", 4)),
                retry_backoff_seconds=float(cfg.get("retry_backoff_seconds", 2)),
                max_retry_after_seconds=float(cfg.get("max_retry_after_seconds", 30)),
            )
            self._provider_cache[stage] = OpenAICompatibleProvider(provider_config)
        return self._provider_cache[stage]

    def _new_session(
        self,
        *,
        stage: str,
        bucket: UsageBucket | str,
        session_type: str,
        occurrence_id: str,
        task_id: str,
        max_turns: int,
        max_tokens: int,
        exhaustion_code: str,
        semantic_max_turns: int | None = None,
    ) -> _SessionProxy:
        session = ReplayAgentSession(
            self._provider(stage),
            system_prompt=_SYSTEM_PROMPTS[stage],
            usage_ledger=self.usage,
            usage_bucket=bucket,
            budget=AgentBudget(max_turns, max_tokens, exhaustion_code),
            semantic_max_turns=semantic_max_turns,
        )
        observed = _ObservedSession(
            session, session_type, occurrence_id, task_id, time.time(), []
        )
        self._observed_sessions.append(observed)
        return _SessionProxy(observed)

    def _shared_tool_builder_tokens(self, session_kind: str) -> int:
        """ToolBuilder never creates a new implicit learning budget pool.

        Shared caps are task-scoped.  The process-wide UsageLedger may contain
        earlier episodes, so only events from the current task boundary are
        charged against the remaining allocation.
        """

        if session_kind == "tool_builder_runtime" or session_kind.startswith("runtime"):
            shared_buckets = (
                UsageBucket.RUNTIME_PREPARATION,
                UsageBucket.RUNTIME_SEEDED,
                UsageBucket.RUNTIME_DYNAMIC,
                UsageBucket.RUNTIME_PROVISIONAL_SEEDED,
                UsageBucket.RUNTIME_DYNAMIC_COLD_START_CONTINUATION,
                UsageBucket.TOOL_BUILDER_RUNTIME,
            )
            cap = int(
                self._stage_config("runtime").get(
                    "max_total_tokens_per_task", 300000,
                )
            )
        else:
            shared_buckets = (
                UsageBucket.EXTRACTOR_E1,
                UsageBucket.EXTRACTOR_E2,
                UsageBucket.TOOL_BUILDER_EVOLUTION,
            )
            cap = int(
                self._stage_config("extractor").get(
                    "max_total_tokens_per_task", 262144,
                )
            )
        events = list(self.usage.events)
        start = int(getattr(self, "_current_task_usage_start", 0) or 0)
        current_events = events[max(0, start):]
        used = sum(
            event.usage.total_tokens
            for event in current_events
            if event.bucket in shared_buckets
        )
        return max(0, cap - int(used))

    def _tool_builder_session(self, session_kind: str, occurrence_id: str) -> _SessionProxy:
        cfg = self._stage_config("tool_builder")
        bucket = (
            UsageBucket.TOOL_BUILDER_RUNTIME
            if session_kind.startswith("runtime")
            else UsageBucket.TOOL_BUILDER_EVOLUTION
        )
        return self._new_session(
            stage="tool_builder", bucket=bucket,
            session_type="ToolBuilderSession", occurrence_id=occurrence_id,
            task_id=str(getattr(self, "_current_task_id", "")),
            max_turns=structured_provider_turn_cap(1),
            max_tokens=int(self._shared_tool_builder_tokens(session_kind)),
            exhaustion_code="tool_builder_token_budget_exhausted",
        )

    def _planner_session(self, task: HarnessTask, _contract: Any) -> _SessionProxy:
        cfg = self._stage_config("planner")
        semantic_max_turns = int(cfg.get("max_turns", 4))
        return self._new_session(
            stage="planner", bucket=UsageBucket.PLANNER_P1,
            session_type="PlannerSession", occurrence_id="", task_id=task.task_id,
            max_turns=structured_provider_turn_cap(semantic_max_turns),
            max_tokens=int(cfg.get("max_total_tokens_per_task", 120000)),
            exhaustion_code="planner_token_budget_exhausted",
            semantic_max_turns=semantic_max_turns,
        )

    def _cold_start_session(self, task: HarnessTask, _contract: Any) -> _SessionProxy:
        cfg = self._stage_config("planner")
        # C1 + the sole C1R are independent from P1/P2 and use a fresh
        # Planner conversation while retaining the frozen Planner budget.
        semantic_max_turns = 2
        return self._new_session(
            stage="planner",
            bucket=UsageBucket.COLD_START_C1,
            session_type="ColdStartPlannerSession",
            occurrence_id="",
            task_id=task.task_id,
            max_turns=structured_provider_turn_cap(semantic_max_turns),
            max_tokens=int(cfg.get("max_total_tokens_per_task", 120000)),
            exhaustion_code="planner_token_budget_exhausted",
            semantic_max_turns=semantic_max_turns,
        )

    def _runtime_session(self, session_kind: str, occurrence_id: str) -> _SessionProxy:
        cfg = self._stage_config("runtime")
        bucket = UsageBucket(session_kind)
        task_level = session_kind in {
            "runtime_dynamic", "runtime_dynamic_cold_start_continuation",
        }
        token_name = "max_total_tokens_per_task" if task_level else "max_total_tokens_per_node"
        return self._new_session(
            stage=(
                "runtime_dynamic"
                if session_kind == "runtime_dynamic_cold_start_continuation"
                else "runtime_seeded"
                if session_kind == "runtime_provisional_seeded"
                else session_kind
            ), bucket=bucket,
            session_type={
                "runtime_preparation": "RuntimePreparationSession",
                "runtime_seeded": "SeededSession",
                "runtime_dynamic": "DynamicTaskSession",
                "runtime_provisional_seeded": "ProvisionalSeededSession",
                "runtime_dynamic_cold_start_continuation": "ColdStartDynamicContinuationSession",
            }[session_kind],
            occurrence_id=occurrence_id, task_id=self._current_task_id,
            max_turns=(
                self._runtime_turn_caps[1]
                if task_level
                else self._runtime_turn_caps[0]
            ),
            max_tokens=int(cfg.get(token_name, cfg.get("max_total_tokens_per_node", 80000))),
            exhaustion_code=(
                "runtime_task_token_budget_exhausted"
                if task_level
                else "runtime_node_token_budget_exhausted"
            ),
        )

    def _extractor_session(self, task_id: str) -> _SessionProxy:
        cfg = self._stage_config("extractor")
        semantic_max_turns = int(cfg.get("max_turns", 2))
        return self._new_session(
            stage="extractor", bucket=UsageBucket.EXTRACTOR_E1,
            session_type="ExtractorSession", occurrence_id="", task_id=task_id,
            max_turns=structured_provider_turn_cap(semantic_max_turns),
            max_tokens=int(cfg.get("max_total_tokens_per_task", cfg.get("max_completion_tokens", 131072) * 2)),
            exhaustion_code="extractor_token_budget_exhausted",
            semantic_max_turns=semantic_max_turns,
        )

    def _failure_extractor_token_cap(self) -> int:
        cfg = self._stage_config("extractor")
        return int(cfg.get(
            "max_total_tokens_per_task",
            cfg.get("max_completion_tokens", 131072) * 2,
        ))

    def _failure_extractor_f1_session(self, task_id: str) -> _SessionProxy:
        # F1 and F2 are deliberately separate conversations.  The only shared
        # resource is the task-level cap, reconstructed from UsageLedger after
        # F1 rather than from a second mutable counter.
        semantic_max_turns = 1
        return self._new_session(
            stage="extractor", bucket=UsageBucket.FAILURE_EXTRACTOR_F1,
            session_type="FailureExtractorF1Session", occurrence_id="",
            task_id=task_id,
            max_turns=structured_provider_turn_cap(semantic_max_turns),
            max_tokens=self._failure_extractor_token_cap(),
            exhaustion_code="extractor_token_budget_exhausted",
            semantic_max_turns=semantic_max_turns,
        )

    def _failure_extractor_f2_allocation(
        self,
        task_id: str,
        f1_session_id: str,
    ) -> FailureExtractorSessionAllocation:
        used = sum(
            event.usage.total_tokens
            for event in self.usage.events
            if event.session_id == f1_session_id
            and event.bucket is UsageBucket.FAILURE_EXTRACTOR_F1
        )
        remaining = max(0, self._failure_extractor_token_cap() - used)
        if remaining == 0:
            return FailureExtractorSessionAllocation(None, 0)
        semantic_max_turns = 1
        session = self._new_session(
            stage="extractor", bucket=UsageBucket.FAILURE_EXTRACTOR_F2,
            session_type="FailureExtractorF2Session", occurrence_id="",
            task_id=task_id,
            max_turns=structured_provider_turn_cap(semantic_max_turns),
            max_tokens=remaining,
            exhaustion_code="extractor_token_budget_exhausted",
            semantic_max_turns=semantic_max_turns,
        )
        return FailureExtractorSessionAllocation(session, remaining)

    def _evolution_repair_session(self, task_id: str) -> _SessionProxy:
        cfg = self._stage_config("evolution_repair")
        semantic_max_turns = int(cfg.get("max_turns", 1))
        batch_limit = int(
            cfg.get(
                "max_total_tokens_per_batch",
                cfg.get("max_completion_tokens", 32768),
            )
        )
        if self._evolution_batch_usage_start is None:
            remaining = batch_limit
        else:
            used = sum(
                event.usage.total_tokens
                for event in self.usage.events[
                    self._evolution_batch_usage_start:
                ]
                if event.bucket is UsageBucket.EVOLUTION_REPAIR
            )
            remaining = max(0, batch_limit - used)
        return self._new_session(
            stage="evolution_repair",
            bucket=UsageBucket.EVOLUTION_REPAIR,
            session_type="EvolutionRepairSession",
            occurrence_id="",
            task_id=task_id,
            max_turns=structured_provider_turn_cap(semantic_max_turns),
            # Every proposal producer in one maintenance milestone receives
            # only the unspent portion of the single batch-wide token cap.
            max_tokens=remaining,
            exhaustion_code="evolution_repair_token_budget_exhausted",
            semantic_max_turns=semantic_max_turns,
        )

    @staticmethod
    def _provisional_trials(trace: TraceRecord) -> list[ProvisionalTrialResult]:
        trials: list[ProvisionalTrialResult] = []
        seen_steps: set[str] = set()
        for raw in trace.metadata.get("provisional_trials", ()):
            if not isinstance(raw, Mapping):
                raise TypeError("provisional trial Trace payload must be a mapping")
            action_span = tuple(map(int, raw.get("action_span", ())))
            if len(action_span) != 2:
                raise ValueError("provisional trial action_span must contain two indexes")
            trial = ProvisionalTrialResult(
                provisional_ref=str(raw["provisional_ref"]),
                step_id=str(raw["step_id"]),
                local_effect_passed=bool(raw["local_effect_passed"]),
                progress_before_digest=str(raw["progress_before_digest"]),
                progress_after_digest=str(raw["progress_after_digest"]),
                action_span=action_span,
                witness_refs=list(map(str, raw.get("witness_refs", ()))),
                failure_code=str(raw.get("failure_code", "")),
                resolved_bindings=dict(raw.get("resolved_bindings") or {}),
            )
            if trial.step_id in seen_steps:
                raise ValueError("a cold-start step may record only one provisional trial")
            seen_steps.add(trial.step_id)
            trials.append(trial)
        return trials

    def _cold_start_authority(
        self,
        trace: TraceRecord,
        task: HarnessTask,
    ) -> tuple[Any, Any, Any]:
        if trace.cold_start_plan is None:
            raise ValueError("failure extraction requires a ColdStartPlan record")
        contract = self.harness.task_contract(task)
        bundle = requirement_bundle_from_dict(dict(trace.requirement_bundle))
        expansion = self.planner.multiplicity_compiler.expand(bundle, contract)
        proposal = cold_start_plan_from_dict(
            dict(trace.cold_start_plan.proposal)
        )
        if to_primitive(expansion) != dict(trace.requirement_expansion):
            raise RuntimeError(
                "Trace RequirementExpansion does not match the frozen P1 bundle"
            )
        return contract, expansion, proposal

    def _cold_candidate_contract_views(self, proposal: Any) -> list[dict[str, Any]]:
        views: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for step in proposal.steps:
            source = str(getattr(step.candidate_source, "value", step.candidate_source))
            ref = str(step.candidate_ref)
            key = (source, ref)
            if key in seen:
                continue
            seen.add(key)
            if source == "verified":
                atomic = self.skills.get_atomic(ref)
                contract = {
                    "summary": atomic.summary,
                    "inputs": atomic.inputs,
                    "outputs": atomic.outputs,
                    "preconditions": atomic.preconditions,
                    "effects": atomic.effects,
                    "validator_spec": atomic.validator_spec,
                }
            elif source == "provisional":
                if self.failure_knowledge is None:
                    raise RuntimeError(
                        "Provisional C1 step exists without failure-side storage"
                    )
                contract = to_primitive(
                    self.failure_knowledge.provisional_candidate_view(ref)
                )
            else:
                contract = {
                    "expected_effects": to_primitive(step.expected_effects),
                    "unresolved": True,
                }
            views.append({
                "candidate_source": source,
                "candidate_ref": ref,
                "contract": to_primitive(contract),
            })
        return views

    def _prepare_failure_extraction(
        self,
        trace: TraceRecord,
        task: HarnessTask,
        *,
        run_mode: RuntimeMode,
    ) -> PreparedFailureExtraction | None:
        valid_plan = bool(
            trace.cold_start_plan is not None
            and dict(trace.cold_start_plan.validation).get("passed") is True
        )
        eligibility = FailureExtractionEligibility(
            cold_start_enabled=self.cold_start_enabled,
            valid_cold_start_plan=valid_plan,
            strict_task_success=trace.strict_task_success,
            infrastructure_failure=trace.infrastructure_failure,
            runtime_mode=run_mode.value,
        )
        trace.metadata["failure_extractor_eligible"] = bool(
            eligibility.passed
        )
        trace.metadata["failure_extractor_eligibility"] = {
            "cold_start_enabled": eligibility.cold_start_enabled,
            "valid_cold_start_plan": eligibility.valid_cold_start_plan,
            "strict_task_success": eligibility.strict_task_success,
            "infrastructure_failure": eligibility.infrastructure_failure,
            "runtime_mode": str(eligibility.runtime_mode),
            "passed": eligibility.passed,
        }
        if self.failure_extraction_coordinator is None or not eligibility.passed:
            return None
        contract, expansion, proposal = self._cold_start_authority(trace, task)
        source_replay = FailureAtomicSourceReplay(
            trace=trace,
            task=task,
            normalizer=self.normalizer,
            atomicizer=self.atomicizer,
            tool_compiler=self.tool_compiler,
            admission=self.admission,
            harness=self.harness,
        )
        record_builder = FailureAssetRecordBuilder(
            source_replay,
            task_contract=contract,
            requirement_expansion=expansion,
            cold_start_plan=proposal,
            trace=trace,
            harness_profile=str(self.harness.profile_name),
        )
        f1_session = self._failure_extractor_f1_session(task.task_id)
        f1_session_id = str(f1_session.session_id)
        extractor = FailureExtractorSession(
            f1_session,
            lambda: self._failure_extractor_f2_allocation(
                task.task_id, f1_session_id,
            ),
        )
        prepared = self.failure_extraction_coordinator.prepare(
            eligibility=eligibility,
            extractor=extractor,
            task_contract=contract,
            requirement_expansion=expansion,
            cold_start_plan=proposal,
            trace=trace,
            task_progress=trace.task_progress_records,
            failures=trace.failures,
            candidate_contracts=self._cold_candidate_contract_views(proposal),
            source_replay=source_replay,
            record_builder=record_builder,
        )
        f1_events = [
            event for event in self.usage.events
            if event.session_id == extractor.f1_session_id
            and event.bucket is UsageBucket.FAILURE_EXTRACTOR_F1
        ]
        f2_events = [
            event for event in self.usage.events
            if extractor.f2_session_id
            and event.session_id == extractor.f2_session_id
            and event.bucket is UsageBucket.FAILURE_EXTRACTOR_F2
        ]
        rejection_code = str(prepared.rejection.get("code") or "")
        rejection_stage = str(prepared.rejection.get("stage") or "")
        if rejection_stage.startswith("f1"):
            rejected_usage_persisted = bool(f1_events)
        elif rejection_stage == "f2_not_started_no_remaining_budget":
            rejected_usage_persisted = bool(f1_events)
        elif rejection_stage.startswith("f2"):
            rejected_usage_persisted = bool(f2_events)
        else:
            rejected_usage_persisted = bool(f1_events or f2_events)
        failure_extractor_metrics = dict(prepared.diagnostics)
        failure_extractor_metrics.update({
            # These four values are snapshots of the authoritative UsageLedger,
            # not an independently updated token/call counter.
            "failure_extractor_f1_tokens": sum(
                event.usage.total_tokens for event in f1_events
            ),
            "failure_extractor_f2_tokens": sum(
                event.usage.total_tokens for event in f2_events
            ),
            "failure_extractor_f1_provider_call_count": sum(
                event.usage.call_count for event in f1_events
            ),
            "failure_extractor_f2_provider_call_count": sum(
                event.usage.call_count for event in f2_events
            ),
            "failure_extractor_budget_exhausted_count": int(
                rejection_code == "failure_extractor_budget_exhausted"
            ),
            "failure_extractor_usage_persisted_after_rejection_count": int(
                rejection_code == "failure_extractor_budget_exhausted"
                and rejected_usage_persisted
            ),
        })
        failure_extractor_metrics.setdefault(
            "failure_extractor_skipped_after_budget_count", 0,
        )
        trace.metadata["failure_extractor_metrics"] = (
            failure_extractor_metrics
        )
        trace.failure_extraction = FailureExtractionRecord(
            f1_alignment=to_primitive(prepared.alignment) if prepared.alignment else {},
            f1_validation=(
                to_primitive(prepared.f1_validation)
                if prepared.f1_validation else {}
            ),
            f2_proposal=(
                {
                    "proposal": to_primitive(
                        prepared.f2_validation.proposal
                    ),
                    "provisional_rejections": list(
                        prepared.f2_validation.provisional_rejections
                    ),
                    "failure_experience_accepted": bool(
                        prepared.f2_validation.failure_experience_accepted
                    ),
                }
                if prepared.f2_validation else {}
            ),
            provisional_refs=[],
            failure_experience_ids=[],
            rejection=copy.deepcopy(prepared.rejection),
        )
        return prepared

    def _prepare_provisional_promotions(
        self,
        trace: TraceRecord,
        task: HarnessTask,
        trials: list[ProvisionalTrialResult],
    ) -> list[PreparedPromotion]:
        if (
            self.provisional_promotion_compiler is None
            or self.failure_knowledge is None
            or not trace.strict_task_success
        ):
            return []
        prepared = self.provisional_promotion_compiler.prepare(
            trace,
            [item for item in trials if item.local_effect_passed],
            provisional_lookup=self.failure_knowledge.get_provisional,
            task=task,
        )
        trace.provisional_promotions.extend({
            "provisional_ref": item.provisional_ref,
            "status": "rejected",
            "code": item.code,
            "detail": item.detail,
        } for item in self.provisional_promotion_compiler.last_rejections)
        return prepared

    def run_task(
        self,
        task: HarnessTask,
        *,
        mode: RuntimeMode | str | None = None,
        attempt_id: str = "",
    ) -> TraceRecord:
        run_mode = RuntimeMode(mode or self.mode)
        if self.readonly and run_mode is not RuntimeMode.FROZEN:
            raise RuntimeError("a read-only knowledge snapshot may run only in frozen mode")
        if not self.readonly and run_mode is RuntimeMode.FROZEN:
            raise RuntimeError("frozen evaluation must open a read-only snapshot")
        self._current_task_id = task.task_id
        self.invocation_compiler.mode = run_mode
        usage_start = len(self.usage.events)
        self._current_task_usage_start = int(usage_start)
        sessions_start = len(self._observed_sessions)
        failure_side_read_start = (
            self.failure_knowledge.failure_side_read_count
            if self.failure_knowledge is not None else 0
        )
        trace_builder = self.orchestrator.create_trace_builder(
            task, attempt_id=attempt_id,
        )
        trace_builder.trace.metadata["method_patch"] = str(
            self.config.get("method_patch", "3.1")
        )
        trace_builder.trace.metadata.setdefault("environment", {}).update({
            "alfworld_version": installed_alfworld_version(),
        })
        provider_offsets = self._provider_request_offsets()
        try:
            return self._run_task_pipeline(
                task,
                run_mode=run_mode,
                trace_builder=trace_builder,
                usage_start=usage_start,
                sessions_start=sessions_start,
                provider_offsets=provider_offsets,
                failure_side_read_start=failure_side_read_start,
            )
        except Exception as primary:
            try:
                self._persist_failure_trace(
                    trace_builder,
                    primary,
                    attempt_id=attempt_id,
                    usage_start=usage_start,
                    sessions_start=sessions_start,
                    provider_offsets=provider_offsets,
                    failure_side_read_start=failure_side_read_start,
                )
            except Exception as audit_error:
                if hasattr(primary, "add_note"):
                    primary.add_note(
                        "failure Trace finalization also failed: "
                        + self._sanitize_failure_message(audit_error)
                    )
            raise
        finally:
            self._current_task_id = ""
            self._current_task_usage_start = 0

    def _run_task_pipeline(
        self,
        task: HarnessTask,
        *,
        run_mode: RuntimeMode,
        trace_builder: TraceBuilder,
        usage_start: int,
        sessions_start: int,
        provider_offsets: dict[int, int],
        failure_side_read_start: int,
    ) -> TraceRecord:

        trace = self.orchestrator.run_task(
            task, mode=run_mode, trace_builder=trace_builder,
            attempt_id=str(trace_builder.trace.metadata.get("attempt_id", "")),
        )
        # Internal HTTP retries are individually auditable.  If any attempt
        # lacks provider usage, the episode is not a valid formal success and
        # must stop before Extractor/Evolution or credit can mutate knowledge.
        self._attach_provider_requests(trace, provider_offsets)
        self._require_resource_usage_complete(trace)
        refresh_learning_eligibility(trace)
        trace.runtime_plan["failure_stage"] = "evolution"
        self.failure_processor.localize(trace)
        provisional_trials = self._provisional_trials(trace)
        prepared_failure = self._prepare_failure_extraction(
            trace, task, run_mode=run_mode,
        )
        # Promotion staging is independent of the normal success Extractor
        # and intentionally happens first.  It recompiles only current-task
        # successful trial spans and never constructs a Composite.
        prepared_promotions = self._prepare_provisional_promotions(
            trace, task, provisional_trials,
        )
        decision = self.extraction_policy.decide(trace)
        trace.extraction_policy = {
            "should_extract": bool(decision.should_extract and run_mode is RuntimeMode.ONLINE),
            "reasons": decision.reasons if run_mode is RuntimeMode.ONLINE else ["frozen_mode_disabled"],
        }
        prepared: _PreparedEvolution | None = None
        if (
            run_mode is RuntimeMode.ONLINE
            and not trace.infrastructure_failure
            and trace.learning_eligible
            and decision.should_extract
        ):
            trace.metadata["extraction"] = {
                "attempted": True,
                "stage": "e1",
                "prepared": False,
                "error_code": "",
                "error_type": "",
                "error": "",
            }
            try:
                prepared = self._prepare_evolution(trace, task)
                composite_prepared = (
                    getattr(prepared, "composite", None) is not None
                )
                composite_rejection = dict(
                    getattr(prepared, "composite_rejection", None) or {}
                )
                trace.metadata["extraction"] = {
                    **dict(trace.metadata.get("extraction") or {}),
                    "attempted": True,
                    "stage": (
                        "prepared" if composite_prepared
                        else "atomic_prepared"
                    ),
                    "prepared": True,
                    "applied": False,
                    "atomic_occurrence_count": len(prepared.compiled),
                    "atomic_prepared": bool(prepared.compiled),
                    "composite_prepared": composite_prepared,
                    "partial_atomic_admission": not composite_prepared,
                    "error_code": str(
                        composite_rejection.get("error_code", "")
                    ),
                    "error_type": str(
                        composite_rejection.get("error_type", "")
                    ),
                    "error": str(composite_rejection.get("error", "")),
                }
            except (AgentProtocolError, ValueError, BudgetExhausted) as exc:
                # Extractor proposals may be rejected either by the native
                # submission protocol or by deterministic Atomic/Composite
                # validators.  The Extractor's own configured token exhaustion
                # is likewise a learning-only rejection: discard staged
                # Evolution and preserve the completed task Trace.
                # Infrastructure, persistence, programming, and unexpected
                # errors still propagate so the runner rolls back the task
                # checkpoint.
                if (
                    isinstance(exc, BudgetExhausted)
                    and exc.code != "extractor_token_budget_exhausted"
                ):
                    raise
                current = dict(trace.metadata.get("extraction") or {})
                stage = str(
                    getattr(exc, "stage", "")
                    or current.get("stage")
                    or "e1"
                )
                error_code = str(
                    getattr(exc, "error_code", "")
                    or getattr(exc, "code", "")
                )
                if error_code == "runtime_agent_schema_error":
                    error_code = f"extractor_{stage}_schema_rejected"
                if not error_code:
                    error_code = (
                        "extractor_e1_occurrence_rejected"
                        if stage == "e1"
                        else "extractor_e2_composite_validation_failed"
                    )
                trace.metadata["extraction"] = {
                    **current,
                    "attempted": True,
                    "stage": stage,
                    "prepared": False,
                    "applied": False,
                    "error_type": type(exc).__name__,
                    "error_code": error_code,
                    "error": self._sanitize_failure_message(exc),
                }

        repair_proposals = []
        if run_mode is RuntimeMode.ONLINE:
            if trace.infrastructure_failure:
                trace.metadata["evolution_branch"] = "infrastructure_neutral"
            elif trace.learning_eligible:
                trace.metadata["evolution_branch"] = "success"
                source_composite_ref = str(
                    trace.runtime_plan.get("source_composite_ref") or ""
                )
                if (
                    trace.task_rescue_required
                    and prepared is not None
                    and getattr(prepared, "composite", None) is not None
                    and source_composite_ref
                ):
                    assert self.evolution_maintenance is not None
                    repair_proposals = [
                        self.evolution_maintenance.prepare_validated_composite_repair(
                            source_composite_ref,
                            prepared.composite,
                            [
                                item.failure_id
                                for item in trace.failures
                                if item.layer is FailureLayer.COMPOSITE
                            ],
                            operation=self._composite_rescue_operation(
                                source_composite_ref, prepared.composite,
                            ),
                        )
                    ]
            elif trace.benchmark_success:
                trace.metadata["evolution_branch"] = "anomaly"
                assert self.evolution_maintenance is not None
                repair_proposals = self.evolution_maintenance.prepare_failure_repairs(
                    trace.failures,
                    trace=trace,
                    tools=self.tools,
                    skills=self.skills,
                    harness_profile=str(self.harness.profile_name),
                )
            else:
                trace.metadata["evolution_branch"] = "failure"
                assert self.evolution_maintenance is not None
                repair_proposals = self.evolution_maintenance.prepare_failure_repairs(
                    trace.failures,
                    trace=trace,
                    tools=self.tools,
                    skills=self.skills,
                    harness_profile=str(self.harness.profile_name),
                )
            if repair_proposals:
                trace.metadata["repair_proposals"] = [
                    to_primitive(item) for item in repair_proposals
                ]

        self._attach_external_sessions(
            trace, self._observed_sessions[sessions_start:]
        )
        task_usage = self.usage.events[usage_start:]
        trace.llm_usage = [event.to_dict() for event in task_usage]
        trace.metadata["usage_reconciliation"] = _reconcile_events(task_usage)
        self._attach_provider_requests(trace, provider_offsets)
        self._require_resource_usage_complete(trace)
        refresh_learning_eligibility(trace)
        if self._provider_override is None:
            _require_formal_usage(task_usage, trace.agent_turns)

        runtime_events = (
            self.credit.assign(trace)
            if run_mode is RuntimeMode.ONLINE and not trace.infrastructure_failure
            else []
        )
        trace.evidence_event_refs = [event.event_id for event in runtime_events]

        failure_metrics, promotion_events = self._commit_failure_side_task_evidence(
            trace,
            task,
            trials=provisional_trials,
            promotions=prepared_promotions,
            failure_extraction=prepared_failure,
            run_mode=run_mode,
        )
        self._finalize_v31_metrics(
            trace,
            failure_side_read_start=failure_side_read_start,
            counters=failure_metrics,
        )

        # Extractor validation and Evolution admission are part of this task's
        # terminal outcome.  Complete them before publishing the immutable
        # success Trace so an exception is persisted on the original skeleton
        # rather than leaving a success-looking Trace for a failed attempt.
        if run_mode is RuntimeMode.ONLINE:
            if trace.infrastructure_failure:
                pass
            elif trace.learning_eligible:
                if prepared is not None:
                    applied = self._apply_evolution(prepared, trace, task)
                    composite_applied = bool(applied["composite_validated"])
                    trace.metadata["extraction"] = {
                        **dict(trace.metadata.get("extraction") or {}),
                        "attempted": True,
                        "stage": (
                            "applied" if composite_applied
                            else "atomic_applied"
                        ),
                        "prepared": True,
                        "applied": True,
                        "atomic_applied": bool(applied["atomic_refs"]),
                        "composite_applied": composite_applied,
                        "partial_atomic_admission": not composite_applied,
                    }
                    trace.metadata["evolution_applied"] = {
                        "atomic_refs": [str(item) for item in applied["atomic_refs"]],
                        "implementation_refs": [
                            str(item) for item in applied["implementation_refs"]
                        ],
                        "tool_refs": [str(item) for item in applied["tool_refs"]],
                        "composite_ref": (
                            str(applied["composite_ref"])
                            if applied["composite_ref"] is not None
                            else ""
                        ),
                        "composite_validated": composite_applied,
                        "partial_atomic_admission": not composite_applied,
                    }
                    self._commit_gap_diagnosis(prepared.gap_diagnosis)
                    if repair_proposals:
                        assert self.evolution_maintenance is not None
                        for proposal in repair_proposals:
                            self.evolution_maintenance.admit_validated_composite_repair(
                                proposal,
                                admitted_ref=str(applied["composite_ref"]),
                                validation_passed=bool(applied["composite_validated"]),
                            )
            elif trace.benchmark_success:
                assert self.evolution_maintenance is not None
                self.evolution_maintenance.commit_repairs(repair_proposals)
            else:
                assert self.evolution_maintenance is not None
                self.evolution_maintenance.commit_repairs(repair_proposals)

        trace.runtime_plan["failure_stage"] = ""
        self._finalize_v31_metrics(
            trace,
            failure_side_read_start=failure_side_read_start,
            counters=failure_metrics,
        )
        self.traces.save_atomic(trace)

        if run_mode is RuntimeMode.ONLINE:
            if trace.learning_eligible:
                self._online_successes += 1
            if not trace.infrastructure_failure:
                # Runtime and provisional-promotion credit remain Trace-first:
                # the immutable evidence source already names every event id
                # before one append-only ledger transaction publishes them.
                self._commit_evidence([*runtime_events, *promotion_events])
            self._maybe_run_maintenance()
            self._persist_maintenance_state()
        return trace

    def _apply_prepared_promotion(
        self,
        prepared: PreparedPromotion,
        trace: TraceRecord,
        task: HarnessTask,
    ) -> tuple[tuple[str, str, str], list[Any]]:
        compiled = prepared.compiled
        atomic_ref = self.aligner.align_atomic(compiled.atomic)
        tool_alignment = self.aligner.align_tool_with_replays(
            compiled.tool,
            admission=self.admission,
            replay=lambda tool, case: bool(
                self.harness.replay_tool(task, tool, case)
            ),
        )
        if not tool_alignment.admitted:
            raise RuntimeError(
                "prepared provisional promotion failed registry Tool admission"
            )
        tool_ref = tool_alignment.ref
        implementation_ref = self.aligner.align_implementation(
            compiled.implementation, atomic_ref, tool_ref,
        )
        self._add_structural_edge(
            str(implementation_ref), str(atomic_ref),
            GlobalRelationType.IMPLEMENTS, trace.trace_id,
        )
        self._add_structural_edge(
            str(implementation_ref), str(tool_ref),
            GlobalRelationType.CONTAINS, trace.trace_id,
        )
        if (
            tool_alignment.operation == "add_replay"
            and tool_alignment.source_ref is not None
            and str(tool_alignment.source_ref) != str(tool_ref)
        ):
            self._add_structural_edge(
                str(tool_ref), str(tool_alignment.source_ref),
                GlobalRelationType.DERIVED_FROM, trace.trace_id,
                evolution_operation="add_replay",
            )

        actual_refs = (
            str(atomic_ref), str(implementation_ref), str(tool_ref),
        )
        attempts = tuple(
            CreditAttempt(
                artifact_ref=ref,
                artifact_kind=kind,
                occurrence_id=f"promotion::{prepared.provisional_ref}",
                attempt_id=(
                    f"promotion:{kind}:{prepared.provisional_ref}:{ref}"
                ),
                sequence_no=index,
                proposed=True,
                validated=True,
                metadata={
                    "source": "provisional_promotion",
                    "provisional_ref": prepared.provisional_ref,
                    "action_span": list(prepared.action_span),
                },
            )
            for index, (kind, ref) in enumerate(zip(
                ("atomic", "implementation", "tool"), actual_refs,
            ))
        )
        evidence = self.credit.assign(CreditTrace(
            trace.task.task_id, trace.trace_id, attempts,
        ))
        return actual_refs, evidence

    def _commit_failure_side_task_evidence(
        self,
        trace: TraceRecord,
        task: HarnessTask,
        *,
        trials: list[ProvisionalTrialResult],
        promotions: list[PreparedPromotion],
        failure_extraction: PreparedFailureExtraction | None,
        run_mode: RuntimeMode,
    ) -> tuple[dict[str, int], list[Any]]:
        counters = {
            "provisional_created_count": 0,
            "provisional_trial_ready_count": 0,
            "provisional_trial_supported_count": 0,
            "provisional_promoted_count": 0,
            "provisional_suppressed_count": 0,
            "failure_experience_observed_count": 0,
            "failure_experience_confirmed_count": 0,
            "failure_experience_resolved_count": 0,
        }
        if run_mode is not RuntimeMode.ONLINE:
            return counters, []
        store = self.failure_knowledge
        if store is None:
            if trials or promotions or failure_extraction is not None:
                raise RuntimeError(
                    "cold-start evidence exists without failure-side storage"
                )
            return counters, []

        promotion_events: list[Any] = []

        # Record the real Seeded attempt before PROMOTED.  A promoted record
        # is terminal and deliberately cannot accept trial evidence later.
        for trial in trials:
            before = store.get_provisional(trial.provisional_ref)
            start, end = trial.action_span
            after = store.record_provisional_trial(
                trial.provisional_ref,
                task_id=task.task_id,
                trace_id=trace.trace_id,
                started=end > start,
                local_effect_passed=trial.local_effect_passed,
                strict_task_success=trace.strict_task_success,
                infrastructure_failure=trace.infrastructure_failure,
                provider_or_protocol_failure=(
                    trial.failure_code
                    == "provisional_provider_or_protocol_failure"
                ),
                suppress_after=self.provisional_suppress_consecutive_failures,
                metadata={
                    "step_id": trial.step_id,
                    "action_span": list(trial.action_span),
                    "witness_refs": list(trial.witness_refs),
                    "failure_code": trial.failure_code,
                },
            )
            if after.status is not before.status:
                status = after.status.value
                trace.provisional_promotions.append({
                    "provisional_ref": trial.provisional_ref,
                    "status": status,
                    "source": "current_task_trial",
                    "step_id": trial.step_id,
                })
                if status == "trial_supported":
                    counters["provisional_trial_supported_count"] += 1
                elif status == "suppressed":
                    counters["provisional_suppressed_count"] += 1

        for prepared in promotions:
            actual_refs, evidence = self._apply_prepared_promotion(
                prepared, trace, task,
            )
            promoted = commit_prepared_promotion(
                prepared,
                store=store,
                verified_refs=actual_refs,
            )
            promotion_events.extend(evidence)
            trace.evidence_event_refs.extend(
                item.event_id for item in evidence
            )
            trace.provisional_promotions.append({
                "provisional_ref": prepared.provisional_ref,
                "status": promoted.status.value,
                "promoted_verified_refs": list(
                    promoted.promoted_verified_refs
                ),
                "action_span": list(prepared.action_span),
                "source": "current_strict_success_trial",
            })
            counters["provisional_promoted_count"] += 1

        if failure_extraction is not None and failure_extraction.accepted:
            prior_provisional: dict[str, Any | None] = {}
            for record in failure_extraction.provisional_records:
                try:
                    prior_provisional[record.provisional_ref] = (
                        store.get_provisional(record.provisional_ref)
                    )
                except KeyError:
                    prior_provisional[record.provisional_ref] = None
            prior_experience = None
            if failure_extraction.failure_experience is not None:
                experience_id = failure_extraction.failure_experience.experience_id
                try:
                    prior_experience = store.get_failure_experience(experience_id)
                except KeyError:
                    prior_experience = None
            provisional_refs, experience_ids = failure_extraction.commit(store)
            if trace.failure_extraction is None:
                raise RuntimeError("accepted failure extraction has no Trace record")
            trace.failure_extraction.provisional_refs = list(provisional_refs)
            trace.failure_extraction.failure_experience_ids = list(experience_ids)
            for ref in provisional_refs:
                if prior_provisional.get(ref) is not None:
                    continue
                current = store.get_provisional(ref)
                trace.provisional_promotions.append({
                    "provisional_ref": ref,
                    "status": current.status.value,
                    "source": "failure_extractor_f2",
                })
                counters["provisional_created_count"] += 1
                if current.status.value == "trial_ready":
                    counters["provisional_trial_ready_count"] += 1
            for experience_id in experience_ids:
                current = store.get_failure_experience(experience_id)
                if prior_experience is None:
                    counters["failure_experience_observed_count"] += int(
                        current.status.value == "observed"
                    )
                elif (
                    prior_experience.status.value != "confirmed"
                    and current.status.value == "confirmed"
                ):
                    counters["failure_experience_confirmed_count"] += 1

        # A successful task resolves only an experience actually referenced
        # by its admitted plan and only when no scaffold step reproduced a
        # divergence.  Dynamic rescue after a failed cold step is insufficient.
        can_resolve = bool(
            trace.strict_task_success
            and trace.cold_start_plan is not None
            and trace.cold_start_steps
            and all(not item.failure_code for item in trace.cold_start_steps)
        )
        if can_resolve:
            referenced = list(map(
                str,
                dict(trace.cold_start_plan.proposal).get(
                    "referenced_failure_experience_ids", (),
                ),
            ))
            for experience_id in referenced:
                before = store.get_failure_experience(experience_id)
                after = store.resolve_failure_experience(
                    experience_id,
                    task_id=task.task_id,
                    trace_id=trace.trace_id,
                    metadata={
                        "cold_start_plan_id": trace.cold_start_plan.plan_id,
                        "divergence_recurred": False,
                    },
                )
                if (
                    before.status.value != "resolved"
                    and after.status.value == "resolved"
                ):
                    counters["failure_experience_resolved_count"] += 1
        return counters, promotion_events

    def _finalize_v31_metrics(
        self,
        trace: TraceRecord,
        *,
        failure_side_read_start: int,
        counters: Mapping[str, int] | None = None,
    ) -> None:
        selected = 0
        if trace.cold_start_plan is not None:
            selected = sum(
                str(item.get("candidate_source", "")) == "provisional"
                for item in dict(trace.cold_start_plan.proposal).get("steps", ())
                if isinstance(item, Mapping)
            )
        current_reads = (
            self.failure_knowledge.failure_side_read_count
            if self.failure_knowledge is not None else 0
        )
        metrics = trace.metadata.setdefault("v31_metrics", {})
        metrics.update({
            **dict(counters or {}),
            "failure_side_read_count": max(
                0, int(current_reads) - int(failure_side_read_start),
            ),
            "provisional_selected_count": int(selected),
        })

    def _provider_instances(self) -> list[Any]:
        values: list[Any] = list(self._provider_cache.values())
        if isinstance(self._provider_override, Mapping):
            values.extend(self._provider_override.values())
        elif self._provider_override is not None:
            values.append(self._provider_override)
        unique: dict[int, Any] = {}
        for value in values:
            unique[id(value)] = value
        return list(unique.values())

    def _provider_request_offsets(self) -> dict[int, int]:
        return {
            id(provider): int(getattr(provider, "request_record_count", 0))
            for provider in self._provider_instances()
        }

    def _attach_provider_requests(
        self, trace: TraceRecord, offsets: Mapping[int, int],
    ) -> None:
        existing = {item.request_id for item in trace.provider_requests}
        payload_fields: set[str] = set()
        for provider in self._provider_instances():
            start = int(offsets.get(id(provider), 0))
            reader = getattr(provider, "request_records_since", None)
            if not callable(reader):
                continue
            for raw in reader(start):
                payload = to_primitive(raw)
                if not isinstance(payload, Mapping):
                    continue
                audit_request_id = str(payload.get("request_id", ""))
                request_id = str(
                    payload.get("provider_request_id") or audit_request_id
                )
                if not request_id or request_id in existing:
                    continue
                record = ProviderRequestRecord(
                    request_id=request_id,
                    session_id=str(payload.get("session_id", "")),
                    stage=str(payload.get("stage", "")),
                    started_at=float(payload.get("started_at", 0.0)),
                    ended_at=float(payload.get("ended_at", 0.0)),
                    outcome=str(payload.get("outcome", "")),
                    http_status=(
                        int(payload["http_status"])
                        if payload.get("http_status") is not None else None
                    ),
                    retry_count=int(payload.get("retry_count", 0)),
                    usage_status=str(payload.get("usage_status", "unavailable")),
                    error_code=str(payload.get("error_code", "")),
                    sanitized_error=str(payload.get("sanitized_error", ""))[:4000],
                    payload_fingerprint=str(payload.get("payload_fingerprint", "")),
                )
                trace.provider_requests.append(record)
                existing.add(request_id)
                payload_fields.update(map(str, payload.get("payload_field_names") or ()))
        trace.resource_usage_complete = all(
            item.usage_status == "reported" for item in trace.provider_requests
        )
        if payload_fields:
            trace.metadata["provider_payload_field_names"] = sorted(payload_fields)

    def _sanitize_failure_message(self, error: BaseException) -> str:
        message = str(error)
        llm = dict(self.config.get("llm") or {})
        secret = os.environ.get(str(llm.get("api_key_env", "MODEL_API_KEY")), "")
        if secret:
            message = message.replace(secret, "[REDACTED]")
        return message[:4000]

    @staticmethod
    def _require_resource_usage_complete(trace: TraceRecord) -> None:
        if trace.resource_usage_complete:
            return
        unavailable = [
            item.request_id
            for item in trace.provider_requests
            if item.usage_status != "reported"
        ]
        raise AtomicSkillGraphError(
            "provider_usage_missing",
            "provider request usage is unavailable; formal learning is blocked"
            + (": " + ", ".join(unavailable[:5]) if unavailable else ""),
            layer=FailureLayer.INFRASTRUCTURE,
        )

    def _persist_failure_trace(
        self,
        builder: TraceBuilder,
        primary: BaseException,
        *,
        attempt_id: str,
        usage_start: int,
        sessions_start: int,
        provider_offsets: Mapping[int, int],
        failure_side_read_start: int,
    ) -> None:
        trace = builder.trace
        if self.traces.exists(trace.trace_id):
            return
        raw_layer = getattr(primary, "layer", FailureLayer.INFRASTRUCTURE)
        try:
            layer = FailureLayer(raw_layer)
        except (TypeError, ValueError):
            layer = FailureLayer.INFRASTRUCTURE
        code = str(getattr(primary, "code", "") or "infrastructure_failure")
        message = self._sanitize_failure_message(primary)
        trace.infrastructure_failure = layer is FailureLayer.INFRASTRUCTURE
        trace.learning_eligible = False
        trace.metadata["failure"] = {
            "error_type": type(primary).__name__,
            "error_code": code,
            "sanitized_error": message,
        }
        trace.runtime_plan = {
            **dict(trace.runtime_plan or {}),
            "failure_stage": str(
                dict(trace.runtime_plan or {}).get("failure_stage") or "system"
            ),
        }
        self._attach_external_sessions(
            trace, self._observed_sessions[sessions_start:]
        )
        usage = self.usage.events[usage_start:]
        trace.llm_usage = [event.to_dict() for event in usage]
        trace.metadata["usage_reconciliation"] = _reconcile_events(usage)
        self._attach_provider_requests(trace, provider_offsets)
        self._finalize_v31_metrics(
            trace,
            failure_side_read_start=failure_side_read_start,
        )
        failure_id = "failure_" + content_hash({
            "trace_id": trace.trace_id,
            "attempt_id": attempt_id,
            "code": code,
            "message": message,
        })[:24]
        if not any(item.failure_id == failure_id for item in trace.failures):
            trace.failures.append(FailureEnvelope(
                failure_id=failure_id,
                layer=layer,
                code=code,
                task_id=trace.task.task_id,
                trace_id=trace.trace_id,
                occurrence_id="",
                attempt_id=attempt_id,
                started=bool(trace.provider_requests or trace.agent_sessions),
                recoverable=False,
                message=message,
            ))
        builder.finish()
        self.traces.save_atomic(trace)

    def _canonical_atomic_for_occurrence(
        self, occurrence: Any,
    ) -> AbstractAtomicSkill | None:
        """Build the Atomic contract without synthesizing an executable Tool."""

        derivations = dict(
            getattr(occurrence, "output_derivations", None) or {}
        )
        if not derivations:
            for output_role, value in sorted(occurrence.output_bindings.items()):
                input_role = next(
                    (
                        role for role, bound in occurrence.input_bindings.items()
                        if bound == value
                    ),
                    None,
                )
                if input_role is None:
                    return None
                derivations[output_role] = {
                    "kind": "input_identity",
                    "input_role": input_role,
                }
        output_identity: list[dict[str, str]] = []
        for output_role, derivation in derivations.items():
            if derivation.get("kind") != "input_identity":
                continue
            input_role = str(derivation.get("input_role", ""))
            output_identity.append({
                "output_role": str(output_role),
                "input_role": input_role,
            })
        return AbstractAtomicSkill(
            occurrence.proposed_ref,
            occurrence.intent,
            occurrence.input_specs,
            occurrence.output_specs,
            occurrence.preconditions,
            occurrence.effects,
            {
                "validator_id": "harness_atomic_effect",
                "identity_strict": True,
                "output_identity": output_identity,
                "output_derivations": {
                    str(role): dict(derivation)
                    for role, derivation in derivations.items()
                },
            },
            [],
            {
                "support_event_ids": [
                    str(item.get("event_id", item.get("action_id", index)))
                    for index, item in enumerate(occurrence.action_events)
                ],
                "envelope_event_range": [
                    int(occurrence.event_start),
                    int(occurrence.event_end),
                ],
            },
            {"source_trace_ids": [occurrence.source_trace_id]},
            SkillStatus.DRAFT,
        )

    def _replay_tool_candidate(
        self,
        task: HarnessTask,
        tool: Any,
        case: dict[str, Any],
    ) -> bool:
        """Run one admission replay through the same ToolRunner authority.

        The Harness only resets, replays validated prefix actions, and exposes
        catalog/evidence; it never interprets ``tool_ir_v1``.
        """

        if str(getattr(tool, "artifact_kind", "")) != "tool_ir_v1":
            return bool(self.harness.replay_tool(task, tool, case))
        expected_task = str((case.get("source_task") or {}).get("task_id", ""))
        if expected_task and expected_task != task.task_id:
            return False
        from atomic_skillgraph.core.bindings import BindingExprKind, BindingExpression
        from atomic_skillgraph.core.results import PrimitiveToolStep, RuntimeLinearPlan
        from atomic_skillgraph.runtime.budget import RuntimeBudget
        from atomic_skillgraph.runtime.task_context import TaskRuntimeContext
        from atomic_skillgraph.runtime.tool_runner import ToolRunner

        contract = self.harness.task_contract(task)
        plan = RuntimeLinearPlan.full_dynamic(
            task.task_id, contract, reason="tool_ir_replay",
        )
        trace_record = TraceRecord.create(
            TaskRecord(
                task.task_id, task.benchmark, task.goal, task.task_type,
                str(task.metadata.get("task_signature") or task.task_id),
                dict(task.metadata),
            ),
            to_primitive(contract),
            {},
            {"source": "full_dynamic", "failure_stage": "tool_ir_replay"},
        )
        trace_record.metadata["method_patch"] = str(
            self.config.get("method_patch", "3.1")
        )
        ctx = TaskRuntimeContext.create(
            task, plan, self.harness, TraceBuilder(trace_record),
            RuntimeBudget(global_action_budget=100, node_action_budget=35),
        )
        try:
            for event in list(case.get("prefix") or []):
                action_type = str(event.get("action_type", ""))
                arguments = dict(event.get("arguments") or {})
                primitive = PrimitiveToolStep(
                    action_type,
                    {
                        role: BindingExpression(
                            BindingExprKind.CONSTANT, constant=value,
                        )
                        for role, value in arguments.items()
                    },
                )
                result = self.harness.execute_primitive(primitive, {})
                if not result.accepted or (result.done and not result.won):
                    return False
                ctx.update_after_action(
                    result,
                    {
                        "action_type": action_type,
                        "arguments": arguments,
                        "accepted": result.accepted,
                        "done": result.done,
                        "won": result.won,
                        "new_revision": result.new_revision,
                        "observation": result.observation,
                        "occurrence_id": "tool_ir_replay_prefix",
                        "origin": "tool_ir_replay",
                    },
                )
            bindings = dict(case.get("bindings") or {})
            result = ToolRunner(self.validation.tool).run(
                tool, bindings, ctx, occurrence_id="tool_ir_replay",
            )
            if result.executed_action_count <= 0:
                return False
            # Admission replay proves the complete Tool program, not merely a
            # benchmark-winning prefix.  A terminal interruption remains valid
            # task/Atomic evidence, but it is never Tool-admission evidence.
            if not result.completed or result.terminal_interrupted:
                return False
            if not result.atomic_effect_passed:
                return False
            if result.failure_code:
                return False
            return True
        except AtomicSkillGraphError:
            raise
        except (KeyError, TypeError, ValueError, RuntimeError):
            return False

    def _existing_executable_reuse(
        self, occurrence: Any, atomic_view: AbstractAtomicSkill,
    ) -> CompiledKnowledge | None:
        """Reuse an exact existing Implementation/Tool without calling ToolBuilder."""

        alignment = self.aligner.resolve_atomic(atomic_view)
        if not alignment.reused:
            return None
        for implementation in self.skills.implementations_for(
            alignment.ref, mode=self.mode,
        ):
            try:
                tools = [
                    self.tools.get(binding.tool_ref)
                    for binding in implementation.tool_bindings
                ]
            except KeyError:
                continue
            if len(tools) != len(implementation.tool_bindings):
                continue
            if len(tools) != 1:
                continue
            return CompiledKnowledge(
                occurrence, atomic_view, tools[0], implementation,
            )
        return None

    def _build_tool_for_occurrence(
        self,
        occurrence: Any,
        atomic_view: AbstractAtomicSkill,
        normalized: dict[str, Any],
        trace: TraceRecord,
    ) -> tuple[CompiledKnowledge | None, dict[str, int]]:
        """Success Evolution Tool path: exact reuse else ToolBuilder + static gate."""

        metrics = {
            "call_count": 0,
            "no_tool_count": 0,
            "static_pass_count": 0,
            "static_reject_count": 0,
        }
        if (
            getattr(self, "config", None) is None
            or getattr(self, "usage", None) is None
            or not hasattr(self, "_tool_builder_session")
        ):
            # Legacy deterministic unit fixtures construct System objects without
            # the v3.2 tooling configuration.  The formal runner always has both.
            compiled = self.tool_compiler.compile([occurrence])
            return compiled[0], metrics
        exact = self._existing_executable_reuse(occurrence, atomic_view)
        if exact is not None:
            return exact, metrics
        provenance = ToolProvenance(
            source="success_evolution",
            atomic_ref=str(atomic_view.ref),
            source_trace_id=str(getattr(trace, "trace_id", normalized.get("trace_id", ""))),
            occurrence_id=occurrence.occurrence_id,
            task_id=str(getattr(getattr(trace, "task", None), "task_id", "")),
        )
        evidence_support = [
            item for item in occurrence.action_events
        ]
        actions = list(normalized.get("actions") or [])
        before_facts = []
        after_facts = []
        for item in normalized.get("before_state_facts", ()):
            if int(item.get("revision", -1)) == int(
                evidence_support[0].get("before_revision", -1)
            ) if evidence_support else False:
                before_facts.append(item)
        for item in normalized.get("after_state_facts", ()):
            if int(item.get("revision", -1)) == int(
                evidence_support[-1].get("after_revision", -1)
            ) if evidence_support else False:
                after_facts.append(item)
        action_schema = getattr(self.harness, "primitive_action_schema", None)
        primitive_actions = (
            [dict(item) for item in action_schema()]
            if callable(action_schema)
            else []
        )
        session = self._tool_builder_session(
            "tool_builder_evolution", occurrence.occurrence_id,
        )
        builder = ToolBuilderSession(session)
        proposal = builder.build(
            atomic=atomic_view,
            provenance=provenance,
            evidence_support=evidence_support,
            semantic_delta={
                "before_facts": before_facts,
                "after_facts": after_facts,
            },
            harness_interface={
                "profile": self.harness.profile_name,
                "predicate_vocabulary": to_primitive(
                    self.harness.semantic_predicate_schema()
                ),
                "primitive_actions": primitive_actions,
            },
            bucket="tool_builder_evolution",
        )
        metrics["call_count"] = 1
        if proposal.decision == "no_tool":
            metrics["no_tool_count"] = 1
            return None, metrics
        static = self.tool_static_validator.validate_proposal(
            proposal, atomic_view, self.harness,
            historical_evidence_support=evidence_support,
        )
        if not static.passed:
            metrics["static_reject_count"] = 1
            raise ValueError(
                "ToolBuilder proposal failed static validation: "
                + "; ".join(static.messages)
            )
        metrics["static_pass_count"] = 1
        item = self.tool_compiler.compile_proposal(
            occurrence, atomic_view, proposal, provenance,
        )
        return item, metrics

    @staticmethod
    def _terminal_empirical_certificate(
        trace: TraceRecord,
        coverage: Any,
    ) -> dict[str, Any]:
        if not bool(getattr(trace, "benchmark_success", False)):
            return {}
        executed: list[str] = []
        skipped: list[str] = []
        for node in getattr(trace, "node_records", ()) or ():
            occurrence_id = str(getattr(node, "occurrence_id", ""))
            status = str(getattr(node, "status", ""))
            if status == "skipped_goal_terminal":
                skipped.append(occurrence_id)
            elif occurrence_id:
                executed.append(occurrence_id)
        terminal_revision = max(
            (int(item.new_revision) for item in getattr(trace, "environment_actions", ())),
            default=0,
        )
        return {
            "benchmark": getattr(getattr(trace, "task", None), "benchmark", "alfworld"),
            "source_trace_id": str(trace.trace_id),
            "terminal_revision": int(terminal_revision),
            "executed_occurrence_ids": list(dict.fromkeys(executed)),
            "skipped_planned_occurrence_ids": list(dict.fromkeys(skipped)),
            "benchmark_won": True,
            "observed_task_contract_coverage": {
                "covered_effects": [
                    item.get("predicate")
                    for item in getattr(coverage, "target_checks", ())
                    if bool(item.get("passed"))
                ],
                "uncovered_effects": [
                    item.get("predicate")
                    for item in getattr(coverage, "target_checks", ())
                    if not bool(item.get("passed"))
                ],
            },
        }

    @staticmethod
    def _runtime_trial_e1_effect_eligible(
        trial: Mapping[str, Any],
    ) -> bool:
        """Keep Tool admission separate from executed-prefix E1 authority."""

        r1 = dict(trial.get("r1") or {})
        if bool(r1.get("admission_eligible", False)):
            return True
        if not bool(r1.get("terminal_interrupted", False)):
            return False
        result = dict(trial.get("result") or {})
        started = bool(r1.get("started", result.get("started", False)))
        intrinsic_failure = bool(r1.get("tool_intrinsic_failure", False))
        if not intrinsic_failure:
            intrinsic_failure = any(
                bool(dict(item).get("intrinsic_failure", False))
                for item in list(result.get("tool_results") or [])
                if isinstance(item, Mapping)
            )
        return bool(
            started
            and r1.get("atomic_effect_passed") is True
            and r1.get("executed_path_effects_passed") is True
            and r1.get("outputs_valid") is True
            and not intrinsic_failure
        )

    @staticmethod
    def _runtime_effect_witness_ref(
        fact: Mapping[str, Any], *, revision: int,
    ) -> str:
        """Rebuild the ALFWorld validator's exact structured fact ref."""

        predicate = str(fact.get("predicate", ""))
        suffix = ",".join(
            f"{role}={normalize_entity(value)}"
            for role, value in sorted(dict(fact.get("args") or {}).items())
        )
        return f"alfworld_action_fact:r{revision}:{predicate}:{suffix}"

    @staticmethod
    def _runtime_effect_event_index(
        trial: Mapping[str, Any],
        actions: list[dict[str, Any]],
        fact: Mapping[str, Any],
    ) -> int | None:
        """Tie a Runtime R1 fact to one accepted action in its actual trial."""

        if not actions:
            return None
        try:
            start = int(trial.get("trial_event_start", -1))
            end = int(trial.get("trial_event_end", -1))
        except (TypeError, ValueError):
            return None
        indexed = [
            action for fallback, action in enumerate(actions)
            if bool(action.get("accepted"))
            and (
                start <= int(action.get("event_index", fallback)) <= end
                if start >= 0 and end >= start
                else int(action.get("after_revision", -1))
                == int(trial.get("after_revision", -2))
            )
        ]
        if not indexed:
            return None
        identity = (
            str(fact.get("predicate", "")).casefold(),
            repr(sorted(dict(fact.get("args") or {}).items())),
        )
        exact = [
            action for action in indexed
            if any(
                (
                    str(raw.get("predicate", "")).casefold(),
                    repr(sorted(dict(raw.get("args") or {}).items())),
                ) == identity
                for raw in list(
                    action.get("authoritative_positive_effects") or []
                )
                if isinstance(raw, Mapping)
            )
        ]
        if exact:
            owner = exact[-1]
            try:
                return int(owner.get("event_index", actions.index(owner)))
            except (TypeError, ValueError):
                return None

        # Rich Harness evidence (for example entity.discovered_at) is not
        # necessarily expressible by the generic action-state reducer. It is
        # admissible only when Runtime recorded the exact occurrence-local
        # fact delta and accepted event/revision that established it.
        explicit: list[tuple[int, int]] = []
        for raw in list(trial.get("r1_effect_event_authorities") or []):
            if not isinstance(raw, Mapping):
                continue
            if str(raw.get("source_kind", "")) != "occurrence_action_delta":
                continue
            raw_identity = (
                str(raw.get("predicate", "")).casefold(),
                repr(sorted(dict(raw.get("args") or {}).items())),
            )
            if raw_identity != identity:
                continue
            raw_domain = str(raw.get("effect_domain", "")).casefold()
            fact_domain = str(fact.get("effect_domain", "")).casefold()
            if not raw_domain or (fact_domain and raw_domain != fact_domain):
                continue
            event_index = raw.get("event_index")
            revision = raw.get("revision")
            if (
                isinstance(event_index, bool)
                or not isinstance(event_index, int)
                or isinstance(revision, bool)
                or not isinstance(revision, int)
            ):
                continue
            owners = [
                action for action in indexed
                if int(action.get("event_index", -1)) == event_index
                and int(action.get("after_revision", -1)) == revision
            ]
            if len(owners) != 1:
                continue
            explicit.append((revision, event_index))
        if not explicit:
            return None
        # If a fact was invalidated and re-established within one trial, its
        # latest explicit establishment owns the still-current R1 witness.
        _revision, event_index = max(explicit)
        return int(event_index)

    @staticmethod
    def _runtime_trial_effect_authorities(
        trial: Mapping[str, Any],
        actions: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Project only an R1-admissible Runtime trial into E1 authorities.

        The model cannot author these facts.  They are reconstructed from the
        frozen Atomic Effect declaration, the Tool's validated RETURN outputs,
        and the AtomicValidator witness refs captured by ImplementationRunner.
        Tool-path selector refs are deliberately excluded.
        """

        if not AtomicSkillGraphSystem._runtime_trial_e1_effect_eligible(trial):
            return []
        witness_refs = list(dict.fromkeys(
            str(ref) for ref in list(trial.get("r1_witness_refs") or [])
            if str(ref)
        ))
        if not witness_refs:
            return []
        bindings = {
            **dict(trial.get("trial_bindings") or {}),
            **dict(trial.get("r1_outputs") or {}),
        }

        def resolve(raw: Any) -> tuple[bool, Any]:
            if isinstance(raw, Mapping):
                kind = str(raw.get("kind", "")).casefold()
                if kind == "skill_input":
                    role = str(raw.get("source_role", ""))
                    return role in bindings, bindings.get(role)
                if kind == "constant":
                    return "constant" in raw, raw.get("constant")
            if isinstance(raw, str) and raw.startswith("$"):
                role = raw[1:]
                return role in bindings, bindings.get(role)
            return True, raw

        resolved_effects: dict[str, list[dict[str, Any]]] = {}
        for raw_effect in list(trial.get("declared_effects") or []):
            if not isinstance(raw_effect, Mapping):
                continue
            predicate = str(raw_effect.get("predicate", ""))
            arguments: dict[str, Any] = {}
            closed = bool(predicate)
            for argument_role, expression in dict(
                raw_effect.get("args") or {}
            ).items():
                resolved, value = resolve(expression)
                if not resolved or value in (None, ""):
                    closed = False
                    break
                arguments[str(argument_role)] = value
            if closed and arguments:
                resolved_effects.setdefault(
                    predicate.casefold(), []
                ).append({
                    "predicate": predicate,
                    "args": arguments,
                    "cardinality": max(
                        1, int(raw_effect.get("cardinality", 1) or 1)
                    ),
                    "distinct_by": str(
                        raw_effect.get("distinct_by", "")
                    ),
                    "effect_domain": str(
                        raw_effect.get("effect_domain", "world")
                    ),
                })

        revision = int(trial.get("after_revision", 0) or 0)
        authorities: list[dict[str, Any]] = []
        for facts in resolved_effects.values():
            for fact in facts:
                expected_ref = (
                    AtomicSkillGraphSystem._runtime_effect_witness_ref(
                        fact, revision=revision,
                    )
                )
                exact_refs = [
                    witness_ref for witness_ref in witness_refs
                    if witness_ref == expected_ref
                ]
                if len(exact_refs) != 1:
                    # A validator ref that cannot be mapped to exactly one
                    # predicate+binding fact is not E1 Effect authority.
                    continue
                event_index = AtomicSkillGraphSystem._runtime_effect_event_index(
                    trial, list(actions or []), fact,
                )
                if actions is not None and event_index is None:
                    continue
                authority = {
                    **fact,
                    "witness_ref": exact_refs[0],
                    "revision": revision,
                    "source_kind": "runtime_trial_r1",
                    "draft_id": str(trial.get("draft_id", "")),
                }
                if event_index is not None:
                    authority["event_index"] = int(event_index)
                authorities.append(authority)
        return authorities

    def _prepare_evolution(self, trace: TraceRecord, task: HarnessTask) -> _PreparedEvolution:
        normalized = self.normalizer.build(trace)
        boundary_inputs: list[dict[str, Any]] = [
            dict(item)
            for item in list(
                dict(normalized.get("boundary_authorities") or {}).get(
                    "inputs"
                )
                or []
            )
            if isinstance(item, Mapping)
        ]
        seen_boundary_inputs = {
            (
                str(item.get("authority_ref", "")),
                str(item.get("role", "")),
                repr(item.get("value")),
            )
            for item in boundary_inputs
        }
        for action in list(normalized.get("actions") or []):
            if not isinstance(action, Mapping) or action.get("accepted") is not True:
                continue
            event_id = str(
                action.get("event_id", action.get("action_id", ""))
            )
            if not event_id:
                continue
            for raw_role, value in dict(
                action.get("arguments") or {}
            ).items():
                role = str(raw_role)
                projected_authority = {
                    "authority_ref": f"action_arg:{event_id}:{role}",
                    "event_id": event_id,
                    "argument_role": role,
                    "kind": "action_argument",
                    "source_kind": "action_argument",
                    "role": role,
                    "value": value,
                }
                identity = (
                    projected_authority["authority_ref"],
                    projected_authority["role"],
                    repr(projected_authority["value"]),
                )
                if identity not in seen_boundary_inputs:
                    seen_boundary_inputs.add(identity)
                    boundary_inputs.append(projected_authority)
        runtime_effect_facts: list[dict[str, Any]] = []
        for trial in list(trace.metadata.get("runtime_tool_trials", {}).values()):
            if not isinstance(trial, dict):
                continue
            if not self._runtime_trial_e1_effect_eligible(trial):
                continue
            for role, authority in dict(trial.get("input_authorities") or {}).items():
                if not isinstance(authority, dict):
                    continue
                trial_event_start = trial.get("trial_event_start", -1)
                trial_event_end = trial.get("trial_event_end", -1)
                if (
                    isinstance(trial_event_start, bool)
                    or not isinstance(trial_event_start, int)
                ):
                    trial_event_start = -1
                if (
                    isinstance(trial_event_end, bool)
                    or not isinstance(trial_event_end, int)
                ):
                    trial_event_end = -1
                source_kind = str(authority.get("kind", "")).casefold()
                projected_authority = {
                    "authority_ref": str(authority.get("authority_ref") or f"runtime_input:{trial.get('draft_id', '')}:{role}"),
                    "draft_id": str(trial.get("draft_id", "")),
                    "trial_event_start": int(trial_event_start),
                    "trial_event_end": int(trial_event_end),
                    "kind": source_kind,
                    "role": str(role),
                    "value": authority.get("value"),
                    "source_kind": source_kind,
                    "source_occurrence_id": str(authority.get("source_occurrence_id", "")),
                    "source_role": str(authority.get("source_role", "")),
                }
                identity = (
                    projected_authority["authority_ref"],
                    projected_authority["role"],
                    repr(projected_authority["value"]),
                )
                if identity not in seen_boundary_inputs:
                    seen_boundary_inputs.add(identity)
                    boundary_inputs.append(projected_authority)
            runtime_effect_facts.extend(
                self._runtime_trial_effect_authorities(
                    trial, list(normalized.get("actions") or []),
                )
            )
        if runtime_effect_facts:
            normalized.setdefault("after_state_facts", []).extend(
                runtime_effect_facts
            )
        predicate_schema = getattr(
            self.harness, "semantic_predicate_schema", None
        )
        predicate_domains = {
            str(item.name): str(item.effect_domain)
            for item in (
                predicate_schema() if callable(predicate_schema) else ()
            )
        }
        boundary_effects: list[dict[str, Any]] = []
        for action in normalized.get("actions", []):
            for raw_fact in action.get("authoritative_positive_effects", []):
                if not isinstance(raw_fact, dict):
                    continue
                witness_ref = str(
                    raw_fact.get("witness_ref")
                    or action.get("event_id", action.get("action_id", ""))
                )
                boundary_effects.append({
                    "witness_ref": witness_ref,
                    "predicate": str(raw_fact.get("predicate", "")),
                    "args": dict(raw_fact.get("args") or {}),
                    "effect_domain": str(
                        raw_fact.get("effect_domain")
                        or predicate_domains.get(
                            str(raw_fact.get("predicate", "")), ""
                        )
                    ),
                })
        boundary_effects.extend({
            "witness_ref": str(fact.get("witness_ref", "")),
            "predicate": str(fact.get("predicate", "")),
            "args": dict(fact.get("args") or {}),
            "effect_domain": str(fact.get("effect_domain", "world")),
            "source_kind": "runtime_trial_r1",
            "draft_id": str(fact.get("draft_id", "")),
            "event_index": int(fact.get("event_index", -1)),
        } for fact in runtime_effect_facts)
        normalized["boundary_authorities"] = {
            "inputs": boundary_inputs,
            "effects": boundary_effects,
        }
        extractor = ExtractorSession(self._extractor_session(task.task_id))
        contract = self.harness.task_contract(task)
        matcher_factory = getattr(self.harness, "contract_matcher", None)
        matcher = (
            matcher_factory()
            if callable(matcher_factory)
            else ExactContractMatcher()
        )
        witness_authority = extraction_coverage_authority(
            normalized,
            contract,
            matcher,
        )
        trace.metadata["extractor_coverage_authority"] = to_primitive(
            witness_authority
        )
        known_atomic_contracts = relevant_known_atomic_contracts(
            normalized,
            self.skills,
            limit=20,
        )
        proposals = extractor.propose_atomics(
            normalized,
            known_atomic_contracts,
            to_primitive(witness_authority),
            runtime_automation_drafts=list(
                trace.metadata.get("runtime_automation_drafts", {}).values()
            ),
            runtime_tool_trials=list(
                trace.metadata.get("runtime_tool_trials", {}).values()
            ),
        )
        try:
            canonical, occurrence_rejections = (
                self.atomicizer.validate_proposed_subset(
                    proposals, normalized,
                )
            )
        except ValueError as exc:
            trace.metadata["extractor_quality"] = {
                "extractor_e1_proposal_count": len(proposals),
                "extractor_e1_validated_occurrence_count": 0,
                "extractor_e1_rejection_count": len(proposals),
                "extractor_e1_contract_coverage_passed": False,
                "known_atomic_contract_payload_count": len(
                    known_atomic_contracts
                ),
            }
            trace.metadata["extraction"] = {
                **dict(trace.metadata.get("extraction") or {}),
                "e1_proposed": len(proposals),
                "e1_validated": 0,
                "e1_rejected": len(proposals),
                "e1_contract_coverage_passed": False,
            }
            raise ExtractionContentError(
                "e1",
                "extractor_e1_occurrence_rejected",
                str(exc),
            ) from exc
        # Compile and canonicalize each independently validated occurrence
        # before considering Composite coverage.  A content-invalid occurrence
        # must not discard unrelated, admission-ready Atomic knowledge from the
        # same E1 response.
        provisional: list[CompiledKnowledge] = []
        tool_builder_calls = 0
        tool_builder_no_tool = 0
        tool_builder_static_pass = 0
        tool_builder_static_reject = 0
        for occurrence in canonical:
            occurrence_stage = "compile"
            try:
                atomic_view = self._canonical_atomic_for_occurrence(occurrence)
                if atomic_view is None:
                    raise RuntimeError("canonical occurrence has no Atomic view")
                item, builder_metrics = self._build_tool_for_occurrence(
                    occurrence, atomic_view, normalized, trace,
                )
                tool_builder_calls += int(builder_metrics.get("call_count", 0))
                tool_builder_no_tool += int(builder_metrics.get("no_tool_count", 0))
                tool_builder_static_pass += int(builder_metrics.get("static_pass_count", 0))
                tool_builder_static_reject += int(builder_metrics.get("static_reject_count", 0))
                if item is None:
                    # NO_TOOL is an explicit, valid Builder decision.  The Atomic
                    # remains learnable and may be executed by a Seeded Agent.
                    alignment = self.aligner.resolve_atomic(atomic_view)
                    staged = self.aligner.stage_atomic(atomic_view)
                    staged_occurrence = (
                        self.aligner.atomic_canonicalizer
                        .rewrite_canonical_occurrence(
                            occurrence,
                            staged,
                            atomic_ref=staged.atomic.ref,
                        )
                    )
                    provisional.append(CompiledKnowledge(
                        staged_occurrence,
                        staged.atomic,
                        None,
                        None,
                    ))
                    continue
                bundle = self.aligner.stage_atomic(
                    item.atomic,
                    item.tool,
                    item.implementation,
                )
                staged_occurrence = (
                    self.aligner.atomic_canonicalizer
                    .rewrite_canonical_occurrence(
                        item.occurrence,
                        bundle,
                        atomic_ref=bundle.atomic.ref,
                    )
                )
            except ValueError as exc:
                occurrence_rejections.append({
                    "phase_id": str(occurrence.phase_id),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "stage": occurrence_stage,
                })
                continue
            provisional.append(CompiledKnowledge(
                staged_occurrence,
                bundle.atomic,
                bundle.tool,
                bundle.implementation,
            ))
        trace.metadata.setdefault("v32_metrics", {}).update({
            "tool_builder_call_count": tool_builder_calls,
            "tool_builder_no_tool_count": tool_builder_no_tool,
            "tool_builder_static_pass_count": tool_builder_static_pass,
            "tool_builder_static_rejection_count": tool_builder_static_reject,
        })

        if not provisional:
            quality = {
                "extractor_e1_proposal_count": len(proposals),
                "extractor_e1_validated_occurrence_count": 0,
                "extractor_e1_rejection_count": len(occurrence_rejections),
                "extractor_e1_contract_coverage_passed": False,
                "known_atomic_contract_payload_count": len(
                    known_atomic_contracts
                ),
            }
            trace.metadata["extractor_quality"] = quality
            trace.metadata["extraction_occurrence_rejections"] = (
                occurrence_rejections
            )
            trace.metadata["extraction"] = {
                **dict(trace.metadata.get("extraction") or {}),
                "e1_proposed": len(proposals),
                "e1_validated": 0,
                "e1_rejected": len(occurrence_rejections),
                "e1_contract_coverage_passed": False,
            }
            raise ExtractionContentError(
                "e1",
                "extractor_e1_occurrence_rejected",
                "Extractor E1 produced no independently compilable Atomic "
                "occurrences",
            )

        quality = {
            "extractor_e1_proposal_count": len(proposals),
            "extractor_e1_validated_occurrence_count": len(provisional),
            "extractor_e1_rejection_count": len(occurrence_rejections),
            "known_atomic_contract_payload_count": len(
                known_atomic_contracts
            ),
        }
        if occurrence_rejections:
            trace.metadata["extraction_occurrence_rejections"] = (
                occurrence_rejections
            )

        # Resolve one label per final staged Atomic ref.  This also makes two
        # alpha-equivalent occurrences in the same batch share a name before
        # either candidate has been registered.
        candidates_by_ref: dict[str, list[CompiledKnowledge]] = {}
        for item in provisional:
            ref = str(item.atomic.ref)
            candidates_by_ref.setdefault(ref, []).append(item)
        labels: dict[str, Any] = {}
        portability_context: dict[str, tuple[set[str], set[str]]] = {}
        for ref, candidates in sorted(candidates_by_ref.items()):
            portability_context[ref] = (
                set().union(*(
                    occurrence_terms(item.occurrence)
                    for item in candidates
                )),
                set().union(*(
                    source_forbidden_terms(item.occurrence)
                    for item in candidates
                )),
            )
            try:
                persisted = self.skills.get_atomic(candidates[0].atomic.ref)
            except KeyError:
                persisted = None
            labels[ref] = resolve_capability_label_group(
                (
                    (item.occurrence, item.atomic)
                    for item in candidates
                ),
                existing_atomic=persisted,
            )
        staged_compiled = [
            rewrite_capability_labels(item, labels[str(item.atomic.ref)])
            for item in provisional
        ]
        staged_occurrences = [item.occurrence for item in staged_compiled]
        coverage = contract_coverage_report(
            contract, staged_occurrences, matcher,
        )
        trace.metadata["extractor_contract_coverage"] = to_primitive(
            coverage
        )
        quality["extractor_e1_contract_coverage_passed"] = coverage.passed
        trace.metadata["extraction"] = {
            **dict(trace.metadata.get("extraction") or {}),
            "e1_proposed": len(proposals),
            "e1_validated": len(staged_compiled),
            "e1_rejected": len(occurrence_rejections),
            "e1_contract_coverage_passed": coverage.passed,
        }
        quality.update({
            "portable_intent_pass_count": sum(
                validate_portability(
                    item.occurrence.intent,
                    episode_terms=portability_context[
                        str(item.atomic.ref)
                    ][0],
                    additional_forbidden_terms=portability_context[
                        str(item.atomic.ref)
                    ][1],
                    require_intent=True,
                ).passed
                for item in provisional
            ),
            "portable_intent_fallback_count": sum(
                labels[str(item.atomic.ref)].source == "contract_fallback"
                for item in provisional
            ),
            "known_contract_name_reuse_count": sum(
                label.source == "existing_contract"
                for label in labels.values()
            ),
            "new_canonical_intent_count": sum(
                label.source != "existing_contract"
                for label in labels.values()
            ),
        })
        compiled = staged_compiled
        composite = None
        composite_rejection: dict[str, str] | None = None
        quality["extractor_e2_attempted"] = False
        if coverage.passed:
            try:
                existing = self.graph.existing_edges(
                    [str(item.proposed_ref) for item in staged_occurrences],
                    mode=RuntimeMode.ONLINE,
                )
                trace.metadata["extraction"] = {
                    **dict(trace.metadata.get("extraction") or {}),
                    "attempted": True,
                    "stage": "e2",
                    "prepared": False,
                    "e2_attempted": True,
                }
                quality["extractor_e2_attempted"] = True
                composite_proposal = extractor.propose_composite(
                    staged_occurrences,
                    existing,
                    contract_matcher=matcher,
                )
                quality.update({
                    "extractor_e2_selected_existing_edge_count": len(
                        composite_proposal.existing_edges
                    ),
                    "extractor_e2_selected_new_edge_count": len(
                        composite_proposal.new_edges
                    ),
                })
                trace.metadata["extraction"] = {
                    **dict(trace.metadata.get("extraction") or {}),
                    "e2_selected_existing_edges": len(
                        composite_proposal.existing_edges
                    ),
                    "e2_selected_new_edges": len(
                        composite_proposal.new_edges
                    ),
                }
                composite = self.composite_builder.validate_and_build(
                    composite_proposal,
                    staged_occurrences,
                    contract,
                    existing_edge_evidence=existing,
                    contract_matcher=matcher,
                    task_bindings=dict(
                        task.context.get("semantic_bindings") or {}
                    ),
                )
            except (AgentProtocolError, ValueError, BudgetExhausted) as exc:
                if (
                    isinstance(exc, BudgetExhausted)
                    and exc.code != "extractor_token_budget_exhausted"
                ):
                    raise
                stage = str(getattr(exc, "stage", "") or "e2")
                error_code = str(
                    getattr(exc, "error_code", "")
                    or getattr(exc, "code", "")
                )
                if error_code == "runtime_agent_schema_error":
                    error_code = f"extractor_{stage}_schema_rejected"
                if not error_code:
                    error_code = "extractor_e2_composite_validation_failed"
                composite_rejection = {
                    "stage": stage,
                    "error_type": type(exc).__name__,
                    "error_code": error_code,
                    "error": self._sanitize_failure_message(exc),
                }
        else:
            composite_rejection = {
                "stage": "e1",
                "error_type": "ExtractionContentError",
                "error_code": (
                    "extractor_e1_task_contract_coverage_incomplete"
                ),
                "error": (
                    "validated E1 occurrences do not cover the authoritative "
                    "TaskContract: " + ", ".join(coverage.failure_codes)
                ),
            }
            terminal_certificate = self._terminal_empirical_certificate(
                trace, coverage,
            )
            if (
                bool(getattr(trace, "benchmark_success", False))
                and staged_occurrences
                and terminal_certificate
            ):
                try:
                    existing = self.graph.existing_edges(
                        [str(item.proposed_ref) for item in staged_occurrences],
                        mode=RuntimeMode.ONLINE,
                    )
                    trace.metadata["extraction"] = {
                        **dict(trace.metadata.get("extraction") or {}),
                        "attempted": True,
                        "stage": "e2_terminal_empirical",
                        "prepared": False,
                        "e2_attempted": True,
                        "completion_authority": "terminal_empirical",
                    }
                    quality["extractor_e2_attempted"] = True
                    composite_proposal = extractor.propose_composite(
                        staged_occurrences,
                        existing,
                        contract_matcher=matcher,
                    )
                    quality.update({
                        "extractor_e2_selected_existing_edge_count": len(
                            composite_proposal.existing_edges
                        ),
                        "extractor_e2_selected_new_edge_count": len(
                            composite_proposal.new_edges
                        ),
                    })
                    composite = self.composite_builder.validate_and_build(
                        composite_proposal,
                        staged_occurrences,
                        contract,
                        existing_edge_evidence=existing,
                        contract_matcher=matcher,
                        task_bindings=dict(
                            task.context.get("semantic_bindings") or {}
                        ),
                        terminal_certificate=terminal_certificate,
                        source_composite_ref=str(
                            trace.runtime_plan.get("source_composite_ref") or ""
                        ),
                    )
                    trace.metadata["extraction"] = {
                        **dict(trace.metadata.get("extraction") or {}),
                        "terminal_empirical_candidate_ref": str(composite.ref),
                        "terminal_certificate": terminal_certificate,
                    }
                except (AgentProtocolError, ValueError, BudgetExhausted) as exc:
                    if (
                        isinstance(exc, BudgetExhausted)
                        and exc.code != "extractor_token_budget_exhausted"
                    ):
                        raise
                    composite_rejection = {
                        "stage": "e2_terminal_empirical",
                        "error_type": type(exc).__name__,
                        "error_code": str(
                            getattr(exc, "error_code", "")
                            or getattr(exc, "code", "")
                            or "extractor_e2_terminal_empirical_failed"
                        ),
                        "error": self._sanitize_failure_message(exc),
                    }
                else:
                    composite_rejection = None
            if composite_rejection is not None:
                composite_rejection = composite_rejection or {
                    "stage": "e1",
                    "error_type": "ExtractionContentError",
                    "error_code": (
                        "extractor_e1_task_contract_coverage_incomplete"
                    ),
                    "error": (
                        "validated E1 occurrences do not cover the authoritative "
                        "TaskContract: " + ", ".join(coverage.failure_codes)
                    ),
                }
                trace.metadata["extraction"] = {
                    **dict(trace.metadata.get("extraction") or {}),
                    "e2_attempted": False,
                }

        if composite_rejection:
            trace.metadata["extraction"] = {
                **dict(trace.metadata.get("extraction") or {}),
                "composite_rejection": dict(composite_rejection),
            }
        label_violations = 0
        for item in compiled:
            terms = occurrence_terms(item.occurrence)
            extra = source_forbidden_terms(item.occurrence)
            values = [
                (item.occurrence.intent, True),
                (item.atomic.summary, False),
                (item.atomic.guideline, False),
            ]
            if item.tool is not None:
                values.append((item.tool.summary, False))
            if item.implementation is not None:
                values.append((
                    item.implementation.metadata.get(
                        "semantic_description", "",
                    ),
                    False,
                ))
            label_violations += sum(
                not validate_portability(
                    value,
                    episode_terms=terms,
                    additional_forbidden_terms=extra,
                    require_intent=require_intent,
                ).passed
                for value, require_intent in values
            )
        if composite is not None:
            label_violations += int(
                composite.metadata.get(
                    "artifact_label_concrete_term_violation_count", 0,
                )
            )
        quality["artifact_label_concrete_term_violation_count"] = (
            label_violations
        )
        # The partial-admission event is emitted only after admission actually
        # runs.  Preparation alone cannot claim an admitted artifact.
        quality["partial_atomic_admission_count"] = 0
        trace.metadata["extractor_quality"] = quality
        diagnosis = self.gap_diagnoser.diagnose(
            trace, [item.atomic for item in compiled],
        )
        if diagnosis:
            trace.metadata["knowledge_gap_diagnosis"] = diagnosis
        return _PreparedEvolution(
            compiled,
            composite,
            diagnosis,
            str(trace.runtime_plan.get("source_composite_ref") or ""),
            composite_rejection,
        )

    def _apply_evolution(
        self, prepared: _PreparedEvolution, trace: TraceRecord, task: HarnessTask,
    ) -> dict[str, Any]:
        atomic_refs = []
        implementation_refs = []
        tool_refs = []
        evidence = EvolutionEvidenceAccumulator()
        by_occurrence: dict[str, Any] = {}
        quality = dict(trace.metadata.get("extractor_quality") or {})
        atomic_reuse_count = 0
        atomic_new_count = 0
        tool_admission_count = 0
        implementation_admission_count = 0
        for item in prepared.compiled:
            atomic_alignment = self.aligner.resolve_atomic(item.atomic)
            atomic_reuse_count += int(atomic_alignment.reused)
            atomic_new_count += int(not atomic_alignment.reused)
            atomic_ref = self.aligner.align_atomic(item.atomic)
            if item.tool is None or item.implementation is None:
                atomic_refs.append(atomic_ref)
                by_occurrence[item.occurrence.occurrence_id] = atomic_ref
                evidence.record(
                    str(atomic_ref),
                    "atomic",
                    occurrence_id=item.occurrence.occurrence_id,
                    passed=True,
                    reason="tool_builder_no_tool_atomic_only",
                )
                continue
            admitted_tool = self.admission.admit_tool(
                item.tool,
                replay=lambda tool, case: self._replay_tool_candidate(
                    task, tool, case,
                ),
                atomic=item.atomic,
                harness=self.harness,
            )
            tool_alignment = self.aligner.align_tool_with_replays(
                admitted_tool,
                admission=self.admission,
                replay=lambda tool, case: self._replay_tool_candidate(
                    task, tool, case,
                ),
            )
            tool_ref = tool_alignment.ref
            tool_admission_count += int(bool(tool_alignment.admitted))
            if (
                tool_alignment.operation == "add_replay"
                and tool_alignment.source_ref is not None
            ):
                assert self.repair_store is not None
                replay_repair = RepairProposal.create(
                    str(tool_alignment.source_ref),
                    "tool",
                    "add_replay",
                    {
                        "evolution_operation": "add_replay",
                        "candidate_ref": str(tool_ref),
                        "source_trace_id": trace.trace_id,
                        "requires_concrete_patch": False,
                    },
                    [],
                )
                replay_repair.status = (
                    "admitted" if tool_alignment.admitted else "rejected"
                )
                replay_repair.replay_result = {
                    "passed": bool(tool_alignment.admitted),
                    "candidate_ref": str(tool_ref),
                    "admission_failures": list(
                        tool_alignment.admission_failures
                    ),
                }
                self.repair_store.save(replay_repair)
                if tool_alignment.admitted:
                    self._add_structural_edge(
                        str(tool_ref),
                        str(tool_alignment.source_ref),
                        GlobalRelationType.DERIVED_FROM,
                        trace.trace_id,
                        evolution_operation="add_replay",
                        proposal_id=replay_repair.proposal_id,
                    )
            admitted_implementation = self.admission.admit_implementation(
                item.implementation,
                admitted_tool,
                atomic=item.atomic,
                harness=self.harness,
            )
            implementation_ref = self.aligner.align_implementation(
                admitted_implementation, atomic_ref, tool_ref
            )
            atomic_refs.append(atomic_ref)
            tool_refs.append(tool_ref)
            implementation_refs.append(implementation_ref)
            by_occurrence[item.occurrence.occurrence_id] = atomic_ref
            occurrence_id = item.occurrence.occurrence_id
            evidence.record(
                str(atomic_ref),
                "atomic",
                occurrence_id=occurrence_id,
                passed=True,
                reason="deterministic_contract_validation_passed",
            )
            evidence.record(
                str(tool_ref),
                "tool",
                occurrence_id=occurrence_id,
                passed=bool(tool_alignment.admitted),
                reason=(
                    "tool_admission_passed"
                    if tool_alignment.admitted
                    else "tool_admission_failed"
                ),
                metadata={
                    "operation": tool_alignment.operation,
                    "admission_failures": list(
                        tool_alignment.admission_failures
                    ),
                },
            )
            implementation_passed = (
                admitted_implementation.status is SkillStatus.CANDIDATE
            )
            implementation_admission_count += int(implementation_passed)
            evidence.record(
                str(implementation_ref),
                "implementation",
                occurrence_id=occurrence_id,
                passed=implementation_passed,
                reason=(
                    "implementation_admission_passed"
                    if implementation_passed
                    else "implementation_admission_failed"
                ),
                metadata={
                    "status": admitted_implementation.status.value,
                },
            )
            self._add_structural_edge(
                str(implementation_ref), str(atomic_ref), GlobalRelationType.IMPLEMENTS,
                trace.trace_id,
            )
            self._add_structural_edge(
                str(implementation_ref), str(tool_ref), GlobalRelationType.CONTAINS,
                trace.trace_id,
            )

        composite_ref = None
        quality.update({
            "atomic_alignment_reuse_count": atomic_reuse_count,
            "atomic_new_contract_count": atomic_new_count,
            "composite_alignment_reuse_count": 0,
            "partial_atomic_admission_count": 0,
            "partial_atomic_alignment_reuse_count": (
                atomic_reuse_count if prepared.composite is None else 0
            ),
            "partial_atomic_new_contract_count": (
                atomic_new_count if prepared.composite is None else 0
            ),
            "partial_atomic_tool_admission_count": (
                tool_admission_count if prepared.composite is None else 0
            ),
            "partial_atomic_implementation_admission_count": (
                implementation_admission_count
                if prepared.composite is None else 0
            ),
        })
        if prepared.composite is not None:
            composite_operation = (
                self._composite_rescue_operation(
                    prepared.source_composite_ref, prepared.composite,
                )
                if prepared.source_composite_ref
                else ""
            )
            composite_refs_before = {
                str(item) for item in self.skills.list_refs("composite")
            }
            composite_ref = self.aligner.align_composite(
                prepared.composite, by_occurrence,
            )
            quality["composite_alignment_reuse_count"] = int(
                str(composite_ref) in composite_refs_before
            )
            evidence.record(
                str(composite_ref),
                "composite",
                occurrence_id="composite",
                passed=True,
                reason="deterministic_graph_validation_passed",
                metadata={
                    "component_occurrence_ids": sorted(by_occurrence),
                },
            )
            if (
                prepared.source_composite_ref
                and str(composite_ref) != prepared.source_composite_ref
            ):
                self._add_structural_edge(
                    str(composite_ref),
                    prepared.source_composite_ref,
                    (
                        GlobalRelationType.DERIVED_FROM
                        if composite_operation == "revise_composite_sequence"
                        else GlobalRelationType.ALTERNATIVE
                    ),
                    trace.trace_id,
                    evolution_operation=(
                        composite_operation or "task_rescue_revision"
                    ),
                )
            for occurrence_id, atomic_ref in by_occurrence.items():
                self._add_structural_edge(
                    str(composite_ref),
                    str(atomic_ref),
                    GlobalRelationType.CONTAINS,
                    trace.trace_id,
                    occurrence_id=occurrence_id,
                )
        elif prepared.compiled:
            quality["partial_atomic_admission_count"] = 1
            trace.metadata.setdefault("r3_events", []).append({
                "revision": max(
                    (
                        int(item.new_revision)
                        for item in getattr(trace, "environment_actions", [])
                    ),
                    default=0,
                ),
                "occurrence_id": "",
                "event_type": "partial_atomic_admission",
                "details": {
                    "admission_count": 1,
                    "alignment_reuse_count": atomic_reuse_count,
                    "new_contract_count": atomic_new_count,
                    "tool_admission_count": tool_admission_count,
                    "implementation_admission_count": (
                        implementation_admission_count
                    ),
                },
            })
        trace.metadata["extractor_quality"] = quality

        attempts = tuple(
            CreditAttempt(
                artifact_ref=state.artifact_ref,
                artifact_kind=state.artifact_kind,
                occurrence_id="evolution",
                attempt_id=(
                    f"evolution:{state.artifact_kind}:{state.artifact_ref}"
                ),
                sequence_no=index,
                proposed=True,
                validated=state.validated_any,
                metadata={
                    "source": "extractor_admission",
                    **state.metadata(),
                },
            )
            for index, state in enumerate(evidence.assets())
        )
        if (
            composite_ref is not None
            and prepared.composite is not None
            and dict(prepared.composite.metadata.get("completion_authority") or {}).get("kind")
            == "terminal_empirical"
            and bool(getattr(trace, "benchmark_success", False))
            and not bool(getattr(trace, "task_rescue_required", False))
        ):
            attempts = attempts + (CreditAttempt(
                artifact_ref=str(composite_ref),
                artifact_kind="composite",
                occurrence_id="graph",
                attempt_id=f"composite:{composite_ref}:source_terminal_success",
                sequence_no=len(attempts),
                started=True,
                outcome=CreditOutcome.SELF_SUFFICIENT_SUCCESS,
                metadata={
                    "completion_authority": "terminal_empirical",
                    "benchmark_won": True,
                    "task_rescue_required": False,
                    "source": "terminal_empirical_candidate_creation",
                },
            ),)
        events = self.credit.assign(CreditTrace(
            trace.task.task_id, trace.trace_id, attempts
        ))
        self._commit_evidence(events)
        return {
            "atomic_refs": atomic_refs,
            "implementation_refs": implementation_refs,
            "tool_refs": tool_refs,
            "composite_ref": composite_ref,
            "composite_validated": composite_ref is not None,
        }

    def _composite_rescue_operation(
        self,
        source_ref: str,
        candidate: Any,
    ) -> str:
        """Classify only an evidence-backed same-node reorder as sequence revision."""
        try:
            source = self.skills.get_composite(source_ref)
        except KeyError:
            return "insert_missing_occurrence"
        source_by_step = {item.step_id: str(item.node_ref) for item in source.occurrences}
        candidate_by_step = {
            item.step_id: str(item.node_ref) for item in candidate.occurrences
        }
        source_sequence = [source_by_step[item] for item in source.control_sequence]
        candidate_sequence = [
            candidate_by_step[item] for item in candidate.control_sequence
        ]
        if (
            sorted(source_sequence) == sorted(candidate_sequence)
            and source_sequence != candidate_sequence
        ):
            return "revise_composite_sequence"
        return "insert_missing_occurrence"

    def _add_structural_edge(
        self,
        source_ref: str,
        target_ref: str,
        relation: GlobalRelationType,
        trace_id: str,
        **metadata: Any,
    ) -> None:
        payload = {
            "source_ref": source_ref,
            "target_ref": target_ref,
            "relation": relation.value,
            "trace_id": trace_id,
            **metadata,
        }
        self.graph.add(GlobalGraphEdge(
            content_hash(payload)[:24], source_ref, target_ref, relation,
            {"support_trace_ids": [trace_id], **metadata},
        ))

    def _commit_evidence(self, events: list[Any]) -> None:
        if not events:
            return
        assert self.ledger is not None and self.projection is not None and self.lifecycle is not None
        self.ledger.append_transaction(events)
        self.projection.consume(events)

    def _commit_gap_diagnosis(self, diagnosis: dict[str, Any]) -> None:
        if not diagnosis or self.readonly:
            return
        key = "evolution.gap_diagnosis_counts"
        row = self.database.execute(
            "SELECT value FROM metadata WHERE key=?", (key,),
        ).fetchone()
        state = json.loads(str(row["value"])) if row is not None else {
            "counts": {}, "trace_ids": [],
        }
        if "counts" not in state:
            state = {"counts": dict(state), "trace_ids": []}
        trace_id = str(diagnosis.get("trace_id") or "")
        if trace_id and trace_id in set(state.get("trace_ids") or []):
            return
        current = dict(state.get("counts") or {})
        for name, count in dict(diagnosis.get("counts") or {}).items():
            current[str(name)] = int(current.get(str(name), 0)) + int(count)
        trace_ids = list(state.get("trace_ids") or [])
        if trace_id:
            trace_ids.append(trace_id)
        state = {"counts": current, "trace_ids": trace_ids}
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO metadata(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(state, sort_keys=True, separators=(",", ":"))),
            )
    def _maybe_run_maintenance(self) -> None:
        if (
            self._online_successes <= 0
            or self._online_successes - self._last_maintenance_success_count
            < self.maintenance_interval
        ):
            return
        self.run_maintenance(
            triggering_task_id=self._current_task_id,
            milestone=f"online_success_{self._online_successes}",
        )

    def expected_periodic_maintenance_milestone_after_success(self) -> str:
        """Return the maintenance milestone a successful next task must produce."""

        expected_successes = self._online_successes + 1
        if (
            expected_successes - self._last_maintenance_success_count
            < self.maintenance_interval
        ):
            return ""
        return f"online_success_{expected_successes}"

    def _persist_maintenance_state(self) -> None:
        if self.readonly:
            return
        with self.database.transaction() as connection:
            for key, value in (
                ("online_success_count", self._online_successes),
                ("last_maintenance_success_count", self._last_maintenance_success_count),
            ):
                connection.execute(
                    "INSERT INTO metadata(key,value) VALUES(?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (key, str(value)),
                )

    def run_maintenance(
        self,
        *,
        triggering_task_id: str = "",
        milestone: str = "manual_final_batch",
        finalize_pending: bool = False,
    ) -> BatchMaintenanceResult:
        """Run one auditable batch and return its queue-empty admission result.

        Evolution-repair Agent usage is persisted in a dedicated immutable
        maintenance Trace before any replay/admission or ledger mutation.
        """
        if self.readonly:
            raise RuntimeError("frozen knowledge cannot run maintenance")
        assert (
            self.projection is not None
            and self.lifecycle is not None
            and self.evolution_maintenance is not None
            and self.repair_store is not None
        )
        self.projection.consume_new_events()
        trigger = triggering_task_id or self._current_task_id or "manual"
        maintenance_task_id = (
            f"maintenance_{self._online_successes}_"
            f"{content_hash({'trigger': trigger, 'milestone': milestone})[:12]}"
        )
        task_record = TaskRecord(
            maintenance_task_id,
            "maintenance",
            "Replay-gated batch evolution maintenance",
            "maintenance",
            content_hash({
                "success_count": self._online_successes,
                "triggering_task_id": trigger,
                "milestone": milestone,
            }),
            {
                "trace_kind": "maintenance",
                "triggering_task_id": trigger,
                "milestone": milestone,
            },
        )
        trace = TraceRecord.create(
            task_record,
            {},
            {},
            {
                "source": "batch_maintenance",
                "triggering_task_id": trigger,
                "milestone": milestone,
            },
        )
        trace.metadata.update({
            "trace_kind": "maintenance",
            "triggering_task_id": trigger,
            "milestone": milestone,
        })
        usage_start = len(self.usage.events)
        if self._evolution_batch_usage_start is not None:
            raise RuntimeError("nested evolution maintenance batch is forbidden")
        self._evolution_batch_usage_start = usage_start
        sessions_start = len(self._observed_sessions)
        provider_offsets = self._provider_request_offsets()
        reviews: list[dict[str, Any]] = []
        typed_reviews = []
        composite_reviews = []
        proposals = []
        typed_proposals = []
        composite_proposals = []
        typed_decisions = []
        composite_decisions = []
        semantic_error = ""
        try:
            reviews = self.evolution_maintenance.build_batch_reviews(self.tools)
            typed_reviews = self.evolution_maintenance.build_typed_reviews(
                skills=self.skills,
                tools=self.tools,
                traces=self.traces,
                harness_profile=str(self.harness.profile_name),
            )
            composite_reviews = (
                self.evolution_maintenance.build_composite_sequence_reviews(
                    skills=self.skills,
                    traces=self.traces,
                    harness_profile=str(self.harness.profile_name),
                )
            )
            if reviews:
                proposals = EvolutionRepairSession(
                    self._evolution_repair_session(maintenance_task_id)
                ).propose(reviews)
                by_review = {item["review_id"]: item for item in reviews}
                seen: set[str] = set()
                for proposal in proposals:
                    if proposal.review_id in seen or proposal.review_id not in by_review:
                        raise ValueError("EvolutionRepairSession returned invalid review authority")
                    seen.add(proposal.review_id)
                    self.evolution_maintenance._validate_agent_proposal(
                        proposal, by_review[proposal.review_id]
                    )
            if typed_reviews:
                typed_session = TypedRepairProposalSession(
                    self._evolution_repair_session(maintenance_task_id)
                )
                typed_decisions = typed_session.propose(typed_reviews)
                typed_proposals = typed_session.build_proposals(
                    typed_decisions, typed_reviews,
                )
                typed_by_failures = {
                    frozenset(item.source_failure_ids): item
                    for item in typed_reviews
                }
                for proposal in typed_proposals:
                    review = typed_by_failures.get(
                        frozenset(proposal.source_failure_ids)
                    )
                    proposal.proposed_patch.update({
                        "maintenance_trace_id": trace.trace_id,
                        "typed_review_id": (
                            "" if review is None else review.review_id
                        ),
                    })
            if composite_reviews:
                composite_session = CompositeSequenceProposalSession(
                    self._evolution_repair_session(maintenance_task_id)
                )
                composite_decisions = composite_session.propose(composite_reviews)
                composite_proposals = composite_session.build_proposals(
                    composite_decisions, composite_reviews,
                )
                composite_authority = {
                    (
                        item.target_ref,
                        frozenset(item.source_failure_ids),
                    ): item
                    for item in composite_reviews
                }
                for proposal in composite_proposals:
                    review = composite_authority.get((
                        proposal.target_ref,
                        frozenset(proposal.source_failure_ids),
                    ))
                    proposal.proposed_patch.update({
                        "maintenance_trace_id": trace.trace_id,
                        "composite_sequence_review_id": (
                            "" if review is None else review.review_id
                        ),
                        "source_proposal_ids": (
                            [] if review is None else list(
                                review.structural_context.get(
                                    "source_proposal_ids", []
                                )
                            )
                        ),
                    })
            self._attach_provider_requests(trace, provider_offsets)
            self._require_resource_usage_complete(trace)
        except AtomicSkillGraphError as exc:
            if exc.layer is FailureLayer.INFRASTRUCTURE:
                trace.infrastructure_failure = True
                trace.metadata["maintenance_failure"] = {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                self._finalize_maintenance_trace(
                    trace, sessions_start=sessions_start,
                    usage_start=usage_start, provider_offsets=provider_offsets,
                )
                raise
            # Protocol/budget failures are attributable Agent outcomes.  They
            # close this proposal batch without mutating semantic assets.
            semantic_error = str(exc)
            trace.metadata["semantic_proposal_error_code"] = str(
                getattr(exc, "code", "")
            )
            proposals = []
            typed_proposals = []
            composite_proposals = []
        except (AssertionError, KeyError, RuntimeError) as exc:
            trace.infrastructure_failure = True
            trace.metadata["maintenance_failure"] = {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            self._finalize_maintenance_trace(
                trace, sessions_start=sessions_start, usage_start=usage_start,
                provider_offsets=provider_offsets,
            )
            raise
        except (TypeError, ValueError) as exc:
            # Invalid semantic content is fail-closed but not infrastructure:
            # the maintenance trace records it and code admits no proposed edit.
            semantic_error = str(exc)
            proposals = []
            typed_proposals = []
            composite_proposals = []
        finally:
            self._evolution_batch_usage_start = None

        trace.metadata["batch_review_ids"] = [
            str(item["review_id"]) for item in reviews
        ]
        trace.metadata["semantic_proposal_count"] = len(proposals)
        trace.metadata["typed_review_ids"] = [
            item.review_id for item in typed_reviews
        ]
        trace.metadata["typed_proposal_count"] = len(typed_proposals)
        trace.metadata["composite_sequence_review_ids"] = [
            item.review_id for item in composite_reviews
        ]
        trace.metadata["composite_sequence_proposal_count"] = len(
            composite_proposals
        )
        if semantic_error:
            trace.metadata["semantic_proposal_error"] = semantic_error
        self._finalize_maintenance_trace(
            trace, sessions_start=sessions_start, usage_start=usage_start,
            provider_offsets=provider_offsets,
        )

        typed_decision_by_id = {
            item.review_id: item for item in typed_decisions
        }
        for review in typed_reviews:
            decision = typed_decision_by_id.get(review.review_id)
            if decision is not None and decision.decision != "no_change":
                continue
            audit = RepairProposal.create(
                review.target_refs[0],
                review.target_layer,
                review.eligible_operations[0],
                {
                    "typed_review_id": review.review_id,
                    "review_outcome": (
                        "no_change" if decision is not None else "omitted"
                    ),
                    "evidence_ids": sorted(
                        item.evidence_id for item in review.evidence
                    ),
                    "maintenance_trace_id": trace.trace_id,
                    "requires_concrete_patch": False,
                },
                list(review.source_failure_ids),
            )
            audit.status = "rejected"
            audit.replay_result = {
                "passed": False,
                "failure_code": "semantic_edit_not_proposed",
            }
            self.repair_store.save(audit)
        composite_decision_by_id = {
            item.review_id: item for item in composite_decisions
        }
        for review in composite_reviews:
            decision = composite_decision_by_id.get(review.review_id)
            if decision is not None and decision.decision != "no_change":
                continue
            audit = RepairProposal.create(
                review.target_ref,
                "composite",
                "revise_composite_sequence",
                {
                    "composite_sequence_review_id": review.review_id,
                    "review_outcome": (
                        "no_change" if decision is not None else "omitted"
                    ),
                    "evidence_ids": sorted(
                        item.evidence_id for item in review.evidence
                    ),
                    "maintenance_trace_id": trace.trace_id,
                    "requires_concrete_patch": False,
                },
                list(review.source_failure_ids),
            )
            audit.status = "rejected"
            audit.replay_result = {
                "passed": False,
                "failure_code": "semantic_edit_not_proposed",
            }
            self.repair_store.save(audit)

        typed_admitted: list[tuple[str, str]] = []
        typed_rejected: list[str] = []
        typed_lineage: list[dict[str, str]] = []
        typed_reviewed = [item.review_id for item in typed_reviews]
        concrete_typed = [
            item for item in self.repair_store.pending()
            if item.target_layer in {"atomic", "implementation"}
            and item.proposed_patch.get("typed_schema")
        ]
        by_proposal_id = {item.proposal_id: item for item in concrete_typed}
        by_proposal_id.update({item.proposal_id: item for item in typed_proposals})
        replay_executor = TraceRepairExecutor(
            harness=self.harness,
            skills=self.skills,
            tools=self.tools,
            validation=self.validation,
            admission=self.admission,
        )
        typed_engine = TypedRepairEngine(self.repair_store, self.skills)
        admitted_source_proposals: set[str] = set()
        review_by_failures = {
            frozenset(item.source_failure_ids): item for item in typed_reviews
        }
        for proposal in by_proposal_id.values():
            outcome = typed_engine.execute(
                proposal,
                replay=replay_executor.replay,
                validate=replay_executor.validate,
                admit=replay_executor.admit,
            )
            if outcome.proposal.status == "admitted":
                typed_admitted.extend(
                    (ref, proposal.target_layer) for ref in outcome.admitted_refs
                )
                review = review_by_failures.get(
                    frozenset(proposal.source_failure_ids)
                )
                if review is not None:
                    admitted_source_proposals.update(
                        map(str, review.context.get("source_proposal_ids") or [])
                    )
                typed_lineage.extend({
                    "source_ref": item.source_ref,
                    "target_ref": item.target_ref,
                    "relation": item.relation.value,
                    "operation": item.operation,
                    "proposal_id": item.proposal_id,
                    "review_id": "" if review is None else review.review_id,
                } for item in outcome.lineage)
            else:
                typed_rejected.append(outcome.proposal.proposal_id)
        for source_id in sorted(admitted_source_proposals):
            source = next(
                (
                    item for item in self.repair_store.pending()
                    if item.proposal_id == source_id
                ),
                None,
            )
            if source is None:
                continue
            source.status = "admitted"
            source.replay_result = {
                "passed": True,
                "resolved_by_typed_batch": True,
                "maintenance_trace_id": trace.trace_id,
            }
            self.repair_store.save(source)

        composite_admitted: list[tuple[str, str]] = []
        composite_rejected: list[str] = []
        composite_lineage: list[dict[str, str]] = []
        concrete_composite = [
            item for item in self.repair_store.pending()
            if item.target_layer == "composite"
            and item.operation == "revise_composite_sequence"
            and item.proposed_patch.get("typed_schema")
        ]
        composite_by_id = {item.proposal_id: item for item in concrete_composite}
        composite_by_id.update({
            item.proposal_id: item for item in composite_proposals
        })
        composite_engine = CompositeSequenceRepairEngine(
            self.repair_store, self.skills,
        )
        resolved_composite_sources: set[str] = set()
        for proposal in composite_by_id.values():
            outcome = composite_engine.execute(
                proposal,
                replay=replay_executor.replay_composite,
                validate=lambda candidate: self.planner.validator.validate(
                    _composite_plan(candidate),
                    mode=RuntimeMode.ONLINE,
                    harness_profile=str(self.harness.profile_name),
                ),
                admit=lambda candidate: candidate,
            )
            review_id = str(
                proposal.proposed_patch.get("composite_sequence_review_id", "")
            )
            if outcome.proposal.status == "admitted" and outcome.admitted_ref:
                composite_admitted.append((outcome.admitted_ref, "composite"))
                resolved_composite_sources.update(map(
                    str,
                    proposal.proposed_patch.get("source_proposal_ids") or [],
                ))
                composite_lineage.extend({
                    "source_ref": item.source_ref,
                    "target_ref": item.target_ref,
                    "relation": item.relation.value,
                    "operation": item.operation,
                    "proposal_id": item.proposal_id,
                    "review_id": review_id,
                } for item in outcome.lineage)
            else:
                composite_rejected.append(outcome.proposal.proposal_id)
        for source_id in sorted(resolved_composite_sources):
            source = next(
                (
                    item for item in self.repair_store.pending()
                    if item.proposal_id == source_id
                ),
                None,
            )
            if source is None:
                continue
            source.status = "admitted"
            source.replay_result = {
                "passed": True,
                "resolved_by_composite_sequence_batch": True,
                "maintenance_trace_id": trace.trace_id,
            }
            self.repair_store.save(source)

        result = self.evolution_maintenance.run_batch(
            maintenance_trace_id=trace.trace_id,
            reviews=reviews,
            agent_proposals=proposals,
            tools=self.tools,
            skills=self.skills,
            admission=self.admission,
            projection=self.projection,
            traces=self.traces,
            planner_validator=self.planner.validator,
            harness_profile=str(self.harness.profile_name),
            replay_tool=self._replay_maintenance_tool,
            replay_composite=replay_executor.replay_composite,
            finalize_pending=finalize_pending,
        )
        result.admitted_assets = tuple(dict.fromkeys([
            *typed_admitted, *composite_admitted, *result.admitted_assets,
        ]))
        result.rejected_proposal_ids = tuple(dict.fromkeys([
            *typed_rejected, *composite_rejected,
            *result.rejected_proposal_ids,
        ]))
        result.reviewed_ids = tuple(dict.fromkeys([
            *typed_reviewed,
            *(item.review_id for item in composite_reviews),
            *result.reviewed_ids,
        ]))
        result.lineage = (
            *typed_lineage, *composite_lineage, *result.lineage,
        )
        result.pending_proposal_ids = tuple(
            item.proposal_id for item in self.repair_store.pending()
        )
        for item in result.lineage:
            self._add_structural_edge(
                item["source_ref"],
                item["target_ref"],
                GlobalRelationType(item["relation"]),
                trace.trace_id,
                evolution_operation=item["operation"],
                proposal_id=item.get("proposal_id", ""),
                review_id=item.get("review_id", ""),
            )
        for ref, kind in result.admitted_assets:
            if kind != "composite":
                continue
            composite = self.skills.get_composite(ref)
            for occurrence in composite.occurrences:
                self._add_structural_edge(
                    ref,
                    str(occurrence.node_ref),
                    GlobalRelationType.CONTAINS,
                    trace.trace_id,
                    occurrence_id=occurrence.occurrence_id,
                )
        if result.admitted_assets:
            attempts = tuple(
                CreditAttempt(
                    artifact_ref=ref,
                    artifact_kind=kind,
                    occurrence_id="maintenance",
                    attempt_id=f"maintenance:{trace.trace_id}:{ref}",
                    sequence_no=index,
                    proposed=True,
                    validated=True,
                    metadata={
                        "source": "batch_replay_admission",
                        "maintenance_trace_id": trace.trace_id,
                    },
                )
                for index, (ref, kind) in enumerate(result.admitted_assets)
            )
            self._commit_evidence(self.credit.assign(CreditTrace(
                maintenance_task_id,
                trace.trace_id,
                attempts,
            )))
        result.lifecycle_result = self.lifecycle.review()
        superseded = self._apply_stable_supersedes(
            task_id=maintenance_task_id,
            trace_id=trace.trace_id,
        )
        if superseded:
            result.lineage = (*result.lineage, *superseded)
            result.lifecycle_result = self.lifecycle.review(
                sorted({item["target_ref"] for item in superseded})
            )
        self._last_maintenance_success_count = self._online_successes
        self._persist_maintenance_state()
        return result

    def _apply_stable_supersedes(
        self,
        *,
        task_id: str,
        trace_id: str,
    ) -> tuple[dict[str, str], ...]:
        """Supersede originals only after their replacement is lifecycle-stable."""
        assert self.projection is not None and self.lifecycle is not None
        edges = self.graph.edges()
        already = {
            (edge.source_ref, edge.target_ref)
            for edge in edges
            if edge.relation is GlobalRelationType.SUPERSEDES
        }
        replacement_operations = {
            "generalize",
            "merge",
            "update",
            "add_replay",
            "remove_redundant_occurrence",
            "revise_composite_sequence",
            "revise_composite_insight",
            "revise_atomic_contract",
            "merge_atomic",
            "split_atomic",
            "revise_implementation_mapping",
            "revise_grounding_constraint",
            "split",
        }
        groups: dict[tuple[str, str, str], list[Any]] = {}
        for edge in edges:
            if edge.relation not in {
                GlobalRelationType.DERIVED_FROM,
                GlobalRelationType.MERGED_FROM,
                GlobalRelationType.SPLIT_FROM,
            }:
                continue
            operation = str(edge.metadata.get("evolution_operation", ""))
            if operation not in replacement_operations:
                continue
            support = list(edge.metadata.get("support_trace_ids") or [])
            group_id = str(
                edge.metadata.get("proposal_id")
                or edge.metadata.get("review_id")
                or (support[0] if support else edge.edge_id)
            )
            groups.setdefault((edge.target_ref, operation, group_id), []).append(edge)

        emitted: list[dict[str, str]] = []
        sequence_no = 0
        for (old_ref, operation, _), group in sorted(groups.items()):
            old_row = self.database.execute(
                "SELECT artifact_kind,status FROM artifact_index WHERE artifact_ref=?",
                (old_ref,),
            ).fetchone()
            if old_row is None or str(old_row["status"]) not in {
                "active", "preferred", "suppressed",
            }:
                continue
            replacements: list[tuple[str, str]] = []
            for edge in sorted(group, key=lambda item: item.source_ref):
                row = self.database.execute(
                    "SELECT artifact_kind,status FROM artifact_index WHERE artifact_ref=?",
                    (edge.source_ref,),
                ).fetchone()
                if row is None or str(row["artifact_kind"]) != str(old_row["artifact_kind"]):
                    replacements = []
                    break
                status = str(row["status"])
                stable = status in (
                    {"active", "preferred"}
                    if str(row["artifact_kind"]) == "tool"
                    else {"active"}
                )
                if not stable:
                    replacements = []
                    break
                replacements.append((edge.source_ref, status))
            if not replacements:
                continue
            if operation in {"split", "split_atomic"} and len(replacements) < 2:
                continue
            for replacement_ref, replacement_status in replacements:
                if (replacement_ref, old_ref) in already:
                    continue
                self._commit_evidence(self.credit.assign_superseded(
                    task_id=task_id,
                    trace_id=trace_id,
                    old_ref=old_ref,
                    old_kind=str(old_row["artifact_kind"]),
                    replacement_ref=replacement_ref,
                    replacement_status=replacement_status,
                    sequence_no=sequence_no,
                ))
                sequence_no += 1
                self._add_structural_edge(
                    replacement_ref,
                    old_ref,
                    GlobalRelationType.SUPERSEDES,
                    trace_id,
                    evolution_operation=operation,
                )
                already.add((replacement_ref, old_ref))
                emitted.append({
                    "source_ref": replacement_ref,
                    "target_ref": old_ref,
                    "relation": GlobalRelationType.SUPERSEDES.value,
                    "operation": operation,
                    "proposal_id": "",
                    "review_id": "",
                })
        return tuple(emitted)

    def _finalize_maintenance_trace(
        self,
        trace: TraceRecord,
        *,
        sessions_start: int,
        usage_start: int,
        provider_offsets: Mapping[int, int],
    ) -> None:
        self._attach_external_sessions(
            trace, self._observed_sessions[sessions_start:]
        )
        usage = self.usage.events[usage_start:]
        trace.llm_usage = [item.to_dict() for item in usage]
        trace.metadata["usage_reconciliation"] = _reconcile_events(usage)
        self._attach_provider_requests(trace, provider_offsets)
        if self._provider_override is None:
            _require_formal_usage(usage, trace.agent_turns)
        trace.finish()
        self.traces.save_atomic(trace)

    def _replay_maintenance_tool(
        self, tool: Any, case: dict[str, Any],
    ) -> bool:
        source = dict(case.get("source_task") or {})
        required = {"task_id", "goal", "benchmark", "task_type"}
        if not required.issubset(source) or not str(source.get("task_id", "")):
            return False
        task = HarnessTask(
            task_id=str(source["task_id"]),
            goal=str(source["goal"]),
            benchmark=str(source["benchmark"]),
            task_type=str(source["task_type"]),
            context=dict(source.get("context") or {}),
            metadata=dict(source.get("metadata") or {}),
        )
        return bool(self.harness.replay_tool(task, tool, case))

    def _attach_external_sessions(
        self, trace: TraceRecord, observations: list[_ObservedSession],
    ) -> None:
        existing = {item.session_id for item in trace.agent_sessions}
        for observed in observations:
            if observed.task_id != trace.task.task_id or observed.session.session_id in existing:
                continue
            snapshot = observed.session.snapshot()
            trace.agent_sessions.append(AgentSessionRecord(
                observed.session.session_id,
                observed.session_type,
                observed.occurrence_id,
                observed.started_at,
                time.time(),
                snapshot,
            ))
            events = snapshot.get("r3_events", [])
            if isinstance(events, list) and events:
                trace.metadata.setdefault("r3_events", []).extend(
                    to_primitive(events)
                )
            for index, turn in enumerate(observed.turns):
                trace.agent_turns.append(AgentTurnRecord(
                    observed.session.session_id,
                    index,
                    turn.content,
                    turn.finish_reason,
                    [call.call_id for call in turn.tool_calls],
                    {
                        "prompt_tokens": turn.prompt_tokens,
                        "completion_tokens": turn.completion_tokens,
                        "total_tokens": turn.total_tokens,
                        "reasoning_tokens": turn.reasoning_tokens,
                        "call_count": 1,
                        "latency_ms": turn.latency_ms,
                    },
                    dict(turn.provider_metadata),
                    turn.reasoning_content,
                ))

    def preflight(
        self,
        *,
        require_api_key: bool = True,
        initialize_harness: bool = True,
        require_empty_bank: bool | None = None,
    ) -> dict[str, Any]:
        """Run the formal, side-effect-bounded experiment preflight."""
        requires_empty = (
            (self.config.get("experiment") or {}).get("initialize_v3_bank") == "empty"
            if require_empty_bank is None
            else bool(require_empty_bank)
        )
        try:
            self.database.validate_integrity()
            database_schema: bool = True
            database_schema_error = ""
        except Exception as exc:
            database_schema = False
            database_schema_error = str(exc)
        try:
            bank_is_empty = database_schema and self.is_empty_knowledge_bank()
        except Exception as exc:
            bank_is_empty = False
            if not database_schema_error:
                database_schema_error = str(exc)
            database_schema = False
        try:
            planner_provider = self._provider("planner")
            completion_parameters = inspect.signature(
                planner_provider.complete
            ).parameters
            provider_adapter_interface = (
                {"messages", "tools"}.issubset(completion_parameters)
                and "structured_output_schema" not in completion_parameters
            )
        except Exception:
            planner_provider = None
            provider_adapter_interface = False
        checks: dict[str, Any] = {
            "schema_version": int(self.config.get("schema_version", 0)) == 3,
            "alfworld_version": installed_alfworld_version() == "0.4.2",
            "method_patch": str(self.config.get("method_patch", "")) in {"3.1", "3.2"},
            "state_patch_level": (
                database_schema
                and self.database.execute(
                    "SELECT value FROM metadata WHERE key='state_patch_level'"
                ).fetchone()["value"] == STATE_PATCH_LEVEL
            ),
            "condition_full": str(
                self.config.get("condition")
                or (self.config.get("experiment") or {}).get("condition")
                or "full"
            ) == "full",
            "empty_bank": bank_is_empty,
            "bank_protocol": bank_is_empty if requires_empty else True,
            "database_schema": database_schema,
            "provider_adapter_interface": provider_adapter_interface,
            "model_configured": (
                True if self._provider_override is not None else
                str((self.config.get("llm") or {}).get("model", "")).strip()
                not in {"", "REPLACE_WITH_MODEL_ID"}
            ),
        }
        failure_side_read_count = (
            int(self.failure_knowledge.failure_side_read_count)
            if self.failure_knowledge is not None
            else 0
        )
        checks["failure_side_read_count"] = failure_side_read_count
        checks["provisional_selected_count"] = 0
        if self.readonly:
            checks["frozen_cold_start_disabled"] = not self.cold_start_enabled
            checks["frozen_failure_components_absent"] = all((
                self.failure_knowledge is None,
                self.provisional_retriever is None,
                self.failure_experience_retriever is None,
                getattr(self, "failure_extractor", None) is None,
                getattr(self.planner, "provisional_retriever", None) is None,
                getattr(self.planner, "failure_experience_retriever", None) is None,
            ))
            checks["frozen_failure_side_zero_read"] = failure_side_read_count == 0
            checks["frozen_provisional_zero_selected"] = (
                checks["provisional_selected_count"] == 0
            )
        if self._provider_override is None and planner_provider is not None:
            llm = dict(self.config.get("llm") or {})
            provider_snapshot = planner_provider.snapshot()
            checks["provider_configuration"] = bool(
                provider_snapshot.get("dialect") == "deepseek_v4_chat"
                and str(provider_snapshot.get("base_url", "")).rstrip("/")
                == "https://api.deepseek.com"
                and provider_snapshot.get("model") == "deepseek-v4-flash"
                and provider_snapshot.get("http_token_limit_field") == "max_tokens"
                and llm.get("dialect") == "deepseek_v4_chat"
            )
        else:
            checks["provider_configuration"] = "injected_or_not_required"
        if database_schema_error:
            checks["database_schema_error"] = database_schema_error
        try:
            self.artifacts.verify_all()
            checks["artifact_integrity"] = True
        except Exception as exc:
            checks["artifact_integrity"] = False
            checks["artifact_integrity_error"] = str(exc)
        if self.failure_knowledge is not None:
            try:
                self.failure_knowledge.verify_all()
                checks["failure_knowledge_integrity"] = True
            except Exception as exc:
                checks["failure_knowledge_integrity"] = False
                checks["failure_knowledge_integrity_error"] = str(exc)
        else:
            checks["failure_knowledge_integrity"] = "injected_or_not_required"
        if require_api_key and self._provider_override is None:
            try:
                self._provider("planner").config.resolve_api_key()
                checks["api_key_source"] = True
            except Exception as exc:
                checks["api_key_source"] = False
                checks["api_key_error"] = str(exc)
        else:
            checks["api_key_source"] = "injected_or_not_required"
        if initialize_harness:
            try:
                count = int(self.harness.initialize()) if hasattr(self.harness, "initialize") else -1
                checks["harness_initialization"] = True
                checks["harness_task_count"] = count
            except Exception as exc:
                checks["harness_initialization"] = False
                checks["harness_error"] = str(exc)
        try:
            self.trace_data_dir.mkdir(parents=True, exist_ok=True)
            handle, probe = tempfile.mkstemp(prefix=".write_probe_", dir=self.trace_data_dir)
            os.close(handle)
            Path(probe).unlink()
            checks["trace_output_writable"] = True
            checks["artifact_output_writable"] = self.readonly or os.access(self.artifacts.root, os.W_OK)
        except OSError as exc:
            checks["trace_output_writable"] = False
            checks["output_error"] = str(exc)
        checks["passed"] = all(
            value is True or value == "injected_or_not_required"
            for key, value in checks.items()
            if key not in {
                "empty_bank",
                "harness_task_count",
                "failure_side_read_count",
                "provisional_selected_count",
            } and not key.endswith("_error")
        )
        return checks

    def is_empty_knowledge_bank(self) -> bool:
        """Return whether this is the canonical fresh schema-v3 knowledge bank.

        Run/task bookkeeping and traces are experiment outputs, so they do not
        participate in this check.  Every table and file that contributes
        long-term executable/evolution knowledge does.  The database-created
        schema-version and v3.1 patch-level rows are the sole allowed metadata
        entries.
        """
        metadata = [
            (str(row["key"]), str(row["value"]))
            for row in self.database.rows("SELECT key,value FROM metadata ORDER BY key")
        ]
        if metadata != [
            ("schema_version", "3"),
            ("state_patch_level", STATE_PATCH_LEVEL),
        ]:
            return False
        for table in _LONG_TERM_KNOWLEDGE_TABLES:
            if self.database.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone() is not None:
                return False
        if self.artifacts.root.exists() and any(
            path.is_file() or path.is_symlink()
            for path in self.artifacts.root.rglob("*")
        ):
            return False
        failure_root = self.data_dir / "failure_knowledge"
        if failure_root.exists() and any(
            path.is_file() or path.is_symlink()
            for path in failure_root.rglob("*")
        ):
            return False
        return True

    def knowledge_digest(self) -> str:
        """Hash semantic knowledge, excluding machine-specific absolute file paths."""
        specs = {
            "metadata": ("key,value", "key"),
            "artifact_index": (
                "artifact_ref,artifact_kind,logical_id,version,content_hash,status,schema_version",
                "artifact_ref",
            ),
            "recommended_pointers": ("logical_id,artifact_ref", "logical_id"),
            "graph_edges": (
                "edge_id,source_ref,target_ref,relation,metadata_json", "edge_id"
            ),
            "evidence_events": (
                "event_id,schema_version,task_id,trace_id,occurrence_id,attempt_id,"
                "sequence_no,artifact_ref,artifact_kind,event_type,failure_layer,confidence,metadata_json",
                "event_id",
            ),
            "lifecycle_projection": (
                "artifact_ref,projection_json,last_event_rowid", "artifact_ref"
            ),
            "projection_checkpoints": (
                "projection_name,last_event_rowid", "projection_name"
            ),
            "provisional_artifacts": (
                "provisional_ref,contract_signature,canonical_intent,status,"
                "harness_profile,content_hash,source_trace_id,source_task_id,"
                "promoted_refs_json,schema_version,created_at,updated_at",
                "provisional_ref",
            ),
            "failure_experiences": (
                "experience_id,cluster_signature,divergence_signature,status,"
                "harness_profile,content_hash,support_count,resolved_count,"
                "schema_version,created_at,updated_at",
                "experience_id",
            ),
            "cold_start_evidence": (
                "event_id,task_id,trace_id,subject_ref,subject_kind,event_type,"
                "sequence_no,metadata_json",
                "event_id",
            ),
        }
        table_records: dict[str, list[list[Any]]] = {}
        for table, (columns, order) in specs.items():
            table_records[table] = [
                [to_primitive(value) for value in row]
                for row in self.database.execute(
                    f"SELECT {columns} FROM {table} ORDER BY {order}"
                ).fetchall()
            ]
        files: list[dict[str, str]] = []
        data_dir = Path(self.artifacts.data_dir)
        if self.artifacts.root.exists():
            for path in sorted(item for item in self.artifacts.root.rglob("*") if item.is_file()):
                files.append({
                    "path": path.relative_to(data_dir).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                })
        failure_root = data_dir / "failure_knowledge"
        if failure_root.exists():
            for path in sorted(
                item for item in failure_root.rglob("*") if item.is_file()
            ):
                files.append({
                    "path": path.relative_to(data_dir).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                })
        return content_hash({"files": files, "tables": table_records})

    def freeze(
        self,
        destination: str | Path,
        *,
        provenance: Mapping[str, Any] | None = None,
    ) -> Path:
        """Create one immutable, relocatable knowledge snapshot."""
        if self.readonly:
            raise RuntimeError("cannot freeze from an already read-only snapshot")
        if self._last_maintenance_success_count != self._online_successes:
            raise RuntimeError(
                "final batch maintenance must cover every online success before freeze"
            )
        assert self.repair_store is not None
        self.repair_store.assert_queue_empty()
        # Keep the source and frozen digest chains identical: resolved repairs
        # already live in immutable history; the empty mutable queue key does
        # not cross the snapshot boundary.
        self.repair_store.remove_queue_metadata()
        destination = Path(destination).expanduser().resolve()
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts.verify_all()
        if self.failure_knowledge is not None:
            self.failure_knowledge.verify_all()
        digest = self.knowledge_digest()
        temporary = Path(tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-", dir=destination.parent
        ))
        try:
            shutil.copytree(self.artifacts.root, temporary / "artifacts")
            source_failure_root = self.data_dir / "failure_knowledge"
            if source_failure_root.is_dir():
                shutil.copytree(
                    source_failure_root,
                    temporary / "failure_knowledge",
                )
            else:
                (temporary / "failure_knowledge").mkdir()
            target_database = temporary / "state.sqlite3"
            target_connection = sqlite3.connect(target_database)
            self.database.connection.backup(target_connection)
            target_connection.close()
            target_connection = sqlite3.connect(target_database)
            target_connection.row_factory = sqlite3.Row
            target_connection.execute(
                "DELETE FROM metadata WHERE key='evolution.repair_proposals'"
            )
            for row in target_connection.execute(
                "SELECT artifact_ref,file_path FROM artifact_index"
            ).fetchall():
                source = Path(row["file_path"])
                relative = source.relative_to(self.artifacts.root)
                final_path = destination / "artifacts" / relative
                target_connection.execute(
                    "UPDATE artifact_index SET file_path=? WHERE artifact_ref=?",
                    (str(final_path), row["artifact_ref"]),
                )
            for table, key_column in (
                ("provisional_artifacts", "provisional_ref"),
                ("failure_experiences", "experience_id"),
            ):
                for row in target_connection.execute(
                    f"SELECT {key_column},file_path FROM {table}"
                ).fetchall():
                    source = Path(row["file_path"]).resolve()
                    try:
                        relative = source.relative_to(source_failure_root.resolve())
                    except ValueError as exc:
                        raise RuntimeError(
                            "failure knowledge index points outside failure_knowledge"
                        ) from exc
                    final_path = destination / "failure_knowledge" / relative
                    target_connection.execute(
                        f"UPDATE {table} SET file_path=? WHERE {key_column}=?",
                        (str(final_path), row[key_column]),
                    )
            target_connection.commit()
            target_connection.close()
            atomic_write_json(temporary / "freeze_manifest.json", {
                "schema_version": 3,
                "state_patch_level": STATE_PATCH_LEVEL,
                "created_at": time.time(),
                "knowledge_digest": digest,
                "source_data_dir": str(self.data_dir),
                "provenance": to_primitive(dict(provenance or {})),
            })
            os.replace(temporary, destination)
            with StateDatabase(destination / "state.sqlite3", readonly=True) as frozen_db:
                frozen_artifacts = ArtifactStore(destination, frozen_db)
                frozen_artifacts.verify_all()
                view = type("FrozenDigestView", (), {})()
                view.database = frozen_db
                view.artifacts = frozen_artifacts
                if AtomicSkillGraphSystem.knowledge_digest(view) != digest:
                    raise RuntimeError("frozen snapshot knowledge digest differs from source")
            return destination
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def close(self) -> None:
        self.database.close()

    def __enter__(self) -> "AtomicSkillGraphSystem":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _reconcile_events(events: tuple[Any, ...]) -> dict[str, int]:
    real = sum(
        event.usage.total_tokens
        for event in events
        if event.bucket is not UsageBucket.UNATTRIBUTED
    )
    unattributed = sum(
        event.usage.total_tokens
        for event in events
        if event.bucket is UsageBucket.UNATTRIBUTED
    )
    return {
        "real_bucket_total_tokens": real,
        "unattributed_total_tokens": unattributed,
        "episode_total_tokens": real + unattributed,
        "token_mismatch": 0,
    }


def _require_formal_usage(events: tuple[Any, ...], turns: list[Any]) -> None:
    if turns and not events:
        raise AtomicSkillGraphError(
            "infrastructure_failure",
            "provider calls were observed without usage records",
            layer=FailureLayer.INFRASTRUCTURE,
        )
    for event in events:
        if event.provider_metadata.get("usage_status") != "reported":
            raise AtomicSkillGraphError(
                "infrastructure_failure",
                "provider usage is unavailable or partial; formal accounting is fail-closed",
                layer=FailureLayer.INFRASTRUCTURE,
            )
        if event.bucket is UsageBucket.UNATTRIBUTED:
            raise AtomicSkillGraphError(
                "infrastructure_failure",
                "provider usage was not attributed to a formal agent bucket",
                layer=FailureLayer.INFRASTRUCTURE,
            )


__all__ = ["AtomicSkillGraphSystem", "load_config"]
