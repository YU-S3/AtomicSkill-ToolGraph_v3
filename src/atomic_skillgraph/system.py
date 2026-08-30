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
from .core.errors import AtomicSkillGraphError, FailureEnvelope, FailureLayer
from .core.refs import content_hash
from .core.serialization import atomic_write_json, to_primitive
from .core.status import RuntimeMode, SkillStatus
from .evolution.admission import Admission
from .evolution.aligner import Aligner
from .evolution.atomicizer import Atomicizer
from .evolution.composite_builder import CompositeBuilder
from .evolution.composite_repair_session import CompositeSequenceProposalSession
from .evolution.composite_repairs import CompositeSequenceRepairEngine
from .evolution.extractor_session import ExtractorSession
from .evolution.failure_processor import FailureProcessor
from .evolution.gap_diagnosis import GapDiagnoser
from .evolution.maintenance import (
    BatchMaintenanceResult,
    EvolutionMaintenance,
    ExtractionPolicy,
    _composite_plan,
)
from .evolution.repair import RepairProposal, RepairStore
from .evolution.repair_session import EvolutionRepairSession
from .evolution.trace_replay import TraceRepairExecutor
from .evolution.tool_compiler import CompiledKnowledge, ToolCompiler
from .evolution.trace_normalizer import TraceNormalizer
from .evolution.typed_repair_session import TypedRepairProposalSession
from .evolution.typed_repairs import TypedRepairEngine
from .governance import (
    CandidateUsePolicy,
    CreditAssigner,
    CreditAttempt,
    CreditTrace,
    EvidenceLedger,
    LifecycleController,
    LifecyclePolicy,
    LifecycleProjection,
    LifecycleThresholds,
)
from .harness.alfworld import AlfWorldAdapter
from .harness.protocol import HarnessTask
from .knowledge import ArtifactStore, GraphStore, SkillRegistry, StateDatabase, ToolRegistry
from .planner import PlannerPipeline
from .runtime.invocation_compiler import InvocationCompiler
from .runtime.orchestrator import RuntimeOrchestrator
from .runtime.budget import required_runtime_turn_caps, validate_runtime_turn_caps
from .traces import (
    AgentSessionRecord,
    AgentTurnRecord,
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
        "You are a runtime preparation agent. Use only native tools offered in the current turn. "
        "Ground missing arguments through current environment evidence, then invoke at most one learned implementation."
    ),
    "runtime_seeded": (
        "You are a fresh seeded runtime agent. Complete only the supplied Atomic contract using current native actions."
    ),
    "runtime_dynamic": (
        "You are a fresh full-dynamic task agent. Solve the stated task using exactly one currently offered native action per turn."
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
}


_LONG_TERM_KNOWLEDGE_TABLES = (
    "artifact_index",
    "recommended_pointers",
    "graph_edges",
    "evidence_events",
    "lifecycle_projection",
    "projection_checkpoints",
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
    composite: Any
    gap_diagnosis: dict[str, Any]
    source_composite_ref: str


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
            candidate_policy=self.candidate_policy,
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
        self.admission = Admission(self.validation.tool)
        self.aligner = Aligner(self.skills, self.tools)
        self.composite_builder = CompositeBuilder()
        self.failure_processor = FailureProcessor(self.validation.failure_localizer)
        self.gap_diagnoser = GapDiagnoser(self.skills)
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

    def _runtime_session(self, session_kind: str, occurrence_id: str) -> _SessionProxy:
        cfg = self._stage_config("runtime")
        bucket = UsageBucket(session_kind)
        token_name = "max_total_tokens_per_task" if session_kind == "runtime_dynamic" else "max_total_tokens_per_node"
        return self._new_session(
            stage=session_kind, bucket=bucket,
            session_type={
                "runtime_preparation": "RuntimePreparationSession",
                "runtime_seeded": "SeededSession",
                "runtime_dynamic": "DynamicTaskSession",
            }[session_kind],
            occurrence_id=occurrence_id, task_id=self._current_task_id,
            max_turns=(
                self._runtime_turn_caps[1]
                if session_kind == "runtime_dynamic"
                else self._runtime_turn_caps[0]
            ),
            max_tokens=int(cfg.get(token_name, cfg.get("max_total_tokens_per_node", 80000))),
            exhaustion_code="runtime_node_token_budget_exhausted",
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
        sessions_start = len(self._observed_sessions)
        trace_builder = self.orchestrator.create_trace_builder(
            task, attempt_id=attempt_id,
        )
        provider_offsets = self._provider_request_offsets()
        try:
            return self._run_task_pipeline(
                task,
                run_mode=run_mode,
                trace_builder=trace_builder,
                usage_start=usage_start,
                sessions_start=sessions_start,
                provider_offsets=provider_offsets,
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

    def _run_task_pipeline(
        self,
        task: HarnessTask,
        *,
        run_mode: RuntimeMode,
        trace_builder: TraceBuilder,
        usage_start: int,
        sessions_start: int,
        provider_offsets: dict[int, int],
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
        trace.runtime_plan["failure_stage"] = "evolution"
        self.failure_processor.localize(trace)
        decision = self.extraction_policy.decide(trace)
        trace.extraction_policy = {
            "should_extract": bool(decision.should_extract and run_mode is RuntimeMode.ONLINE),
            "reasons": decision.reasons if run_mode is RuntimeMode.ONLINE else ["frozen_mode_disabled"],
        }
        prepared: _PreparedEvolution | None = None
        if (
            run_mode is RuntimeMode.ONLINE
            and not trace.infrastructure_failure
            and trace.benchmark_success
            and trace.learning_eligible
            and decision.should_extract
        ):
            try:
                prepared = self._prepare_evolution(trace, task)
                trace.metadata["extraction"] = {
                    "prepared": True,
                    "atomic_occurrence_count": len(prepared.compiled),
                }
            except ValueError as exc:
                # Extractor proposals may be semantically rejected by the
                # deterministic Atomic/Composite validators.  Infrastructure,
                # persistence, programming, and unexpected errors propagate so
                # the runner can roll back the task checkpoint.
                trace.metadata["extraction"] = {
                    "prepared": False,
                    "error_type": type(exc).__name__,
                    "error_code": str(getattr(exc, "code", "")),
                    "error": str(exc),
                }

        repair_proposals = []
        if run_mode is RuntimeMode.ONLINE:
            if trace.infrastructure_failure:
                trace.metadata["evolution_branch"] = "infrastructure_neutral"
            elif trace.benchmark_success and trace.learning_eligible:
                trace.metadata["evolution_branch"] = "success"
                source_composite_ref = str(
                    trace.runtime_plan.get("source_composite_ref") or ""
                )
                if trace.task_rescue_required and prepared is not None and source_composite_ref:
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
        if self._provider_override is None:
            _require_formal_usage(task_usage, trace.agent_turns)

        runtime_events = (
            self.credit.assign(trace)
            if run_mode is RuntimeMode.ONLINE and not trace.infrastructure_failure
            else []
        )
        trace.evidence_event_refs = [event.event_id for event in runtime_events]

        # Extractor validation and Evolution admission are part of this task's
        # terminal outcome.  Complete them before publishing the immutable
        # success Trace so an exception is persisted on the original skeleton
        # rather than leaving a success-looking Trace for a failed attempt.
        if run_mode is RuntimeMode.ONLINE:
            if trace.infrastructure_failure:
                pass
            elif trace.benchmark_success and trace.learning_eligible:
                if prepared is not None:
                    applied = self._apply_evolution(prepared, trace, task)
                    trace.metadata["evolution_applied"] = {
                        "atomic_refs": [str(item) for item in applied["atomic_refs"]],
                        "implementation_refs": [
                            str(item) for item in applied["implementation_refs"]
                        ],
                        "tool_refs": [str(item) for item in applied["tool_refs"]],
                        "composite_ref": str(applied["composite_ref"]),
                        "composite_validated": bool(applied["composite_validated"]),
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
        self.traces.save_atomic(trace)

        if run_mode is RuntimeMode.ONLINE:
            if trace.benchmark_success:
                self._online_successes += 1
            if not trace.infrastructure_failure:
                # Runtime credit remains Trace-first: the immutable evidence
                # source exists before its append-only ledger projections.
                self._commit_evidence(runtime_events)
            self._maybe_run_maintenance()
            self._persist_maintenance_state()
        return trace

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

    def _prepare_evolution(self, trace: TraceRecord, task: HarnessTask) -> _PreparedEvolution:
        normalized = self.normalizer.build(trace)
        extractor = ExtractorSession(self._extractor_session(task.task_id))
        proposals = extractor.propose_atomics(normalized)
        contract = self.harness.task_contract(task)
        matcher_factory = getattr(self.harness, "contract_matcher", None)
        matcher = (
            matcher_factory()
            if callable(matcher_factory)
            else ExactContractMatcher()
        )
        canonical, occurrence_rejections = self.atomicizer.validate_proposed_subset(
            proposals, normalized,
        )
        if occurrence_rejections:
            trace.metadata["extraction_occurrence_rejections"] = occurrence_rejections

        # Compile and canonicalize roles before E2.  Staging is read-only but
        # resolves the exact persistent Atomic refs that graph evidence uses.
        staged_compiled: list[CompiledKnowledge] = []
        staged_occurrences = []
        for item in self.tool_compiler.compile(canonical):
            bundle = self.aligner.stage_atomic(
                item.atomic,
                item.tool,
                item.implementation,
            )
            assert bundle.tool is not None and bundle.implementation is not None
            occurrence = self.aligner.atomic_canonicalizer.rewrite_canonical_occurrence(
                item.occurrence,
                bundle,
                atomic_ref=bundle.atomic.ref,
            )
            staged_occurrences.append(occurrence)
            staged_compiled.append(CompiledKnowledge(
                occurrence,
                bundle.atomic,
                bundle.tool,
                bundle.implementation,
            ))
        existing = self.graph.existing_edges(
            [str(item.proposed_ref) for item in staged_occurrences],
            mode=RuntimeMode.ONLINE,
        )
        composite_proposal = extractor.propose_composite(staged_occurrences, existing)
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
        compiled = staged_compiled
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
        )

    def _apply_evolution(
        self, prepared: _PreparedEvolution, trace: TraceRecord, task: HarnessTask,
    ) -> dict[str, Any]:
        atomic_refs = []
        implementation_refs = []
        tool_refs = []
        validated: dict[str, bool] = {}
        by_occurrence: dict[str, Any] = {}
        for item in prepared.compiled:
            atomic_ref = self.aligner.align_atomic(item.atomic)
            admitted_tool = self.admission.admit_tool(
                item.tool,
                replay=lambda tool, case: bool(self.harness.replay_tool(task, tool, case)),
            )
            tool_alignment = self.aligner.align_tool_with_replays(
                admitted_tool,
                admission=self.admission,
                replay=lambda tool, case: bool(
                    self.harness.replay_tool(task, tool, case)
                ),
            )
            tool_ref = tool_alignment.ref
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
            validated[str(atomic_ref)] = True
            validated[str(tool_ref)] = bool(tool_alignment.admitted)
            validated[str(implementation_ref)] = (
                admitted_implementation.status is SkillStatus.CANDIDATE
            )
            self._add_structural_edge(
                str(implementation_ref), str(atomic_ref), GlobalRelationType.IMPLEMENTS,
                trace.trace_id,
            )
            self._add_structural_edge(
                str(implementation_ref), str(tool_ref), GlobalRelationType.CONTAINS,
                trace.trace_id,
            )

        composite_operation = (
            self._composite_rescue_operation(
                prepared.source_composite_ref, prepared.composite,
            )
            if prepared.source_composite_ref
            else ""
        )
        composite_ref = self.aligner.align_composite(prepared.composite, by_occurrence)
        validated[str(composite_ref)] = True
        if prepared.source_composite_ref and str(composite_ref) != prepared.source_composite_ref:
            self._add_structural_edge(
                str(composite_ref),
                prepared.source_composite_ref,
                (
                    GlobalRelationType.DERIVED_FROM
                    if composite_operation == "revise_composite_sequence"
                    else GlobalRelationType.ALTERNATIVE
                ),
                trace.trace_id,
                evolution_operation=composite_operation or "task_rescue_revision",
            )
        for occurrence_id, atomic_ref in by_occurrence.items():
            self._add_structural_edge(
                str(composite_ref), str(atomic_ref), GlobalRelationType.CONTAINS,
                trace.trace_id, occurrence_id=occurrence_id,
            )

        assets = (
            [(str(ref), "atomic") for ref in atomic_refs]
            + [(str(ref), "implementation") for ref in implementation_refs]
            + [(str(ref), "tool") for ref in tool_refs]
            + [(str(composite_ref), "composite")]
        )
        attempts = tuple(
            CreditAttempt(
                artifact_ref=ref,
                artifact_kind=kind,
                occurrence_id="evolution",
                attempt_id=f"evolution:{ref}",
                sequence_no=index,
                proposed=True,
                validated=validated.get(ref, False),
                metadata={"source": "extractor_admission"},
            )
            for index, (ref, kind) in enumerate(assets)
        )
        events = self.credit.assign(CreditTrace(
            trace.task.task_id, trace.trace_id, attempts
        ))
        self._commit_evidence(events)
        return {
            "atomic_refs": atomic_refs,
            "implementation_refs": implementation_refs,
            "tool_refs": tool_refs,
            "composite_ref": composite_ref,
            "composite_validated": validated[str(composite_ref)],
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
            trace.agent_sessions.append(AgentSessionRecord(
                observed.session.session_id,
                observed.session_type,
                observed.occurrence_id,
                observed.started_at,
                time.time(),
                observed.session.snapshot(),
            ))
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
            if key not in {"empty_bank", "harness_task_count"} and not key.endswith("_error")
        )
        return checks

    def is_empty_knowledge_bank(self) -> bool:
        """Return whether this is the canonical fresh schema-v3 knowledge bank.

        Run/task bookkeeping and traces are experiment outputs, so they do not
        participate in this check.  Every table and file that contributes
        long-term executable/evolution knowledge does.  The database-created
        schema-version metadata row is the sole allowed metadata entry.
        """
        metadata = [
            (str(row["key"]), str(row["value"]))
            for row in self.database.rows("SELECT key,value FROM metadata ORDER BY key")
        ]
        if metadata != [("schema_version", "3")]:
            return False
        for table in _LONG_TERM_KNOWLEDGE_TABLES:
            if self.database.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone() is not None:
                return False
        if self.artifacts.root.exists() and any(
            path.is_file() or path.is_symlink()
            for path in self.artifacts.root.rglob("*")
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
        digest = self.knowledge_digest()
        temporary = Path(tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-", dir=destination.parent
        ))
        try:
            shutil.copytree(self.artifacts.root, temporary / "artifacts")
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
            target_connection.commit()
            target_connection.close()
            atomic_write_json(temporary / "freeze_manifest.json", {
                "schema_version": 3,
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
