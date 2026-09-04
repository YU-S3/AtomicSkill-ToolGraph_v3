"""Static, deterministic, and real-ALFWorld v3 smoke gates."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from atomic_skillgraph.agents import AgentTurn, NativeToolCall
from atomic_skillgraph.agents.provider_probe import (
    ensure_provider_capability,
    run_provider_capability_probe,
)
from atomic_skillgraph.core.serialization import atomic_write_json, to_primitive
from atomic_skillgraph.evolution.failure_extraction_view import (
    DEFAULT_PUBLIC_OBSERVATION_CHAR_LIMIT,
)
from atomic_skillgraph.system import AtomicSkillGraphSystem, load_config

from .protocol import (
    RunManifest,
    TaskManifest,
    hash_code,
    hash_config,
    hash_task_manifest,
    task_signature,
    validate_deepseek_formal_llm,
)
from .report import (
    validate_formal_usage,
    validate_usage_event_persistence,
    write_reports,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

_FAILURE_EXTRACTOR_TASK_ORDINAL = 25
_FAILURE_EXTRACTOR_TASK_ID = "alfworld_train_42_look_at_obj_in_light"
_FAILURE_EXTRACTOR_TASK_SIGNATURE = (
    "73a839b74abb70d40fd5ef84f372d498ee1e13fb855f79ae273f162aa6f2f5d8"
)
_FAILURE_EXTRACTOR_TOKEN_CAP = 262144
_FAILURE_EXTRACTOR_NON_EVENT_PROMPT_CHAR_ALLOWANCE = 200000
_MAX_UTF8_BYTES_PER_CHAR = 4


class _AllUnresolvedC1Provider:
    """Deterministic smoke-only C1 transport for reaching the failure branch.

    It projects the code-generated RequirementExpansion into one unresolved
    step per required instance.  It never reads a benchmark label, task goal,
    entity, observation, or action vocabulary, and it is not used by formal
    training.  All other stages in this smoke still use the configured provider.
    """

    provider_name = "deterministic_c1_fixture"
    model_name = "all-unresolved-v1"

    def __init__(self) -> None:
        self._request_context = {"session_id": "", "stage": ""}
        self._request_records: list[dict[str, Any]] = []

    @property
    def request_record_count(self) -> int:
        return len(self._request_records)

    def set_request_context(self, *, session_id: str, stage: str) -> None:
        self._request_context = {
            "session_id": str(session_id),
            "stage": str(stage),
        }

    def request_records_since(self, index: int) -> tuple[dict[str, Any], ...]:
        return tuple(copy.deepcopy(self._request_records[int(index):]))

    def snapshot(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "model": self.model_name,
            "dialect": "deterministic_smoke_fixture",
            "request_count": len(self._request_records),
            "external_provider": False,
        }

    @staticmethod
    def _requirement_expansion(prompt: str) -> dict[str, Any]:
        marker = "RequirementExpansion:"
        start = prompt.rfind(marker)
        if start < 0:
            raise ValueError("C1 smoke fixture did not receive RequirementExpansion")
        encoded = prompt[start + len(marker):].lstrip()
        value, _ = json.JSONDecoder().raw_decode(encoded)
        if not isinstance(value, dict):
            raise ValueError("C1 smoke RequirementExpansion must be an object")
        return value

    @classmethod
    def _proposal(cls, prompt: str) -> dict[str, Any]:
        expansion = cls._requirement_expansion(prompt)
        required_ids = [
            str(item.get("instance_id", ""))
            for item in expansion.get("instances", ())
            if isinstance(item, dict)
            and isinstance(item.get("requirement"), dict)
            and item["requirement"].get("required") is True
        ]
        if not required_ids or any(not value for value in required_ids):
            raise ValueError("C1 smoke fixture requires non-empty required instances")
        if len(required_ids) != len(set(required_ids)):
            raise ValueError("C1 smoke fixture received duplicate instance ids")
        step_ids = [f"smoke_unresolved_{index:03d}" for index in range(len(required_ids))]
        return {
            "plan_id": "failure_extractor_smoke_all_unresolved",
            "steps": [
                {
                    "step_id": step_id,
                    "requirement_instance_ids": [instance_id],
                    "candidate_source": "unresolved",
                    "candidate_ref": "",
                    "execution_mode": "dynamic",
                    "binding_specs": {},
                    "repeat_role_bindings": {},
                }
                for step_id, instance_id in zip(step_ids, required_ids, strict=True)
            ],
            "control_sequence": step_ids,
            "data_edges": [],
            "dependency_edges": [],
            "requirement_coverage": {
                instance_id: [step_id]
                for step_id, instance_id in zip(step_ids, required_ids, strict=True)
            },
            "referenced_failure_experience_ids": [],
        }

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[Any] | None = None,
    ) -> AgentTurn:
        if not tools or len(tools) != 1 or tools[0].name != "submit_cold_start_plan":
            raise ValueError("C1 smoke fixture requires submit_cold_start_plan")
        prompt = next(
            (
                str(message.get("content", ""))
                for message in reversed(messages)
                if message.get("role") == "user"
            ),
            "",
        )
        started = time.time()
        proposal = self._proposal(prompt)
        sequence = len(self._request_records) + 1
        request_id = f"failure_extractor_smoke_c1_{sequence}"
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "prompt": prompt,
                    "tool": tools[0].name,
                    "proposal": proposal,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        ended = time.time()
        self._request_records.append({
            "request_id": request_id,
            "provider_request_id": request_id,
            "session_id": self._request_context["session_id"],
            "stage": self._request_context["stage"],
            "started_at": started,
            "ended_at": ended,
            "outcome": "success",
            "http_status": None,
            "retry_count": 0,
            "usage_status": "reported",
            "error_code": "",
            "sanitized_error": "",
            "payload_fingerprint": fingerprint,
            "payload_field_names": ["messages", "tools"],
        })
        return AgentTurn(
            content="",
            tool_calls=[NativeToolCall(
                call_id=request_id,
                name=tools[0].name,
                arguments=proposal,
            )],
            finish_reason="tool_calls",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            reasoning_tokens=0,
            latency_ms=max(0.0, (ended - started) * 1000.0),
            provider_metadata={
                "provider": self.provider_name,
                "model": self.model_name,
                "usage_status": "reported",
                "fixture": True,
                "external_provider": False,
            },
        )


def _install_failure_extractor_c1_fixture(
    system: AtomicSkillGraphSystem,
) -> _AllUnresolvedC1Provider:
    """Install a C1-only fixture while preserving every real stage provider."""

    real_planner = system._provider("planner")
    fixture = _AllUnresolvedC1Provider()
    fixture_key = "failure_extractor_smoke_c1_fixture"
    system._provider_cache[fixture_key] = fixture

    def factory(task: object, contract: object) -> object:
        current = system._provider_cache.get("planner")
        if current is not real_planner:
            raise RuntimeError("planner provider changed before C1 smoke fixture")
        system._provider_cache["planner"] = fixture
        try:
            return system._cold_start_session(task, contract)
        finally:
            system._provider_cache["planner"] = real_planner

    system.planner.cold_start_session_factory = factory
    return fixture


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _validate_configured_task_manifest(
    config: dict[str, object], system: AtomicSkillGraphSystem,
) -> dict[str, object]:
    """Materialize the configured deterministic selection during preflight.

    A schema-only dummy manifest cannot establish that the installed ALFWorld
    dataset contains the requested balanced split.  Formal configurations
    therefore scan their configured selection and bind its concrete identities
    before a paid run is allowed to start.
    """

    harness = dict(config.get("harness") or {})
    selection = dict(harness.get("task_selection") or {})
    if not selection:
        task = TaskManifest(0, "preflight_task", "preflight_signature", "preflight")
        manifest_hash = hash_task_manifest((task,))
        return {
            "task_manifest_schema": bool(manifest_hash),
            "task_manifest_selection": "not_configured",
            "task_manifest_hash": manifest_hash,
        }
    if selection.get("policy") != "balanced_fixed_manifest":
        raise ValueError("preflight requires task_selection.policy=balanced_fixed_manifest")
    task_types = [str(item) for item in selection.get("task_types", [])]
    per_type = int(selection.get("tasks_per_type", 0))
    total = int(selection.get("total_tasks", 0))
    if not task_types or per_type <= 0 or total != len(task_types) * per_type:
        raise ValueError("configured balanced task count is inconsistent")
    if selection.get("require_exact_count") is not True:
        raise ValueError("configured task selection must require exact count")
    tasks = system.harness.load_balanced_tasks(task_types, per_type)
    counts = {label: sum(task.task_type == label for task in tasks) for label in task_types}
    task_ids = [task.task_id for task in tasks]
    signatures = [task_signature(task) for task in tasks]
    if (
        len(tasks) != total
        or any(count != per_type for count in counts.values())
        or len(set(task_ids)) != total
        or len(set(signatures)) != total
    ):
        raise ValueError(
            "configured task manifest is not exact, balanced, and identity-unique: "
            f"total={len(tasks)}, counts={counts}, unique_ids={len(set(task_ids))}, "
            f"unique_signatures={len(set(signatures))}"
        )
    items = tuple(
        TaskManifest.from_task(
            task,
            ordinal=index,
            knowledge_milestone="preflight",
            split=str(system.harness.split),
        )
        for index, task in enumerate(tasks)
    )
    return {
        "task_manifest_schema": True,
        "task_manifest_selection": True,
        "task_manifest_task_count": total,
        "task_manifest_counts": counts,
        "task_manifest_hash": hash_task_manifest(items),
    }


def run_preflight(config_path: str | Path) -> int:
    config = load_config(_path(config_path))
    with tempfile.TemporaryDirectory(prefix="asg_v3_preflight_") as temporary:
        isolated = Path(temporary)
        config["data_dir"] = str(isolated / "data_v3")
        config["trace_data_dir"] = str(isolated / "traces")
        experiment = dict(config.get("experiment") or {})
        experiment.update({
            "condition": "full",
            "runtime_mode": "online",
            "freeze_skills": False,
            "initialize_v3_bank": "empty",
        })
        config["experiment"] = experiment
        with AtomicSkillGraphSystem(config, readonly=False) as system:
            checks = system.preflight(require_api_key=True, initialize_harness=True)
            try:
                task_checks = _validate_configured_task_manifest(config, system)
                checks.update(task_checks)
                tasks = (
                    TaskManifest(0, "preflight_task", "preflight_signature", "preflight"),
                )
                RunManifest.create(
                    run_id="preflight", phase="preflight", config_hash="config",
                    code_commit="code", knowledge_digest=system.knowledge_digest(), tasks=tasks,
                )
            except Exception as exc:
                checks["task_manifest_schema"] = False
                checks["task_manifest_selection"] = False
                checks["task_manifest_error"] = str(exc)
            checks["passed"] = bool(
                checks.get("passed")
                and checks["task_manifest_schema"]
                and checks["task_manifest_selection"] in {True, "not_configured"}
            )
            print(json.dumps(checks, ensure_ascii=False, indent=2))
            return 0 if checks["passed"] else 1


def run_provider_probe(config_path: str | Path) -> int:
    config_path = _path(config_path)
    config = load_config(config_path)
    validate_deepseek_formal_llm(config)
    output = _path(
        (config.get("experiment") or {}).get(
            "output_dir", "runs/alfworld_train_full_30"
        )
    )
    try:
        manifest = run_provider_capability_probe(
            config,
            output_dir=output,
            config_hash=hash_config(config_path),
            code_hash=hash_code(REPO_ROOT),
        )
    except Exception as exc:
        print(json.dumps({
            "passed": False,
            "gate": "deepseek_provider_capability",
            "error_type": type(exc).__name__,
            "error_code": str(getattr(exc, "code", "")),
            "error": str(exc),
            "output_dir": str(output),
        }, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps({
        "passed": True,
        "gate": "deepseek_provider_capability",
        "output_dir": str(output),
        "manifest": manifest,
    }, ensure_ascii=False, indent=2))
    return 0


def run_deterministic() -> int:
    command = [
        sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
        "tests/test_deterministic_fullchain.py", "tests/test_agent_finalization.py",
        "tests/test_failure_extraction_view.py", "tests/test_failure_extractor.py",
        "tests/test_deepseek_protocol.py",
        "tests/test_r3_runtime_state.py",
        "tests/test_r3_semantic_anchors.py",
        "tests/test_r21_replay_report.py",
        "tests/test_v32_frozen_design.py",
        "tests/test_extractor_contract_authority.py",
        "tests/test_v32_r1_gates.py",
        "tests/test_v32_r21_cross_task_reuse.py",
        "tests/test_v32_r21_atomicizer_vocabulary_tolerance.py",
        "src/atomic_skillgraph/governance/tests/test_governance.py",
        "experiments/tests/test_protocol_report.py",
        "experiments/tests/test_failure_extractor_smoke.py",
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode:
        return completed.returncode
    print(json.dumps({
        "passed": True,
        "gate": "deterministic_no_api_fullchain",
        "episodes": 4,
        "coverage": [
            "dynamic_to_evolution", "candidate_direct", "preflight_to_fresh_seeded",
            "task_rescue", "ledger_exactly_once", "token_reconciliation", "frozen_digest",
        ],
    }, ensure_ascii=False, indent=2))
    return 0


def _actual_started_direct(trace: object) -> bool:
    for node in getattr(trace, "node_records", ()):
        status = getattr(getattr(node, "status", ""), "value", getattr(node, "status", ""))
        if status not in {
            "direct_autonomous_success", "direct_agent_prepared_success",
        }:
            continue
        occurrence_id = str(getattr(node, "occurrence_id", ""))
        invocations = [
            item for item in getattr(trace, "implementation_invocations", ())
            if str(getattr(item, "occurrence_id", "")) == occurrence_id
            and dict(getattr(item, "preflight", {}) or {}).get("passed") is True
            and dict(getattr(item, "result", {}) or {}).get("started") is True
            and dict(getattr(item, "result", {}) or {}).get("completed") is True
        ]
        tools = [
            item for item in getattr(trace, "tool_executions", ())
            if str(getattr(item, "occurrence_id", "")) == occurrence_id
            and dict(getattr(item, "result", {}) or {}).get("started") is True
            and dict(getattr(item, "result", {}) or {}).get("completed") is True
        ]
        if invocations and tools:
            return True
    return False


def _validated_dataflow(trace: object) -> bool:
    plan = dict(getattr(trace, "runtime_plan", {}) or {})
    changes = [to_primitive(item) for item in getattr(trace, "binding_changes", ())]
    occurrences = {
        str(item.get("step_id", "")): str(item.get("occurrence_id", ""))
        for item in (plan.get("occurrences") or ())
        if isinstance(item, dict)
    }
    invocations = [
        to_primitive(item)
        for item in getattr(trace, "implementation_invocations", ())
    ]

    # A binding-store write alone is not consumption.  Match one declared
    # edge end-to-end: validator-backed source publication -> target DataFlow
    # binding -> passed and actually-started downstream Implementation whose
    # concrete argument contains that same value.
    consumed_edge = False
    for edge in plan.get("data_edges") or ():
        if not isinstance(edge, dict):
            continue
        source_occurrence = occurrences.get(str(edge.get("source_step", "")), "")
        target_occurrence = occurrences.get(str(edge.get("target_step", "")), "")
        source_role = str(edge.get("source_role", ""))
        target_role = str(edge.get("target_role", ""))
        publications = [
            dict(item.get("current") or {})
            for item in changes
            if item.get("reason") == "validated_output_published"
            and str(item.get("occurrence_id", "")) == source_occurrence
            and str(item.get("role", "")) == source_role
        ]
        for publication in publications:
            value = publication.get("value")
            flowed = any(
                item.get("reason") == "data_flow"
                and str(item.get("occurrence_id", "")) == target_occurrence
                and str(item.get("role", "")) == target_role
                and str(dict(item.get("current") or {}).get("source", "")) == "data_flow"
                and dict(item.get("current") or {}).get("value") == value
                for item in changes
            )
            downstream_started = any(
                str(item.get("occurrence_id", "")) == target_occurrence
                and dict(item.get("preflight") or {}).get("passed") is True
                and dict(item.get("arguments") or {}).get(target_role) == value
                and dict(item.get("result") or {}).get("started") is True
                and dict(item.get("result") or {}).get("completed") is True
                and dict(item.get("result") or {}).get("atomic_effect_passed") is True
                for item in invocations
            )
            if flowed and downstream_started:
                consumed_edge = True
                break
        if consumed_edge:
            break
    return bool(
        len(occurrences) >= 2
        and consumed_edge
        and getattr(trace, "graph_self_sufficient_success", False)
        and not getattr(trace, "task_rescue_required", False)
    )


def _failure_extractor_smoke_audit(
    trace: object,
    *,
    harness_max_steps: int,
) -> dict[str, object]:
    """Audit the bounded failure-learning path without interpreting ALFWorld state."""

    failures = [to_primitive(item) for item in getattr(trace, "failures", ())]
    failure_codes = {
        str(item.get("code", ""))
        for item in failures
        if isinstance(item, dict)
    }
    extraction = to_primitive(getattr(trace, "failure_extraction", None))
    extraction_recorded = isinstance(extraction, dict)
    rejection = (
        dict(extraction.get("rejection") or {})
        if extraction_recorded
        else {}
    )

    metadata = dict(getattr(trace, "metadata", {}) or {})
    metrics = dict(metadata.get("failure_extractor_metrics") or {})
    f1_event_count = metrics.get("failure_extractor_f1_input_event_count")
    f1_prompt_chars = metrics.get("failure_extractor_f1_prompt_chars")
    f1_prompt_bytes = metrics.get("failure_extractor_f1_prompt_bytes")
    f2_span_count = metrics.get("failure_extractor_f2_span_count")
    f2_source_event_count = metrics.get(
        "failure_extractor_f2_source_event_count"
    )
    f2_prompt_chars = metrics.get("failure_extractor_f2_prompt_chars")
    f2_prompt_bytes = metrics.get("failure_extractor_f2_prompt_bytes")

    def bounded_integer(value: object, *, lower: int, upper: int) -> bool:
        return (
            not isinstance(value, bool)
            and isinstance(value, int)
            and lower <= value <= upper
        )

    def prompt_limits(event_count: object) -> tuple[int, int]:
        # The view retains every event but bounds each public observation.  The
        # remaining allowance covers the fixed plan/contract/alignment envelope.
        # Deriving this from the recorded event count avoids rejecting a legal
        # 100-event view merely because 100 * the per-event bound exceeds a
        # one-size-fits-all smoke constant.
        normalized_count = (
            int(event_count)
            if bounded_integer(event_count, lower=0, upper=harness_max_steps)
            else int(harness_max_steps)
        )
        char_limit = (
            _FAILURE_EXTRACTOR_NON_EVENT_PROMPT_CHAR_ALLOWANCE
            + normalized_count * DEFAULT_PUBLIC_OBSERVATION_CHAR_LIMIT
        )
        return char_limit, char_limit * _MAX_UTF8_BYTES_PER_CHAR

    f1_prompt_char_limit, f1_prompt_byte_limit = prompt_limits(f1_event_count)
    f2_prompt_char_limit, f2_prompt_byte_limit = prompt_limits(
        f2_source_event_count
    )

    usage = [to_primitive(item) for item in getattr(trace, "llm_usage", ())]
    f1_usage = [
        item for item in usage
        if isinstance(item, dict)
        and item.get("bucket") == "failure_extractor_f1"
        and str(item.get("session_id", ""))
        and int(item.get("call_count", 0)) == 1
    ]
    f2_usage = [
        item for item in usage
        if isinstance(item, dict)
        and item.get("bucket") == "failure_extractor_f2"
        and str(item.get("session_id", ""))
        and int(item.get("call_count", 0)) == 1
    ]
    c1_usage = [
        item for item in usage
        if isinstance(item, dict)
        and item.get("bucket") == "cold_start_c1"
    ]
    c1_repair_usage = [
        item for item in usage
        if isinstance(item, dict)
        and item.get("bucket") == "cold_start_c1_repair"
    ]
    f1_session_ids = {str(item["session_id"]) for item in f1_usage}
    f2_session_ids = {str(item["session_id"]) for item in f2_usage}
    c1_session_ids = {
        str(item.get("session_id", "")) for item in c1_usage
        if str(item.get("session_id", ""))
    }
    provider_requests = [
        to_primitive(item) for item in getattr(trace, "provider_requests", ())
    ]
    f1_provider_requests = [
        item for item in provider_requests
        if isinstance(item, dict)
        and str(item.get("session_id", "")) in f1_session_ids
    ]
    f2_provider_requests = [
        item for item in provider_requests
        if isinstance(item, dict)
        and str(item.get("session_id", "")) in f2_session_ids
    ]
    c1_provider_requests = [
        item for item in provider_requests
        if isinstance(item, dict)
        and str(item.get("session_id", "")) in c1_session_ids
    ]
    cold_start_plan = to_primitive(getattr(trace, "cold_start_plan", None))
    cold_validation = (
        dict(cold_start_plan.get("validation") or {})
        if isinstance(cold_start_plan, dict)
        else {}
    )
    cold_proposal = (
        dict(cold_start_plan.get("proposal") or {})
        if isinstance(cold_start_plan, dict)
        else {}
    )
    cold_steps = [
        item for item in cold_proposal.get("steps", ())
        if isinstance(item, dict)
    ]
    real_stage_buckets = {
        "planner_p1",
        "runtime_dynamic_cold_start_continuation",
        "failure_extractor_f1",
        "failure_extractor_f2",
    }
    real_stage_usage = {
        bucket: [
            item for item in usage
            if isinstance(item, dict) and item.get("bucket") == bucket
        ]
        for bucket in real_stage_buckets
    }

    checks = {
        "benchmark_failure_returned": (
            getattr(trace, "benchmark_success", None) is False
        ),
        "strict_failure_returned": (
            getattr(trace, "strict_task_success", None) is False
        ),
        "infrastructure_neutral": (
            getattr(trace, "infrastructure_failure", None) is False
        ),
        "resource_usage_complete": (
            getattr(trace, "resource_usage_complete", None) is True
        ),
        "runtime_task_token_budget_exhausted": (
            "runtime_task_token_budget_exhausted" in failure_codes
        ),
        "failure_extraction_recorded": extraction_recorded,
        "failure_extractor_budget_not_exhausted": (
            str(rejection.get("code", ""))
            != "failure_extractor_budget_exhausted"
        ),
        "f1_input_event_count_bounded": bounded_integer(
            f1_event_count, lower=0, upper=harness_max_steps,
        ),
        "f1_prompt_chars_bounded": bounded_integer(
            f1_prompt_chars,
            lower=1,
            upper=f1_prompt_char_limit,
        ),
        "f1_prompt_bytes_bounded": bounded_integer(
            f1_prompt_bytes,
            lower=1,
            upper=f1_prompt_byte_limit,
        ),
        "f1_usage_audited": bool(f1_usage),
        "f1_provider_requests_audited": bool(f1_provider_requests) and all(
            item.get("usage_status") == "reported"
            for item in f1_provider_requests
        ),
        "f2_span_count_bounded": bounded_integer(
            f2_span_count, lower=0, upper=harness_max_steps,
        ),
        "f2_source_event_count_bounded": bounded_integer(
            f2_source_event_count, lower=0, upper=harness_max_steps,
        ),
        "f2_prompt_chars_bounded": bounded_integer(
            f2_prompt_chars,
            lower=1,
            upper=f2_prompt_char_limit,
        ),
        "f2_prompt_bytes_bounded": bounded_integer(
            f2_prompt_bytes,
            lower=1,
            upper=f2_prompt_byte_limit,
        ),
        "f2_usage_audited": bool(f2_usage),
        "f2_provider_requests_audited": bool(f2_provider_requests) and all(
            item.get("usage_status") == "reported"
            for item in f2_provider_requests
        ),
        "deterministic_c1_fixture_audited": (
            len(c1_usage) == 1
            and c1_usage[0].get("provider")
            == _AllUnresolvedC1Provider.provider_name
            and c1_usage[0].get("model")
            == _AllUnresolvedC1Provider.model_name
            and int(c1_usage[0].get("total_tokens", -1)) == 0
            and len(c1_provider_requests) == 1
            and c1_provider_requests[0].get("usage_status") == "reported"
            and c1_provider_requests[0].get("http_status") is None
        ),
        "c1_semantic_repair_not_used": not c1_repair_usage,
        "cold_start_plan_valid_unresolved": (
            cold_validation.get("passed") is True
            and bool(cold_steps)
            and all(
                item.get("candidate_source") == "unresolved"
                and item.get("candidate_ref") == ""
                and item.get("execution_mode") == "dynamic"
                for item in cold_steps
            )
        ),
        "real_provider_stages_uncontaminated": all(
            bool(items)
            and all(
                item.get("provider") == "openai_compatible"
                and item.get("model") == "deepseek-v4-flash"
                for item in items
            )
            for items in real_stage_usage.values()
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failure_codes": sorted(failure_codes),
        "failure_extraction_rejection": rejection,
        "failure_extractor_metrics": metrics,
        "failure_extractor_prompt_limits": {
            "f1_chars": f1_prompt_char_limit,
            "f1_bytes": f1_prompt_byte_limit,
            "f2_chars": f2_prompt_char_limit,
            "f2_bytes": f2_prompt_byte_limit,
            "public_observation_chars_per_event": (
                DEFAULT_PUBLIC_OBSERVATION_CHAR_LIMIT
            ),
        },
        "failure_extractor_f1_usage": f1_usage,
        "failure_extractor_f1_provider_requests": f1_provider_requests,
        "failure_extractor_f2_usage": f2_usage,
        "failure_extractor_f2_provider_requests": f2_provider_requests,
        "cold_start_c1_fixture_usage": c1_usage,
        "cold_start_c1_fixture_provider_requests": c1_provider_requests,
    }


def run_failure_extractor_smoke(config_path: str | Path) -> int:
    """Run the frozen ordinal-25 failure-learning gate in an isolated empty bank."""

    config_path = _path(config_path)
    config = copy.deepcopy(load_config(config_path))
    validate_deepseek_formal_llm(config)
    base_output = _path(
        (config.get("experiment") or {}).get(
            "output_dir", "runs/alfworld_train_full_30"
        )
    )
    capability = ensure_provider_capability(
        config,
        output_dir=base_output,
        config_hash=hash_config(config_path),
        code_hash=hash_code(REPO_ROOT),
        run_if_missing=False,
    )

    llm = dict(config.get("llm") or {})
    extractor_llm = dict(llm.get("extractor") or {})
    extractor_cap = int(extractor_llm.get("max_total_tokens_per_task", 0))
    if extractor_cap != _FAILURE_EXTRACTOR_TOKEN_CAP:
        raise ValueError(
            "failure-extractor smoke requires the formal extractor token cap "
            f"{_FAILURE_EXTRACTOR_TOKEN_CAP}, got {extractor_cap}"
        )
    runtime_llm = dict(llm.get("runtime") or {})
    runtime_llm["max_total_tokens_per_task"] = 1
    llm["runtime"] = runtime_llm
    config["llm"] = llm

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    smoke_root = base_output.parent / f"{base_output.name}_failure_extractor_smoke"
    output = smoke_root / f"run_{stamp}_{os.getpid()}"
    experiment = dict(config.get("experiment") or {})
    experiment.update({
        "name": f"failure_extractor_smoke_{stamp}",
        "phase": "smoke",
        "condition": "full",
        "runtime_mode": "online",
        "freeze_skills": False,
        "initialize_v3_bank": "empty",
        "output_dir": str(output),
    })
    config["experiment"] = experiment
    config["data_dir"] = str(output / "data_v3")
    config["trace_data_dir"] = str(output)

    with AtomicSkillGraphSystem(config, readonly=False) as system:
        preflight = system.preflight(
            require_api_key=True,
            initialize_harness=True,
            require_empty_bank=True,
        )
        if not preflight.get("passed") or not system.is_empty_knowledge_bank():
            result = {
                "passed": False,
                "gate": "failure_extractor_real_alfworld",
                "output_dir": str(output),
                "preflight": preflight,
                "empty_bank": system.is_empty_knowledge_bank(),
            }
            atomic_write_json(output / "failure_extractor_smoke_result.json", result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 1

        harness = dict(config.get("harness") or {})
        selection = dict(harness.get("task_selection") or {})
        if selection.get("policy") != "balanced_fixed_manifest":
            raise ValueError(
                "failure-extractor smoke requires balanced_fixed_manifest selection"
            )
        task_types = [str(item) for item in selection.get("task_types", ())]
        tasks_per_type = int(selection.get("tasks_per_type", 0))
        expected_total = int(selection.get("total_tasks", 0))
        tasks = system.harness.load_balanced_tasks(task_types, tasks_per_type)
        counts = {
            label: sum(task.task_type == label for task in tasks)
            for label in task_types
        }
        if (
            expected_total != 30
            or len(tasks) != expected_total
            or selection.get("require_exact_count") is not True
            or any(count != tasks_per_type for count in counts.values())
            or len({task_signature(task) for task in tasks}) != expected_total
        ):
            raise RuntimeError(
                "failure-extractor smoke requires the exact distinct formal 30-task selection"
            )

        selected = tasks[_FAILURE_EXTRACTOR_TASK_ORDINAL]
        selected_signature = task_signature(selected)
        if (
            selected.task_id != _FAILURE_EXTRACTOR_TASK_ID
            or selected_signature != _FAILURE_EXTRACTOR_TASK_SIGNATURE
        ):
            raise RuntimeError(
                "formal ordinal-25 task identity changed: "
                f"task_id={selected.task_id!r}, signature={selected_signature!r}"
            )

        initial_digest = system.knowledge_digest()
        manifest_items = tuple(
            TaskManifest(
                index,
                task.task_id,
                task_signature(task),
                f"isolated_smoke_selection:{initial_digest}",
                task.benchmark,
                str(system.harness.split),
                json.dumps({
                    "task_type": task.task_type,
                    "env_index": task.context.get("env_index"),
                    "game_file": task.context.get("game_file", ""),
                }, ensure_ascii=False, sort_keys=True),
            )
            for index, task in enumerate(tasks)
        )
        atomic_write_json(output / "task_manifest.json", {
            "schema_version": 3,
            "task_manifest_hash": hash_task_manifest(manifest_items),
            "selected_ordinal": _FAILURE_EXTRACTOR_TASK_ORDINAL,
            "selected_task_id": selected.task_id,
            "selected_task_signature": selected_signature,
            "tasks": [item.to_dict() for item in manifest_items],
        })

        _install_failure_extractor_c1_fixture(system)
        try:
            trace = system.run_task(selected)
        except Exception as exc:
            result = {
                "passed": False,
                "gate": "failure_extractor_real_alfworld",
                "task_returned": False,
                "output_dir": str(output),
                "selected_ordinal": _FAILURE_EXTRACTOR_TASK_ORDINAL,
                "selected_task_id": selected.task_id,
                "cold_start_c1_mode": "deterministic_all_unresolved_fixture",
                "error_type": type(exc).__name__,
                "error_code": str(getattr(exc, "code", "")),
                "error": str(exc),
            }
            atomic_write_json(output / "failure_extractor_smoke_result.json", result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 1

        audit = _failure_extractor_smoke_audit(
            trace,
            harness_max_steps=int(harness.get("max_steps", 100)),
        )
        persisted_traces = list(system.traces.iter_payloads())
        validate_formal_usage(persisted_traces)
        validate_usage_event_persistence(system.usage.events, persisted_traces)
        write_reports(
            [trace], output / "reports", stem="failure_extractor_smoke",
        )
        result = {
            **audit,
            "passed": bool(
                capability.get("passed") is True and audit["passed"]
            ),
            "gate": "failure_extractor_real_alfworld",
            "task_returned": True,
            "output_dir": str(output),
            "trace_id": trace.trace_id,
            "trace_path": str(system.traces.root / f"{trace.trace_id}.json"),
            "report_dir": str(output / "reports"),
            "selected_ordinal": _FAILURE_EXTRACTOR_TASK_ORDINAL,
            "selected_task_id": selected.task_id,
            "selected_task_signature": selected_signature,
            "cold_start_c1_mode": "deterministic_all_unresolved_fixture",
            "runtime_task_token_cap": 1,
            "extractor_task_token_cap": extractor_cap,
            "provider_capability_passed": capability.get("passed") is True,
        }
        atomic_write_json(output / "failure_extractor_smoke_result.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["passed"] else 1


def run_real_alfworld(config_path: str | Path) -> int:
    config_path = _path(config_path)
    config = copy.deepcopy(load_config(config_path))
    validate_deepseek_formal_llm(config)
    base_output = _path(
        (config.get("experiment") or {}).get("output_dir", "runs/v3_real_smoke")
    )
    capability = ensure_provider_capability(
        config,
        output_dir=base_output,
        config_hash=hash_config(config_path),
        code_hash=hash_code(REPO_ROOT),
        run_if_missing=False,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = base_output / f"run_{stamp}_{os.getpid()}"
    experiment = dict(config.get("experiment") or {})
    experiment.update({
        "name": f"v3_real_smoke_{stamp}",
        "phase": "smoke",
        "condition": "full",
        "runtime_mode": "online",
        "freeze_skills": False,
        "initialize_v3_bank": "empty",
        "output_dir": str(output),
    })
    config["experiment"] = experiment
    config["data_dir"] = str(output / "data_v3")
    with AtomicSkillGraphSystem(config, readonly=False) as system:
        preflight = system.preflight(require_api_key=True, initialize_harness=True)
        if not preflight["passed"]:
            print(json.dumps(preflight, ensure_ascii=False, indent=2))
            return 1
        pick_tasks = system.harness.load_balanced_tasks(["pick_and_place_simple"], 5)
        multi_tasks = system.harness.load_balanced_tasks(
            ["pick_heat_then_place_in_recep"], 2,
        )
        tasks = [*pick_tasks, *multi_tasks]
        if len(tasks) != 7 or len({task_signature(task) for task in tasks}) != 7:
            raise RuntimeError(
                "real smoke requires 3 cold + 2 unseen warm pick-and-place, "
                "then distinct heat learning and heat data-flow reuse tasks"
            )
        task_items = tuple(
            TaskManifest(
                index,
                task.task_id,
                task_signature(task),
                (
                    "cold_learning" if index < 3
                    else "warm_reuse" if index < 5
                    else "multi_node_learning" if index == 5
                    else "multi_node_dataflow"
                ),
                task.benchmark,
                str(system.harness.split),
                json.dumps({
                    "phase": (
                        "cold_learning" if index < 3
                        else "warm_reuse" if index < 5
                        else "multi_node_learning" if index == 5
                        else "multi_node_dataflow"
                    ),
                    "task_type": task.task_type,
                    "env_index": task.context.get("env_index"),
                    "game_file": task.context.get("game_file", ""),
                }, ensure_ascii=False, sort_keys=True),
            )
            for index, task in enumerate(tasks)
        )
        atomic_write_json(output / "task_manifest.json", {
            "schema_version": 3,
            "task_manifest_hash": hash_task_manifest(task_items),
            "tasks": [item.to_dict() for item in task_items],
        })
        cold_traces = [system.run_task(task) for task in tasks[:3]]
        warm_traces = [system.run_task(task) for task in tasks[3:5]]
        # The first heat task learns the previously absent heat Atomic and its
        # two-node Composite.  Only a distinct subsequent task can prove that
        # persisted graph and DataFlow at Runtime; post-terminal learning must
        # never be counted retroactively as same-task execution.
        multi_traces = [system.run_task(task) for task in tasks[5:7]]
        traces = [*cold_traces, *warm_traces, *multi_traces]
        artifact_counts = {
            str(row["artifact_kind"]): int(row["count"])
            for row in system.database.execute(
                "SELECT artifact_kind,COUNT(*) AS count FROM artifact_index GROUP BY artifact_kind"
            ).fetchall()
        }
        four_layer_assets = all(
            artifact_counts.get(kind, 0) > 0
            for kind in ("atomic", "implementation", "tool", "composite")
        )
        actual_started_direct = any(_actual_started_direct(trace) for trace in warm_traces)
        # Only the second, distinct heat task may satisfy this gate.  Earlier
        # tasks can learn the graph but must not substitute for downstream
        # Runtime consumption of the persisted DataFlow.
        dataflow_trace = multi_traces[-1]
        dataflow_proven = _validated_dataflow(dataflow_trace)
        cold_dynamic_success = any(
            trace.benchmark_success
            and trace.runtime_plan.get("source") == "full_dynamic"
            for trace in cold_traces
        )
        unknown_actions = sum(
            action.action_type == "UNKNOWN"
            for trace in traces for action in trace.environment_actions
        )
        contract_mismatches = sum(
            failure.code in {
                "task_contract_mismatch", "benchmark_goal_contract_mismatch",
            }
            for trace in traces for failure in trace.failures
        )
        passed = (
            capability.get("passed") is True
            and cold_dynamic_success
            and four_layer_assets
            and actual_started_direct
            and dataflow_proven
            and all(not trace.infrastructure_failure for trace in traces)
            and all(trace.resource_usage_complete for trace in traces)
            and unknown_actions == 0
            and contract_mismatches == 0
        )
        persisted_traces = list(system.traces.iter_payloads())
        validate_formal_usage(persisted_traces)
        validate_usage_event_persistence(system.usage.events, persisted_traces)
        write_reports(traces, output / "reports", stem="real_alfworld_smoke")
        result = {
            "passed": passed,
            "output_dir": str(output),
            "tasks": len(traces),
            "cold_tasks": len(cold_traces),
            "warm_unseen_tasks": len(warm_traces),
            "multi_node_tasks": len(multi_traces),
            "validated_dataflow_task_id": dataflow_trace.task.task_id,
            "successes": sum(trace.benchmark_success for trace in traces),
            "strict_task_successes": sum(
                trace.strict_task_success for trace in traces
            ),
            "learning_eligible_successes": sum(
                trace.learning_eligible for trace in traces
            ),
            "artifact_counts": artifact_counts,
            "four_layer_assets": four_layer_assets,
            "cold_dynamic_success": cold_dynamic_success,
            "actual_started_direct": actual_started_direct,
            "validated_dataflow": dataflow_proven,
            "unknown_alfworld_actions": unknown_actions,
            "won_task_contract_mismatches": contract_mismatches,
            "provider_capability_passed": capability.get("passed") is True,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--provider-probe", action="store_true")
    modes.add_argument("--deterministic", action="store_true")
    modes.add_argument("--real-alfworld", action="store_true")
    modes.add_argument("--failure-extractor", action="store_true")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)
    if args.preflight:
        return run_preflight(args.config)
    if args.provider_probe:
        return run_provider_probe(args.config)
    if args.deterministic:
        return run_deterministic()
    if args.failure_extractor:
        return run_failure_extractor_smoke(args.config)
    return run_real_alfworld(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
