"""Static, deterministic, and real-ALFWorld v3 smoke gates."""

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping
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
        "tests",
        "experiments/tests",
        "src/atomic_skillgraph/governance/tests",
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode:
        return completed.returncode
    print(json.dumps({
        "passed": True,
        "gate": "deterministic_no_api_fullchain",
        "collection": "full_ours_test_roots",
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


def _snapshot_state_identity(snapshot: Mapping[str, Any]) -> str | None:
    """Return the canonical semantic state certified by one snapshot.

    Runner-side auditing intentionally validates only the frozen snapshot
    contract.  It never derives a fact from an ALFWorld action or observation.
    """

    facts = snapshot.get("facts")
    if not isinstance(facts, list):
        return None
    canonical_facts: list[str] = []
    fact_identities: set[tuple[str, str]] = set()
    for raw_fact in facts:
        if not isinstance(raw_fact, Mapping):
            return None
        predicate = str(raw_fact.get("predicate", ""))
        arguments = raw_fact.get("args")
        effect_domain = str(raw_fact.get("effect_domain", ""))
        witness_ref = str(raw_fact.get("witness_ref", ""))
        if (
            not predicate
            or not isinstance(arguments, Mapping)
            or effect_domain not in {"world", "evidence"}
            or not witness_ref
        ):
            return None
        arguments_json = json.dumps(
            dict(arguments), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        )
        fact_identity = (predicate, arguments_json)
        if fact_identity in fact_identities:
            return None
        fact_identities.add(fact_identity)
        canonical_facts.append(json.dumps({
            "predicate": predicate,
            "args": dict(arguments),
            "effect_domain": effect_domain,
            "witness_ref": witness_ref,
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    if not isinstance(snapshot.get("done"), bool):
        return None
    if not isinstance(snapshot.get("won"), bool):
        return None
    return json.dumps({
        "done": snapshot["done"],
        "won": snapshot["won"],
        "facts": sorted(canonical_facts),
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _look_at_authority_smoke_audit(
    trace: object,
    normalized: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit the returned trace without manufacturing semantic authority."""

    metadata = dict(getattr(trace, "metadata", {}) or {})
    raw_snapshots = metadata.get("semantic_state_snapshots")
    snapshots = (
        [dict(item) for item in raw_snapshots if isinstance(item, Mapping)]
        if isinstance(raw_snapshots, list)
        else []
    )
    actions: list[dict[str, Any]] = []
    for raw_action in getattr(trace, "environment_actions", ()):
        item = to_primitive(raw_action)
        if isinstance(item, Mapping):
            actions.append(dict(item))
        elif hasattr(item, "__dict__"):
            actions.append(dict(vars(item)))
    normalized_actions = [
        dict(item)
        for item in normalized.get("actions", ())
        if isinstance(item, Mapping)
    ]

    snapshot_contract = bool(snapshots) and len(snapshots) == len(raw_snapshots)
    sequence_complete = snapshot_contract and all(
        not isinstance(snapshot.get("sequence_index"), bool)
        and isinstance(snapshot.get("sequence_index"), int)
        and snapshot["sequence_index"] == index
        and not isinstance(snapshot.get("revision"), bool)
        and isinstance(snapshot.get("revision"), int)
        and isinstance(snapshot.get("origin"), str)
        and isinstance(snapshot.get("action_id"), str)
        and isinstance(snapshot.get("occurrence_id"), str)
        and isinstance(snapshot.get("accepted"), bool)
        and _snapshot_state_identity(snapshot) is not None
        for index, snapshot in enumerate(snapshots)
    )

    revision_states: dict[int, str] = {}
    revision_consistent = sequence_complete
    if revision_consistent:
        for snapshot in snapshots:
            revision = int(snapshot["revision"])
            state = _snapshot_state_identity(snapshot)
            assert state is not None
            previous = revision_states.setdefault(revision, state)
            if previous != state:
                revision_consistent = False
                break

    reset_snapshot_present = bool(snapshots) and all((
        snapshots[0].get("sequence_index") == 0,
        snapshots[0].get("origin") == "reset",
        snapshots[0].get("action_id") == "",
        snapshots[0].get("occurrence_id") == "",
        snapshots[0].get("accepted") is True,
    ))
    every_step_snapshot_present = (
        sequence_complete
        and len(snapshots) == len(actions) + 1
    )
    action_timeline_consistent = every_step_snapshot_present
    if action_timeline_consistent:
        for index, action in enumerate(actions):
            before = snapshots[index]
            after = snapshots[index + 1]
            revision = action.get("revision")
            new_revision = action.get("new_revision")
            if (
                isinstance(revision, bool)
                or not isinstance(revision, int)
                or isinstance(new_revision, bool)
                or not isinstance(new_revision, int)
                or before.get("revision") != revision
                or after.get("revision") != new_revision
                or after.get("action_id") != str(action.get("action_id", ""))
                or after.get("accepted") is not action.get("accepted")
                or after.get("origin") == "reset"
                or revision not in revision_states
                or new_revision not in revision_states
            ):
                action_timeline_consistent = False
                break

    failure_codes = [
        str(
            failure.get("code", "")
            if isinstance(failure, Mapping)
            else getattr(failure, "code", "")
        )
        for failure in getattr(trace, "failures", ())
    ]
    extraction = dict(metadata.get("extraction") or {})
    extraction_error_code = str(extraction.get("error_code", ""))
    semantic_integrity_error_absent = (
        "semantic_snapshot_integrity_error" not in failure_codes
        and extraction_error_code != "semantic_snapshot_integrity_error"
    )
    task = getattr(trace, "task", None)
    task_type = str(
        task.get("task_type", "")
        if isinstance(task, Mapping)
        else getattr(task, "task_type", "")
    )
    checks = {
        "method_patch_3_2": str(metadata.get("method_patch", "")) == "3.2",
        "look_at_task_selected": task_type == "look_at_obj_in_light",
        "benchmark_success": getattr(trace, "benchmark_success", False) is True,
        "task_contract_success": (
            getattr(trace, "task_contract_success", False) is True
        ),
        "infrastructure_failure_absent": (
            getattr(trace, "infrastructure_failure", True) is False
        ),
        "resource_usage_complete": (
            getattr(trace, "resource_usage_complete", False) is True
        ),
        "semantic_state_snapshots_present": bool(snapshots),
        "semantic_snapshot_contract_valid": sequence_complete,
        "reset_snapshot_present": reset_snapshot_present,
        "every_environment_step_snapshotted": every_step_snapshot_present,
        "semantic_revisions_consistent": revision_consistent,
        "environment_action_timeline_consistent": action_timeline_consistent,
        "normalizer_action_projection_complete": (
            len(normalized_actions) == len(actions)
        ),
        "validator_snapshot_authority_used": str(
            normalized.get("semantic_authority_source", "")
        ) == "validator_snapshot_v3_2",
        "success_evolution_attempted": extraction.get("attempted") is True,
        "semantic_snapshot_integrity_error_absent": (
            semantic_integrity_error_absent
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "environment_action_count": len(actions),
        "semantic_snapshot_count": len(snapshots),
        "semantic_revision_count": len(revision_states),
        "normalizer_semantic_authority_source": str(
            normalized.get("semantic_authority_source", "")
        ),
        "extraction_attempted": extraction.get("attempted") is True,
        "extraction_error_code": extraction_error_code,
        "failure_codes": failure_codes,
    }


def run_look_at_authority_smoke(config_path: str | Path) -> int:
    """Run one real look-at task through the normal online System pipeline."""

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
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    smoke_root = base_output.parent / f"{base_output.name}_look_at_authority_smoke"
    output = smoke_root / f"run_{stamp}_{os.getpid()}"
    if output.exists():
        raise FileExistsError(output)

    experiment = dict(config.get("experiment") or {})
    experiment.update({
        "name": f"look_at_authority_smoke_{stamp}",
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
        empty_bank = system.is_empty_knowledge_bank()
        configuration_checks = {
            "preflight_passed": preflight.get("passed") is True,
            "alfworld_0_4_2": preflight.get("alfworld_version") is True,
            "method_patch_3_2": str(config.get("method_patch", "")) == "3.2",
            "runtime_mode_online": experiment.get("runtime_mode") == "online",
            "fresh_empty_bank": empty_bank,
            "provider_capability_passed": capability.get("passed") is True,
        }
        if not all(configuration_checks.values()):
            result = {
                "passed": False,
                "gate": "real_look_at_authority",
                "output_dir": str(output),
                "configuration_checks": configuration_checks,
                "preflight": preflight,
            }
            atomic_write_json(output / "look_at_authority_smoke_result.json", result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 1

        tasks = system.harness.load_balanced_tasks(
            ["look_at_obj_in_light"], 1,
        )
        if (
            len(tasks) != 1
            or tasks[0].task_type != "look_at_obj_in_light"
            or not task_signature(tasks[0])
        ):
            raise RuntimeError(
                "look-at authority smoke requires exactly one distinct "
                "balanced-loader look_at_obj_in_light task"
            )
        task = tasks[0]
        initial_digest = system.knowledge_digest()
        task_item = TaskManifest.from_task(
            task,
            ordinal=0,
            knowledge_milestone=f"isolated_smoke_selection:{initial_digest}",
            split=str(system.harness.split),
        )
        atomic_write_json(output / "task_manifest.json", {
            "schema_version": 3,
            "task_manifest_hash": hash_task_manifest((task_item,)),
            "tasks": [task_item.to_dict()],
        })

        try:
            trace = system.run_task(task)
            normalized = system.normalizer.build(trace)
        except Exception as exc:
            result = {
                "passed": False,
                "gate": "real_look_at_authority",
                "task_returned": False,
                "output_dir": str(output),
                "selected_task_id": task.task_id,
                "selected_task_signature": task_signature(task),
                "error_type": type(exc).__name__,
                "error_code": str(getattr(exc, "code", "")),
                "error": str(exc),
            }
            atomic_write_json(output / "look_at_authority_smoke_result.json", result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 1

        audit = _look_at_authority_smoke_audit(trace, normalized)
        persisted_traces = list(system.traces.iter_payloads())
        validate_formal_usage(persisted_traces)
        validate_usage_event_persistence(system.usage.events, persisted_traces)
        write_reports(
            [trace], output / "reports", stem="look_at_authority_smoke",
        )
        result = {
            **audit,
            "passed": bool(
                capability.get("passed") is True
                and preflight.get("passed") is True
                and empty_bank
                and audit["passed"]
            ),
            "gate": "real_look_at_authority",
            "task_returned": True,
            "output_dir": str(output),
            "trace_id": trace.trace_id,
            "trace_path": str(system.traces.root / f"{trace.trace_id}.json"),
            "report_dir": str(output / "reports"),
            "selected_task_id": task.task_id,
            "selected_task_signature": task_signature(task),
            "provider_capability_passed": capability.get("passed") is True,
            "preflight_passed": preflight.get("passed") is True,
            "fresh_empty_bank": empty_bank,
        }
        atomic_write_json(output / "look_at_authority_smoke_result.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["passed"] else 1


_R7_LOOK_TASK_TYPE = "look_at_obj_in_light"
_R7_LOOK_TASK_COUNT = 5


def _r7_valid_semantic_alias(raw: Mapping[str, Any]) -> bool:
    event_id = str(raw.get("event_id", ""))
    source_role = str(raw.get("source_argument_role", ""))
    semantic_role = str(raw.get("role", ""))
    event_index = raw.get("event_index")
    return bool(
        raw.get("kind") == "semantic_alias"
        and raw.get("source_kind") == "semantic_snapshot_alias"
        and event_id
        and source_role
        and semantic_role == str(raw.get("predicate_argument_role", ""))
        and semantic_role != source_role
        and raw.get("value") not in (None, "")
        and not isinstance(event_index, bool)
        and isinstance(event_index, int)
        and event_index >= 0
        and str(raw.get("authority_ref", ""))
        == f"semantic_alias:{event_id}:{source_role}:{semantic_role}"
        and str(raw.get("source_authority_ref", ""))
        == f"action_arg:{event_id}:{source_role}"
        and str(raw.get("predicate", ""))
        and str(raw.get("witness_ref", ""))
        and raw.get("effect_domain") in {"world", "evidence"}
    )


def _r7_look_targeted_audit(
    *,
    configuration_checks: Mapping[str, Any],
    train_records: list[Mapping[str, Any]],
    composite_records: list[Mapping[str, Any]],
    eval_records: list[Mapping[str, Any]],
    final_maintenance_pending_count: int | None,
    digests: Mapping[str, str],
) -> dict[str, Any]:
    """Join R7 evidence into one fail-closed train-to-frozen chain audit."""

    train_signatures = [str(item.get("task_signature", "")) for item in train_records]
    eval_signatures = [str(item.get("task_signature", "")) for item in eval_records]
    alias_trace_ids = {
        str(item.get("trace_id", ""))
        for item in train_records
        if any(
            _r7_valid_semantic_alias(alias)
            and str(alias.get("role", "")) == "light"
            for alias in item.get("semantic_aliases", ())
            if isinstance(alias, Mapping)
        )
    }
    e1_trace_ids = {
        str(item.get("trace_id", ""))
        for item in train_records
        if item.get("e1_validated_object_observed") is True
    }
    authority_closed_trace_ids = alias_trace_ids & e1_trace_ids
    chain_composite_refs = {
        str(item.get("composite_ref", ""))
        for item in composite_records
        if item.get("task_contract_covered") is True
        and item.get("goal_covers_object_observed") is True
        and str(item.get("status", "")) == "active"
        and authority_closed_trace_ids
        & {str(value) for value in item.get("source_trace_ids", ())}
    }
    heldout_chain_records = [
        item
        for item in eval_records
        if str(item.get("runtime_source", "")) == "stored_composite"
        and str(item.get("source_composite_ref", "")) in chain_composite_refs
        and item.get("benchmark_success") is True
        and item.get("graph_self_sufficient_success") is True
        and item.get("task_rescue_required") is False
    ]
    digest_values = [
        str(digests.get(name, ""))
        for name in (
            "source_before_freeze",
            "source_after_freeze",
            "frozen_before_eval",
            "frozen_after_eval",
        )
    ]
    checks = {
        "configuration": bool(configuration_checks)
        and all(value is True for value in configuration_checks.values()),
        "five_train_look_tasks_returned": (
            len(train_records) == _R7_LOOK_TASK_COUNT
            and all(
                str(item.get("task_type", "")) == _R7_LOOK_TASK_TYPE
                for item in train_records
            )
        ),
        "five_eval_look_tasks_returned": (
            len(eval_records) == _R7_LOOK_TASK_COUNT
            and all(
                str(item.get("task_type", "")) == _R7_LOOK_TASK_TYPE
                for item in eval_records
            )
        ),
        "task_signatures_unique_and_disjoint": (
            len(train_signatures) == _R7_LOOK_TASK_COUNT
            and len(eval_signatures) == _R7_LOOK_TASK_COUNT
            and all(train_signatures)
            and all(eval_signatures)
            and len(set(train_signatures)) == _R7_LOOK_TASK_COUNT
            and len(set(eval_signatures)) == _R7_LOOK_TASK_COUNT
            and not set(train_signatures) & set(eval_signatures)
        ),
        "semantic_alias_light_authority_observed": bool(alias_trace_ids),
        "e1_validated_object_observed": bool(e1_trace_ids),
        "semantic_alias_to_e1_trace_joined": bool(authority_closed_trace_ids),
        "task_contract_covered_active_composite": bool(chain_composite_refs),
        "heldout_stored_composite_success": bool(heldout_chain_records),
        "final_maintenance_queue_empty": (
            not isinstance(final_maintenance_pending_count, bool)
            and isinstance(final_maintenance_pending_count, int)
            and final_maintenance_pending_count == 0
        ),
        "train_runtime_integrity": (
            len(train_records) == _R7_LOOK_TASK_COUNT
            and all(item.get("infrastructure_failure") is False for item in train_records)
            and all(item.get("resource_usage_complete") is True for item in train_records)
        ),
        "eval_runtime_integrity": (
            len(eval_records) == _R7_LOOK_TASK_COUNT
            and all(item.get("infrastructure_failure") is False for item in eval_records)
            and all(item.get("resource_usage_complete") is True for item in eval_records)
        ),
        "freeze_and_eval_digest_unchanged": (
            all(digest_values)
            and len(set(digest_values)) == 1
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "train_task_count": len(train_records),
        "eval_task_count": len(eval_records),
        "train_successes": sum(
            item.get("benchmark_success") is True for item in train_records
        ),
        "eval_successes": sum(
            item.get("benchmark_success") is True for item in eval_records
        ),
        "semantic_alias_trace_ids": sorted(alias_trace_ids),
        "e1_object_observed_trace_ids": sorted(e1_trace_ids),
        "authority_closed_trace_ids": sorted(authority_closed_trace_ids),
        "active_chain_composite_refs": sorted(chain_composite_refs),
        "heldout_stored_composite_task_ids": sorted(
            str(item.get("task_id", "")) for item in heldout_chain_records
        ),
        "digests": dict(digests),
    }


def _r7_task_record(
    trace: object,
    task: object,
    *,
    system: AtomicSkillGraphSystem,
) -> dict[str, Any]:
    normalized = system.normalizer.build(trace)
    aliases = [
        dict(item)
        for item in dict(normalized.get("boundary_authorities") or {}).get(
            "inputs", ()
        )
        if isinstance(item, Mapping) and item.get("kind") == "semantic_alias"
    ]
    metadata = dict(getattr(trace, "metadata", {}) or {})
    quality = dict(metadata.get("extractor_quality") or {})
    applied = dict(metadata.get("evolution_applied") or {})
    observed_atomic_refs: list[str] = []
    for raw_ref in applied.get("atomic_refs", ()):
        try:
            atomic = system.skills.get_atomic(str(raw_ref))
        except (KeyError, ValueError):
            continue
        if any(
            str(effect.predicate) == "object.observed_with"
            for effect in atomic.effects
        ):
            observed_atomic_refs.append(str(raw_ref))
    return {
        "task_id": str(getattr(task, "task_id", "")),
        "task_signature": task_signature(task),
        "task_type": str(getattr(task, "task_type", "")),
        "trace_id": str(getattr(trace, "trace_id", "")),
        "benchmark_success": getattr(trace, "benchmark_success", False) is True,
        "infrastructure_failure": (
            getattr(trace, "infrastructure_failure", True) is True
        ),
        "resource_usage_complete": (
            getattr(trace, "resource_usage_complete", False) is True
        ),
        "semantic_aliases": aliases,
        "observed_atomic_refs": sorted(observed_atomic_refs),
        "e1_validated_object_observed": bool(
            observed_atomic_refs
            and quality.get("extractor_e1_contract_coverage_passed") is True
            and int(quality.get("extractor_e1_validated_occurrence_count", 0)) > 0
        ),
    }


def _r7_composite_records(system: AtomicSkillGraphSystem) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_ref in system.skills.list_refs("composite"):
        composite = system.skills.get_composite(raw_ref)
        records.append({
            "composite_ref": str(composite.ref),
            "status": str(getattr(composite.status, "value", composite.status)),
            "task_contract_covered": (
                dict(composite.validator_spec).get("task_contract_covered") is True
            ),
            "goal_covers_object_observed": any(
                str(effect.predicate) == "object.observed_with"
                for effect in composite.goal_contract.target_effects
            ),
            "source_trace_ids": sorted(
                str(value)
                for value in dict(composite.metadata).get("source_trace_ids", ())
            ),
        })
    return records


def _r7_eval_record(trace: object, task: object) -> dict[str, Any]:
    runtime_plan = dict(getattr(trace, "runtime_plan", {}) or {})
    return {
        "task_id": str(getattr(task, "task_id", "")),
        "task_signature": task_signature(task),
        "task_type": str(getattr(task, "task_type", "")),
        "trace_id": str(getattr(trace, "trace_id", "")),
        "benchmark_success": getattr(trace, "benchmark_success", False) is True,
        "graph_self_sufficient_success": (
            getattr(trace, "graph_self_sufficient_success", False) is True
        ),
        "task_rescue_required": (
            getattr(trace, "task_rescue_required", True) is True
        ),
        "task_contract_success": (
            getattr(trace, "task_contract_success", False) is True
        ),
        "infrastructure_failure": (
            getattr(trace, "infrastructure_failure", True) is True
        ),
        "resource_usage_complete": (
            getattr(trace, "resource_usage_complete", False) is True
        ),
        "runtime_source": str(runtime_plan.get("source", "")),
        "source_composite_ref": str(
            runtime_plan.get("source_composite_ref", "") or ""
        ),
    }


def _write_r7_task_manifest(
    path: Path,
    tasks: list[object],
    *,
    split: str,
    milestone: str,
) -> tuple[TaskManifest, ...]:
    items = tuple(
        TaskManifest.from_task(
            task,
            ordinal=index,
            knowledge_milestone=milestone,
            split=split,
        )
        for index, task in enumerate(tasks)
    )
    atomic_write_json(path, {
        "schema_version": 3,
        "task_manifest_hash": hash_task_manifest(items),
        "tasks": [item.to_dict() for item in items],
    })
    return items


def _require_r7_look_selection(config: Mapping[str, Any]) -> None:
    selection = dict(dict(config.get("harness") or {}).get("task_selection") or {})
    if (
        selection.get("policy") != "balanced_fixed_manifest"
        or list(selection.get("task_types") or []) != [_R7_LOOK_TASK_TYPE]
        or selection.get("tasks_per_type") != _R7_LOOK_TASK_COUNT
        or selection.get("total_tasks") != _R7_LOOK_TASK_COUNT
        or selection.get("require_exact_count") is not True
    ):
        raise ValueError(
            "R7 look targeted config requires one fixed look-at family and five tasks"
        )


def _require_r7_selected_tasks(tasks: list[object], *, split: str) -> None:
    signatures = [task_signature(task) for task in tasks]
    task_ids = [str(getattr(task, "task_id", "")) for task in tasks]
    if (
        len(tasks) != _R7_LOOK_TASK_COUNT
        or any(
            str(getattr(task, "task_type", "")) != _R7_LOOK_TASK_TYPE
            for task in tasks
        )
        or any(not value for value in signatures)
        or any(not value for value in task_ids)
        or len(set(signatures)) != _R7_LOOK_TASK_COUNT
        or len(set(task_ids)) != _R7_LOOK_TASK_COUNT
    ):
        raise RuntimeError(
            f"R7 targeted {split} selection is not five distinct look-at tasks"
        )


def run_r7_look_targeted(config_path: str | Path) -> int:
    """Train five look tasks, freeze, and audit five held-out look tasks."""

    config_path = _path(config_path)
    config = copy.deepcopy(load_config(config_path))
    base_output = _path(
        dict(config.get("experiment") or {}).get(
            "output_dir", "runs/alfworld_r7_look_targeted"
        )
    )
    base_output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = base_output / f"run_{stamp}_{os.getpid()}"
    output.mkdir(parents=False, exist_ok=False)
    result_path = output / "r7_look_targeted_result.json"

    configuration_checks: dict[str, Any] = {}
    train_records: list[Mapping[str, Any]] = []
    composite_records: list[Mapping[str, Any]] = []
    eval_records: list[Mapping[str, Any]] = []
    pending_count: int | None = None
    digests = {
        "source_before_freeze": "",
        "source_after_freeze": "",
        "frozen_before_eval": "",
        "frozen_after_eval": "",
    }
    error: Exception | None = None
    try:
        validate_deepseek_formal_llm(config)
        _require_r7_look_selection(config)
        capability = ensure_provider_capability(
            config,
            output_dir=base_output,
            config_hash=hash_config(config_path),
            code_hash=hash_code(REPO_ROOT),
            run_if_missing=False,
        )
        configuration_checks["provider_capability_passed"] = (
            capability.get("passed") is True
        )
        train_root = output / "train"
        train_config = copy.deepcopy(config)
        train_config["data_dir"] = str(train_root / "data_v3")
        train_config["trace_data_dir"] = str(train_root)
        train_experiment = dict(train_config.get("experiment") or {})
        train_experiment.update({
            "name": f"r7_look_targeted_train_{stamp}",
            "phase": "smoke",
            "condition": "full",
            "runtime_mode": "online",
            "freeze_skills": False,
            "initialize_v3_bank": "empty",
            "allow_long_term_knowledge_writes": True,
            "output_dir": str(train_root),
        })
        train_config["experiment"] = train_experiment
        frozen_dir = output / "frozen" / "data_v3"

        train_traces: list[object] = []
        with AtomicSkillGraphSystem(train_config, readonly=False) as system:
            preflight = system.preflight(
                require_api_key=True,
                initialize_harness=True,
                require_empty_bank=True,
            )
            empty_bank = system.is_empty_knowledge_bank()
            configuration_checks.update({
                "train_preflight_passed": preflight.get("passed") is True,
                "fresh_empty_bank": empty_bank,
                "method_patch_3_2": str(config.get("method_patch", "")) == "3.2",
                "train_split": str(system.harness.split) == "train",
            })
            if not all(configuration_checks.values()):
                raise RuntimeError("R7 targeted train configuration gate failed")
            tasks = system.harness.load_balanced_tasks(
                [_R7_LOOK_TASK_TYPE], _R7_LOOK_TASK_COUNT,
            )
            _require_r7_selected_tasks(tasks, split="train")
            initial_digest = system.knowledge_digest()
            _write_r7_task_manifest(
                output / "train_task_manifest.json",
                tasks,
                split=str(system.harness.split),
                milestone=f"empty_bank:{initial_digest}",
            )
            for task in tasks:
                trace = system.run_task(task)
                train_traces.append(trace)
                train_records.append(_r7_task_record(
                    trace, task, system=system,
                ))
                if trace.infrastructure_failure:
                    raise RuntimeError(
                        f"infrastructure failure at targeted train task {task.task_id}"
                    )

            maintenance = system.run_maintenance(
                triggering_task_id=tasks[-1].task_id,
                milestone="r7_look_targeted_final_batch",
                finalize_pending=True,
            )
            raw_pending = getattr(maintenance, "pending_count", None)
            pending_count = (
                raw_pending
                if not isinstance(raw_pending, bool) and isinstance(raw_pending, int)
                else None
            )
            if pending_count != 0:
                raise RuntimeError(
                    "R7 targeted final maintenance left unresolved proposals"
                )
            composite_records = _r7_composite_records(system)
            digests["source_before_freeze"] = system.knowledge_digest()
            system.freeze(frozen_dir, provenance={
                "gate": "r7_look_targeted",
                "train_task_manifest": str(output / "train_task_manifest.json"),
                "source_final_knowledge_digest": digests["source_before_freeze"],
            })
            digests["source_after_freeze"] = system.knowledge_digest()
            persisted_train = list(system.traces.iter_payloads())
            validate_formal_usage(persisted_train)
            validate_usage_event_persistence(system.usage.events, persisted_train)
            write_reports(
                train_traces,
                train_root / "reports",
                stem="r7_look_targeted_train5",
                title="AtomicSkillGraph v3.2 R7 Look-at Targeted Train-5",
            )

        eval_root = output / "eval"
        eval_config = copy.deepcopy(config)
        eval_config["data_dir"] = str(frozen_dir)
        eval_config["trace_data_dir"] = str(eval_root)
        eval_config["cold_start"] = {
            **dict(eval_config.get("cold_start") or {}),
            "enabled": False,
        }
        eval_config["extraction"] = {
            **dict(eval_config.get("extraction") or {}),
            "extract_full_dynamic_success": False,
            "extract_task_rescue_success": False,
            "extract_novel_seeded_success": False,
        }
        eval_config["harness"] = {
            **dict(eval_config.get("harness") or {}),
            "split": "eval_out_of_distribution",
        }
        eval_experiment = dict(eval_config.get("experiment") or {})
        eval_experiment.pop("initialize_v3_bank", None)
        eval_experiment.update({
            "name": f"r7_look_targeted_eval_{stamp}",
            "phase": "smoke",
            "condition": "full",
            "runtime_mode": "frozen",
            "freeze_skills": True,
            "allow_long_term_knowledge_writes": False,
            "output_dir": str(eval_root),
        })
        eval_config["experiment"] = eval_experiment

        eval_traces: list[object] = []
        with AtomicSkillGraphSystem(eval_config) as system:
            configuration_checks["frozen_system_readonly"] = bool(
                system.readonly and system.database.readonly
            )
            preflight = system.preflight(
                require_api_key=True,
                initialize_harness=True,
            )
            configuration_checks.update({
                "eval_preflight_passed": preflight.get("passed") is True,
                "eval_split_valid_unseen": (
                    str(system.harness.split) == "eval_out_of_distribution"
                ),
            })
            digests["frozen_before_eval"] = system.knowledge_digest()
            freeze_manifest_path = frozen_dir / "freeze_manifest.json"
            freeze_manifest = json.loads(
                freeze_manifest_path.read_text(encoding="utf-8")
            )
            configuration_checks["freeze_manifest_digest_match"] = (
                str(freeze_manifest.get("knowledge_digest", ""))
                == digests["frozen_before_eval"]
            )
            if not all(configuration_checks.values()):
                raise RuntimeError("R7 targeted frozen configuration gate failed")

            tasks = system.harness.load_balanced_tasks(
                [_R7_LOOK_TASK_TYPE], _R7_LOOK_TASK_COUNT,
            )
            _require_r7_selected_tasks(tasks, split="valid_unseen")
            train_signatures = {
                str(item.get("task_signature", "")) for item in train_records
            }
            if train_signatures & {task_signature(task) for task in tasks}:
                raise RuntimeError("R7 targeted train/eval signatures overlap")
            _write_r7_task_manifest(
                output / "eval_task_manifest.json",
                tasks,
                split=str(system.harness.split),
                milestone=f"frozen:{digests['frozen_before_eval']}",
            )
            for task in tasks:
                trace = system.run_task(task)
                eval_traces.append(trace)
                eval_records.append(_r7_eval_record(trace, task))
                if trace.infrastructure_failure:
                    raise RuntimeError(
                        f"infrastructure failure at targeted eval task {task.task_id}"
                    )
                if system.knowledge_digest() != digests["frozen_before_eval"]:
                    raise RuntimeError("R7 targeted frozen task changed knowledge")
            digests["frozen_after_eval"] = system.knowledge_digest()
            persisted_eval = list(system.traces.iter_payloads())
            validate_formal_usage(persisted_eval)
            validate_usage_event_persistence(system.usage.events, persisted_eval)
            write_reports(
                eval_traces,
                eval_root / "reports",
                stem="r7_look_targeted_eval5",
                title="AtomicSkillGraph v3.2 R7 Look-at Targeted Held-out-5",
            )
    except Exception as exc:
        error = exc

    audit = _r7_look_targeted_audit(
        configuration_checks=configuration_checks,
        train_records=train_records,
        composite_records=composite_records,
        eval_records=eval_records,
        final_maintenance_pending_count=pending_count,
        digests=digests,
    )
    result = {
        **audit,
        "gate": "r7_look_targeted_full_chain",
        "output_dir": str(output),
        "train_task_manifest": str(output / "train_task_manifest.json"),
        "eval_task_manifest": str(output / "eval_task_manifest.json"),
        "frozen_snapshot": str(output / "frozen" / "data_v3"),
        "configuration_checks": configuration_checks,
        "composite_records": composite_records,
    }
    if error is not None:
        result.update({
            "passed": False,
            "error_type": type(error).__name__,
            "error_code": str(getattr(error, "code", "")),
            "error": str(error),
        })
    atomic_write_json(result_path, result)
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
    config["trace_data_dir"] = str(output)
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
    modes.add_argument("--look-at-authority", action="store_true")
    modes.add_argument("--r7-look-targeted", action="store_true")
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
    if args.look_at_authority:
        return run_look_at_authority_smoke(args.config)
    if args.r7_look_targeted:
        return run_r7_look_targeted(args.config)
    return run_real_alfworld(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
