"""Paid, fail-closed DeepSeek capability probes for formal v3 experiments."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..core.errors import AtomicSkillGraphError, FailureLayer
from ..core.refs import content_hash
from ..core.serialization import atomic_write_json, read_json, to_primitive
from .protocol import NativeToolSpec
from .provider import OpenAICompatibleConfig, OpenAICompatibleProvider
from .session import PROTOCOL_REPAIR_LIMIT, ReplayAgentSession
from .structured_submission import StructuredSubmissionClient
from .usage import AgentBudget, UsageBucket, UsageLedger


_PROBE_SCHEMA = {
    "type": "object",
    "required": ["ok"],
    "additionalProperties": False,
    "properties": {"ok": {"type": "boolean"}},
}
_FORBIDDEN_PAYLOAD_FIELDS = frozenset({
    "response_format",
    "max_completion_tokens",
    "tool_choice",
    "parallel_tool_calls",
    "temperature",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
})
_PROBE_B_ACCEPTED_TURNS = 2
_PROBE_B_MAX_HTTP_REQUESTS = _PROBE_B_ACCEPTED_TURNS + PROTOCOL_REPAIR_LIMIT


class ProviderCapabilityError(AtomicSkillGraphError):
    def __init__(self, message: str) -> None:
        super().__init__(
            "provider_capability_mismatch",
            message,
            layer=FailureLayer.INFRASTRUCTURE,
        )


def provider_config_for_stage(
    config: Mapping[str, Any], stage: str,
) -> OpenAICompatibleConfig:
    llm = dict(config.get("llm") or {})
    stage_config = {**llm, **dict(llm.get(stage) or {})}
    protocol = dict(llm.get("protocol") or {})
    return OpenAICompatibleConfig(
        dialect=str(llm.get("dialect", "")),
        base_url=str(llm.get("base_url", "")),
        model=str(llm.get("model", "")),
        api_key_env=str(llm.get("api_key_env", "MODEL_API_KEY")),
        max_completion_tokens=int(stage_config.get("max_completion_tokens", 0)),
        thinking_type=str(protocol.get("thinking_type", "")),
        reasoning_effort=str(stage_config.get("reasoning_effort", "")),
        connect_timeout_seconds=float(llm.get("connect_timeout_seconds", 15)),
        request_timeout_seconds=float(stage_config.get("request_timeout_seconds", 120)),
        max_retries=int(llm.get("max_retries", 4)),
        retry_backoff_seconds=float(llm.get("retry_backoff_seconds", 2)),
        max_retry_after_seconds=float(llm.get("max_retry_after_seconds", 30)),
    )


def capability_paths(output_dir: str | Path) -> tuple[Path, Path]:
    root = Path(output_dir).resolve()
    return (
        root / "provider_capability_manifest.json",
        root / "provider_probe_trace.json",
    )


def provider_fingerprint(config: Mapping[str, Any]) -> str:
    llm = dict(config.get("llm") or {})
    protocol = dict(llm.get("protocol") or {})
    return content_hash({
        "provider": llm.get("provider"),
        "dialect": llm.get("dialect"),
        "base_url": str(llm.get("base_url", "")).rstrip("/"),
        "model": llm.get("model"),
        "endpoint_path": protocol.get("endpoint_path"),
        "thinking_type": protocol.get("thinking_type"),
        "token_limit_field": protocol.get("token_limit_field"),
    })


def run_provider_capability_probe(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path,
    config_hash: str,
    code_hash: str,
    provider_factory: Any = OpenAICompatibleProvider,
) -> dict[str, Any]:
    """Run A/B/C against the actual endpoint and always persist a sanitized audit."""

    manifest_path, trace_path = capability_paths(output_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    planner_provider = provider_factory(provider_config_for_stage(config, "planner"))
    extractor_provider = provider_factory(provider_config_for_stage(config, "extractor"))
    providers = [planner_provider, extractor_provider]
    probe_results: dict[str, Any] = {}
    primary_error: Exception | None = None
    usage_events: list[dict[str, Any]] = []
    try:
        result_a, events_a = _probe_structured_submission(planner_provider)
        probe_results["structured_submission"] = result_a
        usage_events.extend(events_a)

        result_b, events_b = _probe_reasoning_replay(planner_provider)
        probe_results["reasoning_replay"] = result_b
        usage_events.extend(events_b)

        try:
            result_c, events_c = _probe_extractor_limit(extractor_provider)
        except Exception as exc:
            if int(getattr(exc, "http_status", 0) or 0) == 400:
                raise ProviderCapabilityError(
                    "extractor_max_tokens_unsupported: endpoint rejected max_tokens=131072"
                ) from exc
            raise
        probe_results["extractor_max_tokens"] = result_c
        usage_events.extend(events_c)
    except Exception as exc:  # persist the negative capability evidence first
        primary_error = exc

    request_records = [
        to_primitive(record)
        for provider in providers
        for record in getattr(provider, "request_records", ())
    ]
    payload_fields = sorted({
        str(field)
        for record in request_records
        for field in record.get("payload_field_names", ())
    })
    forbidden_present = sorted(_FORBIDDEN_PAYLOAD_FIELDS & set(payload_fields))
    http_success = bool(request_records) and all(
        record.get("http_status") == 200
        and record.get("outcome") == "success"
        for record in request_records
    )
    all_reported = bool(request_records) and all(
        record.get("usage_status") == "reported" for record in request_records
    )
    replay_passed = bool(
        dict(probe_results.get("reasoning_replay") or {}).get("passed")
    )
    probe_request_shape_valid = _probe_request_shape_is_valid(
        probe_results, len(request_records),
    )
    passed = bool(
        primary_error is None
        and len(probe_results) == 3
        and all(bool(item.get("passed")) for item in probe_results.values())
        and not forbidden_present
        and "max_tokens" in payload_fields
        and "thinking" in payload_fields
        and "reasoning_effort" in payload_fields
        and all_reported
        and http_success
        and replay_passed
        and probe_request_shape_valid
    )
    sanitized_error = _sanitize_error(primary_error, config) if primary_error else ""
    trace_payload = {
        "schema_version": 3,
        "trace_kind": "provider_capability_probe",
        "provider_fingerprint": provider_fingerprint(config),
        "config_hash": str(config_hash),
        "code_hash": str(code_hash),
        "passed": passed,
        "probes": probe_results,
        "provider_requests": request_records,
        "provider_request_count": len(request_records),
        "usage_events": usage_events,
        "payload_field_names": payload_fields,
        "forbidden_payload_fields_present": forbidden_present,
        "resource_usage_complete": all_reported,
        "all_http_requests_succeeded": http_success,
        "reasoning_content_replay_check": replay_passed,
        "probe_request_shape_valid": probe_request_shape_valid,
        "error_type": type(primary_error).__name__ if primary_error else "",
        "error_code": str(getattr(primary_error, "code", "")) if primary_error else "",
        "sanitized_error": sanitized_error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    # Provider records contain only hashes/counts for reasoning.  This final
    # assertion prevents future adapter changes from leaking the private text.
    encoded_trace = json.dumps(trace_payload, ensure_ascii=False, sort_keys=True)
    for turn_result in probe_results.values():
        if "reasoning_content" in turn_result:
            raise RuntimeError("probe result attempted to persist reasoning_content")
    api_key = os.environ.get(
        str(dict(config.get("llm") or {}).get("api_key_env", "MODEL_API_KEY")), ""
    )
    if api_key and api_key in encoded_trace:
        raise RuntimeError("provider probe audit attempted to persist the API key")
    atomic_write_json(trace_path, trace_payload)
    trace_hash = _sha256_file(trace_path)
    llm = dict(config.get("llm") or {})
    manifest = {
        "schema_version": 3,
        "provider": llm.get("provider"),
        "dialect": llm.get("dialect"),
        "base_url": str(llm.get("base_url", "")).rstrip("/"),
        "model": llm.get("model"),
        "provider_fingerprint": provider_fingerprint(config),
        "config_hash": str(config_hash),
        "code_hash": str(code_hash),
        "payload_field_names": payload_fields,
        "forbidden_payload_fields_present": forbidden_present,
        "passed": passed,
        "probe_pass": {name: bool(item.get("passed")) for name, item in probe_results.items()},
        "http_statuses": [record.get("http_status") for record in request_records],
        "http_outcomes": [record.get("outcome") for record in request_records],
        "request_ids": [
            record.get("provider_request_id") or record.get("request_id")
            for record in request_records
        ],
        "provider_request_count": len(request_records),
        "usage": [record.get("usage") for record in request_records],
        "reasoning_content_replay_check": replay_passed,
        "probe_request_shape_valid": probe_request_shape_valid,
        "resource_usage_complete": all_reported,
        "all_http_requests_succeeded": http_success,
        "probe_trace_sha256": trace_hash,
        "timestamp": trace_payload["timestamp"],
    }
    atomic_write_json(manifest_path, manifest)
    if primary_error is not None:
        raise ProviderCapabilityError(
            f"DeepSeek provider probe failed: {sanitized_error or type(primary_error).__name__}"
        ) from primary_error
    if not passed:
        raise ProviderCapabilityError(
            "DeepSeek provider probe failed its payload, replay, or usage invariants"
        )
    return manifest


def ensure_provider_capability(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path,
    config_hash: str,
    code_hash: str,
    run_if_missing: bool,
) -> dict[str, Any]:
    """Validate an exact matching probe, optionally running it for a fresh run."""

    manifest_path, trace_path = capability_paths(output_dir)
    if manifest_path.is_file() and trace_path.is_file():
        manifest = read_json(manifest_path)
        expected = {
            "schema_version": 3,
            "provider_fingerprint": provider_fingerprint(config),
            "config_hash": str(config_hash),
            "code_hash": str(code_hash),
            "passed": True,
            "reasoning_content_replay_check": True,
            "probe_request_shape_valid": True,
            "resource_usage_complete": True,
            "all_http_requests_succeeded": True,
        }
        mismatches = {
            name: {"expected": wanted, "actual": manifest.get(name)}
            for name, wanted in expected.items()
            if manifest.get(name) != wanted
        }
        required_fields = {"max_tokens", "thinking", "reasoning_effort", "tools"}
        probe_pass = manifest.get("probe_pass")
        statuses = manifest.get("http_statuses")
        outcomes = manifest.get("http_outcomes")
        request_ids = manifest.get("request_ids")
        usage = manifest.get("usage")
        provider_request_count = manifest.get("provider_request_count")
        trace = read_json(trace_path)
        trace_probes = trace.get("probes") if isinstance(trace, dict) else None
        request_shape_valid = bool(
            isinstance(provider_request_count, int)
            and not isinstance(provider_request_count, bool)
            and _probe_request_shape_is_valid(trace_probes, provider_request_count)
        )
        artifact_valid = bool(
            not mismatches
            and manifest.get("probe_trace_sha256") == _sha256_file(trace_path)
            and required_fields.issubset(set(manifest.get("payload_field_names") or ()))
            and not (_FORBIDDEN_PAYLOAD_FIELDS & set(manifest.get("payload_field_names") or ()))
            and manifest.get("forbidden_payload_fields_present") == []
            and isinstance(probe_pass, dict)
            and set(probe_pass) == {
                "structured_submission", "reasoning_replay", "extractor_max_tokens",
            }
            and all(value is True for value in probe_pass.values())
            and request_shape_valid
            and isinstance(statuses, list)
            and len(statuses) == provider_request_count
            and all(status == 200 for status in statuses)
            and isinstance(outcomes, list)
            and len(outcomes) == provider_request_count
            and all(outcome == "success" for outcome in outcomes)
            and isinstance(request_ids, list)
            and len(request_ids) == provider_request_count
            and all(isinstance(request_id, str) and request_id for request_id in request_ids)
            and isinstance(usage, list)
            and len(usage) == provider_request_count
            and all(
                isinstance(item, dict)
                and all(name in item for name in (
                    "prompt_tokens", "completion_tokens", "total_tokens",
                ))
                for item in usage
            )
            and isinstance(trace, dict)
            and trace.get("passed") is True
            and trace.get("provider_fingerprint") == expected["provider_fingerprint"]
            and trace.get("config_hash") == expected["config_hash"]
            and trace.get("code_hash") == expected["code_hash"]
            and trace.get("resource_usage_complete") is True
            and trace.get("all_http_requests_succeeded") is True
            and trace.get("reasoning_content_replay_check") is True
            and trace.get("probe_request_shape_valid") is True
            and trace.get("provider_request_count") == provider_request_count
            and len(trace.get("provider_requests") or ()) == provider_request_count
            and isinstance(trace_probes, dict)
            and set(trace_probes) == set(probe_pass)
            and trace_probes["reasoning_replay"].get(
                "reasoning_content_replayed_verbatim"
            ) is True
            and trace_probes["extractor_max_tokens"].get("max_tokens") == 131072
            and not _contains_exact_key(trace, "reasoning_content")
        )
        if artifact_valid:
            return manifest
        if not run_if_missing:
            raise ProviderCapabilityError(
                "stored provider capability manifest does not match current config/code"
            )
    elif not run_if_missing:
        raise ProviderCapabilityError("matching provider capability manifest is missing")
    return run_provider_capability_probe(
        config,
        output_dir=output_dir,
        config_hash=config_hash,
        code_hash=code_hash,
    )


def _session(
    provider: Any,
    *,
    bucket: UsageBucket,
    max_turns: int,
    max_tokens: int,
) -> tuple[ReplayAgentSession, UsageLedger]:
    ledger = UsageLedger()
    session = ReplayAgentSession(
        provider,
        system_prompt=(
            "You are a provider capability probe. Follow the offered native function schema "
            "exactly and do not return prose instead of a ToolCall."
        ),
        usage_ledger=ledger,
        usage_bucket=bucket,
        budget=AgentBudget(max_turns, max_tokens, "provider_capability_mismatch"),
    )
    return session, ledger


def _probe_structured_submission(provider: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    request_start = int(getattr(provider, "request_record_count", 0))
    session, ledger = _session(
        provider, bucket=UsageBucket.PLANNER_P1, max_turns=3, max_tokens=120000,
    )
    submission = StructuredSubmissionClient().request(
        session,
        prompt="Call submit_probe exactly once with {\"ok\": true}.",
        tool_name="submit_probe",
        description="Submit the structured provider capability result.",
        schema=_PROBE_SCHEMA,
    )
    if submission.value != {"ok": True} or not submission.turn.reasoning_content:
        raise ProviderCapabilityError(
            "Probe A requires ok=true and non-empty reasoning_content"
        )
    request_count = int(getattr(provider, "request_record_count", 0)) - request_start
    turn_count = int(session.snapshot().get("turn_count", 0))
    if request_count != 1 or turn_count != 1 or len(ledger.events) != 1:
        raise ProviderCapabilityError(
            "Probe A must complete as exactly one structured provider turn"
        )
    return {
        "passed": True,
        "request_count": request_count,
        "turn_count": turn_count,
        "metered_turn_count": len(ledger.events),
        "tool_name": submission.tool_name,
        "tool_call_count": len(submission.turn.tool_calls),
        "reasoning_content_present": True,
        "reasoning_content_chars": len(submission.turn.reasoning_content),
        "reasoning_content_sha256": hashlib.sha256(
            submission.turn.reasoning_content.encode("utf-8")
        ).hexdigest(),
    }, [event.to_dict() for event in ledger.events]


def _probe_reasoning_replay(provider: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    request_start = int(getattr(provider, "request_record_count", 0))
    session, ledger = _session(
        provider, bucket=UsageBucket.PLANNER_P1, max_turns=5, max_tokens=160000,
    )
    first_tool = NativeToolSpec(
        "probe_step_one", "Complete step one of the replay probe.", _PROBE_SCHEMA,
    )
    first = session.next_turn(
        "Call probe_step_one exactly once with ok=true.", tools=[first_tool],
    )
    if len(first.tool_calls) != 1 or first.tool_calls[0].arguments != {"ok": True}:
        raise ProviderCapabilityError("Probe B first turn did not call probe_step_one correctly")
    first_hash = hashlib.sha256(first.reasoning_content.encode("utf-8")).hexdigest()
    second_tool = NativeToolSpec(
        "probe_step_two", "Complete step two of the replay probe.", _PROBE_SCHEMA,
    )
    second = session.submit_tool_result(
        first.tool_calls[0].call_id,
        {"accepted": True, "next": "probe_step_two"},
        tools=[second_tool],
    )
    if len(second.tool_calls) != 1 or second.tool_calls[0].arguments != {"ok": True}:
        raise ProviderCapabilityError("Probe B second turn did not call probe_step_two correctly")
    session.acknowledge_tool_result(
        second.tool_calls[0].call_id, {"accepted": True, "complete": True},
    )
    snapshot = session.snapshot()
    assistant_snapshots = [
        message
        for message in snapshot.get("messages", ())
        if message.get("role") == "assistant"
    ]
    first_assistant_snapshots = [
        message
        for message in assistant_snapshots
        if _snapshot_has_tool_call_id(message, first.tool_calls[0].call_id)
    ]
    exact_hash_replayed = bool(
        len(first_assistant_snapshots) == 1
        and first_assistant_snapshots[0].get("reasoning_content_sha256") == first_hash
        and first_assistant_snapshots[0].get("reasoning_content_chars")
        == len(first.reasoning_content)
    )
    if not first.reasoning_content or not second.reasoning_content or not exact_hash_replayed:
        raise ProviderCapabilityError("Probe B did not preserve reasoning_content verbatim")
    request_count = int(getattr(provider, "request_record_count", 0)) - request_start
    provider_turn_count = int(snapshot.get("turn_count", 0))
    protocol_failures = list(snapshot.get("protocol_failures") or ())
    protocol_repair_count = len(protocol_failures)
    request_audit_valid = bool(
        _PROBE_B_ACCEPTED_TURNS <= request_count <= _PROBE_B_MAX_HTTP_REQUESTS
        and provider_turn_count == request_count
        and len(ledger.events) == request_count
        and protocol_repair_count == request_count - _PROBE_B_ACCEPTED_TURNS
        and all(item.get("repair_attempted") is True for item in protocol_failures)
        and snapshot.get("terminal_protocol_failure") is None
    )
    if not request_audit_valid:
        raise ProviderCapabilityError(
            "Probe B request audit mismatch: "
            f"accepted={_PROBE_B_ACCEPTED_TURNS}, http={request_count}, "
            f"metered={len(ledger.events)}, repairs={protocol_repair_count}"
        )
    return {
        "passed": True,
        "request_count": request_count,
        "turn_count": provider_turn_count,
        "metered_turn_count": len(ledger.events),
        "accepted_turn_count": _PROBE_B_ACCEPTED_TURNS,
        "protocol_repair_count": protocol_repair_count,
        "protocol_failure_codes": [
            str(item.get("code", "")) for item in protocol_failures
        ],
        "reasoning_content_replayed_verbatim": exact_hash_replayed,
        "first_reasoning_content_sha256": first_hash,
        "second_reasoning_content_present": True,
    }, [event.to_dict() for event in ledger.events]


def _probe_extractor_limit(provider: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    request_start = int(getattr(provider, "request_record_count", 0))
    session, ledger = _session(
        provider, bucket=UsageBucket.EXTRACTOR_E1, max_turns=3, max_tokens=262144,
    )
    submission = StructuredSubmissionClient().request(
        session,
        prompt="Call submit_extractor_probe exactly once with {\"ok\": true}.",
        tool_name="submit_extractor_probe",
        description="Confirm the configured extractor max_tokens field is accepted.",
        schema=_PROBE_SCHEMA,
    )
    if submission.value != {"ok": True} or not submission.turn.reasoning_content:
        raise ProviderCapabilityError("Probe C failed its structured response invariant")
    request_count = int(getattr(provider, "request_record_count", 0)) - request_start
    turn_count = int(session.snapshot().get("turn_count", 0))
    if request_count != 1 or turn_count != 1 or len(ledger.events) != 1:
        raise ProviderCapabilityError("Probe C must use exactly one provider turn")
    return {
        "passed": True,
        "request_count": request_count,
        "turn_count": turn_count,
        "metered_turn_count": len(ledger.events),
        "max_tokens": int(provider.config.max_completion_tokens),
        "reasoning_content_present": True,
    }, [event.to_dict() for event in ledger.events]


def _sanitize_error(error: Exception | None, config: Mapping[str, Any]) -> str:
    if error is None:
        return ""
    message = str(error)
    env_name = str(dict(config.get("llm") or {}).get("api_key_env", "MODEL_API_KEY"))
    secret = os.environ.get(env_name, "")
    if secret:
        message = message.replace(secret, "[REDACTED]")
    return message[:4000]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contains_exact_key(value: Any, forbidden_key: str) -> bool:
    if isinstance(value, Mapping):
        return forbidden_key in value or any(
            _contains_exact_key(item, forbidden_key) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_exact_key(item, forbidden_key) for item in value)
    return False


def _snapshot_has_tool_call_id(message: Mapping[str, Any], call_id: str) -> bool:
    for item in message.get("tool_calls") or ():
        if not isinstance(item, Mapping):
            continue
        if (item.get("id") or item.get("call_id")) == call_id:
            return True
    return False


def _probe_request_shape_is_valid(
    probes: Any,
    provider_request_count: int,
) -> bool:
    """Separate two accepted replay turns from their metered repair calls."""

    if not isinstance(probes, Mapping) or set(probes) != {
        "structured_submission", "reasoning_replay", "extractor_max_tokens",
    }:
        return False
    if not isinstance(provider_request_count, int) or isinstance(
        provider_request_count, bool,
    ):
        return False
    structured = probes.get("structured_submission")
    replay = probes.get("reasoning_replay")
    extractor = probes.get("extractor_max_tokens")
    if not all(isinstance(item, Mapping) for item in (structured, replay, extractor)):
        return False
    replay_requests = replay.get("request_count")
    if not isinstance(replay_requests, int) or isinstance(replay_requests, bool):
        return False
    replay_repairs = replay.get("protocol_repair_count")
    failure_codes = replay.get("protocol_failure_codes")
    return bool(
        structured.get("request_count") == 1
        and structured.get("turn_count") == 1
        and structured.get("metered_turn_count") == 1
        and _PROBE_B_ACCEPTED_TURNS
        <= replay_requests
        <= _PROBE_B_MAX_HTTP_REQUESTS
        and replay.get("accepted_turn_count") == _PROBE_B_ACCEPTED_TURNS
        and replay.get("turn_count") == replay_requests
        and replay.get("metered_turn_count") == replay_requests
        and replay_repairs == replay_requests - _PROBE_B_ACCEPTED_TURNS
        and isinstance(failure_codes, list)
        and len(failure_codes) == replay_repairs
        and all(isinstance(code, str) and code for code in failure_codes)
        and extractor.get("request_count") == 1
        and extractor.get("turn_count") == 1
        and extractor.get("metered_turn_count") == 1
        and provider_request_count == 1 + replay_requests + 1
    )


__all__ = [
    "ProviderCapabilityError",
    "capability_paths",
    "ensure_provider_capability",
    "provider_config_for_stage",
    "provider_fingerprint",
    "run_provider_capability_probe",
]
