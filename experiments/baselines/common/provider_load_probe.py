"""Concurrent real-provider preflight for one baseline method campaign.

The probe deliberately exercises the same SkillOpt OpenAI-compatible backend,
application retry observer, and campaign-wide file-lock gate used by a formal
run.  It writes one durable provider event per logical call and a small,
sanitized report; prompts and model responses are never persisted.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from experiments.baselines.b3_skillopt.provider_observer import (
    ProviderCallExhausted,
    install_provider_observer,
    uninstall_provider_observer,
)
from experiments.baselines.common.provider_gate import CampaignProviderGate


DEFAULT_CONCURRENCY = 16
DEFAULT_REQUESTS = 32
DEFAULT_MAX_COMPLETION_TOKENS = 256
DEFAULT_RETRY_DELAYS_SECONDS = (2.0, 5.0, 10.0, 20.0)
_REPORT_NAME = "provider_load_probe.json"
_EVENTS_NAME = "provider_calls.jsonl"
_TRANSIENT_FAILURE_CODES = frozenset({
    "connection",
    "empty_choices",
    "empty_message",
    "invalid_usage",
    "rate_limit",
    "server_error",
    "timeout",
})
_REQUIRED_EVENT_FIELDS = frozenset({
    "call_id",
    "logical_call_id",
    "run_id",
    "run_seed",
    "role",
    "stage",
    "status",
    "requested_retry_limit",
    "application_attempts",
    "sdk_boundary_attempts",
    "failure_code_counts",
    "last_failure_code",
    "provider_request_ids",
    "provider_slot_ids",
    "provider_queue_wait_ms",
    "provider_service_latency_ms",
    "logical_call_latency_ms",
    "retry_backoff_ms",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
})


def _validate_probe_shape(
    *,
    concurrency: int,
    requests: int,
    max_completion_tokens: int,
    application_retry_limit: int,
    retry_delays_seconds: Iterable[float],
) -> tuple[float, ...]:
    concurrency = int(concurrency)
    requests = int(requests)
    max_completion_tokens = int(max_completion_tokens)
    application_retry_limit = int(application_retry_limit)
    delays = tuple(float(value) for value in retry_delays_seconds)
    if concurrency <= 0:
        raise ValueError("provider probe concurrency must be positive")
    if requests <= 0:
        raise ValueError("provider probe requests must be positive")
    if max_completion_tokens <= 0:
        raise ValueError("provider probe max_completion_tokens must be positive")
    if application_retry_limit <= 0:
        raise ValueError("provider probe application_retry_limit must be positive")
    if len(delays) != application_retry_limit - 1:
        raise ValueError("provider probe requires one delay per application retry")
    if any(value < 0 for value in delays):
        raise ValueError("provider probe retry delays must be non-negative")
    return delays


def _event_is_complete(
    event: dict[str, Any],
    *,
    run_id: str,
    run_seed: int,
    model: str,
    reasoning_effort: str,
) -> bool:
    if not _REQUIRED_EVENT_FIELDS.issubset(event):
        return False
    try:
        attempts = int(event["application_attempts"])
        sdk_attempts = int(event["sdk_boundary_attempts"])
        failure_counts = dict(event["failure_code_counts"])
        if any(
            not str(code).strip()
            or isinstance(count, bool)
            or int(count) <= 0
            for code, count in failure_counts.items()
        ):
            return False
        failure_total = sum(int(value) for value in failure_counts.values())
        prompt_tokens = int(event["prompt_tokens"])
        completion_tokens = int(event["completion_tokens"])
        total_tokens = int(event.get("total_tokens", -1))
        slots = list(event["provider_slot_ids"])
        request_ids = event["provider_request_ids"]
        status = str(event.get("status", ""))
        last_failure_code = event.get("last_failure_code")
        recovered = event.get("recovered")
        shared_complete = bool(
            int(event.get("schema_version", 0)) >= 2
            and event.get("method") == "b3_skillopt"
            and event.get("phase") == "provider_load_probe"
            and event.get("run_id") == str(run_id)
            and int(event.get("run_seed", -1)) == int(run_seed)
            and event.get("model") == str(model)
            and event.get("reasoning_effort") == str(reasoning_effort)
            and event.get("role") == "target"
            and event.get("stage") == "provider_load_probe"
            and status in {"succeeded", "failed"}
            and str(event["call_id"]).strip()
            and str(event["logical_call_id"]).strip()
            and attempts > 0
            and sdk_attempts == attempts
            and len(slots) == sdk_attempts
            and int(event["provider_queue_wait_ms"]) >= 0
            and int(event["provider_service_latency_ms"]) >= 0
            and int(event["logical_call_latency_ms"]) >= 0
            and int(event["retry_backoff_ms"]) >= 0
            and isinstance(request_ids, list)
            and all(str(value).strip() for value in request_ids)
        )
        if not shared_complete:
            return False
        if status == "succeeded":
            return bool(
                failure_total == attempts - 1
                and prompt_tokens >= 0
                and completion_tokens > 0
                and total_tokens == prompt_tokens + completion_tokens
                and recovered is bool(failure_counts)
                and (
                    last_failure_code is None
                    if not failure_counts
                    else str(last_failure_code) in failure_counts
                )
            )
        return bool(
            failure_total == attempts
            and prompt_tokens == 0
            and completion_tokens == 0
            and total_tokens == 0
            and recovered is False
            and str(last_failure_code) in failure_counts
        )
    except (TypeError, ValueError, KeyError):
        return False


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_provider_load_probe(
    *,
    output_dir: str | Path,
    campaign_gate: CampaignProviderGate,
    model: str,
    reasoning_effort: str,
    run_id: str,
    run_seed: int,
    concurrency: int = DEFAULT_CONCURRENCY,
    requests: int = DEFAULT_REQUESTS,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    application_retry_limit: int = 5,
    retry_delays_seconds: Iterable[float] = DEFAULT_RETRY_DELAYS_SECONDS,
    deterministic_jitter_ratio: float = 0.10,
    expected_sdk_max_retries: int = 0,
) -> dict[str, Any]:
    """Run the paid load probe against the already-configured SkillOpt backend.

    The caller owns provider configuration so the exact same model/base URL/API
    key setup can be shared with the formal worker.  A negative provider result
    is returned as ``passed=false`` (and durably reported), not confused with an
    ALFWorld task failure.
    """

    delays = _validate_probe_shape(
        concurrency=concurrency,
        requests=requests,
        max_completion_tokens=max_completion_tokens,
        application_retry_limit=application_retry_limit,
        retry_delays_seconds=retry_delays_seconds,
    )
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / _REPORT_NAME
    events_path = root / _EVENTS_NAME
    if report_path.exists():
        raise FileExistsError(report_path)
    if events_path.exists():
        raise FileExistsError(events_path)

    # Import only when the probe runs.  This keeps manifest/config tooling usable
    # in the controller environment where the pinned SkillOpt wheel is absent.
    import skillopt.model.openai_compatible_backend as backend

    observer = install_provider_observer(
        output_path=events_path,
        method="b3_skillopt",
        phase="provider_load_probe",
        model=str(model),
        reasoning_effort=str(reasoning_effort),
        run_id=str(run_id),
        run_seed=int(run_seed),
        application_retry_limit=int(application_retry_limit),
        retry_delays_seconds=delays,
        deterministic_jitter_ratio=float(deterministic_jitter_ratio),
        expected_sdk_max_retries=int(expected_sdk_max_retries),
        campaign_gate=campaign_gate,
    )
    started = time.perf_counter()
    outcomes: list[dict[str, Any]] = []
    observer_error_type = ""

    def invoke(index: int) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": "This is a provider load probe. Reply briefly.",
            },
            {
                "role": "user",
                "content": f"Reply with OK for probe request {index}.",
            },
        ]
        try:
            _, usage = backend._chat_messages_impl(
                messages,
                int(max_completion_tokens),
                int(application_retry_limit),
                "provider_load_probe",
                role="target",
            )
            return {
                "index": index,
                "status": "succeeded",
                "prompt_tokens": int(dict(usage or {}).get("prompt_tokens", 0) or 0),
                "completion_tokens": int(
                    dict(usage or {}).get("completion_tokens", 0) or 0
                ),
            }
        except ProviderCallExhausted as exc:
            return {
                "index": index,
                "status": "failed",
                "error_type": type(exc).__name__,
            }
        except Exception as exc:  # fail closed on non-provider probe failures
            return {
                "index": index,
                "status": "failed",
                "error_type": type(exc).__name__,
            }

    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=int(concurrency),
            thread_name_prefix="provider-load-probe",
        ) as executor:
            futures = [executor.submit(invoke, index) for index in range(int(requests))]
            outcomes = [future.result() for future in futures]
    except Exception as exc:
        observer_error_type = type(exc).__name__
    finally:
        try:
            uninstall_provider_observer(observer)
        except Exception as exc:
            observer_error_type = observer_error_type or type(exc).__name__

    try:
        events = observer.events()
    except Exception as exc:
        events = []
        observer_error_type = observer_error_type or type(exc).__name__

    status_counts = Counter(str(event.get("status", "")) for event in events)
    failure_codes = Counter()
    permanent_provider_errors = 0
    for event in events:
        raw_counts = event.get("failure_code_counts", {})
        if isinstance(raw_counts, dict):
            failure_codes.update({
                str(code): int(count) for code, count in raw_counts.items()
            })
        if event.get("status") == "failed" and str(
            event.get("last_failure_code", "")
        ) not in _TRANSIENT_FAILURE_CODES:
            permanent_provider_errors += 1

    completed = sum(item.get("status") == "succeeded" for item in outcomes)
    failed_events = [event for event in events if event.get("status") == "failed"]
    exhausted = sum(
        str(event.get("last_failure_code", "")) in _TRANSIENT_FAILURE_CODES
        and int(event.get("application_attempts", 0))
        >= int(event.get("retry_limit", 1))
        for event in failed_events
    )
    logical_ids = [str(event.get("logical_call_id", "")) for event in events]
    evidence_complete = bool(
        len(events) == int(requests)
        and len(set(logical_ids)) == len(logical_ids)
        and all(
            _event_is_complete(
                event,
                run_id=run_id,
                run_seed=run_seed,
                model=model,
                reasoning_effort=reasoning_effort,
            )
            for event in events
        )
    )
    passed = bool(
        not observer_error_type
        and len(outcomes) == int(requests)
        and completed == int(requests)
        and len(events) == int(requests)
        and status_counts == {"succeeded": int(requests)}
        and exhausted == 0
        and permanent_provider_errors == 0
        and evidence_complete
    )
    report = {
        "schema_version": 1,
        "probe_kind": "campaign_provider_load",
        "passed": passed,
        "run_id": str(run_id),
        "run_seed": int(run_seed),
        "campaign_id": campaign_gate.campaign_id,
        "model": str(model),
        "reasoning_effort": str(reasoning_effort),
        "concurrency": int(concurrency),
        "requests": int(requests),
        "max_completion_tokens": int(max_completion_tokens),
        "campaign_provider_max_inflight": int(campaign_gate.max_inflight),
        "logical_calls_recorded": len(events),
        "completed_logical_calls": int(completed),
        "failed_provider_calls": len(failed_events),
        "exhausted_provider_calls": exhausted,
        "permanent_provider_errors": permanent_provider_errors,
        "provider_evidence_complete": evidence_complete,
        "provider_status_counts": dict(sorted(status_counts.items())),
        "provider_failure_code_counts": dict(sorted(failure_codes.items())),
        "recovered_provider_calls": sum(
            bool(event.get("recovered", False)) for event in events
        ),
        "application_attempts": sum(
            int(event.get("application_attempts", 0)) for event in events
        ),
        "sdk_boundary_attempts": sum(
            int(event.get("sdk_boundary_attempts", 0)) for event in events
        ),
        "provider_queue_wait_ms": sum(
            int(event.get("provider_queue_wait_ms", 0)) for event in events
        ),
        "provider_service_latency_ms": sum(
            int(event.get("provider_service_latency_ms", 0)) for event in events
        ),
        "logical_call_latency_ms": sum(
            int(event.get("logical_call_latency_ms", 0)) for event in events
        ),
        "retry_backoff_ms": sum(
            int(event.get("retry_backoff_ms", 0)) for event in events
        ),
        "elapsed_ms": max(0, int((time.perf_counter() - started) * 1000)),
        "provider_calls_path": str(events_path),
        "provider_calls_sha256": (
            hashlib.sha256(events_path.read_bytes()).hexdigest()
            if events_path.is_file() else None
        ),
        "observer_error_type": observer_error_type or None,
    }
    _write_json_exclusive(report_path, report)
    return report


def _parse_delays(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in str(raw).split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("retry delays must not be empty")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gate-dir", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--run-id", default=f"provider_probe_{uuid.uuid4().hex}")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--api-key-env", default="MODEL_API_KEY")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--requests", type=int, default=DEFAULT_REQUESTS)
    parser.add_argument(
        "--max-completion-tokens", type=int, default=DEFAULT_MAX_COMPLETION_TOKENS
    )
    parser.add_argument("--max-inflight", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--sdk-max-retries", type=int, default=0)
    parser.add_argument("--application-retry-limit", type=int, default=5)
    parser.add_argument(
        "--retry-delays-seconds",
        type=_parse_delays,
        default=DEFAULT_RETRY_DELAYS_SECONDS,
    )
    parser.add_argument("--deterministic-jitter-ratio", type=float, default=0.10)
    args = parser.parse_args(argv)

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        parser.error(f"{args.api_key_env} is missing or empty")

    import skillopt.model as skillopt_model

    skillopt_model.set_backend("openai_compatible")
    skillopt_model.configure_openai_compatible(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        max_tokens=args.max_completion_tokens,
    )
    gate = CampaignProviderGate(
        gate_dir=Path(args.gate_dir),
        campaign_id=args.campaign_id,
        max_inflight=args.max_inflight,
    )
    report = run_provider_load_probe(
        output_dir=args.output_dir,
        campaign_gate=gate,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        run_id=args.run_id,
        run_seed=args.seed,
        concurrency=args.concurrency,
        requests=args.requests,
        max_completion_tokens=args.max_completion_tokens,
        application_retry_limit=args.application_retry_limit,
        retry_delays_seconds=args.retry_delays_seconds,
        deterministic_jitter_ratio=args.deterministic_jitter_ratio,
        expected_sdk_max_retries=args.sdk_max_retries,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
