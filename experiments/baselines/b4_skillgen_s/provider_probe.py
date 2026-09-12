"""B4-native concurrent provider probe used before a formal campaign lock."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from .inference_adapter import AuditedDeepSeekClient, ProviderExecutionError


def _write_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _event_complete(event: dict[str, Any], *, run_id: str, seed: int) -> bool:
    try:
        status = str(event["status"])
        attempts = int(event["application_attempts"])
        return bool(
            event.get("method") == "b4_skillgen_s"
            and event.get("phase") == "provider_load_probe"
            and event.get("run_id") == run_id
            and int(event.get("run_seed", -1)) == seed
            and event.get("stage") == "provider_load_probe"
            and status in {"succeeded", "failed"}
            and str(event.get("call_id", "")).strip()
            and attempts > 0
            and int(event.get("sdk_boundary_attempts", -1)) == attempts
            and isinstance(event.get("failure_codes"), list)
            and isinstance(event.get("provider_slot_ids"), list)
            and int(event.get("provider_queue_wait_ms", -1)) >= 0
            and int(event.get("retry_backoff_ms", -1)) >= 0
            and (
                status == "failed"
                or (
                    int(event.get("prompt_tokens", -1)) >= 0
                    and int(event.get("completion_tokens", 0)) > 0
                    and int(event.get("total_tokens", -1))
                    == int(event["prompt_tokens"]) + int(event["completion_tokens"])
                )
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def run_probe(
    *,
    output_dir: str | Path,
    gate_dir: str | Path,
    campaign_id: str,
    run_id: str,
    run_seed: int,
    model: dict[str, str],
    concurrency: int,
    requests: int,
    max_completion_tokens: int,
    max_inflight: int,
    retry_delays: tuple[float, ...],
    jitter_ratio: float,
) -> dict[str, Any]:
    if concurrency <= 0 or requests <= 0 or max_completion_tokens <= 0:
        raise ValueError("B4 provider probe counts must be positive")
    if concurrency != max_inflight:
        raise ValueError("B4 provider probe concurrency must equal its campaign cap")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    events_path = root / "provider_calls.jsonl"
    report_path = root / "provider_load_probe.json"
    if events_path.exists() or report_path.exists():
        raise FileExistsError("B4 provider probe output already exists")
    client = AuditedDeepSeekClient(
        model=model,
        run_id=run_id,
        run_seed=run_seed,
        phase="provider_load_probe",
        retry_delays=retry_delays,
        jitter_ratio=jitter_ratio,
        campaign={
            "campaign_id": campaign_id,
            "provider_gate_dir": str(Path(gate_dir).expanduser().resolve()),
            "campaign_provider_max_inflight": max_inflight,
        },
    )
    started = time.perf_counter()

    def invoke(index: int) -> tuple[int, dict[str, Any], str | None]:
        try:
            result = client.complete(
                [
                    {
                        "role": "system",
                        "content": "This is a provider load probe. Reply briefly.",
                    },
                    {
                        "role": "user",
                        "content": f"Reply with OK for probe request {index}.",
                    },
                ],
                task_id=f"provider_probe_{index:04d}",
                stage="provider_load_probe",
                call_index=index,
                max_tokens=max_completion_tokens,
                temperature=0.0,
                top_p=1.0,
                seed=run_seed + index,
                do_sample=False,
            )
            event = dict(result.event)
            event.pop("response_text", None)
            return index, event, None
        except ProviderExecutionError as exc:
            return index, dict(exc.event), exc.failure_kind
        except BaseException as exc:
            return index, {
                "schema_version": 1,
                "method": "b4_skillgen_s",
                "phase": "provider_load_probe",
                "run_id": run_id,
                "run_seed": run_seed,
                "stage": "provider_load_probe",
                "status": "failed",
                "error_type": type(exc).__name__,
                "application_attempts": 0,
                "sdk_boundary_attempts": 0,
                "failure_codes": ["probe_runtime"],
                "provider_slot_ids": [],
                "provider_queue_wait_ms": 0,
                "retry_backoff_ms": 0,
            }, "protocol_failure"

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrency,
        thread_name_prefix="b4-provider-probe",
    ) as executor:
        rows = list(executor.map(invoke, range(requests)))
    rows.sort(key=lambda item: item[0])
    events = [event for _, event, _ in rows]
    failure_kinds = [kind for _, _, kind in rows if kind is not None]
    event_bytes = "".join(
        json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
        for event in events
    ).encode("utf-8")
    _write_exclusive(events_path, event_bytes)
    completed = sum(event.get("status") == "succeeded" for event in events)
    failed = len(events) - completed
    exhausted = sum(kind == "infrastructure_failure" for kind in failure_kinds)
    permanent = sum(kind == "protocol_failure" for kind in failure_kinds)
    evidence_complete = bool(
        len(events) == requests
        and len({str(event.get("call_id", "")) for event in events}) == requests
        and all(_event_complete(event, run_id=run_id, seed=run_seed) for event in events)
    )
    passed = bool(
        completed == requests
        and failed == 0
        and exhausted == 0
        and permanent == 0
        and evidence_complete
    )
    report = {
        "schema_version": 1,
        "probe_kind": "campaign_provider_load",
        "passed": passed,
        "campaign_id": campaign_id,
        "run_id": run_id,
        "run_seed": run_seed,
        "model": model["model"],
        "reasoning_effort": model["reasoning_effort"],
        "concurrency": concurrency,
        "requests": requests,
        "max_completion_tokens": max_completion_tokens,
        "campaign_provider_max_inflight": max_inflight,
        "logical_calls_recorded": len(events),
        "completed_logical_calls": completed,
        "failed_provider_calls": failed,
        "exhausted_provider_calls": exhausted,
        "permanent_provider_errors": permanent,
        "provider_evidence_complete": evidence_complete,
        "application_attempts": sum(
            int(event.get("application_attempts", 0)) for event in events
        ),
        "sdk_boundary_attempts": sum(
            int(event.get("sdk_boundary_attempts", 0)) for event in events
        ),
        "provider_queue_wait_ms": sum(
            int(event.get("provider_queue_wait_ms", 0)) for event in events
        ),
        "retry_backoff_ms": sum(
            int(event.get("retry_backoff_ms", 0)) for event in events
        ),
        "elapsed_ms": max(0, int((time.perf_counter() - started) * 1000)),
        "provider_calls_path": str(events_path),
        "provider_calls_sha256": hashlib.sha256(event_bytes).hexdigest(),
        "observer_error_type": None,
    }
    _write_exclusive(
        report_path,
        (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    return report


def _parse_delays(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(result) != 4:
        raise argparse.ArgumentTypeError("B4 probe requires four retry delays")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gate-dir", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="MODEL_API_KEY")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--max-completion-tokens", type=int, required=True)
    parser.add_argument("--max-inflight", type=int, required=True)
    parser.add_argument("--retry-delays-seconds", type=_parse_delays, required=True)
    parser.add_argument("--deterministic-jitter-ratio", type=float, required=True)
    args = parser.parse_args(argv)
    report = run_probe(
        output_dir=args.output_dir,
        gate_dir=args.gate_dir,
        campaign_id=args.campaign_id,
        run_id=args.run_id,
        run_seed=args.seed,
        model={
            "provider": "openai_compatible",
            "base_url": args.base_url,
            "model": args.model,
            "api_key_env": args.api_key_env,
            "reasoning_effort": args.reasoning_effort,
        },
        concurrency=args.concurrency,
        requests=args.requests,
        max_completion_tokens=args.max_completion_tokens,
        max_inflight=args.max_inflight,
        retry_delays=args.retry_delays_seconds,
        jitter_ratio=args.deterministic_jitter_ratio,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
