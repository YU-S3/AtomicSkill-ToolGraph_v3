"""Separate recorded infrastructure uncertainty from missing model-turn usage."""
from collections.abc import Mapping


def recorded_unmetered_infrastructure(request) -> bool:
    get = request.get if isinstance(request, Mapping) else lambda key, default=None: getattr(request, key, default)
    return (
        get("outcome") == "error"
        and get("usage_status") == "unavailable"
        and get("error_code") in {
            "provider_invalid_response", "provider_usage_missing",
            "provider_timeout", "provider_transport_error", "provider_rate_limited",
            "provider_auth_error", "provider_invalid_request", "provider_capability_mismatch",
        }
        and bool(get("request_id"))
        and bool(get("payload_fingerprint"))
        and isinstance(get("started_at"), (int, float))
        and isinstance(get("ended_at"), (int, float))
        and get("ended_at") >= get("started_at")
    )


def decision_usage_auditable(trace) -> bool:
    if trace.resource_usage_complete:
        return True
    return bool(trace.provider_requests) and all(
        r.usage_status == "reported" or recorded_unmetered_infrastructure(r)
        for r in trace.provider_requests
    )
