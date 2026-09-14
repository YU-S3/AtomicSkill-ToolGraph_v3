"""Fixed reasoning allowance, independent of upstream visible-output hints."""
from experiments.baselines.common.usage import visible_completion_tokens
POLICY_VERSION = "reasoning-aware-v2.2"
DEFAULT_CAP = 65536
OUTPUT_HINTS = dict(solver=512, stuck_recovery=512, trajectory_condensation=512,
    failure_diagnosis=512, episode_reflection=512, manual_revision=2048, manual_refactor=2048)


def summarize_budget_events(events):
    """Billed physical usage, including unsuccessful returned responses."""
    rows = [dict(attempt, role=event["role"]) for event in events
            for attempt in event.get("physical_attempt_usage", [event])]
    result = dict(provider_attempts=len(rows),
        budget_exhaustion_count=sum(bool(e.get("budget_exhausted")) for e in rows),
        finish_reason_length_count=sum(e.get("finish_reason") == "length" for e in rows),
        api_cost=None, api_cost_unpriced=True)
    for role in ("target", "evolution"):
        group = [e for e in rows if ("target" if e["role"] == "target" else "evolution") == role]
        for field in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "visible_completion_tokens"):
            values = [e.get(field) for e in group]
            subtotal = sum(v for v in values if isinstance(v, int))
            result[role+"_"+field] = subtotal if all(isinstance(v, int) for v in values) else None
            result[role+"_"+field+"_known_subtotal"] = subtotal
    return result


def policy_metadata(model):
    cap = int(model.get("provider_completion_cap", DEFAULT_CAP))
    if cap < DEFAULT_CAP:
        raise ValueError("High-reasoning provider cap must be at least 65536")
    return dict(provider_completion_cap=cap, reasoning_effort="high",
        response_consumption="content_only", reasoning_content_used_by_method=False,
        budget_policy_version=POLICY_VERSION)


def response_evidence(response, *, cap, hint, parser=None):
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    def token(owner, name):
        value = getattr(owner, name, None)
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
    prompt = token(usage, "prompt_tokens")
    completion = token(usage, "completion_tokens")
    reasoning = token(details, "reasoning_tokens")
    visible = visible_completion_tokens(completion, reasoning)
    choices = getattr(response, "choices", None) or []
    choice = choices[0] if choices else None
    content = getattr(getattr(choice, "message", None), "content", None)
    present = isinstance(content, str) and bool(content.strip())
    parse_success = present
    if present and parser is not None:
        try:
            parsed = parser(content)
            parse_success = parsed is not None and parsed is not False and (not isinstance(parsed, str) or bool(parsed.strip()))
        except (ValueError, TypeError, KeyError, AttributeError):
            parse_success = False
    finish = getattr(choice, "finish_reason", None)
    exhausted = finish == "length" and not parse_success and completion is not None and completion >= cap * .98
    values = (prompt, completion, reasoning, visible)
    return dict(method_output_token_hint=hint, provider_completion_cap=cap,
        reasoning_effort="high", prompt_tokens=prompt, completion_tokens=completion,
        reasoning_tokens=reasoning, visible_completion_tokens=visible,
        usage_status="known" if all(v is not None for v in values) else "partial" if any(v is not None for v in values) else "unavailable",
        finish_reason=finish, content_present=present,
        content_length_chars=len(content) if isinstance(content, str) else 0,
        content_parse_success=parse_success, budget_exhausted=exhausted)
