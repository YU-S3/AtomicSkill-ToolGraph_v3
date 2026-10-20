"""Idempotent answer envelope; deliberately no lexical or factual normalization."""
import re


def run(ctx, inputs):
    value = inputs.get("answer")
    if not isinstance(value, str) or not value.strip():
        return {"status": "needs_input", "outputs": {}, "error": "A nonempty answer is required"}
    wrapped = re.fullmatch(r"\s*<answer>(.*?)</answer>\s*", value, re.DOTALL)
    body = wrapped.group(1) if wrapped else value
    if not body.strip() or re.search(r"<\s*/?\s*answer\b", body, re.IGNORECASE):
        return {"status": "needs_input", "outputs": {}, "error": "Supply one plain answer or one complete answer tag pair"}
    return {"status": "ok", "outputs": {"answer": "<answer>" + body + "</answer>"}}
