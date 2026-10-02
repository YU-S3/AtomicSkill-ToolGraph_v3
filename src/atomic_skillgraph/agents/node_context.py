"""Conservative node-local view; no grounding, model calls or policy decisions.

Keep the existing native-call field vocabulary. The original goal is omitted
only when its exact text is already present in the current Atomic contract;
contract coverage alone does not prove that all natural-language qualifiers
were preserved. Unknown observations and all binding/Repeat obligations stay.
"""
from __future__ import annotations

import copy
from .runtime_policy_projection import canonical_bytes, digest

VERSION = 'skillcompiler.node-context.v1'


def project_node_context(payload, *, native_tool_specs=()):
    out = copy.deepcopy(payload)
    atomic = out.get('current_state_snapshot', {}).get('current_atomic', {})
    goal = out.get('task_goal')
    exact = isinstance(goal, str) and bool(goal) and goal == atomic.get('summary')
    if exact:
        del out['task_goal']
    specs = list(native_tool_specs or ())
    schema_bytes = canonical_bytes([s.to_openai() for s in specs]) if specs else b'[]'
    audit = {
        'version': VERSION,
        'context_projection_fallback': not exact,
        'reason': 'goal_exactly_in_node_contract' if exact else 'unproven_task_qualifier_projection',
        'goal_sha256': digest(goal),
        'preserved_fields_sha256': digest({k: v for k, v in payload.items() if k != 'task_goal'}),
        'before_body_utf8_bytes': len(canonical_bytes(payload)),
        'after_body_utf8_bytes': len(canonical_bytes(out)),
        'native_tools_utf8_bytes': len(schema_bytes),
        'measurement': 'utf8_bytes_not_tokens',
    }
    return out, audit
