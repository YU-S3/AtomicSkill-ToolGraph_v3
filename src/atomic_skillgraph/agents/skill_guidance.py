"""Bounded, portable soft guidance; never an executable policy."""

from ..evolution.portability import validate_portability


def normalize_guideline(value, *, forbidden_terms=(), formal_roles=()):
    if not value:
        return {}
    if not isinstance(value, dict) or set(value) - {"steps", "notes"} or "steps" not in value:
        raise ValueError("guideline requires only steps and notes")
    steps, notes = value["steps"], value.get("notes", [])
    if not isinstance(steps, list) or not 1 <= len(steps) <= 6:
        raise ValueError("guideline requires 1-6 steps")
    if not isinstance(notes, list) or len(notes) > 2:
        raise ValueError("guideline allows 0-2 notes")
    for text in [*steps, *notes]:
        if not isinstance(text, str) or not text.strip() or len(text) > 200:
            raise ValueError("guideline items require 1-200 characters")
        from ..tooling.validator import episode_literal_matches, _mask_formal_reference_spans
        checked = _mask_formal_reference_spans(text, set(formal_roles))
        if episode_literal_matches(checked, annotation=True) or not validate_portability(
            checked, additional_forbidden_terms=forbidden_terms
        ).passed:
            raise ValueError("guideline is not portable")
    return {"steps": list(steps), "notes": list(notes)}


def guidance_view(atomic):
    try:
        fields = (atomic.get("inputs", []) + atomic.get("outputs", []) if isinstance(atomic, dict)
                  else [*atomic.inputs, *atomic.outputs])
        roles = [p["name"] if isinstance(p, dict) else p.name for p in fields]
        guidance = normalize_guideline(atomic.get("guideline") if isinstance(atomic, dict) else atomic.guideline,
                                       formal_roles=roles)
    except ValueError as exc:
        return {"soft_reference": True, "guidance_absent": True, "rejection_reason": str(exc)}
    return organize_guidance_view({"soft_reference": True, "guidance_absent": not bool(guidance), **guidance})


def organize_guidance_view(view):
    """Stable typed main/note grouping, downstream of the original validity gate.

    Adapted from EmbodiSkill format_task_prompt_with_skills, commit
    760126030eab1d33ec6a6f30988f0f1fb58df3a7 (MIT). See THIRD_PARTY_NOTICES.md.
    Unlike upstream, no prefix classification, deduplication or rewriting.
    """
    import copy
    result = {}
    for key in ("soft_reference", "guidance_absent", "rejection_reason", "steps", "notes"):
        if key in view:
            result[key] = copy.deepcopy(view[key])
    for key, value in view.items():
        if key not in result:
            result[key] = copy.deepcopy(value)
    return result
