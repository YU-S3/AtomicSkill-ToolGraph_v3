"""Read-only descriptions for persisted runtime Support; contracts are unchanged."""
from dataclasses import replace

from ..evolution.portability import contract_label


def portable_support_view(atomic):
    if not getattr(atomic, 'metadata', {}).get('runtime_support_promotion'):
        return atomic
    from .skill_guidance import normalize_guideline
    metadata = dict(atomic.metadata)
    try:
        guidance = normalize_guideline(atomic.guideline, formal_roles=[p.name for p in [*atomic.inputs, *atomic.outputs]])
    except ValueError as exc:
        guidance = {}
        metadata["guidance_rejection"] = str(exc)
    return replace(atomic, metadata=metadata, summary=contract_label(atomic.effects, atomic.outputs),
        inputs=[replace(item, description='') for item in atomic.inputs],
        outputs=[replace(item, description='') for item in atomic.outputs],
        guideline=guidance)
