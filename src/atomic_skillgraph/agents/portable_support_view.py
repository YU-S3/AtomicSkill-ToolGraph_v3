"""Read-only descriptions for persisted runtime Support; contracts are unchanged."""
from dataclasses import replace

from ..evolution.portability import contract_label


def portable_support_view(atomic):
    if not getattr(atomic, 'metadata', {}).get('runtime_support_promotion'):
        return atomic
    return replace(atomic, summary=contract_label(atomic.effects, atomic.outputs),
        inputs=[replace(item, description='') for item in atomic.inputs],
        outputs=[replace(item, description='') for item in atomic.outputs],
        guideline={'runtime_automation': True, 'steps': []})
