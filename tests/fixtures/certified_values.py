"""Explicit concrete-entity certificates for publisher unit-test fixtures.

These fixtures represent a prior validator result; integration tests still
obtain certificates from the real validator rather than this constructor.
"""
from atomic_skillgraph.core.bindings import RuntimeBinding, BindingSource, BindingStatus, BindingResolution


def concrete_entities(values, refs, revision):
    return {role: RuntimeBinding(role, value, 'entity', BindingSource.TOOL_OUTPUT,
        BindingStatus.GROUNDED, BindingResolution.CONCRETE, list(refs), revision)
        for role, value in values.items()}
