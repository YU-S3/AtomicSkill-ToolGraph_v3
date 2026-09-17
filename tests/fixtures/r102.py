"""Authored entry declarations for deterministic legacy primitive fixtures.

These test tools declare no extra entry requirement. Atomic preconditions and
actual ACTION availability remain checked by production code. Source-only
replay tests should call ToolCompiler.compile directly instead.
"""
from atomic_skillgraph.evolution.tool_compiler import ToolCompiler


def compile_fixture(occurrences):
    return ToolCompiler().compile(occurrences, entry_contracts={
        occurrence.phase_id: {"conditions": [], "grounding_constraints": []}
        for occurrence in occurrences
    })
