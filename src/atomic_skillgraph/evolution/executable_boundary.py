"""Read an existing one-Tool route in its Atomic's declared boundary names.

This is alpha-renaming only, not parameter inference or specialization. The
source artifact is immutable; unsupported non-bijective adapters keep Builder's
ordinary opportunity instead of executing a replay with the wrong arguments.
"""
from dataclasses import replace
from ..core.bindings import BindingExpression, BindingExprKind
from .contract_canonicalizer import AtomicContractCanonicalizer
from .identity_matching import match_tool


def atomic_boundary_view(atomic, tool, implementation):
    if len(implementation.tool_bindings) != 1:
        return None
    binding = implementation.tool_bindings[0]
    inputs = {}
    for formal, raw in binding.parameter_mapping.items():
        expr = BindingExpression.from_dict(raw)
        if expr.kind is not BindingExprKind.SKILL_INPUT:
            return None
        inputs[formal] = expr.source_role
    outputs = {}
    for role, raw in implementation.execution_policy.get("output_mapping", {}).items():
        expr = BindingExpression.from_dict(raw)
        if expr.kind is not BindingExprKind.TOOL_OUTPUT or expr.source_step != binding.role:
            return None
        if expr.source_role in outputs:
            return None
        outputs[expr.source_role] = role
    for mapping, original, target in (
        (inputs, tool.signature.get("properties", {}), {p.name for p in atomic.inputs}),
        (outputs, tool.interface.get("output_schema", {}).get("properties", {}), {p.name for p in atomic.outputs}),
    ):
        if set(mapping) != set(original) or set(mapping.values()) != target or len(mapping) != len(target):
            return None
    # Retain exact-byte fast path, including conservative legacy artifacts.
    if all(k == v for mapping in (inputs, outputs) for k,v in mapping.items()):
        return tool, implementation, None
    renamed = AtomicContractCanonicalizer._rewrite_tool(tool, inputs, outputs)
    proof = match_tool(tool, renamed, input_role_map=inputs, output_role_map=outputs)
    if proof.status != "exact":
        return None
    policy = dict(implementation.execution_policy)
    if "output_mapping" in policy:
        policy["output_mapping"] = {role:replace(BindingExpression.from_dict(raw),
            source_role=outputs[BindingExpression.from_dict(raw).source_role])
            for role,raw in policy["output_mapping"].items()}
    remapped = replace(implementation, tool_bindings=[replace(binding,
        parameter_mapping={inputs[k]:v for k,v in binding.parameter_mapping.items()})], execution_policy=policy)
    return renamed, remapped, proof.proof
