"""Tool formals are independent from Atomic roles; replay uses one namespace."""
import copy
from dataclasses import replace
from atomic_skillgraph.core.bindings import BindingExpression, BindingExprKind
from atomic_skillgraph.evolution.contract_canonicalizer import AtomicContractCanonicalizer
from atomic_skillgraph.evolution.executable_boundary import atomic_boundary_view
from atomic_skillgraph.evolution.identity_matching import match_tool, verify_tool_proof
from test_r92_replay_source_authority import _source
from fixtures.r102 import compile_fixture


def test_D_alias_formals_are_rewritten_via_explicit_route_not_guessed():
    source = compile_fixture([_source("a","trace_a","object","apple_1")])[0]
    tool = AtomicContractCanonicalizer._rewrite_tool(source.tool,{"object":"arg"},{"held_object":"out"})
    b = source.implementation.tool_bindings[0]
    impl = replace(source.implementation,tool_bindings=[replace(b,parameter_mapping={"arg":b.parameter_mapping["object"]})],
        execution_policy={**source.implementation.execution_policy,"output_mapping":{
            "held_object":BindingExpression(BindingExprKind.TOOL_OUTPUT,source_step=b.role,source_role="out")}})
    before = copy.deepcopy(tool)
    renamed, adapted, proof = atomic_boundary_view(source.atomic,tool,impl)
    assert verify_tool_proof(tool,renamed,proof)
    assert match_tool(source.tool,renamed).status == "exact"
    assert set(renamed.signature["properties"]) == {"object"}
    assert set(adapted.tool_bindings[0].parameter_mapping) == {"object"}
    assert adapted.execution_policy["output_mapping"]["held_object"].source_role == "held_object"
    assert tool == before
    bad = copy.deepcopy(impl)
    bad.tool_bindings[0].parameter_mapping["arg"] = BindingExpression(BindingExprKind.CONSTANT,constant="apple_1")
    assert atomic_boundary_view(source.atomic,tool,bad) is None
