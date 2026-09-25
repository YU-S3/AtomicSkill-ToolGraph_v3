from copy import deepcopy
from dataclasses import replace

import pytest

from atomic_skillgraph.core.bindings import BindingExpression, BindingExprKind
from atomic_skillgraph.core.contracts import AbstractAtomicSkill, ParameterSpec, SemanticPredicate, ToolAsset, ImplementationAtom
from atomic_skillgraph.core.refs import SkillRef, ToolRef
from atomic_skillgraph.core.status import SkillStatus
from atomic_skillgraph.evolution.contract_canonicalizer import AtomicContractCanonicalizer, atomic_contract_signature
from atomic_skillgraph.evolution.identity_matching import (match_atomic, verify_atomic_proof, raw_hash,
    match_tool, verify_tool_proof, match_implementation, tool_view)


def atomic(inputs=("a", "b"), outputs=("x", "y"), *, crossed=False):
    return AbstractAtomicSkill(
        ref=SkillRef("draft", "1.0.0"), summary="reference",
        inputs=[ParameterSpec(n, "entity") for n in inputs],
        outputs=[ParameterSpec(n, "entity") for n in outputs], preconditions=[],
        effects=[SemanticPredicate("known", {"object": BindingExpression(
            BindingExprKind.SKILL_INPUT, source_role=n)}) for n in inputs],
        validator_spec={"output_derivations": {outputs[0]: {"kind": "input_identity", "input_role": inputs[crossed]},
            outputs[1]: {"kind": "input_identity", "input_role": inputs[not crossed]}}},
        failure_modes=[], guideline={}, metadata={}, status=SkillStatus.CANDIDATE)


def test_bounded_count_reference_survives_canonical_promotion():
    from atomic_skillgraph.deployment.scienceworld_reference import bounded_wait
    a,i,t=bounded_wait()
    canonical=AtomicContractCanonicalizer().canonicalize(a,t,i,
        input_role_map={'wait_steps':'count'},output_role_map={'evidence_ref':'witness'})
    result=match_implementation(i,canonical.implementation,source_atomic=a,target_atomic=canonical.atomic,
        source_tools={str(t.ref):t},target_tools={str(canonical.tool.ref):canonical.tool})
    assert result.status=='exact',result
    bad=deepcopy(canonical.tool)
    bad.artifact['program'][0]['collection_source']['count']={'source':'constant','value':3}
    assert match_tool(t,bad).status!='exact'


def test_tool_field_projection_survives_role_renaming_without_renaming_keys():
    from atomic_skillgraph.evolution.contract_canonicalizer import _rewrite_tool_ir
    reference={'kind':'skill_input','source_role':'pair','field_path':['pair','left']}
    rewritten=_rewrite_tool_ir(reference,{'pair':'input_000'}, {})
    assert rewritten['source_role']=='input_000'
    assert rewritten['field_path']==['pair','left']
    projected_effect={'predicate':'seen','args':{'entity':reference}}
    projected=_rewrite_tool_ir(projected_effect,{'pair':'input_000'}, {})
    assert projected['args']['entity']['source_role']=='input_000'
    assert projected['args']['entity']['field_path']==['pair','left']
    local={**reference,'kind':'local_variable'}
    assert _rewrite_tool_ir(local,{'pair':'input_000'}, {})['source_role']=='pair'
    predicate={'predicate':'seen','args':{'entity':{'kind':'local_variable','source_role':'pair'}}}
    assert _rewrite_tool_ir(predicate,{'pair':'input_000'}, {})['args']['entity']['source_role']=='pair'


def test_full_relation_renaming_and_proof_without_mutation():
    a = atomic()
    b = atomic(("z", "c"), ("q", "p"))
    b = replace(b, inputs=list(reversed(b.inputs)), effects=list(reversed(b.effects)))
    before = (raw_hash(a), raw_hash(b))
    result = match_atomic(a, b)
    assert result.status == "exact"
    assert verify_atomic_proof(a, b, result.proof)
    assert atomic_contract_signature(a) == atomic_contract_signature(b)
    assert before == (raw_hash(a), raw_hash(b))
    assert not verify_atomic_proof(a, replace(b, summary="changed bytes"), result.proof)


def test_symmetric_atomic_search_continues_to_joint_compatible_mapping():
    a, b = atomic(), atomic(("c", "d"), ("p", "q"))
    seen = []
    def executable(inputs, outputs):
        seen.append(inputs)
        return inputs["a"] == "d" and outputs["x"] == "q"
    result = match_atomic(a, b, compatible_mapping=executable)
    assert result.status == "exact"
    assert result.proof.input_role_map == {"a": "d", "b": "c"}
    assert verify_atomic_proof(a, b, result.proof)


def test_relation_multiplicity_not_just_role_descriptors():
    a = atomic()
    bad = deepcopy(a)
    bad.validator_spec["output_derivations"]["y"]["input_role"] = "a"
    assert match_atomic(a, bad).status == "different"


@pytest.mark.parametrize("field,value", [("required", False), ("runtime_resolvable", True),
                                           ("required_resolution", "concrete")])
def test_boundary_attributes_are_semantic(field, value):
    a = atomic()
    b = replace(a, inputs=[replace(a.inputs[0], **{field: value}), a.inputs[1]])
    assert match_atomic(a, b).status == "different"


@pytest.mark.parametrize("field,value", [("effect_domain","evidence"),("cardinality",2),("distinct_by","object")])
def test_I04_predicate_obligations_are_not_erased(field,value):
    a = atomic()
    b = deepcopy(a)
    b.effects[0] = replace(b.effects[0], **{field:value})
    assert match_atomic(a,b).status == "different"


def test_I04_argument_position_not_formal_name_is_semantic():
    a,b = atomic(),atomic()
    b.effects[0] = replace(b.effects[0], args={"location":b.effects[0].args["object"]})
    assert match_atomic(a,b).status == "different"


@pytest.mark.parametrize("literal", ["a", "$a", "cabinet_1", True, 1, 1.0, None])
def test_typed_constants_not_role_references(literal):
    a = atomic()
    a.effects.append(SemanticPredicate("constant", {"literal": BindingExpression(
        BindingExprKind.CONSTANT, constant=literal)}))
    b = AtomicContractCanonicalizer().canonicalize(a,
        input_role_map={"a": "z", "b": "c"}, output_role_map={"x": "p", "y": "q"}).atomic
    assert b.effects[-1].args["literal"].constant == literal
    assert raw_hash(b.effects[-1].args["literal"].constant) == raw_hash(literal)
    assert match_atomic(a, b).status == "exact"


def test_bare_constant_same_as_role_is_not_rewritten():
    a = atomic()
    a.effects.append(SemanticPredicate("literal", {"value": "a"}))
    b = AtomicContractCanonicalizer().canonicalize(a,
        input_role_map={"a": "c", "b": "d"}, output_role_map={"x": "p", "y": "q"}).atomic
    assert b.effects[-1].args["value"] == "a"
    assert match_atomic(a, b).status == "exact"


def test_bool_number_string_missing_are_not_equal():
    a = atomic()
    for x, y in [(True, 1), (1, 1.0), (1, "1"), (None, "")]:
        left, right = deepcopy(a), deepcopy(a)
        left.validator_spec["unknown_semantic_extension"] = x
        right.validator_spec["unknown_semantic_extension"] = y
        assert match_atomic(left, right).status == "different"
    other = deepcopy(a)
    other.validator_spec["unknown_semantic_extension"] = None
    assert match_atomic(a, other).status == "different"


def test_search_limit_is_unknown_and_has_no_proof():
    result = match_atomic(atomic(), atomic(), max_states=0)
    assert result.status == "unknown" and result.proof is None
    assert result.search_states == 0


def test_provenance_only_whitelist_is_excluded():
    a, b = atomic(), atomic()
    a.validator_spec["output_derivations"]["x"]["witness_refs"] = ["trace:first"]
    b.validator_spec["output_derivations"]["x"]["witness_refs"] = ["trace:second"]
    assert match_atomic(a, b).status == "exact"
    b.validator_spec["output_derivations"]["x"]["unrecognized_semantic_field"] = True
    assert match_atomic(a, b).status == "different"


def tool(item="item", output="result", node="take", local="current"):
    return ToolAsset(ToolRef("draft_tool", "1.0.0"), "take in loop",
        {"type": "object", "properties": {item: {"type": "string"}}, "required": [item]},
        {"entry_contract": {"conditions": [], "grounding_constraints": []},
         "output_schema": {"type": "object", "properties": {output: {"type": "string"}}, "required": [output]}},
        "tool_ir_v1", {"schema_version": 1, "max_actions": 2, "program": [
            {"node_id": node + "_loop", "op": "FOR_EACH", "max_iterations": 2,
             "iteration_variable": local,
             "collection_source": {"source": "action_catalog", "where": {"action_type": "TAKE"},
                 "project": {"kind": "argument", "role": "item"}},
             "body": [{"node_id": node, "op": "ACTION", "action_type": "TAKE",
                 "argument_mapping": {"item": {"kind": "local_variable", "source_role": local}}}]},
            {"node_id": node + "_return", "op": "RETURN", "output_sources": {
                output: {"source": "tool_input", "field": item}}}],
            "final_effects": [], "evidence_outputs": [],
            "path_expectations": [{"path": "program/" + node + "_loop/body/" + node}]},
        [], {"reviewed": True}, {}, {})


def test_tool_ast_scoped_renaming_has_reverifiable_proof():
    a, b = tool(), tool("thing", "answer", "pick", "element")
    result = match_tool(a, b)
    assert result.status == "exact", result
    assert result.proof.node_id_map["take"] == "pick"
    assert result.proof.local_symbol_map["take_loop::current"] == "pick_loop::element"
    assert verify_tool_proof(a, b, result.proof)
    assert "item" in tool_view(a).payload["artifact"]["program"][0]["body"][0]["argument_mapping"]


@pytest.mark.parametrize("change", ["order", "selector", "entry", "refresh", "max_actions", "return"])
def test_tool_executable_changes_are_not_equivalent(change):
    a, b = tool(), tool()
    if change == "order":
        b.artifact["program"].reverse()
    elif change == "selector":
        b.artifact["program"][0]["collection_source"]["where"]["action_type"] = "PUT"
    elif change == "entry":
        b.interface["entry_contract"]["conditions"] = [{"predicate": "ready", "args": {}}]
    elif change == "refresh":
        b.artifact["program"][0]["collection_source"]["refresh_each_iteration"] = True
    elif change == "max_actions":
        b.artifact["max_actions"] = 3
    else:
        b.artifact["program"][1]["output_sources"]["result"] = {"source": "constant", "value": "item"}
    assert match_tool(a, b).status == "different"


def test_local_cannot_escape_loop_or_capture_input():
    a, b = tool(), tool()
    b.artifact["program"][1]["output_sources"]["result"] = {"source": "local_variable", "field": "current"}
    assert match_tool(a, b).status == "unknown"
    b = tool()
    b.artifact["program"][0]["body"][0]["argument_mapping"]["item"]["kind"] = "skill_input"
    assert match_tool(a, b).status == "unknown"


def test_tool_schema_annotations_only_excluded_in_schema_positions():
    a, b = tool(), tool()
    b.signature["description"] = "additional explanation"
    b.signature["properties"]["item"]["title"] = "different explanation"
    assert match_tool(a, b).status == "exact"
    a.signature["properties"]["item"]["const"] = {"description": "literal one"}
    b.signature["properties"]["item"]["const"] = {"description": "literal two"}
    assert match_tool(a, b).status == "different"


def test_joint_implementation_checks_every_tool_and_mapping():
    from atomic_skillgraph.core.bindings import ToolBinding
    a, b = atomic(), atomic(("c", "d"), ("p", "q"))
    ta, tb = tool(), tool("thing", "answer", "pick", "element")
    def impl(atom, first_input, output, t):
        return ImplementationAtom(SkillRef("impl", "1.0.0"), atom.ref, [
            ToolBinding(t.ref, "primary", {next(iter(t.signature["properties"])): BindingExpression(
                BindingExprKind.SKILL_INPUT, source_role=first_input)}, 0)], [],
            {"output_mapping": {output: BindingExpression(BindingExprKind.TOOL_OUTPUT,
                source_role=next(iter(t.interface["output_schema"]["properties"])), source_step="primary")}}, {}, {})
    ia, ib = impl(a, "a", "x", ta), impl(b, "d", "q", tb)
    kwargs = dict(source_atomic=a, target_atomic=b,
        source_tools={str(ta.ref): ta}, target_tools={str(tb.ref): tb})
    match = match_implementation(ia, ib, **kwargs)
    assert match.status == "exact", match
    assert match.proof.input_role_map["a"] == "d"
    ib.execution_policy["output_mapping"] = {"p": ib.execution_policy["output_mapping"]["q"]}
    assert match_implementation(ia, ib, **kwargs).status == "different"
    ia.tool_bindings.append(replace(ia.tool_bindings[0], role="second", order=1))
    assert match_implementation(ia, ib, **kwargs).status == "different"


def test_I_multitool_identity_uses_declared_order_not_storage_order():
    from atomic_skillgraph.core.bindings import ToolBinding
    a, t = atomic(), tool()
    first = ToolBinding(t.ref, "first", {"item": BindingExpression(
        BindingExprKind.SKILL_INPUT, source_role="a")}, 0)
    second = ToolBinding(t.ref, "second", {"item": BindingExpression(
        BindingExprKind.TOOL_OUTPUT, source_step="first", source_role="result")}, 1)
    source = ImplementationAtom(SkillRef("two", "1.0.0"), a.ref, [first,second], [],
        {"output_mapping":{"x":BindingExpression(BindingExprKind.TOOL_OUTPUT,
            source_step="second",source_role="result")}}, {}, {})
    target = replace(source, tool_bindings=[second,first])
    from atomic_skillgraph.knowledge.identity_index import bucket_key
    assert bucket_key("implementation", source) == bucket_key("implementation",target)
    kwargs = dict(source_atomic=a,target_atomic=a,
                  source_tools={str(t.ref):t},target_tools={str(t.ref):t})
    result = match_implementation(source,target,**kwargs)
    assert result.status == "exact", result
    target = replace(target, tool_bindings=[replace(second,order=0),first])
    assert match_implementation(source,target,**kwargs).status == "unknown"


def test_tool_identity_does_not_depend_on_hash_equality(monkeypatch):
    import atomic_skillgraph.evolution.identity_matching as matching
    a, b = tool(), tool()
    b.artifact["max_actions"] = 99
    monkeypatch.setattr(matching, "raw_hash", lambda _: "collision")
    assert matching.match_tool(a, b).status == "different"
    x, y = atomic(), atomic()
    y.validator_spec["identity_strict"] = True
    assert matching.match_atomic(x, y).status == "different"


def test_unknown_validator_text_is_not_a_role_reference():
    a = atomic()
    a.validator_spec["future_extension"] = {"literal": "$a"}
    b = AtomicContractCanonicalizer().canonicalize(a,
        input_role_map={"a": "c", "b": "d"}, output_role_map={"x": "p", "y": "q"}).atomic
    assert b.validator_spec["future_extension"] == {"literal": "$a"}
    assert match_atomic(a, b).status == "exact"


def test_registration_preserves_suppressed_identity_and_guideline(tmp_path):
    from atomic_skillgraph.knowledge.database import StateDatabase
    from atomic_skillgraph.knowledge.artifact_store import ArtifactStore
    from atomic_skillgraph.knowledge.skill_registry import SkillRegistry
    from atomic_skillgraph.knowledge.tool_registry import ToolRegistry
    from atomic_skillgraph.knowledge.identity_index import IdentityIndex
    from atomic_skillgraph.evolution.aligner import Aligner
    with StateDatabase(tmp_path / "state.sqlite3") as db:
        store = ArtifactStore(tmp_path, db)
        skills, tools = SkillRegistry(store, db), ToolRegistry(store, db)
        original = replace(atomic(), status=SkillStatus.SUPPRESSED, guideline={"steps": ["Check", "Act", "Check"]})
        skills.register_atomic(original)
        path = store.path_for("atomic", original.ref)
        before = path.read_bytes()
        candidate = replace(atomic(("c", "d"), ("p", "q")), guideline={"steps": ["new instructions"]})
        assert Aligner(skills, tools).align_atomic(candidate) == original.ref
        assert skills.get_atomic(original.ref).status is SkillStatus.SUPPRESSED
        assert path.read_bytes() == before
        IdentityIndex(db, tmp_path).verify(str(original.ref))
        assert len(db.rows("SELECT * FROM artifact_identity_index")) == 1


def test_identity_index_is_in_freeze_digest_and_checkpoint(tmp_path):
    from atomic_skillgraph.system import AtomicSkillGraphSystem
    from experiments.fakes import FakeHarness
    from test_batch_evolution import _system_config
    from experiments.protocol import TaskCheckpointStore
    from atomic_skillgraph.knowledge.identity_index import IdentityIndex
    from atomic_skillgraph.core.serialization import read_json
    with AtomicSkillGraphSystem(_system_config(tmp_path / "knowledge"), harness=FakeHarness()) as system:
        before = system.knowledge_digest()
        checkpoint = TaskCheckpointStore(tmp_path / "checkpoint", system.data_dir)
        checkpoint.create(system.database, run_id="fixture", task_id="identity",
            before_digest=before, config_hash="fixture", code_commit="fixture")
        system.skills.register_atomic(atomic())
        index = IdentityIndex(system.database, system.data_dir)
        index.verify(str(atomic().ref))
        assert system.knowledge_digest() != before
        # The proof is under the original artifact root, so the existing
        # checkpoint/freeze copier cannot silently omit the new evidence file.
        proof = system.database.rows("SELECT proof_path FROM artifact_identity_index")[0][0]
        assert proof.startswith("artifacts/")
        frozen = system.freeze(tmp_path / "snapshot")
        assert (frozen / proof).is_file()
        system.artifacts.verify_all()
