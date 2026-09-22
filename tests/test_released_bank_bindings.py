"""R06/R07: control authorization does not create entity/world authority."""
import copy
import pytest
from atomic_skillgraph.core.bindings import BindingResolution, BindingSource
from atomic_skillgraph.core.contracts import AbstractAtomicSkill, ParameterSpec
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.serialization import to_primitive, dataclass_from_dict
from atomic_skillgraph.runtime.evidence_store import GroundingEvidenceStore
from atomic_skillgraph.runtime.input_authorization import authorize, validate_declarations
from atomic_skillgraph.harness.protocol import HarnessActionSpec

def atomic():
    return AbstractAtomicSkill(SkillRef('control_test','1.0.0'),'control',
        [ParameterSpec('permission','bool'),ParameterSpec('scope','list')],[],[],[],
        {'input_authorization':{'permission':{'kind':'caller_boolean'},'scope':{
            'kind':'ordered_entity_scope','element_semantic_type':'entity','min_items':1,'max_items':8,'unique_items':True}}},[],{}, {})

@pytest.mark.parametrize('value',[True,False])
def test_boolean_is_explicit_pure_and_semantic(value):
    store=GroundingEvidenceStore();before=copy.deepcopy(store.__dict__)
    b=authorize(atomic(),'permission',value,call_id='call_a',evidence_store=store,revision=0)
    assert b.value is value and b.source is BindingSource.CALLER_AUTHORIZED
    assert b.resolution is BindingResolution.SEMANTIC
    assert store.__dict__==before
    assert dataclass_from_dict(type(b),to_primitive(b))==b
    other=authorize(atomic(),'permission',value,call_id='call_b',evidence_store=store,revision=0)
    assert b.evidence_refs!=other.evidence_refs

@pytest.mark.parametrize('value',[1,0,'true',None,[],{}])
def test_boolean_rejects_coercion(value):
    with pytest.raises(ValueError):authorize(atomic(),'permission',value,call_id='c',evidence_store=GroundingEvidenceStore(),revision=0)

@pytest.mark.parametrize('value',[[],['unseen'],['a','a'],list('abcdefghi'),[1]])
def test_scope_rejects_unproven_or_invalid(value):
    with pytest.raises(ValueError):authorize(atomic(),'scope',value,call_id='c',evidence_store=GroundingEvidenceStore(),revision=0)

def test_entity_cannot_borrow_control_authority():
    a=atomic();a.inputs[0].semantic_type='entity'
    with pytest.raises(ValueError):validate_declarations(a)
    a=atomic();a.inputs[0].required_resolution='concrete'
    with pytest.raises(ValueError):validate_declarations(a)
