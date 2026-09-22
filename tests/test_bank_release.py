import copy
import json
import zipfile
from pathlib import Path
import pytest
from atomic_skillgraph.deployment.asset_revision import ref_map,revise
from atomic_skillgraph.deployment.bank_release import _extract,CLASSES
from atomic_skillgraph.deployment.release_protocol import ReleaseError,VerifiedFrozenSource
from atomic_skillgraph.core.serialization import dataclass_from_dict,to_primitive
from atomic_skillgraph.agents.skill_guidance import normalize_guideline
from atomic_skillgraph.runtime.input_authorization import validate_declarations

@pytest.fixture(params=[42,43,44])
def authored(request):
    path=Path(__file__).resolve().parents[2]/f'ASTRG_R103_bank_edit_20260922/seed{request.param}_edited_bank.zip'
    if not path.exists():pytest.skip('local authored delivery input not installed')
    assets=[]
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if '/artifacts/' not in name or not name.endswith('.json'):continue
            payload=json.loads(z.read(name))
            if payload.get('metadata',{}).get('authoring_source')!='llm_bank_edit':continue
            directory=name.split('/artifacts/')[1].split('/')[0]
            kind='tool' if directory=='tools' else directory
            assets.append((kind,dataclass_from_dict(CLASSES[kind],payload)))
    assert len(assets)==33
    return assets

def test_revision_exact_closure_and_legal_guidance(authored):
    refs=ref_map([a for _,a in authored]);before=copy.deepcopy(to_primitive(authored))
    revised=[(kind,revise(kind,a,refs)) for kind,a in authored]
    assert to_primitive(authored)==before
    assert len(refs)==33
    for kind,a in revised:
        assert a.ref.version=='1.0.1'
        if kind in {'atomic','composite'}:assert normalize_guideline(a.guideline)['steps']
        if kind=='atomic':validate_declarations(a)
        if kind=='implementation':
            assert a.abstract_ref.version=='1.0.1'
            assert all(b.tool_ref.version=='1.0.1' for b in a.tool_bindings)
        if kind=='composite':
            assert all(o.node_ref.version=='1.0.1' for o in a.occurrences)
            assert all(e.origin=='existing_active' and e.existing_edge_id==e.edge_id
                       and not e.evidence_refs for e in [*a.data_edges,*a.dependency_edges])
            for o in a.occurrences:
                assert all(b.source_role in {'object','destination','light_source'}
                           for b in o.binding_specs.values() if b.kind.value=='skill_input')
            if a.metadata['catalog_id']=='G05':
                assert next(o for o in a.occurrences if o.step_id=='process').binding_specs['light'].source_role=='light_source'

def test_fresh_outputs_and_control_identity(authored):
    refs=ref_map([a for _,a in authored])
    atomics={a.metadata['catalog_id']:revise(k,a,refs) for k,a in authored if k=='atomic'}
    c2,c4=atomics['C02'],atomics['C04']
    assert {v['predicate'] for k,v in c2.validator_spec['output_derivations'].items() if k!='allow_open'}=={'entity.discovered_at'}
    assert {p.name for p in c4.outputs}=={'object','allow_open'}
    assert c4.validator_spec['output_derivations']['object']['predicate']=='agent.holds'
    for code in ('C01','C02','C03','C04','C05'):
        a=atomics[code]
        assert next(p for p in a.outputs if p.name=='allow_open').required_resolution=='semantic'
        assert a.validator_spec['output_derivations']['allow_open']=={'kind':'input_identity','input_role':'allow_open'}

@pytest.mark.parametrize('name',['../outside','/absolute','a/../../outside','C:/escape','a\\escape'])
def test_zip_escape_is_rejected(tmp_path,name):
    archive=tmp_path/'input.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr(name,'payload')
    with pytest.raises(ReleaseError):_extract(archive,tmp_path/'out')
    assert not (tmp_path/'outside').exists()

def test_unverified_source_not_constructible():
    with pytest.raises(ReleaseError):VerifiedFrozenSource({},Path('/arbitrary'))

def test_stage_receipts_are_immutable_and_idempotent(tmp_path):
    from atomic_skillgraph.deployment.bank_release import _stage
    digest=_stage(tmp_path,'01_import',{'source':'fixed'},{'count':3})
    assert _stage(tmp_path,'01_import',{'source':'fixed'},{'count':3})==digest
    with pytest.raises(ReleaseError):_stage(tmp_path,'01_import',{'source':'changed'},{'count':3})

def test_navigation_guard_requires_current_arrival_not_missing_action():
    from atomic_skillgraph.deployment.asset_revision import _fix_navigation
    source={'kind':'tool_input','source_role':'destination'}
    program=[{'op':'ACTION','node_id':'go','action_type':'GO_TO','argument_mapping':{'destination':source}},
        {'op':'RETURN','node_id':'done','output_sources':{}}]
    _fix_navigation(program)
    assert [n['op'] for n in program]==['IF','FOR_EACH']
    guard=program[1]['collection_source']
    assert guard['source']=='semantic_evidence'
    assert guard['where']['predicate']=='agent.at_location'
    assert guard['where']['semantic_compatible_with']['field']=='destination'
    assert program[1]['body'][0]['op']=='RETURN'
