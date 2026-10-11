"""Production replay of saved proposals; requests.post is intercepted, never paid."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from atomic_skillgraph.agents.protocol import NativeToolSpec
from atomic_skillgraph.empirical.budget_governor import input_token_bound
from atomic_skillgraph.empirical.contracts import digest
from atomic_skillgraph.empirical.model_view import structure_repair_material, expand_material
from atomic_skillgraph.empirical.prompts import LEARNING, LEARNER_PROMPT, BUILD
from atomic_skillgraph.experiments.benchmark_config import benchmark_config
from test_cf2_contracts import office, http, response
from test_office_r1 import record


@pytest.fixture
def archive():
    root=os.environ.get('OFFICE_R2_ARCHIVE')
    if not root:pytest.skip('Set the immutable UID0115 archive for production replay')
    return Path(root)


def read(root,name):return json.loads((root/(name+'.json')).read_text())


def test_actual_uid0115_payload_and_host_validation(archive,tmp_path,monkeypatch):
    original=read(archive,'original_proposal');before=read(archive,'reconstructed_provider_payload')
    errors=read(archive,'host_validation_errors')['full_reconstructed_feedback']
    trace=read(archive,'original_trace')
    request=next(r for r in trace['requests'] if r['stage']=='extractor' and r['repair']==0)
    material=expand_material(json.loads(request['messages'][1]['content']))
    directories=[material['experience']['local_evidence']]
    repair=structure_repair_material(original,errors,directories)
    previous=read(archive,'structure_repair_material')
    assert repair['proposal']==previous['proposal'] and repair['errors']==previous['errors']
    assert all(set(r)=={'local_evidence_ref','operation_name'} for r in repair['legal_operations'])
    s=office(tmp_path);s.config=benchmark_config(s.config,'officeqa')
    sent=http(monkeypatch,[response([original],name='submit_learning'),response([{'decision':'no_change'}],name='submit_learning')])
    try:
        with pytest.raises(ValueError,match='args'):s.learner.validate_learning_proposal(original)
        # This separate simulated model submission passes Host validation; production never fills args.
        valid=deepcopy(original)
        for node in valid['workflow']['nodes']:node['args']={}
        s.learner.validate_learning_proposal(valid)
        assert original==read(archive,'original_proposal')
        provider=s.provider('extractor','learning_structure_repair')
        payload=provider._build_payload([before['messages'][0],{'role':'user','content':json.dumps(repair,ensure_ascii=False,separators=(',',':'))}],
            [NativeToolSpec(t['function']['name'],t['function']['description'],t['function']['parameters']) for t in before['tools']],before['tool_choice'])
        assert {k:v for k,v in payload.items() if k!='messages'}=={k:v for k,v in before.items() if k!='messages'}
        assert input_token_bound(before)==8662 and input_token_bound(payload)==7490
        assert s.agent('extractor',LEARNER_PROMPT,material,'submit_learning',request['tools'][0]['function']['parameters'],
            validator=s.learner.validate_learning_proposal,repair_limit=1)=={'decision':'no_change'}
        actual=sent[1];body=json.loads(actual['messages'][1]['content'])
        assert len(sent)==2 and body==repair and actual['tools']==before['tools']
        assert input_token_bound(actual)==7490 and actual['thinking']['type']=='disabled' and actual['max_tokens']==2048
        record('uid0115_r2_actual',{'old_wire_bytes':8662,'new_wire_bytes':input_token_bound(actual),
            'proposal_hash':digest(body['proposal']),'errors_hash':digest(body['errors']),'tools_hash':digest(actual['tools']),
            'payload':actual,'actual_http':0,'intercepted_fixture_responses':2})
    finally:s.close()


def test_binding_cards_and_explicit_provenance_errors_keep_details():
    card={'case_id':'real','local_evidence_ref':'e1','operation_name':'read','kind':'native_operation',
          'effect':'inspection_or_json','published_files':[], 'output_references':[{'kind':'json','path':['data']}],
          'input_sources':[{'value':'real-path','kind':'operation_arg'}]}
    proposal={'realization_request':{'case_bindings':[{'local_evidence_ref':'e1','inputs':{'path':'real-path'}}]}}
    assert structure_repair_material(proposal,[],[[card]])['legal_operations'][0]==card
    error=[{'binding_index':None,'errors':[{'code':'invalid_output_reference','field':'reference_fields.x','detail':'missing'}]}]
    assert structure_repair_material({},error,[[card]])['legal_operations'][0]==card


def test_other_saved_structure_repairs(archive,tmp_path):
    s=office(tmp_path);s.config=benchmark_config(s.config,'officeqa');receipts=[]
    try:
        for path in (archive.parent/'04_office_v4/traces').glob('*.json'):
            trace=json.loads(path.read_text())
            initial=next(r for r in trace['requests'] if r['stage']=='extractor' and r['repair']==0)
            directory=expand_material(json.loads(initial['messages'][1]['content']))['experience']['local_evidence']
            for request in trace['requests']:
                if request['purpose']!='learning_structure_repair':continue
                saved=json.loads(request['messages'][1]['content'])
                errors=[{'binding_index':g['binding_index'],'errors':[
                    {**e,'detail':e.get('expected','')} for e in g['errors']]} for g in saved['errors']]
                material=structure_repair_material(saved['proposal'],errors,[directory])
                assert material['proposal']==saved['proposal'] and material['errors']==saved['errors']
                payload=s.provider('extractor','learning_structure_repair')._build_payload(
                    [request['messages'][0],{'role':'user','content':json.dumps(material,ensure_ascii=False,separators=(',',':'))}],
                    [NativeToolSpec(t['function']['name'],t['function']['description'],t['function']['parameters']) for t in request['tools']],request['tool_choice'])
                assert input_token_bound(payload)<=8192
                receipts.append({'task_id':trace['task']['task_id'],'wire_bytes':input_token_bound(payload),'http':0})
        assert len(receipts)>=2
        record('v4_structure_repairs_r2',{'samples':receipts})
    finally:s.close()


def test_unavoidable_payload_refusal_receipt_has_no_http(tmp_path,monkeypatch):
    s=office(tmp_path);sent=http(monkeypatch,[])
    try:
        with pytest.raises(ValueError,match='material_too_large'):
            s.agent('tool_builder','Build',{'required_source':'x'*20000},'submit_program',BUILD,repair_limit=1)
        refusal=s.requests[-1]['material_refusal']
        assert not sent and s.requests[-1]['admission_status']=='local_refusal'
        assert refusal['wire_bytes']==input_token_bound(refusal['payload'])>refusal['limit_bytes']
        assert sum(refusal['parts_bytes'].values())==refusal['wire_bytes'] and refusal['reason']
        assert not s.requests[-1].get('http_attempts') and 'response' not in s.requests[-1]
    finally:s.close()
