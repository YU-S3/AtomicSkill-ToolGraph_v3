"""Production wire targets and Host tolerance remain independent; all HTTP is intercepted."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest
import requests

from atomic_skillgraph.empirical import capability_batch as cb
from atomic_skillgraph.empirical.contracts import PublicTask
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.experiments import run_choice_capability_batch as batch
from atomic_skillgraph.harness.benchmarks import AnswerAdapter
from test_capability_batch import fixture_source,fixture_transport


@pytest.mark.parametrize('reason_size,status',[(210,'procedural_only'),(240,'procedural_only'),
    (241,'procedural_only'),(380,'procedural_only'),(436,'procedural_only'),(512,'procedural_only'),
    (513,'procedural_only'),(436,'unsupported')])
def test_strict_generation_targets_do_not_replace_host_or_content_gates(tmp_path,monkeypatch,reason_size,status):
    calls=fixture_transport(monkeypatch,check_status=status);post=requests.post
    def bounded_output(url,**kwargs):
        response=post(url,**kwargs);body=response.json()
        function=body['choices'][0]['message']['tool_calls'][0]['function']
        args=json.loads(function['arguments'])
        if function['name']=='submit_capability':args['required_conditions']=args['required_conditions'].ljust(231)
        elif function['name']=='submit_learning':
            args['guidance']=args['guidance'].ljust(248);args['applicability']=args['applicability'].ljust(75)
        else:args['reason']='Executable comparisons only.'.ljust(reason_size)
        function['arguments']=json.dumps(args);response.json=lambda:body
        return response
    monkeypatch.setattr(requests,'post',bounded_output)
    _,out,plan,_=fixture_source(tmp_path,monkeypatch);system=EmpiricalSystem(batch.read(out/'config.json'),harness=AnswerAdapter('livemath',{}))
    try:
        for row in batch.read(out/'source_cases.json'):system.bank.save_case(PublicTask(**row['task']),row['experience'])
        batch.learning_boundary(system,out/'learning/0','candidate:0')
        selected=[t['physical_key'] for stage in plan['selection_tasks'].values() for t in stage]
        unit=plan['candidate_units'][0];log=system.learner.learn_capability_batch(unit,selected)
        assert len(calls)==3 and log['prompt_revision']==cb.PROMPT_REVISION,log
        for wire,field in zip(calls,('required_conditions','guidance','reason')):
            limit=180
            assert wire['tools'][0]['function']['parameters']['properties'][field]['maxLength']==limit
            assert field+' must be 1-'+str(limit)+' Unicode characters' in wire['messages'][0]['content']
        assert cb.CAPABILITY['properties']['required_conditions']['maxLength']==256
        assert cb.PROPOSAL['properties']['guidance']['maxLength']==256
        assert cb.CHECK['properties']['reason']['maxLength']==512
        assert calls[1]['tools'][0]['function']['parameters']['properties']['applicability']['maxLength']==80
        assert len(log['capability']['definition']['required_conditions'])==231
        assert len(log['proposal']['guidance'])==248
        assert len(json.dumps(cb.render(log['proposed_candidate']),ensure_ascii=False))+2==500
        if reason_size>512:
            assert log['decision']=='rejected' and '$.reason' in log['reason'] and 'expected <= 512' in log['reason']
        else:
            assert log['source_check']['status']==status
            assert log['decision']==('candidate' if status=='procedural_only' else 'rejected')
        assert not system.bank.all('skill')
        audit=batch.read(out/'learning/0/requests.json')['requests']
        assert all(q['repair']==0 for q in audit)
        assert system.learner.learn_capability_batch(unit,selected)==log and len(calls)==3
        prior=deepcopy(log);prior['prompt_revision']='procedural-actions.r3'
        system.checkpoint.state['capability_batch_'+log['unit_id']]=prior
        with pytest.raises(ValueError,match='checkpoint prompt revision changed'):
            system.learner.learn_capability_batch(unit,selected)
        assert len(calls)==3
    finally:system.close()


def test_actual_r3_checker_failures_and_billing_are_not_rewritten(monkeypatch):
    value=os.environ.get('LIVEMATH_P2_R3_PAID_RUN')
    if not value:pytest.skip('Set the immutable actual completed P2-R3 run')
    source=Path(value);before={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    monkeypatch.setattr(requests,'post',lambda *a,**k:pytest.fail('Saved response validation may not send HTTP'))
    for i,size in enumerate((380,436)):
        folder=source/'learning'/str(i);log=batch.read(folder/'learning.json')
        q=next(q for q in batch.read(folder/'requests.json')['requests'] if q['purpose']=='guidance_grounding')
        raw=q['response']['tool_calls'][0]['arguments']
        assert raw['status']=='unsupported' and len(raw['reason'])==size and log['decision']=='rejected'
        with pytest.raises(ValueError,match='expected <= 240, actual '+str(size)):
            cb.validate_check(raw,log['proposed_candidate'],prompt_revision='procedural-actions.r3')
        assert cb.validate_check(raw,log['proposed_candidate']) is False
    result=batch.read(source/'result.json')
    assert result['status']=='completed_no_candidate' and result['total_p2_tokens']==58444 and result['total_p2_http']==11
    assert not result['candidates'] and result['paired_tasks']==0
    assert before=={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}
