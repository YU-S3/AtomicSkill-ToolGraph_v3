"""Real Train files and production Provider/scorer; intercepted transport, never paid."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import yaml

from atomic_skillgraph.experiments import run_math_effect_p7 as run
from atomic_skillgraph.empirical import math_effect_contract as c, math_effect_runtime as runtime
from atomic_skillgraph.empirical.contracts import PublicTask, digest
from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.budget_governor import BudgetGovernor
from test_math_effect_p7_contract import proposal
from test_cf2_contracts import response, http
REAL_AUTHORIZE=run.authorize

SOURCE=Path('/home/yangchengyu/livemath_p2_formal_seed42_20261013/livemath/seed42')
DATA=Path('/home/yangchengyu/main_experiment_v1_resources_cf4_r3_20261009_v1')
pytestmark=pytest.mark.skipif(not SOURCE.exists(),reason='Actual immutable local Train60 audit fixture required')


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    settings=yaml.safe_load(run.CONTRACT.read_text());ledgers=[]
    # Snapshot real parallel ledgers. Offline replay retains their charges; no actual run is modified.
    for i,p in enumerate(settings['ledger_sources']):
        target=tmp_path/('ledger_'+str(i)+'.json');run.write_json(target,run.read(p));ledgers.append(str(target))
    settings['ledger_sources']=ledgers
    contract=tmp_path/'contract.yaml';contract.write_text(yaml.safe_dump(settings))
    monkeypatch.setattr(run,'code_identity',lambda:{'git_sha':'offline_fixture_only','source_sha256':'offline','tracked_dirty':False})
    root=tmp_path/'p7';run.prepare(root,contract,SOURCE,DATA)
    # Simulation governor: retain real source accounting in prepare evidence but use a separate test-only ledger.
    # This synthetic override is never written into real P7 output or presented as paid authorization.
    monkeypatch.setattr(run,'authorize',lambda *args:None)
    empty={'limits':settings['cumulative_limits'],'attempts':{},'unknown_billing':False}
    run.write_json(root/'budget/ledger.json',empty);run.write_json(root/'budget/prefix.json',empty)
    return root,run.read(root/'plan.json')


def transport(monkeypatch,root,*,count=1,gate_fails=False):
    calls=[];tasks=run.read(root/'tasks.json');partition=run.read(root/'partition.json')['members']
    records={}
    for split in tasks:
        gold=run.read(DATA/'livemath'/('evaluator_records_'+split+'.json'))
        records.update({t['goal']:(t,gold[t['task_id']]) for t in tasks[split]})
    def post(url,**kwargs):
        wire=deepcopy(kwargs['json']);calls.append(wire);body=json.loads(wire['messages'][1]['content'])
        if wire.get('tools'):
            if isinstance(body,list):
                assert {r['case_id'] for r in body}==set(partition['proposal_train'])
                if not count:value={'decision':'no_change'}
                else:
                    values=[]
                    for i in range(count):
                        v=proposal('global' if i==0 else 'local');v['source_case_ids']=partition['proposal_train'][:2]
                        if i==2:v['applicability']['required_public_indicators']=['comparable_conclusions']
                        values.append(v)
                    value={'decision':'propose','candidates':values}
            else:
                # Correct evaluator answers, old holdout history and gate evidence must never enter Editor.
                assert {r['case_id'] for r in body['paired_feedback']}==set(partition['edit_feedback'])
                assert 'correct_text' not in json.dumps(body) and 'correct_label' not in json.dumps(body)
                value={'decision':'no_change'}
            data=response([value],name='submit_effect')
        else:
            assert 'tools' not in wire and set(body)<={'goal','choices','skills','previous_reasoning_tail'}
            t,record=records[body['goal']];correct=record['correct_choice']['label']
            wrong=next(c['label'] for c in record['choices'] if c['label']!=correct)
            on=bool(body.get('skills')) and not (gate_fails and t['physical_key'] in partition['protected_gate'])
            data=response(content=correct if on else wrong,finish='stop')
        return SimpleNamespace(ok=True,status_code=200,headers={},text='',raise_for_status=lambda:None,json=lambda:data)
    monkeypatch.setenv('MODEL_API_KEY','fixture');monkeypatch.setenv('DEEPSEEK_API_KEY','fixture')
    monkeypatch.setattr('requests.post',post);return calls


@pytest.mark.parametrize('count',[0,1,2,3])
def test_production_flow_and_normal_negative(prepared,monkeypatch,count):
    root,plan=prepared;calls=transport(monkeypatch,root,count=count)
    initial=run.run_learn(root,plan);assert len(initial['candidates'])==count
    selected=run.run_edit(root,plan)
    if not count:
        assert selected['status']=='completed_no_candidate' and len(calls)==1
        with pytest.raises(ValueError):run.run_gate(root,plan)
        return
    assert selected['status']=='feedback_qualified'
    gate=run.run_gate(root,plan);assert gate['status']=='gate_qualified'
    before=len(calls);run.run_gate(root,plan);assert len(calls)==before
    with pytest.raises(ValueError,match='sealed'):run.run_edit(root,plan)
    frozen=run.freeze(root,plan)
    result=run.run_eval(root,plan,'val');assert result['recommend_release']
    assert run.verify_frozen(root)==frozen and not (root/'test').exists()
    if count==1:
        with pytest.raises(ValueError,match='authorization'):REAL_AUTHORIZE(root,'test',plan)
        run.write_json(root/'test_authorization.json',{'plan_hash':digest(plan),'human_instruction':'OFFLINE TEST FIXTURE ONLY',
            'limits':plan['settings']['phase_limits']['test'],'review_identity':run.test_identity(root,plan)})
        REAL_AUTHORIZE(root,'test',plan)
        result=run.run_eval(root,plan,'test');assert len(result['rows'])==100
        assert run.verify_frozen(root)==frozen
    # No release claim uses mock results outside this temporary fixture.
    from atomic_skillgraph.empirical.system import EmpiricalSystem
    cfg=run.read(root/'config.json');cfg['data_dir']=str(root/'frozen_bank')
    s=EmpiricalSystem(cfg,harness=run.AnswerAdapter('livemath',{}))
    assert s.bank.readonly;s.close()


def test_gate_failure_cannot_freeze_or_edit(prepared,monkeypatch):
    root,plan=prepared;transport(monkeypatch,root,gate_fails=True)
    run.run_learn(root,plan);run.run_edit(root,plan)
    assert run.run_gate(root,plan)['status']=='method_not_released'
    with pytest.raises(ValueError):run.freeze(root,plan)
    with pytest.raises(ValueError):run.run_edit(root,plan)


def test_real_saved_truncations_replay(tmp_path,monkeypatch):
    from atomic_skillgraph.harness.benchmarks import AnswerAdapter
    from test_math_effect_p7_runtime import cfg
    history=Path('/home/yangchengyu/livemath_p2_routing_recovery_seed42_20261013/val')
    found=[]
    for path in sorted((history/'traces').glob('*.json')):
        trace=run.read(path)
        for request in trace.get('requests',[]):
            resp=request.get('response',{})
            if request.get('stage')=='runtime' and resp.get('finish_reason')=='length' and not resp.get('content'):
                found.append((trace,resp,request));break
    assert len(found)>=2
    for i,(trace,saved,request) in enumerate(found[:2]):
        task=PublicTask(**trace['task']);record=run.read(DATA/'livemath/evaluator_records_val.json')[task.task_id]
        a=AnswerAdapter('livemath',{task.task_id:record})
        envelopes=[run.read(p) for p in (history/'checkpoints'/task.task_id/'1/responses').glob('*.json')]
        envelope=next(e for e in envelopes if e['request_id']==request['id'])
        raw=response(content=saved['content'],finish=saved['finish_reason'])
        raw['choices'][0]['message']['reasoning_content']=envelope['turn']['reasoning_content']
        assert raw['choices'][0]['message']['reasoning_content']
        raw['usage']=request['http_attempts'][-1]['raw_usage']
        calls=http(monkeypatch,[raw,response(content=task.inputs['choices'][0]['label'],finish='stop')])
        g=BudgetGovernor(tmp_path/str(i)/'budget.json',token_limit=1000000,request_limit=4,finish_reserve=0)
        result=runtime.solve(task,a,[],cfg(),tmp_path/str(i)/'task',g)
        assert result['first_truncated'] and result['valid'] and not result['budget_censored'] and len(calls)==2
        assert calls[0]['max_tokens']==65536 and len(result['raw_usage'])==2
