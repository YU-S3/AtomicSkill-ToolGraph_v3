"""Regression for actual Office return types, final submission and lost learning sources."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from atomic_skillgraph.empirical import LEARNING_MATERIAL_VERSION
from atomic_skillgraph.empirical.contracts import PublicTask, validate_schema_instance
from atomic_skillgraph.empirical.model_view import learning_case, expand_material
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.harness.benchmarks import OfficeAdapter, score_answer
from atomic_skillgraph.harness.simple_protocol import Broker
from skillcompiler_bench_contracts.answer import answer_contract
from test_cf2_contracts import office, http, response
from test_empirical import worker


def test_declared_rpc_types_match_real_docker(tmp_path, worker):
    s=office(tmp_path);s.adapter.worker=worker
    broker=Broker(s.adapter,24)
    try:
        specs={t['name']:t for t in broker.available_tools()}
        for name,args in [('grep',{'pattern':'target'}),('read',{'path':'a.txt','offset':0}),
                          ('grep',{'pattern':'absent'}),('read',{'path':'missing.txt','offset':0})]:
            validate_schema_instance(broker.call(name,args),specs[name]['result_schema'])
        result=broker.call('execute_python',{'source':'''import json
hits=ctx.call('grep',{'pattern':'target'})['data']
assert isinstance(hits[0]['line'],int) and isinstance(hits[0]['text'],str)
page=ctx.call('read',{'path':hits[0]['path'],'offset':hits[0]['offset']})['data']
assert isinstance(page['text'],str)
print(json.dumps({'text':page['text'],'next_offset':page['next_offset']}))
'''})
        assert result['accepted'] and 'target' in result['data']['computed']['text']
    finally:s.close()


def test_confirmed_answer_is_submitted_once_without_rechecking(tmp_path,monkeypatch):
    s=office(tmp_path)
    sent=http(monkeypatch,[response([{'mode':'compose','workflow':{'goal':'answer','nodes':[
        {'id':'n','execution_mode':'dynamic','goal':'answer','args':{}}],'outputs':{}}}],name='submit_plan'),
        response([{'action':'finish','answer':'14.04%'}])])
    try:
        task=s.adapter.task
        from atomic_skillgraph.empirical.task_context import TaskContext
        broker=Broker(s.adapter,24,context=TaskContext(s.config['runtime']))
        result=s.executor.run(task,s.adapter,broker,s.planner.plan(task,s.adapter))
        assert result['prediction']=='14.04%' and result['reason']=='agent_submitted'
        assert len(sent)==2 and result['finish_trigger'] is None and not result['history']
        assert 'plain text' in answer_contract('officeqa',{})
        assert 'No answer tags' in answer_contract('officeqa',{})
        assert score_answer('officeqa','14.04%',{'answer':'14.04%'})['hard']
        assert not score_answer('officeqa','&lt;answer&gt;14.04%&lt;/answer&gt;',{'answer':'14.04%'})['hard']
    finally:s.close()


def test_latest_completed_chain_and_whole_dictionary_remain_selectable():
    public={'goal':'read and summarize','inputs':{}}
    def evidence(key,source,value):
        return {'id':key,'kind':'native_operation','public_task':public,
                'operation':{'name':'execute_python','arguments':{'source':source}},
                'context_result_protocols':['native.text-page.v1'],
                'reference':{'accepted':True,'data':{'computed':value}},
                'executed_calls':[{'name':'read','result':{'accepted':True}}]}
    values={str(i):list(range(12)) for i in range(18)}
    prefix="page=ctx.call('read',{'path':'a.txt','offset':0})\n"
    first=evidence('local:first',prefix+"print('probe')",{'probe':'early'})
    last=evidence('local:last',prefix+'values='+repr(values)+'\nprint(sum(map(sum,values.values())))',{'total':1188})
    selected=learning_case({'task':public,'local_evidence':[first,last]},'case')['local_evidence']
    assert selected[0]['local_evidence_ref']=='local:last'
    refs=selected[0]['input_sources'];aggregate=next(r for r in refs if r['value']==values)
    assert aggregate['path']==['body',1,'value']
    assert not any(r['path'][:3]==aggregate['path'] and len(r['path'])>3 for r in refs)


def test_all_saved_prefix_learning_requests_fit_and_keep_actual_sources(tmp_path,monkeypatch,record_property):
    root=os.environ.get('OFFICE_INTERFACE_ARCHIVE')
    if not root:pytest.skip('Set the stopped Office prefix for zero-HTTP production replay')
    root=Path(root);config=json.loads((root/'train/config.json').read_text())
    records=json.loads(Path(config['harness']['evaluator_records']).read_text())
    traces=[json.loads(p.read_text()) for p in sorted((root/'train/traces').glob('*.json'))]
    expected={'officeqa:UID0115':'8ee1128e13b84fef9ce82dcabe07ccb4:call_00_a8y5tsRPzSNBXHqWuzyx4478',
              'officeqa:UID0148':'e76c4a0f2fbc420f9e822c0a52eca591:call_00_EO7iuNhDHV7Bzj2p2Q3D0471'}
    sent=http(monkeypatch,[response([{'decision':'no_change'}],name='submit_learning')]*len(traces))
    sizes=[]
    for i,trace in enumerate(traces):
        cfg=deepcopy(config);cfg['data_dir']=str(tmp_path/str(i)/'bank');cfg['experiment']['output_dir']=str(tmp_path/str(i))
        cfg['learning']['material_version']=LEARNING_MATERIAL_VERSION
        s=EmpiricalSystem(cfg,harness=OfficeAdapter(records,cfg));s.phase='train'
        task=PublicTask(**trace['task']);s.adapter.reset(task);s.budget_scope=task.task_id
        try:
            result=s.learner.learn(task,trace)
            assert result['decision']=='no_change' and not result['errors'], (task.task_id,result)
            shown=expand_material(json.loads(sent[-1]['messages'][1]['content']))
            cards=shown['experience']['local_evidence']
            if task.task_id in expected:assert cards[0]['local_evidence_ref'].endswith(expected[task.task_id])
            from atomic_skillgraph.experiments.audit_officeqa_prefix import verify_directory,saved_case_directory
            verify_directory(shown,{c['local_evidence_ref']:c for c in saved_case_directory(trace,cfg)})
            from atomic_skillgraph.empirical.budget_governor import input_token_bound
            size=input_token_bound(sent[-1]);assert size<=16384
            sizes.append({'task_id':task.task_id,'wire_bytes':size,'source_ids':[c['local_evidence_ref'] for c in cards]})
            assert not s.bank.all('program')
        finally:s.close()
    assert len(sent)==len(traces)
    record_property('saved_prefix_requests',json.dumps(sizes))
