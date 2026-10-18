"""Manual bridge protocol regression; real archived materials, intercepted transport only."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from atomic_skillgraph.empirical.budget_governor import BudgetGovernor, input_token_bound
from atomic_skillgraph.empirical.contracts import PublicTask
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.experiments.formal_log import tree_identity
from atomic_skillgraph.harness.benchmarks import OfficeAdapter
from test_cf2_contracts import http, response


def module(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


@pytest.fixture(scope='module')
def frontier(tmp_path_factory):
    source=os.environ.get('HUMAN_FRONTIER_ORIGINAL')
    if not source:pytest.skip('Set HUMAN_FRONTIER_ORIGINAL to the immutable supplied package')
    patcher=module(Path(__file__).parents[1]/'experiments/patch_human_frontier.py','frontier_patcher')
    target=tmp_path_factory.mktemp('frontier')/'package'
    patcher.patch(source,target)
    original=Path(source)
    assert tree_identity(original/'bank')==tree_identity(target/'bank')
    assert {p.name:p.read_bytes() for p in (original/'programs').glob('*.py')}=={p.name:p.read_bytes() for p in (target/'programs').glob('*.py')}
    assert tree_identity(original/'skills')==tree_identity(target/'skills')
    return module(target/'integration/frontier.py','frontier_under_test')


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args,**kwargs):raise AssertionError('Real HTTP forbidden in offline regression')
    monkeypatch.setattr('requests.sessions.Session.request',refuse)


@pytest.mark.parametrize('name,purpose',[(None,'finish_only'),('finish_answer',None)])
def test_finish_is_exact_passthrough(frontier,monkeypatch,name,purpose):
    seen=[]
    monkeypatch.setattr(EmpiricalSystem,'agent',lambda *a,**kw:seen.append((a,kw)))
    view=frontier.HumanBankView.__new__(frontier.HumanBankView)
    system=SimpleNamespace(bank=view)
    materials={'finish_evidence':[{'content':'already acquired result'}]}
    kwargs={'decision_purpose':purpose} if purpose else {}
    with frontier.bridge(SimpleNamespace()):
        EmpiricalSystem.agent(system,'runtime','native final contract',materials,name,None,**kwargs)
    args,actual_kwargs=seen[0]
    assert args[2]=='native final contract' and args[3] is materials and args[4]==name
    assert actual_kwargs==kwargs


def test_runtime_preserves_bank_but_native_protocol_is_last(frontier,monkeypatch):
    seen=[]
    monkeypatch.setattr(EmpiricalSystem,'agent',lambda *a,**kw:seen.append((a,kw)))
    skill={'id':'authored','execution_intent':'guidance_only','applicability':{'selector':'always'},
           'goal':'format','guidance':'Output one <answer>...</answer>.'}
    bank=SimpleNamespace(readonly=True,all=lambda kind:[skill] if kind=='skill' else [],environment=None)
    view=frontier.HumanBankView(bank,{'benchmark':'officeqa','method_version':'fixture'})
    task=PublicTask('fixture','fixture','Calculate',{},'test')
    system=SimpleNamespace(bank=view,adapter=SimpleNamespace(task=task))
    material={'calls':{'programs':[]}}
    with frontier.bridge(SimpleNamespace()):
        EmpiricalSystem.agent(system,'runtime','Use runtime_step; answer is plain text.',material,'runtime_step',{})
    args,_=seen[0]
    assert 'Active authored SkillBank' in args[2] and '<answer>' in args[2]
    assert args[2].endswith('Use runtime_step; answer is plain text.')
    assert 'current request governs the response protocol' in args[2]
    assert args[3]['human_frontier_skill_ids']==['authored'] and material=={'calls':{'programs':[]}}


def test_all_saved_finish_refusals_reach_native_transport(frontier,tmp_path,monkeypatch):
    archive=os.environ.get('HUMAN_FRONTIER_ARCHIVE')
    if not archive:pytest.skip('Set HUMAN_FRONTIER_ARCHIVE to the immutable failed run')
    root=Path(archive);read=frontier.read
    failed=[read(p) for p in sorted((root/'test/traces').glob('*.json'))
            if read(p)['execution']['reason']=='finish_only_protocol_error']
    assert len(failed)==32
    before=tree_identity(root/'test/bank');records=[]
    for trace in failed:
        task=PublicTask(**trace['task']);cfg=deepcopy(read(root/'config.json'))
        out=tmp_path/task.task_id.replace(':','_');cfg['experiment']['output_dir']=str(out)
        cfg['data_dir']=str(root/'test/bank')
        adapter=OfficeAdapter({task.task_id:{'answer':'offline-placeholder'}},cfg);adapter.reset(task)
        bundle=read(root/'test/bank/imported_bank.json')
        governor=BudgetGovernor(out/'budget.json',token_limit=192000,request_limit=1,finish_reserve=0)
        state=read(root/'test/checkpoints'/task.task_id/'1/state.json')
        evidence=state['finish_prepared']['evidence'];materials=deepcopy(evidence['materials'])
        request=next(q for q in trace['requests'] if q['purpose']=='finish_only')
        assert request['admission_status']=='local_refusal' and not request.get('http_attempts')
        prompt=request['messages'][0]['content'].split('\n\n## Active authored SkillBank\n')[0]
        with frontier.bridge(SimpleNamespace()):
            system=EmpiricalSystem(cfg,harness=adapter,readonly=True,budget_governor=governor,
                bank_view_factory=lambda b:frontier.HumanBankView(b,bundle))
            system.phase='test';system.task_context=TaskContext(cfg['runtime'])
            for k,v in state['executor_state']['context'].items():setattr(system.task_context,k,deepcopy(v))
            try:
                sent=http(monkeypatch,[response(content='offline-fixture-answer',finish='stop')])
                value=system.agent('runtime',prompt,materials,None,None,decision_purpose='finish_only')
                assert value=='offline-fixture-answer' and len(sent)==1
                payload=sent[0]
                assert not payload.get('tools') and payload['max_tokens']==2048
                assert payload['thinking']['type']=='disabled'
                assert payload['messages'][0]['content']==prompt
                assert json.loads(payload['messages'][1]['content'])==evidence['materials']
                assert input_token_bound(payload)==evidence['serialized_input_bytes']<=16384
                assert len(governor.state['attempts'])==1 and not governor.state['unknown_billing']
                records.append({'task':task.task_id,'original_bytes':request['material_refusal']['wire_bytes'],
                    'fixed_bytes':input_token_bound(payload),'evidence_unchanged':True,'model_http':0})
            finally:system.close()
    assert tree_identity(root/'test/bank')==before
    output=os.environ.get('HUMAN_FRONTIER_EVIDENCE')
    if output:
        Path(output).write_text(json.dumps({'passed':True,'model_http':0,'saved_refusals':records},indent=2)+'\n')
