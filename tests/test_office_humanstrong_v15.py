"""Saved failures and real Docker, with zero external model calls."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
from atomic_skillgraph.empirical.contracts import PublicTask, program_digest
from atomic_skillgraph.empirical.final_submission import plain_answer, VERSION, LIST_VERSION
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.harness.benchmarks import OfficeAdapter
from test_cf2_contracts import http, response
from test_office_human_phase_v3 import load
import test_office_humanstrong_v5 as fixtures

ROOT=Path(__file__).parents[1]/'resources/office_humanstrong_v15'
RUN=Path('/home/yangchengyu/officeqa_humanstrong_v14_three_model_test102_seed42_20261023')


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr('requests.sessions.Session.request',lambda *a,**kw:pytest.fail('Real HTTP forbidden'))


def setup(tmp_path,monkeypatch):
    monkeypatch.setattr(fixtures,'ROOT',ROOT)
    return fixtures.setup.__wrapped__(tmp_path)


def saved(uid,model='kimi_k26'):
    plan=json.loads((RUN/'plan.json').read_text())
    e=next(e for e in plan['episodes'] if e['task_id']=='officeqa:'+uid and e['model_key']==model)
    return json.loads((Path(e['path'])/'summary.json').read_text())['cases'][0]


@pytest.mark.parametrize('text,expected',[
    ('[1444, 3174]','[1444, 3174]'),('[7.1, 82]','[7.1, 82]'),
    ('[37.48, unusual]','[37.48, unusual]'),('Answer: [7.1, 82]',None),
    ('[7.1, 82] because ...',None),('[[1], [2]]',None),('[NaN, 4]',None),
    ('[7.1, ]',None)])
def test_whole_list_submission_is_opt_in(text,expected):
    turn=SimpleNamespace(content=text,finish_reason='stop',tool_calls=[])
    assert plain_answer(turn,{'final_submission_version':VERSION}) is None
    assert plain_answer(turn,{'final_submission_version':LIST_VERSION})==expected
    turn.finish_reason='length'
    assert plain_answer(turn,{'final_submission_version':LIST_VERSION}) is None


@pytest.mark.parametrize('uid',['UID0232','UID0124'])
@pytest.mark.parametrize('final',[False,True])
def test_actual_correct_list_is_not_discarded(tmp_path,monkeypatch,uid,final):
    cfg,bridge,bundle=setup(tmp_path,monkeypatch);old=saved(uid)
    turn=old['requests'][-1]['response'];gold=old['scoring_audit']['raw_scorer_output']['gold_answer']
    assert turn['content']==gold and turn['finish_reason']=='stop' and old['execution']['prediction']==''
    seen=http(monkeypatch,[response(content=turn['content'],finish='stop')])
    task=PublicTask(**old['task']);adapter=OfficeAdapter({task.task_id:{'answer':gold}},cfg);adapter.task=task
    with bridge.bridge():
        system=EmpiricalSystem(cfg,harness=adapter,readonly=True,bank_view_factory=lambda b:bridge.HumanBankView(b,bundle))
        try:
            if final:
                system.task_context=TaskContext(cfg['runtime']);system.phase='test'
                answer=system.agent('runtime','Return the final answer.',{'original_goal':task.goal},None,None,decision_purpose='finish_only')
                assert answer==gold
            else:
                trace=system.run_task(task,learn=False)
                assert trace['score']['hard'] and trace['execution']['reason']=='agent_submitted',trace['execution']
                assert trace['requests'][0]['normalized_response']['raw_content_preserved']
        finally:system.close()
    assert len(seen)==1


CASES=[
    ('UID0237','midpoint_submit',{'values':[4034853,4156876],'unit':'millions USD','absolute':True,'output_percent':False,'decimals':[2]},'0.03'),
    ('UID0101','growth_submit',{'values':[135030,35810],'unit':'millions USD','intervals':8,'decimals':[3,3,3]},'[-0.153, 0.847, -1.162]'),
    ('UID0123','paired_statistics',{'values':[1186,1317,1439],'other_values':[3839,3636,3981],
        'unit':'thousands tons','periods':['1941-01','1941-02','1941-03'],'expected_periods':['1941-01','1941-02','1941-03'],
        'statistic':'share_and_correlation','decimals':[1,3]},'[34.4, 0.391]'),
    ('UID0202','baseline_z',{'values':[1.3,1.4],'unit':'millions USD increase','target':4,
        'spread':'sample','two_sided':False,'threshold':1.96,'decimals':[2]},'[37.48, unusual]')]


@pytest.mark.parametrize('uid,operation,arguments,expected',CASES)
def test_native_implementation_calculates_and_submits_in_docker(tmp_path,monkeypatch,uid,operation,arguments,expected):
    cfg,bridge,bundle=setup(tmp_path,monkeypatch);old=saved(uid,'deepseek');task=PublicTask(**old['task'])
    program=next(p for p in bundle['program'] if p['operation']==operation)
    seen=http(monkeypatch,[response([{'action':'call_program','name':program['id'],
        'arguments':{'operation':operation,'submit_answer':True,**arguments}}])])
    with bridge.bridge():
        system=EmpiricalSystem(cfg,harness=OfficeAdapter({task.task_id:{'answer':expected}},cfg),readonly=True,
            bank_view_factory=lambda b:bridge.HumanBankView(b,bundle))
        try:trace=system.run_task(task,learn=False)
        finally:system.close()
    assert trace['score']['hard'],trace['execution']
    assert len(seen)==1 and trace['execution']['reason']=='plan_submitted'
    attempt=trace['execution']['attempts'][0]
    assert attempt['submission_by_program'] and attempt['worker_status']=='ok'
    assert trace['knowledge_before']==trace['knowledge_after']
    assert program['id'] in {p['id'] for p in json.loads(seen[0]['messages'][1]['content'])['calls']['programs']}


def test_paired_period_mask_cannot_silently_expand():
    module=load(ROOT/'programs/verified_math.py','v15_math')
    args={'operation':'paired_statistics','submit_answer':True,**deepcopy(CASES[2][2])}
    args['periods'].append('1941-04');args['values'].append(1386);args['other_values'].append(4605)
    assert module.run(None,args)['status']=='needs_input'
    args=deepcopy(CASES[0][2]);args.update(operation='midpoint_submit',submit_answer=True,percent_suffix=True)
    assert module.run(None,args)['status']=='needs_input'


def test_scoped_rule_conflicts_and_identity():
    bridge=load(ROOT/'integration/frontier.py','v15_selector')
    tasks=json.loads((RUN/'tasks.json').read_text())['tasks'];bank=json.loads((ROOT/'bank/bank.json').read_text())
    for task in tasks:
        titles=bridge.selected_titles(PublicTask(**task))
        assert len(titles)<=6
        assert not {'normalized_midpoint','arc_change'}<=set(titles)
        assert not {'endpoint_growth','arc_change'}<=set(titles)
    for p in bank['program']:
        assert p['id']=='program_'+program_digest(p)
        assert p['source']==(ROOT/p['source_file']).read_text()
    for p in [ROOT/'cards.json',*ROOT.rglob('*.py')]:assert 'UID0' not in p.read_text()
    assert len(bank['implementation'])==5


@pytest.mark.parametrize('model',['deepseek','kimi_k26','glm53_flash','minimax_m3','gemini38_flash','qwen38_flash'])
@pytest.mark.parametrize('final',[False,True])
def test_existing_six_provider_configurations_keep_final_lists(tmp_path,monkeypatch,model,final):
    sys.path.insert(0,str(ROOT.parents[1]/'experiments'))
    import run_office_humanstrong_v15 as runner
    from test_office_v13_crossmodel import Stream
    cfg,bridge,bundle=setup(tmp_path,monkeypatch)
    configured=runner.config_for(model,fixtures.RESOURCES)
    configured['data_dir']=cfg['data_dir'];configured['experiment']['output_dir']=cfg['experiment']['output_dir'];cfg=configured
    monkeypatch.setenv(cfg['llm']['api_key_env'],'offline-fixture')
    seen=[]
    def post(*a,**kw):
        wire=kw['json'];seen.append(wire)
        if cfg['llm'].get('stream'):return Stream(wire['model'],content='[7.1, 82]')
        return SimpleNamespace(status_code=200,ok=True,headers={},json=lambda:response(content='[7.1, 82]',finish='stop'))
    monkeypatch.setattr('requests.post',post)
    task=PublicTask(**saved('UID0124')['task']);adapter=OfficeAdapter({task.task_id:{'answer':'[7.1, 82]'}},cfg);adapter.task=task
    with bridge.bridge():
        system=EmpiricalSystem(cfg,harness=adapter,readonly=True,bank_view_factory=lambda b:bridge.HumanBankView(b,bundle))
        system.task_context=TaskContext(cfg['runtime']);system.phase='test'
        system.request_attribution={'parent_scope':'fixture','parent_token_limit':runner.TASK_CAPS[model]}
        try:
            if final:
                prepared=system.prepare_finish({'original_goal':task.goal},'Return the final answer.')
                assert system.agent('runtime','Return the final answer.',prepared['materials'],None,None,decision_purpose='finish_only')=='[7.1, 82]'
            else:
                trace=system.run_task(task,learn=False)
                assert trace['score']['hard'],trace['execution']
            entries=list(system.budget_governor.state['attempts'].values())
            assert len(entries)==1 and entries[0]['status']=='metered'
            entry=entries[0]
            assert entry['reserved_tokens']+entry['context']['parent_finish_reserve']<=runner.TASK_CAPS[model]
            if model=='qwen38_flash':
                assert entry['reserved_tokens']-entry['input_estimate']==seen[0]['max_tokens']+131072+10
                if not final:assert entry['context']['parent_finish_reserve']>=32768+131072+10
        finally:system.close()
    assert len(seen)==1 and seen[0]['model']==cfg['llm']['model']
    assert [t['function']['name'] for t in seen[0]['tools']]==(['finish_answer'] if final else ['runtime_step'])
    assert cfg['llm']['runtime']['max_total_tokens_per_task']==runner.TASK_CAPS[model]
    if model=='qwen38_flash':assert seen[0]['thinking_budget']==131072 and 'reasoning_effort' not in seen[0]
    if model=='gemini38_flash':assert cfg['llm']['base_url']=='https://api.rootflowai.com/v1'


def test_six_model_controller_preserves_parent_and_task_caps(tmp_path,monkeypatch):
    sys.path.insert(0,str(ROOT.parents[1]/'experiments'))
    import run_office_humanstrong_v15 as runner
    from test_office_v13_crossmodel import Stream
    # Use an already settled ledger; the ongoing V14 run remains untouched.
    read=runner.common.read;v14_config=read(runner.PARENT/'base_config.json')
    monkeypatch.setattr(runner,'PARENT',runner.previous.PARENT)
    monkeypatch.setattr(runner.common,'read',lambda p:deepcopy(v14_config) if Path(p)==runner.PARENT/'base_config.json' else read(p))
    settled=runner.previous.previous.fixed.paired.settled_parent
    def isolated_parent(*a,**kw):
        plan,ledger,_=settled(*a,**kw)
        return plan,ledger,[str(tmp_path/'offline-ledger.lock')]
    monkeypatch.setattr(runner.previous.previous.fixed.paired,'settled_parent',isolated_parent)
    runner.bind_runner();c=runner.common
    identity=lambda:{'git_sha':'offline-fixture','tracked_dirty':False,'source_sha256':'offline-fixture'}
    monkeypatch.setattr(c,'code_identity',identity)
    monkeypatch.setattr('atomic_skillgraph.experiments.run_empirical.code_identity',identity)
    monkeypatch.setattr(c,'load_env',lambda _:None)
    for model in runner.MODELS:
        monkeypatch.setenv(runner.config_for(model,fixtures.RESOURCES)['llm']['api_key_env'],'offline-fixture')
    parent_hash=c.sha(runner.PARENT/'budget.json');root=tmp_path/'campaign'
    runner.prepare(root);plan=c.verify(root)
    assert len(plan['episodes'])==72 and plan['new_token_cap']==17794560
    assert plan['ledger_limits']['token_limit']==plan['inherited_used_or_reserved']+17794560
    sent=[]
    def post(*a,**kw):
        wire=kw['json'];sent.append(wire)
        if wire.get('stream'):return Stream(wire['model'],content='0.03')
        return SimpleNamespace(status_code=200,ok=True,headers={},json=lambda:response(content='0.03',finish='stop'))
    monkeypatch.setattr('requests.post',post)
    assert runner.previous.previous.execute(root,'unused')==0
    result=runner.previous.previous.status(root)
    assert result['new_http']==72 and result['new_unsettled']==0 and len(sent)==72
    assert all(row['completed']==12 for row in result['models'].values())
    assert c.sha(runner.PARENT/'budget.json')==parent_hash
    for episode in plan['episodes']:
        cfg=c.read(Path(episode['path'])/'config.json')
        assert cfg['budget']['eval_task_tokens']==runner.TASK_CAPS[episode['model_key']]
        assert c.read(Path(episode['path'])/'freeze_record.json')['unchanged']
