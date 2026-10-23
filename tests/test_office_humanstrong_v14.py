"""Production-path submission, immutable saved responses and Docker-failure checks."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from atomic_skillgraph.empirical.contracts import PublicTask, program_digest
from atomic_skillgraph.empirical.final_submission import scalar_answer, VERSION
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.empirical.program_worker import ProgramWorker
from atomic_skillgraph.harness.benchmarks import OfficeAdapter
from test_cf2_contracts import http, response
from test_office_human_phase_v3 import load
import test_office_humanstrong_v5 as fixtures

ROOT=Path(__file__).parents[1]/'resources/office_humanstrong_v14'
PRIOR=Path('/home/yangchengyu/officeqa_humanstrong_v13_crossmodel_stream_resume_seed42_20261022')


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr('requests.sessions.Session.request',lambda *a,**kw:pytest.fail('Real HTTP forbidden'))


def setup(tmp_path,monkeypatch):
    monkeypatch.setattr(fixtures,'ROOT',ROOT)
    return fixtures.setup.__wrapped__(tmp_path)


def saved(uid,model='kimi_k26'):
    plan=json.loads((PRIOR/'plan.json').read_text())
    e=next(e for e in plan['episodes'] if e['task_id']=='officeqa:'+uid and e['model_key']==model)
    return json.loads((Path(e['path'])/'traces'/('officeqa:'+uid+'.json')).read_text())


def run_case(cfg,bridge,bundle,task,gold):
    with bridge.bridge():
        system=EmpiricalSystem(cfg,harness=OfficeAdapter({task.task_id:{'answer':gold}},cfg),readonly=True,
            bank_view_factory=lambda b:bridge.HumanBankView(b,bundle))
        try:return system.run_task(task,learn=False)
        finally:system.close()


@pytest.mark.parametrize('text,finish,expected',[
    ('182','stop','182'),('  -1.600%  ','stop','-1.600%'),('907,654','stop','907,654'),
    ('182','length',None),('Answer: 182','stop',None),('182\nReason: ...','stop',None),
    ('1,60','stop',None),('NaN','stop',None),('182 or 183','stop',None)])
def test_only_complete_scalar_is_alternative_submission(text,finish,expected):
    assert scalar_answer(SimpleNamespace(content=text,finish_reason=finish,tool_calls=[]))==expected


def test_original_correct_runtime_scalar_is_submitted_once(tmp_path,monkeypatch):
    cfg,bridge,bundle=setup(tmp_path,monkeypatch);old=saved('UID0070')
    original=old['requests'][0]['response']
    assert original['content']=='182' and original['finish_reason']=='stop' and not original['tool_calls']
    seen=http(monkeypatch,[response(content=original['content'],finish=original['finish_reason'])])
    trace=run_case(cfg,bridge,bundle,PublicTask(**old['task']),'182')
    assert trace['score']['hard'] and trace['execution']['reason']=='agent_submitted'
    assert len(seen)==1 and trace['requests'][0]['normalized_response']['raw_content_preserved']
    assert trace['knowledge_before']==trace['knowledge_after']


def test_rules_and_submission_schema_survive_finalization(tmp_path,monkeypatch):
    cfg,bridge,bundle=setup(tmp_path,monkeypatch);task=PublicTask(**saved('UID0206')['task'])
    adapter=OfficeAdapter({},cfg);adapter.task=task
    seen=http(monkeypatch,[response([{'answer':'9.987%'}],name='finish_answer')])
    with bridge.bridge():
        system=EmpiricalSystem(cfg,harness=adapter,readonly=True,
            bank_view_factory=lambda b:bridge.HumanBankView(b,bundle))
        system.task_context=TaskContext(cfg['runtime']);system.phase='test'
        try:
            prepared=system.prepare_finish({'original_goal':task.goal},'Return final answer.')
            final=system.agent('runtime','Return final answer.',prepared['materials'],None,None,decision_purpose='finish_only')
            assert final=='9.987%' and len(seen)==1
            wire=seen[0];system_text=wire['messages'][0]['content']
            for title in ('answer_only','arc_change'):
                guidance=next(s['guidance'] for s in bundle['skill'] if s['title']==title)
                assert system_text.count(guidance)==1
            assert [t['function']['name'] for t in wire['tools']]==['finish_answer']
            assert len(json.dumps(wire,ensure_ascii=False,separators=(',',':')).encode())<=prepared['input_max_bytes']
            assert wire['thinking']['type']=='enabled'
        finally:system.close()


def test_real_saved_math_source_runs_in_docker(tmp_path,monkeypatch):
    cfg,bridge,bundle=setup(tmp_path,monkeypatch);old=saved('UID0053')
    source=old['tools'][0]['arguments']['source']
    seen=http(monkeypatch,[response([{'action':'call_tool','name':'execute_python','arguments':{'source':source}}]),
                           response([{'action':'finish','answer':'23.587'}])])
    trace=run_case(cfg,bridge,bundle,PublicTask(**old['task']),'23.587')
    result=trace['tools'][0]['result']
    assert result['accepted'] and result['data']['computed']['absolute_difference_rounded']==23.587
    assert trace['score']['hard'] and len(seen)==2


def test_docker_disconnect_stops_before_another_model_turn(tmp_path,monkeypatch):
    cfg,bridge,bundle=setup(tmp_path,monkeypatch);old=saved('UID0053')
    seen=http(monkeypatch,[response([{'action':'call_tool','name':'execute_python',
        'arguments':{'source':old['tools'][0]['arguments']['source']}}])])
    monkeypatch.setattr(ProgramWorker,'command',lambda *a,**kw:['python3','-c',
        'import sys;sys.stderr.write("Cannot connect to the Docker daemon\\n");sys.exit(125)'])
    with pytest.raises(Exception,match='Docker control failed'):
        run_case(cfg,bridge,bundle,PublicTask(**old['task']),'23.587')
    assert len(seen)==1


def test_native_bank_links_and_scoped_fact_cards(tmp_path,monkeypatch):
    cfg,bridge,bundle=setup(tmp_path,monkeypatch)
    assert cfg['runtime']['final_submission_version']==VERSION
    for p in json.loads((ROOT/'bank/bank.json').read_text())['program']:
        assert p['id']=='program_'+program_digest(p)
        assert p['source']==(ROOT/p['source_file']).read_text()
    task=PublicTask(**saved('UID0140')['task']);titles=bridge.selected_titles(task)
    assert 'ffo_public_layout' in titles and 'table_block_binding' in titles
    assert 'arc_change' in bridge.selected_titles(PublicTask(**saved('UID0206')['task']))
    assert 'arc_change' in bridge.selected_titles(PublicTask(**saved('UID0101')['task']))
    titles=bridge.selected_titles(PublicTask(**saved('UID0118')['task']))
    assert 'median_absolute_deviation' in titles and 'median_deviation' not in titles
    assert 'percent_difference' in bridge.selected_titles(PublicTask(**saved('UID0220')['task']))
    for path in [ROOT/'cards.json',*ROOT.rglob('*.py')]:assert 'UID0' not in path.read_text()
    assert all(x['skill_id'] in {s['id'] for s in bundle['skill']} for x in bundle['implementation'])


@pytest.mark.parametrize('value,places,fixed,expected',[
    (-118255.5,2,False,'-118255.5'),(12.6,2,False,'12.6'),(1.6,3,True,'1.600'),
    (27.025,2,False,'27.03'),(-0.0001,2,False,'0')])
def test_program_rounding_separates_precision_from_fixed_width(value,places,fixed,expected):
    program=load(ROOT/'programs/scalar_display.py','v14_render')
    assert program.render(value,'quantity','quantity','number',places,'amount',fixed_decimals=fixed)==expected


@pytest.mark.parametrize('model',['deepseek','kimi_k26','glm53_flash'])
def test_three_provider_final_submission_wire(tmp_path,monkeypatch,model):
    cfg,bridge,bundle=setup(tmp_path,monkeypatch)
    sys.path.insert(0,str(ROOT.parents[1]/'experiments'))
    import run_office_humanstrong_v14 as runner
    from test_office_v13_crossmodel import Stream
    cfg=runner.config_for(model,fixtures.RESOURCES)|{'data_dir':cfg['data_dir'],
        'experiment':{**cfg['experiment'],'output_dir':str(tmp_path/'run')}}
    for key in ('MODEL_API_KEY','MOONSHOT_API_KEY','ZHIPU_API_KEY'):monkeypatch.setenv(key,'offline-fixture')
    seen=[]
    def post(*a,**kw):
        wire=kw['json'];seen.append(wire)
        if cfg['llm'].get('stream'):
            reply=Stream(wire['model'],{'answer':'5'})
            reply.chunks[0]['choices'][0]['delta']['tool_calls'][0]['function']['name']='finish_answer'
            return reply
        return SimpleNamespace(status_code=200,ok=True,headers={},json=lambda:response([{'answer':'5'}],name='finish_answer'))
    monkeypatch.setattr('requests.post',post)
    task=PublicTask(**saved('UID0070')['task']);adapter=OfficeAdapter({},cfg);adapter.task=task
    with bridge.bridge():
        system=EmpiricalSystem(cfg,harness=adapter,readonly=True,
            bank_view_factory=lambda b:bridge.HumanBankView(b,bundle))
        system.task_context=TaskContext(cfg['runtime']);system.phase='test'
        try:
            prompt='Do not call tools, execute programs, read more results or replan.'
            prepared=system.prepare_finish({'original_goal':task.goal},prompt)
            assert system.agent('runtime',prompt,prepared['materials'],None,None,decision_purpose='finish_only')=='5'
        finally:system.close()
    assert len(seen)==1 and seen[0]['tools'][0]['function']['name']=='finish_answer'
    from atomic_skillgraph.empirical.budget_governor import input_token_bound
    assert input_token_bound(seen[0])==prepared['serialized_input_bytes']
    assert seen[0]['thinking']['type']=='enabled' and seen[0]['max_tokens']==32768
    assert 'Do not call tools' not in seen[0]['messages'][0]['content']


def test_all_test_inputs_admitted_without_private_scores(tmp_path,monkeypatch):
    cfg,bridge,bundle=setup(tmp_path,monkeypatch)
    rows=json.loads((fixtures.RESOURCES/'test.json').read_text())['tasks'];sent=[]
    def post(*args,**kwargs):
        wire=kwargs['json'];sent.append(wire)
        name=wire['tools'][0]['function']['name']
        action={'answer':'offline fixture'} if name=='finish_answer' else {'action':'finish','answer':'offline fixture'}
        return SimpleNamespace(status_code=200,ok=True,headers={},json=lambda:response([action],name=name))
    monkeypatch.setenv('MODEL_API_KEY','test-only-key')
    monkeypatch.setattr('atomic_skillgraph.agents.provider.requests.post',post)
    for row in rows:
        trace=run_case(cfg,bridge,bundle,PublicTask(**row),'PRIVATE_TEST_GOLD_SENTINEL')
        assert trace['solve_status']=='completed' and not trace['score']['hard']
    assert len(sent)==102
    assert all('PRIVATE_TEST_GOLD_SENTINEL' not in json.dumps(wire) for wire in sent)
    assert all(wire['tools'] for wire in sent)


def test_final_scalar_program_submits_without_model_rewrite(tmp_path,monkeypatch):
    cfg,bridge,bundle=setup(tmp_path,monkeypatch)
    old=fixtures.v2.fixture_task(bridge)
    task=PublicTask(**{**old.__dict__,'goal':'Report the final amount with exactly three digits after the decimal point.',
                      'inputs':{**old.inputs,'source_pages':[]}})
    sent=[]
    def post(*args,**kwargs):
        wire=kwargs['json'];sent.append(wire);body=json.loads(wire['messages'][1]['content'])
        if len(sent)==1:
            action={'action':'call_tool','name':'execute_python','arguments':{'source':'import json\nprint(json.dumps({"final_amount":1.6}))'}}
        else:
            assert len(sent)==2
            ref=body['acquired_result_refs']['numeric_refs'][0]['value_ref']
            assert len(body['calls']['programs'])==1
            action={'action':'call_program','name':bundle['program'][0]['id'],'arguments':{
                'submit_answer':True,'value_ref':ref,'source_scale':'quantity','unit':'widgets',
                'format':{'output_scale':'quantity','style':'number','decimals':3,'fixed_decimals':True}}}
        return SimpleNamespace(status_code=200,ok=True,headers={},json=lambda:response([action]))
    monkeypatch.setenv('MODEL_API_KEY','test-only-key')
    monkeypatch.setattr('atomic_skillgraph.agents.provider.requests.post',post)
    trace=run_case(cfg,bridge,bundle,task,'1.600')
    assert trace['score']['hard'] and len(sent)==2
    assert trace['execution']['reason']=='plan_submitted'
    assert trace['execution']['attempts'][0]['submission_by_program']


def test_three_model_plan_carries_costs_and_rejects_insufficient_ceiling(tmp_path,monkeypatch):
    sys.path.insert(0,str(ROOT.parents[1]/'experiments'))
    import run_office_humanstrong_v14 as runner
    for obj,names in [(runner.common,('PACKAGE','frontier','VERSION')),(runner.previous,('MODELS',))]:
        for name in names:monkeypatch.setattr(obj,name,getattr(obj,name))
    runner.bind_runner();common=runner.common
    identity=lambda:{'git_sha':'offline-fixture','tracked_dirty':False,'source_sha256':'offline-fixture'}
    monkeypatch.setattr(common,'code_identity',identity)
    before=common.sha(runner.PARENT/'budget.json');amount=runner.preview()
    assert amount['episodes']==306 and amount['new_token_cap']==66272256
    with pytest.raises(ValueError,match='Insufficient'):
        runner.prepare(tmp_path/'not_created',262000000)
    assert not (tmp_path/'not_created').exists()
    root=tmp_path/'plan';runner.prepare(root,amount['cumulative_max'])
    plan=common.verify(root)
    assert len(plan['episodes'])==306 and len({e['task_id'] for e in plan['episodes']})==102
    assert {e['model_key'] for e in plan['episodes']}==set(runner.MODELS)
    assert common.sha(runner.PARENT/'budget.json')==before
    old=common.read(runner.PARENT/'budget.json')
    assert common.read(root/'budget.json')['attempts']==old['attempts']
    assert plan['development_exposed'] and plan['test_runs_authorized_in_this_plan']==3
    for model in runner.MODELS:
        cfg=common.read(root/(model+'_config.json'))
        assert cfg['runtime']['final_submission_version']==VERSION
        assert cfg['llm']['purpose_overrides']['finish_only']['protocol']['thinking_type']=='enabled'
        assert cfg['llm']['runtime']['max_total_tokens_per_task']==216576
