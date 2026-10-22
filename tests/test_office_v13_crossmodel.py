"""Streamed production requests, frozen Bank identity and shared campaign billing."""
import json
from pathlib import Path
import sys
import pytest
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.empirical.contracts import PublicTask
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.harness.benchmarks import OfficeAdapter
from atomic_skillgraph.agents.provider import OpenAICompatibleConfig,OpenAICompatibleProvider,AgentProviderError
from test_office_humanstrong_v13 import setup
from test_office_humanstrong_v5 import v2
sys.path.insert(0,str(Path(__file__).parents[1]/'experiments'))
import run_office_humanstrong_v13_crossmodel as runner


class Stream:
    status_code=200;ok=True;headers={};content=b'';text=''
    def __init__(self,model,action=None,content='',complete=True):
        message={'content':content,'reasoning_content':'fixture reasoning'}
        if action is not None:
            message['tool_calls']=[{'index':0,'id':'call_1','type':'function',
                'function':{'name':'runtime_step','arguments':json.dumps(action)}}]
        self.chunks=[{'id':'fixture-response','model':model,'choices':[{'index':0,'delta':message,'finish_reason':None}]},
            {'id':'fixture-response','model':model,'choices':[{'index':0,'delta':{},
                'finish_reason':'tool_calls' if action is not None else 'stop'}],
             'usage':{'prompt_tokens':3,'completion_tokens':2,'total_tokens':5}}]
        self.complete=complete
    def iter_lines(self,**kwargs):
        for chunk in self.chunks:yield b'data: '+json.dumps(chunk).encode()
        if self.complete:yield b'data: [DONE]'
    def close(self):pass


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr('requests.sessions.Session.request',lambda *a,**k:pytest.fail('Real HTTP forbidden'))
    for name in ('MODEL_API_KEY','MOONSHOT_API_KEY','ZHIPU_API_KEY'):monkeypatch.setenv(name,'offline-fixture')


@pytest.mark.parametrize('model',runner.MODELS)
def test_streamed_native_computation_and_finish(tmp_path,monkeypatch,model):
    cfg,bridge,bundle=setup(tmp_path,monkeypatch);cfg=runner.model_config(cfg,model)
    old=v2.fixture_task(bridge)
    task=PublicTask(**{**old.__dict__,'goal':'Calculate the absolute difference between the two amounts.',
        'inputs':{**old.inputs,'source_pages':[]}})
    seen=[]
    def post(*args,**kwargs):
        wire=kwargs['json'];seen.append(wire)
        assert wire['model']==cfg['llm']['model'] and wire['stream'] and kwargs['stream']
        assert wire['max_tokens']==32768 and 'seed' not in wire
        assert ('reasoning_effort' in wire)==(model=='glm53_flash')
        assert ('stream_options' in wire)==(model=='kimi_k26')
        if model=='kimi_k26':
            assert wire['tools'][0]['function']['parameters']['properties']['action']['type']=='string'
            assert 'oneOf' not in json.dumps(wire['tools'])
        body=json.loads(wire['messages'][1]['content'])
        assert body.get('human_frontier_skill_ids') and not body['calls']['programs']
        action={'action':'call_tool','name':'execute_python',
                'arguments':{'source':'import json\nprint(json.dumps({"difference":abs(9-4)}))'}} if len(seen)==1 else {'action':'finish','answer':'5'}
        return Stream(wire['model'],action)
    monkeypatch.setattr('requests.post',post)
    with bridge.bridge():
        system=EmpiricalSystem(cfg,harness=OfficeAdapter({task.task_id:{'answer':'5'}},cfg),readonly=True,
            bank_view_factory=lambda b:bridge.HumanBankView(b,bundle,'rules'))
        try:trace=system.run_task(task,learn=False)
        finally:system.close()
    assert trace['score']['hard'] and len(seen)==2
    assert trace['execution']['reason']=='agent_submitted'


@pytest.mark.parametrize('model',runner.MODELS)
def test_finish_protocol_and_usage(tmp_path,monkeypatch,model):
    cfg,bridge,bundle=setup(tmp_path,monkeypatch);cfg=runner.model_config(cfg,model)
    task=v2.fixture_task(bridge);adapter=OfficeAdapter({},cfg);adapter.task=task;seen=[]
    def post(*args,**kwargs):seen.append(kwargs['json']);return Stream(kwargs['json']['model'],content='5')
    monkeypatch.setattr('requests.post',post)
    with bridge.bridge():
        system=EmpiricalSystem(cfg,harness=adapter,readonly=True,bank_view_factory=lambda b:bridge.HumanBankView(b,bundle,'rules'))
        system.task_context=TaskContext(cfg['runtime']);system.phase='test'
        try:
            result=system.agent('runtime','Return only the answer.',{'original_goal':'Amount?'},None,None,decision_purpose='finish_only')
            assert result=='5'
        finally:system.close()
    assert seen[0]['thinking']['type']==('enabled' if model=='glm53_flash' else 'disabled')


@pytest.mark.parametrize('dialect,model',[('moonshot_k2_chat','kimi-k2.6'),('glm_53_chat','glm-5.3-flash')])
def test_incomplete_stream_is_not_a_success(monkeypatch,dialect,model):
    provider=OpenAICompatibleProvider(OpenAICompatibleConfig('https://example.invalid',model,'MODEL_API_KEY',32,
        dialect=dialect,stream=True,max_retries=0))
    monkeypatch.setattr('requests.post',lambda *a,**k:Stream(model,content='5',complete=False))
    with pytest.raises(AgentProviderError):provider.complete([{'role':'user','content':'Amount?'}],tools=[])
    assert len(provider.request_records)==1 and provider.request_records[0]['outcome']=='error'


def test_complete_two_model_campaign(tmp_path,monkeypatch):
    runner.fixed.bind_runner(True);c=runner.common
    identity=lambda:{'git_sha':'offline-fixture','tracked_dirty':False,'source_sha256':'offline-fixture'}
    monkeypatch.setattr(c,'code_identity',identity)
    monkeypatch.setattr('atomic_skillgraph.experiments.run_empirical.code_identity',identity)
    monkeypatch.setattr(c,'load_env',lambda _:None)
    root=tmp_path/'campaign';parent_hash=c.sha(runner.PARENT/'budget.json')
    runner.prepare(root);plan=c.verify(root)
    assert len(plan['episodes'])==204 and plan['new_token_cap']==44181504
    assert plan['ledger_limits']['token_limit']==234188758
    assert [e['model_key'] for e in plan['episodes'][:4]]==list(runner.MODELS)*2
    sent=[]
    def post(*args,**kwargs):
        wire=kwargs['json'];sent.append(wire['model'])
        assert not json.loads(wire['messages'][1]['content'])['calls']['programs']
        return Stream(wire['model'],{'action':'finish','answer':'offline fixture'})
    monkeypatch.setattr('requests.post',post)
    assert runner.execute(root,'unused')==0
    result=runner.status(root)
    assert result['new_http']==204 and result['new_tokens_used_or_reserved']==1020
    assert result['new_unsettled']==0 and len(sent)==204
    assert all(m['completed']==102 and m['http']==102 for m in result['models'].values())
    assert c.sha(runner.PARENT/'budget.json')==parent_hash
    assert {r['model_key'] for r in c.read(root/'comparison.json')['results']}==set(runner.MODELS)
    assert all(c.read(Path(e['path'])/'freeze_record.json')['unchanged'] for e in plan['episodes'])


def test_schema_refusal_prefix_keeps_full_reservation(tmp_path,monkeypatch):
    runner.fixed.bind_runner(True);c=runner.common
    monkeypatch.setattr(c,'code_identity',lambda:{'git_sha':'offline-fixture','tracked_dirty':False,'source_sha256':'offline-fixture'})
    failed=Path('/home/yangchengyu/officeqa_humanstrong_v13_rules_crossmodel_test102_seed42_20261022')
    before=c.sha(failed/'budget.json')
    runner.prepare(tmp_path/'recovery',failed)
    plan=c.verify(tmp_path/'recovery');ledger=c.read(tmp_path/'recovery/budget.json')
    assert plan['inherited_used_or_reserved']==190082144
    assert plan['ledger_limits']['token_limit']==234263648
    assert c.read(failed/'budget.json')['attempts']==ledger['attempts']
    assert not c.billing_blocked(ledger) and c.sha(failed/'budget.json')==before


def test_wire_schema_keeps_host_validation():
    from atomic_skillgraph.agents.moonshot_schema import wire_schema
    from atomic_skillgraph.empirical.prompts import STEP
    from jsonschema import Draft202012Validator
    before=json.dumps(STEP);wire=wire_schema(STEP)
    assert json.dumps(STEP)==before
    for action in [{'action':'finish','answer':'5'}, {'action':'finish','answer':5},
                   {'action':'call_tool','name':'execute_python','arguments':{'source':'print(5)'}},
                   {'action':'complete_node','outputs':{'number':5},'output_refs':{'number':{'result_id':'r1','path':['computed','number']}}}]:
        assert Draft202012Validator(STEP).is_valid(action) and Draft202012Validator(wire).is_valid(action)
    assert not Draft202012Validator(STEP).is_valid({'action':'invented'})
