"""Production V6 boundaries with intercepted HTTP, never live model calls."""
from copy import deepcopy
from dataclasses import asdict
import json

import pytest

from atomic_skillgraph.empirical.checkpoint import save_json
from atomic_skillgraph.empirical.contracts import PublicTask, digest
from atomic_skillgraph.empirical.office_effect_contract import validate_candidate, VERSION
from atomic_skillgraph.empirical.office_effect_handoff import observed_hits, resolve_hit, bind_read, recover_handoffs
from atomic_skillgraph.empirical.office_effect_sources import partition, load_case
from atomic_skillgraph.empirical.office_effect_optimizer import evaluate_pairs, choose_version
from atomic_skillgraph.empirical.office_effect_runtime import OfficeEffectView, runtime_evidence
from atomic_skillgraph.empirical.model_view import expand_material
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.harness.benchmarks import OfficeAdapter
from atomic_skillgraph.harness.simple_protocol import Broker
from test_office_trace_v3 import local_system
from test_cf2_contracts import http, response


@pytest.fixture(autouse=True)
def forbid_http(monkeypatch):
    monkeypatch.setattr('requests.sessions.Session.request',lambda *a,**k:pytest.fail('Real HTTP forbidden'))


def policy(kind='all'):
    return {'decision':'propose','reason':'offline fixture only','source_case_ids':['source'],
        'goal':'Retrieve evidence','applicability':kind,'procedure':['Identify a source.','Read the observed hit.'],
        'stop_condition':'Continue native retrieval if evidence is missing.'}


def search(path='a.txt',offset=12,legacy=False):
    hits = [{'path':path,'offset':offset,'line':'actual Unicode 文本'}]
    return {'name':'grep' if legacy else 'grep_page','event_id':'e1','result_id':'r1','state':'finished',
        'backend_invoked':True,'arguments':{'pattern':'target','paths':[path]},
        'result':{'accepted':True,'data':hits if legacy else {'hits':hits,'complete':True,'next_cursor':None}}}


def read_event(path='a.txt',offset=12):
    return {'name':'read','event_id':'e2','result_id':'r2','state':'finished','backend_invoked':True,
        'arguments':{'path':path,'offset':offset},'result':{'accepted':True,'data':{'text':'actual acquired text'}}}


@pytest.mark.parametrize('legacy',[False,True])
def test_hit_identity_same_producer_and_unicode(legacy):
    event = search(legacy=legacy); ref = observed_hits(event,'corpus')[0]
    assert resolve_hit(ref,[event],'corpus')['offset']==12
    assert bind_read(ref,{},[event],'corpus')['status']=='needs_input'
    assert bind_read(ref,{'before':2,'after':30},[event],'corpus')['arguments']['path']=='a.txt'
    for change in ({'corpus_identity':'wrong'},{'producer_result_id':'invented'},{'result_path':['data',2]}):
        with pytest.raises(ValueError):resolve_hit({**ref,**change},[event],'corpus')
    assert recover_handoffs([event,read_event()],'corpus')[0]['state']=='observed_handoff'
    assert recover_handoffs([event,read_event('b.txt')],'corpus',binding_claimed=True)[0]['reason']=='cross_file_handoff_invalid'
    assert recover_handoffs([event,read_event('b.txt')],'corpus')[0]['reason']=='cross_file_handoff_unproven'
    second = {**deepcopy(event),'event_id':'another','result_id':'another_result'}
    assert recover_handoffs([event,second,read_event()],'corpus')[0]['state']=='unproven_handoff'


def test_no_backend_no_proof():
    e=search(); e['backend_invoked']=False
    assert observed_hits(e,'corpus')==[]
    e=read_event();e['result']['data']['text']=''
    assert recover_handoffs([search(),e],'corpus')[0]['state']=='unproven_handoff'
    with pytest.raises(ValueError):resolve_hit([],[],'corpus')


def test_partition_and_loading_reject_heldout(tmp_path):
    records=[{'task':{'physical_key':str(i),'split':'train'}} for i in range(120)]
    split=partition(records)
    assert split==partition(list(reversed(records)))
    assert [len(split[k]) for k in ('induction','edit_feedback','train_gate')]==[80,20,20]
    for role in ('edit_feedback','train_gate'):
        with pytest.raises(ValueError):load_case({'task':{'physical_key':split[role][0]}},split,'induction')


def test_single_local_edit_and_content_budget():
    parent=validate_candidate(policy(),['source'])
    raw=policy();raw['procedure'][1]='Use a current result reference.';raw['edit']={'operation':'replace','index':1}
    child=validate_candidate(raw,['source'],parent)
    assert child['parent_hash']==digest(parent)
    raw['procedure'][0]='Another change.'
    with pytest.raises(ValueError):validate_candidate(raw,['source'],parent)
    raw=policy();raw['procedure'][0]='x'*1000
    with pytest.raises(ValueError):validate_candidate(raw,['source'])
    raw=policy();raw['source_case_ids']=['heldout']
    with pytest.raises(ValueError):validate_candidate(raw,['source'])


def pair(key,a,b,empty=False):
    def arm(correct):return {'complete':True,'unknown_billing':False,'score':{'hard':correct},'empty':False,'tokens':10,'http':1,'tool_calls':1}
    value={'physical_key':key,'same_protocol':True,'bank_unchanged':True,'off':arm(a),'on':arm(b)}
    value['on']['empty']=empty
    return value


def test_negative_and_empty_edits_roll_back():
    r=evaluate_pairs([pair('a',True,False)])
    assert not r['accepted'] and choose_version('old','new',r)=='old'
    assert not evaluate_pairs([pair('a',False,True),pair('b',True,False)])['accepted']
    assert not evaluate_pairs([pair('a',False,True,True)])['accepted']
    assert evaluate_pairs([pair('a',False,True)])['accepted']
    bad=pair('a',False,True);bad['on']['unknown_billing']=True
    with pytest.raises(ValueError):evaluate_pairs([bad])


@pytest.mark.parametrize('enabled,kind,expected',[(True,'all',True),(False,'all',False),(True,'time_series_or_comparison',False)])
def test_actual_planner_executor_payloads_and_no_fallback(tmp_path,monkeypatch,enabled,kind,expected):
    old,task=local_system(tmp_path);cfg=deepcopy(old.config);bank_hash=old.bank.digest();old.close()
    candidate=validate_candidate(policy(kind),['source']);path=tmp_path/'view.json'
    save_json(path,{'method_version':VERSION,'bank_digest':bank_hash,'candidate':candidate,'enabled':enabled})
    cfg['experiment'].update(runtime_mode='frozen',office_effect_version=VERSION,office_effect_manifest=str(path))
    cfg['runtime']['global_action_budget']=48
    plan={'mode':'compose','workflow':{'interface_version':'empirical.workflow.v2','goal':task.goal,
        'nodes':[{'id':'task','execution_mode':'dynamic','goal':'unrelated node wording','args':{}}]}}
    payloads=http(monkeypatch,[response([plan],name='submit_plan'),response([{'action':'finish','answer':'private'}])])
    adapter=OfficeAdapter({task.task_id:{'answer':'private'}},cfg)
    s=EmpiricalSystem(cfg,harness=adapter,readonly=True)
    try:
        trace=s.run_task(task,learn=False)
        assert trace['score']['hard']
        assert len(payloads)==2
        p,r=[expand_material(json.loads(v['messages'][1]['content'])) for v in payloads]
        ids=[candidate['id']] if expected else []
        assert p['active_guidance_ids']==ids==r['recovery']['active_guidance_ids']
        assert bool(p['related_interfaces'])==expected==bool(r['memory']['guidance'])
        assert len(r['memory']['guidance'])<=1
        assert s.bank.digest()==bank_hash
    finally:s.close()


def test_observed_progress_and_one_loop_notice(tmp_path):
    s,task=local_system(tmp_path)
    try:
        context=TaskContext(s.config['runtime']);b=Broker(s.adapter,48,context=context)
        assert not runtime_evidence(context,task.goal)['progress']['text_acquired']
        b.call('grep',{'pattern':'target'}); b.call('grep',{'pattern':'target'})
        # Broker may deduplicate before dispatch; neither repetition can certify full evidence.
        state=runtime_evidence(context,task.goal)
        assert state['progress']['all_periods']=='unverified'
        context.effect_owner_version += 1
        assert 'loop_feedback' not in runtime_evidence(context,task.goal)
    finally:s.close()


def test_real_runner_receipts_budget_scope_and_idempotent_resume(tmp_path,monkeypatch):
    monkeypatch.setattr('atomic_skillgraph.experiments.run_empirical.code_identity',lambda:{'git_sha':'offline-fixture','tracked_dirty':False})
    from atomic_skillgraph.experiments.run_office_effect_v6 import episode, PhaseGovernor
    from atomic_skillgraph.empirical.office_operation_mining import sha, read
    from atomic_skillgraph.experiments.formal_log import tree_identity
    old,task=local_system(tmp_path);cfg=deepcopy(old.config);bank_hash=old.bank.digest();old.close()
    cfg['experiment'].update(runtime_mode='frozen',office_effect_version=VERSION)
    cfg['runtime']['global_action_budget']=48
    labels=tmp_path/'evaluator_records_train.json';save_json(labels,{task.task_id:{'answer':'private'}})
    cfg['harness']['evaluator_records']=str(labels)
    root=tmp_path/'run';root.mkdir();limits={'token_limit':2000000,'request_limit':100,'finish_reserve':0}
    plan={'effective_limits':limits,'empty_bank_digest':bank_hash,
          'settings':{'phase_limits':{'feedback':{'tokens':384000,'http':40}}}}
    save_json(root/'manifest.json',plan);save_json(root/'sibling_reservations.json',[])
    save_json(root/'authorization.json',{'manifest_sha256':sha(root/'manifest.json'),'user_authorization':'offline fixture',
        'approved_phases':['feedback'],'phase_limits':plan['settings']['phase_limits']})
    save_json(root/'zero_http_check.json',{'passed':True,'manifest_sha256':sha(root/'manifest.json')})
    save_json(root/'budget/usage.json',{'limits':limits,'attempts':{},'unknown_billing':False})
    candidate=validate_candidate(policy(),['source'])
    flow={'mode':'compose','workflow':{'interface_version':'empirical.workflow.v2','goal':task.goal,
        'nodes':[{'id':'task','execution_mode':'dynamic','goal':task.goal,'args':{}}]}}
    seen=http(monkeypatch,[response([flow],name='submit_plan'),response([{'action':'finish','answer':'private'}])])
    first=episode(root,'feedback',root/'arm',cfg,task,candidate,True,plan)
    assert first['http']==2 and first['tokens']==10 and first['score']['hard']
    assert all(r['ids']==[candidate['id']] for r in first['exposure'])
    rows=read(root/'budget/usage.json')['attempts'].values()
    assert all(r['context']['parent_token_limit']==192000 and not r['context']['train_parent'] for r in rows)
    again=episode(root,'feedback',root/'arm',cfg,task,candidate,True,plan)
    assert first==again and len(seen)==2
    state=read(root/'budget/usage.json');state['unknown_billing']=True;save_json(root/'budget/usage.json',state)
    governor=PhaseGovernor(root,'feedback')
    assert governor.headroom({})['unknown_billing']
