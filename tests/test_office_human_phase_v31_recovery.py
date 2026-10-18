"""Real interrupted prefix replay; all Provider HTTP is intercepted."""
import json
import os
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'experiments'))
import run_office_human_phase_v31 as runner
import recover_office_human_phase_v31 as recovery
from atomic_skillgraph.empirical.budget_governor import billing_blocked, input_token_bound
from atomic_skillgraph.empirical.contracts import PublicTask
from atomic_skillgraph.experiments.formal_log import FormalLog, tree_identity
from test_cf2_contracts import http,response


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def refuse(*a,**kw):raise AssertionError('Real HTTP forbidden')
    monkeypatch.setattr('requests.sessions.Session.request',refuse)


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    archive=os.environ.get('OFFICE_PHASE_V31_ARCHIVE')
    if not archive:pytest.skip('Set OFFICE_PHASE_V31_ARCHIVE to the real interrupted run')
    base=Path(archive);root=tmp_path/'child'
    monkeypatch.setattr(runner,'code_identity',lambda:{'git_sha':'offline-test','tracked_dirty':False,'source_sha256':'offline-test'})
    recovery.prepare(root,base)
    return root,base


def test_exact_prefix_and_unchanged_authority(prepared):
    root,base=prepared;plan=runner.verify(root)
    old=runner.read(base/'budget.json');new=runner.read(root/'budget.json')
    assert old['attempts']==new['attempts'] and old['limits']==new['limits']
    assert billing_blocked(old) and not billing_blocked(new)
    assert plan['recovery']['retained_episodes']==4
    assert plan['recovery']['unstarted_episodes']==19
    assert plan['new_token_cap']==3955281 and plan['new_http_cap']==2352
    finish=[e for e in plan['episodes'] if e.get('recovery_source')]
    assert len(finish)==1 and finish[0]['finish_reserve']==18432
    assert plan['package_identity']==runner.read(base/'plan.json')['package_identity']
    assert runner.read(root/'budget_prefix.json')==old


def test_real_context_finish_once_and_preserve_unknown(prepared,monkeypatch):
    root,base=prepared;plan=runner.read(root/'plan.json')
    episode=next(e for e in plan['episodes'] if e.get('recovery_source'))
    out=Path(episode['path']);config=runner.read(out/'config.json')
    task=next(PublicTask(**t) for t in runner.read(root/'tasks.json')['tasks'] if t['task_id']==episode['task_id'])
    bundle=runner.read(out/'bank/imported_bank.json');prior=runner.read(root/'budget.json')
    governor=runner.GuardedGovernor(root,plan)
    seen=http(monkeypatch,[response(content='123.45',finish='stop')])
    def no_native(*a,**kw):raise AssertionError('Recovery must not replay native tools')
    monkeypatch.setattr('atomic_skillgraph.harness.simple_protocol.Broker.call',no_native)
    formal=FormalLog(out/'formal',{'purpose':'offline-real-recovery'},config)
    with runner.frontier.bridge():
        result=recovery.finish_episode(config,task,out,episode,governor,formal,runner.frontier,bundle)
        repeated=recovery.finish_episode(config,task,out,episode,governor,formal,runner.frontier,bundle)
    assert len(seen)==1 and not seen[0].get('tools')
    assert seen[0]['thinking']['type']=='disabled'
    assert input_token_bound(seen[0])<=16384
    assert input_token_bound(seen[0])+2048<=episode['finish_reserve']
    after=runner.read(root/'budget.json')
    assert all(after['attempts'][k]==v for k,v in prior['attempts'].items())
    added=[v for k,v in after['attempts'].items() if k not in prior['attempts']]
    assert len(added)==1 and added[0]['context']['parent_scope']==episode['original_parent_scope']
    assert added[0]['context']['purpose']=='finish_only'
    assert result['known_total_tokens']==111100 and result['total_tokens'] is None
    assert result['unknown_billing_attempts']==1 and repeated==result
    trace=result['cases'][0]
    assert trace['episode_result']['sealed_prediction']['prediction']=='123.45'
    assert trace['episode_result']['termination_reason']=='finish_only'
    assert trace['tools']==runner.read(Path(episode['recovery_source'])/'summary.json')['cases'][0]['tools']
    assert runner.read(out/'execution_manifest.json')['code']==plan['code']
    with sqlite3.connect(out/'run.sqlite3') as db:
        stored=json.loads(db.execute('SELECT result FROM tasks').fetchone()[0])
    assert stored==trace
    assert any(r['purpose']=='finish_only' for r in formal.rows('llm_calls'))
    runner.verify(root)  # Original files and all frozen Banks are unchanged.


def test_runner_stops_before_next_episode_after_unknown(tmp_path,monkeypatch):
    root=tmp_path/'run';save=runner.save_json
    config={'harness':{'corpus_root':str(tmp_path/'corpus')}}
    (tmp_path/'corpus').mkdir();save(root/'base_config.json',config)
    save(root/'tasks.json',{'tasks':[{'task_id':'x','physical_key':'x','goal':'Synthetic','split':'train'}]})
    episodes=[]
    for arm in ('manual','off'):
        out=root/arm;save(out/'bank/imported_bank.json',{});save(out/'config.json',config)
        episodes.append({'task_id':'x','arm':arm,'path':str(out),'bank_identity':tree_identity(out/'bank')})
    plan={'episodes':episodes,'locks':[],'code':{},'asset_origin':'test','new_token_cap':1000,'new_http_cap':2,
          'corpus_identity':tree_identity(tmp_path/'corpus')}
    save(root/'plan.json',plan);save(root/'budget.json',{'attempts':{}});save(root/'budget_prefix.json',{'attempts':{}})
    governor=SimpleNamespace(billing_blocked=False);calls=[]
    monkeypatch.setattr(runner,'verify',lambda r:plan)
    monkeypatch.setattr(runner,'load_env',lambda e:None)
    monkeypatch.setattr(runner,'GuardedGovernor',lambda *a:governor)
    def interrupted(config,tasks,out,**kw):
        calls.append(str(out));governor.billing_blocked=True
        result={'complete':True,'successes':0,'total_tokens':None,'program_invocations':0,
                'known_total_tokens':0,'unknown_billing_attempts':1,
                'cases':[{'execution':{'reason':'token_budget_exhausted'}}]}
        save(out/'summary.json',result);return result
    monkeypatch.setattr(runner,'run',interrupted)
    assert runner.execute(root,None)==1
    assert len(calls)==1 and not (root/'off/summary.json').exists()
    assert runner.read(root/'launch_status.json')['status']=='failed'
    assert not (root/'comparison.json').exists()
