"""Replay the metered P7 stop with no real HTTP; original evidence stays read-only."""
from copy import deepcopy
import shutil
from pathlib import Path
import pytest

from atomic_skillgraph.experiments import run_math_effect_p7 as run, recover_math_effect_p7 as recovery
from atomic_skillgraph.empirical.contracts import PublicTask
from test_cf2_contracts import http, response

PARENT=Path('/home/yangchengyu/livemath_effect_p7_seed42_ready_20261015')
CHECKOUT='/home/yangchengyu/asg_livemath_effect_p7_release_seed42_20261015'
pytestmark=pytest.mark.skipif(not PARENT.exists(),reason='Actual stopped P7 billing/trace fixture required')


def new_code(monkeypatch):
    monkeypatch.setattr(run,'code_identity',lambda:{'git_sha':'offline_scope_fixture','tracked_dirty':False,'source_sha256':'offline'})


def test_recovery_reuses_real_receipts_and_unblocks_unsent_arm(tmp_path,monkeypatch):
    new_code(monkeypatch);root=tmp_path/'continued'
    before={str(p):run.sha256(p) for p in PARENT.rglob('*') if p.is_file()}
    receipt=recovery.recover(root,PARENT,CHECKOUT,'OFFLINE RECOVERY FIXTURE ONLY')
    assert receipt['reused_completed_traces']==7 and receipt['inherited_p7_tokens']==253456
    assert len(receipt['inherited_p7_request_ids'])==9 and len(receipt['not_sent_requests_retained_only_in_parent'])==1
    plan=run.checked(root);g=run.StageContext(run.phase_governor(root,'feedback',plan),'feedback')
    assert len(g.attempts())==8 and sum(v['accounted_tokens'] for v in g.attempts().values())==221597
    cfg=run.read(root/'config.json');candidates=run.read(root/'learning/initial/learning.json')['candidates']
    monkeypatch.setenv(cfg['llm']['api_key_env'],'offline-fixture')
    records=run.read(Path(plan['datasets'])/'livemath/evaluator_records_train.json')
    adapter=run.AnswerAdapter('livemath',records)
    calls=http(monkeypatch,[response(content='A',finish='stop')])
    for p in (root/'feedback').rglob('trace.json'):
        saved=run.read(p);policy=[] if p.relative_to(root).parts[1]=='off' else candidates
        assert run.runtime.solve(PublicTask(**saved['task']),adapter,policy,cfg,p.parent,g)==saved
    assert len(calls)==0
    request=root/receipt['not_sent_requests_retained_only_in_parent'][0]
    assert not request.exists()
    task=next(t for t in run.read(root/'tasks.json')['train'] if t['physical_key']==request.parent.parent.name)
    result=run.runtime.solve(PublicTask(**task),adapter,candidates,cfg,request.parent.parent,g)
    assert result['valid'] and len(calls)==1 and not result['selection']['selected']
    assert len(g.attempts())==9
    assert {str(p):run.sha256(p) for p in PARENT.rglob('*') if p.is_file()}==before
    # Inherited P7 charges are an immutable prefix, not a new phase allowance.
    state=deepcopy(g.state);state['attempts'].pop(receipt['inherited_p7_request_ids'][0])
    run.write_json(root/'budget/ledger.json',state)
    with pytest.raises(ValueError,match='Historical charge'):run.checked(root)


def test_recovery_refuses_ambiguous_unsent_marker(tmp_path,monkeypatch):
    new_code(monkeypatch);parent=tmp_path/'parent';shutil.copytree(PARENT,parent)
    monkeypatch.setattr(recovery,'inspect_parent',lambda *_:run.read(parent/'plan.json'))
    pending=next(p for p in (parent/'feedback').rglob('request.json') if not (p.parent/'response.json').exists())
    run.write_json(pending.parent/'started.json',{'identity':run.read(pending)['identity']})
    with pytest.raises(ValueError,match='Interrupted physical request'):
        recovery.recover(tmp_path/'continued',parent,CHECKOUT,'OFFLINE RECOVERY FIXTURE ONLY')
    assert not (tmp_path/'continued').exists()
