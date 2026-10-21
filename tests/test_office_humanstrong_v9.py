"""One narrow result-scale contract; independent fresh paired orchestration."""
import json
from pathlib import Path
import sys

import pytest
from test_cf2_contracts import http,response
from atomic_skillgraph.empirical.contracts import digest,validate_schema_instance
import test_office_humanstrong_v5 as fixtures
import test_office_humanstrong_v8 as sparse
from test_office_human_phase_v3 import load

ROOT=Path(__file__).parents[1]/'resources/office_humanstrong_v9'
PARENT=Path('/home/yangchengyu/officeqa_humanstrong_v8_targeted_test15_pairs_seed42_20261021')


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a,**kw):raise AssertionError('Actual HTTP forbidden')
    monkeypatch.setattr('requests.sessions.Session.request',refuse)


def test_fixed_scale_real_docker(tmp_path,monkeypatch):
    monkeypatch.setattr(fixtures,'ROOT',ROOT)
    setup=fixtures.setup.__wrapped__(tmp_path)
    cfg,bridge,bundle=setup
    assert [len(bundle[k]) for k in ('skill','program','implementation','workflow')]==[1,1,1,0]
    assert bridge.VERSION=='office.humanstrong.sparse.v9.20261021'
    assert bridge.matching('mid-point normalized difference')
    assert not bridge.matching('absolute percent difference')
    assert not bridge.matching('correlation between two series')
    args=dict(operation='symmetric_difference',unit='ratio',decimals=2,output_percent=False,values=[3,5])
    validate_schema_instance(args,bundle['program'][0]['input_schema'])
    with pytest.raises(ValueError):
        validate_schema_instance({**args,'output_percent':True},bundle['program'][0]['input_schema'])
    sparse.test_real_docker_narrow_calculators(setup,monkeypatch,'symmetric_difference',
        dict(values=[3,5],output_percent=False,decimals=2),'Absolute midpoint-normalized difference of 3 and 5.','0.50')


def test_fresh_second_observation(tmp_path,monkeypatch):
    source=ROOT.parents[1];sys.path.insert(0,str(source/'experiments'))
    import run_office_humanstrong_v9 as launcher
    launcher.bind_runner();common=launcher.common
    identity=lambda:{'git_sha':'offline-fixture','tracked_dirty':False,'source_sha256':'offline-fixture'}
    monkeypatch.setattr(common,'code_identity',identity)
    monkeypatch.setattr('atomic_skillgraph.experiments.run_empirical.code_identity',identity)
    monkeypatch.setattr(common,'load_env',lambda _:None)
    from filelock import FileLock
    locks=tmp_path/'locks';locks.mkdir()
    monkeypatch.setattr(common,'FileLock',lambda path,timeout=0:FileLock(str(locks/(digest(str(path))+'.lock')),timeout=timeout))
    root=tmp_path/'pairs';parent_hash=common.sha(PARENT/'budget.json')
    launcher.prepare(root)
    plan=common.verify(root)
    assert len(plan['episodes'])==10 and plan['new_token_cap']==2165760
    assert plan['inherited_used_or_reserved']==164741320
    assert plan['prior_scores_used_for_selection'] and plan['development_exposed']
    assert plan['ledger_limits']['token_limit']==166907080
    seen=http(monkeypatch,[response([{'action':'finish','answer':'offline fixture'}]) for _ in range(10)])
    assert common.execute(root,'unused')==0
    for episode,wire in zip(plan['episodes'],seen):
        body=json.loads(wire['messages'][1]['content'])
        count=int(episode['task_id']=='officeqa:UID0237' and episode['arm']=='manual')
        assert len(body.get('human_frontier_skill_ids',[]))==count
        assert len(body['calls']['programs'])==count
        assert common.read(Path(episode['path'])/'freeze_record.json')['unchanged']
    assert common.sha(PARENT/'budget.json')==parent_hash
    assert common.status(root)['new_http']==10
    common.verify(root)
