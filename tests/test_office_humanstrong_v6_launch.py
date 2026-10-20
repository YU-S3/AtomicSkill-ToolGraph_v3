"""Zero-HTTP end-to-end paired Val runner, including immutable billing ancestry."""
import json
from pathlib import Path
import sys
import pytest
from test_cf2_contracts import http, response

SOURCE = Path(__file__).parents[1]
PARENT = Path('/home/yangchengyu/officeqa_humanstrong_v5_test102_pairs_seed42_20261020')
RESOURCES = Path('/home/yangchengyu/officeqa_external_minimax_resources_20261020_v1')


def test_fresh_val_pairs_real_runner(tmp_path, monkeypatch):
    sys.path.insert(0, str(SOURCE/'experiments'))
    import run_office_humanstrong_v6 as launcher
    common = launcher.common
    launcher.bind_runner()
    identity = lambda: {'git_sha':'offline-fixture','tracked_dirty':False,'source_sha256':'offline-fixture'}
    monkeypatch.setattr(common,'code_identity',identity)
    monkeypatch.setattr('atomic_skillgraph.experiments.run_empirical.code_identity',identity)
    monkeypatch.setattr(common,'load_env',lambda _:None)
    def refuse(*a, **kw): raise AssertionError('Actual HTTP forbidden')
    monkeypatch.setattr('requests.sessions.Session.request',refuse)
    from filelock import FileLock
    from atomic_skillgraph.empirical.contracts import digest
    lock_root = tmp_path/'locks'; lock_root.mkdir()
    monkeypatch.setattr(common,'FileLock',lambda path,timeout=0: FileLock(str(lock_root/(digest(str(path))+'.lock')),timeout=timeout))
    root = tmp_path/'pairs'
    parent_hash = common.sha(PARENT/'budget.json')
    launcher.prepare(root,PARENT,RESOURCES,PARENT)
    plan = common.verify(root)
    assert len(plan['episodes']) == 48
    assert plan['inherited_used_or_reserved'] == 155960797
    assert plan['new_token_cap'] == 10395648
    assert plan['ledger_limits']['token_limit'] == 166356445
    assert plan['authorized_cumulative_ceiling'] == 190000000
    assert plan['new_http_cap'] == 100080
    assert plan['test_runs_authorized_in_this_plan'] == 0
    cfg = common.read(root/'base_config.json')
    assert cfg['experiment']['input_alignment'] == 'legacy_val_internal_only'
    assert cfg['experiment']['office_human_bank_version'] == 'office.humanstrong.programmatic.v6.20261021'
    assert common.read(root/'budget.json')['attempts'] == common.read(PARENT/'budget.json')['attempts']
    with monkeypatch.context() as patched:
        patched.setattr(launcher,'CUMULATIVE_CAP',166356444)
        with pytest.raises(ValueError,match='Insufficient cumulative authority'):
            launcher.prepare(tmp_path/'too_small',PARENT,RESOURCES,PARENT)
    assert not (tmp_path/'too_small').exists()
    seen = http(monkeypatch,[response([{'action':'finish','answer':'offline fixture'}]) for _ in range(48)])
    assert common.execute(root,'offline-unused-env') == 0
    state = common.status(root)
    assert state['completed_episodes'] == state['new_http'] == 48
    rows = {t['task_id']:t for t in common.read(root/'tasks.json')['tasks']}
    for episode, wire in zip(plan['episodes'],seen):
        body = json.loads(wire['messages'][1]['content'])
        assert body['task']['inputs'] == rows[episode['task_id']]['inputs']
        assert common.frontier.ANSWER_CONTRACT in wire['messages'][0]['content']
        assert len(body['calls']['programs']) == (9 if episode['arm']=='manual' else 0)
        assert common.read(Path(episode['path'])/'freeze_record.json')['unchanged']
    assert common.sha(PARENT/'budget.json') == parent_hash
    common.verify(root)
