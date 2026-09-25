from pathlib import Path
import sys
import threading

import pytest

from experiments.baselines.scienceworld import campaign


def test_campaign_keeps_each_method_seed_lane_independent(tmp_path, monkeypatch):
    monkeypatch.setenv('MODEL_API_KEY','unit-test-not-real')
    monkeypatch.setattr(campaign.os,'environ',dict(campaign.os.environ))
    calls=[]
    lock=threading.Lock()
    def run(command, **kwargs):
        assert kwargs['check'] is True
        with lock:
            calls.append(command)
    monkeypatch.setattr(campaign.subprocess, 'run', run)
    monkeypatch.setattr(sys, 'argv', ['campaign', '--root', str(tmp_path),
        '--methods', 'b0_dynamic', 'b3_skillopt', '--workers', '3', '--smoke', '--resume'])
    campaign.main()
    assert len(calls)==6
    assert {(c[c.index('--method')+1], c[c.index('--seed')+1]) for c in calls}=={
        (method, str(seed)) for method in ('b0_dynamic','b3_skillopt') for seed in (42,43,44)}
    assert all('--smoke' in c and '--resume' in c for c in calls)
    assert len(list(tmp_path.glob('*.log')))==6


@pytest.mark.parametrize('extra', [['--seeds','42','42'], ['--workers','16']])
def test_campaign_rejects_duplicate_or_unbounded_lanes(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(sys, 'argv', ['campaign','--root',str(tmp_path),*extra])
    monkeypatch.setattr(campaign.subprocess, 'run', lambda *a,**k:pytest.fail('must not launch'))
    with pytest.raises(SystemExit):
        campaign.main()


def test_missing_key_fails_before_any_lane_or_output(tmp_path,monkeypatch):
    monkeypatch.delenv('MODEL_API_KEY',raising=False)
    monkeypatch.setattr(sys,'argv',['campaign','--root',str(tmp_path/'absent')])
    with pytest.raises(ValueError,match='MODEL_API_KEY'):
        campaign.main()
    assert not (tmp_path/'absent').exists()


def test_explicit_shared_env_file_without_executing_shell(tmp_path,monkeypatch):
    monkeypatch.delenv('MODEL_API_KEY',raising=False)
    path=tmp_path/'key.env';path.write_text("MODEL_API_KEY='not-real-secret' # comment\n")
    campaign.credentials(path)
    assert campaign.os.environ['MODEL_API_KEY']=='not-real-secret'


def test_ordered_parallel_results_and_worker_limit(monkeypatch):
    from experiments.baselines.scienceworld.parallel import ordered_map
    monkeypatch.setenv('SW_EPISODE_WORKERS','2')
    barrier=threading.Barrier(2,timeout=5)
    lock=threading.Lock()
    active=peak=0
    def evaluate(value):
        nonlocal active,peak
        with lock:
            active+=1
            peak=max(peak,active)
        barrier.wait()
        with lock:
            active-=1
        return value*10
    assert ordered_map(evaluate,[3,1,4,2])==[30,10,40,20]
    assert peak==2


def test_failed_lane_does_not_cancel_other_lanes(tmp_path,monkeypatch):
    import json
    monkeypatch.setenv('MODEL_API_KEY','unit-test-not-real')
    monkeypatch.setattr(campaign.os,'environ',dict(campaign.os.environ))
    monkeypatch.setattr(sys,'argv',['campaign','--root',str(tmp_path),
        '--methods','b0_dynamic','--seeds','42','43','44','--workers','3'])
    calls=[]
    def run(command,**kwargs):
        seed=command[command.index('--seed')+1]
        calls.append(seed)
        if seed=='42':raise campaign.subprocess.CalledProcessError(1,command)
    monkeypatch.setattr(campaign.subprocess,'run',run)
    assert campaign.main()==1
    assert set(calls)=={'42','43','44'}
    records=json.loads((tmp_path/'campaign_results.json').read_text())
    assert [r['passed'] for r in records]==[False,True,True]
