from pathlib import Path
import sys
import threading

import pytest

from experiments.baselines.scienceworld import campaign


def test_campaign_keeps_each_method_seed_lane_independent(tmp_path, monkeypatch):
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


@pytest.mark.parametrize('extra', [['--seeds','42','42'], ['--workers','4']])
def test_campaign_rejects_duplicate_or_unbounded_lanes(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(sys, 'argv', ['campaign','--root',str(tmp_path),*extra])
    monkeypatch.setattr(campaign.subprocess, 'run', lambda *a,**k:pytest.fail('must not launch'))
    with pytest.raises(SystemExit):
        campaign.main()
