"""Zero-HTTP checks of the production fresh launcher and budget carryforward."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from atomic_skillgraph.empirical.budget_governor import BudgetGovernor


def fixture(tmp_path, monkeypatch, *, approval=True, conflict=False):
    script=Path(__file__).resolve().parents[1]/'scripts/run_officeqa_r1_seed42.sh'
    block=script.read_text().split("<<'PY'\n",1)[1].split('\nPY\n',1)[0]
    monkeypatch.chdir(tmp_path)
    original={'token_limit':36096000,'finish_reserve':0,'request_limit':284070}
    requested={**original,'token_limit':85000000,'request_limit':480000}
    configs=tmp_path/'configs';configs.mkdir()
    (configs/'atomic_unified_seed42_budget.json').write_text(json.dumps({'cells':{'officeqa':original}}))
    (tmp_path/'benchmark_profiles.json').write_text(json.dumps({'profiles':{'officeqa':{'image_digest':'fixture-image'}}}))
    source=tmp_path/'old';source.mkdir();extra=tmp_path/'extra';extra.mkdir()
    first={'status':'metered','accounted_tokens':100,'reserved_tokens':120,'context':{}}
    second={**first,'accounted_tokens':200}
    ledger={'limits':original,'attempts':{'old-1':first},'unknown_billing':False}
    (source/'budget.json').write_text(json.dumps(ledger))
    added={'old-1':{**first,'accounted_tokens':101} if conflict else first,'diagnostic-1':second}
    (extra/'budget.json').write_text(json.dumps({**ledger,'attempts':added}))
    (tmp_path/'base.json').write_text(json.dumps({'budget':requested}))
    (tmp_path/'spec.json').write_text(json.dumps({'base_config':str(tmp_path/'base.json')}))
    receipt={'schema':'budget-revision.v1','authorization':'explicit user increase (offline fixture)',
             'old_limits':original,'new_limits':requested}
    (tmp_path/'revision.json').write_text(json.dumps(receipt))
    output=tmp_path/'fresh'
    monkeypatch.setattr(sys,'argv',['-',str(output),str(source),'--execute',str(extra),
        str(tmp_path/'spec.json'),str(tmp_path/'revision.json') if approval else ''])
    def docker_only(command, **kwargs):
        assert command[:3]==['docker','run','--rm'] and '--network' in command
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(subprocess,'run',docker_only)
    return block,source,output,original,requested,receipt


def test_explicit_increase_keeps_original_costs_and_governor_identity(tmp_path,monkeypatch):
    block,source,output,original,requested,receipt=fixture(tmp_path,monkeypatch)
    before=(source/'budget.json').read_bytes()
    exec(compile(block,'production-launcher-budget','exec'),{})
    assert (source/'budget.json').read_bytes()==before
    ledger=json.loads((output/'budget.json').read_text())
    assert ledger['limits']==requested and len(ledger['attempts'])==2
    assert sum(a['accounted_tokens'] for a in ledger['attempts'].values())==300
    carry=json.loads((output/'cost_carryforward_manifest.json').read_text())
    assert carry['bank_inherited'] is False and carry['limits_unchanged'] is False
    assert carry['budget_revision']==receipt
    assert carry['source_budget_sha256']==hashlib.sha256(before).hexdigest()
    governor=BudgetGovernor(output/'budget.json',**requested)
    assert governor.headroom({})['tokens']==85000000-300
    assert governor.headroom({})['requests']==480000-2


def test_increase_without_receipt_stops_before_creating_fresh(tmp_path,monkeypatch):
    block,source,output,*_=fixture(tmp_path,monkeypatch,approval=False)
    before=(source/'budget.json').read_bytes()
    with pytest.raises(ValueError,match='explicit authorization'):
        exec(compile(block,'production-launcher-budget','exec'),{})
    assert not output.exists() and (source/'budget.json').read_bytes()==before


def test_mismatched_receipt_or_duplicate_cost_is_rejected(tmp_path,monkeypatch):
    block,source,output,original,requested,receipt=fixture(tmp_path,monkeypatch,conflict=True)
    before=(source/'budget.json').read_bytes()
    changed=deepcopy(receipt);changed['new_limits']['token_limit']+=1
    (tmp_path/'revision.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='resolved configuration'):
        exec(compile(block,'production-launcher-budget','exec'),{})
    (tmp_path/'revision.json').write_text(json.dumps(receipt))
    with pytest.raises(AssertionError,match='Conflicting HTTP identity'):
        exec(compile(block,'production-launcher-budget','exec'),{})
    assert not output.exists() and (source/'budget.json').read_bytes()==before


def test_unchanged_caps_remain_backward_compatible(tmp_path,monkeypatch):
    block,source,output,original,*_=fixture(tmp_path,monkeypatch,approval=False)
    (tmp_path/'base.json').write_text(json.dumps({'budget':original}))
    exec(compile(block,'production-launcher-budget','exec'),{})
    carry=json.loads((output/'cost_carryforward_manifest.json').read_text())
    assert carry['limits_unchanged'] is True and carry['budget_revision'] is None
    assert json.loads((output/'budget.json').read_text())['limits']==original
