"""Intercept real Provider HTTP; assert wire, billing, scorer and replay invariants."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
import yaml

from atomic_skillgraph.empirical import math_effect_runtime as r
from atomic_skillgraph.empirical.budget_governor import BudgetGovernor
from atomic_skillgraph.empirical.contracts import PublicTask
from atomic_skillgraph.harness.benchmarks import AnswerAdapter
from test_math_effect_p7_contract import candidate
from test_cf2_contracts import http, response


def cfg():
    s=yaml.safe_load(Path('configs/livemath_effect_p7_seed42.yaml').read_text())
    return {'effect_p7':s,'llm':{'base_url':'https://test.invalid/v1','model':'deepseek-v4-flash','api_key_env':'MODEL_API_KEY'}}


def setup(tmp_path):
    t=PublicTask('x','key','For every positive integer x, compare the claims.',
        {'choices':[{'label':'A','text':'x>=1'},{'label':'B','text':'x<0'}]},'val')
    a=AnswerAdapter('livemath',{'x':{'choices':t.inputs['choices'],'correct_choice':t.inputs['choices'][0]}})
    g=BudgetGovernor(tmp_path/'budget.json',token_limit=1000000,request_limit=10,finish_reserve=0)
    return t,a,g


def test_single_wire_and_exact_resume(tmp_path,monkeypatch):
    t,a,g=setup(tmp_path);calls=http(monkeypatch,[response(content='<answer>A</answer>',finish='stop')]*2)
    off=r.solve(t,a,[],cfg(),tmp_path/'off',g);on=r.solve(t,a,[candidate()],cfg(),tmp_path/'on',g)
    assert off['score']['hard'] and on['score']['hard'] and on['selection']['selected']
    x,y=deepcopy(calls[0]),deepcopy(calls[1])
    # Helper records raw payloads; no tools/action directories anywhere in wire.
    assert 'tools' not in x and 'tools' not in y
    b=json.loads(y['messages'][1]['content']);b.pop('skills');y['messages'][1]['content']=json.dumps(b,ensure_ascii=False)
    assert x==y and x['max_tokens']==65536 and x['thinking']=={'type':'enabled'}
    assert r.solve(t,a,[candidate()],cfg(),tmp_path/'on',g)==on and len(calls)==2


def test_truncation_recovery_once_and_no_censored_success(tmp_path,monkeypatch):
    t,a,g=setup(tmp_path)
    raw=response(content='',finish='length');raw['choices'][0]['message']['reasoning_content']='数学'*10000
    calls=http(monkeypatch,[raw,response(content='<answer>A</answer>',finish='stop')])
    result=r.solve(t,a,[],cfg(),tmp_path/'run',g)
    assert result['first_truncated'] and result['finish_state']=='valid' and not result['budget_censored']
    assert len(calls)==2 and calls[1]['max_tokens']==16384 and calls[1]['reasoning_effort']=='low'
    tail=json.loads(calls[1]['messages'][1]['content'])['previous_reasoning_tail']
    assert len(tail['text'].encode())<=16384 and tail['truncated']
    assert len(result['raw_usage'])==2
    r.verify_trace(result)
    r.solve(t,a,[],cfg(),tmp_path/'run',g);assert len(calls)==2


def test_invalid_twice_stops_and_unknown_bill_cannot_replay(tmp_path,monkeypatch):
    t,a,g=setup(tmp_path);calls=http(monkeypatch,[response(content='undecided',finish='stop')]*2)
    result=r.solve(t,a,[],cfg(),tmp_path/'bad',g)
    assert not result['valid'] and result['http']==2
    unresolved=tmp_path/'unknown/main';unresolved.mkdir(parents=True)
    (unresolved/'started.json').write_text('{}')
    with pytest.raises(RuntimeError,match='no blind'):r.solve(t,a,[],cfg(),tmp_path/'unknown',g)
    assert len(calls)==2


def test_tampered_physical_receipt_rejected(tmp_path,monkeypatch):
    t,a,g=setup(tmp_path);http(monkeypatch,[response(content='A',finish='stop')])
    result=r.solve(t,a,[],cfg(),tmp_path/'run',g)
    p=tmp_path/'run/main/requests.json';p.write_text('[]')
    with pytest.raises(ValueError,match='Physical request'):r.verify_trace(result)
