"""Exact saved refusal through the production agent; intercepted HTTP only."""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from atomic_skillgraph.empirical.budget_governor import input_token_bound
from atomic_skillgraph.empirical import IMPLEMENTATION_REVISION
from atomic_skillgraph.empirical.model_view import expand_material,executed_source_capacity
from atomic_skillgraph.empirical.prompts import LEARNER_PROMPT
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.harness.benchmarks import AnswerAdapter
from test_cf2_contracts import response


def test_exact_17408_byte_extractor_replay_preserves_source_and_scope(tmp_path,monkeypatch,record_property):
    source=Path(os.environ.get('LIVEMATH_CURRENT_SOURCE','/home/yangchengyu/livemath_p2_formal_seed42_20261013/livemath/seed42'))
    if not source.exists():pytest.skip('Real source trace is not available on this host')
    raw=json.loads((source/'train/requests/livemath:202512:31_attempt1.json').read_text())
    saved=next(row for row in raw['requests'] if row.get('material_refusal'))
    old=saved['material_refusal']['payload'];material=expand_material(json.loads(old['messages'][1]['content']))
    assert input_token_bound(old)==17408 and saved['purpose']=='extractor' and saved['stage']=='extractor'
    assert saved['material_refusal']['limit_bytes']==16384 and not saved['http_attempts']
    calls=[]
    def post(url,**kwargs):
        calls.append(deepcopy(kwargs['json']))
        value=response([{'decision':'no_change'}],name='submit_learning')
        return SimpleNamespace(ok=True,status_code=200,headers={},text='',raise_for_status=lambda:None,json=lambda:value)
    monkeypatch.setattr('requests.post',post);monkeypatch.setenv('MODEL_API_KEY','intercepted-not-a-secret')
    original=json.loads((source/'train/config.json').read_text())
    for enabled in (False,True):
        cfg=deepcopy(original);cfg['data_dir']=str(tmp_path/str(enabled))
        cfg['experiment']['output_dir']=str(tmp_path/str(enabled)/'replay-output')
        cfg['experiment']['implementation_revision']=IMPLEMENTATION_REVISION
        if enabled:cfg['learning']['source_program_payload_max_bytes']=24576
        system=EmpiricalSystem(cfg,harness=AnswerAdapter('livemath',{}))
        system.phase='train';system.budget_scope='exact-refusal-replay';system.audit_path=tmp_path/(str(enabled)+'.json')
        try:
            invoke=lambda:system.agent('extractor',old['messages'][0]['content'],material,'submit_learning',
                saved['tools'][0]['function']['parameters'],completion_override=saved['completion_cap'])
            if not enabled:
                with pytest.raises(ValueError,match='17408 > 16384'):invoke()
                assert not calls and system.requests[-1]['material_refusal']['wire_bytes']==17408
            else:
                invoke();assert len(calls)==1 and input_token_bound(calls[0])<=24576
                shown=expand_material(json.loads(calls[0]['messages'][1]['content']))
                assert shown==material and calls[0]['tools']==old['tools']
                assert shown['experience']['local_evidence']==material['experience']['local_evidence']
                assert shown['experience']['case_source']==material['experience']['case_source']
                assert system.requests[-1]['payload_max_bytes']==24576
                record_property('saved_prompt_wire_bytes',input_token_bound(calls[0]))
                system.agent('extractor',LEARNER_PROMPT,material,'submit_learning',
                    saved['tools'][0]['function']['parameters'],completion_override=saved['completion_cap'])
                assert len(calls)==2 and input_token_bound(calls[-1])<=24576
                assert expand_material(json.loads(calls[-1]['messages'][1]['content']))==material
                assert calls[-1]['tools']==old['tools']
                record_property('current_prompt_wire_bytes',input_token_bound(calls[-1]))
        finally:system.close()
    # Unconfigured callers, non-source materials, and altered source kinds retain 16 KiB.
    assert executed_source_capacity(material,{},16384)==16384
    altered=deepcopy(material);altered['experience']['local_evidence'][0]['kind']='invented'
    assert executed_source_capacity(altered,{'source_program_payload_max_bytes':24576},16384)==16384
