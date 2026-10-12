"""Final window metadata must fit the real Provider request, including saved failures."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from atomic_skillgraph.empirical.budget_governor import BudgetGovernor, input_token_bound
from atomic_skillgraph.empirical.contracts import digest
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.empirical.finish_evidence import build_finish_evidence
from atomic_skillgraph.empirical.public_results import FILE_COMPUTE, PROGRAM_EXECUTION, TEXT_PAGE
from atomic_skillgraph.experiments import audit_officeqa_prefix as gate
from atomic_skillgraph.experiments.benchmark_config import benchmark_config
from test_cf2_contracts import office, http, response
from test_office_r1 import record


@pytest.mark.parametrize('content', ['引号"\\\n'*100, list(range(40))])
def test_fitted_window_counts_final_metadata(content):
    context=TaskContext();rid=context.register('source', {'data':content})
    window=context.result_window({'result_id':rid,'path':['data']})
    limit=input_token_bound(window)
    fitted=context.fit_window(window, lambda w:input_token_bound(w)<=limit)
    assert fitted['content'] and len(fitted['content'])<len(window['content'])
    assert input_token_bound(fitted)<=limit and fitted['truncated']
    assert fitted['content_hash']==digest(fitted['content'])
    assert fitted['end']==fitted['next_offset']==fitted['offset']+len(fitted['content'])


@pytest.mark.parametrize('uid', ['UID0208','UID0148'])
def test_saved_finish_overflow_reaches_production_transport(tmp_path,monkeypatch,uid):
    archive=os.environ.get('OFFICE_FINISH_ARCHIVE')
    if not archive:pytest.skip('Set the immutable Office prefix for saved finish playback')
    root=Path(archive);task_id='officeqa:'+uid
    state=json.loads((root/'train/checkpoints'/task_id/'1/state.json').read_text())
    request=next(r for r in state['trace']['requests'] if r['purpose']=='finish_only' and r.get('material_refusal'))
    assert request['material_refusal']['wire_bytes']>16384 and not request.get('http_attempts')
    s=office(tmp_path);s.config=benchmark_config(s.config,'officeqa');s.phase='test'
    s.budget_governor=BudgetGovernor(tmp_path/'budget.json',token_limit=96000,request_limit=1,finish_reserve=0)
    s.request_attribution={'parent_scope':'saved-finish-fixture','parent_token_limit':96000}
    context=TaskContext(s.config['runtime'])
    for key,value in state['executor_state']['context'].items():setattr(context,key,deepcopy(value))
    s.task_context=context
    material=deepcopy(state['finish_prepared']['evidence']['materials']);material.pop('finish_evidence')
    prompt=request['messages'][0]['content'];sent=http(monkeypatch,[response(content='fixture',finish='stop')])
    try:
        finish=s.prepare_finish(material,prompt)
        assert finish['serialized_input_bytes']<=finish['input_max_bytes']==16384
        assert finish['materials']['original_goal']==material['original_goal']
        assert finish['materials']['finish_evidence']
        assert s.agent('runtime',prompt,finish['materials'],None,None,decision_purpose='finish_only')=='fixture'
        assert len(sent)==1 and finish['serialized_input_bytes']==input_token_bound(sent[0])
        assert 'tools' not in sent[0] and sent[0]['max_tokens']==2048
        assert len(s.budget_governor.state['attempts'])==1 and not s.budget_governor.state['unknown_billing']
        record('finish_window_'+uid, {'original_wire_bytes':request['material_refusal']['wire_bytes'],
            'new_wire_bytes':input_token_bound(sent[0]),'output_cap':2048,'actual_model_http':0})
    finally:s.close()


def test_gate_still_blocks_runtime_payload_overflow():
    with pytest.raises(AssertionError,match='non-learning payload overflow'):
        gate.capacity_refusal({'stage':'runtime','purpose':'finish_only','error':'material_too_large',
            'material_refusal':{'payload':{}}}, {})


@pytest.mark.parametrize('origin,field', [(FILE_COMPUTE,'data'), (PROGRAM_EXECUTION,'outputs')])
def test_finish_keeps_acquired_computation_before_long_pages(origin,field):
    context=TaskContext()
    page=context.register('page',{'data':{'text':'long acquired table row '*900}},origin=TEXT_PAGE)
    host_field='worker_cpu_seconds' if origin==FILE_COMPUTE else 'cpu_seconds'
    output=context.register('calculation',{field:{'rounded':0.162},host_field:99},origin=origin)
    before=digest(context.results)
    context.sent_windows=context.displayed_windows(context.view(page))
    material={'original_goal':'Report the acquired calculation rounded to three places',
              'recent':[context.view(page),context.view(output)]}
    finish=build_finish_evidence(context,material,'finish',max_bytes=4500)
    windows=finish['materials']['finish_evidence']
    calculation=next(w for w in windows if w['source_ref']=={'result_id':output,'path':[field,'rounded']})
    assert calculation['content']=='0.162' and not calculation['truncated']
    assert windows.index(calculation)==0 and any(w['source_ref']['result_id']==page for w in windows)
    assert finish['serialized_input_bytes']<=4500 and digest(context.results)==before
    assert not any(w['source_ref']['path']==[host_field] for w in windows)
    assert all(w['content_hash']==digest(w['content']) for w in windows)
    blocked=build_finish_evidence(context,material,'finish',max_bytes=4500,allowed_result_ids={page})
    assert all(w['source_ref']['result_id']!=output for w in blocked['materials']['finish_evidence'])
    assert any(w['result_id']==output and w['reason']=='not_in_obtained_results' for w in blocked['unresolved_refs'])
