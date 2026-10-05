"""T23: execute the two naturally learned CF2 Programs against real ALFWorld, no LLM."""
from copy import deepcopy
from pathlib import Path
import json
import os

import pytest

from atomic_skillgraph.empirical import IMPLEMENTATION_REVISION
from atomic_skillgraph.empirical.contracts import ValueStore
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.experiments.formal_log import tree_identity
from atomic_skillgraph.experiments.run_empirical import resolve_alfworld_tasks
from atomic_skillgraph.harness.registry import create_simple_harness
from test_cf3_takeover import transport, workflow


@pytest.fixture(scope='module')
def actual_program_cases(tmp_path_factory):
    location=os.environ.get('CF3_REVIEW_RUN')
    if not location:
        pytest.skip('T23 requires the preserved naturally learned 12+6 run; no replacement Bank')
    root=Path(location)
    assert json.loads((root/'completion.json').read_text())['status']=='completed'
    cases=[]
    for path in sorted((root/'val/traces').glob('*.json')):
        trace=json.loads(path.read_text())
        if trace['execution']['attempts']:
            attempt=trace['execution']['attempts'][0]
            assert attempt['terminal_by_program'] and trace['score']['hard']
            cases.append((trace,attempt['program_id']))
    assert len(cases)==2
    settings=json.loads((root/'val/config.json').read_text())
    settings['experiment']['implementation_revision']=IMPLEMENTATION_REVISION
    settings['runtime'].update(plan_execution_policy='empirical.recoverable-takeover.v1',dynamic_escape_limit=1)
    settings['experiment']['output_dir']=str(tmp_path_factory.mktemp('cf3_actual_executor'))
    if os.environ.get('CF3_EVIDENCE_DIR'):
        evidence=Path(os.environ['CF3_EVIDENCE_DIR']).parent/'actual_takeover'
        evidence.mkdir(parents=True,exist_ok=True)
        (evidence/'resolved_config.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2))
    adapter=create_simple_harness(settings)
    selected={trace['task']['task_id'] for trace,_ in cases}
    rows=[t for t in json.loads((root/'val_selection.json').read_text())['tasks'] if t['task_id'] in selected]
    tasks=resolve_alfworld_tasks(adapter,rows,canonical_split='val',
                                mapping_path=Path(settings['experiment']['output_dir'])/'task_identity_resolution.json')
    bank=root/'train/frozen_bank'
    before=tree_identity(bank)
    yield root,settings,adapter,{t.task_id:t for t in tasks},cases,before
    adapter.close()
    assert tree_identity(bank)==before


@pytest.mark.parametrize('index',[0,1])
def test_t23_real_usable_program_keeps_automatic_native_takeover(actual_program_cases,monkeypatch,index):
    root,settings,adapter,tasks,cases,before=actual_program_cases
    trace,program_id=cases[index]
    task=tasks[trace['task']['task_id']]
    node=trace['execution']['plan']['nodes'][0]
    inputs,missing=ValueStore(task).resolve(node['args'])
    assert not missing
    if not inputs:
        calls=[c['arguments'] for r in trace['requests'] for c in r.get('response',{}).get('tool_calls',[])
               if c['arguments'].get('action')=='call_program' and c['arguments'].get('name')==program_id]
        assert len(calls)==1 and not calls[0].get('argument_refs')
        inputs=calls[0]['arguments']
    plan=workflow([{'id':'validated','execution_mode':'program','program_id':program_id,
                    'args':{k:{'literal':v} for k,v in inputs.items()}}])
    sent=transport(monkeypatch,[('submit_plan',{'mode':'compose','workflow':plan})])
    monkeypatch.setenv(settings['llm']['api_key_env'],'test-only-key')
    system=EmpiricalSystem(deepcopy(settings),harness=adapter,readonly=True)
    try:
        result=system.run_task(task,learn=False,attempt_id='cf3_engineering_'+str(index))
        assert result['score']['hard'] and result['execution']['reason']=='environment_terminal'
        assert len(result['execution']['attempts'])==1 and result['execution']['attempts'][0]['program_id']==program_id
        assert result['execution']['attempts'][0]['terminal_by_program']
        assert [r['stage'] for r in result['requests']]==['planner'] and len(sent)==1
        events=result['tools']
        assert len(events)==result['execution']['attempts'][0]['calls']>=2
        assert all(e['state']=='finished' for e in events)
        assert result['native_call_attempts']==len(events)
        assert result['environment_steps']==sum(e.get('environment_step',0) for e in events)
        assert tree_identity(root/'train/frozen_bank')==before
        if os.environ.get('CF3_EVIDENCE_DIR'):
            evidence=Path(os.environ['CF3_EVIDENCE_DIR']).parent/'actual_takeover'
            (evidence/(str(index)+'_trace.json')).write_text(json.dumps({
                'kind':'execution-layer regression; real environment and naturally learned Program; intercepted Planner',
                'fresh_cf3_learning':False,'live_model_requests':0,'trace':result,
                'frozen_before_after':before},ensure_ascii=False,indent=2))
    finally:
        system.close()
