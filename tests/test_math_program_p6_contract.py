"""P6 protocol fixtures are artificial regression inputs, never learned production assets."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from atomic_skillgraph.empirical import math_operation_contract_p6 as c, math_program_p6 as m
from atomic_skillgraph.empirical.contracts import PublicTask, digest
from atomic_skillgraph.empirical.program_worker import ProgramWorker
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.harness.benchmarks import AnswerAdapter
from atomic_skillgraph.experiments import run_math_program_p6 as runner
from test_cf2_contracts import http, response
from test_math_program_p5 import cfg as old_cfg

# Handwritten ONLY for zero-HTTP contract/Worker tests.
FIXTURE="""import re
def run(ctx, inputs):
    text=inputs.get('statement')
    if not isinstance(text,str) or not text or len(text)>16000:
        return {'status':'needs_input','outputs':{}}
    pattern=PATTERN
    tokens=[{'operator':m.group(1),'variable':('\\\\'+m.group(2)+' ' if m.group(2) else '')+m.group(3),'start':m.start(),'end':m.end()} for m in re.finditer(pattern,text)]
    if not tokens:return {'status':'needs_input','outputs':{}}
    return {'status':'ok','outputs':{'quantifiers':tokens,'uninterpreted_predicates':True,'tested_cases':len(tokens),'proof_scope':'syntactic_order_only_not_truth_or_scope_proof'}}
""".replace('PATTERN',repr(c.QUANTIFIER.pattern))


def rows():
    return [{'task':{'task_id':f'fixture:{i}','physical_key':str(i),'split':'train',
        'goal':r'Given $\forall x\exists y\, P(x,y)$ compare the order of the displayed quantifiers. '+str(i),
        'inputs':{'choices':[{'label':'A','text':'A statement'},{'label':'B','text':'Another statement'}]}},
        'parent_outcome':'wrong','trace_sha256':digest(i),'computations':[],'public_process':None} for i in range(3)]


def proposal(unit=None, sources=None):
    sources=sources or rows();unit=unit or m.catalog(sources)[1][0];task=sources[0]['task'];kind=unit['operation_kind'];io=c.interface(kind)
    binding=c.ready_bindings(task,kind)
    return {'contract_version':c.CONTRACT,'decision':'program_candidate','source_case_ids':unit['source_case_ids'],
        'public_source_refs':[next(c.public_spans(r['task']))['span_id'] for r in sources if r['task']['physical_key'] in unit['source_case_ids']],
        'atomic':{'goal':'List explicit quantifier order','applicability':'Formal quantifiers are displayed',
            'preconditions':['Public text is available'],'operation':'Extract literal tokens and original offsets',
            'stop_conditions':['Do not infer predicate truth or full scope']},
        'program_spec':{'operation_kind':kind,**{k:io[k] for k in ('input_schema','output_schema','result_role','allowed_tools','proof_scope','probe_contract')},
            'input_bindings':binding['input_bindings']}}


def config(tmp):
    cfg=old_cfg(tmp);cfg['experiment']['math_program_method_version']=c.VERSION
    return cfg


@pytest.mark.parametrize('command',['prepare','check','report','execute'])
def test_cli_dispatch_preserves_output_and_options(command,monkeypatch,capsys):
    import sys
    argv=['p6',command,'--output','/fixed/run']
    if command=='execute':argv+=['--env-file','/unused.env']
    seen=[]
    def entry(root,**kwargs):
        seen.append((root,kwargs));return {'status':'fixture_no_http'}
    monkeypatch.setattr(runner,command,entry);monkeypatch.setattr(sys,'argv',argv)
    runner.main()
    assert seen[0][0]=='/fixed/run' and json.loads(capsys.readouterr().out)['status']=='fixture_no_http'
    if command=='execute':assert seen[0][1]['paid_approved'] is False


def test_types_diagnostics_and_exact_contract():
    data=rows();unit=m.catalog(data)[1][0];v=proposal(unit,data)
    assert c.validate_spec(v,unit,data)==v
    mutations=[('input_bindings',[]),('input_schema',[]),('result_role','final_answer'),
        ('probe_contract',{'kind':'independent_finite_oracle','output_field':'correct_answer'}),
        ('input_bindings',{'wrong':{'kind':'runtime_required','reason':'missing'}})]
    for key,bad in mutations:
        value=deepcopy(v);value['program_spec'][key]=bad
        with pytest.raises(c.ContractError) as caught:c.validate_spec(value,unit,data)
        assert all({'field_path','expected','actual_type','source_ref'}<=set(e) for e in caught.value.diagnostics)
    for bad in (None,[],True,4,{'decision':[]}):
        with pytest.raises(c.ContractError):c.validate_spec(bad,unit,data)
    value=deepcopy(v);value['source_case_ids']=[{}]
    with pytest.raises(c.ContractError):c.validate_spec(value,unit,data)
    value=deepcopy(v);value['program_spec']['input_bindings']['statement']['span_id']='made_up'
    with pytest.raises(c.ContractError,match='source_reference_unknown'):c.validate_spec(value,unit,data)


def test_abstention_and_source_isolation():
    unit=m.catalog(rows())[1][0]
    v={'contract_version':c.CONTRACT,'decision':'no_change','atomic':[],'source_case_ids':['test']}
    assert c.validate_spec(v,unit,rows())=={'contract_version':c.CONTRACT,'decision':'no_change'}
    v=proposal();v['source_case_ids']=['unassigned']
    with pytest.raises(c.ContractError,match='source_identity_invalid'):c.validate_spec(v,unit,rows())
    demands,units,transfer=m.catalog(rows())
    assert transfer==['2'] and units[0]['source_case_ids']==['0','1']
    assert {r['task']['physical_key'] for r in m.material(units[0],rows(),[])['train_sources']}=={'0','1'}


def test_no_fictitious_matrix_or_semantics():
    task=rows()[0]['task'];task['goal']='For every solution of an infinite PDE there exists a unique limit.'
    assert c.ready_bindings(task,'finite_relation_quantifiers') is None
    assert c.ready_bindings(task,'quantifier_dependency_audit') is None
    task['goal']='The complete finite relation is given as finite_relation = [[true,false],[false,true]].'
    binding=c.ready_bindings(task,'finite_relation_quantifiers')
    assert binding['grounded_task_inputs']=={'relation':[[True,False],[False,True]]}
    spec={'operation_kind':'finite_relation_quantifiers',**{k:v for k,v in c.interface('finite_relation_quantifiers').items() if k!='semantics'},
        'input_bindings':{'relation':{'kind':'runtime_required','reason':'not given'}},'demonstration_input':{'relation':[[True]]}}
    assert c.bind(spec,{}) is None
    assert c.oracle('finite_relation_quantifiers',spec['demonstration_input'])['tested_cases']==1


def test_pairwise_exact_not_cartesian():
    inputs={'left':[[1,2],[2,3],[-1,2],[1000000,999999],[0,1]],'right':[[2,4],[3,4],[1,-2],[999999,1000000],[1,1]]}
    out=c.oracle('exact_rational_comparison',inputs)
    assert out['tested_cases']==5 and out['comparisons']==[0,-1,0,1,-1]
    for right in (inputs['right'][:-1],[[1,0]]*5):
        with pytest.raises(c.ContractError):c.oracle('exact_rational_comparison',{**inputs,'right':right})


def test_task_input_cannot_mix_sources():
    data=rows()[:2]
    for row in data:row['task']['goal']='rational_pairs = {"left":[[1,2]],"right":[[3,4]]}'
    spans={s['span_id']:s for row in data for s in c.public_spans(row['task'])}
    kind='exact_rational_comparison';io=c.interface(kind)
    spec={'operation_kind':kind,**{k:v for k,v in io.items() if k!='semantics'},
        'input_bindings':c.ready_bindings(data[0]['task'],kind)['input_bindings']}
    spec['input_bindings']['right']=c.ready_bindings(data[1]['task'],kind)['input_bindings']['right']
    with pytest.raises(c.ContractError,match='cross_source_binding'):c.bind(spec,spans)


def test_rejected_recovery_never_repays_or_changes_feedback(tmp_path,monkeypatch):
    cfg=config(tmp_path);s=EmpiricalSystem(cfg,harness=AnswerAdapter('livemath',{}))
    unit=m.catalog(rows())[1][0];plan={'code':{},'run_id':'p6_fixture'};p=proposal(unit)
    p['program_spec']['input_bindings']=[]
    seen=http(monkeypatch,[response([p],name='submit_math_program_spec')])
    def invoke():
        return runner.receive(s,plan,unit,tmp_path/'unit','extractor',c.PROMPT,m.material(unit,rows(),[]),
            'submit_math_program_spec',c.PROPOSAL,'math_program_spec',lambda x:c.validate_spec(x,unit,rows()))
    try:
        with pytest.raises(c.ContractError) as first:invoke()
        with pytest.raises(c.ContractError) as recovered:invoke()
        assert first.value.diagnostics==recovered.value.diagnostics and len(seen)==1
        next((tmp_path/'unit').glob('rejected_*.json')).unlink()
        with pytest.raises(RuntimeError,match='without its diagnostic receipt'):invoke()
        assert len(seen)==1
    finally:s.close()


@pytest.mark.parametrize('bad,code',[
    ({'source':FIXTURE,'note':'extra'},'unexpected_field'),
    ({'source':'import sympy\ndef run(ctx,inputs):return {}'},'disallowed_import'),
    ({'source':'def run(ctx,inputs):return open("/hidden")'},'disallowed_authority'),
    ({'source':'def run(ctx,inputs):return ctx.call("score",{})'},'disallowed_authority'),
    ({'source':'def run(ctx,inputs):return {"correct_answer":"A"}'},'answer_content_forbidden')])
def test_code_boundary(bad,code):
    with pytest.raises(c.ContractError,match=code):c.validate_source(bad)


def test_real_provider_response_recovery_and_abstention(tmp_path,monkeypatch):
    cfg=config(tmp_path);s=EmpiricalSystem(cfg,harness=AnswerAdapter('livemath',{}))
    unit=m.catalog(rows())[1][0];plan={'code':{},'run_id':'p6_fixture'};p=proposal(unit)
    seen=http(monkeypatch,[response([p],name='submit_math_program_spec'),response([{'source':FIXTURE}],name='submit_program')])
    for _ in range(2):
        v=runner.receive(s,plan,unit,tmp_path/'unit','extractor',c.PROMPT,m.material(unit,rows(),[]),
            'submit_math_program_spec',c.PROPOSAL,'math_program_spec',lambda x:c.validate_spec(x,unit,rows()))
        assert v==p
        a=runner.receive(s,plan,unit,tmp_path/'unit','tool_builder',c.BUILD_PROMPT,{'program_spec':p['program_spec']},
            'submit_program',c.BUILD,'math_program_build',lambda x:m.program(x,p['program_spec'],cfg['program_environment']))
        assert a=={'source':FIXTURE}
    assert len(seen)==2;s.close()
    assert all(x['thinking']['type']=='disabled' for x in seen)
    s=EmpiricalSystem(config(tmp_path/'other'),harness=AnswerAdapter('livemath',{}))
    seen=http(monkeypatch,[response([{'contract_version':c.CONTRACT,'decision':'no_change','atomic':[]}],name='submit_math_program_spec')])
    value=runner.receive(s,plan,unit,tmp_path/'otherunit','extractor',c.PROMPT,m.material(unit,rows(),[]),
        'submit_math_program_spec',c.PROPOSAL,'math_program_spec',lambda x:c.validate_spec(x,unit,rows()))
    assert value['decision']=='no_change' and 'atomic' not in value;s.close()


def test_real_docker_local_transfer_and_interruption(tmp_path):
    if not runner.p5.docker_ready()['ready']:pytest.skip('Locked Linux Docker required')
    cfg=config(tmp_path);data=rows();unit=m.catalog(data)[1][0];p=proposal(unit,data)
    program=m.program({'source':FIXTURE},p['program_spec'],cfg['program_environment'])
    worker=ProgramWorker(cfg['program_worker']);result=m.qualify(worker,program,p,data,['2'],tmp_path/'probe')
    assert result['local_algorithm_checked'] and result['task_transfer_checked'] and not result['formal_usable']
    assert all(r['grounding']['physical_key']=='2' for r in result['transfer_receipts'])
    assert m.qualify(worker,program,p,data,['2'],tmp_path/'probe')==result
    receipt=result['receipts'][0];path=tmp_path/'probe/worker'/(receipt['id']+'.json');path.unlink()
    with pytest.raises(RuntimeError,match='Unresolved'):m.qualify(worker,program,p,data,['2'],tmp_path/'probe')


def test_real_p5_failures_and_original_negative(tmp_path):
    from atomic_skillgraph.experiments.math_program_p6_audit import replay
    root=Path('/home/yangchengyu/livemath_p5_trace_seed42_20261015')
    if not root.exists() or not runner.p5.docker_ready()['ready']:pytest.skip('Real P5 and locked Docker required')
    result=replay(root,tmp_path)
    assert result['rejected_count']==20 and result['old_passed_program_count']==51
    assert result['diagnostic_counts']=={'schema_contract_mismatch':15,'unexpected_field':1,'response_json_invalid':3,'disallowed_import':1}
    assert len(result['old_capability_families'])==2
    assert result['worker_negative_replay'] and all(r['unchanged_original_failure'] for r in result['worker_negative_replay'])
    assert result['historical_tree_before']==result['historical_tree_after']
