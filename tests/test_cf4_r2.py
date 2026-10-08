"""CF4 R2 Z01–Z14: production modules, saved records, intercepted HTTP only."""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.bank_view import BankView
from atomic_skillgraph.empirical.budget_governor import BudgetGovernor
from atomic_skillgraph.empirical.checkpoint import TaskCheckpoint
from atomic_skillgraph.empirical.contracts import PublicTask, digest, object_schema
from atomic_skillgraph.empirical.finish_evidence import build_finish_evidence
from atomic_skillgraph.empirical.program_submission import (ProgramContractError, effective_output_schema,
    normalize_program_result, prepare_program_submission, submission_contract, validate_program_declaration)
from atomic_skillgraph.empirical.system import EmpiricalSystem, validate_config
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.experiments.episode_projection import episode_projection
from atomic_skillgraph.experiments.recover_empirical import inspect_source, recover, read
from atomic_skillgraph.experiments.run_cf4_r2_diagnostic import verify_d3_wire, prepare, execute, view_identity, d1, d2
from atomic_skillgraph.experiments.run_empirical import run
from atomic_skillgraph.harness.benchmarks import SpreadsheetAdapter
from atomic_skillgraph.harness.simple_protocol import Broker
from atomic_skillgraph.agents.provider import OpenAICompatibleConfig, OpenAICompatibleProvider
from atomic_skillgraph.core.errors import BudgetExhausted
from test_empirical import config_for, program, worker
from test_cf2_contracts import office, http, response

REVIEW = Path(os.environ.get('CF4_R2_REVIEW','/home/yangchengyu/cf4_r1_review_snapshots/20261007_201002'))
SOURCE = Path(os.environ.get('CF4_R2_SOURCE','/home/yangchengyu/cf4_r1_nonalf_formal_seed42_20261007/spreadsheetbench/seed42'))


def snapshot_bank(benchmark):
    if not REVIEW.exists(): pytest.skip('Requires the archived CF4 R1 review records')
    return Bank(REVIEW/'runs'/benchmark/'seed42/train/bank',readonly=True)


def sheet(tmp_path):
    config=config_for(tmp_path/'bank')
    adapter=SpreadsheetAdapter({'sheet':{'public_files':{}}},config)
    task=PublicTask('sheet','sheet-physical','Publish a workbook')
    adapter.reset(task)
    return config,adapter,task


def file_program(*,declared=('solution.py','case1_result.xlsx'), count=1):
    source="def run(ctx, inputs):\n    open('/workspace/solution.py','w').write('new solution')\n    open('/workspace/case1_result.xlsx','wb').write(b'new workbook')\n    return "+repr({'status':'ok','outputs':{'count':count,'files':list(declared)}})
    p=program(source,outputs=object_schema({'count':{'type':'integer'}},['count']))
    p.update(result_role='final_files',allowed_tools=[])
    return p


def test_z01_real_57_declaration_rejected_without_keyerror():
    bank=snapshot_bank('spreadsheetbench')
    try:
        p=bank.get('program_f9b533e5085af63f923d3914aa6b1562ad1eabe197cfc0975e508a8dec06acb3')
        with pytest.raises(ProgramContractError,match='program_submission_kind_mismatch'):
            validate_program_declaration(p,{'final_submission_kind':'files','publication_contract':{'supported':True,'required_files':[]}})
    finally: bank.close()


@pytest.mark.parametrize('schema,answer,code',[(object_schema(), 'x','program_answer_schema_invalid'),
    (object_schema({'answer':{'type':'integer'}},['answer']),1,'program_answer_schema_invalid'),
    (object_schema({'answer':{'type':'string'}},['answer']),'','program_answer_empty')])
def test_z02_answer_contract_before_scorer(tmp_path,schema,answer,code):
    s=office(tmp_path); p=program('',outputs=schema); p['result_role']='final_answer'
    s.adapter.evaluate=lambda *a:pytest.fail('Invalid contract must not reach scoring')
    result=normalize_program_result(s.adapter,p,{'status':'ok','outputs':{'answer':answer}})
    assert result['status']=='execution_error' and result['error_code']==code
    s.close()


@pytest.mark.parametrize('declared,expected',[(('solution.py','case1_result.xlsx'),'ok'),(('case1_result.xlsx',),'execution_error'),((),'execution_error')])
def test_z03_publication_requires_this_invocation(tmp_path,worker,declared,expected):
    worker.settings['wall_timeout_seconds']=10
    _,adapter,task=sheet(tmp_path)
    bank=Bank(tmp_path/'bank'); p=bank.put('program',file_program(declared=declared))
    result=worker.execute(p,{},Broker(adapter,24))
    assert result['status']==expected
    if expected=='ok':
        assert result['publication_receipt']['host_verified'] and result['submission_preparation']['status']=='ready'
        unchanged=deepcopy(result); unchanged['publication_receipt']['before']=unchanged['publication_receipt']['after']
        assert prepare_program_submission(adapter,p,unchanged)['error_code']=='program_publication_unchanged'
    adapter.close(); bank.close()


def test_z04_strict_schema_composition_paths_and_datetime(tmp_path,worker):
    worker.settings['wall_timeout_seconds']=10
    _,adapter,_=sheet(tmp_path)
    schema=object_schema({'count':{'type':'integer'}},['count'])
    schema['allOf']=[deepcopy(schema)]
    effective=effective_output_schema(schema,result_role='final_files',publication_contract={'supported':True})
    assert effective['additionalProperties'] is False and schema['properties'].keys()=={'count'}
    assert 'files' in effective['allOf'][0]['properties']
    bank=Bank(tmp_path/'bank')
    for source in ("def run(ctx, inputs):\n    return {'status':'ok','outputs':{'count':1,'files':['../escape']}}",
                   "def run(ctx, inputs):\n    import datetime\n    return {'status':'ok','outputs':{'count':datetime.datetime.now()}}"):
        p=bank.put('program',{**file_program(),'source':source})
        assert worker.execute(p,{},Broker(adapter,24))['status']=='execution_error'
    adapter.close(); bank.close()


def test_z05_worker_complete_strict_and_direct_provenance(tmp_path,worker):
    worker.settings['wall_timeout_seconds']=10
    from atomic_skillgraph.empirical.executor import Executor
    config,adapter,task=sheet(tmp_path); bank=Bank(tmp_path/'bank')
    p=bank.put('program',file_program())
    for i in range(2): bank.record({'id':str(i),'program_id':p['id'],'task_key':str(i),'origin':'train_test','outcome':'positive','basis':'local_check'})
    executor=Executor(bank,lambda *a,**k:pytest.fail('Ready Program must bypass Agent'),worker,object())
    executor.checkpoint=TaskCheckpoint(tmp_path/'checkpoint')
    result=executor.run(task,adapter,Broker(adapter,24),{'nodes':[{'id':'files','execution_mode':'program','program_id':p['id'],'args':{}}]})
    assert result['reason']=='plan_submitted' and result['attempts'][0]['submission_by_program']
    assert result['attempts'][0]['output_contract_status']=='valid'
    adapter.reset(task)
    cold=executor.run(task,adapter,Broker(adapter,24),{})
    assert cold==result and adapter.submission_ready({},result_role='intermediate')
    adapter.close(); bank.close()


@pytest.mark.parametrize('interruption',['copy','before_commit','after_commit'])
def test_z06_z08_z09_recovery_zero_provider_and_idempotence(tmp_path,monkeypatch,interruption):
    if not SOURCE.exists(): pytest.skip('Requires original stopped task57')
    monkeypatch.setattr(EmpiricalSystem,'run_task',lambda *a,**k:pytest.fail('Recovery must not solve'))
    monkeypatch.setattr(EmpiricalSystem,'agent',lambda *a,**k:pytest.fail('Recovery must not call a model'))
    output=tmp_path/'child'
    with pytest.raises(InterruptedError): recover(SOURCE,output,'spreadsheet:51-12',interrupt_after=interruption)
    receipt=recover(SOURCE,output,'spreadsheet:51-12')
    assert receipt['committed_tasks']==57 and receipt['successes']==44 and receipt['learning_cases']==42
    assert receipt['inherited_task_tokens']==397659 and len(receipt['original_request_ids'])==12
    assert receipt['new_model_calls']==0 and recover(SOURCE,output,'spreadsheet:51-12')==receipt
    projection=read(output/'episode_projection.json'); assert len(projection)==57
    bank=Bank(output/'train/bank',readonly=True)
    assert len(bank.train_cases())==42
    bad='program_f9b533e5085af63f923d3914aa6b1562ad1eabe197cfc0975e508a8dec06acb3'
    assert bank.get(bad)['state']=='disabled' and not any(p['id']==bad for p in bank.program_options('count',allow_candidate=True))
    bank.close()


def test_z07_worker_finished_resumes_scoring_not_worker(tmp_path,worker):
    worker.settings['wall_timeout_seconds']=10
    s=office(tmp_path); config=s.config; s.checkpoint=TaskCheckpoint(tmp_path/'checkpoint')
    original=s.adapter.evaluate; failures=[True]
    def factory():
        from atomic_skillgraph.harness.benchmarks import OfficeAdapter
        a=OfficeAdapter({'office':{'answer':'42'}},config)
        evaluate=a.evaluate
        def score(sealed):
            if failures.pop() if failures else False: raise OSError('Injected scorer infrastructure failure')
            return evaluate(sealed)
        a.evaluate=score
        return a
    s.adapter_factory=factory; s.worker=worker
    p=s.bank.put('program',{**program("def run(ctx, inputs):\n    return {'status':'ok','outputs':{'answer':'42'}}",outputs=object_schema({'answer':{'type':'string'}},['answer'])),'result_role':'final_answer','allowed_tools':[]})
    with pytest.raises(OSError): s.test_program(p,{},s.adapter.task,trial_id='saved-worker',continuation=False)
    count=len(worker.invocations)
    record=s.test_program(p,{},s.adapter.task,trial_id='saved-worker',continuation=False)
    assert record['outcome']=='positive' and len(worker.invocations)==count
    assert record['trial_execution_id']
    s.close()


def test_z10_views_block_direct_ids_and_preserve_digest(tmp_path):
    bank=Bank(tmp_path/'bank'); p=bank.put('program',program('def run(ctx, inputs):\n    return {}'))
    for i in range(2): bank.record({'id':str(i),'program_id':p['id'],'task_key':'same','origin':'train_test','outcome':'positive','basis':'local_check'})
    assert bank.get(p['id'])['state']=='candidate'
    bank.freeze(tmp_path/'frozen'); bank.close()
    frozen=Bank(tmp_path/'frozen',readonly=True); before=frozen.digest(); view=BankView(frozen,'learned_assets_off')
    assert view.get(p['id']) is None and view.all('skill')==[] and view.train_cases()==[] and view.digest()==before
    identities=[view_identity(SimpleNamespace(bank=frozen),mode) for mode in ('on','guidance_off','learned_assets_off')]
    assert len({i['view_hash'] for i in identities})==3
    assert {i['source_bank_digest'] for i in identities}=={before}
    with pytest.raises(RuntimeError): view.save_case(None,None)
    frozen.close()


def test_z11_saved_finish_evidence_and_77_responses():
    from atomic_skillgraph.empirical.answer_status import classify_answer
    if not REVIEW.exists(): pytest.skip('Requires archived responses')
    math=REVIEW/'runs/livemath/seed42'; classifications=[]
    for split in ('train','val'):
        for path in (math/split/'requests').glob('*.json'):
            r=next(r for r in read(path)['requests'] if r['stage']=='runtime')
            classifications.append(classify_answer(r['response']['content'],r['response']['finish_reason']))
    assert len(classifications)==77 and sum(c['answer_status']=='truncated_empty' for c in classifications)==19
    for tid in ('officeqa:UID0115','officeqa:UID0148'):
        root=REVIEW/'runs/officeqa/seed42/train'
        state=read(root/'checkpoints'/tid/'1/state.json')['executor_state']
        context=TaskContext()
        for k,v in state['context'].items(): setattr(context,k,v)
        r=next(r for r in reversed(read(root/'requests'/(tid+'_attempt1.json'))['requests']) if r.get('purpose')=='finish_only')
        before=digest(context.results)
        evidence=build_finish_evidence(context,json.loads(r['messages'][1]['content']),r['messages'][0]['content'])
        assert evidence['serialized_input_bytes']<=65536 and evidence['included_refs'] and not evidence['unresolved_refs']
        assert digest(context.results)==before


def test_z12_actual_http_admission_purpose_and_unknown_billing(tmp_path,monkeypatch):
    s=office(tmp_path)
    s.config['llm']['purpose_overrides']={'finish_only':{'protocol':{'thinking_type':'disabled'},'max_completion_tokens':512}}
    s.budget_governor=BudgetGovernor(tmp_path/'ledger.json',token_limit=100000,finish_reserve=1000,request_limit=2)
    seen=http(monkeypatch,[response(content='42',finish='stop')])
    value=s.agent('runtime','finish',{},None,None,decision_purpose='finish_only')
    assert value=='42' and seen[0]['max_tokens']==512 and seen[0]['thinking']=={'type':'disabled'}
    assert s.provider('runtime','runtime').config.thinking_type=='enabled'
    assert len(s.budget_governor.state['attempts'])==1
    with pytest.raises(BudgetExhausted): s.budget_governor.admit('long',{'max_tokens':1,'messages':['x'*100001]}, {})
    s.budget_governor.admit('unknown',{'max_tokens':1},{'purpose':'finish_only'})
    s.budget_governor.complete({'request_id':'unknown','raw_usage':None,'outcome':'error'})
    with pytest.raises(BudgetExhausted): s.budget_governor.admit('next',{'max_tokens':1},{})
    s.close()


def test_z12_retry_attempt_cannot_escape_unknown_billing(tmp_path,monkeypatch):
    import requests
    monkeypatch.setenv('MODEL_API_KEY','fixture-key')
    provider=OpenAICompatibleProvider(OpenAICompatibleConfig(base_url='https://fixture.invalid',model='fixture',
        api_key_env='MODEL_API_KEY',max_completion_tokens=32,max_retries=1))
    provider.budget_governor=BudgetGovernor(tmp_path/'retry-ledger.json',token_limit=100000,finish_reserve=0,request_limit=4)
    calls=[]
    def post(*a,**k): calls.append(k['json']); raise requests.exceptions.Timeout('intercepted')
    monkeypatch.setattr('atomic_skillgraph.agents.provider.requests.post',post)
    monkeypatch.setattr(provider,'_backoff',lambda *a,**k:None)
    with pytest.raises(BudgetExhausted): provider.complete([{'role':'user','content':'x'}],tools=[])
    assert len(calls)==1 and len(provider.request_records)==1 and provider.budget_governor.state['unknown_billing']


def test_z12_runtime_reserve_admits_only_one_finish(tmp_path,monkeypatch):
    s=office(tmp_path)
    s.config['llm']['runtime']['max_total_tokens_per_task']=1000
    s.config['llm']['purpose_overrides']={'finish_only':{'protocol':{'thinking_type':'disabled'},'max_completion_tokens':512}}
    seen=http(monkeypatch,[response(content='42',finish='stop')])
    with pytest.raises(BudgetExhausted,match='preserves'):
        s.agent('runtime','ordinary',{},'runtime_step',object_schema({'action':{'type':'string'}},['action']))
    assert not seen
    assert s.agent('runtime','finish',{},None,None,decision_purpose='finish_only')=='42' and len(seen)==1
    s.close()


def test_z13_d3_exact_historical_wire_and_wrong_resume(tmp_path):
    if not REVIEW.exists(): pytest.skip('Requires archived responses')
    manifest=prepare(REVIEW,tmp_path/'diagnostic')
    assert len(verify_d3_wire(manifest))==41
    config=config_for(tmp_path/'bank'); config['experiment']['output_dir']=str(tmp_path/'phase')
    task=PublicTask('x','physical','question',split='val')
    from atomic_skillgraph.harness.benchmarks import AnswerAdapter
    adapter=AnswerAdapter('searchqa',{'x':{'answers':['x']}})
    from atomic_skillgraph.experiments.run_empirical import write_json
    write_json(tmp_path/'phase/execution_manifest.json',{'wrong_identity':True})
    with pytest.raises(ValueError,match='identity'):
        run(config,[task],tmp_path/'phase',resume=True,adapter=adapter)


def test_z11_d2_production_entry_meters_saved_finish(tmp_path,monkeypatch):
    if not REVIEW.exists(): pytest.skip('Requires archived finish snapshot')
    seen=http(monkeypatch,[response(content='<answer>42</answer>',finish='stop')])
    from atomic_skillgraph.experiments.run_cf4_r2_diagnostic import settings
    def fixture_settings(*a,**k):
        config=settings(*a,**k); config['llm']['api_key_env']='MODEL_API_KEY'; return config
    monkeypatch.setattr('atomic_skillgraph.experiments.run_cf4_r2_diagnostic.settings',fixture_settings)
    result=d2('officeqa:UID0115',{'review_source':str(REVIEW)},tmp_path,
              BudgetGovernor(tmp_path/'ledger.json'))
    assert result['status']=='completed' and result['tokens']==5 and len(seen)==1
    assert seen[0]['max_tokens']==512 and seen[0]['thinking']=={'type':'disabled'}


def test_z12_rejected_builder_still_counts_actual_http(tmp_path,monkeypatch):
    if not REVIEW.exists(): pytest.skip('Requires archived candidate snapshot')
    seen=http(monkeypatch,[response(content='',finish='length')])
    from atomic_skillgraph.experiments.run_cf4_r2_diagnostic import settings
    def fixture_settings(*a,**k):
        config=settings(*a,**k); config['llm']['api_key_env']='MODEL_API_KEY'; return config
    monkeypatch.setattr('atomic_skillgraph.experiments.run_cf4_r2_diagnostic.settings',fixture_settings)
    manifest=prepare(REVIEW,tmp_path/'diagnostic')
    result=d1('officeqa',manifest,tmp_path/'diagnostic',BudgetGovernor(tmp_path/'ledger.json'))
    assert result['status']=='failed' and 'builder_submission_tool_mismatch' in result['error']
    assert result['builder_http']==1 and len(seen)==1 and not result['usable']


def test_z13_diagnostic_rejects_config_drift_before_http(tmp_path,monkeypatch):
    if not REVIEW.exists(): pytest.skip('Requires archived source config')
    root=tmp_path/'diagnostic'; manifest=prepare(REVIEW,root)
    next(iter(manifest['config_sources'].values()))['config_hash']='changed'
    (root/'diagnostic_manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr('atomic_skillgraph.agents.provider.requests.post',lambda *a,**k:pytest.fail('Config drift must not send HTTP'))
    with pytest.raises(ValueError,match='Diagnostic source config changed'): execute(root/'diagnostic_manifest.json')
    assert not (root/'STARTED.json').exists()


def test_z14_new_commit_boundary_skips_history_and_no_freeze(tmp_path,monkeypatch):
    config=config_for(tmp_path/'bank'); output=tmp_path/'phase'
    config['experiment']['output_dir']=str(output)
    from atomic_skillgraph.harness.benchmarks import AnswerAdapter
    tasks=[PublicTask(str(i),'physical'+str(i),'question') for i in range(4)]
    adapter=AnswerAdapter('searchqa',{t.task_id:{'answers':['x']} for t in tasks})
    seen=[]
    def solve(self,task,**kwargs):
        seen.append(task.task_id)
        return {'task':asdict(task),'execution':{'reason':'single_answer','attempts':[]},
                'score':{'hard':True,'raw_score':1},'usage':[],'requests':[]}
    from dataclasses import asdict
    monkeypatch.setattr(EmpiricalSystem,'run_task',solve)
    first=run(config,tasks,output,adapter=adapter,max_new_tasks=1)
    second=run(config,tasks,output,resume=True,adapter=adapter,max_new_tasks=2)
    assert first['tasks']==1 and second['tasks']==3 and seen==['0','1','2']
    assert not second['complete'] and second['frozen'] is None and not (output/'frozen_bank').exists()
