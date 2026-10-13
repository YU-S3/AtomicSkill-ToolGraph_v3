"""Versioned Office source selection and real zero-HTTP material replay."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import zipfile

import pytest

from atomic_skillgraph.empirical import OFFICE_SOURCE_METHOD_VERSION, OFFICE_SOURCE_MATERIAL_VERSION
from atomic_skillgraph.empirical.contracts import PublicTask, digest
from atomic_skillgraph.empirical.source_catalog import (catalog_material, bind_proposal, verified_sources,
    proposal_schema, PROMPT, repair_material)
from atomic_skillgraph.empirical.model_view import learning_case, expand_material, executed_source_capacity, shrink_learning_preview
from atomic_skillgraph.empirical.program_worker import validate_context_source, CONTEXT_EXAMPLES
from atomic_skillgraph.empirical.prompts import LEARNING, BUILDER_PROMPT, BUILD
from atomic_skillgraph.empirical.system import EmpiricalSystem, validate_config
from atomic_skillgraph.empirical.budget_governor import input_token_bound
from atomic_skillgraph.experiments.benchmark_config import benchmark_config
from atomic_skillgraph.harness.registry import create_simple_harness
from test_cf2_contracts import office, http, response


def configure(config):
    config=deepcopy(config)
    config['experiment'].update(benchmark='officeqa',officeqa_method_version=OFFICE_SOURCE_METHOD_VERSION)
    return validate_config(benchmark_config(config,'officeqa'))


@pytest.mark.parametrize('source', ['ctx = {}', 'for ctx in []: pass', 'context.call("read", {})', 'context.read_result("x")'])
def test_context_misuse_rejected_without_rewrite(source):
    with pytest.raises(ValueError,match='context_api_misuse'):validate_context_source(source)
    validate_context_source('def run(ctx, inputs):\n result = ctx.call("read", inputs)\n return result')


def test_catalog_binding_and_native_capacity_are_host_owned(tmp_path):
    s=office(tmp_path)
    task=s.adapter.task
    event={'event_id':'real','state':'finished','backend_invoked':True,'name':'execute_python',
        'arguments':{'source':'import json\nvalues=[2,4,6]\nprint(json.dumps({"mean":sum(values)/len(values)}))'},
        'result':{'accepted':True,'data':{'computed':{'mean':4.0}},'observation':'','done':False},
        'result_protocol':'file_compute.v1'}
    trace={'tools':[event],'execution':{'attempts':[],'reason':'finish'},'score':{'hard':False}}
    case=s.learner._experience(task,trace)
    material=catalog_material({'experience':learning_case(case,task.physical_key)})
    material['learning_material_version']=OFFICE_SOURCE_MATERIAL_VERSION
    card=material['experience']['local_evidence'][0]
    value=next(r for r in card['input_sources'] if r['value']==[2,4,6])
    output=next(r for r in card['output_references'] if r.get('path')==['data','computed','mean'])
    skill={'goal':'mean','result_role':'intermediate','input_schema':{'type':'object','properties':{'values':{'type':'array','items':{'type':'integer'}}},'required':['values']},
        'output_schema':{'type':'object','properties':{'mean':{'type':'number'}},'required':['mean']}}
    proposal={'decision':'propose_skill_and_program_spec','skill':skill,'realization_request':{'skill_id':'$new','action':'build',
        'case_bindings':[{'operation_id':card['operation_id'],'input_ids':{'values':value['id']},'output_ids':{'mean':output['id']}}]}}
    cases={task.physical_key:(task,case)}
    bound=bind_proposal(proposal,material,cases,s.bank)
    assert bound['realization_request']['case_bindings'][0]['inputs']=={'values':[2,4,6]}
    for field in ('operation_id','input_ids','output_ids'):
        bad=deepcopy(proposal);binding=bad['realization_request']['case_bindings'][0]
        binding[field]='fake' if field=='operation_id' else {'values':'fake'}
        with pytest.raises(ValueError):bind_proposal(bad,material,cases,s.bank)
    limits={'source_program_payload_max_bytes':24576}
    assert executed_source_capacity(material,limits,16384)==16384
    trusted=verified_sources([(task,case)])
    assert trusted and executed_source_capacity(material,limits,16384,trusted=trusted)==24576
    fake=deepcopy(case);fake['events'][0]['backend_invoked']=False
    assert not verified_sources([(task,fake)])
    fake=deepcopy(case);fake['local_evidence'][0]['reference']['data']={}
    assert not verified_sources([(task,fake)])
    original=card['source_excerpt']
    while shrink_learning_preview(material):pass
    assert card['source_excerpt']==original
    s.close()


def test_new_config_is_office_only_and_old_banks_cannot_be_upgraded(tmp_path):
    s=office(tmp_path);old=deepcopy(s.config)
    old['budget']={'token_limit':85000000,'request_limit':480000,'finish_reserve':0}
    new=configure(old)
    assert new['learning']['builder_payload_max_bytes']==24576
    assert new['learning']['structural_repair_payload_max_bytes']==16384
    assert old['learning']['builder_payload_max_bytes']==16384
    new['experiment']['benchmark']='searchqa'
    with pytest.raises(ValueError):validate_config(new)
    s.bank.save_case(s.adapter.task,{'local_evidence':[]})
    with pytest.raises(ValueError):s.bank.bind_office_method(OFFICE_SOURCE_METHOD_VERSION)
    s.close()


@pytest.fixture
def actual_archive():
    path=os.environ.get('OFFICE_SOURCE_ARCHIVE')
    if not path:pytest.skip('Set OFFICE_SOURCE_ARCHIVE for immutable real material gate')
    return zipfile.ZipFile(path)


def test_actual_74_hashes_and_26_production_payloads(actual_archive,tmp_path,monkeypatch):
    z=actual_archive;index=json.loads(z.read('FILE_INDEX.json'))
    for name,row in index.items():
        data=z.read(name)
        assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
    assert len(index)==74
    root=Path(json.loads(z.read('train/execution_manifest.json'))['config']['experiment']['output_dir'])
    config=configure(json.loads(z.read('train/config.json')))
    config['data_dir']=str(tmp_path/'bank');config['experiment']['output_dir']=str(tmp_path/'replay')
    config['llm']['api_key_env']='MODEL_API_KEY'
    Path(config['data_dir']).mkdir()
    source_db=sqlite3.connect((root/'bank/bank.sqlite3').as_uri()+'?mode=ro',uri=True)
    copy_db=sqlite3.connect(Path(config['data_dir'])/'bank.sqlite3');source_db.backup(copy_db);source_db.close()
    with copy_db:
        copy_db.execute('DELETE FROM realization_jobs')
        copy_db.execute('INSERT INTO metadata VALUES(?,?)',('officeqa_method_version',OFFICE_SOURCE_METHOD_VERSION))
    copy_db.close()
    s=EmpiricalSystem(config)
    s.phase='train';s.audit_path=tmp_path/'requests.json'
    sent=http(monkeypatch, [response([{'decision':'no_change'}],name='submit_learning') for _ in range(25)] +
              [response([{'source':'def run(ctx, inputs): return {"status":"ok","outputs":{}}'}],name='submit_program')])
    results=[]
    try:
        for name in sorted(n for n in z.namelist() if n.startswith('train/capacity_rejected/') and n.endswith('requests.json')):
            raw=json.loads(z.read(name));refusals=[r for r in raw['requests'] if r.get('material_refusal')]
            r=next((r for r in refusals if r['stage']=='extractor'),None)
            if not r:continue
            uid=name.split('/')[2]
            trace=json.loads((root/'traces'/('officeqa:'+uid+'.json')).read_text())
            task=PublicTask(**trace['task']);s.adapter.reset(task)
            exp=s.learner._experience(task,trace)
            before=len(sent)
            learned=s.learner.learn(task,trace)
            assert learned['decision']=='no_change' and len(sent)==before+1
            wire=sent[-1];shown=expand_material(json.loads(wire['messages'][1]['content']))
            original=expand_material(json.loads(r['material_refusal']['payload']['messages'][1]['content']))
            ref=original['experience']['local_evidence'][0]['local_evidence_ref']
            card=next(c for c in shown['experience']['local_evidence'] if c['local_evidence_ref']==ref)
            assert card['source_excerpt'] and card['input_sources'] and card['output_references']
            assert input_token_bound(wire)<=24576 and s.requests[-1]['payload_max_bytes']==24576
            results.append({'uid':uid,'stage':'extractor','old_wire':r['material_refusal']['wire_bytes'],
                'new_wire':input_token_bound(wire),'source_chars':len(card['source_excerpt']),'selector_ids':True})
        assert len(results)==25
        r=json.loads(z.read('builder/UID0069/request.json'))
        material=expand_material(json.loads(r['material_refusal']['payload']['messages'][1]['content']))
        # Reconstruct via current builder_material, not material_audit's projected length.
        request=material['build_request']
        db=sqlite3.connect((root/'bank/bank.sqlite3').as_uri()+'?mode=ro',uri=True)
        job=next(json.loads(x[0]) for x in db.execute('SELECT payload FROM realization_jobs')
                 if digest(json.loads(x[0])['case_bindings'])==request['binding_hash']);db.close()
        rebuilt=s.learner.builder_material(request['skill'],job['case_bindings'],material['examples'],
            {**material['future_program_api'],'workspace_capabilities':material['workspace_capabilities']})
        rebuilt['learning_material_version']=OFFICE_SOURCE_MATERIAL_VERSION
        s.agent('tool_builder',s.learner.builder_prompt(),rebuilt,'submit_program',BUILD,
            completion_override=r['completion_cap'])
        assert len(sent)==26 and input_token_bound(sent[-1])<=24576
        results.append({'uid':'UID0069','stage':'tool_builder','old_wire':r['material_refusal']['wire_bytes'],
            'new_wire':input_token_bound(sent[-1])})
        report={'verified_hashes':len(index),'admissible':len(results),'actual_model_http':0,'payloads':results}
        output=os.environ.get('OFFICE_SOURCE_GATE_DIR')
        if output:
            Path(output).mkdir(parents=True,exist_ok=True)
            (Path(output)/'material_replay.json').write_text(json.dumps(report,indent=2))
    finally:s.close()


def test_actual_val15_budget_counterfactual(actual_archive,tmp_path):
    from atomic_skillgraph.empirical.budget_governor import BudgetGovernor
    from atomic_skillgraph.core.errors import BudgetExhausted
    z=actual_archive;report=[]
    for name in sorted(n for n in z.namelist() if n.startswith('val/requests/') and n.endswith('requests.json')):
        uid=name.split('/')[2];audit=json.loads(z.read(name))
        trace=json.loads(z.read('val/sqlite_completed_results/'+uid+'_result.json'))
        runtime=[r for r in audit['requests'] if r['purpose']=='runtime' and r.get('http_attempts')]
        request=runtime[-1];actual=request['http_attempts'][-1]['final_payload_audit']
        wire={**actual['serialized_parameters'],'messages':actual['messages'],'tools':actual['tools']}
        assert input_token_bound(wire)==actual['serialized_payload_utf8_bytes']
        used=sum(h['raw_usage']['total_tokens'] for r in audit['requests'] if r['purpose']!='finish_only'
                 for h in r.get('http_attempts',[]))
        wire['max_tokens']=8192
        decisions={}
        for cap in (96000,192000):
            g=BudgetGovernor(tmp_path/(uid+str(cap)+'.json'),token_limit=85000000,request_limit=480000,finish_reserve=0)
            g.state['attempts']={'inherited_task_usage':{'status':'metered','reserved_tokens':used,
                'accounted_tokens':used,'context':{'parent_scope':uid}}}
            context={'parent_scope':uid,'parent_token_limit':cap,'parent_finish_reserve':18432,'purpose':'runtime'}
            try:g.admit('counterfactual',wire,context);decisions[str(cap)]='admitted'
            except BudgetExhausted:decisions[str(cap)]='refused'
        assert decisions['192000']=='admitted'
        report.append({'task':uid,'actual_finish_trigger':trace['execution'].get('finish_trigger'),
            'spent_before_finish':used,'replayed_wire_bytes':input_token_bound(wire),'next_output_floor':8192,
            'finish_reserve':18432,'admission':decisions,'old_tool_limit':24,'new_tool_limit':48})
    assert len(report)==15
    output=os.environ.get('OFFICE_SOURCE_GATE_DIR')
    if output:
        (Path(output)/'val15_budget_simulation.json').write_text(json.dumps({'actual_model_http':0,
            'counterfactual':'Replay the last actual Runtime payload at the post-response spending boundary; no invented next response or success.',
            'cases':report},indent=2))


def test_phase_cap_no_borrowing_and_frozen_method(tmp_path):
    from atomic_skillgraph.experiments.office_source_recompile import PhaseGovernor,PHASE_LIMITS
    from atomic_skillgraph.core.errors import BudgetExhausted
    from atomic_skillgraph.empirical.bank import Bank
    g=PhaseGovernor(tmp_path/'budget.json','recompile')
    g.state['attempts']={'old':{'reserved_tokens':3000000,'accounted_tokens':3000000,'status':'metered',
        'context':{'office_phase':'recompile'}}}
    with pytest.raises(BudgetExhausted):g.admit('new',{'max_tokens':1},{})
    assert len(g.state['attempts'])==1
    bank=Bank(tmp_path/'bank');bank.bind_office_method(OFFICE_SOURCE_METHOD_VERSION)
    manifest=bank.freeze(tmp_path/'frozen');bank.close()
    assert manifest['learning_material_version']==OFFICE_SOURCE_MATERIAL_VERSION
    frozen=Bank(tmp_path/'frozen',readonly=True);frozen.bind_office_method(OFFICE_SOURCE_METHOD_VERSION);frozen.close()


def test_actual_recompile_entry_checkpoint_no_runtime_no_duplicate_charge(actual_archive,tmp_path,monkeypatch):
    from atomic_skillgraph.experiments import office_source_recompile as entry
    z=actual_archive
    source=Path(json.loads(z.read('train/config.json'))['experiment']['output_dir']).parent
    monkeypatch.setattr(entry,'code_identity',lambda:{'git_sha':'offline-fixture-identity','tracked_dirty':False})
    root=tmp_path/'derived';plan=entry.prepare(source,root)
    configs=json.loads((root/'resolved_config.json').read_text())
    # Intercepted provider still uses the production request/admission/checkpoint path.
    seen=http(monkeypatch,[response([{'decision':'no_change'}],name='submit_learning')]*2)
    prior=json.loads((root/'budget.json').read_text())
    first=entry.recompile(root,max_new_records=1)
    second=entry.recompile(root,max_new_records=1)
    assert first['checked']==1 and second['checked']==2 and len(seen)==2
    assert not first['complete'] and not (root/'train/frozen_bank').exists()
    ledger=json.loads((root/'budget.json').read_text())
    assert len(ledger['attempts'])==len(prior['attempts'])+2
    assert all(ledger['attempts'][key]==value for key,value in prior['attempts'].items())
    assert all(p['tools'][0]['function']['name']=='submit_learning' for p in seen)
    assert configs['val']['runtime']['global_action_budget']==48 and configs['val']['budget']['eval_task_tokens']==192000
    assert configs['test']['runtime']['global_action_budget']==48
    entry.checked(root)
    with pytest.raises(ValueError,match='approved Val review'):entry.evaluate(root,'test')


def test_id_proposal_one_repair_builder_and_real_docker_trial(tmp_path,monkeypatch):
    if not os.environ.get('PROGRAM_IMAGE_DIGEST'):pytest.skip('Requires the locked real Docker worker')
    from atomic_skillgraph.harness.simple_protocol import Broker
    from atomic_skillgraph.harness.benchmarks import OfficeAdapter
    s=office(tmp_path)
    old=deepcopy(s.config);old['budget']={'token_limit':85000000,'request_limit':480000,'finish_reserve':0}
    s.config=configure(old);s.adapter.config=s.config;s.bank.bind_office_method(OFFICE_SOURCE_METHOD_VERSION)
    s.adapter_factory=lambda:OfficeAdapter({'office':{'answer':'private'}},s.config)
    task=s.adapter.task
    broker=Broker(s.adapter,64,context=s.task_context)
    result=broker.call('execute_python',{'source':'import json\nvalues=[2,4,6]\nprint(json.dumps({"mean":sum(values)/len(values)}))'})
    assert result['accepted']
    trace={'tools':broker.events,'score':{'hard':False},'execution':{'attempts':[],'prediction':'wrong','reason':'finish'}}
    case=s.learner._experience(task,trace)
    material=catalog_material({'experience':learning_case(case,task.physical_key)})
    card=material['experience']['local_evidence'][0]
    source_id=next(r['id'] for r in card['input_sources'] if r['value']==[2,4,6])
    output_id=next(r['id'] for r in card['output_references'] if r.get('path')==['data','computed','mean'])
    proposal={'decision':'propose_skill_and_program_spec','generate_program':True,'skill':{
        'goal':'Compute mean of a numerical series','guidance':'Use the supplied numerical series.',
        'execution_intent':'program_requested','result_role':'intermediate','entry_constraints':'Nonempty numeric series',
        'input_schema':{'type':'object','properties':{'values':{'type':'array','items':{'type':'number'}}},'required':['values']},
        'output_schema':{'type':'object','properties':{'mean':{'type':'number'}},'required':['mean']}},
        'realization_request':{'skill_id':'$new','action':'build','case_bindings':[
            {'operation_id':card['operation_id'],'input_ids':{'values':source_id},'output_ids':{'mean':output_id}}]}}
    bad=deepcopy(proposal);bad['realization_request']['case_bindings'][0]['input_ids']['values']='invented'
    seen=http(monkeypatch,[response([bad],name='submit_learning'),response([proposal],name='submit_learning'),
        response([{'source':"def run(ctx, inputs):\n values=inputs['values']\n return {'status':'ok','outputs':{'mean':sum(values)/len(values)}}"}],name='submit_program')])
    try:
        log=s.learner.learn(task,trace)
        assert len(seen)==3 and input_token_bound(seen[1])<=16384
        assert log['tests'][0]['validation']['reason']=='source_output_match'
        assert s.bank.program_eligible(s.bank.get(log['program']))
        assert s.requests[1]['repair']==1
    finally:s.close()
