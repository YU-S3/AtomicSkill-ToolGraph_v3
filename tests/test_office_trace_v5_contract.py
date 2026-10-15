"""V5 production-boundary regression; every HTTP is intercepted or forbidden."""
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path

import pytest

from atomic_skillgraph.agents.protocol import parse_json_strict, SchemaValidationError
from atomic_skillgraph.core.errors import BudgetExhausted
from atomic_skillgraph.empirical.checkpoint import save_json
from atomic_skillgraph.empirical.contracts import digest, object_schema
from atomic_skillgraph.empirical.office_operation_mining import read, sha
from atomic_skillgraph.empirical.office_operation_mining_v5 import mine, material, select, inventory
from atomic_skillgraph.empirical.office_program_contract_v5 import (VERSION, CANDIDATE_SCHEMA_VERSION, ContractError,
    MATERIAL_VERSION, PROBE_VERSION, validate_candidate, validate_probe_contract, check_schema, field_contract, CANDIDATE_SCHEMA)
from atomic_skillgraph.empirical.office_trace_program_induction_v5 import (program_from_response, bundle, EXTRACTOR,
    RequestStore, ModelOutputError, repair_material)
from atomic_skillgraph.empirical.office_program_probe_v5 import expected, probe_cases, verify_result, execute_probe, transfer_inputs
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.experiments import run_office_trace_v5_pilot as runner
from atomic_skillgraph.experiments.office_v5_consumption import execute_action, consumption_metrics
from test_office_trace_v4_pilot import fixture_case, provider
from test_cf2_contracts import http, response


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def reject(*a, **k): raise AssertionError('Real model HTTP forbidden')
    monkeypatch.setattr('requests.sessions.Session.request', reject)


@pytest.fixture
def case(tmp_path):
    system, task, trace, _, _ = fixture_case(tmp_path)
    spans, ops = mine(trace, digest(trace), system.config['program_environment'], [])
    op = next(o for o in ops if o['origin'] == 'observed_local_step' and o['events'][-1]['name'] == 'read')
    yield system, task, trace, spans, op
    system.close()


def proposal(op):
    return {'decision': 'propose_program', 'source_opportunity_ids': [op['id']], 'reason': 'Parameterize a local path discovery',
        'evidence_limitations': ['Observed inputs do not prove successful reuse'],
        'atomic': {'goal': 'Discover files using a public scope', 'applicability': 'Known public file pattern',
            'preconditions': ['A real public scope is available'], 'stop_condition': 'Missing or rejected scope',
            'result_role': 'intermediate', 'source_opportunity_ids': [op['id']]},
        'program_spec': {'contract_version': CANDIDATE_SCHEMA_VERSION, 'operation_kind': 'source_locator',
            'input_schema': object_schema({'pattern': {'type': 'string'}}, ['pattern']),
            'output_schema': object_schema({'paths': {'type': 'array', 'items': {'type': 'string'}}}, ['paths']),
            'allowed_tools': ['glob'], 'expected_output_fields': ['paths'],
            'input_bindings': {'pattern': {'kind': 'observed_field', 'span_id': op['events'][-1]['id'], 'section': 'arguments', 'path': ['path']}},
            'probe_contract': {'kind': 'source_locator', 'pattern_input_field': 'pattern', 'output_field': 'paths'}}}


SOURCE = '''def run(ctx, inputs):
    result = ctx.call('glob', {'pattern':inputs['pattern']})
    if not result.get('accepted'): return {'status':'blocked','outputs':{},'detail':'scope rejected'}
    if not result['data']: return {'status':'not_found','outputs':{},'detail':'no files'}
    return {'status':'ok','outputs':{'paths':result['data']}}
'''


def validate(p, case):
    _, _, _, spans, op = case
    return validate_candidate(p, op, spans, ['glob', 'read', 'grep'], [], material(op, spans))


def diagnostic(p, case, code):
    with pytest.raises(ContractError) as exc: validate(p, case)
    assert exc.value.diagnostic['code'] == code
    assert exc.value.diagnostic['evidence_hash'] == digest(material(case[4], case[3]))
    return exc.value.diagnostic


def test_single_contract_and_strict_decision_branches(case):
    p = proposal(case[4]); result = validate(p, case)
    assert result['status'] == 'spec_valid' and result['bound_inputs']['pattern'] == 'a.txt'
    assert field_contract()['candidate_schema'] == CANDIDATE_SCHEMA
    assert 'column_input_field' in EXTRACTOR and 'Surplus, or deficit (-)' in EXTRACTOR
    for decision in ('no_change', 'reuse_existing'):
        q = {**p, 'decision': decision}
        if decision == 'reuse_existing': q['existing_program_id'] = 'missing'
        diagnostic(q, case, 'proposal_shape_invalid')
    q = {k:v for k,v in p.items() if k not in ('atomic', 'program_spec')}; q['decision'] = 'no_change'
    assert validate(q, case)['decision'] == 'no_change'


@pytest.mark.parametrize('bindings', [[{'span_id': 'x', 'path': ['p']}], [], None, 5, 'name'])
def test_v4_list_bindings_never_set_or_get_before_typecheck(case, bindings):
    p = proposal(case[4]); p['program_spec']['input_bindings'] = bindings
    d = diagnostic(p, case, 'binding_shape_invalid')
    assert d['field_path'] == 'program_spec.input_bindings' and 'observed_field' in d['allowed_binding_kinds']


def test_probe_fields_never_literals_and_output_types(case):
    p = proposal(case[4]); spec = p['program_spec']; spec['operation_kind'] = 'table_cell'
    spec['input_schema'] = object_schema({k:{'type':'string'} for k in ['text', 'row', 'column', 'unit']}, ['text', 'row', 'column', 'unit'])
    spec['output_schema'] = object_schema({'value': {'type': 'number'}}, ['value'])
    spec['probe_contract'] = {'kind':'table_cell','text_input_field':'text','row_input_field':'row',
        'column_input_field':'column','unit_input_field':'unit','output_field':'value'}
    validate_probe_contract(spec, 'hash')
    for field, bad in [('text_input_field', 'text_input'), ('column_input_field', 'Surplus, or deficit (-)'), ('unit_input_field', 'millions')]:
        q = deepcopy(p); q['program_spec']['probe_contract'][field] = bad
        diagnostic(q, case, 'probe_input_field_missing')
    spec['output_schema']['properties']['value']['type'] = 'string'
    diagnostic(p, case, 'value_type_mismatch')


@pytest.mark.parametrize('kind', ['source_locator','retrieval','table_cell','series_extract','numeric','evidence_check'])
def test_actual_prompt_advertises_all_host_probe_types_and_required_fields(kind):
    published=json.loads(EXTRACTOR.split('validator:\n',1)[1])
    def schema(name):
        return {'type':'array','items':schema(name[6:])} if name.startswith('array:') else {'type':name}
    fields={**published['probe_field_types'][kind],**published['optional_probe_field_types'].get(kind,{})}
    contract={'kind':kind,**{name:name.removesuffix('_field') for name in fields}}
    inputs={contract[name]:schema(typ) for name,typ in fields.items()}
    if kind in published['probe_output_types']:
        contract['output_field']='value';outputs={'value':schema(published['probe_output_types'][kind])}
    else:outputs={name:schema(typ) for name,typ in published['probe_fixed_outputs'][kind].items()}
    if kind=='numeric':contract['operation']='sum'
    spec={'operation_kind':next(op for op,probe in published['operation_probes'].items() if probe==kind),
        'input_schema':object_schema(inputs,list(inputs)), 'output_schema':object_schema(outputs,list(outputs)),
        'probe_contract':contract}
    validate_probe_contract(spec,'published-contract')
    for side in ('input_schema','output_schema'):
        for name in spec[side]['properties']:
            bad=deepcopy(spec);bad[side]['required'].remove(name)
            with pytest.raises(ContractError) as exc:validate_probe_contract(bad,'published-contract')
            assert exc.value.diagnostic['field_required'] is False
            bad=deepcopy(spec);bad[side]['properties'][name]={'type':'null'}
            with pytest.raises(ContractError) as exc:validate_probe_contract(bad,'published-contract')
            assert exc.value.diagnostic['declared_type']=='null'
    assert kind in published['probe_semantics']


def test_paid_schema_error_reports_declared_type_and_no_cross_version_admission(case):
    p=proposal(case[4]);p['program_spec']['output_schema']['properties']['paths']={'type':'array','items':object_schema()}
    d=diagnostic(p,case,'value_type_mismatch')
    assert d['declared_type']=='array:object' and d['expected']=='required array:string'
    p=proposal(case[4]);p['program_spec']['contract_version']='office.program-candidate.v5'
    diagnostic(p,case,'proposal_shape_invalid')


@pytest.mark.parametrize('mutation', [lambda s:s.update(required=[{}]), lambda s:s.update(properties=[]),
    lambda s:s.update(required=['absent']), lambda s:s.update(additionalProperties=True),
    lambda s:s.update(required=['pattern','pattern']), lambda s:s['properties']['pattern'].update(minLength='many'),
    lambda s:s['properties']['pattern'].update(type='array', items=['number','null']),
    lambda s:s['properties']['pattern'].update(type=[]), lambda s:s['properties']['pattern'].update(type='number',minimum=4,maximum=2)])
def test_malformed_schema_is_classified_not_host_typeerror(case, mutation):
    p = proposal(case[4]); mutation(p['program_spec']['input_schema'])
    diagnostic(p, case, 'schema_invalid')


def test_pending_is_not_default_or_source_probe_and_builder_keeps_schema(case):
    p = proposal(case[4]); p['program_spec']['input_bindings']['pattern'] = {'kind': 'runtime_required', 'source_requirement': 'public_requested_scope'}
    candidate = validate(p, case)
    assert candidate['bound_inputs'] == {} and candidate['status'] == 'spec_valid_pending_inputs'
    assert candidate['pending_inputs'][0]['schema'] == {'type':'string'}
    program = program_from_response({'source': SOURCE}, candidate, case[0].config['program_environment'], ['glob'])
    value = bundle(candidate, program, 'fixture', {'source':SOURCE})
    assert probe_cases(value)['status'] == 'pending_inputs' and probe_cases(value)['cases'] == []
    assert program['input_schema'] == p['program_spec']['input_schema'] and not value['formal_publication']


def test_transfer_slices_rebind_to_new_observed_offsets_and_same_span(case):
    p=proposal(case[4]);ref=p['program_spec']['input_bindings']['pattern'];ref.update(kind='observed_text_slice',start=0,end=1)
    candidate=validate(p,case);span=deepcopy(next(s for s in case[3] if s['id']==ref['span_id']))
    span['id']='new_train_span';span['source_task']['physical_key']='new_train';span['arguments']['path']='prefix_a.txt'
    values,refs=transfer_inputs(candidate,[span])
    assert values=={'pattern':'a'} and refs['pattern']['start']==7 and refs['pattern']['span_id']=='new_train_span'
    span['arguments']['path']='a_a.txt'
    assert transfer_inputs(candidate,[span])==({}, {})


def test_binding_value_types_ids_unicode_visibility_and_no_inference(case):
    p = proposal(case[4]); ref = p['program_spec']['input_bindings']['pattern']
    ref['path'] = ['offset']; diagnostic(p, case, 'value_type_mismatch')
    ref['path'] = ['path']; ref['span_id'] = 'invented'; diagnostic(p, case, 'unknown_source_span')
    ref['span_id'] = case[4]['events'][-1]['id']; ref['path'] = ['unobserved']; diagnostic(p, case, 'path_not_observed')
    ref['path'] = ['path']; ref.update(kind='observed_text_slice', start=0, end=90000)
    diagnostic(p, case, 'text_slice_out_of_bounds')
    spans = deepcopy(case[3]); span = next(s for s in spans if s['id'] == ref['span_id']); span['arguments']['path'] = 'A中文😀B'
    ref.update(start=1,end=4)
    candidate = validate_candidate(p,case[4],spans,['glob'],[],material(case[4],spans))
    assert candidate['bound_inputs']['pattern'] == '中文😀'
    hidden = material(case[4],spans,window_chars=2)
    with pytest.raises(ContractError,match='path_not_observed'):validate_candidate(p,case[4],spans,['glob'],[],hidden)
    ref.clear(); ref.update(kind='observed_field',span_id=span['id'],section='arguments',path=['path'])
    with pytest.raises(ContractError,match='source_incomplete'):validate_candidate(p,case[4],spans,['glob'],[],hidden)
    span['source_task']['physical_key'] = 'another_task'
    # A Host material cannot silently substitute a cross-task span for this opportunity.
    with pytest.raises(ContractError,match='unknown_source_span'):validate_candidate(p,case[4],spans,['glob'],[],material(case[4],case[3]))
    with pytest.raises(ValueError,match='duplicate'):parse_json_strict('{"input_bindings":{"a":1,"a":2}}')


TABLE = 'Title\n(In millions of dollars)\n| Period | Value |\n|---|---|\n| 1934 | 507 |\n| 1935 | 556 |\n'
CONTRACT = {'kind':'table_cell','text_input_field':'text','row_input_field':'row','column_input_field':'column','unit_input_field':'unit','output_field':'value'}
VALUES = {'text':TABLE,'row':'1934','column':'Value','unit':'millions of dollars'}


def test_oracles_realistic_bounded_table_and_negative_layouts():
    assert expected(CONTRACT,VALUES) == {'value':507}
    series = {k:v for k,v in CONTRACT.items() if k != 'row_input_field'} | {'kind':'series_extract','rows_input_field':'rows'}
    assert expected(series,{**VALUES,'rows':['1934','1935']}) == {'value':[507,556]}
    variants = [{'row':'1936'}, {'unit':'billions'}, {'column':'Not declared'}, {'text':''},
        {'text':TABLE.replace('(In millions of dollars)','(In millions and billions of dollars)')},
        {'text':TABLE + '\n' + TABLE}, {'text':TABLE.replace('| 1934 | 507 |','| 1934 |')},
        {'text':TABLE.replace('| 1934 | 507 |','| 1934 | |')}, {'text':TABLE.replace('| 1935 |','| 1934 |')}]
    for change in variants:
        with pytest.raises(ValueError): expected(CONTRACT,{**VALUES,**change})
    assert expected(CONTRACT,{**VALUES,'text':'In millions of dollars\nPeriod  Value\n1934    507\n1935    556'}) == {'value':507}
    event = {'name':'grep_page','arguments':{'pattern':'p'},'result':{'accepted':True,'data':{'complete':False}}}
    assert not verify_result({'kind':'retrieval','query_input_field':'query'}, {'label':'negative','inputs':{'query':'p'}},
        {'status':'not_found','detail':'none'},[event])[0]
    assert not verify_result(CONTRACT,{'label':'positive','expected':{'value':507}}, {'status':'ok','outputs':{'value':999}},[])[0]


def test_context_api_error_and_pending_runtime_does_not_start_worker(case):
    candidate = validate(proposal(case[4]), case)
    with pytest.raises(ContractError,match='context_api_misuse'):
        program_from_response({'source':'def run(ctx, inputs):\n return ctx.read_context(inputs)'},candidate,case[0].config['program_environment'],['glob'])
    context = TaskContext(); broker = type('Broker',(),{'context':context})()
    program = program_from_response({'source':SOURCE},candidate,case[0].config['program_environment'],['glob'])
    state = {'actions':[],'producers':[]}
    worker = type('Forbidden',(),{'execute':lambda *a:pytest.fail('Missing inputs must not start Worker')})()
    result = execute_action({'action':'program','program_id':program['id'],'input_refs':{}},broker,[{'id':program['id'],'program':program}],worker,state)
    assert result['status']=='needs_input' and not result['program_invoked']
    ref = {'result_id':'another-task-result','path':['outputs','value']}
    with pytest.raises(ValueError):execute_action({'action':'program','program_id':program['id'],'input_refs':{'pattern':ref}},broker,[],worker,state)


def test_consumption_requires_actual_ordered_fields_not_prose_or_exposure():
    parent = {'kind':'program','program_invoked':True,'status':'ok','accepted':True,'program_id':'p','result_id':'r1',
        'resolved_refs':{},'bound_inputs':{},'result':{'status':'ok','outputs':{'value':12}}}
    child = {'kind':'program','program_invoked':True,'status':'ok','accepted':True,'program_id':'q','result_id':'r2',
        'resolved_refs':{'x':{'result_id':'r1','path':['outputs','value']}},'bound_inputs':{'x':12},'result':{'status':'ok','outputs':{'value':24}}}
    programs = {'p':{'id':'p','source':'fixture','input_schema':object_schema(),'output_schema':object_schema({'value':{'type':'number'}},['value'])},
        'q':{'id':'q','source':'fixture','input_schema':object_schema({'x':{'type':'number'}},['x']),'output_schema':object_schema({'value':{'type':'number'}},['value'])}}
    m = consumption_metrics([parent,child],['r1'],programs)
    assert m['downstream_bound']==1 and m['partial_workflow']['partial'] and m['result_exposed']==1
    assert not consumption_metrics([parent],['r1'],programs)['outputs_consumed']
    wrong = {**child,'bound_inputs':{'x':13}}
    assert not consumption_metrics([parent,wrong],['r1'],programs)['outputs_consumed']
    assert not consumption_metrics([child,parent],['r1'],programs)['outputs_consumed']


def test_inventory_selection_uses_public_structure_not_scores(case):
    system,task,trace,_,_ = case; spans,ops=[],[]
    for i in range(10):
        tr = deepcopy(trace); tr['task'].update(task_id='t'+str(i),physical_key='k'+str(i)); tr['score']={'hard':i%2==0,'secret':'GOLD_SECRET'}
        for event in tr['tools']: event['arguments']['scope_metadata_for_fixture'] = str(i)
        ss,oo=mine(tr,digest(tr),system.config['program_environment'],[]);spans+=ss;ops+=oo
    focus=[next(o['id'] for o in ops if o['source_task']['task_id']=='t'+str(i) and o['events'][-1]['name']=='read') for i in range(2)]
    selection=select(ops,focus_ids=focus,transfer_tasks=['t8','t9'])
    assert len(selection['opportunities'])==10 and len(selection['source_tasks'])==8
    assert not set(selection['source_tasks']) & {'t8','t9'}
    assert 'GOLD_SECRET' not in json.dumps([ops,spans]) and inventory(ops,spans)['tasks']==10


def setup_run(tmp_path, case, monkeypatch):
    root=tmp_path/'v5';root.mkdir();(root/'candidate_bank').mkdir();(root/'programs').mkdir()
    plan={'run_id':'test_v5','config':{'wire_max_bytes':65536,'source_window_chars':12000},'method_version':VERSION,
        'parent_bank':str(tmp_path/'old'),'parent_frozen':runner.tree_identity(tmp_path/'old'),
        'inherited_tokens':0,'inherited_http':0,'effective_cumulative_limits':{'token_limit':90000000,'request_limit':480000,'finish_reserve':0}}
    ledger={'limits':plan['effective_cumulative_limits'],'attempts':{},'unknown_billing':False}
    save_json(root/'pilot_manifest.json',plan);save_json(root/'budget/inherited.json',ledger);save_json(root/'budget/usage.json',ledger)
    save_json(root/'budget/sibling_reservations.json',[])
    save_json(root/'authorization.json',{'method_version':VERSION,'approved':True,'user_authorization':'INTERCEPTED TEST ONLY',
        'manifest_sha256':sha(root/'pilot_manifest.json'),'phase_limits':runner.PHASE_LIMITS,'approved_phases':['learn','consume']})
    save_json(root/'zero_http_check.json',{'passed':True,'manifest_sha256':sha(root/'pilot_manifest.json')})
    save_json(root/'selection.json',{'paid_opportunities':[case[4]['id']]});save_json(root/'parent_programs.json',[])
    runner.jsonl(root/'source_span_catalog.jsonl',case[3]);runner.jsonl(root/'opportunity_catalog.jsonl',[case[4]])
    monkeypatch.setattr(runner,'checked',lambda *a,**k:(plan,case[0].config))
    return root,plan


@pytest.mark.parametrize('mode',['valid','abstain','bad_then_repair','bad_twice','host_bug'])
def test_intercepted_learning_resume_negative_reporting_and_one_repair(tmp_path,case,monkeypatch,mode):
    root,plan=setup_run(tmp_path,case,monkeypatch);p=proposal(case[4]); bad=deepcopy(p);bad['program_spec']['input_bindings']=[{}]
    abstain={k:v for k,v in p.items() if k not in ('atomic','program_spec')}|{'decision':'no_change'}
    messages=([abstain] if mode=='abstain' else [bad,bad] if mode=='bad_twice' else [bad,p,{'source':SOURCE}]
              if mode=='bad_then_repair' else [p,{'source':SOURCE}])
    seen=http(monkeypatch,[response(finish='stop',content=json.dumps(m)) for m in messages])
    if mode=='host_bug':
        monkeypatch.setattr(runner,'validate_candidate',lambda *a:(_ for _ in ()).throw(TypeError('actual Host bug')))
        with pytest.raises(TypeError,match='Host bug'):runner.learn(root)
        assert len(seen)==1;return
    result=runner.learn(root);before=len(seen);runner.learn(root);assert len(seen)==before
    assert result['formal_promotions']==0 and result['new_http']==len(messages)
    if mode in {'abstain','bad_twice'}:assert result['status']=='completed_no_candidate'
    else:
        bundles=runner.verify_candidates(root);assert len(bundles)==1 and bundles[0]['status']=='candidate_built'
        assert bundles[0]['candidate']['method_version']==VERSION
    if mode.startswith('bad'): assert (root/'diagnostics/op0/proposal.diagnostic.json').exists()


def test_phase_and_cumulative_budget_no_old_authorization_or_unresolved_retry(tmp_path,case,monkeypatch):
    root,plan=setup_run(tmp_path,case,monkeypatch)
    governor=runner.PhaseGovernor(root,'learn')
    with pytest.raises(BudgetExhausted):governor.admit('over',{'max_tokens':500001},{})
    assert not governor.state['attempts']
    governor.admit('uncertain',{'max_tokens':100},{})
    with pytest.raises(BudgetExhausted):runner.PhaseGovernor(root,'learn').admit('again',{'max_tokens':100},{})
    auth=read(root/'authorization.json');auth['method_version']='office-trace-v4-pilot.v1';save_json(root/'authorization.json',auth)
    with pytest.raises(ValueError,match='V5'):runner.PhaseGovernor(root,'learn')


def test_prior_attempts_are_carried_once_and_share_original_phase_allowance(tmp_path,case,monkeypatch):
    root,plan=setup_run(tmp_path,case,monkeypatch)
    prior=tmp_path/'prior';ledger=read(root/'budget/usage.json')
    paid={'status':'metered','accounted_tokens':490000,'reserved_tokens':490000,
        'context':{'office_v5_run':'old_v5','office_v5_phase':'learn'}}
    save_json(prior/'pilot_manifest.json',{'method_version':'office-trace-v5-program-induction.v1','v4_run':str(tmp_path/'v4')})
    save_json(prior/'launch_status.json',{'status':'completed'})
    save_json(prior/'budget/usage.json',{**ledger,'attempts':{'physical_1':paid}})
    inherited,paths=runner.carry_prior_runs(ledger,[str(prior),str(prior)],tmp_path/'v4')
    assert len(inherited['attempts'])==1 and paths and ledger['attempts']=={}
    plan['phase_budget_run_ids']=['old_v5',plan['run_id']];save_json(root/'pilot_manifest.json',plan)
    auth=read(root/'authorization.json');auth['manifest_sha256']=sha(root/'pilot_manifest.json');save_json(root/'authorization.json',auth)
    save_json(root/'budget/usage.json',inherited)
    governor=runner.PhaseGovernor(root,'learn')
    assert governor.headroom({})['tokens']==10000
    with pytest.raises(BudgetExhausted):governor.admit('new',{'max_tokens':10001},{})
    paid['accounted_tokens']=489999;save_json(prior/'budget/usage.json',{**ledger,'attempts':{'physical_1':paid}})
    with pytest.raises(ValueError,match='Conflicting'):runner.carry_prior_runs(inherited,[str(prior)],tmp_path/'v4')
    paid['status']='admitted';save_json(prior/'budget/usage.json',{**ledger,'attempts':{'physical_1':paid}})
    with pytest.raises(ValueError,match='Resolve prior billing'):runner.carry_prior_runs(ledger,[str(prior)],tmp_path/'v4')


def test_request_store_no_duplicate_http_or_altered_payload(tmp_path,monkeypatch):
    seen=http(monkeypatch,[response(finish='stop',content='{"decision":"no_change"}')])
    client=RequestStore(tmp_path/'requests',provider(tmp_path),phase='learn')
    assert client.ask('one','V5 contract',{})[0]['decision']=='no_change'
    client.ask('one','V5 contract',{});assert len(seen)==1
    with pytest.raises(RuntimeError,match='identity_changed'):client.ask('one','V4 contract',{})
    payload=client.provider._build_payload([{'role':'system','content':'p'},{'role':'user','content':'{}'}],[])
    save_json(tmp_path/'requests/pending.json',{'payload_sha256':digest(payload),'status':'prepared'})
    with pytest.raises(RuntimeError,match='unresolved_http'):client.ask('pending','p',{})
    assert len(seen)==1


def test_consumption_finishes_scores_and_resumes_without_new_http(tmp_path,case,monkeypatch):
    from atomic_skillgraph.experiments import office_v5_consumption as consumption
    root,plan=setup_run(tmp_path,case,monkeypatch)
    task=asdict(case[1]);other={**task,'task_id':'officeqa:transfer_two','physical_key':'transfer_two'}
    plan['source']=str(tmp_path/'train')
    plan['config'].update(transfer_tasks=[task['task_id'],other['task_id']],
        paired_arm_limits={'tokens':62500,'http':5,'native_calls':48})
    save_json(root/'pilot_manifest.json',plan)
    auth=read(root/'authorization.json');auth['manifest_sha256']=sha(root/'pilot_manifest.json');save_json(root/'authorization.json',auth)
    labels=tmp_path/'private_train_labels.json'
    save_json(labels,{t['task_id']:{'answer':'42'} for t in (task,other)})
    save_json(Path(plan['source'])/'config.json',{'harness':{'evaluator_records':str(labels)}})
    save_json(root/'source_records.json',[{'task':t} for t in (task,other)])
    candidate=validate(proposal(case[4]),case)
    program=program_from_response({'source':SOURCE},candidate,case[0].config['program_environment'],['glob'])
    value=bundle(candidate,program,'intercepted fixture',{'source':SOURCE});value['status']='transfer_checked'
    monkeypatch.setattr(consumption,'checked',lambda *a,**k:(plan,case[0].config))
    monkeypatch.setattr(consumption,'verify_candidates',lambda *a:[value])
    seen=http(monkeypatch,[response(finish='stop',content=json.dumps({'action':'finish','answer':answer}))
        for answer in ('42','0','42','0')])
    result=consumption.consume(root)
    assert len(seen)==4 and len(result['consumption'])==4
    assert [t['task_correct'] for t in result['consumption']].count(True)==2
    traces=[read(p) for p in (root/'consumption').glob('*/*/trace.json')]
    assert all(t['terminal_reason']=='submitted' and t['http']==1 for t in traces)
    assert all(not t['metrics']['program_invoked'] for t in traces)
    assert all(bool(t['visible_program_ids'])==(t['arm']=='with_program') for t in traces)
    consumption.consume(root)
    assert len(seen)==4


def test_repair_material_keeps_whole_text_but_drops_duplicate_catalogs(case):
    p=proposal(case[4]);data={'source':material(case[4],case[3]),'existing_programs':[{'id':'p','source':'HIDDEN_UNUSED_SOURCE'}],
        'permissions':{'allowed_names':['glob','read'],'public_program_abi':{'entry':'run'},
            'runtime_tool_definitions':[{'name':'glob'},{'name':'read'}]}}
    record={'turn':{'content':json.dumps(p)},'payload_sha256':'old','payload':{'messages':['OLD_FULL_REQUEST']}}
    repaired=repair_material(record,{'code':'binding_shape_invalid'},data)
    assert 'OLD_FULL_REQUEST' not in json.dumps(repaired) and 'HIDDEN_UNUSED_SOURCE' not in json.dumps(repaired)
    assert repaired['source_material']['permissions']['runtime_tool_definitions']==[{'name':'glob'}]
    before=next(s for s in data['source']['spans'] if s['id']==p['program_spec']['input_bindings']['pattern']['span_id'])
    assert repaired['source_material']['source']['spans']==[before]


def test_worker_unresolved_intent_not_replayed_and_all_fail_normal_report(tmp_path,case,monkeypatch):
    root,plan=setup_run(tmp_path,case,monkeypatch);plan['config']['transfer_tasks']=[]
    save_json(root/'pilot_manifest.json',plan)
    # The identity changed before this intercepted test's authorization, never after a real launch.
    auth=read(root/'authorization.json');auth['manifest_sha256']=sha(root/'pilot_manifest.json');save_json(root/'authorization.json',auth)
    gate=read(root/'zero_http_check.json');gate['manifest_sha256']=auth['manifest_sha256'];save_json(root/'zero_http_check.json',gate)
    seen=http(monkeypatch,[response(finish='stop',content=json.dumps(proposal(case[4]))),response(finish='stop',content=json.dumps({'source':SOURCE}))])
    runner.learn(root);value=runner.verify_candidates(root)[0]
    save_json(root/'source_records.json',[{'task':asdict(case[1])}])
    folder=root/'probes'/value['id'];save_json(folder/'positive.intent.json',{'started':'interrupted'})
    monkeypatch.setattr(runner,'execute_probe',lambda *a:pytest.fail('Must not replay unresolved Worker'))
    with pytest.raises(RuntimeError,match='without receipt'):runner.probe(root)
    # Test-only removal simulates an explicit external resolution, not a production recovery rule.
    (folder/'positive.intent.json').unlink()
    def failed(value,c,config,task):
        return {'case':c,'source_sha256':digest(value['program']['source']),'passed':False,'result':{'status':'needs_input'},'reason':'missing data'}
    monkeypatch.setattr(runner,'execute_probe',failed)
    result=runner.probe(root);assert result['status']=='completed_no_qualified_candidate'
    assert result['candidate_statuses']=={'worker_failed':1} and len(seen)==2
    runner.probe(root);assert len(seen)==2


def test_exact_code_method_budget_and_old_artifacts_resume_checks(tmp_path,case,monkeypatch):
    root,plan=setup_run(tmp_path,case,monkeypatch)
    # Restore the real identity checker; only the runtime source fingerprint is an explicit fixture.
    import importlib
    module=importlib.reload(runner)
    identity={'git_sha':'fixture','tracked_dirty':False,'source_sha256':'fixed'}
    monkeypatch.setattr(module,'code_identity',lambda:identity)
    original=tmp_path/'old_response.json';save_json(original,{'content':'original'})
    save_json(root/'source_records.json',[]);save_json(root/'runtime_config.json',case[0].config)
    plan.update(method_version=VERSION,material_version=MATERIAL_VERSION,probe_version=PROBE_VERSION,code=deepcopy(identity),
        files={'budget/inherited.json':sha(root/'budget/inherited.json')},parent_files={str(original):sha(original)},
        permissions_sha256=digest(module.permission(case[0].config)))
    save_json(root/'pilot_manifest.json',plan);save_json(root/'manifest_identity.json',{'sha256':sha(root/'pilot_manifest.json')})
    assert module.checked(root)[0]['code']==identity
    identity['source_sha256']='altered'
    with pytest.raises(RuntimeError,match='Code/manifest'):module.checked(root)
    identity['source_sha256']='fixed'
    old=read(root/'budget/inherited.json');save_json(root/'budget/inherited.json',{**old,'limits':{'token_limit':999999999}})
    with pytest.raises(RuntimeError,match='immutable input'):module.checked(root)
    save_json(root/'budget/inherited.json',old);save_json(original,{'content':'changed'})
    with pytest.raises(RuntimeError,match='Historical audit'):module.checked(root)


@pytest.mark.skipif(os.environ.get('OFFICE_V5_DOCKER')!='1',reason='pinned Docker required')
def test_actual_docker_program_and_host_network_file_isolation(case,tmp_path):
    system,task,_,_,op=case
    system.config['program_environment']['image_digest']='sha256:4db520e20cd121f830731c2d0c0bcfbe9cacd254d1817262404cc18e1e757783'
    candidate=validate(proposal(op),case)
    program=program_from_response({'source':SOURCE},candidate,system.config['program_environment'],['glob'])
    value=bundle(candidate,program,'fixture',{'source':SOURCE});plan=probe_cases(value)
    assert plan['status']=='ready'
    receipts=[execute_probe(value,c,system.config,asdict(task)) for c in plan['cases']]
    assert all(r['passed'] for r in receipts)
    # Containers can run ordinary Python, but neither eval nor open provides access to the Host.
    sentinel=tmp_path/'host_secret.txt';sentinel.write_text('host-only')
    from atomic_skillgraph.experiments.office_v4_offline import engineering_program
    from atomic_skillgraph.empirical.program_worker import ProgramWorker
    from atomic_skillgraph.harness.simple_protocol import Broker
    source="def run(ctx, inputs):\n import socket\n failures=[]\n try:\n  open(inputs['path']).read()\n except OSError:\n  failures.append('file')\n try:\n  socket.create_connection(('1.1.1.1',443),timeout=1)\n except OSError:\n  failures.append('network')\n return {'status':'ok','outputs':{'failures':failures}}"
    p=engineering_program(source,object_schema({'path':{'type':'string'}},['path']),
        object_schema({'failures':{'type':'array','items':{'type':'string'}}},['failures']),[],system.config)
    broker=Broker(system.adapter,10,context=TaskContext(system.config['runtime']))
    result=ProgramWorker({'record_execution_receipt':True}).execute(p,{'path':str(sentinel)},broker)
    assert result['status']=='ok' and result['outputs']['failures']==['file','network']
    assert '--network=none' in result['execution_receipt']['command']
