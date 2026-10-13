"""Pinned Train-only experience recompile; prepare/check never call a model."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from uuid import uuid4

import yaml

from ..core.errors import BudgetExhausted
from ..empirical import POLICY_DEFAULTS
from ..empirical.bank import Bank
from ..empirical.budget_governor import BudgetGovernor, input_token_bound
from ..empirical.capability_batch import (VERSION, MATERIAL_VERSION, PROMPT_REVISION, CAPABILITY, CapabilityView, records, publish, utility_receipt,
    validate_policy, validate_match, validate_capability, validate_check, generation_schema, CHECK, gate, pair_fact)
from ..empirical.model_view import expand_material
from ..empirical.checkpoint import TaskCheckpoint
from ..empirical.contracts import PublicTask, digest
from ..empirical.system import EmpiricalSystem, validate_config
from ..harness.benchmarks import AnswerAdapter
from .run_empirical import code_identity, write_json, load_env
from .canonical_manifest import sha256
from .formal_log import utc

SCOPE_POLICY='choice-capability-batch.run-local-budget.v1'


def learning_scope(plan,key):
    return plan['run_id']+':'+key if plan.get('scope_policy')==SCOPE_POLICY else key


def read(path):return json.loads(Path(path).read_text())


def progress(root,event,**values):
    record={'event':event,'at':utc(),**values};write_json(Path(root)/'progress.json',record)
    print(json.dumps(record,ensure_ascii=False),flush=True)


def policy(spec):
    evaluation=spec['paired_evaluation'];rules={}
    for name,key in [('screen','screen'),('promotion','promotion_after_both_stages')]:
        rule=evaluation[key]
        rules[name]={'min_exposed':rule['min_actual_exposed_tasks' if name=='screen' else 'min_actual_exposed_tasks_total'],
            'min_net_correct':rule['min_net_correct_on_minus_off'],'max_regressions':rule['max_correct_to_wrong_regressions'],
            'require_empty_on_not_increase':rule['require_empty_on_not_increase']}
    split=spec['split']
    value={**rules,'min_precomputed_applicable':evaluation['min_precomputed_applicable_tasks_to_attempt_paid_selection'],
        'selection':{stage:[{'task_id':k,'physical_key':split['selected_physical_keys'][k]} for k in split['selection_'+stage+'_ids']]
            for stage in ('screen','confirmation')}}
    validate_policy(value)
    return value


def partition(tasks,spec):
    split=spec['split'];selector=split['selector'];by_id={t['task_id']:t for t in tasks}
    if len(by_id)!=len(tasks) or len({t['physical_key'] for t in tasks})!=len(tasks) or any(t['split']!='train' for t in tasks):
        raise ValueError('Distinct public Train tasks required')
    if selector['algorithm']!='sha256_hex_ascending_lexicographic' or selector['input_field']!='task_id':raise ValueError('Unsupported declared selector')
    ranked=sorted(by_id,key=lambda k:hashlib.sha256((selector['salt_prefix']+k).encode()).hexdigest())
    stages={}
    for name in ('screen','confirmation'):
        start,end=selector[name+'_rank_slice'];ids=ranked[start:end]
        if ids!=split['selection_'+name+'_ids'] or len(ids)!=split['size']['selection_'+name]:raise ValueError('Declared selection differs from public SHA ranking')
        stages[name]=[by_id[k] for k in ids]
    selected=[t for rows in stages.values() for t in rows]
    if len({t['physical_key'] for t in selected})!=len(selected):raise ValueError('Selection stages overlap')
    if split['selected_physical_keys']!={t['task_id']:t['physical_key'] for t in selected}:raise ValueError('Selection physical identity differs')
    generation=[by_id[k] for k in sorted(set(by_id)-{t['task_id'] for t in selected})]
    if [t['task_id'] for t in generation]!=split['candidate_source_task_ids'] or len(generation)!=split['size']['candidate_source_train']:
        raise ValueError('Candidate pool differs from the declared disjoint Train complement')
    if split['reuse_selection_cases_in_learning'] or not split['exclude_selection_from_all_candidate_material_and_stats'] or split['old_train_score_may_choose_selection']:
        raise ValueError('Selection feedback must remain excluded from candidate learning')
    return generation,stages


def configuration(original,output):
    config=deepcopy(original)
    for section,defaults in POLICY_DEFAULTS.items():config[section].update(defaults)
    config['data_dir']=str(output/'bank');config['experiment'].update(output_dir=str(output),runtime_mode='online',choice_method_version=VERSION)
    for purpose in ('capability_induction','capability_routing'):
        config['llm']['purpose_overrides'][purpose]={'protocol':{'thinking_type':'disabled'},'max_completion_tokens':2048}
    if config['runtime']['choice_bounded_finalization']['enabled']:raise ValueError('P2 keeps bounded finalization disabled')
    return validate_config(config)


def finalization_audit(summary,config):
    """Conservative saved-request feasibility; no retroactive answers or result tuning."""
    from ..agents.provider import OpenAICompatibleConfig, OpenAICompatibleProvider
    from ..empirical.prompts import finish_text_prompt
    llm=config['llm'];cap=llm['purpose_overrides']['finish_only']['max_completion_tokens']
    provider=OpenAICompatibleProvider(OpenAICompatibleConfig(base_url=llm['base_url'],model=llm['model'],api_key_env=llm['api_key_env'],
        dialect=llm['dialect'],input_modalities=tuple(llm['input_modalities']),token_limit_field=llm['token_limit_field'],
        thinking_type='disabled',max_completion_tokens=cap,capability_profile=llm['capability_profile']))
    results=[]
    for trace in summary['cases']:
        if trace['execution'].get('reason')!='completion_truncated_no_answer':continue
        task=PublicTask(**trace['task']);material={'goal':task.goal,'inputs':task.inputs,'local_results':[],'local_reads':[]}
        payload=provider._build_payload([{'role':'system','content':finish_text_prompt('Return one actual displayed option label.')},
            {'role':'user','content':json.dumps(material,ensure_ascii=False)}],[])
        required=input_token_bound(payload)+cap+4096
        attempts=[a for r in trace['requests'] if r['stage']=='runtime' for a in r.get('http_attempts',[])]
        spent=sum(a['usage']['total_tokens'] for a in attempts)
        first=attempts[0]['final_payload_audit'];parent=min(config['budget']['train_solve_tokens'],llm['runtime']['max_total_tokens_per_task'])
        results.append({'task_id':task.task_id,'finish_wire_bytes':input_token_bound(payload),'finish_output_cap':cap,
            'public_process_reserve_bytes':4096,'bounded_finish_reserve':required,'original_runtime_tokens':spent,'parent_limit':parent,
            'could_fit_after_saved_response':spent+required<=parent,
            'would_preserve_initial_runtime_cap':first['serialized_payload_utf8_bytes']+llm['runtime']['max_completion_tokens']+required<=parent,
            'retained_result_windows_checked':False})
    return {'enabled':False,'new_model_http':0,'cases':results,'interpretation':'Base-input feasibility, not answer improvement or finalization admission.'}


def prepare(settings_path,source_run,output,*,allow_unresolved_billing=False):
    spec=yaml.safe_load(Path(settings_path).read_text());output=Path(output).resolve();source=Path(source_run).resolve()
    if output.exists():raise ValueError('Preparation requires a fresh independent directory')
    code=code_identity()
    if code['tracked_dirty'] or not code['git_sha']:raise ValueError('Prepare only committed clean source')
    if spec['schema']!='atomic.livemath.p2.selection-contract.v1' or spec['state']!='PREDECLARED_NOT_EXECUTED':raise ValueError('Unsupported frozen selection contract')
    rules=policy(spec);manifest=read(source/'run_manifest.json');execution=read(source/'train/execution_manifest.json');summary=read(source/'train/summary.json')
    authority=spec['source'];original=read(source/'train/config.json')
    expected={'source_commit':authority['historical_run_commit'],'run_seed':authority['run_seed'],'split_seed':authority['split_seed'],
        'provider_model_id':authority['actual_model_request_label']}
    if any(manifest[k]!=v for k,v in expected.items()) or manifest['split_hashes']['train']!=authority['public_train_manifest_sha256'] or not summary['complete']:
        raise ValueError('Historical source commit, Train hash, model, seed or completion differs')
    if original['llm']['generation_seed_supported']!=authority['generation_seed_supported'] or original['llm']['model']!=authority['actual_model_request_label']:
        raise ValueError('Provider model or generation seed identity differs')
    if execution['config']!=original or execution['tasks']!=[t['task'] for t in summary['cases']]:raise ValueError('Source public task/config order differs')
    canonical=Path(original['manifest']).resolve()
    if sha256(canonical)!=authority['public_train_manifest_sha256'] or {t['task_id'] for t in read(canonical)['tasks']}!={t['task_id'] for t in execution['tasks']}:
        raise ValueError('Actual canonical Train manifest hash or members differ')
    generation,stages=partition(execution['tasks'],spec);selected={t['physical_key'] for rows in stages.values() for t in rows}
    bank=Bank(source/'train/bank',readonly=True,seed=authority['run_seed'],environment=original['program_environment'])
    try:
        raw=bank.train_cases()
        if bank.digest()!=summary['knowledge_digest']:raise ValueError('Source Bank changed')
    finally:bank.close()
    if [r['task'] for r in raw]!=execution['tasks']:raise ValueError('Completed Train evidence differs from execution identity')
    by_key={r['task']['physical_key']:r for r in raw};rows=[by_key[t['physical_key']] for t in generation];public=records(rows)
    ledger=read(source/'budget.json');budget=spec['budget_additional_authority_proposed_not_spent']
    unresolved=ledger['unknown_billing'] or any(a['status']!='metered' for a in ledger['attempts'].values())
    if unresolved and not allow_unresolved_billing:raise ValueError('Unresolved inherited billing')
    inherited=sum(a.get('accounted_tokens',a['reserved_tokens']) for a in ledger['attempts'].values());http=len(ledger['attempts'])
    limits={'token_limit':budget['max_new_total_tokens'],'request_limit':budget['max_new_physical_http_attempts']}
    if any(type(v) is not int or v<=0 for v in limits.values()) or budget['selection_retry_max']!=0 or budget['unknown_billing']!='fail_closed':
        raise ValueError('Finite authority and fail-closed zero-retry selection required')
    if ledger['limits']['token_limit']!=budget['same_original_total_governor_limit_tokens'] or ledger['limits']['request_limit']!=budget['same_original_total_governor_limit_http']:
        raise ValueError('Cumulative authority changed')
    if inherited+limits['token_limit']>ledger['limits']['token_limit'] or http+limits['request_limit']>ledger['limits']['request_limit']:
        raise ValueError('Additional budget exceeds unchanged cumulative authority')
    count=spec['candidate']['max_candidates_to_selection'];width=spec['candidate']['min_distinct_physical_train_source_cases']
    if count!=spec['paired_evaluation']['number_of_candidates_that_may_use_selection'] or not 1<=count<=2 or not 2<=width<=spec['candidate']['max_evidence_cases_per_candidate']<=4:
        raise ValueError('Unsupported bounded candidate contract')
    if spec['candidate']['min_distinct_physical_train_source_cases']!=2 or spec['candidate']['knowledge_type']!='procedural_advisory':
        raise ValueError('Independent procedural sources required')
    config=configuration(original,output)
    scheduling={}
    if spec.get('source_grouping'):
        from .capability_schedule import schedule,unit_limits,configure
        config=validate_config(configure(config,spec))
        source_schedule=schedule(generation,spec)
        units=[[t['physical_key'] for t in unit['sources']] for unit in source_schedule['units']]
        scheduling={'source_schedule':source_schedule,'unit_limits':unit_limits(config,spec)}
    else:units=[[t['physical_key'] for t in generation[i*width:(i+1)*width]] for i in range(count)]
    names=['run_manifest.json','train/execution_manifest.json','train/config.json','train/summary.json','train/bank/bank.sqlite3','budget.json']
    inputs={str(source/name):sha256(source/name) for name in names};inputs[str(Path(settings_path).resolve())]=sha256(Path(settings_path))
    inputs[str(canonical)]=sha256(canonical)
    write_json(output/'source_records.json',public);write_json(output/'source_cases.json',rows);write_json(output/'config.json',config)
    plan={'schema':VERSION,'prompt_revision':PROMPT_REVISION,'scope_policy':SCOPE_POLICY,'run_id':'p2_'+uuid4().hex,'code':code,'prepared_at':utc(),'source_run':str(source),'source_run_id':manifest['run_id'],
        'derivation_mode':'offline_recompile_from_completed_train','initial_bank':'fresh independent derived Bank; no P1 assets',
        'settings':spec,'input_hashes':inputs,'prepared_hashes':{n:sha256(output/n) for n in ('config.json','source_records.json','source_cases.json')},
        'generation_tasks':generation,'selection_tasks':stages,'candidate_units':units,**scheduling,
        'candidate_unit_order':'frozen public-goal relation; no outcome selection' if scheduling else 'public task_id lexicographic; first bounded disjoint units; no outcome selection',
        'utility_policy':rules,'new_limits':limits,'inherited_tokens':inherited,'inherited_http':http,'cumulative_limits':ledger['limits'],
        'source_train_tokens':summary['known_total_tokens'],
        'source_train_runtime_tokens':sum(a['usage']['total_tokens'] for t in summary['cases'] for r in t['requests'] if r['stage']=='runtime' for a in r.get('http_attempts',[])),
        'status':'prepared_blocked_unknown_billing' if unresolved else 'prepared_not_executed',
        'billing_ready':not unresolved,'new_model_http':0,'no_val_or_test_selection':True}
    if any(set(unit)&selected for unit in units):raise ValueError('Learning unit overlaps selection')
    write_json(output/'finalization_feasibility.json',finalization_audit(summary,original));write_json(output/'plan.json',plan)
    write_json(output/'plan_identity.json',{'plan_hash':digest(plan)})
    return plan


def check(plan_path):
    plan=read(plan_path);root=Path(plan_path).parent
    if read(root/'plan_identity.json')['plan_hash']!=digest(plan) or plan['schema']!=VERSION or plan['code']!=code_identity():raise ValueError('Pinned plan or source changed')
    for path,expected in plan['input_hashes'].items():
        if sha256(Path(path))!=expected:raise ValueError('Inherited source or settings changed: '+path)
    for name,expected in plan['prepared_hashes'].items():
        if sha256(root/name)!=expected:raise ValueError('Prepared material changed: '+name)
    validate_config(read(root/'config.json'));policy(plan['settings'])
    generation,stages=partition(read(Path(plan['source_run'])/'train/execution_manifest.json')['tasks'],plan['settings'])
    if generation!=plan['generation_tasks'] or stages!=plan['selection_tasks']:raise ValueError('Prepared split changed')
    if 'source_schedule' in plan:
        from .capability_schedule import validate_plan
        validate_plan(plan,read(root/'config.json'))
    if 'recovery' in plan:
        recovered,ledger,_=recovery_inputs(plan['recovery']['parent_plan'])
        if recovered!=plan['recovery'] or read(root/'inherited_p2_budget.json')!=ledger:raise ValueError('Recovery provenance changed')
        if plan['inherited_tokens']!=recovered['inherited_tokens'] or plan['inherited_http']!=recovered['inherited_http'] or plan['new_limits']!=recovered['allowance']:
            raise ValueError('Recovery budget authority changed')
        if (root/'budget.json').exists() and any(read(root/'budget.json')['attempts'].get(k)!=v for k,v in ledger['attempts'].items()):
            raise ValueError('Inherited paid attempts changed')
    if 'paid_prefix' in plan:
        if 'budget_continuation' in plan:
            restored,ledger,_=budget_continuation_inputs(plan['budget_continuation']['parent_plan'])
            if restored!=plan['budget_continuation']:raise ValueError('Budget continuation provenance changed')
            prefix=restored['paid_prefix']
        elif 'protocol_continuation' in plan:
            restored,ledger,_=protocol_inputs(plan['protocol_continuation']['receipt'])
            if restored!=plan['protocol_continuation']:raise ValueError('Revalidated protocol provenance changed')
            prefix=restored['paid_prefix']
        elif plan['paid_prefix']['experimental_revision'] in {'P2-R4','P2-R5'}:
            prefix,ledger,_=revision_inputs(plan['paid_prefix']['parent_plan'],experimental_revision=plan['paid_prefix']['experimental_revision'])
        else:prefix,ledger,_=(continuation_inputs if plan.get('continuation') else revision_inputs)(plan['paid_prefix']['parent_plan'])
        if prefix!=plan['paid_prefix'] or read(root/'inherited_p2_budget.json')!=ledger or plan['new_limits']!=prefix['allowance']:
            raise ValueError('Paid revision provenance or allowance changed')
        if plan['inherited_tokens']!=prefix['inherited_tokens'] or plan['inherited_http']!=prefix['inherited_http']:
            raise ValueError('Paid revision inherited cost changed')
        if (root/'budget.json').exists() and any(read(root/'budget.json')['attempts'].get(k)!=v for k,v in ledger['attempts'].items()):
            raise ValueError('Inherited paid attempts changed')
    return {'status':'passed','new_model_http':0,'generation':len(generation),'screen':len(stages['screen']),'confirmation':len(stages['confirmation'])}


def recovery_inputs(plan_path):
    """Only completed induction-only batches; replay exact paid outputs, never new evidence."""
    path=Path(plan_path).resolve();root=path.parent;old=read(path);result=read(root/'result.json');ledger=read(root/'budget.json')
    if old['schema']!=VERSION or 'recovery' in old or digest(old)!=read(root/'plan_identity.json')['plan_hash'] or result['plan_hash']!=digest(old):
        raise ValueError('Original batch identity changed or already recovered')
    inputs={str(path):sha256(path)}
    for p,expected in {**old['input_hashes'],**{str(root/n):h for n,h in old['prepared_hashes'].items()}}.items():
        if sha256(Path(p))!=expected:raise ValueError('Original batch source/material changed')
        inputs[p]=expected
    if result['status']!='completed_no_candidate' or result['candidates'] or read(root/'candidate_order.json') or ledger['unknown_billing']:
        raise ValueError('Recovery requires a completed induction-only batch with known billing')
    if ledger['limits']!={**old['new_limits'],'finish_reserve':0} or any(a['status']!='metered' for a in ledger['attempts'].values()):
        raise ValueError('Original allowance or billing changed')
    source=read(root/'source_cases.json');rows=records(source)
    if rows!=read(root/'source_records.json'):raise ValueError('Original public source records changed')
    by_key={r['task']['physical_key']:r for r in rows};selected=[t['physical_key'] for stage in old['selection_tasks'].values() for t in stage]
    recovered=[];attempt_ids=set()
    for i,unit in enumerate(old['candidate_units']):
        folder=root/'learning'/str(i);audit=read(folder/'requests.json');log=read(folder/'learning.json')
        if len(audit['requests'])!=1 or log['decision']!='rejected':raise ValueError('Recovery cannot repeat candidate learning or selection')
        request=audit['requests'][0];attempts=request['http_attempts']
        if request['purpose']!='capability_induction' or request['stage']!='extractor' or request['repair']!=0 or len(attempts)!=1:
            raise ValueError('Only the original unrepaired induction response may be recovered')
        attempt=attempts[0];wire=attempt['final_payload_audit'];unit_rows=[by_key[k] for k in unit]
        public=[{k:r[k] for k in ('task','task_hash','spans')} for r in unit_rows];unit_id=digest([VERSION,unit_rows,sorted(selected)])
        if expand_material(json.loads(wire['messages'][1]['content']))!={'public_cases':public,'learning_material_version':MATERIAL_VERSION}:
            raise ValueError('Paid demand was not issued for the exact public source material')
        tools=wire['tools'];params=wire['serialized_parameters']
        if len(tools)!=1 or tools[0]['function']['name']!='submit_capability' or tools[0]['function']['parameters']!=generation_schema(CAPABILITY,prompt_revision=old['prompt_revision']) or params['model']!=old['settings']['source']['actual_model_request_label']:
            raise ValueError('Paid demand schema or model changed')
        if params['max_tokens']!=2048 or params['thinking']!={'type':'disabled'} or wire['tool_choice']!={'type':'function','function':{'name':'submit_capability'}}:
            raise ValueError('Paid demand inference parameters changed')
        if request['owner_state_version']!='capability_batch_'+unit_id+'_demand_'+MATERIAL_VERSION.replace('.','_').replace('-','_'):
            raise ValueError('Paid induction owner changed')
        calls=request['response']['tool_calls'];choices=attempt['public_response']['choices']
        if len(calls)!=1 or calls[0]['name']!='submit_capability' or len(choices)!=1 or choices[0]['finish_reason']!='tool_calls' or attempt['outcome']!='success':
            raise ValueError('Paid demand response is incomplete')
        raw_calls=choices[0]['message']['tool_calls']
        if len(raw_calls)!=1 or raw_calls[0]['function']['name']!='submit_capability' or json.loads(raw_calls[0]['function']['arguments'])!=calls[0]['arguments']:
            raise ValueError('Parsed demand differs from actual provider response')
        identity=attempt['request_id'];entry=ledger['attempts'][identity]
        if identity in attempt_ids or entry['actual_usage']!=attempt['raw_usage'] or entry['accounted_tokens']!=attempt['raw_usage']['total_tokens'] or entry['context']['parent_scope']!=learning_scope(old,'candidate:'+str(i)):
            raise ValueError('Paid demand billing or attribution changed')
        validate_capability(calls[0]['arguments'],unit_rows);attempt_ids.add(identity)
        recovered.append({'unit_id':unit_id,'public_cases_hash':digest(public),'value':calls[0]['arguments'],
            'request_path':str(folder/'requests.json'),'request_sha256':sha256(folder/'requests.json'),
            'request_id':request['id'],'http_attempt_id':identity,'original_code':old['code'],
            'recovery_mode':'original_paid_induction_revalidated_without_http'})
        for name in ('requests.json','learning.json'):inputs[str(folder/name)]=sha256(folder/name)
    spent=sum(a['accounted_tokens'] for a in ledger['attempts'].values());http=len(attempt_ids)
    if attempt_ids!=set(ledger['attempts']) or result['new_tokens']!=spent or result['new_http']!=http or result['unknown_billing']:
        raise ValueError('Original batch cost differs from the physical ledger')
    if spent>=old['new_limits']['token_limit'] or http>=old['new_limits']['request_limit']:raise ValueError('No unchanged P2 allowance remains')
    for name in ('result.json','budget.json','plan_identity.json','candidate_order.json'):inputs[str(root/name)]=sha256(root/name)
    return {'parent_plan':str(path),'parent_run_id':old['run_id'],'demands':recovered,'prior_p2_tokens':spent,'prior_p2_http':http,
        'allowance':old['new_limits'],'remaining_tokens':old['new_limits']['token_limit']-spent,'remaining_http':old['new_limits']['request_limit']-http,
        'inherited_tokens':old['inherited_tokens']+spent,'inherited_http':old['inherited_http']+http},ledger,inputs


def recover(plan_path,output):
    recovery,ledger,inputs=recovery_inputs(plan_path);old=read(plan_path);root=Path(plan_path).resolve().parent;output=Path(output).resolve()
    code=code_identity()
    if output.exists() or code['tracked_dirty'] or not code['git_sha']:raise ValueError('Recovery requires fresh output and committed clean source')
    config=read(root/'config.json');new_config=configuration(config,output)
    expected=deepcopy(config);expected['data_dir']=new_config['data_dir'];expected['experiment']['output_dir']=new_config['experiment']['output_dir']
    if expected!=new_config:raise ValueError('Recovery may not change method parameters')
    for name in ('source_records.json','source_cases.json'):write_json(output/name,read(root/name))
    write_json(output/'config.json',new_config);write_json(output/'inherited_p2_budget.json',ledger)
    plan={**old,'prompt_revision':PROMPT_REVISION,'scope_policy':SCOPE_POLICY,'run_id':'p2_'+uuid4().hex,'code':code,'prepared_at':utc(),'derivation_mode':'explicit_induction_contract_recovery',
        'recovery':recovery,'input_hashes':inputs,'inherited_tokens':recovery['inherited_tokens'],'inherited_http':recovery['inherited_http'],
        'prepared_hashes':{n:sha256(output/n) for n in ('config.json','source_records.json','source_cases.json','inherited_p2_budget.json')}}
    write_json(output/'plan.json',plan);write_json(output/'plan_identity.json',{'plan_hash':digest(plan)})
    return plan


def revision_inputs(plan_path,*,continuation=False,source_version=VERSION,experimental_revision='P2-R2'):
    """A named explicit revision; completed negative evidence stays immutable."""
    path=Path(plan_path).resolve();root=path.parent;old=read(path);result=read(root/'result.json');ledger=read(root/'budget.json')
    if experimental_revision in {'P2-R4','P2-R5'}:
        parent,revision=('P2-R3','procedural-actions.r3') if experimental_revision=='P2-R4' else ('P2-R4','procedural-actions.r5')
        if (old.get('paid_prefix',{}).get('experimental_revision')!=parent or old.get('prompt_revision')!=revision or
            (experimental_revision=='P2-R4' and not old.get('protocol_continuation'))):
            raise ValueError(experimental_revision+' requires its completed original parent; no automatic further revision')
        continuation=True
    elif experimental_revision!='P2-R2':raise ValueError('Unknown paid experimental revision')
    if old['schema']!=source_version or ('paid_prefix' in old and not continuation) or digest(old)!=read(root/'plan_identity.json')['plan_hash'] or result['plan_hash']!=digest(old):
        raise ValueError('Original batch identity changed or already revised')
    inputs={str(path):sha256(path)}
    for p,expected in {**old['input_hashes'],**{str(root/n):h for n,h in old['prepared_hashes'].items()}}.items():
        if sha256(Path(p))!=expected:raise ValueError('Original batch source/material changed')
        inputs[p]=expected
    if result['status']!='completed_no_candidate' or result['candidates'] or read(root/'candidate_order.json') or ledger['unknown_billing']:
        raise ValueError('Revision requires a completed negative batch with known billing')
    if ledger['limits']!={**old['new_limits'],'finish_reserve':0} or any(a['status']!='metered' for a in ledger['attempts'].values()):
        raise ValueError('Original allowance or billing changed')
    tokens=sum(a['accounted_tokens'] for a in ledger['attempts'].values());http=len(ledger['attempts']);prior=old.get('paid_prefix',old.get('recovery',{}))
    inherited=old['inherited_tokens']+tokens-prior.get('prior_p2_tokens',0);inherited_http=old['inherited_http']+http-prior.get('prior_p2_http',0)
    if result['cumulative_tokens']!=inherited or result['cumulative_http']!=inherited_http or result['unknown_billing']:
        raise ValueError('Original cumulative cost differs from ledger')
    if tokens>=old['new_limits']['token_limit'] or http>=old['new_limits']['request_limit']:raise ValueError('No unchanged P2 allowance remains')
    for folder in (root/'learning').iterdir():
        for name in ('learning.json','requests.json'):
            p=folder/name
            if p.exists():inputs[str(p)]=sha256(p)
    for name in ('result.json','budget.json','plan_identity.json','candidate_order.json'):inputs[str(root/name)]=sha256(root/name)
    return {'parent_plan':str(path),'parent_run_id':old['run_id'],'experimental_revision':experimental_revision,
        'prior_p2_tokens':tokens,'prior_p2_http':http,'allowance':old['new_limits'],
        'remaining_tokens':old['new_limits']['token_limit']-tokens,'remaining_http':old['new_limits']['request_limit']-http,
        'inherited_tokens':inherited,'inherited_http':inherited_http},ledger,inputs


def continuation_inputs(plan_path):
    root=Path(plan_path).resolve().parent;old=read(plan_path)
    if old.get('paid_prefix',{}).get('experimental_revision')!='P2-R2' or old.get('continuation'):
        raise ValueError('Only the authorized R2 pending repair may continue')
    prefix,ledger,inputs=revision_inputs(plan_path,continuation=True)
    pending=[];namespaces={}
    for i,unit in enumerate(old['candidate_units']):
        folder=root/'learning'/str(i);audit=read(folder/'requests.json');log=read(folder/'learning.json')
        refused=[q for q in audit['requests'] if q.get('admission_status')=='local_refusal' and q.get('error','').startswith('material_too_large:')]
        if refused:
            if len(refused)!=1 or refused[0]['purpose']!='guidance_learning' or refused[0]['repair']!=1 or refused[0].get('http_attempts'):
                raise ValueError('Continuation cannot add a repair HTTP or learning round')
            paid=[q for q in audit['requests'] if q.get('http_attempts')]
            if len(paid)!=2 or [q['purpose'] for q in paid]!=['capability_induction','guidance_learning'] or any(q['repair']!=0 or len(q['http_attempts'])!=1 for q in paid):
                raise ValueError('Pending repair paid provenance changed')
            state=read(folder/'checkpoint/state.json');unit_key='capability_batch_'+log['unit_id']
            decision=state['decisions'][refused[0]['logical_decision_id']]
            if state.get(unit_key)!=log or decision['status']!='prepared' or decision['repair_index']!=1:
                raise ValueError('Repair is not the original unissued checkpoint decision')
            pending.append({'index':i,'unit_key':unit_key,'logical_decision_id':refused[0]['logical_decision_id']})
        elif log['decision']!='rejected' or len(audit['requests'])!=1 or audit['requests'][0]['purpose']!='capability_induction':
            raise ValueError('Other learning units must remain terminal model rejections')
        namespaces[str(i)]=str(folder/'checkpoint')
        for file in (folder/'checkpoint').rglob('*'):
            if file.is_file():inputs[str(file)]=sha256(file)
    if len(pending)!=1:raise ValueError('Exactly one original unissued repair is required')
    return {**prefix,'continuation_kind':'unissued_guidance_repair_capacity_fix','pending':pending,'response_namespaces':namespaces},ledger,inputs


def continue_repair(plan_path,output):
    prefix,ledger,inputs=continuation_inputs(plan_path);old=read(plan_path);root=Path(plan_path).resolve().parent;output=Path(output).resolve();code=code_identity()
    if output.exists() or code['tracked_dirty'] or not code['git_sha']:raise ValueError('Continuation requires fresh output and committed clean source')
    config=read(root/'config.json');new_config=configuration(config,output)
    expected=deepcopy(config);expected['data_dir']=new_config['data_dir'];expected['experiment']['output_dir']=new_config['experiment']['output_dir']
    if expected!=new_config:raise ValueError('Repair continuation may not change method parameters')
    for name in ('source_records.json','source_cases.json'):write_json(output/name,read(root/name))
    write_json(output/'config.json',new_config);write_json(output/'inherited_p2_budget.json',ledger)
    for i in prefix['response_namespaces']:
        folder=output/'learning'/i;shutil.copytree(root/'learning'/i/'checkpoint',folder/'checkpoint')
        shutil.copy2(root/'learning'/i/'requests.json',folder/'requests.json')
    for pending in prefix['pending']:
        state_path=output/'learning'/str(pending['index'])/'checkpoint/state.json';state=read(state_path)
        state.pop(pending['unit_key']);write_json(state_path,state)
    plan={**old,'scope_policy':SCOPE_POLICY,'run_id':'p2_'+uuid4().hex,'code':code,'prepared_at':utc(),'derivation_mode':'explicit_unissued_repair_continuation',
        'continuation':True,'paid_prefix':prefix,'input_hashes':inputs,'inherited_tokens':prefix['inherited_tokens'],'inherited_http':prefix['inherited_http'],
        'prepared_hashes':{n:sha256(output/n) for n in ('config.json','source_records.json','source_cases.json','inherited_p2_budget.json')}}
    write_json(output/'plan.json',plan);write_json(output/'plan_identity.json',{'plan_hash':digest(plan)})
    return plan


def prior_content_checks(plan_path,ledger):
    """Revalidate immutable checker outputs; a longer reason never changes their classification."""
    root=Path(plan_path).resolve().parent;old=read(plan_path);rows=[]
    for i in range(len(old['candidate_units'])):
        folder=root/'learning'/str(i);log=read(folder/'learning.json')
        requests=[q for q in read(folder/'requests.json')['requests'] if q['purpose']=='guidance_grounding']
        if not requests:
            rows.append({'unit':i,'original_decision':log['decision'],'original_reason':log.get('reason'),
                'stage':log['stage'],'status':'not_run','published':False,'new_model_http':0})
            continue
        if len(requests)!=1 or len(requests[0]['http_attempts'])!=1:raise ValueError('Original content-check request changed')
        candidate=log['proposed_candidate']
        request=requests[0];attempt=request['http_attempts'][0];calls=request['response']['tool_calls']
        raw=attempt['public_response']['choices'];wire=attempt['final_payload_audit'];entry=ledger['attempts'][attempt['request_id']]
        if (len(calls)!=1 or calls[0]['name']!='submit_capability_check' or len(raw)!=1 or raw[0]['finish_reason']!='tool_calls' or
            len(raw[0]['message']['tool_calls'])!=1 or len(wire['tools'])!=1 or request['repair']!=0 or
            raw[0]['message']['tool_calls'][0]['function']['name']!='submit_capability_check' or
            json.loads(raw[0]['message']['tool_calls'][0]['function']['arguments'])!=calls[0]['arguments'] or
            wire['tools'][0]['function']['parameters']!=generation_schema(CHECK,prompt_revision=old['prompt_revision']) or
            attempt['outcome']!='success' or entry['actual_usage']!=attempt['raw_usage'] or
            entry['context']['logical_decision_id']!=request['logical_decision_id']):
            raise ValueError('Original content-check response, schema or billing changed')
        value=calls[0]['arguments'];error=None;passed=False
        try:passed=validate_check(value,candidate)
        except ValueError as exc:error=str(exc)
        rows.append({'unit':i,'original_decision':log['decision'],'original_reason':log.get('reason'),
            'physical_request_id':attempt['request_id'],'request_sha256':sha256(folder/'requests.json'),
            'reason_characters':len(value.get('reason','')),'status':value['status'],'structure_valid':error is None,'schema_error':error,
            'content_passed':passed,'published':False,'new_model_http':0})
    return {'prompt_revision':PROMPT_REVISION,'new_model_http':0,'new_tokens':0,'units':rows,'published':False}


def revise(plan_path,output,*,experimental_revision='P2-R2'):
    prefix,ledger,inputs=revision_inputs(plan_path,experimental_revision=experimental_revision);old=read(plan_path);root=Path(plan_path).resolve().parent;output=Path(output).resolve()
    code=code_identity()
    if output.exists() or code['tracked_dirty'] or not code['git_sha']:raise ValueError('Revision requires fresh output and committed clean source')
    config=read(root/'config.json');new_config=configuration(config,output)
    expected=deepcopy(config);expected['data_dir']=new_config['data_dir'];expected['experiment']['output_dir']=new_config['experiment']['output_dir']
    if experimental_revision=='P2-R2' and expected!=new_config:raise ValueError('Prompt revision may not change method parameters')
    replay=prior_content_checks(plan_path,ledger) if experimental_revision in {'P2-R4','P2-R5'} else None
    for name in ('source_records.json','source_cases.json'):write_json(output/name,read(root/name))
    write_json(output/'config.json',new_config);write_json(output/'inherited_p2_budget.json',ledger)
    plan={**{k:v for k,v in old.items() if k not in {'recovery','protocol_continuation'}},'prompt_revision':PROMPT_REVISION,'scope_policy':SCOPE_POLICY,'run_id':'p2_'+uuid4().hex,'code':code,'prepared_at':utc(),
        'derivation_mode':('explicit_'+experimental_revision+'_from_same_completed_train' if replay is not None else 'user_authorized_P2_R2_from_same_completed_train'),
        'paid_prefix':prefix,'input_hashes':inputs,
        'inherited_tokens':prefix['inherited_tokens'],'inherited_http':prefix['inherited_http'],
        'prepared_hashes':{n:sha256(output/n) for n in ('config.json','source_records.json','source_cases.json','inherited_p2_budget.json')}}
    if replay is not None:
        write_json(output/'prior_content_checks.json',replay)
        plan['prepared_hashes']['prior_content_checks.json']=sha256(output/'prior_content_checks.json')
        plan['config_transition']={'parent':config,'current':new_config}
    write_json(output/'plan.json',plan);write_json(output/'plan_identity.json',{'plan_hash':digest(plan)})
    return plan


def learning_boundary(system,root,key,*,response_namespace=None,unit_limits=None):
    system.checkpoint=TaskCheckpoint(root/'checkpoint',response_namespace=response_namespace);system.audit_path=str(root/'requests.json')
    system.phase='train';system.budget_scope=key;system._task_start=len(system.usage.events);system._request_start=len(system.requests)
    system._learning_start=system._task_start
    system.prior_usage=read(system.audit_path)['usage'] if Path(system.audit_path).exists() else []
    system.request_attribution=({'parent_scope':key,'parent_token_limit':unit_limits['token_limit'],'train_parent':False,
        'episode_scope':key,'episode_request_limit':unit_limits['request_limit']} if unit_limits else
        {'parent_scope':key,'parent_token_limit':system.config['budget']['train_learning_tokens'],'train_parent':True})


def budget_continuation_inputs(plan_path):
    """Reuse one metered induction before an unissued local pool refusal, retaining every global charge."""
    path=Path(plan_path).resolve();root=path.parent;old=read(path);result=read(root/'result.json');ledger=read(root/'budget.json')
    if (old['schema']!=VERSION or old.get('prompt_revision') not in {'procedural-actions.r6',PROMPT_REVISION} or old.get('paid_prefix',{}).get('experimental_revision')!='P2-R5' or old.get('budget_continuation') or
        digest(old)!=read(root/'plan_identity.json')['plan_hash'] or result['plan_hash']!=digest(old) or
        result['status']!='stopped_budget_censored' or result.get('reason')!='learning pool HTTP admission refused'):
        raise ValueError('Only the original R5 local pool refusal may continue')
    inputs={str(path):sha256(path)}
    for p,h in {**old['input_hashes'],**{str(root/n):h for n,h in old['prepared_hashes'].items()}}.items():
        if sha256(Path(p))!=h:raise ValueError('Original budget continuation source changed')
        inputs[p]=h
    if ledger['unknown_billing'] or ledger['limits']!={**old['new_limits'],'finish_reserve':0} or any(a['status']!='metered' for a in ledger['attempts'].values()):
        raise ValueError('Original billing or allowance changed')
    previous=read(root/'inherited_p2_budget.json');added=set(ledger['attempts'])-set(previous['attempts'])
    if len(added)!=1 or any(ledger['attempts'].get(k)!=v for k,v in previous['attempts'].items()):raise ValueError('Original paid prefix changed')
    audit=read(root/'learning/0/requests.json')['requests']
    if (len(audit)!=2 or [q['purpose'] for q in audit]!=['capability_induction','guidance_learning'] or any(q['repair']!=0 for q in audit) or
        audit[1].get('admission_status')!='local_refusal' or audit[1].get('error')!=result['reason'] or audit[1]['http_attempts']):
        raise ValueError('Continuation cannot add a proposal attempt or repair')
    request=audit[0]
    if len(request['http_attempts'])!=1 or len(request['response']['tool_calls'])!=1:raise ValueError('Original induction must have one physical response')
    attempt=request['http_attempts'][0];value=request['response']['tool_calls'][0]['arguments'];entry=ledger['attempts'][attempt['request_id']]
    rows=records(read(root/'source_cases.json'));by_key={r['task']['physical_key']:r for r in rows}
    source=[by_key[k] for k in old['candidate_units'][0]];public=[{k:r[k] for k in ('task','task_hash','spans')} for r in source]
    wire=attempt['final_payload_audit'];native=attempt['public_response']['choices'][0]
    if (attempt['request_id'] not in added or entry['actual_usage']!=attempt['raw_usage'] or
        entry['accounted_tokens']!=attempt['raw_usage']['total_tokens'] or entry['context']['parent_scope']!=learning_scope(old,'candidate:0') or
        native['finish_reason']!='tool_calls' or len(native['message']['tool_calls'])!=1 or native['message']['tool_calls'][0]['function']['name']!='submit_capability' or
        json.loads(native['message']['tool_calls'][0]['function']['arguments'])!=value or
        wire['tools'][0]['function']['parameters']!=generation_schema(CAPABILITY,prompt_revision=old['prompt_revision']) or
        expand_material(json.loads(wire['messages'][1]['content']))!={'public_cases':public,'learning_material_version':MATERIAL_VERSION}):
        raise ValueError('Original induction response, source or charge changed')
    validate_capability(value,source)
    selected=[t['physical_key'] for stage in old['selection_tasks'].values() for t in stage]
    demand={'unit_id':digest([VERSION,source,sorted(selected)]),'public_cases_hash':digest(public),'value':value,
        'request_id':request['id'],'http_attempt_id':attempt['request_id'],'original_code':old['code'],'new_model_http':0}
    tokens=sum(a['accounted_tokens'] for a in ledger['attempts'].values());http=len(ledger['attempts']);prior=old['paid_prefix']
    inherited=old['inherited_tokens']+tokens-prior['prior_p2_tokens'];inherited_http=old['inherited_http']+http-prior['prior_p2_http']
    if result['total_p2_tokens']!=tokens or result['total_p2_http']!=http or result['cumulative_tokens']!=inherited or result['cumulative_http']!=inherited_http:
        raise ValueError('Continuation cost differs from the complete physical ledger')
    if tokens>=old['new_limits']['token_limit'] or http>=old['new_limits']['request_limit']:raise ValueError('No original allowance remains')
    for p in [*(root/'learning').rglob('*'),*[root/n for n in ('result.json','budget.json','plan_identity.json')]]:
        if p.is_file():inputs[str(p)]=sha256(p)
    prefix={'parent_plan':str(path),'parent_run_id':old['run_id'],'experimental_revision':'P2-R5','continuation_kind':'run_local_learning_budget_scope_fix',
        'prior_p2_tokens':tokens,'prior_p2_http':http,'allowance':old['new_limits'],'remaining_tokens':old['new_limits']['token_limit']-tokens,
        'remaining_http':old['new_limits']['request_limit']-http,'inherited_tokens':inherited,'inherited_http':inherited_http}
    return {'parent_plan':str(path),'paid_prefix':prefix,'demands':[demand]+[None]*(len(old['candidate_units'])-1)},ledger,inputs


def continue_budget(plan_path,output):
    restored,ledger,inputs=budget_continuation_inputs(plan_path);prefix=restored['paid_prefix'];old=read(plan_path)
    output=Path(output).resolve();root=Path(plan_path).resolve().parent;code=code_identity()
    if output.exists() or code['tracked_dirty'] or not code['git_sha']:raise ValueError('Budget continuation requires fresh output and committed clean source')
    original=read(root/'config.json');config=configuration(original,output)
    for name in ('source_cases.json','source_records.json'):write_json(output/name,read(root/name))
    write_json(output/'config.json',config);write_json(output/'inherited_p2_budget.json',ledger)
    plan={**old,'prompt_revision':PROMPT_REVISION,'scope_policy':SCOPE_POLICY,'run_id':'p2_'+uuid4().hex,'code':code,'prepared_at':utc(),
        'derivation_mode':'explicit_unissued_proposal_budget_scope_fix','budget_continuation':restored,'paid_prefix':prefix,
        'config_transition':{'parent':original,'current':config},
        'input_hashes':inputs,'inherited_tokens':prefix['inherited_tokens'],'inherited_http':prefix['inherited_http'],
        'prepared_hashes':{n:sha256(output/n) for n in ('config.json','source_records.json','source_cases.json','inherited_p2_budget.json')}}
    write_json(output/'plan.json',plan);write_json(output/'plan_identity.json',{'plan_hash':digest(plan)})
    return plan


def protocol_inputs(receipt_path):
    """Recompute saved evidence before crossing an explicitly recorded code boundary."""
    from . import replay_choice_protocol as replay
    receipt_path=Path(receipt_path).resolve();receipt=read(receipt_path);root=receipt_path.parent
    receipt_hash=digest(receipt)
    if receipt_hash!=read(root/'receipt_identity.json')['receipt_hash']:raise ValueError('Protocol receipt identity changed')
    with TemporaryDirectory(prefix='p2-protocol-check-') as temporary:
        recomputed=replay.replay(receipt['parent_plan'],Path(temporary)/'verified')
    stable=lambda r:{k:v for k,v in r.items() if k not in {'code','created_at'}}
    if stable(receipt)!=stable(recomputed):raise ValueError('Saved protocol receipt differs from production revalidation')
    ledger=read(root/'inherited_p2_budget.json')
    _,original,inputs=revision_inputs(receipt['parent_plan'],continuation=True,source_version=replay.LEGACY)
    if ledger!=original:raise ValueError('Protocol continuation cannot refill the allowance')
    demands=[];proposals=[]
    by_key={r['task']['physical_key']:r for r in read(Path(receipt['parent_plan']).parent/'source_records.json')}
    for unit in receipt['units']:
        if unit['status'] not in {'ready_for_independent_content_check','ready_for_guidance_generation'}:
            raise ValueError('Only a pending legal demand or proposal may enter paid validation')
        rows=[by_key[k] for k in unit['source_case_ids']]
        public=[{k:r[k] for k in ('task','task_hash','spans')} for r in rows]
        demands.append({'unit_id':unit['new_unit_id'],'public_cases_hash':digest(public),
            'value':unit['demand_response']['arguments'],'original_response':unit['demand_response'],'receipt_hash':receipt_hash})
        proposals.append({'unit_id':unit['new_unit_id'],'capability_hash':digest(unit['capability']),
            'source_records_hash':digest(rows),'value':unit['proposal_responses'][-1]['arguments'],
            'original_response':unit['proposal_responses'][-1],'receipt_hash':receipt_hash} if unit['candidate'] else None)
    for name in ('protocol_replay.json','receipt_identity.json','inherited_p2_budget.json'):
        inputs[str(root/name)]=sha256(root/name)
    return {'receipt':str(receipt_path),'receipt_hash':receipt_hash,'parent_code':receipt['code'],
        'paid_prefix':{**receipt['paid_prefix'],'derivation_mode':'paid_protocol_continuation'},
        'demands':demands,'proposals':proposals},ledger,inputs


def continue_protocol(receipt_path,output):
    restored,ledger,inputs=protocol_inputs(receipt_path);prefix=restored['paid_prefix'];old=read(prefix['parent_plan'])
    output=Path(output).resolve();code=code_identity()
    if output.exists() or code['tracked_dirty'] or not code['git_sha']:raise ValueError('Protocol continuation requires fresh output and committed clean source')
    parent=Path(prefix['parent_plan']).parent;original=read(parent/'config.json');config=configuration(original,output)
    for name in ('source_cases.json','source_records.json'):write_json(output/name,read(parent/name))
    write_json(output/'config.json',config);write_json(output/'inherited_p2_budget.json',ledger)
    plan={**{k:v for k,v in old.items() if k not in {'recovery','continuation','paid_prefix'}},
        'schema':VERSION,'prompt_revision':PROMPT_REVISION,'scope_policy':SCOPE_POLICY,'run_id':'p2_'+uuid4().hex,'code':code,'prepared_at':utc(),
        'derivation_mode':'explicit_P2_R3_paid_protocol_continuation','protocol_continuation':restored,'paid_prefix':prefix,
        'config_transition':{'parent':original,'current':config},'input_hashes':inputs,
        'inherited_tokens':prefix['inherited_tokens'],'inherited_http':prefix['inherited_http'],
        'prepared_hashes':{n:sha256(output/n) for n in ('config.json','source_records.json','source_cases.json','inherited_p2_budget.json')}}
    write_json(output/'plan.json',plan);write_json(output/'plan_identity.json',{'plan_hash':digest(plan)})
    return plan


def evaluator_records(summary):
    records={}
    for trace in summary['cases']:
        raw=trace['scoring_audit']['raw_scorer_output'];choices=trace['task']['inputs']['choices']
        choice=next(c for c in choices if c['label']==raw['correct_label'])
        if choice['text']!=raw['correct_text']:raise ValueError('Saved evaluator label/text identity differs')
        records[trace['task']['task_id']]={'correct_choice':deepcopy(choice),'choices':deepcopy(choices)}
    return records


def evaluate_pair(root,task,candidate,matches,config,governor,trial,evaluator,rank,candidate_index,*,run_id=None,observer=None):
    pair={};order=('off','on') if (rank+candidate_index)%2==0 else ('on','off')
    for arm in order:
        task_root=root/str(rank)/arm;cfg=deepcopy(config);cfg['data_dir']=str(trial);cfg['llm']['max_retries']=0
        cfg['experiment'].update(runtime_mode='frozen',output_dir=str(task_root))
        target=EmpiricalSystem(cfg,harness=AnswerAdapter('livemath',evaluator),budget_governor=governor,
            bank_view_factory=lambda b:CapabilityView(b,matches,off=arm=='off',candidate=candidate))
        target.observer=observer
        try:
            target.checkpoint=TaskCheckpoint(task_root/'checkpoint');target.audit_path=str(task_root/'requests.json')
            trace_path=task_root/'trace.json';attempt=VERSION+':'+str(candidate_index)+':'+str(rank)+':'+arm
            if run_id:attempt=run_id+':'+attempt
            if observer:observer.begin_task(task,rank,attempt,{'purpose':'train_internal_paired_selection','arm':arm})
            if trace_path.exists():
                trace=read(trace_path)
                if trace['task']!=task.__dict__ or trace['attempt_id']!=attempt or trace['method_identity']['config_hash']!=digest(target.config):
                    raise ValueError('Checkpoint method identity changed')
                if target.checkpoint.state.get('paired_trace_hash')!=digest(trace) or trace['knowledge_before']!=target.bank.digest():
                    raise ValueError('Saved pair trace or Frozen Bank changed')
            else:
                trace=target.run_task(task,learn=False,attempt_id=attempt)
                write_json(trace_path,trace)
                target.checkpoint.advance(target.checkpoint.state['stage'],paired_trace_hash=digest(trace))
            if trace.get('error') or trace['execution'].get('budget_censored'):raise BudgetExhausted('parent_task_budget_exhausted','Selection arm censored')
            if observer:
                observer.requests(trace['requests'])
                observer.scorer(task.task_id,attempt,trace['score'],phase='selection',audit=trace.get('scoring_audit',{}))
                observer.training(attempt,'paired_validation',observer.current_task['task_start_time'],utc(),1,'completed',
                    consumed_physical_keys=[task.physical_key],candidate_id=candidate['id'],arm=arm,
                    injected_guidance_ids=trace.get('injected_guidance_ids',[]))
            pair[arm]=trace
        finally:target.close()
        if governor.billing_blocked:raise BudgetExhausted('unknown_billing','Selection billing unresolved')
    pair_fact(candidate,task,pair)
    return pair


def execute(plan_path,env_file,*,governor=None,observer=None,frozen_destination=None):
    check(plan_path);plan=read(plan_path);root=Path(plan_path).parent;config=read(root/'config.json')
    recovery=plan.get('recovery',{})
    restored=plan.get('protocol_continuation',plan.get('budget_continuation',{}))
    prefix=plan.get('paid_prefix',recovery)
    if prefix and not (root/'budget.json').exists():write_json(root/'budget.json',read(root/'inherited_p2_budget.json'))
    if env_file:load_env(env_file)
    shared=governor is not None
    if not shared:governor=BudgetGovernor(root/'budget.json',**plan['new_limits'],finish_reserve=0)
    baseline=plan.get('campaign_budget_prefix',{}) if shared else {}
    result_path=root/'result.json';result=read(result_path) if result_path.exists() else {'status':'running','run_id':plan['run_id'],'plan_hash':digest(plan),'candidates':[],'started_at':utc()}
    if result['plan_hash']!=digest(plan):raise ValueError('Batch identity changed')
    if result['status']!='running':return result
    write_json(result_path,result);progress(root,'started',run_id=plan['run_id'],status='running')
    system=EmpiricalSystem(config,harness=AnswerAdapter('livemath',{}),budget_governor=governor)
    system.observer=observer;system.bank.observer=observer
    try:
        if governor.billing_blocked:raise BudgetExhausted('unknown_billing','No retry after unresolved billing')
        for row in read(root/'source_cases.json'):system.bank.save_case(PublicTask(**row['task']),row['experience'])
        selection=[t for rows in plan['selection_tasks'].values() for t in rows];selected=[t['physical_key'] for t in selection];candidates=[]
        learning=[]
        result.update(planned_units=len(plan['candidate_units']),attempted_units=0,seen_sources=[],cited_support_sources=[],
            unattempted_source_keys=[key for group in plan['candidate_units'] for key in group])
        for i,unit in enumerate(plan['candidate_units']):
            result.update(attempted_units=i+1,
                unattempted_source_keys=[key for group in plan['candidate_units'][i+1:] for key in group])
            write_json(result_path,result);stop=None
            if observer:
                observer.begin_task(PublicTask(plan['run_id'],plan['run_id'],'Post-Train capability learning'),60,
                    learning_scope(plan,'candidate:'+str(i)),{'source_physical_keys':unit})
                started=utc()
            learning_boundary(system,root/'learning'/str(i),learning_scope(plan,'candidate:'+str(i)),
                response_namespace=prefix.get('response_namespaces',{}).get(str(i)),unit_limits=plan.get('unit_limits'))
            cached=root/'learning'/str(i)/'learning.json'
            if cached.exists():
                log=read(cached)
                if 'source_schedule' in plan and (log.get('source_case_ids')!=unit or
                    log.get('receipt_hash')!=digest({k:v for k,v in log.items() if k!='receipt_hash'})):
                    raise ValueError('Saved learning terminal receipt changed')
            else:
                try:
                    log=system.learner.learn_capability_batch(unit,selected,
                        recovered_demand=restored['demands'][i] if restored else recovery['demands'][i] if recovery else None,
                        recovered_proposal=restored['proposals'][i] if restored.get('proposals') else None)
                except ValueError as exc:
                    if getattr(exc,'code',None)!='material_too_large':raise
                    from .capability_schedule import full_contract
                    log={'decision':'capacity_rejected' if full_contract(plan['settings']) else 'rejected','reason':getattr(exc,'learning_outcome','material_too_large'),
                         'error':str(exc),'source_case_ids':unit,'stage':'material_admission'}
                except BudgetExhausted as exc:
                    log={'decision':'budget_censored','reason':str(exc),'source_case_ids':unit,'stage':'budget_admission'}
                    limits=plan.get('unit_limits',{});owned=[a for a in governor.state['attempts'].values()
                        if a['context'].get('parent_scope')==system.budget_scope]
                    room=governor.headroom({})
                    local=(not governor.billing_blocked and room['requests']>0 and room['tokens']>0 and
                        (exc.code in {'parent_task_budget_exhausted','empirical_token_budget_exhausted'} or
                         len(owned)>=limits.get('request_limit',float('inf'))))
                    if not local:stop=exc
                if 'source_schedule' in plan:
                    log['source_case_ids']=unit
                    log['receipt_hash']=digest(log)
            write_json(root/'learning'/str(i)/'learning.json',log)
            learning.append({**{k:log.get(k) for k in ('decision','stage','reason')},
                'unit_id':plan.get('source_schedule',{}).get('units',[{}]*len(plan['candidate_units']))[i].get('unit_id',str(i)),
                'source_case_ids':unit,
                'cited_support_sources':sorted(set((log.get('capability') or {}).get('source_case_ids',[])) |
                    set((log.get('proposed_candidate') or {}).get('source_case_ids',[]))),
                'proposal_created':bool(log.get('proposed_candidate')),
                **({'proposal_reused':True} if log.get('recovered_proposal') else {}),
                'source_check_status':log.get('source_check',{}).get('status')})
            seen=set(result['seen_sources']);unit_seen=set()
            audit=read(system.audit_path) if Path(system.audit_path).exists() else {'requests':[]}
            for request in audit['requests']:
                if not request.get('http_attempts'):continue
                body=expand_material(json.loads(request['messages'][1]['content']))
                for row in [*body.get('public_cases',[]),*body.get('train_cases',[]),*body.get('candidate',{}).get('source_records',[])]:
                    key=row.get('task',{}).get('physical_key')
                    if key in unit:seen.add(key);unit_seen.add(key)
            if observer:observer.training(learning_scope(plan,'candidate:'+str(i)),'capability_batch',started,utc(),len(unit_seen),
                log['decision'],log.get('reason'),planned_physical_keys=unit,consumed_physical_keys=sorted(unit_seen))
            result.update(learning_units=learning,completed_units=len(learning),
                attempted_source_keys=[key for group in plan['candidate_units'][:len(learning)] for key in group],
                seen_sources=sorted(seen),
                unattempted_source_keys=[key for group in plan['candidate_units'][len(learning):] for key in group],
                cited_support_sources=sorted({key for row in learning for key in row.get('cited_support_sources',[])}),
                natural_candidate_count=sum(r['proposal_created'] and not r.get('proposal_reused',False) for r in learning),
                revalidated_candidate_count=sum(r['proposal_created'] and r.get('proposal_reused',False) for r in learning))
            write_json(result_path,result)
            progress(root,'learning_unit_completed',unit=i,**learning[-1])
            if governor.billing_blocked:raise BudgetExhausted('unknown_billing','Learning billing unresolved')
            if stop:raise stop
            if log['decision']=='candidate':candidates.append(log['candidate'])
        if any(row['decision']=='budget_censored' for row in learning):
            raise BudgetExhausted('learning_budget_censored','At least one fixed learning unit was budget censored')
        if any(row['decision']=='capacity_rejected' for row in learning):
            result.update(status='stopped_material_capacity',reason='Required source material exceeded its declared wire capacity')
            return result
        candidates=sorted({c['id']:c for c in candidates}.values(),key=lambda c:(-len(c['source_case_ids']),digest(c)))
        result['source_checked_candidate_count']=len(candidates)
        count=plan['settings']['candidate']['max_candidates_to_selection']
        write_json(root/'candidate_order.json',[{'id':c['id'],'digest':digest(c),'source_coverage':len(c['source_case_ids']),
            'status':'selected_for_routing' if i<count else 'checked_not_selected'} for i,c in enumerate(candidates)])
        for i,candidate in enumerate(candidates):
            write_json(root/'candidates'/candidate['id']/'candidate.json',candidate)
            if i>=count:write_json(root/'candidates'/candidate['id']/'selection_status.json',{'status':'checked_not_selected'})
        candidates=candidates[:count];result['selected_for_routing_count']=len(candidates);write_json(result_path,result)
        trial=root/'trial_bank'
        if not trial.exists():system.bank.freeze(trial)
        evaluator=evaluator_records(read(Path(plan['source_run'])/'train/summary.json')) if candidates else {}
        receipts=[]
        for i,candidate in enumerate(candidates):
            folder=root/'candidates'/candidate['id'];preflight=[]
            for rank,row in enumerate(selection):
                if observer:observer.begin_task(PublicTask(**row),rank,learning_scope(plan,'routing:'+candidate['id']+':'+str(rank)),
                    {'purpose':'train_public_applicability'})
                task=PublicTask(**row);learning_boundary(system,folder/'routing'/str(rank),learning_scope(plan,'routing:'+candidate['id']+':'+str(rank)),unit_limits=plan.get('unit_limits'))
                match=system.learner.match_capability(candidate,task);validate_match(match,candidate['capability'],task)
                preflight.append({'task':row,'stage':'screen' if rank<len(plan['selection_tasks']['screen']) else 'confirmation','rank':rank,'match':match})
                progress(root,'public_match_completed',candidate_id=candidate['id'],task_id=task.task_id,rank=rank,applicable=match['applicable'])
                if governor.billing_blocked:raise BudgetExhausted('unknown_billing','Routing billing unresolved')
            write_json(folder/'preflight.json',preflight);eligible=[r for r in preflight if r['match']['applicable']];pairs={};screen=[]
            if len(eligible)>=plan['utility_policy']['min_precomputed_applicable']:
                for row in eligible:
                    if row['stage']=='confirmation' and not gate(screen,plan['utility_policy']['screen']):break
                    task=PublicTask(**row['task']);match={candidate['capability']['id']:row['match']}
                    progress(root,'pair_started',candidate_id=candidate['id'],task_id=task.task_id,rank=row['rank'])
                    pair=evaluate_pair(folder/'selection',task,candidate,match,config,governor,trial,evaluator,row['rank'],i,
                        run_id=plan['run_id'] if plan.get('scope_policy')==SCOPE_POLICY else None,observer=observer)
                    pairs[task.physical_key]=pair
                    progress(root,'pair_completed',candidate_id=candidate['id'],task_id=task.task_id,rank=row['rank'],
                        off_correct=pair['off']['score']['hard'],on_correct=pair['on']['score']['hard'])
                    if row['stage']=='screen':screen.append(pair_fact(candidate,task,pair))
            receipt=utility_receipt(candidate,pairs,plan['utility_policy'],preflight,{'code':plan['code'],'plan_hash':digest(plan)})
            write_json(folder/'utility_receipt.json',receipt);receipts.append((candidate,receipt))
            result['candidates']=[{'id':c['id'],'passed':r['passed'],'status':r['status'],'actual_exposed':len(r['paired'])} for c,r in receipts]
        for candidate,receipt in receipts:
            if receipt['passed']:
                if observer:observer.begin_task(PublicTask(plan['run_id'],plan['run_id'],'Post-Train capability publication'),60,
                    plan['run_id'],{'candidate_id':candidate['id']})
                publish(system.bank,candidate,receipt)
        frozen=Path(frozen_destination) if frozen_destination else root/'frozen_bank'
        newly_created=not frozen.exists()
        if frozen.exists():
            result['frozen']=read(frozen/'freeze.json')
            if result['frozen']['source_digest']!=system.bank.digest():raise ValueError('Frozen source changed')
        else:result['frozen']=system.bank.freeze(frozen)
        if observer:observer.freeze(frozen,newly_created=newly_created)
        result['status']='completed_selection' if candidates else 'completed_no_candidate'
        result.update(learning_units=learning,completion_reason='selection_finished' if candidates else 'no_source_checked_candidate',
            natural_candidate_count=sum(r['proposal_created'] and not r.get('proposal_reused',False) for r in learning),
            revalidated_candidate_count=sum(r['proposal_created'] and r.get('proposal_reused',False) for r in learning),
            paired_tasks=sum(c['actual_exposed'] for c in result['candidates']))
    except BudgetExhausted as exc:
        result.update(status='stopped_unknown_billing' if governor.billing_blocked else 'stopped_budget_censored',
            reason=str(exc),censored=True,learning_budget_censored=result.get('completed_units',0)<len(plan['candidate_units']) or
            any(row['decision']=='budget_censored' for row in result.get('learning_units',[])))
    except Exception as exc:
        result.update(status='stopped_unknown_billing' if governor.billing_blocked else 'stopped_engineering_error',reason=str(exc))
        if governor.billing_blocked:result['censored']=True
        if not governor.billing_blocked:raise
    finally:
        system.close();state=governor.state;charged=sum(a.get('accounted_tokens',a['reserved_tokens']) for a in state['attempts'].values())
        known=sum(a.get('accounted_tokens',0) for a in state['attempts'].values())
        known-=baseline.get('known_tokens',baseline.get('tokens',prefix.get('prior_p2_tokens',0)))
        charged-=baseline.get('tokens',prefix.get('prior_p2_tokens',0))
        http=len(state['attempts'])-baseline.get('http',prefix.get('prior_p2_http',0))
        result.update(ended_at=utc(),new_tokens=known,new_reserved_or_accounted_tokens=charged,new_http=http,unknown_billing=governor.billing_blocked,
            inherited_unknown_billing=bool(state.get('inherited_reservation_recovery')),
            inherited_tokens=plan['inherited_tokens'],inherited_http=plan['inherited_http'],
            cumulative_tokens=plan.get('inherited_known_tokens',plan['inherited_tokens'])+known,cumulative_reserved_or_accounted_tokens=plan['inherited_tokens']+charged,
            cumulative_http=plan['inherited_http']+http,
            source_train_runtime_tokens_counted_once=plan['source_train_runtime_tokens'],source_train_runtime_reexecuted=False)
        if prefix:result.update(prior_p2_tokens=prefix['prior_p2_tokens'],prior_p2_http=prefix['prior_p2_http'],
            total_p2_tokens=prefix['prior_p2_tokens']+known,total_p2_http=prefix['prior_p2_http']+http)
        if recovery:result['recovery_parent_run_id']=recovery['parent_run_id']
        if 'paid_prefix' in plan:result.update(revision_parent_run_id=prefix['parent_run_id'],experimental_revision=prefix['experimental_revision'])
        write_json(result_path,result)
        progress(root,'finished',status=result['status'],reason=result.get('completion_reason',result.get('reason')),
            new_tokens=known,new_http=http,candidates=result['candidates'])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('prepare');p.add_argument('--settings',required=True);p.add_argument('--source-run',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('recover');p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('revise');p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    p.add_argument('--revision',choices=('P2-R2','P2-R4','P2-R5'),default='P2-R2')
    p=sub.add_parser('continue-repair');p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('continue-budget');p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('continue-protocol');p.add_argument('--receipt',required=True);p.add_argument('--output',required=True)
    for mode in ('check','execute'):
        p=sub.add_parser(mode);p.add_argument('--plan',required=True)
        if mode=='execute':p.add_argument('--env-file',required=True)
    args=parser.parse_args()
    result=prepare(args.settings,args.source_run,args.output) if args.mode=='prepare' else recover(args.plan,args.output) if args.mode=='recover' else revise(args.plan,args.output,experimental_revision=args.revision) if args.mode=='revise' else continue_repair(args.plan,args.output) if args.mode=='continue-repair' else continue_budget(args.plan,args.output) if args.mode=='continue-budget' else continue_protocol(args.receipt,args.output) if args.mode=='continue-protocol' else check(args.plan) if args.mode=='check' else execute(args.plan,args.env_file)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
