"""Pinned Train-only experience recompile; prepare/check never call a model."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import yaml

from ..core.errors import BudgetExhausted
from ..empirical import POLICY_DEFAULTS
from ..empirical.bank import Bank
from ..empirical.budget_governor import BudgetGovernor, input_token_bound
from ..empirical.capability_batch import (VERSION, MATERIAL_VERSION, PROMPT_REVISION, CAPABILITY, CapabilityView, records, publish, utility_receipt,
    validate_policy, validate_match, validate_capability, gate, pair_fact)
from ..empirical.model_view import expand_material
from ..empirical.checkpoint import TaskCheckpoint
from ..empirical.contracts import PublicTask, digest
from ..empirical.system import EmpiricalSystem, validate_config
from ..harness.benchmarks import AnswerAdapter
from .run_empirical import code_identity, write_json, load_env
from .canonical_manifest import sha256
from .formal_log import utc


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


def prepare(settings_path,source_run,output):
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
    if ledger['unknown_billing'] or any(a['status']!='metered' for a in ledger['attempts'].values()):raise ValueError('Unresolved inherited billing')
    inherited=sum(a['accounted_tokens'] for a in ledger['attempts'].values());http=len(ledger['attempts'])
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
    units=[[t['physical_key'] for t in generation[i*width:(i+1)*width]] for i in range(count)]
    config=configuration(original,output)
    names=['run_manifest.json','train/execution_manifest.json','train/config.json','train/summary.json','train/bank/bank.sqlite3','budget.json']
    inputs={str(source/name):sha256(source/name) for name in names};inputs[str(Path(settings_path).resolve())]=sha256(Path(settings_path))
    inputs[str(canonical)]=sha256(canonical)
    write_json(output/'source_records.json',public);write_json(output/'source_cases.json',rows);write_json(output/'config.json',config)
    plan={'schema':VERSION,'prompt_revision':PROMPT_REVISION,'run_id':'p2_'+uuid4().hex,'code':code,'prepared_at':utc(),'source_run':str(source),'source_run_id':manifest['run_id'],
        'derivation_mode':'offline_recompile_from_completed_train','initial_bank':'fresh independent derived Bank; no P1 assets',
        'settings':spec,'input_hashes':inputs,'prepared_hashes':{n:sha256(output/n) for n in ('config.json','source_records.json','source_cases.json')},
        'generation_tasks':generation,'selection_tasks':stages,'candidate_units':units,
        'candidate_unit_order':'public task_id lexicographic; first bounded disjoint units; no outcome selection',
        'utility_policy':rules,'new_limits':limits,'inherited_tokens':inherited,'inherited_http':http,'cumulative_limits':ledger['limits'],
        'source_train_tokens':summary['known_total_tokens'],
        'source_train_runtime_tokens':sum(a['usage']['total_tokens'] for t in summary['cases'] for r in t['requests'] if r['stage']=='runtime' for a in r.get('http_attempts',[])),
        'status':'prepared_not_executed','new_model_http':0,'no_val_or_test_selection':True}
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
    if 'recovery' in plan:
        recovered,ledger,_=recovery_inputs(plan['recovery']['parent_plan'])
        if recovered!=plan['recovery'] or read(root/'inherited_p2_budget.json')!=ledger:raise ValueError('Recovery provenance changed')
        if plan['inherited_tokens']!=recovered['inherited_tokens'] or plan['inherited_http']!=recovered['inherited_http'] or plan['new_limits']!=recovered['allowance']:
            raise ValueError('Recovery budget authority changed')
        if (root/'budget.json').exists() and any(read(root/'budget.json')['attempts'].get(k)!=v for k,v in ledger['attempts'].items()):
            raise ValueError('Inherited paid attempts changed')
    if 'paid_prefix' in plan:
        prefix,ledger,_=revision_inputs(plan['paid_prefix']['parent_plan'])
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
        if len(tools)!=1 or tools[0]['function']['name']!='submit_capability' or tools[0]['function']['parameters']!=CAPABILITY or params['model']!=old['settings']['source']['actual_model_request_label']:
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
        if identity in attempt_ids or entry['actual_usage']!=attempt['raw_usage'] or entry['accounted_tokens']!=attempt['raw_usage']['total_tokens'] or entry['context']['parent_scope']!='candidate:'+str(i):
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
    plan={**old,'prompt_revision':PROMPT_REVISION,'run_id':'p2_'+uuid4().hex,'code':code,'prepared_at':utc(),'derivation_mode':'explicit_induction_contract_recovery',
        'recovery':recovery,'input_hashes':inputs,'inherited_tokens':recovery['inherited_tokens'],'inherited_http':recovery['inherited_http'],
        'prepared_hashes':{n:sha256(output/n) for n in ('config.json','source_records.json','source_cases.json','inherited_p2_budget.json')}}
    write_json(output/'plan.json',plan);write_json(output/'plan_identity.json',{'plan_hash':digest(plan)})
    return plan


def revision_inputs(plan_path):
    """One explicitly authorized prompt revision; completed negative evidence stays immutable."""
    path=Path(plan_path).resolve();root=path.parent;old=read(path);result=read(root/'result.json');ledger=read(root/'budget.json')
    if old['schema']!=VERSION or 'paid_prefix' in old or digest(old)!=read(root/'plan_identity.json')['plan_hash'] or result['plan_hash']!=digest(old):
        raise ValueError('Original batch identity changed or already revised')
    inputs={str(path):sha256(path)}
    for p,expected in {**old['input_hashes'],**{str(root/n):h for n,h in old['prepared_hashes'].items()}}.items():
        if sha256(Path(p))!=expected:raise ValueError('Original batch source/material changed')
        inputs[p]=expected
    if result['status']!='completed_no_candidate' or result['candidates'] or read(root/'candidate_order.json') or ledger['unknown_billing']:
        raise ValueError('Revision requires a completed negative batch with known billing')
    if ledger['limits']!={**old['new_limits'],'finish_reserve':0} or any(a['status']!='metered' for a in ledger['attempts'].values()):
        raise ValueError('Original allowance or billing changed')
    tokens=sum(a['accounted_tokens'] for a in ledger['attempts'].values());http=len(ledger['attempts']);prior=old.get('recovery',{})
    inherited=old['inherited_tokens']+tokens-prior.get('prior_p2_tokens',0);inherited_http=old['inherited_http']+http-prior.get('prior_p2_http',0)
    if result['cumulative_tokens']!=inherited or result['cumulative_http']!=inherited_http or result['unknown_billing']:
        raise ValueError('Original cumulative cost differs from ledger')
    if tokens>=old['new_limits']['token_limit'] or http>=old['new_limits']['request_limit']:raise ValueError('No unchanged P2 allowance remains')
    for folder in (root/'learning').iterdir():
        for name in ('learning.json','requests.json'):
            p=folder/name
            if p.exists():inputs[str(p)]=sha256(p)
    for name in ('result.json','budget.json','plan_identity.json','candidate_order.json'):inputs[str(root/name)]=sha256(root/name)
    return {'parent_plan':str(path),'parent_run_id':old['run_id'],'experimental_revision':'P2-R2',
        'prior_p2_tokens':tokens,'prior_p2_http':http,'allowance':old['new_limits'],
        'remaining_tokens':old['new_limits']['token_limit']-tokens,'remaining_http':old['new_limits']['request_limit']-http,
        'inherited_tokens':inherited,'inherited_http':inherited_http},ledger,inputs


def revise(plan_path,output):
    prefix,ledger,inputs=revision_inputs(plan_path);old=read(plan_path);root=Path(plan_path).resolve().parent;output=Path(output).resolve()
    code=code_identity()
    if output.exists() or code['tracked_dirty'] or not code['git_sha']:raise ValueError('Revision requires fresh output and committed clean source')
    config=read(root/'config.json');new_config=configuration(config,output)
    expected=deepcopy(config);expected['data_dir']=new_config['data_dir'];expected['experiment']['output_dir']=new_config['experiment']['output_dir']
    if expected!=new_config:raise ValueError('Prompt revision may not change method parameters')
    for name in ('source_records.json','source_cases.json'):write_json(output/name,read(root/name))
    write_json(output/'config.json',new_config);write_json(output/'inherited_p2_budget.json',ledger)
    plan={**{k:v for k,v in old.items() if k!='recovery'},'prompt_revision':PROMPT_REVISION,'run_id':'p2_'+uuid4().hex,'code':code,'prepared_at':utc(),
        'derivation_mode':'user_authorized_P2_R2_from_same_completed_train','paid_prefix':prefix,'input_hashes':inputs,
        'inherited_tokens':prefix['inherited_tokens'],'inherited_http':prefix['inherited_http'],
        'prepared_hashes':{n:sha256(output/n) for n in ('config.json','source_records.json','source_cases.json','inherited_p2_budget.json')}}
    write_json(output/'plan.json',plan);write_json(output/'plan_identity.json',{'plan_hash':digest(plan)})
    return plan


def learning_boundary(system,root,key):
    system.checkpoint=TaskCheckpoint(root/'checkpoint');system.audit_path=str(root/'requests.json')
    system.phase='train';system.budget_scope=key;system._task_start=len(system.usage.events);system._request_start=len(system.requests)
    system._learning_start=system._task_start
    system.prior_usage=read(system.audit_path)['usage'] if Path(system.audit_path).exists() else []
    system.request_attribution={'parent_scope':key,'parent_token_limit':system.config['budget']['train_learning_tokens'],'train_parent':True}


def evaluator_records(summary):
    records={}
    for trace in summary['cases']:
        raw=trace['scoring_audit']['raw_scorer_output'];choices=trace['task']['inputs']['choices']
        choice=next(c for c in choices if c['label']==raw['correct_label'])
        if choice['text']!=raw['correct_text']:raise ValueError('Saved evaluator label/text identity differs')
        records[trace['task']['task_id']]={'correct_choice':deepcopy(choice),'choices':deepcopy(choices)}
    return records


def evaluate_pair(root,task,candidate,matches,config,governor,trial,evaluator,rank,candidate_index):
    pair={};order=('off','on') if (rank+candidate_index)%2==0 else ('on','off')
    for arm in order:
        task_root=root/str(rank)/arm;cfg=deepcopy(config);cfg['data_dir']=str(trial);cfg['llm']['max_retries']=0
        cfg['experiment'].update(runtime_mode='frozen',output_dir=str(task_root))
        target=EmpiricalSystem(cfg,harness=AnswerAdapter('livemath',evaluator),budget_governor=governor,
            bank_view_factory=lambda b:CapabilityView(b,matches,off=arm=='off',candidate=candidate))
        try:
            target.checkpoint=TaskCheckpoint(task_root/'checkpoint');target.audit_path=str(task_root/'requests.json')
            trace_path=task_root/'trace.json';attempt=VERSION+':'+str(candidate_index)+':'+str(rank)+':'+arm
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
            pair[arm]=trace
        finally:target.close()
        if governor.state['unknown_billing']:raise BudgetExhausted('unknown_billing','Selection billing unresolved')
    pair_fact(candidate,task,pair)
    return pair


def execute(plan_path,env_file):
    check(plan_path);plan=read(plan_path);root=Path(plan_path).parent;config=read(root/'config.json')
    recovery=plan.get('recovery',{})
    prefix=plan.get('paid_prefix',recovery)
    if prefix and not (root/'budget.json').exists():write_json(root/'budget.json',read(root/'inherited_p2_budget.json'))
    load_env(env_file);governor=BudgetGovernor(root/'budget.json',**plan['new_limits'],finish_reserve=0)
    result_path=root/'result.json';result=read(result_path) if result_path.exists() else {'status':'running','run_id':plan['run_id'],'plan_hash':digest(plan),'candidates':[],'started_at':utc()}
    if result['plan_hash']!=digest(plan):raise ValueError('Batch identity changed')
    if result['status']!='running':return result
    write_json(result_path,result);progress(root,'started',run_id=plan['run_id'],status='running')
    system=EmpiricalSystem(config,harness=AnswerAdapter('livemath',{}),budget_governor=governor)
    try:
        if governor.state['unknown_billing']:raise BudgetExhausted('unknown_billing','No retry after unresolved billing')
        for row in read(root/'source_cases.json'):system.bank.save_case(PublicTask(**row['task']),row['experience'])
        selection=[t for rows in plan['selection_tasks'].values() for t in rows];selected=[t['physical_key'] for t in selection];candidates=[]
        learning=[]
        for i,unit in enumerate(plan['candidate_units']):
            learning_boundary(system,root/'learning'/str(i),'candidate:'+str(i))
            log=system.learner.learn_capability_batch(unit,selected,recovered_demand=recovery['demands'][i] if recovery else None)
            write_json(root/'learning'/str(i)/'learning.json',log)
            learning.append({**{k:log.get(k) for k in ('decision','stage','reason')},
                'proposal_created':bool(log.get('proposed_candidate')),'source_check_status':log.get('source_check',{}).get('status')})
            result.update(learning_units=learning,natural_candidate_count=sum(r['proposal_created'] for r in learning))
            write_json(result_path,result)
            progress(root,'learning_unit_completed',unit=i,**learning[-1])
            if governor.state['unknown_billing']:raise BudgetExhausted('unknown_billing','Learning billing unresolved')
            if log['decision']=='candidate':candidates.append(log['candidate'])
        candidates=sorted({c['id']:c for c in candidates}.values(),key=lambda c:(-len(c['source_case_ids']),digest(c)))
        result['source_checked_candidate_count']=len(candidates);write_json(result_path,result)
        write_json(root/'candidate_order.json',[{'id':c['id'],'digest':digest(c),'source_coverage':len(c['source_case_ids'])} for c in candidates])
        trial=root/'trial_bank'
        if not trial.exists():system.bank.freeze(trial)
        evaluator=evaluator_records(read(Path(plan['source_run'])/'train/summary.json')) if candidates else {}
        receipts=[]
        for i,candidate in enumerate(candidates):
            folder=root/'candidates'/candidate['id'];preflight=[]
            for rank,row in enumerate(selection):
                task=PublicTask(**row);learning_boundary(system,folder/'routing'/str(rank),'routing:'+candidate['id']+':'+str(rank))
                match=system.learner.match_capability(candidate,task);validate_match(match,candidate['capability'],task)
                preflight.append({'task':row,'stage':'screen' if rank<len(plan['selection_tasks']['screen']) else 'confirmation','rank':rank,'match':match})
                progress(root,'public_match_completed',candidate_id=candidate['id'],task_id=task.task_id,rank=rank,applicable=match['applicable'])
                if governor.state['unknown_billing']:raise BudgetExhausted('unknown_billing','Routing billing unresolved')
            write_json(folder/'preflight.json',preflight);eligible=[r for r in preflight if r['match']['applicable']];pairs={};screen=[]
            if len(eligible)>=plan['utility_policy']['min_precomputed_applicable']:
                for row in eligible:
                    if row['stage']=='confirmation' and not gate(screen,plan['utility_policy']['screen']):break
                    task=PublicTask(**row['task']);match={candidate['capability']['id']:row['match']}
                    progress(root,'pair_started',candidate_id=candidate['id'],task_id=task.task_id,rank=row['rank'])
                    pair=evaluate_pair(folder/'selection',task,candidate,match,config,governor,trial,evaluator,row['rank'],i)
                    pairs[task.physical_key]=pair
                    progress(root,'pair_completed',candidate_id=candidate['id'],task_id=task.task_id,rank=row['rank'],
                        off_correct=pair['off']['score']['hard'],on_correct=pair['on']['score']['hard'])
                    if row['stage']=='screen':screen.append(pair_fact(candidate,task,pair))
            receipt=utility_receipt(candidate,pairs,plan['utility_policy'],preflight,{'code':plan['code'],'plan_hash':digest(plan)})
            write_json(folder/'utility_receipt.json',receipt);receipts.append((candidate,receipt))
            result['candidates']=[{'id':c['id'],'passed':r['passed'],'status':r['status'],'actual_exposed':len(r['paired'])} for c,r in receipts]
        for candidate,receipt in receipts:
            if receipt['passed']:publish(system.bank,candidate,receipt)
        frozen=root/'frozen_bank'
        if frozen.exists():
            result['frozen']=read(frozen/'freeze.json')
            if result['frozen']['source_digest']!=system.bank.digest():raise ValueError('Frozen source changed')
        else:result['frozen']=system.bank.freeze(frozen)
        result['status']='completed_selection' if candidates else 'completed_no_candidate'
        result.update(learning_units=learning,completion_reason='selection_finished' if candidates else 'no_source_checked_candidate',
            natural_candidate_count=sum(r['proposal_created'] for r in learning),source_checked_candidate_count=len(candidates),
            paired_tasks=sum(c['actual_exposed'] for c in result['candidates']))
    except BudgetExhausted as exc:
        result.update(status='stopped_unknown_billing' if governor.state['unknown_billing'] else 'stopped_budget_censored',reason=str(exc),censored=True)
    except Exception as exc:
        result.update(status='stopped_unknown_billing' if governor.state['unknown_billing'] else 'stopped_engineering_error',reason=str(exc))
        if governor.state['unknown_billing']:result['censored']=True
        if not governor.state['unknown_billing']:raise
    finally:
        system.close();state=governor.state;charged=sum(a.get('accounted_tokens',a['reserved_tokens']) for a in state['attempts'].values())
        known=sum(a.get('accounted_tokens',0) for a in state['attempts'].values())
        known-=prefix.get('prior_p2_tokens',0);charged-=prefix.get('prior_p2_tokens',0);http=len(state['attempts'])-prefix.get('prior_p2_http',0)
        result.update(ended_at=utc(),new_tokens=known,new_reserved_or_accounted_tokens=charged,new_http=http,unknown_billing=state['unknown_billing'],
            inherited_tokens=plan['inherited_tokens'],inherited_http=plan['inherited_http'],
            cumulative_tokens=plan['inherited_tokens']+known,cumulative_reserved_or_accounted_tokens=plan['inherited_tokens']+charged,
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
    for mode in ('check','execute'):
        p=sub.add_parser(mode);p.add_argument('--plan',required=True)
        if mode=='execute':p.add_argument('--env-file',required=True)
    args=parser.parse_args()
    result=prepare(args.settings,args.source_run,args.output) if args.mode=='prepare' else recover(args.plan,args.output) if args.mode=='recover' else revise(args.plan,args.output) if args.mode=='revise' else check(args.plan) if args.mode=='check' else execute(args.plan,args.env_file)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
