"""Zero HTTP revalidation of paid v1 responses under the explicit P2-R3 contract."""
import argparse
import json
from pathlib import Path

from . import run_choice_capability_batch as batch
from ..empirical import capability_batch as protocol
from ..empirical.contracts import digest
from ..empirical.model_view import expand_material, reference_exact_span_quotes

SCHEMA='choice-capability-batch.protocol-replay.v1'
LEGACY='choice-capability-batch.v1'
LEGACY_MATERIAL='choice-capability-batch.material.v1'


def saved_response(request,ledger,unit,rows,model):
    calls=request['response']['tool_calls'];attempts=request['http_attempts']
    if len(calls)!=1 or len(attempts)!=1:raise ValueError('Revalidation requires one original metered ToolCall')
    call=calls[0];attempt=attempts[0];entry=ledger['attempts'][attempt['request_id']]
    name='submit_capability' if request['purpose']=='capability_induction' else 'submit_learning'
    schema=protocol.schema_for(protocol.CAPABILITY if name=='submit_capability' else protocol.PROPOSAL,LEGACY)
    wire=attempt['final_payload_audit'];params=wire['serialized_parameters'];actual=attempt['public_response']['choices']
    if call['name']!=name or len(actual)!=1 or actual[0]['finish_reason']!='tool_calls' or attempt['outcome']!='success':
        raise ValueError('Original native response or finish reason changed')
    raw=actual[0]['message']['tool_calls']
    if len(raw)!=1 or raw[0]['function']['name']!=name or json.loads(raw[0]['function']['arguments'])!=call['arguments']:
        raise ValueError('Parsed saved arguments differ from the provider response')
    if wire['tools'][0]['function']['parameters']!=schema or len(wire['tools'])!=1 or wire['tools'][0]['function']['name']!=name:
        raise ValueError('Original field schema changed')
    if params['model']!=model or params['max_tokens']!=2048 or params['thinking']!={'type':'disabled'} or wire['tool_choice']!={'type':'function','function':{'name':name}}:
        raise ValueError('Original learning inference parameters changed')
    if entry['status']!='metered' or entry['actual_usage']!=attempt['raw_usage'] or entry['accounted_tokens']!=attempt['raw_usage']['total_tokens'] or entry['context']['parent_scope']!='candidate:'+str(unit):
        raise ValueError('Original physical request cost or attribution changed')
    if request['repair']!=attempt['repair_in_progress'] or request['logical_decision_id']!=entry['context']['logical_decision_id']:
        raise ValueError('Original repair or logical decision identity changed')
    body=expand_material(json.loads(wire['messages'][1]['content']))
    if name=='submit_capability':
        expected={'public_cases':[{k:r[k] for k in ('task','task_hash','spans')} for r in rows],'learning_material_version':LEGACY_MATERIAL}
    else:
        capability=protocol.validate_capability(body['capability']['definition'],rows,method_version=LEGACY)
        expected={'capability':capability,'train_cases':rows,'learning_material_version':LEGACY_MATERIAL}
    if body not in (expected,reference_exact_span_quotes(expected)):raise ValueError('Saved request was not issued for the exact source material')
    return {'arguments':call['arguments'],'physical_request_id':attempt['request_id'],'logical_decision_id':request['logical_decision_id'],
        'repair_index':request['repair'],'raw_usage':attempt['raw_usage'],'original_model':params['model'],
        'original_payload_fingerprint':attempt['payload_fingerprint'],'original_schema':schema,
        'revalidation_only':True,'new_model_http':0}


def replay(plan_path,output):
    root=Path(plan_path).resolve().parent;old=batch.read(plan_path);output=Path(output).resolve();code=batch.code_identity()
    if output.exists() or code['tracked_dirty'] or not code['git_sha']:raise ValueError('Protocol replay requires fresh output and committed clean source')
    if old['schema']!=LEGACY or old.get('prompt_revision')!='procedural-actions.r2':raise ValueError('Only the original R2 admission contract is a replay source')
    prefix,ledger,inputs=batch.revision_inputs(plan_path,continuation=True,source_version=LEGACY)
    inherited=batch.read(root/'inherited_p2_budget.json')
    if any(ledger['attempts'].get(k)!=v for k,v in inherited['attempts'].items()):raise ValueError('Inherited paid attempts changed')
    rows=protocol.records(batch.read(root/'source_cases.json'))
    if rows!=batch.read(root/'source_records.json'):raise ValueError('Saved public source records changed')
    by_key={r['task']['physical_key']:r for r in rows};selected=[t['physical_key'] for stage in old['selection_tasks'].values() for t in stage]
    units=[]
    for i,keys in enumerate(old['candidate_units']):
        if len(set(keys))!=len(keys) or set(keys)&set(selected):raise ValueError('Source physical identity or selection independence changed')
        source=[by_key[k] for k in keys];folder=root/'learning'/str(i)
        for file in (folder/'checkpoint').rglob('*'):
            if file.is_file():inputs[str(file)]=batch.sha256(file)
        audit=batch.read(folder/'requests.json')['requests'];log=batch.read(folder/'learning.json')
        expected_unit=digest([LEGACY,source,sorted(selected)])
        if log['unit_id']!=expected_unit or log['decision']!='rejected':raise ValueError('Original unit or terminal decision changed')
        paid=[q for q in audit if q.get('http_attempts')]
        demand=[q for q in paid if q['purpose']=='capability_induction'];proposals=[q for q in paid if q['purpose']=='guidance_learning']
        if len(demand)!=1 or demand[0]['repair']!=0 or any(q['purpose'] not in {'capability_induction','guidance_learning'} for q in paid):
            raise ValueError('Protocol replay cannot substitute completed grounding or selection')
        if sorted(q['repair'] for q in proposals)!=list(range(len(proposals))) or len(proposals)>2:raise ValueError('Saved repair count changed')
        model=old['settings']['source']['actual_model_request_label']
        parsed=[saved_response(q,ledger,i,source,model) for q in sorted(proposals,key=lambda q:q['repair'])]
        unit={'original_unit_id':expected_unit,'new_unit_id':digest([protocol.VERSION,source,sorted(selected)]),
            'source_case_ids':keys,'original_decision':log['decision'],'original_reason':log.get('reason'),
            'demand_response':saved_response(demand[0],ledger,i,source,model),'proposal_responses':parsed,'candidate':None,
            'independent_content_check':'not_run','utility_check':'not_run','published':False}
        try:
            unit['capability']=protocol.validate_capability(unit['demand_response']['arguments'],source)
            if unit['capability'] is None:unit['status']='no_change'
            elif proposals:
                # The already-paid final attempt is revalidated; never fall back by utility or feedback.
                unit['candidate']=protocol.validate_candidate(parsed[-1]['arguments'],unit['capability'],source)
                unit['card_characters']=len(json.dumps(protocol.render(unit['candidate']),ensure_ascii=False))+2 if unit['candidate'] else 0
                unit['status']='ready_for_independent_content_check' if unit['candidate'] else 'no_change'
            else:unit['status']='ready_for_guidance_generation'
        except ValueError as exc:
            unit.update(status='revalidation_rejected',reason=str(exc))
        units.append(unit)
    receipt={'schema':SCHEMA,'method_version':protocol.VERSION,'material_version':protocol.MATERIAL_VERSION,'prompt_revision':protocol.PROMPT_REVISION,
        'code':code,'created_at':batch.utc(),'source_method_version':LEGACY,'parent_plan':str(Path(plan_path).resolve()),
        'parent_run_id':old['run_id'],'status':'completed_offline_revalidation','new_model_http':0,'new_tokens':0,
        'source_kind':'saved paid v1 outputs; not responses generated under the new prompts',
        'paid_prefix':{**prefix,'experimental_revision':'P2-R3','derivation_mode':'offline_protocol_revalidation'},
        'candidate_units':old['candidate_units'],'selection_tasks':old['selection_tasks'],'utility_policy':old['utility_policy'],
        'input_hashes':inputs,'units':units,'published_assets':0,'val_test_started':False}
    batch.write_json(output/'protocol_replay.json',receipt);batch.write_json(output/'receipt_identity.json',{'receipt_hash':digest(receipt)})
    batch.write_json(output/'inherited_p2_budget.json',ledger)
    return receipt


def check(receipt_path):
    receipt=batch.read(receipt_path);root=Path(receipt_path).parent
    if receipt['schema']!=SCHEMA or receipt['method_version']!=protocol.VERSION or receipt['code']!=batch.code_identity() or digest(receipt)!=batch.read(root/'receipt_identity.json')['receipt_hash']:
        raise ValueError('Revalidation identity or source code changed')
    for path,expected in receipt['input_hashes'].items():
        if batch.sha256(Path(path))!=expected:raise ValueError('Saved paid source changed: '+path)
    prefix,ledger,_=batch.revision_inputs(receipt['parent_plan'],continuation=True,source_version=LEGACY)
    if ledger!=batch.read(root/'inherited_p2_budget.json') or {**prefix,'experimental_revision':'P2-R3','derivation_mode':'offline_protocol_revalidation'}!=receipt['paid_prefix']:
        raise ValueError('Revalidation cannot refill or replace the inherited allowance')
    return {'status':'passed','new_model_http':0,'new_tokens':0,'units':[r['status'] for r in receipt['units']]}


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='mode',required=True)
    p=commands.add_parser('replay');p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    p=commands.add_parser('check');p.add_argument('--receipt',required=True)
    args=parser.parse_args();result=replay(args.plan,args.output) if args.mode=='replay' else check(args.receipt)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
