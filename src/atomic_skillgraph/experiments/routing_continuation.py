"""Explicit child run for a metered model-routing rejection after completed learning."""
from copy import deepcopy
from pathlib import Path
import json
import shutil
from uuid import uuid4

from ..empirical.budget_governor import billing_blocked
from ..empirical.capability_batch import abstain_match, public_identity, validate_match, valid_receipt
from ..empirical.contracts import PublicTask, digest
from ..empirical.model_view import expand_material
from . import run_choice_capability_batch as batch
from .canonical_manifest import sha256
from .formal_log import tree_identity, utc
from .run_empirical import code_identity, write_json

VERSION='choice-capability-routing-continuation.v1'


def replay_match(request, candidate, task):
    """Revalidate the saved answer without replacing it or issuing a repair."""
    body=expand_material(json.loads(request['messages'][1]['content']))
    if (request['purpose']!='capability_routing' or body['capability']!=candidate['capability'] or
        body['task']!={'goal':task.goal,'inputs':task.inputs} or body['task_hash']!=public_identity(task)):
        raise ValueError('Inherited routing material identity changed')
    response=request['response'];calls=response['tool_calls']
    try:
        if response['finish_reason']=='length':raise ValueError('Truncated choice guidance response')
        if len(calls)!=1 or calls[0]['name']!='submit_capability_match':raise ValueError('Expected one submit_capability_match ToolCall')
        match=calls[0]['arguments'];validate_match(match,candidate['capability'],task)
        return {'match':match}
    except ValueError as exc:
        return {'match':abstain_match(candidate['capability'],task),'rejection':{
            'status':'model_response_rejected','reason':str(exc),'host_action':'withhold_asset',
            'logical_decision_id':request['logical_decision_id'],'original_response_hash':digest(response)}}


def prepare(parent, output):
    from .run_derived_choice import VERSION as DERIVED_VERSION, check
    parent,root=Path(parent).resolve(),Path(output).resolve();code=code_identity()
    if root.exists() or code['tracked_dirty'] or not code['git_sha']:
        raise ValueError('Routing continuation requires a fresh output and committed clean source')
    old=batch.read(parent/'p2/plan.json');record=batch.read(parent/'derivation_manifest.json')
    result=batch.read(parent/'p2/result.json');ledger=batch.read(parent/'budget.json')
    phase=batch.read(parent/'p2/budget.json')['attempts']
    if (record['version']!=DERIVED_VERSION or old['code']!=record['code'] or
        batch.read(parent/'p2/plan_identity.json')['plan_hash']!=digest(old)):
        raise ValueError('Parent derivation identity changed')
    for path,expected in {**old['input_hashes'],**{str(parent/'p2'/p):v for p,v in old['prepared_hashes'].items()}}.items():
        if sha256(Path(path))!=expected:raise ValueError('Parent source/material changed: '+path)
    if (result.get('plan_hash')!=digest(old) or result.get('run_id')!=old['run_id'] or
        result['status']!='stopped_engineering_error' or result['completed_units']!=len(old['candidate_units']) or
        (parent/'train/frozen_bank').exists() or (parent/'val').exists() or (parent/'test').exists()):
        raise ValueError('Continuation requires completed learning and an unpublished routing stop')
    if billing_blocked(ledger) or not phase or any(ledger['attempts'].get(k)!=v or v['status']!='metered' for k,v in phase.items()):
        raise ValueError('Routing continuation requires a fully metered P2 prefix')
    audited=set()
    for path in (parent/'p2').rglob('requests.json'):
        for request in batch.read(path)['requests']:
            for attempt in request.get('http_attempts',[]):
                key=attempt['request_id'];row=phase.get(key)
                if not row or row['actual_usage']!=attempt['raw_usage']:
                    raise ValueError('Inherited request audit differs from physical billing')
                audited.add(key)
    if audited!=set(phase):raise ValueError('Missing physical request audit in inherited P2')
    spent=sum(a['accounted_tokens'] for a in phase.values())
    if spent>=old['new_limits']['token_limit'] or len(phase)>=old['new_limits']['request_limit']:
        raise ValueError('No original P2 allowance remains')
    logs=[batch.read(parent/'p2/learning'/str(i)/'learning.json') for i in range(len(old['candidate_units']))]
    if any(log.get('source_case_ids')!=unit or log.get('receipt_hash')!=digest({k:v for k,v in log.items() if k!='receipt_hash'})
           for log,unit in zip(logs,old['candidate_units'])):
        raise ValueError('Completed learning identity changed')
    candidates=sorted({log['candidate']['id']:log['candidate'] for log in logs if log['decision']=='candidate'}.values(),
        key=lambda c:(-len(c['source_case_ids']),digest(c)))[:old['settings']['candidate']['max_candidates_to_selection']]
    selection=[row for rows in old['selection_tasks'].values() for row in rows]
    imported={'version':VERSION,'parent_root':str(parent),'parent_code':old['code'],
        'parent_plan_hash':digest(old),'completed_candidates':{},'matches':{},'prior_tokens':spent,'prior_http':len(phase)}
    rejected=[]
    for candidate in candidates:
        folder=parent/'p2/candidates'/candidate['id'];receipt_path=folder/'utility_receipt.json'
        if receipt_path.exists():
            receipt=batch.read(receipt_path)
            pairs={}
            for row in receipt['preflight']:
                pair_root=folder/'selection'/str(row['rank'])
                if pair_root.exists():pairs[row['task']['physical_key']]={arm:batch.read(pair_root/arm/'trace.json') for arm in ('off','on')}
            rebuilt=batch.utility_receipt(candidate,pairs,old['utility_policy'],receipt['preflight'],receipt['identity'])
            if not valid_receipt(candidate,receipt) or receipt!=rebuilt:raise ValueError('Inherited paired evidence changed')
            imported['completed_candidates'][candidate['id']]=str(receipt_path)
            continue
        if (folder/'selection').exists():raise ValueError('Partial paired evaluation requires its own explicit recovery')
        matches={}
        for rank,row in enumerate(selection):
            audit_path=folder/'routing'/str(rank)/'requests.json'
            if not audit_path.exists():continue
            requests=batch.read(audit_path)['requests']
            if len(requests)!=1 or not requests[0].get('response') or not requests[0].get('http_attempts'):
                raise ValueError('Incomplete routing response; do not silently repeat paid requests')
            request=requests[0]
            if any(a['request_id'] not in phase for a in request['http_attempts']):raise ValueError('Unaccounted routing request')
            replay=replay_match(request,candidate,PublicTask(**row))
            if replay.get('rejection'):rejected.append(replay['rejection']['reason'])
            matches[str(rank)]={**replay,'source':str(audit_path),'source_sha256':sha256(audit_path)}
        imported['matches'][candidate['id']]=matches
    if result['reason'] not in rejected:raise ValueError('Parent stop is not a saved model-authored routing rejection')
    inputs={**old['input_hashes'],**{str(p):sha256(p) for p in parent.rglob('*') if p.is_file()}}
    shutil.copytree(parent/'train/bank',root/'train/bank')
    for i in range(len(logs)):
        for name in ('learning.json','requests.json'):
            source=parent/'p2/learning'/str(i)/name
            if source.exists():
                destination=root/'p2/learning'/str(i)/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
    for name in ('source_cases.json','source_records.json'):
        shutil.copy2(parent/'p2'/name,root/'p2'/name)
    for name in ('budget_revision.json','request_bound.json','billing_reservation_recovery.json'):
        if (parent/name).exists():shutil.copy2(parent/name,root/name)
    config=batch.read(parent/'p2/config.json');config['data_dir']=str(root/'train/bank');config['experiment']['output_dir']=str(root/'p2')
    write_json(root/'p2/config.json',config);write_json(root/'budget.json',ledger)
    write_json(root/'p2/campaign_budget_prefix.json',ledger);write_json(root/'p2/inherited_stage_attempts.json',phase)
    charged=sum(a.get('accounted_tokens',a['reserved_tokens']) for a in ledger['attempts'].values())
    known=sum(a.get('accounted_tokens',0) for a in ledger['attempts'].values())
    plan={**old,'run_id':'p2_'+uuid4().hex,'code':code,'prepared_at':utc(),'routing_continuation':imported,
        'input_hashes':inputs,'inherited_tokens':charged,'inherited_known_tokens':known,'inherited_http':len(ledger['attempts']),
        'campaign_budget_prefix':{'tokens':charged,'known_tokens':known,'http':len(ledger['attempts'])},
        'prior_phase_tokens':spent,'prior_phase_http':len(phase),'new_model_http':0,'status':'prepared_not_executed'}
    identity={**record['identity'],'source_commit':code['git_sha'],'source_sha256':code['source_sha256'],
        'continuation_version':VERSION,'parent_run_id':batch.read(parent/'run_manifest.json')['run_id'],'parent_run':str(parent)}
    configs=deepcopy(record['configs'])
    for cfg in configs.values():
        cfg['data_dir']=str(root/'train/frozen_bank')
        cfg['experiment']['output_dir']=str(root/Path(cfg['experiment']['output_dir']).name)
    record={**record,'prepared_at':utc(),'code':code,'identity':identity,'configs':configs,'source_hashes':inputs,
        'initial_bank_hash':tree_identity(root/'train/bank')['sha256'],'routing_continuation':imported}
    write_json(root/'derivation_manifest.json',record);write_json(root/'continuation_manifest.json',imported)
    plan['prepared_hashes']={str(p.relative_to(root/'p2')):sha256(p) for p in (root/'p2').rglob('*') if p.is_file()}
    for name in ('derivation_manifest.json','continuation_manifest.json','request_bound.json','budget_revision.json','billing_reservation_recovery.json'):
        if (root/name).exists():plan['prepared_hashes']['../'+name]=sha256(root/name)
    write_json(root/'p2/plan.json',plan);write_json(root/'p2/plan_identity.json',{'plan_hash':digest(plan)})
    return check(root)
