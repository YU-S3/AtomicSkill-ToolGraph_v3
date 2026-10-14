"""P4 item-level admission and Train-trace materials; no historical utility is imported."""
from copy import deepcopy
import json
from pathlib import Path

from . import math_capability as m
from .contracts import digest,object_schema,validate_schema_instance


def procedure_identity(candidate):
    def normalized(value):
        return [normalized(v) for v in value] if isinstance(value,list) else ' '.join(value.casefold().split())
    return digest({k:normalized(candidate[k]) for k in ('operation','guidance','applicability','prerequisites','stop_condition')})


def review_batch(value,rows,*,parents=None):
    if isinstance(value,dict) and value.get('decision')=='no_change':return [],[]
    schema=m.batch_schema(8 if parents is not None else 4,m.P4)
    schema['properties']['candidates']['items']={}
    validate_schema_instance(value,schema)
    if set(value)!={'decision','candidates'}:raise ValueError('Missing P4 candidate list')
    accepted=[];receipts=[];seen=set()
    for index,raw in enumerate(value['candidates']):
        normalized={k:v for k,v in raw.items() if k!='parent_ids'} if isinstance(raw,dict) and parents is None else raw
        receipt={'index':index,'raw_hash':digest(raw),'normalized_hash':digest(normalized),
            'discarded_fields':['parent_ids'] if normalized!=raw else [],'source_case_ids':[]}
        try:
            candidate=m.validate_candidate(normalized,rows,parents=parents,version=m.P4)
            if candidate is None:raise ValueError('Use top-level no_change for abstention')
            if parents is not None and len(candidate['source_case_ids'])>4:raise ValueError('Consolidation cites at most four physical sources')
            signature=procedure_identity(candidate)
            if signature in seen:raise ValueError('Duplicate procedure steps and prerequisites')
            seen.add(signature);accepted.append(candidate)
            receipt.update(status='accepted',reason='source_checked',candidate_id=candidate['id'],source_case_ids=candidate['source_case_ids'])
        except (ValueError,TypeError,KeyError) as exc:
            receipt.update(status='rejected',reason=str(exc))
        receipts.append(receipt)
    return accepted,receipts


def review_checks(value,candidates):
    schema=object_schema({'checks':{'type':'array','maxItems':len(candidates),'items':{}}},['checks'])
    validate_schema_instance(value,schema)
    accepted={};receipts=[];expected={digest(c):c for c in candidates}
    for h,c in expected.items():
        matches=[v for v in value['checks'] if isinstance(v,dict) and v.get('candidate_hash')==h]
        receipt={'candidate_hash':h,'candidate_id':c['id'],'source_case_ids':c['source_case_ids']}
        try:
            if len(matches)!=1:raise ValueError('Content check must identify this exact candidate once')
            item=matches[0]
            if not m.validate_check(item,c):raise ValueError('Content classification: '+item['status'])
            accepted[c['id']]=item
            receipt.update(status='accepted',reason=item['status'])
        except (ValueError,TypeError,KeyError) as exc:receipt.update(status='rejected',reason=str(exc))
        receipts.append(receipt)
    for item in value['checks']:
        if not isinstance(item,dict) or not isinstance(item.get('candidate_hash'),str) or item['candidate_hash'] not in expected:
            receipts.append({'status':'rejected','reason':'Unknown exact candidate hash','raw_hash':digest(item)})
    return accepted,receipts


def learn(learner,rows,*,parents=None):
    s=learner.system
    if s.readonly or s.phase!='train':raise ValueError('P4 trace learning requires writable Train mode')
    key='math_learning_'+digest([m.P4,rows,parents])
    if key in s.checkpoint.state:return s.checkpoint.state[key]
    consolidation=parents is not None
    if consolidation:
        if any(not m.checked_candidate(c) or c['method_version']!=m.P4 for c in parents.values()):
            raise ValueError('P4 consolidation requires checked P4 parents only')
        material={'parent_candidates':[{k:c[k] for k in ('id','category','operation','guidance','applicability',
            'prerequisites','stop_condition','references')} for c in parents.values()], 'operations':m.OPERATIONS}
        prompt=m.PROMPT.replace('up to four','up to eight')+(
            ' This is the sole Train-only consolidation. Preserve exact source references and real parent_ids. '
            'Deduplicate concrete steps with their prerequisites, then cover distinct operations. '
            'Cite two to four original physical sources per output. No Val/Test feedback is supplied.')
    else:
        material={'train_cases':rows,'public_task_analysis':[m.public_structure(m.PublicTask(**r['task'])) for r in rows],
            'operations':m.OPERATIONS}
        prompt=m.PROMPT+' Do not emit parent_ids in unit learning. '
    prompt+=(' Keep the short generation targets above. Host tolerance only: guidance 640, each prerequisite 128, '
        'stop_condition 192, rendered card 1536 characters. Outcome feedback is not a proof or a diagnosed failure cause. '
        'Use observed computations only within their actual finite scope; no fabricated executions.')
    log={'method_version':m.P4,'source_case_ids':[r['task']['physical_key'] for r in rows],
        'material_hash':digest(material),'decision':'no_change','stage':'consolidation' if consolidation else 'learning',
        'candidates':[],'proposed_candidates':[],'proposal_receipts':[],'checks':[],'check_receipts':[]}
    def validate(value):
        try:
            _,receipts=review_batch(value,rows,parents=parents)
            errors=[r for r in receipts if r['status']=='rejected']
            if errors:raise ValueError('; '.join(f"candidates[{r['index']}]: {r['reason']}" for r in errors))
        except ValueError as exc:
            exc.repair_material={**material,'previous_proposal':deepcopy(value),'field_errors':str(exc),
                'learning_material_version':m.P4_MATERIAL,
                'repair_rule':'Preserve candidate list positions and valid items. Fix only rejected members; no extra paid repair.'}
            raise
    value=None
    try:
        value=learner._receive(key+'_proposal','extractor',prompt,material,'submit_learning',
            m.batch_schema(8 if consolidation else 4,m.P4),material_version=m.P4_MATERIAL,
            decision_purpose='math_consolidation' if consolidation else 'guidance_learning',validator=validate,repair_limit=1)
        log['proposal']=value
    except ValueError as exc:
        if not getattr(exc,'model_authored',False):raise
        log.update(decision='rejected',reason=str(exc))
    # Only real, correctly named ToolCalls are eligible. Keep valid slots even if the
    # single repair fails or rewrites them; cached paid responses follow the same path.
    owner=key+'_proposal_'+m.P4_MATERIAL.replace('.','_').replace('-','_')
    proposals=[]
    audit=Path(s.audit_path)
    requests={r['id']:r for r in json.loads(audit.read_text())['requests']} if audit.exists() else {}
    requests.update({r['id']:r for r in s.requests if r.get('owner_state_version')==owner})
    for request in requests.values():
        calls=request.get('response',{}).get('tool_calls',[])
        if (request.get('owner_state_version')==owner and len(calls)==1 and calls[0]['name']=='submit_learning'
                and request['response'].get('finish_reason')!='length'):
            proposals.append((request['id'],request.get('normalized_response',{}).get('arguments',calls[0]['arguments'])))
    if value is not None and not proposals:raise ValueError('P4 proposal lacks its original physical request')
    slots={};procedures=set()
    for request_id,proposal in proposals:
        try:valid,receipts=review_batch(proposal,rows,parents=parents)
        except ValueError as exc:
            log['proposal_receipts'].append({'request_id':request_id,'status':'rejected','reason':str(exc)});continue
        by_id={c['id']:c for c in valid}
        for receipt in receipts:
            item={**receipt,'request_id':request_id};index=item['index']
            if item['status']=='accepted':
                candidate=by_id[item['candidate_id']]
                if index in slots:item['retention']='original_valid_slot_preserved'
                elif procedure_identity(candidate) in procedures:item.update(status='rejected',reason='Duplicate retained procedure')
                else:slots[index]=candidate;procedures.add(procedure_identity(candidate));item['retention']='retained'
            log['proposal_receipts'].append(item)
    proposed=[slots[k] for k in sorted(slots)];log['proposed_candidates']=proposed
    groups=[[c] for c in proposed] if consolidation else [proposed] if proposed else []
    for index,group in enumerate(groups):
        schema=object_schema({'checks':{'type':'array','minItems':len(group),'maxItems':len(group),
            'items':m.check_schema(m.P4)}},['checks'])
        common={r['task']['physical_key']:r for c in group for r in c['source_records']}
        check_material={'train_cases':list(common.values()),'candidates':[
            {'candidate_hash':digest(c),'proposal':c['proposal']} for c in group]}
        try:
            answer=learner._receive(key+'_check_'+str(index),'extractor',m.CHECK_PROMPT,check_material,
                'submit_math_checks',schema,material_version=m.P4_MATERIAL,decision_purpose='guidance_grounding',
                validator=lambda v:review_checks(v,group),repair_limit=0)
            accepted,receipts=review_checks(answer,group)
            log['checks'].extend(answer['checks']);log['check_receipts'].extend(receipts)
            for c in group:
                if c['id'] in accepted:
                    extra={'parent_candidates':{k:parents[k] for k in c['parent_ids']}} if consolidation else {}
                    log['candidates'].append({**c,'source_check':accepted[c['id']],**extra})
        except ValueError as exc:
            if not getattr(exc,'model_authored',False):raise
            log['check_receipts'].extend({'candidate_id':c['id'],'status':'rejected','reason':str(exc)} for c in group)
    if proposed or log['proposal_receipts']:log['decision']='candidates' if log['candidates'] else 'rejected'
    log['receipt_hash']=digest(log);s.checkpoint.advance(s.checkpoint.state['stage'],**{key:log})
    return log


def trace_records(cases,traces):
    rows=m.records(cases,span_factory=m.math_spans)
    if set(traces)!={r['task']['physical_key'] for r in rows}:raise ValueError('P4 requires all original Train traces')
    audit=[]
    for row,case in zip(rows,cases):
        trace=traces[row['task']['physical_key']];execution=trace['execution'];experience=case['experience']
        if (trace['task']!=row['task'] or trace['score']!=experience['score'] or
                execution.get('prediction')!=experience.get('submission')):raise ValueError('Train trace/Bank feedback identity differs')
        row['raw_trace_hash']=digest(trace)
        row['observation']['training_feedback']={'score':deepcopy(trace['score']),'qualification':'observed_feedback_not_mathematical_proof'}
        process=execution.get('public_process_evidence')
        row['observation']['public_process']=deepcopy(process) if isinstance(process,dict) and process.get('verification')=='source_compatible' else None
        computations=[]
        store=trace.get('result_store',{}).get('program_results',{})
        for call in execution.get('temporary_executions',[])+execution.get('attempts',[]):
            result=store.get(call['result_id'])
            computations.append({'execution':deepcopy(call),'result':deepcopy(result),
                'qualification':'actual_finite_execution_not_general_proof'})
            audit.append({'physical_key':row['task']['physical_key'],'execution_id':call['id'],
                'status':call['status'],'output_contract_status':call.get('output_contract_status'),
                'outputs_consumed':call.get('outputs_consumed',False),'has_recorded_output':bool(result and result.get('outputs'))})
        row['observation']['computations']=computations
    return rows,{'executions':audit,'program_learning':'guidance_only; no Program promotion from a numerical example or unconsumed output',
        'composite_support':False}
