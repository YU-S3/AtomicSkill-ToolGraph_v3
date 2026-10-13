"""Bounded public self-reports and observable intent; never a proof or tool receipt."""
from copy import deepcopy
import json
import re

from .contracts import object_schema, validate_schema_instance

VERSION='choice-public-procedure.v1'
FINALIZATION_VERSION='choice_bounded_finalization.v1'
TRUNCATION_FINALIZATION_VERSION='choice_truncation_finalization.v2'
KINDS=('check_assumptions','compare_claim_strength','inspect_quantifiers','deterministic_computation')
REQUIRED={
    'check_assumptions':{'multiple_public_claims','stated_assumptions'},
    'compare_claim_strength':{'multiple_public_claims','explicit_strength_comparison'},
    'inspect_quantifiers':{'multiple_public_claims','explicit_quantifiers'},
    'deterministic_computation':{'multiple_public_claims','explicit_finite_computation'}}
REFERENCE=object_schema({'task_path':{'type':'array','maxItems':8},'result_id':{'type':'string'},
    'path':{'type':'array','maxItems':8},'quote':{'type':'string','minLength':1,'maxLength':240}},['quote'])
PUBLIC_STEP=object_schema({'kind':{'enum':list(KINDS)},
    'references':{'type':'array','items':REFERENCE,'minItems':2,'maxItems':4}},['kind','references'])


def features(task):
    choices=task.inputs.get('choices',[])
    text=task.goal+'\n'+'\n'.join(c.get('text','') for c in choices)
    found=set()
    if len(choices)>=2:found.add('multiple_public_claims')
    for name,pattern in (
        ('stated_assumptions',r'\b(assume|assuming|suppose|given|let|under|if)\b'),
        ('explicit_strength_comparison',r'\b(strongest|stronger|weakest|weaker|claim strength)\b'),
        ('explicit_quantifiers',r'\\(?:forall|exists)\b|\b(for all|for every|there exists|every|any|some)\b'),
        ('explicit_finite_computation',r'\b(calculate|compute|enumerate|finite enumeration|numerical value)\b')):
        if re.search(pattern,text,re.I):found.add(name)
    return found


def intents(task):
    # A requested formal proof cannot be discharged by advisory comparison.
    if re.search(r'^\s*(prove\b|give (?:a |the )?proof\b|provide (?:a |the )?proof\b)',task.goal,re.I):return set()
    visible=features(task)
    return {kind for kind,needed in REQUIRED.items() if needed<=visible}


def validate_process(step,task,context,results,used,index,request_id):
    text=step.get('public_process')
    steps=step.get('public_steps',[])
    if text is None and not steps:return None
    if not isinstance(text,str) or not text.strip() or len(text)>512:raise ValueError('Invalid public_process')
    validated=[]
    for item in steps:
        validate_schema_instance(item,PUBLIC_STEP)
        if item['kind'] not in intents(task):raise ValueError('Public operation intent is absent from this task')
        resolved=[]
        for ref in item['references']:
            validate_schema_instance(ref,REFERENCE)
            if 'task_path' in ref:
                if set(ref)!={'task_path','quote'}:raise ValueError('Ambiguous public task reference')
                path=ref['task_path'];value={'goal':task.goal,'inputs':task.inputs}
                if not path or path[0] not in {'goal','inputs'}:raise ValueError('Unknown public task field')
                try:
                    for part in path:
                        if type(part) not in {str,int} or type(part)==int and part<0:raise ValueError('Invalid task path')
                        value=value[part]
                except (KeyError,IndexError,TypeError) as exc:raise ValueError('Unknown public task reference') from exc
            else:
                if set(ref)!={'result_id','path','quote'} or ref['result_id'] not in set(results)&set(used):
                    raise ValueError('Public process needs an actually used result')
                value=context.resolve_public({'result_id':ref['result_id'],'path':ref['path']})
            visible=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,allow_nan=False)
            if ref['quote'] not in visible:raise ValueError('Public process quote is absent from its source')
            resolved.append(deepcopy(ref))
        if len({json.dumps({k:v for k,v in r.items() if k!='quote'},sort_keys=True) for r in resolved})<2:
            raise ValueError('Public comparison needs distinct references')
        if item['kind']=='deterministic_computation' and not any('result_id' in r and
                context.results[r['result_id']].get('status')=='ok' for r in resolved):
            raise ValueError('Self-report cannot invent a computation')
        validated.append({'kind':item['kind'],'references':resolved})
    from datetime import datetime,timezone
    return {'public_process':text,'public_steps':validated,'kind':'self_reported_process',
        'verification':'source_compatible' if validated else 'unstructured_self_report',
        'mathematically_proved':False,'decision_index':index,'request_id':request_id,
        'observed_at':datetime.now(timezone.utc).isoformat(),'used_results':sorted(used),
        'result_ids':sorted({r['result_id'] for p in validated for r in p['references'] if 'result_id' in r}),
        'version':VERSION}
