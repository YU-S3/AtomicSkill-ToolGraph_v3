"""Incremental public evidence for the isolated authored Office Bank V3.1."""
from copy import deepcopy
from decimal import Decimal, InvalidOperation
import re

from . import office_human_policy as v3
from .contracts import digest
from .public_results import TEXT_PAGE, TEXT_CONTEXT, PROGRAM_EXECUTION, FILE_COMPUTE

VERSION = 'office.human.phase.v3.1.20261017'
REF, cards = v3.REF, v3.cards
MAX_OPERANDS = 40
GLOBAL_SUBMISSION_RULE = (
    'Once the requested result is established, immediately use native runtime_step(action="finish", answer=...). '
    'Return only the requested numeric, list, or date payload, without explanation. For a numeric percentage, '
    'preserve the required numeric scale but omit a trailing percent symbol unless the question explicitly '
    'requires that symbol in its template. Preserve sign and decimal precision. Do not restart retrieval '
    'after a complete answer is ready. You may finish without reporting an answer_candidate.')
NOTE = (v3.NOTE + ' Operands accumulate across rounds. Use FYyyyy for fiscal years and CYyyyy for calendar years. '
    'Retrieve only missing observation periods; preserve acquired values. Do not compute a full-range statistic '
    'from a partial series. Correct one saved operand with replaces_source_ref; conflicts never silently overwrite. '
    'Choose an exact scalar result_ref from recent_scalar_result_refs, not a parent object. '
    'answer_candidate.operand_bindings maps executed operand names to exact source refs when ambiguous. '
    'Temporary Python can print {"operand_values":{...},"answer":...} to make executed inputs explicit.')
STATE = deepcopy(v3.STATE)
STATE['properties']['operands']['items']['properties']['replaces_source_ref'] = REF
STATE['properties']['answer_candidate']['properties']['operand_bindings'] = {
    'type':'object', 'additionalProperties':REF}


def number(value):
    try:
        n = Decimal(str(value).replace(',', ''))
        return n.normalize() if n.is_finite() else str(value)
    except InvalidOperation:
        return str(value)


def slot(row):
    return row['name'], row['period'], row['unit']


def operand_identity(row):
    return (*slot(row), digest(row['source_ref']), number(row['value']))


def merge_operands(previous, incoming):
    rows, feedback = deepcopy(previous), []
    for incoming_row in incoming:
        row = deepcopy(incoming_row)
        replaces = row.pop('replaces_source_ref', None)
        targets = [x for x in rows if slot(x)==slot(row) and x['source_ref']==replaces] if replaces else []
        if replaces and len(targets)!=1:
            feedback.append('replacement_requires_one_exact_saved_source_in_the_same_slot')
            continue
        remaining = [x for x in rows if x not in targets]
        if any(operand_identity(x)==operand_identity(row) for x in remaining):
            rows = remaining
            continue
        if len(remaining)>=MAX_OPERANDS:
            feedback.append('operand_capacity_exceeded: preserve saved records; no incoming record added')
            continue
        rows = remaining + [row]
    conflicts=[]
    for key in dict.fromkeys(slot(x) for x in rows):
        group=[x for x in rows if slot(x)==key]
        if len({number(x['value']) for x in group})>1:
            conflicts.append({'name':key[0], 'period':key[1], 'unit':key[2],
                'source_refs':[x['source_ref'] for x in group]})
    return rows, conflicts, feedback


def validate_operand(context, row):
    ref=row['source_ref']
    if context.origin(ref['result_id']) not in {TEXT_PAGE,TEXT_CONTEXT}:
        raise ValueError('operand requires acquired source text, not a search hit/model literal')
    text=context.resolve_public(ref)
    if not isinstance(text,str) or not row['quote'] or row['quote'] not in text:
        raise ValueError('operand quote not found in the referenced text')
    numbers=re.findall(r'(?<![\w.])[+-]?\d+(?:\.\d+)?(?!\w|\.\d)',row['quote'].replace(',',''))
    if not any(number(x)==number(row['value']) for x in numbers):
        raise ValueError('operand value is absent from its quote')
    if not row['name'] or not row['period'] or not row['unit']:
        raise ValueError('operand name, period and unit are required')


def period_coverage(task, state):
    goal=getattr(task,'goal','') if task is not None else ''
    # Require an explicit observation-year basis and a continuous interval. Publication dates,
    # isolated dates and two-year comparisons do not establish a complete annual series.
    matches=list(re.finditer(r'\b(fiscal\s+years?|calendar\s+years?|FY|CY)\s*(?:from\s+)?'
        r'(\d{4})\s*(?:[-–—]|to|through)\s*(?:(?:FY|CY)\s*)?(\d{4})\b',goal,re.I))
    result={'required_periods':[], 'observed_periods':[], 'missing_periods':[],
            'coverage_status':'unknown', 'scope':'observation_periods_only_not_metric_semantics'}
    if len(matches)!=1:return result
    basis,start,end=matches[0].groups();start,end=int(start),int(end)
    if not (1800<=start<=end<=2200) or end-start>=MAX_OPERANDS:return result
    prefix='FY' if basis.lower().startswith(('fiscal','fy')) else 'CY'
    required=[prefix+str(y) for y in range(start,end+1)]
    observed=set()
    conflicted={tuple(c[k] for k in ['name','period','unit']) for c in state.get('operand_conflicts',[])}
    for row in state.get('operands',[]):
        if slot(row) in conflicted:continue
        m=re.fullmatch(r'(FY|CY|fiscal\s+year|calendar\s+year)\s*(\d{4})',row['period'],re.I)
        if m:
            b='FY' if m[1].lower().startswith(('fy','fiscal')) else 'CY'
            observed.add(b+m[2])
    missing=[y for y in required if y not in observed]
    result.update(required_periods=required, observed_periods=[y for y in required if y in observed],
                  missing_periods=missing, coverage_status='partial' if missing else 'complete')
    return result


def scalar_children(value, path):
    if isinstance(value,dict):
        for k,v in value.items():yield from scalar_children(v,path+[k])
    elif isinstance(value,list):
        for i,v in enumerate(value):yield from scalar_children(v,path+[i])
    elif isinstance(value,(str,int,float)) and not isinstance(value,bool) and len(str(value))<=160:
        yield path,value


def recent_scalar_result_refs(context, limit=6):
    if context is None:return []
    result=[]
    for row in reversed(list(context.memory.values())):
        rid=row['result_id'];origin=context.origin(rid)
        if origin not in {PROGRAM_EXECUTION,FILE_COMPUTE}:continue
        base=['outputs'] if origin==PROGRAM_EXECUTION else ['data','computed']
        try:value=context.resolve_public({'result_id':rid,'path':base})
        except (KeyError,TypeError,ValueError):continue
        fields=list(scalar_children(value,base))
        fields.sort(key=lambda pair:(not any(k in str(pair[0][-1]).lower() for k in ['answer','value','result']),len(pair[0])))
        for path,value in fields:
            if path[-1] in {'unit','formula','direction','proof_scope','convention','scope'}:continue
            if 'operand_values' in path or 'operand_refs' in path:continue
            result.append({'result_id':rid,'path':path,'display_value':value,'origin':origin,
                           'proof_scope':'local_computation_only'})
            if len(result)>=min(6,limit):return result
    return result


def executed_operands(context, ref, operation):
    row=next(r for r in context.memory.values() if r['result_id']==ref['result_id'])
    if context.origin(ref['result_id'])==PROGRAM_EXECUTION:
        bound=row.get('arguments',{})
        if bound.get('operation')!=operation:raise ValueError('candidate operation differs from executed operation')
        keys={'ratio':['numerator','denominator'], 'difference':['from_value','to_value']}.get(operation)
        if keys is None:
            keys=['base_value','new_value'] if operation in {'percent_change','log_return','cagr','annual_log_growth'} else ['values']
            if 'other_values' in bound:keys=keys+['other_values']
        relevant={k:bound[k] for k in keys}
    else:
        # A temporary script may nest its calculation. Only explicit operand_values from
        # the actual result parent are used; no arithmetic is inferred from source code.
        relevant={}
        for end in range(len(ref['path'])-1,1,-1):
            parent=context.resolve_public({'result_id':ref['result_id'],'path':ref['path'][:end]})
            if isinstance(parent,dict) and 'operand_values' in parent:
                relevant=parent['operand_values'];break
    def flatten(value,prefix=''):
        if isinstance(value,dict):return {k:v for n,x in value.items() for k,v in flatten(x,n if not prefix else prefix+'.'+n).items()}
        if isinstance(value,list):return {k:v for i,x in enumerate(value) for k,v in flatten(x,prefix+'.'+str(i)).items()}
        return {prefix:value}
    values=flatten(relevant)
    if not values:raise ValueError('no explicit executed operand_values')
    return values


def validate_state(context, reported, previous=None, *, task=None):
    state=deepcopy(previous or {'operands':[], 'answer_candidate':None})
    errors=[];valid=[]
    for row in reported.get('operands',[]):
        try:validate_operand(context,row);valid.append(row)
        except (KeyError,TypeError,ValueError) as exc:errors.append(str(exc))
    state['operands'],state['operand_conflicts'],feedback=merge_operands(state.get('operands',[]),valid)
    errors.extend(feedback)
    candidate=deepcopy(reported.get('answer_candidate',state.get('answer_candidate')))
    if candidate:
        fresh='answer_candidate' in reported
        candidate.update(ready=False,proof_scope='reference_integrity_only')
        try:
            ref=candidate['result_ref']
            if context.origin(ref['result_id']) not in {PROGRAM_EXECUTION,FILE_COMPUTE}:
                raise ValueError('candidate must reference an actual computation result')
            actual=context.resolve_public(ref)
            if isinstance(actual,(dict,list)):
                state['scalar_reference_choices']=[{'result_id':ref['result_id'],'path':p,'display_value':v}
                    for p,v in list(scalar_children(actual,ref['path']))[:6]]
                raise ValueError('non_scalar_result_reference: Select one of the actual scalar child fields.')
            if actual is None or number(actual)!=number(candidate['value']):
                raise ValueError('candidate differs from referenced output')
            if not candidate['output_unit'] or not candidate['operation']:raise ValueError('candidate requires output unit and operation')
            explicit=candidate.get('operand_bindings',{})
            selected=[]
            for name,value in executed_operands(context,ref,candidate['operation']).items():
                group=[x for x in state['operands'] if x['name']==name and number(x['value'])==number(value)]
                if name in explicit:group=[x for x in group if x['source_ref']==explicit[name]]
                if len(group)!=1:raise ValueError('ambiguous_or_missing_operand_binding: '+name)
                row=group[0];validate_operand(context,row)
                conflict=any(slot(row)==tuple(c[k] for k in ['name','period','unit']) for c in state['operand_conflicts'])
                if conflict and not (fresh and name in explicit):raise ValueError('operand_conflict_requires_explicit_source: '+name)
                selected.append(row)
            candidate['operand_refs']=[x['source_ref'] for x in selected]
            # Pin actual sources for revalidation; unrelated later operands cannot invalidate them.
            candidate['operand_bindings']={x['name']:x['source_ref'] for x in selected}
            coverage=period_coverage(task,state)
            if coverage['coverage_status']=='partial':raise ValueError('incomplete_observation_periods')
            candidate['ready']=candidate['covers_task'] is True
        except (KeyError,IndexError,TypeError,ValueError,StopIteration) as exc:errors.append(str(exc))
    state['answer_candidate']=candidate
    state['binding_feedback']=errors
    return state


def acquired(context):
    pages=v3.acquired(context)
    if context:
        for row in context.memory.values():
            rid=row['result_id']
            if context.origin(rid)!=PROGRAM_EXECUTION:continue
            public=context.public_result(context.results[rid],origin=context.origin(rid))
            if public.get('status')!='ok':continue
            for w in (public.get('outputs') or {}).get('windows',[]):
                if isinstance(w,dict) and isinstance(w.get('text'),str) and w['text']:
                    pages.append({'result_id':rid,'text':w['text']})
    return pages


def progress(context, state, remaining=48, *, task=None):
    result=v3.progress(context,state,remaining)
    pages=acquired(context)
    result.update(acquired_pages=len(pages),table_window_available=any(
        len(re.findall(r'^.*\|.*\|.*$',p['text'],re.M))>=2 for p in pages),
        period_coverage=period_coverage(task,state))
    if result['phase']=='locate' and pages:result['phase']='extract'
    if result['period_coverage']['coverage_status']=='partial' and result['phase']=='compute':result['phase']='extract'
    return result


def finish_evidence(*args, **kwargs):
    context=args[0]
    task=kwargs.pop('task',None)
    extra={'recent_scalar_result_refs':recent_scalar_result_refs(context),
           'period_coverage':period_coverage(task,kwargs['state'])}
    payload=kwargs['payload_builder']
    kwargs['payload_builder']=lambda value,cap:payload({**value,**extra},cap)
    result=v3.finish_evidence(*args,**kwargs)
    result['materials'].update(extra)
    result['projection_version']=VERSION
    return result
