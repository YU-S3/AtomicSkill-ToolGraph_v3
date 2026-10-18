"""Exact paired arithmetic, immutable evidence and train-only policy ranking."""
from copy import deepcopy
from .contracts import digest
from .math_effect_contract import VERSION, EDIT, require, validate_candidate, candidate


def pairs(off, on):
    if len(off)!=len(on) or len({r['task']['physical_key'] for r in off})!=len(off):
        raise ValueError('Missing/duplicate paired tasks')
    rows=[]
    for a,b in zip(off,on):
        if (a['task']!=b['task'] or a['public_input_hash']!=b['public_input_hash'] or
                a['runtime_contract_hash']!=b['runtime_contract_hash']):raise ValueError('Off/on input or protocol differs')
        if set(a.get('response_models',[]))!=set(b.get('response_models',[])):
            raise ValueError('Off/on actual response model differs')
        for r in (a,b):
            if r['trace_hash']!=digest({k:v for k,v in r.items() if k!='trace_hash'}):raise ValueError('Changed task receipt')
            if not r['request_ids'] or len(r['request_ids'])!=r['http'] or len(r['raw_usage'])!=r['http']:
                raise ValueError('Missing actual physical response/usage')
            if any(u is None for u in r['raw_usage']):raise ValueError('Unqualified usage')
        x,y=a['score']['hard'],b['score']['hard']
        rows.append({'physical_key':a['task']['physical_key'],'off':deepcopy(a),'on':deepcopy(b),
            'outcome':'win' if not x and y else 'loss' if x and not y else 'tie_correct' if x else 'tie_wrong'})
    return rows


def receipt(off, on, candidates, source_hash, evaluation_set, policy):
    rows=pairs(off,on)
    count=lambda key:sum(r['outcome']==key for r in rows)
    empty=lambda arm:sum(not r[arm]['valid'] for r in rows)
    result={'schema':'math.effect-utility.v1','method_version':VERSION,
        'candidate_ids':[c['id'] for c in candidates],'candidate_hashes':[digest(c) for c in candidates],
        'source_manifest_hash':source_hash,'evaluation_set':evaluation_set,'rows':rows,'policy':deepcopy(policy),
        'evaluated_task_keys':[r['physical_key'] for r in rows],
        'off_on_contract_equal_except_skill':True,'wins':count('win'),'losses':count('loss'),
        'ties_correct':count('tie_correct'),'ties_wrong':count('tie_wrong'),
        'empty_off':empty('off'),'empty_on':empty('on'),
        'tokens_off':sum(r['off']['tokens'] for r in rows),'tokens_on':sum(r['on']['tokens'] for r in rows),
        'http_off':sum(r['off']['http'] for r in rows),'http_on':sum(r['on']['http'] for r in rows),
        'injections':sum(bool(r['on']['selection']['selected']) for r in rows),
        'usage_qualified':'observed_model_usage'}
    result['net_correct']=result['wins']-result['losses']
    result['qualified']=(len(rows)==policy['tasks'] and result['net_correct']>=policy['min_net'] and
        result['losses']<=policy['max_losses'] and result['empty_on']<=result['empty_off'] and result['injections']>0)
    result['receipt_hash']=digest(result);return result


def verify_receipt(value):
    from .math_effect_contract import R2
    if value.get('method_version')==R2:return verify_receipt_r2(value)
    try:
        if value['receipt_hash']!=digest({k:v for k,v in value.items() if k!='receipt_hash'}):return False
        if value['method_version']!=VERSION or value['off_on_contract_equal_except_skill'] is not True:return False
        rows=pairs([r['off'] for r in value['rows']],[r['on'] for r in value['rows']])
        if rows!=value['rows'] or value['wins']!=sum(r['outcome']=='win' for r in rows) or value['losses']!=sum(r['outcome']=='loss' for r in rows):return False
        n={'edit_feedback':12,'protected_gate':20,'val':17,'test':100}[value['evaluation_set']]
        p=value['policy']
        for key,outcome in [('ties_correct','tie_correct'),('ties_wrong','tie_wrong')]:
            if value[key]!=sum(r['outcome']==outcome for r in rows):return False
        for arm in ('off','on'):
            for key in ('tokens','http'):
                if value[key+'_'+arm]!=sum(r[arm][key] for r in rows):return False
        if value['injections']!=sum(bool(r['on']['selection']['selected']) for r in rows):return False
        if value['evaluated_task_keys']!=[r['physical_key'] for r in rows]:return False
        if any(r['on']['selection']['selected'] and not set(r['on']['selection']['selected'])<=set(value['candidate_ids']) for r in rows):return False
        return (len(rows)==n and p['tasks']==n and p['min_net']==2 and p['max_losses']==1
            and value['net_correct']==value['wins']-value['losses'] and
            value['empty_on']==sum(not r['on']['valid'] for r in rows) and
            value['empty_off']==sum(not r['off']['valid'] for r in rows) and
            value['qualified']==(value['net_correct']>=2 and value['losses']<=1 and value['empty_on']<=value['empty_off'] and value['injections']>0))
    except (KeyError,TypeError,ValueError):return False


def rank(r):
    from .math_effect_contract import R2
    if r.get('method_version')==R2:
        x=r['groups']['intervened']
        return (-x['net_correct'],x['losses'],-r['net_correct'],r['tokens_on']+r['tokens_off'],r['candidate_ids'])
    return (-r['net_correct'],r['losses'],r['tokens_on']+r['tokens_off'],r['http_on'],r['candidate_ids'])


def apply_edits(value, parents, sources, evidence, tokenizer_path):
    require(value,EDIT)
    if value['decision']=='no_change':return []
    if not value.get('edits'):raise ValueError('edit decision requires edits')
    children=[];used=set()
    for edit in value['edits']:
        pid=edit['parent_id']
        if pid not in parents or pid in used:raise ValueError('Unknown or duplicate editor parent')
        used.add(pid);parent=parents[pid];v=deepcopy(parent['candidate'])
        for op in edit['operations']:
            field=op['field'];kind=op['op']
            if field=='stop_condition':
                if kind!='replace' or 'index' in op or 'value' not in op:raise ValueError('stop_condition supports replacement only')
                v[field]=op['value'];continue
            index=op.get('index');steps=v['steps']
            if type(index) is not int or index>len(steps) or (kind!='add' and index==len(steps)):
                raise ValueError('Step edit index out of range')
            if kind=='delete':
                if 'value' in op:raise ValueError('Delete has no value')
                steps.pop(index)
            elif 'value' not in op:raise ValueError('Add/replace requires value')
            elif kind=='add':steps.insert(index,op['value'])
            else:steps[index]=op['value']
        v['edit_rationale']=edit['rationale']
        validate_candidate(v,sources,tokenizer_path=tokenizer_path)
        from .math_effect_contract import card
        if card(v)==card(parent['candidate']):raise ValueError('Unchanged edit is not a new candidate')
        children.append(candidate(v,sources,evidence,parent=parent,delta=edit['operations']))
    return children


def main_wire_hash(trace):
    from .math_effect_source import read
    paths=[p for p in trace['request_files'] if p.endswith('/main/request.json')]
    if len(paths)!=1:raise ValueError('Missing unique actual main payload')
    request=read(paths[0])
    if digest(request)!=trace['request_files'][paths[0]]:raise ValueError('Actual request changed')
    result=digest(request['payload'])
    if trace.get('main_wire_hash',result)!=result:raise ValueError('Trace main payload hash differs')
    return result


def pairs_r2(off,on):
    from .math_effect_selection import stronger_meta
    from .contracts import PublicTask
    rows=pairs(off,on)
    for row in rows:
        a,b=row['off'],row['on'];x,y=main_wire_hash(a),main_wire_hash(b)
        row.update(on_selected_ids=list(b['selection']['selected']),off_wire_hash=x,on_wire_hash=y,
            intervention_applied=x!=y,meta_option_present=stronger_meta(PublicTask(**a['task']))['has_stronger_meta'],
            off_correct=a['score']['hard'],on_correct=b['score']['hard'],
            attribution='policy_intervention_observation' if x!=y else 'same_policy_sampling_difference')
    return rows


def statistics_r2(rows):
    def group(values):
        wins=sum(r['outcome']=='win' for r in values);losses=sum(r['outcome']=='loss' for r in values)
        return {'tasks':len(values),'wins':wins,'losses':losses,'net_correct':wins-losses,
            **{a+'_correct':sum(r[a]['score']['hard'] for r in values) for a in ('off','on')},
            **{'empty_'+a:sum(not r[a]['valid'] for r in values) for a in ('off','on')}}
    groups={'all':group(rows),'intervened':group([r for r in rows if r['intervention_applied']]),
        'not_intervened':group([r for r in rows if not r['intervention_applied']]),
        'meta':group([r for r in rows if r['meta_option_present']]),
        'non_meta':group([r for r in rows if not r['meta_option_present']])}
    for intervention in (True,False):
        for meta in (True,False):
            groups[('intervened' if intervention else 'not_intervened')+('_meta' if meta else '_non_meta')]=group(
                [r for r in rows if r['intervention_applied']==intervention and r['meta_option_present']==meta])
    return groups


def receipt_r2(off,on,candidates,source_hash,evaluation_set,policy=None):
    from .math_effect_contract import R2
    return build_receipt_r2(pairs_r2(off,on),[c['id'] for c in candidates],[digest(c) for c in candidates],source_hash,evaluation_set)


def build_receipt_r2(rows,ids,hashes,source_hash,evaluation_set):
    from .math_effect_contract import R2
    groups=statistics_r2(rows);total=groups['all']
    value={'schema':'math.effect-utility.v2','method_version':R2,'statistics_version':'actual-wire-intervention.v1',
        'candidate_ids':ids,'candidate_hashes':hashes,'source_manifest_hash':source_hash,
        'evaluation_set':evaluation_set,'rows':rows,'groups':groups,'evaluated_task_keys':[r['physical_key'] for r in rows],
        'off_on_contract_equal_except_skill':True,'wins':total['wins'],'losses':total['losses'],
        'net_correct':total['net_correct'],'empty_off':total['empty_off'],'empty_on':total['empty_on'],
        'tokens_off':sum(r['off']['tokens'] for r in rows),'tokens_on':sum(r['on']['tokens'] for r in rows),
        'http_off':sum(r['off']['http'] for r in rows),'http_on':sum(r['on']['http'] for r in rows),
        'injections':sum(bool(r['on_selected_ids']) for r in rows),'qualified':False,
        'observed_gain':groups['intervened']['net_correct']>0,
        'evaluation_status':'experimental_not_claimed_effective','usage_qualified':'observed_model_usage'}
    value['receipt_hash']=digest(value);return value


def verify_receipt_r2(value):
    try:
        rows=pairs_r2([r['off'] for r in value['rows']],[r['on'] for r in value['rows']])
        expected=build_receipt_r2(rows,value['candidate_ids'],value['candidate_hashes'],value['source_manifest_hash'],value['evaluation_set'])
        return (value==expected and len(value['candidate_ids'])==len(value['candidate_hashes']) and
            all(set(r['on_selected_ids'])<=set(value['candidate_ids']) for r in rows))
    except (KeyError,TypeError,ValueError,OSError):return False


def apply_edit_r2(edit,parents,sources,evidence,tokenizer_path):
    from . import math_effect_contract as c
    c.require(edit,c.EDIT_ITEM_R2)
    if edit['parent_id'] not in parents:
        raise c.ContractError([c.error('unknown_parent','$.parent_id','shown actual candidate ID',edit['parent_id'])])
    parent=parents[edit['parent_id']];v=deepcopy(edit['replacement'])
    c.validate_candidate_r2(v,sources,tokenizer_path=tokenizer_path)
    if c.card(v)==c.card(parent['candidate']):return None
    delta={key:{'before':parent['candidate'].get(key),'after':v.get(key)} for key in v if v.get(key)!=parent['candidate'].get(key)}
    return c.candidate(v,sources,evidence,parent=parent,delta={'diff':delta,'rationale':edit['rationale']},version=c.R2)
