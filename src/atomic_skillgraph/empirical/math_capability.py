"""Explicit LiveMath P3 mathematical procedures; old capability policies stay frozen."""
from copy import deepcopy
import json
import re

from . import MATH_CAPABILITY_VERSION as VERSION, MATH_CAPABILITY_MATERIAL_VERSION as MATERIAL_VERSION
from .contracts import PublicTask,digest,object_schema,validate_schema_instance
from .capability_batch import TEXT,REFERENCE as REF,records,source_refs,spans,public_identity,abstain_match,MATCH,field_errors

CATEGORIES=('quantifiers_implications','convergence_asymptotics','counterexample_construction',
    'existence_uniqueness','finite_computation')
OPERATIONS={'quantifiers_implications':['quantifier_instantiation','negation_scope','implication_direction'],
    'convergence_asymptotics':['limit_order','uniformity_conditions','asymptotic_scale'],
    'counterexample_construction':['admissible_counterexample','boundary_case'],
    'existence_uniqueness':['existence_construction','uniqueness_comparison'],
    'finite_computation':['invariant_check','finite_enumeration']}
POLICY={'screen':{'min_exposed':3,'min_net_correct':0,'max_regressions':1,'require_empty_on_not_increase':True},
    'confirmation':{'min_exposed':1,'min_net_correct':1,'max_regressions':1,'require_empty_on_not_increase':True},
    'promotion':{'min_exposed':8,'min_net_correct':2,'max_regressions':1,'require_empty_on_not_increase':True}}
PROPOSAL=object_schema({'decision':{'enum':['candidate','no_change']},'category':{'enum':list(CATEGORIES)},
    'operation':{'enum':[op for ops in OPERATIONS.values() for op in ops]},
    'guidance':TEXT(512),'applicability':TEXT(128),
    'prerequisites':{'type':'array','minItems':1,'maxItems':3,'items':TEXT(80)},'stop_condition':TEXT(128),
    'references':{'type':'array','minItems':2,'maxItems':8,'items':REF},
    'parent_ids':{'type':'array','minItems':1,'maxItems':15,'uniqueItems':True,'items':TEXT(80)}},['decision'])
CHECK=object_schema({'candidate_hash':TEXT(64),'status':{'enum':['procedural_only','disputed','domain_fact','unsupported']},
    'proposal_quote':TEXT(512),'references':{'type':'array','minItems':2,'maxItems':8,'items':REF},'reason':TEXT(512)},
    ['candidate_hash','status','proposal_quote','references','reason'])
PROMPT=('Extract up to four distinct reusable mathematical procedures from the supplied actual Train records, or choose no_change. '
    'Use a listed category. Specify applicability, prerequisite checks, concrete guidance steps, and a stopping condition. '
    'Guidance <=512 Unicode characters (aim 280); applicability <=128 (aim 60); prerequisites <=3 short checks; '
    'stop_condition <=128 (aim 70); complete rendered card <=1024. Avoid generic advice alone. '
    'Useful operations include instantiating or negating quantifiers, checking limit order and uniformity hypotheses, '
    'constructing an admissible boundary example, separating existence from uniqueness, or bounded deterministic computation. '
    'Finite examples do not prove universal claims. Do not copy a source theorem conclusion, select an answer label, '
    'infer a true failure cause from a wrong/truncated answer, or equate a stronger/weaker option with truth. '
    'Cite exact supplied case_id/span_id pairs from at least two distinct physical Train cases. '
    'Compressed p2_text_ref identity values are resolved only against this request. '
    'Use decision=no_change when evidence is insufficient; unused accompanying fields do not publish a candidate.')


def configure(base):
    from .public_process import TRUNCATION_FINALIZATION_VERSION
    from . import POLICY_DEFAULTS
    config=deepcopy(base)
    for section,defaults in POLICY_DEFAULTS.items():config.setdefault(section,{}).update(defaults)
    config['learning'].update(capability_material_version=MATERIAL_VERSION,capability_payload_max_bytes=65536,guidance_repair_limit=1)
    config['llm']['runtime'].update(max_completion_tokens=65536,max_total_tokens_per_task=128000,
        request_timeout_seconds=600,protocol={'thinking_type':'enabled'},reasoning_effort='high')
    config['llm']['extractor']['max_total_tokens_per_task']=1000000
    for purpose,cap in [('guidance_learning',4096),('guidance_grounding',4096),('capability_routing',2048),('math_consolidation',8192)]:
        config['llm']['purpose_overrides'][purpose]={'protocol':{'thinking_type':'disabled'},'max_completion_tokens':cap,'max_retries':0}
    config['llm']['purpose_overrides']['finish_only']={'protocol':{'thinking_type':'enabled'},'reasoning_effort':'low','max_completion_tokens':16384,'max_retries':0}
    config.setdefault('budget',{'finish_reserve':0}).update(token_limit=45000000,request_limit=9165,eval_task_tokens=128000)
    config['runtime']['choice_bounded_finalization']={'enabled':True,'version':TRUNCATION_FINALIZATION_VERSION}
    return config


def validate_runtime_contract(config):
    from .public_process import TRUNCATION_FINALIZATION_VERSION
    if config['harness'].get('adapter')!='livemath':raise ValueError('P3 is restricted to the LiveMath adapter')
    llm=config['llm'];runtime=llm['runtime'];finish=llm['purpose_overrides'].get('finish_only',{})
    if (runtime.get('max_completion_tokens')!=65536 or runtime.get('request_timeout_seconds')!=600 or
        runtime.get('protocol')!={'thinking_type':'enabled'} or runtime.get('reasoning_effort')!='high' or
        runtime.get('max_total_tokens_per_task')!=128000 or config['budget'].get('eval_task_tokens')!=128000):
        raise ValueError('P3 requires the frozen 64K/high/600s Runtime and 128K parent contract')
    if finish!={'protocol':{'thinking_type':'enabled'},'reasoning_effort':'low','max_completion_tokens':16384,'max_retries':0}:
        raise ValueError('P3 requires one 16K enabled/low finish without HTTP retry')
    if config['runtime']['choice_bounded_finalization']!={'enabled':True,'version':TRUNCATION_FINALIZATION_VERSION}:
        raise ValueError('P3 requires its truncation-only finalizer')


def render(candidate):
    return {'skill_id':candidate['id'],'category':candidate['category'],'operation':candidate['operation'],'applicability':candidate['applicability'],
        'prerequisites':candidate['prerequisites'],'guidance':candidate['guidance'],'stop_condition':candidate['stop_condition'],
        'qualification':'procedure_not_proof'}


def validate_candidate(value,rows,*,parents=None):
    if isinstance(value,dict) and value.get('decision')=='no_change':return None
    errors=field_errors(value,PROPOSAL)
    if errors:raise ValueError('; '.join(errors))
    validate_schema_instance(value,PROPOSAL)
    if value['decision']=='no_change':return None
    required={'decision','category','operation','guidance','applicability','prerequisites','stop_condition','references'}
    if set(value)!=(required|({'parent_ids'} if parents is not None else set())):raise ValueError('Incomplete P3 procedure')
    if value['operation'] not in OPERATIONS[value['category']]:raise ValueError('Operation does not belong to category')
    covered=source_refs(value['references'],rows,span_factory=math_spans)
    if parents is not None:
        ids=value['parent_ids']
        if not ids or any(k not in parents for k in ids):raise ValueError('Unknown consolidation parent')
        permitted={k for key in ids for k in parents[key]['source_case_ids']}
        if not set(covered)<=permitted:raise ValueError('Consolidation invented cross-parent evidence')
    text=' '.join([value['guidance'],value['applicability'],*value['prerequisites'],value['stop_condition']])
    if re.search(r'\b(?:choose|select|answer(?: is)?)\s+(?:option\s+)?[A-E]\b',text,re.I):
        raise ValueError('A mathematical procedure cannot select a fixed option label')
    content={k:deepcopy(v) for k,v in value.items() if k!='decision'}
    candidate={'id':'skill_'+digest([VERSION,content]),'method_version':VERSION,'state':'candidate',
        'knowledge_type':'procedural_advisory','execution_intent':'guidance_only','goal':value['applicability'],
        **content,'proposal':deepcopy(value),'source_case_ids':covered,
        'source_records':deepcopy([r for r in rows if r['task']['physical_key'] in covered])}
    scope={k:candidate[k] for k in ('category','operation','applicability','prerequisites','stop_condition')}
    candidate['capability']={'id':'cap_'+digest([VERSION,scope]),**scope}
    if len(json.dumps(render(candidate),ensure_ascii=False))+2>1024:raise ValueError('P3 complete card exceeds 1024 Unicode characters')
    return candidate


def validate_check(value,candidate):
    validate_schema_instance(value,CHECK)
    if value['candidate_hash']!=digest(candidate):raise ValueError('P3 content check identity changed')
    if value['proposal_quote'] not in candidate['guidance']:raise ValueError('Unlocated P3 procedure quote')
    source_refs(value['references'],candidate['source_records'],span_factory=math_spans)
    return value['status'] in {'procedural_only','disputed'}


def public_structure(task):
    """Deterministic public demand features with exact locations, never answer predictions."""
    patterns={
        'quantifiers_implications':r'\b(?:every|all|some|each|implies|equivalent|necessarily|sufficient|necessary)\b|\\(?:forall|exists|implies|iff)\b',
        'convergence_asymptotics':r'\b(?:converg\w*|asymptot\w*|uniform\w*|limit|limsup|liminf|bounded|stabili\w*)\b',
        'counterexample_construction':r'\b(?:counterexample|fails?|false|not necessarily|cannot|boundary|exception)\b',
        'existence_uniqueness':r'\b(?:exist\w*|unique\w*|there is|there are|solution|fixed point)\b',
        'finite_computation':r'\b(?:compute|calculate|finite|integer|polynomial|matrix|determinant|rank|dimension|cardinality)\b'}
    evidence=[]
    for category,pattern in patterns.items():
        found=[]
        for path,text in [(['goal'],task.goal),*[(['inputs','choices',i,'text'],c['text']) for i,c in enumerate(task.inputs['choices'])]]:
            match=re.search(pattern,text,re.I)
            if match:found.append({'category':category,'task_path':path,'start':match.start(),'end':match.end(),'quote':match.group()})
        evidence.extend(found[:2])
    return {'version':VERSION,'task_hash':public_identity(task),'categories':sorted({e['category'] for e in evidence}),
        'evidence':evidence,'qualification':'public_demand_indicators_not_mathematical_truth'}


def math_spans(task):
    result=spans(task)
    for feature in public_structure(task)['evidence']:
        path=feature['task_path'];text={'goal':task.goal,'inputs':task.inputs}
        for key in path:text=text[key]
        start=max(0,feature['start']-48);end=min(len(text),start+160)
        span={'task_path':path,'start':start,'end':end,'quote':text[start:end]}
        row={'span_id':'span_'+digest(span),**span}
        if row not in result:result.append(row)
    return result


def validate_route(value,candidate,task):
    validate_schema_instance(value,MATCH)
    if value['task_hash']!=public_identity(task) or value['capability_id']!=candidate['capability']['id']:
        raise ValueError('P3 public routing identity changed')
    available={s['span_id']:s for s in math_spans(task)}
    if any(s not in available for s in value['span_ids']):raise ValueError('Invented P3 routing span')
    if value['applicable'] and len({tuple(available[s]['task_path']) for s in value['span_ids']})<2:
        raise ValueError('P3 routing requires two distinct public fields')
    return value['applicable']


def match(learner,candidate,task):
    s=learner.system;key='math_match_'+digest([candidate['capability'],public_identity(task)])
    if key+'_rejection' in s.checkpoint.state:return abstain_match(candidate['capability'],task)
    try:
        return learner._receive(key,'extractor','Check whether this conditional mathematical procedure applies to this public task. '
            'Cite exact supplied span IDs from two distinct public fields. Abstain if premises or scope cannot be established. '
            'Do not solve, predict an answer or use gold.',
            {'capability':candidate['capability'],'task':{'goal':task.goal,'inputs':task.inputs},'spans':math_spans(task),'task_hash':public_identity(task)},
            'submit_capability_match',MATCH,material_version=MATERIAL_VERSION,decision_purpose='capability_routing',
            validator=lambda v:validate_route(v,candidate,task),repair_limit=0)
    except ValueError as exc:
        if not getattr(exc,'model_authored',False):raise
        s.checkpoint.advance(s.checkpoint.state['stage'],**{key+'_rejection':{'reason':str(exc),'logical_decision_id':exc.logical_decision_id}})
        return abstain_match(candidate['capability'],task)


def select(assets,task,matches,*,candidate=False):
    selected=[];audit=[]
    for asset in sorted(assets,key=lambda a:a.get('publication_rank',0)):
        route=matches.get(asset['capability']['id']);reason='unverified'
        if (candidate or checked_asset(asset)) and route:
            reason='applicable' if validate_route(route,asset,task) else 'not_applicable'
            if reason=='applicable':
                if selected:reason='higher_ranked_skill_selected'
                else:selected.append(asset);reason='selected'
        audit.append({'asset_id':asset['id'],'reason':reason})
    return {'selected':selected,'audit':audit,'injected_ids':[a['id'] for a in selected],
        'injected_chars':sum(len(json.dumps(render(a),ensure_ascii=False))+2 for a in selected),'selection_version':VERSION}


class MathView:
    def __init__(self,bank,matches,*,off=False,candidate=None):
        if not bank.readonly:raise ValueError('P3 pair requires a readonly Bank')
        self.bank,self.matches,self.off,self.candidate=bank,matches,off,candidate
    def select_guidance(self,task,policy):
        assets=[] if self.off else [self.candidate] if self.candidate else self.bank.all('skill')
        return select(assets,task,self.matches,candidate=self.candidate is not None)
    def __getattr__(self,name):return getattr(self.bank,name)


def utility_receipt(candidate,pairs,preflight,policy,identity):
    from .capability_batch import pair_fact,gate
    if any(r['task']['split']!='val' for r in preflight):raise ValueError('P3 utility requires genuine Val records')
    if policy!=POLICY:raise ValueError('P3 frozen selection gates changed')
    facts=[pair_fact(candidate,PublicTask(**r['task']),pairs[r['task']['physical_key']]) for r in preflight if r['task']['physical_key'] in pairs]
    screen=[f for f in facts if next(r['rank'] for r in preflight if r['task']['physical_key']==f['physical_key'])<6]
    confirm=[f for f in facts if f not in screen]
    initial=gate(screen,policy['screen']);confirmed=gate(confirm,policy['confirmation']) and gate(facts,policy['promotion'])
    expected={r['task']['physical_key'] for r in preflight if r['match']['applicable']}
    if set(pairs)!=expected:raise ValueError('Incomplete P3 paired selection')
    if any(r['task']['physical_key'] in candidate['source_case_ids'] for r in preflight):raise ValueError('P3 learning overlaps selection')
    if any(not validate_route(r['match'],candidate,PublicTask(**r['task'])) for r in preflight if r['match']['applicable']):
        raise ValueError('Invalid P3 public route')
    totals={arm:{'correct':sum(f[arm]['correct'] for f in facts),'empty':sum(f[arm]['empty'] for f in facts),
        'total_tokens':sum(f[arm]['usage']['total_tokens'] for f in facts),'http':sum(f[arm]['physical_http'] for f in facts)} for arm in ('off','on')}
    value={'schema':VERSION+'.val-utility.v1','candidate_hash':digest(candidate),'preflight':deepcopy(preflight),'paired':facts,
        'policy':deepcopy(policy),'identity':identity,'screen_passed':initial,'passed':initial and confirmed and len(preflight)==17,
        'totals':totals,'qualification':'development_selection_not_generalization'}
    return {**value,'receipt_hash':digest(value)}


def checked_candidate(candidate):
    try:
        parents=candidate.get('parent_candidates')
        if parents is not None and any(c.get('parent_candidates') is not None or not checked_candidate(c) for c in parents.values()):return False
        rebuilt=validate_candidate(candidate['proposal'],candidate['source_records'],parents=parents)
        expected={k:v for k,v in candidate.items() if k not in {'source_check','parent_candidates'}}
        return expected==rebuilt and validate_check(candidate['source_check'],rebuilt)
    except (ValueError,KeyError,TypeError):return False


def checked_asset(asset):
    from .capability_batch import gate
    try:
        if asset['method_version']!=VERSION or asset['state']!='usable':return False
        c=asset['candidate'];receipt=asset['utility_receipt']
        if not checked_candidate(c):return False
        if receipt['schema']!=VERSION+'.val-utility.v1' or receipt['candidate_hash']!=digest(c) or not receipt['passed']:return False
        if receipt['receipt_hash']!=digest({k:v for k,v in receipt.items() if k!='receipt_hash'}):return False
        rows=receipt['preflight'];facts=receipt['paired'];policy=receipt['policy']
        if policy!=POLICY or receipt['identity']['method_version']!=VERSION or receipt['identity']['public_val_hash']!=digest([r['task'] for r in rows]):return False
        if len(rows)!=17 or [r['rank'] for r in rows]!=list(range(17)) or any(r['task']['split']!='val' for r in rows):return False
        if len({r['task']['physical_key'] for r in rows})!=17 or any(r['task']['physical_key'] in c['source_case_ids'] for r in rows):return False
        expected_keys={r['task']['physical_key'] for r in rows if validate_route(r['match'],c,PublicTask(**r['task']))}
        if {f['physical_key'] for f in facts}!=expected_keys or len(facts)!=len(expected_keys):return False
        for f in facts:
            if not f['exposed'] or c['id'] in f['off']['injected_ids'] or c['id'] not in f['on']['injected_ids']:return False
            if any(f['off'][k]!=f['on'][k] for k in ('bank_hash','prompt_without_candidate_hash','effective_settings','wire_parameters','wire_tools_hash')):return False
        first={r['task']['physical_key'] for r in rows[:6]}
        screen=[f for f in facts if f['physical_key'] in first];confirm=[f for f in facts if f['physical_key'] not in first]
        return (gate(screen,policy['screen']) and gate(confirm,policy['confirmation']) and gate(facts,policy['promotion']) and
            asset=={**c,'state':'usable','candidate':c,'utility_receipt':receipt,'publication_rank':asset['publication_rank']})
    except (ValueError,KeyError,TypeError):return False


def batch_schema(limit):
    return object_schema({'decision':{'enum':['candidates','no_change']},
        'candidates':{'type':'array','minItems':1,'maxItems':limit,'items':PROPOSAL}},['decision'])


def validate_batch(value,rows,*,parents=None):
    schema=batch_schema(8 if parents is not None else 4)
    schema['properties']['candidates']['items']={'type':'object'}
    validate_schema_instance(value,schema)
    if value['decision']=='no_change':return []
    if set(value)!={'decision','candidates'}:raise ValueError('Missing P3 candidate list')
    result=[];errors=[]
    for i,proposal in enumerate(value['candidates']):
        try:
            candidate=validate_candidate(proposal,rows,parents=parents)
            if candidate is None:raise ValueError('Use top-level no_change, not an empty candidate')
            if parents is not None and len(candidate['source_case_ids'])>4:raise ValueError('A consolidated procedure cites at most four physical sources')
            result.append(candidate)
        except ValueError as exc:errors.append(f'candidates[{i}]: {exc}')
    if len({c['operation'] for c in result})!=len(result):errors.append('Duplicate operation within this batch')
    if errors:raise ValueError('; '.join(errors))
    return result


CHECK_PROMPT=('Check each exact mathematical procedure against its cited actual Train sources. '
    'procedural_only means concrete conditional operations with a stop condition, not a theorem conclusion. '
    'disputed means real sources but uncertain mathematical applicability requiring controlled utility validation. '
    'domain_fact copies an answer or theorem conclusion; unsupported invents evidence, facts or failure causes. '
    'Return one check per candidate hash, quote exact guidance and cite two exact physical source references. '
    'Keep each reason <=180 characters; do not rewrite candidates.')


def learn(learner,rows,*,parents=None):
    s=learner.system
    if s.readonly or s.phase!='train' or s.config['experiment'].get('choice_method_version')!=VERSION:
        raise ValueError('P3 learning requires its own writable Train version')
    key='math_learning_'+digest([VERSION,rows,parents])
    if key in s.checkpoint.state:return s.checkpoint.state[key]
    consolidation=parents is not None
    if consolidation:
        # One Train-only integration pass: the model sees every checked procedure and exact
        # source references. Original records are checked again for each revised output.
        material={'parent_candidates':[{k:c[k] for k in ('id','category','operation','guidance','applicability',
            'prerequisites','stop_condition','references')} for c in parents.values()],
            'operations':OPERATIONS}
        prompt=PROMPT.replace('up to four','up to eight')+(
            ' This is the sole cross-group consolidation. Deduplicate operations and merge or revise only using '
            'the supplied Train-derived parents. Set parent_ids. Retain exact original references; '
            'cite two to four distinct physical cases per output. No Val or Test input is supplied.')
    else:
        material={'train_cases':rows,'public_task_analysis':[public_structure(PublicTask(**r['task'])) for r in rows],
            'operations':OPERATIONS}
        prompt=PROMPT
    log={'method_version':VERSION,'source_case_ids':[r['task']['physical_key'] for r in rows],
        'decision':'no_change','candidates':[],'stage':'consolidation' if consolidation else 'learning',
        'material_hash':digest(material),'checks':[]}
    def validator(value):
        try:return validate_batch(value,rows,parents=parents)
        except ValueError as exc:
            exc.repair_material={**material,'previous_proposal':deepcopy(value),'field_errors':str(exc),
                'learning_material_version':MATERIAL_VERSION}
            raise
    try:
        value=learner._receive(key+'_proposal','extractor',prompt,material,'submit_learning',batch_schema(8 if consolidation else 4),
            material_version=MATERIAL_VERSION,decision_purpose='math_consolidation' if consolidation else 'guidance_learning',
            validator=validator,repair_limit=1)
        log['proposal']=value;proposed=validate_batch(value,rows,parents=parents)
        log['proposed_candidates']=proposed
        groups=[[c] for c in proposed] if consolidation else [proposed] if proposed else []
        for index,group in enumerate(groups):
            check_schema=object_schema({'checks':{'type':'array','minItems':len(group),'maxItems':len(group),'items':CHECK}},['checks'])
            def validate_checks(answer):
                validate_schema_instance(answer,check_schema)
                if {v['candidate_hash'] for v in answer['checks']}!={digest(c) for c in group}:
                    raise ValueError('P3 content checks must cover each exact candidate once')
                by_hash={digest(c):c for c in group}
                for v in answer['checks']:validate_check(v,by_hash[v['candidate_hash']])
            common={r['task']['physical_key']:r for c in group for r in c['source_records']}
            check_material={'train_cases':list(common.values()),'candidates':[
                {'candidate_hash':digest(c),'proposal':c['proposal']} for c in group]}
            try:
                answer=learner._receive(key+'_check_'+str(index),'extractor',CHECK_PROMPT,check_material,
                    'submit_math_checks',check_schema,material_version=MATERIAL_VERSION,
                    decision_purpose='guidance_grounding',validator=validate_checks,repair_limit=0)
            except ValueError as exc:
                if not getattr(exc,'model_authored',False):raise
                log['checks'].append({'candidate_ids':[c['id'] for c in group],'rejected':str(exc)});continue
            log['checks'].extend(answer['checks'])
            for c in group:
                checked=next(v for v in answer['checks'] if v['candidate_hash']==digest(c))
                if validate_check(checked,c):
                    extra={'parent_candidates':{k:parents[k] for k in c['parent_ids']}} if consolidation else {}
                    log['candidates'].append({**c,'source_check':checked,**extra})
        if proposed:log['decision']='candidates' if log['candidates'] else 'rejected'
    except ValueError as exc:
        if not getattr(exc,'model_authored',False):raise
        log.update(decision='rejected',reason=str(exc))
    log['receipt_hash']=digest(log);s.checkpoint.advance(s.checkpoint.state['stage'],**{key:log})
    return log
