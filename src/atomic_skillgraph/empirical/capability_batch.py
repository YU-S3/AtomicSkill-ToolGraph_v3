"""Train-only capability proposals; quoted demands are not proofs or utility receipts."""
from copy import deepcopy
import json
import re

from .contracts import PublicTask, digest, object_schema, validate_schema_instance
from .choice_guidance import valid_choices
from .public_results import public_result

VERSION='choice-capability-batch.v1'
MATERIAL_VERSION='choice-capability-batch.material.v1'
PROMPT_REVISION='procedural-actions.r2'
OPERATIONS=('extract_assumptions','compare_implications','inspect_quantifiers','seek_counterexample','check_finite_invariant')
TEXT=lambda n:{'type':'string','minLength':1,'maxLength':n}
REFERENCE=object_schema({'case_id':TEXT(160),'span_id':TEXT(80)},['case_id','span_id'])
REFERENCES={'type':'array','items':REFERENCE,'minItems':2,'maxItems':4,'uniqueItems':True}
CAPABILITY=object_schema({'decision':{'enum':['no_change','capability']},'required_conditions':TEXT(180),
    'operations':{'type':'array','items':{'enum':list(OPERATIONS)},'minItems':1,'maxItems':3,'uniqueItems':True},
    'references':REFERENCES},['decision'])
PROPOSAL=object_schema({'decision':{'enum':['no_change','candidate']},'guidance':TEXT(180),
    'applicability':TEXT(80),'references':REFERENCES},['decision'])
CHECK=object_schema({'candidate_hash':TEXT(64),'status':{'enum':['procedural_only','domain_fact','unsupported']},
    'proposal_quote':TEXT(240),'references':REFERENCES,'reason':TEXT(240)},['candidate_hash','status','proposal_quote','references'])
MATCH=object_schema({'task_hash':TEXT(64),'capability_id':TEXT(80),'applicable':{'type':'boolean'},
    'span_ids':{'type':'array','items':TEXT(80),'minItems':0,'maxItems':4,'uniqueItems':True}},
    ['task_hash','capability_id','applicable','span_ids'])


def public_identity(task):
    return digest({'goal':task.goal,'inputs':task.inputs})


def spans(task):
    """Exact source slices, preserving LaTeX and Unicode; no paraphrased quotes."""
    rows=[]
    fields=[(['goal'],task.goal), *[(['inputs','choices',i,'text'],c['text']) for i,c in enumerate(task.inputs['choices'])]]
    for path,value in fields:
        for start in dict.fromkeys([0,max(0,len(value)-160)]):
            span={'task_path':path,'start':start,'end':min(len(value),start+160),'quote':value[start:start+160]}
            if span['quote']:rows.append({'span_id':'span_'+digest(span),**span})
    return rows


def records(cases):
    result=[];physical=set()
    for row in cases:
        task=PublicTask(**row['task']);experience=row['experience']
        if task.split!='train' or not valid_choices(task):raise ValueError('Capability sources require public Train choices')
        if task.physical_key in physical:raise ValueError('Repeated physical source is not independent evidence')
        physical.add(task.physical_key)
        status=experience.get('answer_status')
        outcome='no_answer' if status in {'empty','truncated_empty','invalid_format'} or not experience.get('submission') else (
            'correct' if (experience.get('score') or {}).get('hard') is True else 'incorrect')
        result.append({'task':deepcopy(row['task']),'task_hash':public_identity(task),'spans':spans(task),
            'observation':{'submission':experience.get('submission'),'outcome':outcome,
                'answer_status':status,'termination_reason':experience.get('termination_reason'),
                'public_process':deepcopy(experience.get('public_process_evidence')),
                'tools':[{'name':e.get('name'),'arguments':deepcopy(e.get('arguments')),
                    'result':public_result(e.get('result'),e.get('result_protocol'))} for e in experience.get('events',[]) if e.get('state')=='finished']},
            'source_record_hash':digest(row)})
    return result


def source_refs(refs,rows):
    by_id={r['task']['physical_key']:r for r in rows};found=set()
    if len(by_id)!=len(rows) or any(PublicTask(**r['task']).split!='train' or r['task_hash']!=public_identity(PublicTask(**r['task'])) or
        r['spans']!=spans(PublicTask(**r['task'])) for r in rows):raise ValueError('Source identity or exact spans changed')
    aliases={}
    for physical,row in by_id.items():
        for alias in (physical,row['task']['task_id']):
            if alias in aliases and aliases[alias]!=physical:raise ValueError('Ambiguous public source ID')
            aliases[alias]=physical
    for ref in refs:
        physical=aliases.get(ref['case_id'])
        if physical is None or ref['span_id'] not in {s['span_id'] for s in by_id[physical]['spans']}:
            raise ValueError('Unknown exact public source span')
        found.add(physical)
    if len(found)<2:raise ValueError('Procedure requires at least two distinct physical sources')
    return sorted(found)


def no_change(value,rows):
    if set(value)-{'decision','references'}:raise ValueError('no_change cannot supply a capability or candidate')
    if 'references' in value:source_refs(value['references'],rows)
    return None


def abstract_advisory(text,rows):
    # A procedural card cannot carry formulas, numerical answers or label instructions.
    if re.search(r'\d|\\[A-Za-z]|[=⇒]|<answer>|\b(?:prove[sd]? that|mathematically prove[dn]|always (?:true|correct))\b',text,re.I):
        raise ValueError('Specific mathematical conclusions are not procedural advisories')
    labels={c['label'] for r in rows for c in r['task']['inputs']['choices']}
    if any(re.search(r'\b(?:select|choose|answer|option)\s+'+re.escape(label)+r'\b',text,re.I) for label in labels):
        raise ValueError('Advisory must not prescribe a displayed answer label')
    normalized=' '.join(text.casefold().split())
    for row in rows:
        for choice in row['task']['inputs']['choices']:
            conclusion=' '.join(choice['text'].casefold().split())
            if len(conclusion)>=12 and conclusion in normalized:raise ValueError('Choice conclusion copied into a procedural card')


def validate_capability(value,rows):
    validate_schema_instance(value,CAPABILITY)
    if value['decision']=='no_change':
        return no_change(value,rows)
    if set(value)!={'decision','required_conditions','operations','references'}:raise ValueError('Incomplete capability')
    covered=source_refs(value['references'],rows)
    abstract_advisory(value['required_conditions'],rows)
    return {'id':'cap_'+digest(value),'definition':deepcopy(value),'source_case_ids':covered,
        'coverage':len(covered),'source_records_hash':digest(rows),'qualification':'public_task_demand_not_proof'}


def render(asset):
    return {'skill_id':asset['id'],'guidance':asset['guidance'],'applicability':asset['applicability'],
        'qualification':'paired_train_advisory_not_proof'}


def validate_candidate(value,capability,rows):
    validate_schema_instance(value,PROPOSAL)
    if value['decision']=='no_change':
        return no_change(value,rows)
    if set(value)!={'decision','guidance','applicability','references'}:raise ValueError('Incomplete procedural candidate')
    covered=source_refs(value['references'],rows)
    if not set(covered)<=set(capability['source_case_ids']):raise ValueError('Candidate is outside its capability evidence')
    abstract_advisory(value['guidance']+'\n'+value['applicability'],rows)
    candidate={'id':'skill_'+digest([value,capability]),'method_version':VERSION,'state':'candidate',
        'knowledge_type':'procedural_advisory','execution_intent':'guidance_only','goal':capability['definition']['required_conditions'],
        'guidance':value['guidance'],'applicability':value['applicability'],'capability':deepcopy(capability),
        'proposal':deepcopy(value),'source_records':deepcopy(rows),'source_case_ids':covered,'verification_strength':['source_compatible'],
        'mathematically_proved':False}
    if len(json.dumps(render(candidate),ensure_ascii=False))+2>512:raise ValueError('Complete procedural advisory must fit 512 Unicode characters')
    return candidate


def validate_check(value,candidate):
    validate_schema_instance(value,CHECK)
    if value['candidate_hash']!=digest(candidate):raise ValueError('Candidate check identity changed')
    if value['proposal_quote'] not in candidate['guidance']+'\n'+candidate['applicability']:raise ValueError('Unlocated proposal quote')
    source_refs(value['references'],candidate['source_records'])
    return value['status']=='procedural_only'


def validate_match(value,capability,task):
    validate_schema_instance(value,MATCH)
    if value['task_hash']!=public_identity(task) or value['capability_id']!=capability['id']:raise ValueError('Public demand match identity changed')
    available={s['span_id']:s for s in spans(task)}
    if any(s not in available for s in value['span_ids']):raise ValueError('Invented task matching span')
    if value['applicable'] and len({tuple(available[s]['task_path']) for s in value['span_ids']})<2:
        raise ValueError('Applicability requires distinct public locations')
    return value['applicable']


def compile_candidate(learner,case_ids,selection_ids,*,recovered_demand=None):
    s=learner.system
    if s.config['experiment'].get('choice_method_version')!=VERSION or s.readonly or s.phase!='train' or not s.checkpoint or not s.audit_path:
        raise ValueError('Capability learning requires an explicitly versioned durable writable Train unit')
    if len(set(case_ids))!=len(case_ids) or not 2<=len(case_ids)<=4 or set(case_ids)&set(selection_ids):
        raise ValueError('Generation evidence must be independent of fixed selection')
    by_id={r['task']['physical_key']:r for r in s.bank.train_cases()}
    if not set(case_ids)<=set(by_id):raise ValueError('Missing completed Train source')
    rows=records([by_id[k] for k in case_ids]);unit=digest([VERSION,rows,sorted(selection_ids)])
    key='capability_batch_'+unit
    if s.checkpoint.state.get(key):return s.checkpoint.state[key]
    log={'method_version':VERSION,'prompt_revision':PROMPT_REVISION,'unit_id':unit,'source_case_ids':case_ids,
        'decision':'no_change','candidate':None,'stage':'capability_induction'}
    receive=lambda suffix,prompt,material,name,schema,purpose,validator,repair:learner._receive(
        key+'_'+suffix,'extractor',prompt,material,name,schema,material_version=MATERIAL_VERSION,
        decision_purpose=purpose,validator=validator,repair_limit=repair)
    try:
        public=[{k:r[k] for k in ('task','task_hash','spans')} for r in rows]
        if recovered_demand is not None:
            if recovered_demand['unit_id']!=unit or recovered_demand['public_cases_hash']!=digest(public):
                raise ValueError('Recovered demand source identity changed')
            value=deepcopy(recovered_demand['value']);log['recovered_demand']=deepcopy(recovered_demand)
        else:
            value=receive('demand','Infer one shared executable comparison capability from public demands only. '
                'Do not infer from answers or feedback. Cite exact supplied span IDs from different cases; '
                'case_id must be the supplied task_id or physical_key. Use no_change if no shared demand; optional references must remain exact.',
                {'public_cases':public},'submit_capability',CAPABILITY,'capability_induction',lambda v:validate_capability(v,rows),0)
        capability=validate_capability(value,rows);log['capability']=capability
        if capability:
            log['stage']='guidance_learning'
            proposal=receive('proposal','Propose a short cross-case procedural advisory, or no_change. '
                'Use observed public demands and saved outcomes. Incorrect/truncated responses disclose no true reasoning cause or alternative proof. '
                'Describe checks the solver can perform, leaving their outcomes to the actual task. '
                'Do not prescribe a correct option from strength, fewer dependencies, extra hypotheses or equality wording alone: these do not establish mathematical truth. '
                'No theorem conclusions, formulas, answer labels, lookup tables or guarantees. Cite supplied source spans. The entire card must fit 512 characters.',
                {'capability':capability,'train_cases':rows},'submit_learning',PROPOSAL,'guidance_learning',
                lambda v:validate_candidate(v,capability,rows),s.config['learning']['guidance_repair_limit'])
            candidate=validate_candidate(proposal,capability,rows)
            log['proposal']=proposal
            if candidate:
                log['stage']='guidance_grounding';log['proposed_candidate']=candidate
                check=receive('check','Classify the proposed guidance itself; mathematical statements in its source questions are context, not the guidance. '
                    'procedural_only means executable checks whose outcomes remain to be established by the solver. '
                    'domain_fact means the guidance asserts a particular mathematical conclusion. unsupported includes invented failure causes or '
                    'choosing truth from claim strength, simplicity, dependencies, extra hypotheses or equality wording without justification. '
                    'Reject these shortcuts even if they sound general. Explain the classification in reason and quote actual guidance text. '
                    'Cite exact supplied source spans. This is not a mathematical proof or utility test.',
                    {'candidate':{**{k:candidate[k] for k in ('guidance','applicability','source_records')},
                        'references':candidate['proposal']['references']},
                        'candidate_hash':digest(candidate)},'submit_capability_check',CHECK,'guidance_grounding',
                    lambda v:validate_check(v,candidate),0)
                log['source_check']=check
                if validate_check(check,candidate):log.update(decision='candidate',candidate={**candidate,'source_check':check})
                else:log.update(decision='rejected',reason=check['status'])
    except ValueError as exc:log.update(decision='rejected',reason=str(exc))
    s.checkpoint.advance(s.checkpoint.state['stage'],**{key:log})
    return log


def match_public(learner,candidate,task):
    # Routing sees only the target's public input; never its score or answer.
    capability=candidate['capability'];key='capability_match_'+digest([capability,public_identity(task)])
    return learner._receive(key,'extractor','Match the capability to this public task demand; abstain if unsupported. '
        'Cite exact supplied spans from distinct fields; do not solve or predict a correct option.',
        {'capability':capability,'task':{'goal':task.goal,'inputs':task.inputs},'spans':spans(task),
            'task_hash':public_identity(task)},'submit_capability_match',MATCH,material_version=MATERIAL_VERSION,
        decision_purpose='capability_routing',validator=lambda v:validate_match(v,capability,task),repair_limit=0)


def validate_policy(policy):
    if not isinstance(policy,dict) or set(policy)!={'screen','promotion','min_precomputed_applicable','selection'}:
        raise ValueError('Explicit two-stage utility policy required')
    for stage in ('screen','promotion'):
        rule=policy[stage]
        if set(rule)!={'min_exposed','min_net_correct','max_regressions','require_empty_on_not_increase'} or any(
            type(rule[k]) is not int or rule[k]<0 for k in ('min_exposed','min_net_correct','max_regressions')) or rule['require_empty_on_not_increase'] is not True:
            raise ValueError('Invalid correctness, exposure or empty-answer gate')
    if policy['promotion']['min_net_correct']<1 or policy['promotion']['min_exposed']<1 or type(policy['min_precomputed_applicable']) is not int or policy['min_precomputed_applicable']<1:
        raise ValueError('Promotion requires positive measured correctness and exposure')
    if set(policy['selection'])!={'screen','confirmation'} or any(not rows or any(set(r)!={'task_id','physical_key'} for r in rows) for rows in policy['selection'].values()):
        raise ValueError('Both preregistered selection stages are required')


def gate(facts,rule):
    return (sum(f['exposed'] for f in facts)>=rule['min_exposed'] and
        sum(f['on']['correct']-f['off']['correct'] for f in facts)>=rule['min_net_correct'] and
        sum(f['on']['empty'] for f in facts)<=sum(f['off']['empty'] for f in facts) and
        sum(f['off']['correct'] and not f['on']['correct'] for f in facts)<=rule['max_regressions'])


def arm_fact(trace,task,candidate,arm):
    if trace['task']!=task.__dict__ or trace.get('error') or trace.get('learning') or trace['knowledge_before']!=trace['knowledge_after']:
        raise ValueError('Paired evaluation requires unchanged readonly Train input and no learning')
    if type(trace['score']['hard']) is not bool:raise ValueError('Missing independent task score')
    requests=[r for r in trace['requests'] if r['stage']=='runtime']
    attempts=[a for r in trace['requests'] for a in r.get('http_attempts',[])]
    if not requests or not attempts or any(not isinstance(a.get('raw_usage'),dict) or type(a['raw_usage'].get('total_tokens')) is not int for a in attempts):
        raise ValueError('Paired requests require metered physical attempts')
    if any(r['effective_call_settings']['max_retries']!=0 for r in requests):raise ValueError('Selection retries must be disabled')
    wire=requests[0]['http_attempts'][0]['final_payload_audit']
    messages=deepcopy(wire['messages']);body=json.loads(messages[1]['content'])
    cards=body.get('guidance',[]);ids=trace.get('injected_guidance_ids',[])
    visible=[c['skill_id'] for c in cards]
    if ids!=visible or (candidate['id'] in ids)!=(arm=='on'):raise ValueError('Actual candidate exposure differs from the declared arm')
    body['guidance']=[c for c in cards if c['skill_id']!=candidate['id']]
    messages[1]['content']=json.dumps(body,ensure_ascii=False,sort_keys=True)
    usage={}
    for field,raw_field in [('total_tokens','total_tokens'),('input_tokens','prompt_tokens'),('output_tokens','completion_tokens')]:
        values=[a['raw_usage'].get(raw_field) for a in attempts]
        usage[field]=sum(values) if all(type(v) is int for v in values) else None
    reasoning=[a['raw_usage'].get('completion_tokens_details',{}).get('reasoning_tokens') for a in attempts]
    usage['reasoning_tokens']=sum(reasoning) if all(type(v) is int for v in reasoning) else None
    return {'correct':int(trace['score']['hard']),'empty':int(not bool(trace['execution'].get('prediction'))),
        'answer_status':trace['execution'].get('answer_status'),'termination_reason':trace['execution'].get('reason'),
        'injected_ids':ids,'usage':usage,'physical_http':len(attempts),'retries':len(attempts)-len(trace['requests']),
        'bank_hash':trace['knowledge_before'],'prompt_without_candidate_hash':digest(messages),
        'effective_settings':requests[0]['effective_call_settings'],'wire_parameters':wire['serialized_parameters'],
        'wire_tools_hash':wire['tools_sha256'],'response_models':[(a.get('public_response') or {}).get('model') for a in attempts],
        'trace_hash':digest(trace)}


def pair_fact(candidate,task,pair):
    arms={a:arm_fact(pair[a],task,candidate,a) for a in ('off','on')}
    if any(arms['off'][k]!=arms['on'][k] for k in ('bank_hash','prompt_without_candidate_hash','effective_settings','wire_parameters','wire_tools_hash')):
        raise ValueError('Paired public prompt, other guidance, Bank or effective model settings differ')
    off,on=arms['off']['correct'],arms['on']['correct']
    return {'task_id':task.task_id,'physical_key':task.physical_key,'task_hash':public_identity(task),**arms,
        'exposed':True,'outcome':'win' if on>off else 'loss' if on<off else 'tie'}


def utility_receipt(candidate,pairs,policy,preflight,identity):
    validate_policy(policy)
    facts={key:pair_fact(candidate,PublicTask(**next(r['task'] for r in preflight if r['task']['physical_key']==key)),pair) for key,pair in pairs.items()}
    screen=[facts[r['task']['physical_key']] for r in preflight if r['stage']=='screen' and r['task']['physical_key'] in facts]
    screen_passed=gate(screen,policy['screen'])
    eligible=[r['task']['physical_key'] for r in preflight if validate_match(r['match'],candidate['capability'],PublicTask(**r['task']))]
    expected=[r['task']['physical_key'] for r in preflight if r['task']['physical_key'] in eligible and (r['stage']=='screen' or screen_passed)] if len(eligible)>=policy['min_precomputed_applicable'] else []
    if set(facts)!=set(expected):raise ValueError('Incomplete or noneligible paired selection')
    evidence=list(facts.values())
    totals={}
    for arm in ('off','on'):
        rows=[f[arm] for f in evidence]
        totals[arm]={k:sum(r[k] for r in rows) for k in ('correct','empty','physical_http','retries')}
        totals[arm].update(valid=sum(r['answer_status']=='valid' for r in rows),invalid=sum(r['answer_status']=='invalid_format' for r in rows),
            truncated=sum(r['answer_status']=='truncated_empty' for r in rows),
            usage={k:sum(r['usage'][k] for r in rows) if rows and all(type(r['usage'][k]) is int for r in rows) else None
                for k in ('total_tokens','input_tokens','output_tokens','reasoning_tokens')})
    passed=screen_passed and gate(evidence,policy['promotion'])
    receipt={'schema':VERSION,'candidate_hash':digest(candidate),'preflight':deepcopy(preflight),
        'policy':deepcopy(policy),'identity':deepcopy(identity),'paired':evidence,'screen_passed':screen_passed,
        'totals':totals,'passed':passed,'status':'promoted' if passed else 'coverage_insufficient' if len(eligible)<policy['min_precomputed_applicable'] else
            'screen_rejected' if not screen_passed else 'utility_rejected',
        'qualification':'paired_train_observation_not_proof'}
    if not valid_receipt(candidate,receipt):raise ValueError('Invalid utility receipt identity or evidence')
    return {**receipt,'receipt_hash':digest(receipt)}


def valid_receipt(candidate,receipt):
    preflight=receipt['preflight'];facts=receipt['paired'];policy=receipt['policy'];validate_policy(policy)
    keys=[r['task']['physical_key'] for r in preflight]
    if not keys or len(set(keys))!=len(keys) or set(keys)&{r['task']['physical_key'] for r in candidate['source_records']}:return False
    if any(r['stage'] not in {'screen','confirmation'} or PublicTask(**r['task']).split!='train' for r in preflight):return False
    actual={stage:[{k:r['task'][k] for k in ('task_id','physical_key')} for r in preflight if r['stage']==stage] for stage in ('screen','confirmation')}
    if actual!=policy['selection']:return False
    eligible={r['task']['physical_key']:r for r in preflight if validate_match(r['match'],candidate['capability'],PublicTask(**r['task']))}
    screen=[f for f in facts if f['physical_key'] in eligible and eligible[f['physical_key']]['stage']=='screen']
    screen_passed=gate(screen,policy['screen'])
    expected={k for k,r in eligible.items() if r['stage']=='screen' or screen_passed} if len(eligible)>=policy['min_precomputed_applicable'] else set()
    if len(facts)!=len(expected) or {f['physical_key'] for f in facts}!=expected:return False
    for f in facts:
        if not f['exposed'] or f['task_hash']!=public_identity(PublicTask(**eligible[f['physical_key']]['task'])):return False
        if candidate['id'] in f['off']['injected_ids'] or candidate['id'] not in f['on']['injected_ids']:return False
        if any(f['off'][k]!=f['on'][k] for k in ('bank_hash','prompt_without_candidate_hash','effective_settings','wire_parameters','wire_tools_hash')):return False
        if any(type(f[a][k]) is not int or f[a][k] not in (0,1) for a in ('off','on') for k in ('correct','empty')):return False
    return (receipt['candidate_hash']==digest(candidate) and receipt['screen_passed']==screen_passed and
        receipt['passed']==(screen_passed and gate(facts,policy['promotion'])))


def checked_asset(asset):
    try:
        candidate=asset['candidate'];receipt=asset['utility_receipt'];rows=candidate['source_records']
        rebuilt=validate_candidate(candidate['proposal'],candidate['capability'],rows)
        if validate_capability(candidate['capability']['definition'],rows)!=candidate['capability']:return False
        if {k:v for k,v in candidate.items() if k!='source_check'}!=rebuilt:return False
        if not validate_check(candidate['source_check'],rebuilt):return False
        return (receipt['passed'] is True and valid_receipt(candidate,receipt) and
            receipt['receipt_hash']==digest({k:v for k,v in receipt.items() if k!='receipt_hash'}) and
            asset=={**candidate,'state':'usable','candidate':candidate,'utility_receipt':receipt})
    except (ValueError,KeyError,TypeError):return False


def publish(bank,candidate,receipt):
    asset={**candidate,'state':'usable','candidate':deepcopy(candidate),'utility_receipt':deepcopy(receipt)}
    if not checked_asset(asset):raise ValueError('Only a source-checked, paired-utility-passed advisory can publish')
    return bank.put('skill',asset)


def selection(assets,task,policy,*,candidate=False):
    matches=policy.get('capability_matches',{});selected=[];audit=[]
    for asset in assets:
        reason='unchecked_asset'
        if asset.get('method_version')==VERSION and (candidate or checked_asset(asset)):
            match=matches.get(asset['capability']['id'])
            reason='public_demand_match_missing'
            if match:
                reason='eligible' if validate_match(match,asset['capability'],task) else 'public_demand_mismatch'
        if reason=='eligible':
            if selected or len(json.dumps(render(asset),ensure_ascii=False))+2>512:reason='procedural_bound'
            else:selected.append(deepcopy(asset));reason='selected'
        audit.append({'asset_id':asset['id'],'reason':reason,'route':VERSION})
    return {'selected':selected,'audit':audit,'injected_ids':[a['id'] for a in selected],
        'injected_chars':sum(len(json.dumps(render(a),ensure_ascii=False))+2 for a in selected),'selection_version':VERSION}


class CapabilityView:
    """Paired public matching is shared; the off intervention changes only exposure."""
    def __init__(self,bank,matches,*,off=False,candidate=None):
        if not bank.readonly:raise ValueError('Capability interventions require a readonly Bank')
        self.bank=bank;self.matches=matches;self.off=off;self.candidate=candidate

    def select_guidance(self,task,policy):
        baseline=self.bank.select_guidance(task,{**policy,'capability_matches':self.matches})
        if not self.candidate:return baseline
        if self.candidate['id'] in baseline['injected_ids']:raise ValueError('Trial candidate is already published')
        if self.off:return {**baseline,'view':'candidate_off'}
        result=selection([self.candidate],task,{**policy,'capability_matches':self.matches},candidate=True)
        size=baseline['injected_chars']+result['injected_chars']
        if size>policy['max_total_chars']:raise ValueError('Candidate and preserved guidance exceed the frozen bound')
        return {**result,'selected':baseline['selected']+result['selected'],'injected_ids':baseline['injected_ids']+result['injected_ids'],
            'injected_chars':size,'audit':baseline['audit']+result['audit']}

    def all(self,kind):
        return self.bank.all(kind)

    def get(self,asset_id):
        return next((a for kind in ('skill','program','implementation','workflow') for a in self.all(kind) if a['id']==asset_id),None)

    def retrieve(self,query,limit=8):
        return [a for a in self.bank.retrieve(query,limit) if self.get(a['id'])]

    def planning_cards(self,query):
        return [a for a in self.bank.planning_cards(query) if self.get(a['id'])]

    def __getattr__(self,name):return getattr(self.bank,name)
