"""P7 text policy contracts. Source validity never certifies mathematical truth."""
from copy import deepcopy
from functools import lru_cache
import json
import re

from .contracts import digest, object_schema
from .math_operation_contract_p6 import ContractError, error, require

VERSION = 'livemath.effect-evolution.p7'
R2 = VERSION + '.r2'
MATERIAL = VERSION + '.train-experience.v1'
ANSWER = 'effect_answer.v1'
TEXT = {'type':'string','minLength':1,'maxLength':512}
INDICATORS = ['comparable_conclusions','quantifier_syntax','equivalence_syntax',
              'bound_syntax','scope_syntax','meta_option_present']
INDICATOR_LIST = {'type':'array','items':{'type':'string','enum':INDICATORS},'maxItems':6}
CANDIDATE = object_schema({
    'category':{'type':'string','enum':['claim_strength','necessity_sufficiency','hypothesis_scope',
        'quantifier_order','maximality','combined_decision']},
    'scope':{'type':'string','enum':['global','local']}, 'goal':TEXT,
    'applicability':object_schema({'required_public_indicators':INDICATOR_LIST,
        'excluded_indicators':INDICATOR_LIST},['required_public_indicators','excluded_indicators']),
    'steps':{'type':'array','items':TEXT,'minItems':1,'maxItems':8}, 'stop_condition':TEXT,
    'source_case_ids':{'type':'array','items':{'type':'string','minLength':1},'minItems':2,'maxItems':28},
    'edit_rationale':TEXT}, ['category','scope','goal','applicability','steps','stop_condition',
                            'source_case_ids','edit_rationale'])
BATCH = object_schema({'decision':{'type':'string','enum':['propose','no_change']},
    'candidates':{'type':'array','items':CANDIDATE,'maxItems':3}},['decision'])
EDIT = object_schema({'decision':{'type':'string','enum':['edit','no_change']},
    'edits':{'type':'array','maxItems':2,'items':object_schema({
        'parent_id':{'type':'string','minLength':1},
        'operations':{'type':'array','minItems':1,'maxItems':3,'items':object_schema({
            'op':{'type':'string','enum':['add','delete','replace']},
            'field':{'type':'string','enum':['steps','stop_condition']},
            'index':{'type':'integer','minimum':0,'maximum':7}, 'value':TEXT},['op','field'])},
        'rationale':TEXT},['parent_id','operations','rationale'])}},['decision'])


def card(candidate):
    keys=('goal','applicability','steps','stop_condition')
    if 'rule_basis' in candidate:keys=(*keys,'rule_basis')
    return {k:deepcopy(candidate[k]) for k in keys}


@lru_cache(maxsize=4)
def tokenizer(tokenizer_path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(tokenizer_path))


def token_count(value, tokenizer_path):
    """Measure with a pinned local tokenizer, never label byte counts as tokens."""
    text=json.dumps(value,ensure_ascii=False,separators=(',',':'))
    return len(tokenizer(tokenizer_path).encode(text,add_special_tokens=False).ids)


def validate_candidate(value, sources, *, tokenizer_path=None, max_tokens=600):
    require(value,CANDIDATE)
    ids=value['source_case_ids'];errors=[]
    if len(set(ids))!=len(ids) or not set(ids)<=set(sources):
        errors.append(error('invalid_source','$.source_case_ids','distinct locked proposal Train keys',ids))
    a=value['applicability'];required=a['required_public_indicators'];excluded=a['excluded_indicators']
    if set(required)&set(excluded) or (value['scope']=='local' and not required) or (value['scope']=='global' and (required or excluded)):
        errors.append(error('invalid_applicability','$.applicability','global unconditional or local positive public conditions',a))
    text=json.dumps(card(value),ensure_ascii=False)
    if any(k in text for k in sources) or re.search(r'livemath:\d|(?:choose|select|answer\s*(?:is|:|=))\s*(?:option\s*)?[A-E]\b',text,re.I):
        errors.append(error('answer_or_id_shortcut','$','no task IDs or fixed answer labels',text))
    if re.search(r'(?:always|automatically|unconditionally)\s+(?:choose|select)|(?:choose|select).{0,60}(?:whenever|on seeing)',text,re.I):
        errors.append(error('unconditional_shortcut','$','no string-to-answer policy',text))
    if tokenizer_path and token_count(card(value),tokenizer_path)>max_tokens:
        errors.append(error('injection_too_long','$','card <= '+str(max_tokens)+' tokenizer tokens',text))
    if errors:raise ContractError(errors)
    return deepcopy(value)


def candidate(value, sources, evidence, *, parent=None, delta=None, version=VERSION):
    (validate_candidate_r2 if version==R2 else validate_candidate)(value,sources)
    record={'schema':'math.effect-candidate.v2' if version==R2 else 'math.effect-candidate.v1','method_version':version,'candidate':deepcopy(value),
        'parent_id':parent['id'] if parent else None,'parent_hash':digest(parent) if parent else None,
        'source_manifest_hash':digest(sources),'source_checked':True,'model_evidence':evidence,
        'edit_delta':deepcopy(delta),'qualification':'unproven_hypothesis'}
    record['id']='effect_'+digest(record)
    return record


def checked_candidate(value):
    try:
        if set(value)!={'id','schema','method_version','candidate','parent_id','parent_hash','source_manifest_hash',
            'source_checked','model_evidence','edit_delta','qualification'}:return False
        raw={k:v for k,v in value.items() if k!='id'}
        if value['method_version'] not in {VERSION,R2} or value['id']!='effect_'+digest(raw) or not value['source_checked']:return False
        if value['method_version']==R2:
            if value['schema']!='math.effect-candidate.v2':return False
            validate_candidate_r2(value['candidate'],value['candidate']['source_case_ids'])
        else:require(value['candidate'],CANDIDATE)
        return bool(value['model_evidence']['response_hash'])
    except (KeyError,TypeError,ValueError):return False


def checked_effect_asset(asset, *, usable=True):
    if asset.get('method_version')==R2:
        return False if usable else checked_experimental_asset(asset)
    try:
        if set(asset)-{'id','method_version','effect_candidate','goal','guidance','knowledge_type','execution_intent','state','train_utility_receipts'}:return False
        c=asset['effect_candidate']
        if asset['method_version']!=VERSION or not checked_candidate(c):return False
        if asset['goal']!=c['candidate']['goal'] or asset['guidance']!=json.dumps(card(c['candidate']),ensure_ascii=False):return False
        if not usable:return asset['state']=='candidate' or checked_effect_asset(asset)
        from .math_effect_optimizer import verify_receipt
        receipts=asset['train_utility_receipts']
        return (asset['state']=='active_usable' and len(receipts)==2 and
            all(verify_receipt(r) and r['qualified'] and c['id'] in r['candidate_ids']
                and r['candidate_hashes'][r['candidate_ids'].index(c['id'])]==digest(c)
                and r['source_manifest_hash']==c['source_manifest_hash'] for r in receipts) and
            [r['evaluation_set'] for r in receipts]==['edit_feedback','protected_gate'])
    except (KeyError,TypeError,ValueError):return False


def asset(c, receipts=None):
    if c['method_version']==R2:
        value={'method_version':R2,'effect_candidate':deepcopy(c),'goal':c['candidate']['goal'],
            'guidance':json.dumps(card(c['candidate']),ensure_ascii=False),'knowledge_type':c['candidate']['rule_basis'],
            'execution_intent':'guidance_only','state':'candidate','evaluation_status':'experimental_not_claimed_effective',
            'train_utility_receipts':deepcopy(receipts or [])}
        value['id']='skill_'+digest(value)
        return value
    value={'method_version':VERSION,'effect_candidate':deepcopy(c),'goal':c['candidate']['goal'],
        'guidance':json.dumps(card(c['candidate']),ensure_ascii=False),'knowledge_type':'procedural',
        'execution_intent':'guidance_only','state':'active_usable' if receipts else 'candidate',
        'train_utility_receipts':deepcopy(receipts or [])}
    value['id']='skill_'+digest(value)
    return value


# R2 is deliberately separate: frozen v1 candidates/receipts keep their old meaning.
R2_TEXT={'type':'string','minLength':1}
R2_INDICATORS=[*INDICATORS,'has_stronger_meta']
R2_CONDITIONS={'type':'array','items':{'type':'string','enum':R2_INDICATORS},'maxItems':len(R2_INDICATORS)}
CANDIDATE_R2=object_schema({
    'category':{'type':'string','minLength':1,'maxLength':80},'goal':R2_TEXT,
    'applicability':object_schema({'mode':{'type':'string','enum':['all_tasks','when']},
        'require_all':R2_CONDITIONS,'exclude_any':R2_CONDITIONS},['mode','require_all','exclude_any']),
    'steps':{'type':'array','items':R2_TEXT,'minItems':1,'maxItems':8},'stop_condition':R2_TEXT,
    'rule_basis':{'type':'string','enum':['reasoning_procedure','benchmark_prior','mixed']},
    'source_case_ids':{'type':'array','items':R2_TEXT,'minItems':1,'maxItems':60},'edit_rationale':R2_TEXT},
    ['category','goal','applicability','steps','stop_condition','rule_basis','source_case_ids','edit_rationale'])
BATCH_R2=object_schema({'decision':{'type':'string','enum':['propose','no_change']},
    'candidates':{'type':'array','items':CANDIDATE_R2,'minItems':1,'maxItems':3}},['decision'])
EDIT_ITEM_R2=object_schema({'parent_id':R2_TEXT,'replacement':CANDIDATE_R2,'rationale':R2_TEXT},
    ['parent_id','replacement','rationale'])
EDIT_R2=object_schema({'decision':{'type':'string','enum':['edit','no_change']},
    'edits':{'type':'array','items':EDIT_ITEM_R2,'minItems':1,'maxItems':2}},['decision'])
REPAIR_R2=object_schema({'decision':{'type':'string','enum':['repair','no_change']},
    'repairs':{'type':'array','maxItems':3,'items':object_schema({'index':{'type':'integer','minimum':0,'maximum':2},
        'replacement':CANDIDATE_R2},['index','replacement'])}},['decision'])
REPAIR_EDIT_R2=object_schema({'decision':{'type':'string','enum':['repair','no_change']},
    'repairs':{'type':'array','maxItems':2,'items':object_schema({'index':{'type':'integer','minimum':0,'maximum':1},
        'replacement':EDIT_ITEM_R2},['index','replacement'])}},['decision'])


def validate_candidate_r2(value,sources,*,tokenizer_path=None,max_tokens=600):
    require(value,CANDIDATE_R2)  # All item types are checked before set operations.
    errors=[];ids=value['source_case_ids'];a=value['applicability']
    yes,no=a['require_all'],a['exclude_any']
    if len(set(ids))!=len(ids) or not set(ids)<=set(sources):
        errors.append(error('invalid_source','$.source_case_ids','distinct shown Train physical IDs',ids))
    if (len(set(yes))!=len(yes) or len(set(no))!=len(no) or set(yes)&set(no) or
        a['mode']=='all_tasks' and (yes or no) or a['mode']=='when' and not yes):
        errors.append(error('invalid_applicability','$.applicability','all_tasks empty lists or when with positive noncontradictory conditions',a))
    text=json.dumps(card(value),ensure_ascii=False)
    # Benchmark priors may refer to public option semantics, never a fixed answer label.
    if (any(k in text for k in sources) or re.search(r'livemath:\d',text) or
        re.search(r'(?i:choose|select|return|output|predict|answer\s*(?:is|:|=))\s*(?i:option\s*|label\s*)?[A-Z]\b|(?i:<answer>)[A-Za-z](?i:</answer>)',text)):
        errors.append(error('answer_or_id_shortcut','$','no physical IDs or fixed answer labels',text))
    if tokenizer_path and token_count(card(value),tokenizer_path)>max_tokens:
        errors.append(error('injection_too_long','$','card <= '+str(max_tokens)+' tokenizer tokens',value))
    if errors:raise ContractError(errors)
    return deepcopy(value)


def migrate_candidate_r2(value,sources,*,all_have_choices):
    """Lossless v1 routing migration; only proven redundant global guards are removed."""
    require(value,CANDIDATE);a=value['applicability'];yes=a['required_public_indicators'];no=a['excluded_indicators']
    if value['scope']=='global' and (yes or no) and not (all_have_choices and yes==['comparable_conclusions'] and not no):
        raise ContractError([error('migration_needs_model','$.applicability','retain real conditions as when or explicitly repair all_tasks',a)])
    result={k:deepcopy(v) for k,v in value.items() if k not in {'scope','applicability'}}
    result['applicability']={'mode':'all_tasks' if value['scope']=='global' else 'when',
        'require_all':[] if value['scope']=='global' else yes,'exclude_any':[] if value['scope']=='global' else no}
    result['rule_basis']='reasoning_procedure'
    return validate_candidate_r2(result,sources)


def checked_experimental_asset(a):
    try:
        expected=asset(a['effect_candidate'],a['train_utility_receipts'])
        if a!=expected or a['method_version']!=R2 or not checked_candidate(a['effect_candidate']):return False
        from .math_effect_optimizer import verify_receipt_r2
        return all(verify_receipt_r2(r) and a['effect_candidate']['id'] in r['candidate_ids'] and
            digest(a['effect_candidate'])==r['candidate_hashes'][r['candidate_ids'].index(a['effect_candidate']['id'])]
            for r in a['train_utility_receipts'])
    except (KeyError,TypeError,ValueError):return False
