"""P7 text policy contracts. Source validity never certifies mathematical truth."""
from copy import deepcopy
from functools import lru_cache
import json
import re

from .contracts import digest, object_schema
from .math_operation_contract_p6 import ContractError, error, require

VERSION = 'livemath.effect-evolution.p7'
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
    return {k:deepcopy(candidate[k]) for k in ('goal','applicability','steps','stop_condition')}


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


def candidate(value, sources, evidence, *, parent=None, delta=None):
    validate_candidate(value,sources)
    record={'schema':'math.effect-candidate.v1','method_version':VERSION,'candidate':deepcopy(value),
        'parent_id':parent['id'] if parent else None,'parent_hash':digest(parent) if parent else None,
        'source_manifest_hash':digest(sources),'source_checked':True,'model_evidence':evidence,
        'edit_delta':deepcopy(delta),'qualification':'unproven_hypothesis'}
    record['id']='effect_'+digest(record)
    return record


def checked_candidate(value):
    try:
        raw={k:v for k,v in value.items() if k!='id'}
        if value['method_version']!=VERSION or value['id']!='effect_'+digest(raw) or not value['source_checked']:return False
        require(value['candidate'],CANDIDATE)
        return bool(value['model_evidence']['response_hash'])
    except (KeyError,TypeError,ValueError):return False


def checked_effect_asset(asset, *, usable=True):
    try:
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
    value={'method_version':VERSION,'effect_candidate':deepcopy(c),'goal':c['candidate']['goal'],
        'guidance':json.dumps(card(c['candidate']),ensure_ascii=False),'knowledge_type':'procedural',
        'execution_intent':'guidance_only','state':'active_usable' if receipts else 'candidate',
        'train_utility_receipts':deepcopy(receipts or [])}
    value['id']='skill_'+digest(value)
    return value
