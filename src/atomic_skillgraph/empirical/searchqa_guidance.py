"""Versioned public-structure selection for derived SearchQA capabilities."""
import re
import hashlib
from .contracts import digest, object_schema, validate_schema_instance

VERSION = 'searchqa.train-trace-capability.r2'
MATERIAL = 'searchqa.train-trace-material.r2'
SELECTION = 'searchqa.public-structure-selection.r2'
OUTPUT = 'searchqa.public-span-output.r2'
QUALIFICATION = 'searchqa.worker-transfer-qualification.r2'
IMAGE = 'sha256:4db520e20cd121f830731c2d0c0bcfbe9cacd254d1817262404cc18e1e757783'
FAMILIES = ('entity', 'relation', 'temporal_numeric', 'alias', 'multi_doc')
REQUIREMENTS = ('documents', 'multiple_documents', 'question_terms', 'temporal_question', 'alias_surface')
POLICY = {'version': SELECTION, 'max_items': 2, 'max_total_chars': 1200,
          'max_card_chars': 900, 'min_source_cases': 2}
STOP = set('a an the of to in on at for from by is are was were be been being this that these those it its who what which where when how and or with as his her their he she they you your named called'.split())


def quote_hash(text): return hashlib.sha256(text.encode('utf-8')).hexdigest()


def terms(question):
    return sorted({w.casefold() for w in re.findall(r'[\w]+', question) if len(w) > 2 and w.casefold() not in STOP})


def documents(context):
    """Unicode offsets are absolute in the unchanged public context."""
    markers = list(re.finditer(re.escape('[DOC]'), context))
    spans = [(m.end(), markers[i+1].start() if i+1 < len(markers) else len(context)) for i,m in enumerate(markers)]
    if not markers and context.strip(): spans = [(0, len(context))]
    return [{'doc_index': i+1, 'start': a, 'end': b, 'quote': context[a:b],
             'quote_sha': quote_hash(context[a:b])} for i,(a,b) in enumerate(spans) if context[a:b].strip()]


def features(task):
    question, context = task.goal, task.inputs.get('context', '')
    docs = documents(context)
    temporal = bool(re.search(r'\b(?:year|date|century|oldest|earliest|latest|before|after|first|last|largest|smallest|number|many|much)\b|\d', question, re.I))
    alias = bool(re.search(r'\b(?:also known|aka|a\.k\.a\.|abbreviat|nicknam|called)\b|\([A-Z]{2,8}\)', context, re.I))
    flags = {'documents': bool(docs), 'multiple_documents': len(docs) >= 2,
             'question_terms': bool(terms(question)), 'temporal_question': temporal, 'alias_surface': alias}
    families = {'entity', 'relation'} if flags['documents'] and flags['question_terms'] else set()
    if temporal: families.add('temporal_numeric')
    if alias: families.add('alias')
    if len(docs) >= 2: families.add('multi_doc')
    return {'requirements': flags, 'families': sorted(families), 'question_terms': terms(question), 'doc_count': len(docs)}


def applicability(task, contract):
    public = features(task)
    if contract['question_pattern'] not in public['families']: return False, 'question_structure_mismatch'
    missing = [k for k in contract['public_evidence_required'] if not public['requirements'][k]]
    return (False, 'missing_public_prerequisite:' + ','.join(missing)) if missing else (True, 'public_prerequisites_met')


APPLICABILITY = object_schema({'question_pattern': {'enum': list(FAMILIES)},
    'public_evidence_required': {'type': 'array', 'items': {'enum': list(REQUIREMENTS)}, 'minItems': 1, 'maxItems': 5}},
    ['question_pattern', 'public_evidence_required'])
REF = object_schema({'case_id': {'type':'string'}, 'doc_index': {'type':'integer','minimum':1},
    'start': {'type':'integer','minimum':0}, 'end': {'type':'integer','minimum':1},
    'quote_sha': {'type':'string'}, 'basis': {'enum':['posthoc_public_check']}},
    ['case_id','doc_index','start','end','quote_sha','basis'])
ATOMIC = object_schema({'kind': {'enum':['atomic_procedure']}, 'method_version': {'enum':[VERSION]},
    'goal': {'type':'string','minLength':1,'maxLength':160}, 'applicability': APPLICABILITY,
    'procedure': {'type':'array','items':{'type':'string','minLength':1,'maxLength':300},'minItems':1,'maxItems':6},
    'stop_condition': {'type':'string','minLength':1,'maxLength':180},
    'source_case_ids': {'type':'array','items':{'type':'string'},'minItems':2,'maxItems':4},
    'evidence_refs': {'type':'array','items':REF,'minItems':2,'maxItems':12},
    'parent_ids': {'type':'array','items':{'type':'string'},'maxItems':4},
    'execution_intent': {'enum':['guidance_only']}},
    ['kind','method_version','goal','applicability','procedure','stop_condition','source_case_ids','evidence_refs','parent_ids','execution_intent'])


def render_procedure(candidate):
    return '\n'.join([*candidate['procedure'], 'Fallback: ' + candidate['stop_condition']])


def checked_asset(asset):
    meta = asset.get('searchqa_metadata', {})
    try:
        validate_schema_instance(meta['candidate'], ATOMIC)
        c = meta['candidate']
        if (asset.get('method_version') != VERSION or meta.get('schema') != MATERIAL or
                meta.get('source_consistent') is not True or len(set(c['source_case_ids'])) < 2 or
                asset.get('goal') != c['goal'] or asset.get('guidance') != render_procedure(c) or
                meta.get('candidate_hash') != digest(c) or len(asset['guidance']) > 600 or
                meta.get('utility') not in {'not_tested','passed'}): return False
        card = {'skill_id': asset['id'], 'goal':asset['goal'], 'guidance':asset['guidance']}
        return len(__import__('json').dumps(card, ensure_ascii=False, separators=(',',':'))) <= POLICY['max_card_chars']
    except (KeyError,TypeError,ValueError): return False


def select(assets, task, policy=None):
    if policy is not None and policy != POLICY: raise ValueError('SearchQA selection policy changed')
    selected, rejected, families, chars = [], [], set(), 0
    valid = []
    for a in assets:
        if not checked_asset(a): rejected.append({'id':a.get('id'), 'reason':'invalid_r2_asset'}); continue
        c = a['searchqa_metadata']['candidate']
        okay, reason = applicability(task, c['applicability'])
        if not okay: rejected.append({'id':a['id'], 'reason':reason}); continue
        valid.append(a)
    valid.sort(key=lambda a:(-len(a['searchqa_metadata']['candidate']['applicability']['public_evidence_required']),
                            -len(set(a['searchqa_metadata']['candidate']['source_case_ids'])), a['id']))
    for a in valid:
        family = a['searchqa_metadata']['candidate']['applicability']['question_pattern']
        reason = ('similar_capability' if family in families else 'item_limit' if len(selected) >= 2 else
                  'text_budget' if chars + len(a['guidance']) > 1200 else None)
        if reason: rejected.append({'id':a['id'], 'reason':reason}); continue
        selected.append(a); families.add(family); chars += len(a['guidance'])
    return {'version':SELECTION, 'selected':selected, 'rejected':rejected,
            'public_features':features(task), 'guidance_chars':chars}


def validate_wire(cards):
    import json
    if len(cards) > 2 or sum(len(c['guidance']) for c in cards) > 1200:
        raise ValueError('R2 wire Guidance exceeds locked budget')
    for c in cards:
        if set(c) != {'skill_id','goal','guidance'} or len(c['goal']) > 160 or len(c['guidance']) > 600 or len(
                json.dumps(c, ensure_ascii=False, separators=(',',':'))) > 900:
            raise ValueError('Invalid R2 wire Guidance card')


class RuntimeView:
    """Same readonly Bank, exact intervention in learned visibility only."""
    def __init__(self, bank, arm):
        if not bank.readonly or arm not in {'NoSkill','Guidance-only','Atomic-full'}:
            raise ValueError('R2 evaluation requires a readonly arm')
        self.bank,self.arm,self.task=bank,arm,None

    def all(self,kind):
        if self.arm=='NoSkill' or self.arm=='Guidance-only' and kind!='skill': return []
        return self.bank.all(kind)

    def get(self,key):
        return next((a for kind in ('skill','program','implementation') for a in self.all(kind) if a['id']==key),None)

    def retrieve_searchqa_guidance(self,task,policy):
        self.task=task
        return {**select(self.all('skill'),task,policy),'arm':self.arm}

    def program_eligible(self,program):
        if not program or self.arm!='Atomic-full' or not self.bank.program_eligible(program) or self.task is None:return False
        return applicability(self.task,program['entry_constraints']['spec']['applicability'])[0]

    def program_options(self,query,**kwargs):
        if self.arm!='Atomic-full':return []
        return [{**c,'host_bound_inputs':{'question':'task.goal','context':'task.inputs.context'},
                 'call_contract':'Use arguments={} to bind the unchanged public question/context; do not copy source examples.'}
                for c in self.bank.program_options(query,**kwargs) if self.program_eligible(self.bank.get(c['id']))]

    def __getattr__(self,name):return getattr(self.bank,name)
