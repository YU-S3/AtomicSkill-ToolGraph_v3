"""R3 selection uses public, necessary cues; it never predicts answer correctness."""
from copy import deepcopy
import json
import re

from .contracts import digest, object_schema, validate_schema_instance
from .searchqa_guidance import IMAGE, REF, documents, terms, quote_hash, render_procedure

VERSION = 'searchqa.train-trace-capability.r3'
MATERIAL = 'searchqa.train-trace-material.r3'
SELECTION = 'searchqa.public-prerequisite-selection.r3'
OUTPUT = 'searchqa.typed-public-evidence.r3'
QUALIFICATION = 'searchqa.all-source-transfer-qualification.r3'
FINALIZATION = VERSION + '.length-only.v1'
FAMILIES = ('evidence_locate', 'explicit_alias', 'typed_numeric_compare',
            'explicit_relation', 'multi_doc_join', 'source_check')
POLICY = {'version': SELECTION, 'max_items': 1, 'max_total_chars': 600,
          'max_card_chars': 900, 'min_source_cases': 2, 'execution_order': 'model_first'}
REQUIREMENTS = ('question_term_overlap', 'explicit_alias_cue', 'comparable_numbers',
                'explicit_relation_cue', 'shared_explicit_identifier')
NECESSARY = dict(zip(FAMILIES, REQUIREMENTS[:4] + ('shared_explicit_identifier', 'explicit_relation_cue')))
RELATIONS = {'born_in': 'was born in', 'written_by': 'was written by',
             'capital_of': 'is the capital of', 'founded_by': 'was founded by'}
ALIAS_CUE = r'\balso known as\b|\bstands for\b|\([A-Z]{2,8}\)'
IDENTIFIER = r'\b(?:ISBN|ISSN|DOI|identifier|registration number)\s*[:#]?\s*([\w/-](?:[\w./-]*[\w/-])?)'
NUMBER = r'(?<![\w.])(-?\d+(?:\.\d+)?)\s+(kg|km|miles|meters|years|days|percent|%)\b'
APPLICABILITY = object_schema({'family': {'enum': list(FAMILIES)},
    'public_evidence_required': {'type': 'array', 'items': {'enum': list(REQUIREMENTS)}, 'minItems': 1, 'maxItems': 5}},
    ['family', 'public_evidence_required'])
ATOMIC = object_schema({'kind': {'enum': ['atomic_procedure']}, 'method_version': {'enum': [VERSION]},
    'family': {'enum': list(FAMILIES)}, 'operation': {'type': 'string', 'minLength': 1, 'maxLength': 100},
    'goal': {'type': 'string', 'minLength': 1, 'maxLength': 160}, 'applicability': APPLICABILITY,
    'procedure': {'type': 'array', 'items': {'type': 'string', 'minLength': 1, 'maxLength': 300}, 'minItems': 1, 'maxItems': 6},
    'stop_condition': {'type': 'string', 'minLength': 1, 'maxLength': 180},
    'source_case_ids': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 2, 'maxItems': 4},
    'evidence_refs': {'type': 'array', 'items': REF, 'minItems': 2, 'maxItems': 8},
    'parent_ids': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 4},
    'execution_intent': {'enum': ['guidance_only']}, 'output_role': {'enum': ['intermediate']}},
    ['kind', 'method_version', 'family', 'operation', 'goal', 'applicability', 'procedure',
     'stop_condition', 'source_case_ids', 'evidence_refs', 'parent_ids', 'execution_intent', 'output_role'])


def features(task):
    docs = documents(task.inputs.get('context', ''))
    query = set(terms(task.goal))
    overlap = [d['doc_index'] for d in docs if query & set(terms(d['quote']))]
    aliases = [d['doc_index'] for d in docs if re.search(ALIAS_CUE, d['quote']) and query & set(terms(d['quote']))]
    relations = [d['doc_index'] for d in docs if any(cue in d['quote'].lower() for cue in RELATIONS.values())
                 and query & set(terms(d['quote']))]
    numeric = [(m.group(2).lower(), d['doc_index']) for d in docs for m in re.finditer(NUMBER, d['quote'], re.I)
               if query & set(terms(d['quote']))]
    identifiers = {}
    for d in docs:
        for m in re.finditer(IDENTIFIER, d['quote'], re.I):
            identifiers.setdefault(m.group(1).casefold(), set()).add(d['doc_index'])
    flags = {'question_term_overlap': len(overlap) >= 2, 'explicit_alias_cue': bool(aliases),
             'comparable_numbers': bool(re.search(r'\b(?:larger|smaller|largest|smallest|more|less|difference|longer|shorter)\b', task.goal, re.I))
                 and any(sum(u == unit for u, _ in numeric) >= 2 for unit, _ in numeric),
             'explicit_relation_cue': bool(relations),
             'shared_explicit_identifier': any(len(ids) >= 2 for ids in identifiers.values()) and len(overlap) >= 2}
    return {'requirements': flags, 'question_terms': sorted(query), 'overlap_docs': overlap,
            'alias_docs': aliases, 'relation_docs': relations, 'doc_count': len(docs)}


def applicability(task, contract):
    validate_schema_instance(contract, APPLICABILITY)
    needed = contract['public_evidence_required']
    if NECESSARY[contract['family']] not in needed:
        return False, 'missing_kind_specific_contract'
    public = features(task)
    missing = [k for k in needed if not public['requirements'][k]]
    return (False, 'missing_public_prerequisite:' + ','.join(missing)) if missing else (True, 'necessary_public_cue_present')


def checked_asset(asset):
    try:
        meta = asset['searchqa_metadata']; c = meta['candidate']
        validate_schema_instance(c, ATOMIC)
        card = {'skill_id': asset['id'], 'goal': asset['goal'], 'guidance': asset['guidance']}
        validate_wire([card])
        return (asset['method_version'] == VERSION and meta['schema'] == MATERIAL and meta['source_consistent'] is True
            and meta['candidate_hash'] == digest(c) and asset['goal'] == c['goal']
            and asset['guidance'] == render_procedure(c) and len(set(c['source_case_ids'])) >= 2
            and c['family'] == c['applicability']['family'] and NECESSARY[c['family']] in c['applicability']['public_evidence_required']
            and meta['utility'] == 'not_tested' and meta['historical_execution_claim'] is False)
    except (KeyError, TypeError, ValueError):
        return False


def validate_wire(cards):
    if len(cards) > 1 or sum(len(c['guidance']) for c in cards) > 600:
        raise ValueError('R3 wire exceeds one short actionable card')
    for c in cards:
        if set(c) != {'skill_id', 'goal', 'guidance'} or len(c['goal']) > 160 or len(c['guidance']) > 600 or len(
                json.dumps(c, ensure_ascii=False, separators=(',', ':'))) > 900:
            raise ValueError('R3 card exceeds its locked capacity')


def select(assets, task, policy=None):
    if policy is not None and policy != POLICY: raise ValueError('R3 selection policy changed')
    audit, eligible = [], []
    for asset in assets:
        okay, reason = applicability(task, asset['entry_constraints']) if checked_asset(asset) else (False, 'invalid_r3_asset')
        audit.append({'id': asset.get('id'), 'applicable': okay, 'reason': reason})
        if okay: eligible.append(asset)
    # Narrow, mechanically checkable operations precede broad localization. No outcome is read.
    eligible.sort(key=lambda a: (a['searchqa_metadata']['candidate']['family'] == 'evidence_locate',
        -len(a['entry_constraints']['public_evidence_required']),
        -len(a['searchqa_metadata']['candidate']['source_case_ids']), a['id']))
    selected = eligible[:1]
    for row in audit:
        row['selected'] = bool(selected and row['id'] == selected[0]['id'])
        if row['applicable'] and not row['selected']: row['reason'] = 'one_card_deterministic_priority'
    return {'version': SELECTION, 'selected': selected, 'all_candidates': audit,
            'public_features': features(task), 'guidance_chars': sum(len(a['guidance']) for a in selected)}


class RuntimeView:
    def __init__(self, bank, arm):
        if not bank.readonly or arm not in {'NoSkill', 'Guidance-only', 'Atomic-full'}:
            raise ValueError('R3 evaluation requires a readonly arm')
        bank.bind_searchqa_method(VERSION)
        self.bank, self.arm, self.task, self.selected = bank, arm, None, set()

    def all(self, kind):
        if self.arm == 'NoSkill' or self.arm == 'Guidance-only' and kind != 'skill': return []
        return self.bank.all(kind)

    def get(self, key):
        return next((a for kind in ('skill', 'program', 'implementation') for a in self.all(kind) if a['id'] == key), None)

    def retrieve_searchqa_guidance(self, task, policy):
        self.task = task
        result = select(self.all('skill'), task, policy)
        self.selected = {a['id'] for a in result['selected']}
        return {**result, 'arm': self.arm}

    def program_eligible(self, program):
        return bool(program and self.arm == 'Atomic-full' and self.task is not None
            and program['entry_constraints']['atomic_skill_id'] in self.selected
            and self.bank.program_eligible(program)
            and applicability(self.task, program['entry_constraints']['spec']['preconditions'])[0])

    def program_options(self, query, **kwargs):
        if self.arm != 'Atomic-full': return []
        return [{**c, 'host_bound_inputs': {'question': 'task.goal', 'context': 'task.inputs.context'},
                 'call_contract': 'arguments={} binds unchanged full question/context; output is intermediate evidence only.'}
                for c in self.bank.program_options(query, **kwargs) if self.program_eligible(self.bank.get(c['id']))]

    def __getattr__(self, name): return getattr(self.bank, name)


def normalize_finish(value, schema):
    """Only annotation keys on an otherwise valid finish may be removed."""
    if not isinstance(value, dict) or value.get('action') != 'finish' or not isinstance(value.get('answer'), str) or not value['answer'].strip():
        return value, None
    extras = set(value) - set(schema['properties'])
    if not extras or extras - {'unit', 'end'}: return value, None
    normalized = {k: v for k, v in value.items() if k not in extras}
    validate_schema_instance(normalized, schema)
    return normalized, {'version': VERSION + '.answer.v1', 'removed': sorted(extras),
        'raw_arguments': deepcopy(value), 'raw_arguments_hash': digest(value), 'normalized_arguments_hash': digest(normalized)}


def validate_step(value, schema):
    validate_schema_instance(value, schema)
    fields = {'finish': {'answer'}, 'read_result': {'result_id', 'path'},
              'call_program': {'name', 'arguments', 'argument_refs'},
              'execute_python': {'source', 'arguments', 'argument_refs', 'input_schema', 'output_schema'}}
    action = value['action']
    if set(value) - fields[action] - {'action', 'used_results', 'replaced_results'}:
        raise ValueError('R3 action contains unrelated fields')
    required = {'finish': {'answer'}, 'read_result': {'result_id', 'path'},
                'call_program': {'name'}, 'execute_python': {'source', 'input_schema', 'output_schema'}}
    if required[action] - set(value): raise ValueError('R3 action is missing required fields')
