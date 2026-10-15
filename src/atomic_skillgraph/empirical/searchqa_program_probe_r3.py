"""Typed public operations with exact provenance and all-source Worker qualification."""
from copy import deepcopy
import ast
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import time

from .contracts import PublicTask, digest, object_schema, program_digest, validate_program, validate_schema_instance
from .checkpoint import save_json
from .searchqa_guidance import REF, documents, terms
from .searchqa_guidance_r3 import (VERSION, MATERIAL, OUTPUT, QUALIFICATION, IMAGE, FAMILIES,
    APPLICABILITY, RELATIONS, IDENTIFIER)
from .searchqa_program_probe import INPUT, SPAN, BUILD, inputs, asdict_task
from .task_context import TaskContext
from ..harness.simple_protocol import Broker
from ..harness.benchmarks import AnswerAdapter


def array(item, minimum=1, maximum=8):
    return {'type': 'array', 'items': item, 'minItems': minimum, 'maxItems': maximum}


TEXT = {'type': 'string', 'minLength': 1, 'maxLength': 160}
ALIAS = object_schema({'alias_pair': array(TEXT, 2, 2), 'cue': {'enum': ['also known as', 'stands for', 'parenthesized_acronym']},
    'source_ref': SPAN}, ['alias_pair', 'cue', 'source_ref'])
NUMBER = object_schema({'subject': TEXT, 'value': {'type': 'string', 'pattern': r'^-?\d+(\.\d+)?$'},
    'unit': TEXT, 'time': {'type': 'string', 'maxLength': 80}, 'span': SPAN}, ['subject', 'value', 'unit', 'time', 'span'])
RELATION = object_schema({'relation': {'enum': list(RELATIONS)}, 'subject': TEXT, 'candidate': TEXT,
    'source_ref': SPAN, 'ambiguity': {'enum': [False]}}, ['relation', 'subject', 'candidate', 'source_ref', 'ambiguity'])
OUTPUTS = {
    'evidence_locate': object_schema({'matches': array(object_schema({**SPAN['properties'],
        'rank_reason': {'enum': ['question_term_overlap']}}, [*SPAN['required'], 'rank_reason']))}, ['matches']),
    'explicit_alias': object_schema({'aliases': array(ALIAS)}, ['aliases']),
    'typed_numeric_compare': object_schema({'values': array(NUMBER, 2), 'comparison': object_schema({
        'left': {'type': 'integer', 'minimum': 0, 'maximum': 7}, 'right': {'type': 'integer', 'minimum': 0, 'maximum': 7},
        'relation': {'enum': ['less', 'equal', 'greater']}}, ['left', 'right', 'relation'])}, ['values', 'comparison']),
    'explicit_relation': object_schema({'facts': array(RELATION)}, ['facts']),
    'multi_doc_join': object_schema({'entity_keys': array(TEXT, 1, 4), 'evidence_refs': array(SPAN, 2),
        'conflicts': {'type': 'array', 'maxItems': 0}}, ['entity_keys', 'evidence_refs', 'conflicts']),
    'source_check': object_schema({'candidate': TEXT, 'public_support_refs': array(RELATION),
        'abstain': {'enum': [False]}}, ['candidate', 'public_support_refs', 'abstain']),
}
SPEC_R3 = object_schema({'kind': {'enum': list(FAMILIES)}, 'goal': TEXT,
    'input_contract': {'type': 'object'}, 'output_contract': {'type': 'object'},
    'source_refs': array(REF, 2),
    'source_case_ids': array(TEXT, 2, 4), 'preconditions': APPLICABILITY,
    'abstention': array({'enum': ['needs_input', 'not_found', 'blocked']}, 3, 3),
    'result_role': {'enum': ['intermediate']}, 'proposal_origin': {'enum': ['candidate_derived_from_text']}},
    ['kind', 'goal', 'input_contract', 'output_contract', 'source_refs', 'source_case_ids',
     'preconditions', 'abstention', 'result_role', 'proposal_origin'])
BOUNDARIES = {'boundary:empty_context', 'boundary:empty_question', 'boundary:unicode_parameter',
              'boundary:ambiguous', 'boundary:unit_conflict', 'boundary:offset_guard'}


def validate_spec(spec, asset):
    validate_schema_instance(spec, SPEC_R3)
    c = asset['searchqa_metadata']['candidate']
    if (spec['input_contract'] != INPUT or spec['output_contract'] != OUTPUTS[spec['kind']]
            or spec['kind'] != c['family'] or spec['preconditions'] != c['applicability']
            or spec['source_case_ids'] != c['source_case_ids'] or spec['source_refs'] != c['evidence_refs']
            or set(spec['abstention']) != {'needs_input', 'not_found', 'blocked'}):
        raise ValueError('R3 typed ProgramSpec changed its Atomic/source/interface contract')


def prepare(source, spec, asset, environment, *, parent=None):
    validate_spec(spec, asset)
    program = {'source': source, 'entry': 'run', 'input_schema': deepcopy(INPUT),
        'output_schema': deepcopy(OUTPUTS[spec['kind']]), 'allowed_tools': [], 'environment': deepcopy(environment),
        'result_role': 'intermediate', 'method_version': VERSION, 'proposal_origin': 'candidate_derived_from_text',
        'entry_constraints': {'method_version': VERSION, 'output_contract': OUTPUT, 'spec': deepcopy(spec),
            'atomic_skill_id': asset['id'], 'parent_program_id': parent}, 'state': 'candidate'}
    validate_program(program); validate_declaration(program)
    return program


def validate_declaration(program):
    c = program.get('entry_constraints', {}); spec = c.get('spec', {})
    validate_schema_instance(spec, SPEC_R3)
    if (program.get('method_version') != VERSION or c.get('method_version') != VERSION or c.get('output_contract') != OUTPUT
            or program.get('proposal_origin') != 'candidate_derived_from_text' or program['input_schema'] != INPUT
            or program['output_schema'] != OUTPUTS[spec['kind']] or spec['input_contract'] != INPUT
            or spec['output_contract'] != OUTPUTS[spec['kind']] or program['allowed_tools']
            or program['environment'].get('image_digest') != IMAGE or program['environment'].get('adapter_abi') != 'simple.v2'
            or program.get('result_role') != 'intermediate' or not c.get('atomic_skill_id', '').startswith('skill_')):
        raise ValueError('Invalid R3 typed Program authority/interface')
    allowed = {'re', 'json', 'datetime', 'decimal', 'math', 'collections', 'unicodedata', 'string', 'statistics', 'itertools'}
    forbidden = {'open', 'eval', 'exec', 'compile', 'globals', 'locals', 'vars', 'getattr', 'setattr', 'delattr', '__import__', 'input'}
    for node in ast.walk(ast.parse(program['source'])):
        modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ''] if isinstance(node, ast.ImportFrom) else []
        if any(name.split('.')[0] not in allowed for name in modules) or isinstance(node, ast.ImportFrom) and node.level:
            raise ValueError('R3 permits only its bounded public-text standard library')
        if isinstance(node, ast.Name) and node.id in forbidden or isinstance(node, ast.Attribute) and (
                node.attr.startswith('__') or isinstance(node.value, ast.Name) and node.value.id == 'ctx'):
            raise ValueError('R3 public-text Program cannot access files, environment, reflection or Host context')


def check(program, arguments, result):
    validate_declaration(program); validate_schema_instance(arguments, INPUT)
    if result.get('status') != 'ok': return {'passed': False, 'reason': result.get('status', 'missing_status')}
    kind = program['entry_constraints']['spec']['kind']; output = result.get('outputs')
    validate_schema_instance(output, OUTPUTS[kind])
    context = arguments['context']; question = arguments['question']; docs = documents(context); count = 0
    if not question.strip() or not context.strip(): raise ValueError('Empty public input cannot succeed')
    def span(ref):
        nonlocal count
        a, b = ref['start'], ref['end']; doc = next((d for d in docs if d['doc_index'] == ref['doc_index']), None)
        if not doc or not doc['start'] <= a < b <= doc['end'] or context[a:b] != ref['quote']:
            raise ValueError('Exact Unicode offset/DOC boundary/quote mismatch')
        count += 1
        if count > 8: raise ValueError('At most eight evidence spans')
        return ref['quote']
    def relation(fact):
        text = span(fact['source_ref']); subject, candidate = fact['subject'], fact['candidate']
        literal = subject + ' ' + RELATIONS[fact['relation']] + ' ' + candidate
        if not re.search(re.escape(literal) + r'(?=\s*(?:[.!?;\n]|$))', text) or not set(terms(subject)) & set(terms(question)):
            raise ValueError('Relation lacks literal subject/cue/candidate/question provenance')
        # A competing value for the same explicit relation is an abstention, not a guessed fact.
        pattern = re.escape(subject + ' ' + RELATIONS[fact['relation']] + ' ') + r'([^.!?\n;]+)'
        values = {m.group(1).strip().casefold() for m in re.finditer(pattern, context)}
        if len(values) != 1 or any(re.search(r'\b(?:or|and)\b', v) for v in values):
            raise ValueError('Ambiguous explicit relation')
    if kind == 'evidence_locate':
        seen = set()
        for ref in output['matches']:
            text = span(ref); key = (ref['start'], ref['end'])
            if key in seen or not set(terms(question)) & set(terms(text)):
                raise ValueError('Duplicate span or missing question-term overlap')
            seen.add(key)
    elif kind == 'explicit_alias':
        for record in output['aliases']:
            text = span(record['source_ref']); a, b = record['alias_pair']; cue = record['cue']
            literal = a + (' (' + b + ')' if cue == 'parenthesized_acronym' else ' ' + cue + ' ' + b)
            if a == b or literal not in text or not set(terms(question)) & set(terms(a + ' ' + b)):
                raise ValueError('Alias requires an explicit public cue, not co-occurrence')
            if cue != 'parenthesized_acronym':
                pattern = re.escape(a + ' ' + cue + ' ') + r'([^.!?\n;]+)'
                targets = {m.group(1).strip() for m in re.finditer(pattern, context)}
                if targets != {b} or any(re.search(r'\b(?:or|and)\b', target) for target in targets):
                    raise ValueError('Ambiguous explicit alias')
            if cue == 'parenthesized_acronym' and (not re.fullmatch('[A-Z]{2,8}', b)
                    or ''.join(w[0].upper() for w in a.split()) != b):
                raise ValueError('Parenthetical string is not a full-name acronym')
    elif kind == 'typed_numeric_compare':
        values = output['values']
        if len({(v['subject'], v['unit'], v['time']) for v in values}) != 1:
            raise ValueError('Numeric comparison mixes subjects, units or periods')
        for value in values:
            text = span(value['span'])
            if (value['subject'] not in text or not set(terms(question)) & set(terms(value['subject']))
                    or not re.search(r'(?<![\w.])' + re.escape(value['value']) + r'\s+' + re.escape(value['unit']) + r'(?!\w)', text)
                    or value['time'] and value['time'] not in text):
                raise ValueError('Numeric value/unit/time lacks literal provenance')
        c = output['comparison']; left, right = c['left'], c['right']
        if left == right or max(left, right) >= len(values): raise ValueError('Invalid comparison indexes')
        a, b = Decimal(values[left]['value']), Decimal(values[right]['value'])
        if c['relation'] != ('less' if a < b else 'greater' if a > b else 'equal'):
            raise ValueError('Incorrect deterministic numeric comparison')
    elif kind == 'multi_doc_join':
        refs = output['evidence_refs']; texts = [span(r) for r in refs]
        if len({r['doc_index'] for r in refs}) < 2: raise ValueError('Join needs distinct DOCs')
        for key in output['entity_keys']:
            if any(key.casefold() not in {m.group(1).casefold() for m in re.finditer(IDENTIFIER, text, re.I)} for text in texts):
                raise ValueError('Join needs an explicit shared identifier, not matching names')
    else:
        facts = output['facts'] if kind == 'explicit_relation' else output['public_support_refs']
        for fact in facts:
            relation(fact)
            if kind == 'source_check' and fact['candidate'] != output['candidate']:
                raise ValueError('Source check candidate differs from supported relation')
    return {'passed': True, 'reason': 'typed_public_operation_checked',
            'scope': 'intermediate local operation; no official QA correctness or recall assertion'}


def boundary_inputs(kind):
    """Synthetic contract probes contain no benchmark answers and are never model material."""
    examples = {
        'evidence_locate': ('Where is Ζήτα?', '[DOC] Ζήτα is here. [DOC] Ζήτα is there.'),
        'explicit_alias': ('What is Ζήτα also known as?', '[DOC] Ζήτα also known as Theta.'),
        'typed_numeric_compare': ('What is the difference for Ζήτα?', '[DOC] Ζήτα 12 kg. [DOC] Ζήτα 18 kg.'),
        'explicit_relation': ('Where was Ζήτα born?', '[DOC] Ζήτα was born in Theta.'),
        'multi_doc_join': ('Which record describes Ζήτα?', '[DOC] Ζήτα identifier: a-42. [DOC] Ζήτα identifier: a-42.'),
        'source_check': ('Where was Ζήτα born?', '[DOC] Ζήτα was born in Theta.'),
    }
    q, c = examples[kind]
    return {'boundary:empty_question': {'question': '', 'context': c},
        'boundary:empty_context': {'question': q, 'context': ''},
        'boundary:unicode_parameter': {'question': q, 'context': c},
        'boundary:ambiguous': {'question': 'Where was Ζήτα born?', 'context': '[DOC] Ζήτα was born in Theta or Delta.'},
        'boundary:unit_conflict': {'question': 'What is the difference for Ζήτα?', 'context': '[DOC] Ζήτα 12 kg. [DOC] Ζήτα 18 miles.'},
        'boundary:offset_guard': {'question': q, 'context': 'αβ\n' + c}}


def checked_result(program, arguments, result, role):
    if role in {'boundary:empty_context', 'boundary:empty_question'}:
        return {'passed': result.get('status') in {'not_found', 'needs_input', 'blocked'}, 'reason': 'empty_input_decline'}
    if role in {'boundary:ambiguous', 'boundary:unit_conflict'}:
        kind = program['entry_constraints']['spec']['kind']
        must_decline = kind in {'explicit_relation', 'source_check'} and role == 'boundary:ambiguous' or kind == 'typed_numeric_compare' and role == 'boundary:unit_conflict'
        if must_decline or result.get('status') != 'ok':
            return {'passed': result.get('status') in {'not_found', 'needs_input', 'blocked'}, 'reason': 'honest_ambiguity_or_unit_fallback'}
    return check(program, arguments, result)


def execute(bank, worker, program, task, directory, *, role, arguments=None, governor=None):
    arguments = inputs(task) if arguments is None else arguments
    identity = digest([program['id'], task.physical_key, role, arguments]); directory = Path(directory)
    path, claim = directory / (identity + '.json'), directory / (identity + '.started.json')
    if path.exists():
        row = json.loads(path.read_text())
        if not receipt_identity(program, row) or row['arguments'] != arguments: raise ValueError('Worker receipt identity changed')
        return row
    if claim.exists(): raise RuntimeError('Unresolved Worker invocation; explicit review required')
    validate_schema_instance(arguments, INPUT)
    if governor: governor.admit_validation(identity)
    save_json(claim, {'id': identity, 'program_digest': program_digest(program), 'task_key': task.physical_key, 'role': role})
    adapter = AnswerAdapter('searchqa', {task.task_id: {}}); adapter.reset(task)
    broker = Broker(adapter, 1, context=TaskContext({})); begin = time.time()
    try:
        result = worker.execute(program, arguments, broker)
        try: validation = checked_result(program, arguments, result, role)
        except (ValueError, TypeError, KeyError, InvalidOperation) as exc: validation = {'passed': False, 'reason': str(exc)}
        row = {'id': identity, 'policy': QUALIFICATION, 'program_id': program['id'], 'program_digest': program_digest(program),
            'environment_hash': digest(program['environment']), 'task_key': task.physical_key, 'task': asdict_task(task),
            'role': role, 'arguments': arguments, 'input_hash': digest(arguments), 'result': result,
            'output_hash': digest(result.get('outputs', {})), 'check': validation, 'start_time': begin, 'end_time': time.time(),
            'tools': broker.events, 'historical_execution': False}
        if not receipt_identity(program, row): raise RuntimeError('R3 requires a real locked Docker receipt')
        save_json(path, row); return row
    finally: adapter.close()


def receipt_identity(program, row):
    try:
        receipt = row['result']['execution_receipt']
        return (row['policy'] == QUALIFICATION and row['program_digest'] == program_digest(program)
            and row['task_key'] == row['task']['physical_key'] and row['task']['split'] == 'train'
            and row['id'] == digest([program['id'], row['task_key'], row['role'], row['arguments']])
            and row['environment_hash'] == digest(program['environment']) and row['input_hash'] == digest(row['arguments'])
            and row['output_hash'] == digest(row['result'].get('outputs', {}))
            and receipt['source_sha256'] == hashlib.sha256(program['source'].encode()).hexdigest()
            and receipt['input_hash'] == digest(row['arguments'])
            and receipt['result_hash'] == digest({k: v for k, v in row['result'].items() if k != 'execution_receipt'})
            and receipt['image_digest'] == IMAGE and receipt['container'].startswith('skillcompiler-')
            and '--network=none' in receipt['command'] and '--read-only' in receipt['command']
            and row['historical_execution'] is False)
    except (ValueError, TypeError, KeyError): return False


def valid_receipt(program, row):
    if not receipt_identity(program, row): return False
    try:
        if row['role'] in {'source', 'transfer'}:
            if row['arguments'] != inputs(PublicTask(**row['task'])): return False
        elif row['role'] in BOUNDARIES:
            if row['arguments'] != boundary_inputs(program['entry_constraints']['spec']['kind'])[row['role']]: return False
        else: return False
        return checked_result(program, row['arguments'], row['result'], row['role']) == row['check'] and row['check']['passed'] is True
    except (ValueError, TypeError, KeyError, InvalidOperation): return False


def qualification(program, validation):
    rows = validation.get('receipts', [])
    if (validation.get('policy') != QUALIFICATION or validation.get('passed') is not True
            or validation.get('program_digest') != program_digest(program) or validation.get('environment_hash') != digest(program['environment'])
            or not rows or len({r.get('id') for r in rows}) != len(rows) or any(not valid_receipt(program, r) for r in rows)):
        return False
    sources = {r['task_key'] for r in rows if r['role'] == 'source'}
    transfers = [r for r in rows if r['role'] == 'transfer']
    source_hashes = {r['input_hash'] for r in rows if r['role'] == 'source'}
    return (len(sources) >= 2 and sources == set(program['entry_constraints']['spec']['source_case_ids'])
        and bool(transfers) and all(r['task_key'] not in sources and r['input_hash'] not in source_hashes for r in transfers)
        and {r['role'] for r in rows if r['role'].startswith('boundary:')} == BOUNDARIES)


def qualify(bank, program, rows):
    good = [r for r in rows if valid_receipt(program, r)]
    validation = {'policy': QUALIFICATION, 'program_digest': program_digest(program),
        'environment_hash': digest(program['environment']), 'passed': True, 'receipts': good,
        'reason': 'all_declared_sources_distinct_train_and_boundaries'}
    if not qualification(program, validation):
        return {'passed': False, 'reason': 'partial_or_not_qualified', 'successful_source_count': len({r['task_key'] for r in good if r['role'] == 'source'})}
    bank.record({'id': 'r3_qualification:' + digest(validation), 'program_id': program['id'], 'task_key': good[0]['task_key'],
        'origin': 'train_test', 'split': 'train', 'outcome': 'positive', 'basis': 'local_check', 'calls': 0, 'validation': validation})
    bank.record_validation(program['id'], validation)
    bank.put('implementation', {'skill_id': program['entry_constraints']['atomic_skill_id'], 'program_id': program['id'],
        'method_version': VERSION, 'qualification_hash': digest(validation)})
    return {'passed': True, 'program_id': program['id'], 'receipt_count': len(good)}


def repair_feedback(previous, bank):
    if not previous or 'worker_feedback' not in previous:
        return deepcopy(previous)
    material = deepcopy(previous)
    program = bank.get(previous['program_id'])
    if not program: raise ValueError('Repair Program identity changed')
    material['original_source'] = program['source']
    feedback = material['worker_feedback']; checks = feedback.pop('checks')
    grouped = {}
    for row in checks:
        key = (row['role'], row['passed'], row['reason'])
        if key not in grouped:
            grouped[key] = {k: row[k] for k in ('role', 'passed', 'reason')}
            grouped[key].update(count=0, example_receipt_id=row['id'])
        grouped[key]['count'] += 1
    feedback.update(check_groups=list(grouped.values()), total_checks=len(checks), checks_hash=digest(checks))
    return material


def builder_sources(sources):
    projected = deepcopy(sources)
    for source in projected:
        for doc in source['documents']:
            quote = source['context'][doc['start']:doc['end']]
            if quote != doc['quote'] or hashlib.sha256(quote.encode()).hexdigest() != doc['quote_sha']:
                raise ValueError('Builder source document identity changed')
            del doc['quote']
    return projected


def build(learner, spec, asset, sources, *, generation=0, previous=None):
    from .prompts import SEARCHQA_R3_BUILDER_PROMPT
    from .program_worker import public_program_abi
    if generation not in {0, 1} or bool(previous) != bool(generation): raise ValueError('Exactly one bounded Builder repair')
    material = {'atomic': asset['searchqa_metadata']['candidate'], 'spec': spec,
        'sources': builder_sources([sources[k] for k in spec['source_case_ids']]),
        'document_quote_contract': 'Each document quote is exactly source.context[start:end] using Unicode indexes; the complete context is retained once.',
        'input_schema': INPUT,
        'output_schema': OUTPUTS[spec['kind']], 'public_program_abi': public_program_abi(),
        'allowed_tools': [], 'output_contract': OUTPUT, 'previous_failure': repair_feedback(previous, learner.system.bank),
        'environment': learner.system.config['program_environment']}
    if asset['searchqa_metadata'].get('legacy_parent'):
        parent = asset['searchqa_metadata']['legacy_parent']
        material['historical_failed_spec_parent'] = {k: deepcopy(parent[k]) for k in
            ('old_spec', 'old_asset', 'old_failures', 'parent_spec_hash', 'old_asset_hash', 'qualification')}
    def validate(value):
        validate_schema_instance(value, BUILD)
        prepare(value['source'], spec, asset, learner.system.config['program_environment'])
        from ..harness.scorers.searchqa import extract_answer
        literals = [n.value for n in ast.walk(ast.parse(value['source'])) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        for key in spec['source_case_ids']:
            answer = extract_answer(sources[key]['original_public_prediction'] or '').strip()
            if len(answer) >= 3 and any(re.search(r'(?<!\w)' + re.escape(answer) + r'(?!\w)', literal, re.I) for literal in literals):
                raise ValueError('Generated Program embeds a historical answer literal')
            if any(len(literal) >= 40 and literal in sources[key]['context'] for literal in literals):
                raise ValueError('Generated Program embeds historical source prose')
    value = learner._receive('r3_build_' + asset['id'] + '_' + str(generation), 'tool_builder', SEARCHQA_R3_BUILDER_PROMPT,
        material, 'submit_searchqa_code', BUILD, material_version=MATERIAL, validator=validate, repair_limit=0,
        decision_purpose='searchqa_builder', job_key=[asset['id'], generation])
    return prepare(value['source'], spec, asset, learner.system.config['program_environment'],
                   parent=previous.get('program_id') if previous else None)
