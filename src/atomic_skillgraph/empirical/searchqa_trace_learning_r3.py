"""R3 Train-only source projection, full-schema induction and one bounded repair."""
from copy import deepcopy
import hashlib
import re

from .contracts import PublicTask, digest, object_schema, validate_schema_instance
from .searchqa_guidance import render_procedure, quote_hash
from .searchqa_guidance_r3 import (VERSION, MATERIAL, ATOMIC, FAMILIES, NECESSARY,
    applicability, checked_asset, features)
from .searchqa_trace_learning import public_case as original_public_case
from ..harness.scorers.searchqa import extract_answer

INDUCTION_R3 = object_schema({'decision': {'enum': ['candidate', 'no_change']}, 'candidate': ATOMIC,
    'reason': {'type': 'string', 'maxLength': 400}}, ['decision', 'reason'])


def partition(tasks, exposed):
    keys = {t.physical_key for t in tasks}
    if len(tasks) != 300 or len(keys) != 300 or any(t.split != 'train' for t in tasks) or set(exposed) - keys:
        raise ValueError('R3 partition requires canonical Train300 and real R2 exposure IDs')
    ordered = sorted(keys - set(exposed), key=lambda key: hashlib.sha256((VERSION + key).encode()).hexdigest())
    if len(ordered) < 60: raise ValueError('Insufficient unexposed Train identities')
    probe, transfer = ordered[:30], ordered[30:60]
    induction = sorted(keys - set(probe) - set(transfer), key=lambda key: hashlib.sha256((VERSION + key).encode()).hexdigest())
    return {'schema': VERSION, 'rule': 'SHA256(method_version + physical_key); ID only, no outcomes',
            'r2_exposed': sorted(exposed), 'induction': induction, 'worker_probe': transfer, 'utility_probe': probe}


def public_case(row):
    value = original_public_case(row)
    value.pop('posthoc_located_answer_surface')
    value['public_features'] = features(PublicTask(**row['task']))
    value['feedback_qualification'] = 'correct_answer_advisory' if row['correct'] else 'failed_task_demand'
    value['prediction_qualification'] = 'model_reported_not_executed_evidence'
    value['source_trace_hash'] = row['trace_sha256']
    return value


def learning_units(sources):
    units = []
    # Two deterministic groups per family, including failures whenever public cues qualify.
    for turn in range(2):
        for family in FAMILIES:
            eligible = [s for s in sources.values() if s['public_features']['requirements'][NECESSARY[family]]]
            eligible.sort(key=lambda s: hashlib.sha256((VERSION + family + s['case_id']).encode()).hexdigest())
            failures = [s for s in eligible if s['feedback_qualification'] == 'failed_task_demand']
            successes = [s for s in eligible if s['feedback_qualification'] == 'correct_answer_advisory']
            mixed = []
            for i in range(max(len(failures), len(successes))):
                mixed.extend(pool[i] for pool in (failures, successes) if i < len(pool))
            chosen = mixed[turn * 4:(turn + 1) * 4]
            if len(chosen) < 2: continue
            units.append({'id': 'unit_' + str(len(units) + 1).zfill(2), 'family': family,
                          'case_ids': [s['case_id'] for s in chosen]})
    return units


def validate_candidate(candidate, sources, known_parents=()):
    validate_schema_instance(candidate, ATOMIC)
    ids = candidate['source_case_ids']
    if len(ids) != len(set(ids)) or len(set(ids)) < 2 or set(ids) - set(sources):
        raise ValueError('source_case_ids: require two distinct supplied physical Train sources')
    if set(candidate['parent_ids']) - set(known_parents): raise ValueError('parent_ids: unknown parent')
    if candidate['family'] != candidate['applicability']['family']:
        raise ValueError('family: inconsistent applicability')
    guidance = render_procedure(candidate)
    if len(guidance) > 600: raise ValueError('total_rendered_chars: procedure plus fallback exceeds 600')
    text = candidate['goal'] + ' ' + candidate['operation'] + ' ' + guidance
    if re.search(r'searchqa:|[0-9a-f]{24,}|\b(?:gold|correct_answer|task_id|physical_key)\b', text, re.I):
        raise ValueError('procedure: identity or answer lookup forbidden')
    seen = set()
    for ref in candidate['evidence_refs']:
        if ref['case_id'] not in ids: raise ValueError('evidence_refs.case_id: outside declared sources')
        source = sources[ref['case_id']]
        doc = next((d for d in source['documents'] if d['doc_index'] == ref['doc_index']), None)
        a, b = ref['start'], ref['end']
        if not doc or not doc['start'] <= a < b <= doc['end'] or quote_hash(source['context'][a:b]) != ref['quote_sha']:
            raise ValueError('evidence_refs: DOC boundary/offset/quote_sha mismatch')
        seen.add(ref['case_id'])
    if seen != set(ids): raise ValueError('evidence_refs: each declared source needs a valid exact reference')
    for key in ids:
        source = sources[key]
        if source['historical_operation_count'] != 0 or source['historical_executed_evidence']:
            raise ValueError('Historical answer cannot become an executed operation')
        if not applicability(PublicTask('source', key, source['question'], {'context': source['context']}), candidate['applicability'])[0]:
            raise ValueError('preconditions: required public cue absent on source')
        answer = extract_answer(source['original_public_prediction'] or '').strip()
        if len(answer) >= 3 and re.search(r'(?<!\w)' + re.escape(answer) + r'(?!\w)', text, re.I):
            raise ValueError('procedure: copied model answer instead of a general operation')
        if any(len(p.strip()) >= 40 and p.strip() in text for d in source['documents'] for p in re.split(r'[.!?\n]', d['quote'])):
            raise ValueError('procedure: copied source prose')
    asset = {'method_version': VERSION, 'goal': candidate['goal'], 'guidance': guidance,
        'input_schema': object_schema(), 'output_schema': object_schema(), 'execution_intent': 'guidance_only',
        'result_role': 'intermediate', 'entry_constraints': candidate['applicability'],
        'searchqa_metadata': {'schema': MATERIAL, 'candidate': deepcopy(candidate), 'candidate_hash': digest(candidate),
            'source_consistent': True, 'utility': 'not_tested', 'historical_execution_claim': False,
            'source_roles': {k: sources[k]['feedback_qualification'] for k in ids},
            'source_trace_hashes': {k: sources[k]['source_trace_hash'] for k in ids}}}
    asset['id'] = 'skill_' + digest(asset)
    if not checked_asset(asset): raise ValueError('total_rendered_chars: complete card exceeds 900 or invalid metadata')
    return asset


def full_validator(sources, material, parents=()):
    original_ids = None
    def validate(value):
        nonlocal original_ids
        c = value.get('candidate') if isinstance(value, dict) else None
        c = c if isinstance(c, dict) else {}
        ids = c.get('source_case_ids')
        if original_ids is None and isinstance(ids, list) and all(isinstance(k, str) for k in ids):
            original_ids = deepcopy(ids)
        try:
            validate_schema_instance(value, INDUCTION_R3)
            if value['decision'] == 'no_change':
                if 'candidate' in value: raise ValueError('no_change cannot include candidate')
            else:
                if ids != original_ids: raise ValueError('repair cannot replace the original candidate source IDs')
                validate_candidate(value.get('candidate'), sources, parents)
        except ValueError as exc:
            members = []
            raw_refs = c.get('evidence_refs')
            for key in ids if isinstance(ids, list) else []:
                source = sources.get(key) if isinstance(key, str) else None
                refs = [r for r in (raw_refs if isinstance(raw_refs, list) else []) if isinstance(r, dict) and r.get('case_id') == key]
                valid = bool(source and refs)
                for ref in refs:
                    doc = next((d for d in source['documents'] if d['doc_index'] == ref.get('doc_index')), None) if source else None
                    a, b = ref.get('start'), ref.get('end')
                    valid = valid and bool(doc and type(a) is int and type(b) is int and doc['start'] <= a < b <= doc['end']
                                           and quote_hash(source['context'][a:b]) == ref.get('quote_sha'))
                members.append({'source_id': key, 'reference_consistent': valid})
            exc.repair_material = {'original_proposal': deepcopy(value), 'field_errors': [str(exc)],
                'original_material_hash': digest(material), 'legal_source_ids': sorted(sources),
                'source_member_checks': members, 'sources': list(sources.values()), 'known_parent_ids': list(parents), 'candidate_contract': ATOMIC}
            raise
    return validate


def induce(learner, unit, sources):
    from .prompts import SEARCHQA_R3_INDUCTION_PROMPT
    bound = {k: sources[k] for k in unit['case_ids']}
    material = {'unit': unit, 'sources': list(bound.values()), 'candidate_contract': ATOMIC,
                'known_parent_ids': []}
    proposal = learner._receive('r3_' + unit['id'], 'extractor', SEARCHQA_R3_INDUCTION_PROMPT,
        material, 'submit_searchqa_candidate', INDUCTION_R3, material_version=MATERIAL,
        validator=full_validator(bound, material), repair_limit=1, decision_purpose='searchqa_induction')
    if proposal['decision'] == 'no_change': return proposal
    asset = validate_candidate(proposal['candidate'], bound)
    saved = learner.system.bank.put('skill', asset)
    return {'decision': 'candidate', 'asset_id': saved['id'], 'candidate': proposal['candidate'],
            'source_consistent': True, 'utility': 'not_tested'}


def specification(asset):
    """Typed interfaces are Host contracts, not another invented model response."""
    from .searchqa_program_probe_r3 import INPUT, OUTPUTS
    c = asset['searchqa_metadata']['candidate']
    return {'kind': c['family'], 'goal': c['goal'], 'input_contract': deepcopy(INPUT),
        'output_contract': deepcopy(OUTPUTS[c['family']]), 'source_refs': deepcopy(c['evidence_refs']),
        'source_case_ids': list(c['source_case_ids']), 'preconditions': deepcopy(c['applicability']),
        'abstention': ['needs_input', 'not_found', 'blocked'], 'result_role': 'intermediate',
        'proposal_origin': 'candidate_derived_from_text'}
