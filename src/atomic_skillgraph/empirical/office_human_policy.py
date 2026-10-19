"""Public evidence routing for the isolated authored Office Bank, not learned Ours."""
from copy import deepcopy
import json
import re

from .contracts import digest, object_schema
from .budget_governor import input_token_bound
from .public_results import TEXT_PAGE, TEXT_CONTEXT, PROGRAM_EXECUTION, FILE_COMPUTE

VERSION = 'office.human.phase.v3.20261017'
TEXT = {'type': 'string', 'maxLength': 512}
REF = object_schema({'result_id': {'type': 'string'}, 'path': {'type': 'array',
    'items': {'type': ['string', 'integer']}}}, ['result_id', 'path'])
OPERAND = object_schema({'name': TEXT, 'value': {'type': ['string', 'number']},
    'period': TEXT, 'unit': TEXT, 'source_ref': REF, 'quote': {'type': 'string', 'maxLength': 2048}},
    ['name', 'value', 'period', 'unit', 'source_ref', 'quote'])
STATE = object_schema({'operands': {'type': 'array', 'items': OPERAND, 'maxItems': 40},
    'answer_candidate': object_schema({'value': {'type': ['string', 'number']},
        'output_unit': TEXT, 'operation': TEXT, 'result_ref': REF, 'covers_task': {'type': 'boolean'}},
        ['value', 'output_unit', 'operation', 'result_ref', 'covers_task'])}, [])
NOTE = ('Optional evidence_state accompanies the same runtime_step (no separate reporting call). '
    'Persist operands with exact quotes and public source_ref to acquired read/read_context text. '
    'For a computed candidate supply its actual result_ref, value, output_unit, operation, covers_task. '
    'Use period/unit="not_applicable" only when actually inapplicable. Host checks reference integrity, '
    'not whether your interpretation is correct. Once the full target is supported, submit now with '
    'runtime_step(action="finish", answer=<requested payload>). Never invent evidence to change phase.')


def numerical_match(value, text):
    text = text.replace(',', '')
    try:
        from decimal import Decimal, InvalidOperation
        number = Decimal(str(value).replace(',', ''))
        return any(Decimal(s) == number for s in re.findall(r'(?<![\w.])[+-]?\d+(?:\.\d+)?(?![\w.])', text))
    except (ValueError, InvalidOperation):
        return str(value) in text


def validate_state(context, reported, previous=None):
    """No reference integrity check certifies task semantics or changes consumption flags."""
    state = deepcopy(previous or {'operands': [], 'answer_candidate': None})
    errors = []
    if 'operands' in reported:
        valid = []
        for row in reported['operands']:
            try:
                ref = row['source_ref']
                if context.origin(ref['result_id']) not in {TEXT_PAGE, TEXT_CONTEXT}:
                    raise ValueError('operand requires acquired source text, not a search hit/model literal')
                text = context.resolve_public(ref)
                if not isinstance(text, str) or not row['quote'] or row['quote'] not in text:
                    raise ValueError('operand quote not found in the referenced text')
                if not numerical_match(row['value'], row['quote']):
                    raise ValueError('operand value is absent from its quote')
                if not row['name'] or not row['period'] or not row['unit']:
                    raise ValueError('operand name, period and unit are required')
                valid.append(deepcopy(row))
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(str(exc))
        state['operands'] = valid
        state['answer_candidate'] = None  # changed bindings invalidate an earlier candidate
    if 'answer_candidate' in reported:
        candidate = deepcopy(reported['answer_candidate'])
        candidate.update(ready=False, proof_scope='reference_integrity_only',
                         operand_refs=[x['source_ref'] for x in state['operands']])
        try:
            ref = candidate['result_ref']
            if context.origin(ref['result_id']) not in {PROGRAM_EXECUTION, FILE_COMPUTE}:
                raise ValueError('candidate must reference an actual computation result')
            actual = context.resolve_public(ref)
            if str(actual) != str(candidate['value']):
                raise ValueError('candidate differs from referenced output')
            if not candidate['output_unit'] or not candidate['operation'] or not state['operands']:
                raise ValueError('candidate requires operation, output unit and source-bound operands')
            producer=next((r for r in context.memory.values() if r['result_id']==ref['result_id']), {})
            if context.origin(ref['result_id'])==PROGRAM_EXECUTION:
                bound=producer.get('arguments', {})
                keys={'values','other_values','base_value','new_value','numerator','denominator','from_value','to_value'}
                relevant={k:v for k,v in bound.items() if k in keys}
                if bound.get('operation') != candidate['operation']:raise ValueError('candidate operation differs from executed operation')
            else:
                computed=context.results[ref['result_id']].get('data', {}).get('computed', {})
                bound=computed.get('operand_values', {}) if isinstance(computed,dict) else {}
                relevant=bound
            def names(value, prefix=''):
                if isinstance(value,dict):return {n for k,v in value.items() for n in names(v,k if not prefix else prefix+'.'+k)}
                if isinstance(value,list):return {n for i,v in enumerate(value) for n in names(v,prefix+'.'+str(i))}
                return {prefix}
            if not relevant or not names(relevant)<=set(r['name'] for r in state['operands']):
                raise ValueError('not all executed operands have source bindings')
            for operand in state['operands']:
                current=bound
                for part in operand['name'].split('.'):
                    current=current[int(part)] if isinstance(current,list) and part.isdigit() else current[part]
                if str(current)!=str(operand['value']):raise ValueError('candidate operand does not match executed computation input')
            candidate['ready'] = candidate['covers_task'] is True
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            errors.append(str(exc))
        state['answer_candidate'] = candidate
    state['binding_feedback'] = errors
    return state


def acquired(context):
    if context is None:
        return []
    result = []
    for row in context.memory.values():
        rid = row['result_id']
        try:
            public = context.public_result(context.results[rid], origin=context.origin(rid))
        except (KeyError, ValueError):
            continue
        if row['name'] in {'read', 'read_context'} and public.get('accepted'):
            text = (public.get('data') or {}).get('text', '')
            if text:
                result.append({'result_id': rid, 'text': text})
    return result


def progress(context, state, remaining=48):
    pages = acquired(context)
    table = any(len(re.findall(r'^.*\|.*\|.*$', p['text'], re.M)) >= 2 for p in pages)
    cells = False
    if context:
        cells = any(isinstance(r, dict) and (r.get('outputs') or {}).get('table_header_bound') is True
                    and (r.get('outputs') or {}).get('target_periods_complete') is True
                    and (r.get('outputs') or {}).get('values') for r in context.results.values())
    candidate = state.get('answer_candidate') or {}
    phase = ('finalize' if candidate.get('ready') or remaining <= 3 else
             'compute' if state.get('operands') or cells else 'extract' if pages else 'locate')
    return {'phase': phase, 'acquired_pages': len(pages), 'table_window_available': table,
            'bound_operands': len(state.get('operands', [])), 'remaining_calls': remaining,
            'candidate_ready': bool(candidate.get('ready'))}


def cards(phase):
    return {'locate': ['OA-Target', 'OA-Locate'], 'extract': ['OA-Target', 'OA-Align'],
            'compute': ['OA-Align', 'OA-Compute'], 'finalize': ['OA-Commit']}[phase]


def finish_evidence(context, materials, prompt, *, max_bytes, output_cap, payload_builder,
                    state, policy, allowed_result_ids=None):
    """Fit the exact native wire payload; acquired source windows remain traceable."""
    candidate = state.get('answer_candidate')
    value = {'original_goal': materials.get('original_goal', ''), 'finish_evidence': [],
             'answer_candidate': deepcopy(candidate), 'authored_commit_policy': policy,
             'candidate_note': 'Reference integrity is not task/semantic certification. Match the requested target.'}
    from . import evidence_handoff
    if evidence_handoff.enabled(context.settings):
        outputs,required=evidence_handoff.output_material(materials)
        value.update(outputs)
        required.extend({'ref':r['source_ref'],'quote':r['quote']} for r in state.get('operands',[]))
        if candidate and candidate.get('ready'):required.insert(0,{'ref':candidate['result_ref']})
        return evidence_handoff.prepare(context,value,max_bytes=max_bytes,output_cap=output_cap,
            payload_builder=payload_builder,priority_refs=required,allowed_result_ids=allowed_result_ids)
    bound = lambda v: input_token_bound(payload_builder(v, output_cap))
    omitted, unresolved, included, refs = [], [], [], []
    # Directory/history summaries are represented by actual cited windows below.
    for key in ('working_memory', 'recent', 'pending_outputs', 'completed_results', 'public_notes'):
        if materials.get(key):omitted.append({'material_key': key, 'reason': 'replaced_by_ranked_public_windows'})
    if candidate:
        refs.append((0, candidate.get('result_ref'), None))
    for row in state.get('operands', []):
        refs.append((1, row['source_ref'], row['quote']))
    from .finish_evidence import source_text_refs
    refs.extend((3, ref, None) for ref in source_text_refs(context))
    for row in reversed(list(context.memory.values())):
        rid = row['result_id']
        try:
            leaves = context.business_refs({'result_id': rid, 'path': []})
        except ValueError:
            continue
        for ref in leaves:
            path = ref['path']
            if context.origin(rid) in {PROGRAM_EXECUTION, FILE_COMPUTE} and path[:1] in [['outputs'], ['data']]:
                rank = 2 if path[-1:] in [['answer'], ['value'], ['formula'], ['operand_refs'], ['unit'], ['direction']] else 5
            elif row['name'] in {'read', 'read_context'} and path == ['data', 'text']:
                rank = 3
            else:
                rank = 6
            refs.append((rank, ref, None))
    if bound(value) > max_bytes:
        omitted.append({'material_key': 'answer_candidate', 'reason': 'candidate_metadata_input_limit'})
        value['answer_candidate'] = None
    if bound(value) > max_bytes:
        raise ValueError('finish_unavailable:base_input_exceeds_limit')
    seen = set()
    for rank, ref, quote in sorted(refs, key=lambda r: r[0]):
        if not ref or digest([ref, quote]) in seen:
            continue
        seen.add(digest([ref, quote]))
        if allowed_result_ids is not None and ref['result_id'] not in allowed_result_ids:
            unresolved.append({**ref, 'reason': 'not_in_obtained_results'});continue
        try:
            raw = context.resolve_public(ref)
            if raw is None or isinstance(raw, dict) or raw=='' or raw==[]:continue
            offset = max(0, raw.find(quote)-600) if quote and isinstance(raw, str) else 0
            limit = context.settings['result_preview_max_items'] if isinstance(raw, list) else min(
                context.settings['result_read_window_chars'], max(3000, len(quote or '')+1200))
            window = context.result_window(ref, offset, limit)
        except (KeyError, TypeError, ValueError):
            unresolved.append({**ref, 'reason': 'invalid_public_reference'});continue
        value['finish_evidence'].append(window)
        def fits(w):
            value['finish_evidence'][-1] = w
            return bound(value) <= max_bytes
        fitted = context.fit_window(window, fits)
        if not fitted['content']:
            value['finish_evidence'].pop()
            omitted.append({**ref, 'offset': offset, 'end': window['end'], 'reason': 'input_limit'})
        else:
            value['finish_evidence'][-1] = fitted
            included.append({**{k:v for k,v in fitted.items() if k!='content'}, 'priority': rank})
            if fitted['end'] < fitted['size']:
                omitted.append({**ref, 'offset': fitted['end'], 'end': fitted['size'], 'reason': 'window_or_input_limit'})
    return {'materials': value, 'included_refs': included, 'omitted_refs': omitted,
            'unresolved_refs': unresolved, 'evidence_digest': digest(value['finish_evidence']),
            'serialized_input_bytes': bound(value), 'input_max_bytes': max_bytes,
            'output_cap': output_cap, 'projection_version': VERSION}
