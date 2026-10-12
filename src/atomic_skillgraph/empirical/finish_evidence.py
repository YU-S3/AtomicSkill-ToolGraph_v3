"""One bounded, tool-free payload containing only already obtained business windows."""
from copy import deepcopy
import json

from .contracts import digest
from .budget_governor import input_token_bound


def wire_bytes(prompt, materials):
    return input_token_bound({'messages': [{'role': 'system', 'content': prompt},
        {'role': 'user', 'content': json.dumps(materials, ensure_ascii=False, allow_nan=False)}], 'tools': []})


def build_finish_evidence(context, materials, prompt, *, max_bytes=65536, window_chars=12000,
                          allowed_result_ids=None, payload_builder=None, output_cap=2048):
    materials = deepcopy(materials)
    included, omitted, unresolved, refs = [], [], [], []
    def bound(value):
        if payload_builder: return input_token_bound(payload_builder(value, output_cap))
        return wire_bytes(prompt, value)
    def reference(value):
        if isinstance(value, dict):
            if value.get('source_ref'):
                refs.append({**value['source_ref'], **{k: value[k] for k in ('offset', 'end') if k in value}})
                return {'ref': value['source_ref']}
            if isinstance(value.get('result_id'), str):
                ref = {'result_id': value['result_id'], 'path': value.get('path', [])}
                refs.append(ref)
                return {'ref': ref}
            return {k: reference(v) for k, v in value.items()}
        if isinstance(value, list): return [reference(v) for v in value]
        return value
    for key in ('completed_results', 'pending_outputs'):
        if materials.get(key): materials[key] = reference(materials[key])
    refs.extend({**w['source_ref'], 'offset': w['offset'], 'end': w['end']}
                for w in reversed(context.sent_windows))
    for w in reversed(context.local_reads):
        refs.append({**w.get('source_ref', {k: w[k] for k in ('result_id', 'path') if k in w}),
                     'offset': w.get('offset', 0),
                     **({'end': w['end']} if 'end' in w else {'limit': w.get('limit', window_chars)})})
    for key in ('recent', 'working_memory'):
        materials[key] = reference(materials.get(key, []))
    refs.extend({'result_id': row['result_id'], 'path': []}
                for row in reversed(list(context.memory.values())))
    # Literal dataflow outputs get an exact local source, rather than a second body.
    for key in ('completed_results', 'pending_outputs'):
        def bind_literals(v, path):
            if isinstance(v, dict):
                if set(v) == {'ref'}: return v
                return {k: bind_literals(x, [*path,k]) for k, x in v.items()}
            ref = context.reference(v, 'finish_handoff:'+json.dumps(path,ensure_ascii=False))
            refs.insert(0, ref)
            return {'ref': ref}
        if materials.get(key): materials[key] = bind_literals(materials[key], [key])
    materials['finish_evidence'] = []
    for key in ('working_memory', 'recent', 'pending_outputs', 'completed_results', 'public_notes'):
        if bound(materials) <= max_bytes: break
        if materials.get(key):
            omitted.append({'material_key': key, 'reason': 'optional_reference_directory_input_limit'})
            materials[key] = []
    if bound(materials) > max_bytes: raise ValueError('finish_unavailable:base_input_exceeds_limit')
    ranges = {}
    for ref in refs:
        rid, path = ref.get('result_id'), ref.get('path', [])
        if allowed_result_ids is not None and rid not in allowed_result_ids:
            unresolved.append({**ref, 'reason': 'not_in_obtained_results'}); continue
        alias = context.window_sources.get(rid)
        if alias and (not path or path[0] == 'data'):
            source = alias['source_ref']
            if len(path) > 1:
                tail = list(path[1:])
                if alias['unit'] == 'array_item_0_based' and type(tail[0]) is int: tail[0] += alias['offset']
                ref = {**ref, 'result_id': source['result_id'], 'path': [*source['path'], *tail]}
            else: ref = {**source, 'offset': alias['offset'], 'end': alias['end']}
        try: leaves = context.business_refs({'result_id': ref['result_id'], 'path': ref.get('path', [])})
        except ValueError:
            unresolved.append({**ref, 'reason': 'invalid_public_reference'}); continue
        for leaf in leaves:
            if not ref.get('path') and leaf['path'][:1] in [['accepted'], ['done'], ['error']]: continue
            offset = ref.get('offset', 0)
            limit = min(ref.get('end', offset + ref.get('limit', window_chars)) - offset, window_chars)
            if limit <= 0: continue
            try:
                if isinstance(context.resolve_public(leaf), list): limit = min(limit, context.settings['result_preview_max_items'])
                window = context.result_window(leaf, offset, limit)
            except ValueError:
                unresolved.append({**leaf, 'reason': 'invalid_public_window'}); continue
            key = digest([leaf, window['unit']])
            segments = [(window['offset'], window['end'])]
            for start, end in ranges.get(key, []):
                segments = [(a, min(b, start)) for a, b in segments if a < start] + [(max(a, end), b) for a, b in segments if b > end]
            for start, end in segments:
                if start >= end: continue
                fragment = context.result_window(leaf, start, end - start)
                materials['finish_evidence'].append(fragment)
                def fits(w):
                    materials['finish_evidence'][-1] = w
                    return bound(materials) <= max_bytes
                fragment = context.fit_window(fragment, fits)
                if not fragment['content']:
                    materials['finish_evidence'].pop()
                    omitted.append({**leaf, 'offset': start, 'end': end, 'reason': 'input_limit',
                                    **({'field_refs': fragment['field_refs']} if 'field_refs' in fragment else {})})
                else:
                    materials['finish_evidence'][-1] = fragment
                    ranges.setdefault(key, []).append((fragment['offset'], fragment['end']))
                    included.append({k: v for k, v in fragment.items() if k != 'content'})
    return {'materials': materials, 'included_refs': included, 'omitted_refs': omitted,
            'unresolved_refs': unresolved, 'evidence_digest': digest(materials['finish_evidence']),
            'serialized_input_bytes': bound(materials), 'input_max_bytes': max_bytes, 'output_cap': output_cap}


def finalize_text(agent, context, materials, prompt, *, owner_state_version, allowed_result_ids=None):
    owner = getattr(agent, '__self__', None)
    checkpoint = getattr(owner, 'checkpoint', None)
    saved = checkpoint.state.get('finish_prepared') if checkpoint else None
    if saved and saved['owner_state_version'] == owner_state_version:
        evidence = saved['evidence']
    elif owner and hasattr(owner, 'prepare_finish'):
        evidence = owner.prepare_finish(materials, prompt, allowed_result_ids=allowed_result_ids)
    else: evidence = build_finish_evidence(context, materials, prompt, allowed_result_ids=allowed_result_ids)
    if checkpoint and (not saved or saved['owner_state_version'] != owner_state_version):
        checkpoint.advance(checkpoint.state['stage'], finish_prepared={
            'owner_state_version':owner_state_version, 'evidence':evidence})
    answer = agent('runtime', prompt, evidence['materials'], None, None, repair_limit=0,
                   owner_state_version=owner_state_version, decision_purpose='finish_only')
    return answer, {k: v for k, v in evidence.items() if k != 'materials'}
