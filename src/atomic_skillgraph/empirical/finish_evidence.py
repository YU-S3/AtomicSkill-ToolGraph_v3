"""Materialize only already obtained public results for a tool-free finish."""
from copy import deepcopy
import json

from .contracts import digest


def wire_bytes(prompt, materials):
    from .budget_governor import input_token_bound
    return input_token_bound({'messages': [{'role': 'system', 'content': prompt},
        {'role': 'user', 'content': json.dumps(materials, ensure_ascii=False, allow_nan=False)}],
        'tools': []})


def build_finish_evidence(context, materials, prompt, *, max_bytes=65536, window_chars=12000,
                          allowed_result_ids=None):
    materials = deepcopy(materials)
    included, omitted, unresolved, refs = [], [], [], []
    def discover(value):
        if isinstance(value, dict):
            if isinstance(value.get('result_id'), str):
                refs.append({'result_id': value['result_id'], 'path': value.get('path', [])})
            for child in value.values(): discover(child)
        elif isinstance(value, list):
            for child in value: discover(child)
    # Explicit pending/finished references, then previously read windows, then recent results.
    for key in ('completed_results', 'pending_outputs'): discover(materials.get(key, {}))
    refs.extend(reversed(context.local_reads))
    discover(materials.get('recent', []))
    refs.extend({'result_id': row['result_id'], 'path': []}
                for row in reversed(list(context.memory.values())))
    materials['finish_evidence'] = []
    if wire_bytes(prompt, materials) > max_bytes:
        for key in ('working_memory', 'recent', 'completed_results', 'pending_outputs'):
            if key in materials:
                omitted.append({'material_key': key, 'reason': 'base_input_limit'})
                materials[key] = []
                if wire_bytes(prompt, materials) <= max_bytes: break
    if wire_bytes(prompt, materials) > max_bytes:
        raise ValueError('finish_input_base_exceeds_limit')
    seen = set()
    for ref in refs:
        rid, path = ref['result_id'], ref.get('path', [])
        key = digest([rid, path, ref.get('offset', 0)])
        if key in seen: continue
        seen.add(key)
        if allowed_result_ids is not None and rid not in allowed_result_ids:
            unresolved.append({**ref, 'reason': 'not_in_obtained_results'}); continue
        try: value = context.resolve({'result_id': rid, 'path': path})
        except ValueError:
            unresolved.append(ref); continue
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False)
        offset = ref.get('offset', 0)
        limit = min(ref.get('limit', window_chars), window_chars)
        content = text[offset:offset+limit]
        fragment = {'result_id': rid, 'source_ids': [s for s,r in context.sources.items() if r == rid],
                    'path': path, 'offset': offset, 'window_chars': len(content),
                    'content': content, 'content_hash': digest(content)}
        materials['finish_evidence'].append(fragment)
        while content and wire_bytes(prompt, materials) > max_bytes:
            content = content[:len(content)//2]
            fragment.update(content=content, window_chars=len(content), content_hash=digest(content))
        if not content:
            materials['finish_evidence'].pop()
            omitted.append({**ref, 'reason': 'input_limit'})
        else: included.append({k:v for k,v in fragment.items() if k != 'content'})
    return {'materials': materials, 'included_refs': included, 'omitted_refs': omitted,
            'unresolved_refs': unresolved, 'evidence_digest': digest(materials['finish_evidence']),
            'serialized_input_bytes': wire_bytes(prompt, materials)}


def finalize_text(agent, context, materials, prompt, *, owner_state_version, allowed_result_ids=None):
    evidence = build_finish_evidence(context, materials, prompt, allowed_result_ids=allowed_result_ids)
    answer = agent('runtime', prompt, evidence['materials'], None, None, repair_limit=0,
                   owner_state_version=owner_state_version, decision_purpose='finish_only')
    return answer, {k:v for k,v in evidence.items() if k != 'materials'}
