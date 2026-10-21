"""Compile independent manual rule variants; selection never sees outcomes or IDs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'bank'


def skills(selected):
    catalog = json.loads((ROOT / 'rule_search_v10.json').read_text())
    core = json.loads((ROOT / catalog['core_source']).read_text())['core']
    choices = {v['id']: (d, v) for d in catalog['directions'] for v in d['variants']}
    if len(selected) != len(set(selected)) or set(selected) - choices.keys():
        raise ValueError('Expected unique frozen rule variant IDs')
    families = [choices[k][0]['id'] for k in selected]
    if len(families) != len(set(families)):
        raise ValueError('Choose at most one wording per direction')
    result = [{'title': 'Document evidence', 'goal': 'Extract the requested source span',
        'guidance': core, 'method_version': catalog['version'], 'principle_id': 'core'}]
    # Comparisons take precedence over overlapping entity words. Phone is disjoint.
    priority = {'comparison': 0, 'entity': 1, 'phone': 2}
    for key in sorted(selected, key=lambda k: priority[choices[k][0]['id']]):
        direction, variant = choices[key]
        result.append({'title': direction['id'], 'goal': direction['id'],
            'guidance': variant['text'], 'method_version': catalog['version'], 'principle_id': key,
            'public_intent_trigger': direction['trigger'], 'priority': priority[direction['id']]})
    return result
