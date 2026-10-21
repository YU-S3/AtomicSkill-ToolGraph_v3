"""Compile an explicitly selected manual principle set; no task routing or scoring."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def compile_skill(selected):
    catalog = json.loads((ROOT / 'bank/principles_v8.json').read_text())
    rules = catalog['principles']
    ids = [r['id'] for r in rules]
    if len(selected) != len(set(selected)) or set(selected) - set(ids):
        raise ValueError('Expected unique principle IDs from the frozen catalog')
    chosen = [r for r in rules if r['id'] in selected]
    skill = json.loads((ROOT / 'bank/single_node.json').read_text())
    skill.update(title='Document reading: selected short principles',
        goal='Answer from the requested source constituent',
        guidance=catalog['core'] + ''.join('\n- ' + r['text'] for r in chosen),
        method_version=catalog['version'], principle_ids=[r['id'] for r in chosen])
    return skill
