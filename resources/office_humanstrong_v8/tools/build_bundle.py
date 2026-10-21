"""Six short principles; four narrow contracts on unchanged audited arithmetic."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3

from atomic_skillgraph.empirical.contracts import digest, program_digest, validate_program

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'office.humanstrong.sparse.v8.20261021'


def build():
    parent = json.loads((ROOT.parent/'office_humanstrong_v7/bank/bank.json').read_text())
    bank = {**deepcopy(parent), 'method_version': VERSION, 'parent_bank_digest': digest(parent),
            'measured_performance': None, 'status': 'development_candidate_not_accepted',
            'source_basis': 'Public method principles and observed V7 interference; no task-specific answers.',
            'default_workflow_id': None, 'workflow': [], 'skill': [], 'program': [], 'implementation': []}
    original = next(p for p in parent['program'] if p['source_file'].endswith('series_metrics.py'))
    common = ['operation', 'unit', 'decimals', 'operand_refs', 'output_percent']
    fields = {'annual_log_growth': ['base_value', 'new_value', 'periods'],
              'cagr': ['base_value', 'new_value', 'periods'],
              'share_difference': ['values', 'absolute'], 'symmetric_difference': ['values']}
    for spec in json.loads((ROOT/'principles.json').read_text()):
        skill = deepcopy(parent['skill'][0]); skill.pop('id')
        skill.update(title=spec['title'], goal=spec['goal'], guidance=spec['guidance'], method_version=VERSION)
        skill['id'] = 'skill_'+digest(skill); bank['skill'].append(skill)
        if 'operation' not in spec: continue
        op = spec['operation']; program = deepcopy(original)
        props = {k: deepcopy(original['input_schema']['properties'][k]) for k in common+fields[op]}
        props['operation'] = {'const': op}
        if op == 'annual_log_growth': props['output_percent'] = {'const': False}
        if 'values' in props:
            count = 4 if op == 'share_difference' else 2
            props['values'].update(minItems=count, maxItems=count,
                description='[part_before,total_before,part_after,total_after]' if count == 4 else '[first,second]')
        required = [k for k in props if k != 'operand_refs']
        program.update(title=spec['goal']+'; returns a numeric answer, not a task-success claim',
            operation=op, method_version=VERSION,
            input_schema={'type': 'object', 'properties': props, 'required': required, 'additionalProperties': False})
        program['id'] = 'program_'+program_digest(program)
        validate_program(program); bank['program'].append(program)
        link = dict(kind='implementation', skill_id=skill['id'], program_id=program['id'],
                    goal=spec['goal'], origin='human_authored', permission='author_trusted', method_version=VERSION)
        link['id'] = 'implementation_'+digest(link); bank['implementation'].append(link)
    destination = ROOT/'bank'; destination.mkdir(exist_ok=True)
    (destination/'bank.json').write_text(json.dumps(bank, ensure_ascii=False, indent=2)+'\n')
    with sqlite3.connect(destination/'bank.sqlite3') as db:
        db.execute('DROP TABLE IF EXISTS assets')
        db.execute('CREATE TABLE assets (id TEXT PRIMARY KEY, kind TEXT NOT NULL, json TEXT NOT NULL)')
        for kind in ('skill', 'program', 'implementation', 'workflow'):
            db.executemany('INSERT INTO assets VALUES (?,?,?)',
                [(a['id'], kind, json.dumps(a, ensure_ascii=False)) for a in bank[kind]])
    print(json.dumps({'digest': digest(bank), 'counts': {k: len(bank[k]) for k in ('skill','program','implementation','workflow')},
                      'guidance_characters': sum(len(s['guidance']) for s in bank['skill'])}))


if __name__ == '__main__': build()
