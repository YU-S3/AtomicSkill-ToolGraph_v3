"""Build the versioned reference Bank; V5 assets remain byte-for-byte unchanged."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]/'src'))
from atomic_skillgraph.empirical.contracts import digest, program_digest, validate_program

VERSION = 'office.humanstrong.programmatic.v6.20261021'


def build():
    parent_root = ROOT.parent/'office_humanstrong_v5'
    parent = json.loads((parent_root/'bank/bank.json').read_text())
    bank = deepcopy(parent)
    bank.update(method_version=VERSION, parent_bank_digest=digest(parent),
        measured_performance=None, status='authored_reference_offline_validation_only',
        source_basis='V5 public execution failures reviewed; authored development Bank, not unseen-Test evidence.')
    specs = json.loads((ROOT/'skills.json').read_text())
    by_name = {s['title']: s for s in bank['skill']}
    bank['skill'], remap = [], {}
    for name, spec in specs.items():
        old = by_name.get(name, by_name['OA-Advanced'])
        skill = {**deepcopy(old), **spec, 'title': name, 'method_version': VERSION}
        skill.pop('id')
        skill['id'] = 'skill_'+digest(skill)
        bank['skill'].append(skill)
        if name in by_name: remap[old['id']] = skill['id']
    remap[by_name['OA-Advanced']['id']] = next(s['id'] for s in bank['skill'] if s['title'] == 'OA-Polynomial')
    old_extract = (parent_root/'programs/table_extract.py').read_text()
    new_extract = (ROOT/'programs/table_extract.py').read_text()
    combined = (parent_root/'programs/table_calculate.py').read_text()
    old_part = old_extract.replace('def run(', 'def extract_table(', 1)
    assert combined.count(old_part) == 1
    (ROOT/'programs/table_calculate.py').write_text(
        combined.replace(old_part, new_extract.replace('def run(', 'def extract_table(', 1)))
    for program in bank['program']:
        path = (parent_root/program['source_file']).resolve()
        program['source_file'] = '../'+path.relative_to(ROOT.parent).as_posix()
        name = path.name
        if name not in ('table_extract.py', 'table_calculate.py'): continue
        old = program['id']
        program.update(source_file='programs/'+name, source=(ROOT/'programs'/name).read_text(),
                       method_version=VERSION)
        if name == 'table_calculate.py':
            # Expose method/operation constraints where the model actually sees the interface.
            calc = program['input_schema']['properties']['calculation']
            calc['description'] = 'Choose exactly one named method. Bind operands separately; never guess option names.'
            calc['oneOf'] = [
                {'properties': {'method': {'const': 'series_metrics'}, 'operation': {'enum': [
                    op for op in calc['properties']['operation']['enum']
                    if op not in ('predict', 'absolute_error', 'coefficient_variation', 'z_score', 'mean_spread')]}},
                 'not': {'anyOf': [{'required': [key]} for key in ('transform', 'ddof', 'observation',
                    'degree', 'first_x', 'x_step', 'prediction_x')]}},
                {'properties': {'method': {'const': 'statistics'}, 'operation': {'enum': [
                    'coefficient_variation', 'z_score', 'mean_spread']}},
                 'required': ['transform'],
                 'allOf': [
                     {'if': {'properties': {'operation': {'enum': ['coefficient_variation', 'z_score']}}},
                      'then': {'required': ['ddof']}},
                     {'if': {'properties': {'operation': {'const': 'z_score'}}},
                      'then': {'required': ['observation']}}]},
                {'properties': {'method': {'const': 'polynomial_forecast'},
                                'operation': {'enum': ['predict', 'absolute_error']}},
                 'required': ['degree', 'prediction_x']}
            ]
        program['id'] = 'program_'+program_digest(program)
        remap[old] = program['id']
    def replace(value):
        if isinstance(value, dict): return {k: replace(v) for k, v in value.items()}
        if isinstance(value, list): return [replace(v) for v in value]
        return remap.get(value, value) if isinstance(value, str) else value
    for kind in ('implementation', 'workflow'):
        bank[kind] = [replace(asset) for asset in bank[kind]]
        for asset in bank[kind]:
            asset.pop('id')
            asset['method_version'] = VERSION
            if kind == 'implementation':
                program = next(p for p in bank['program'] if p['id'] == asset['program_id'])
                asset['goal'] = program['title']
            asset['id'] = kind+'_'+digest(asset)
    for program in bank['program']: validate_program(program)
    destination = ROOT/'bank'
    destination.mkdir(exist_ok=True)
    (destination/'bank.json').write_text(json.dumps(bank, ensure_ascii=False, indent=2)+'\n')
    with NamedTemporaryFile(dir=destination, suffix='.tmp', delete=False) as temp:
        temporary = Path(temp.name)
    try:
        with sqlite3.connect(temporary) as db:
            db.execute('CREATE TABLE assets (id TEXT PRIMARY KEY, kind TEXT NOT NULL, json TEXT NOT NULL)')
            for kind in ('skill', 'program', 'implementation', 'workflow'):
                db.executemany('INSERT INTO assets VALUES (?,?,?)',
                    [(a['id'], kind, json.dumps(a, ensure_ascii=False)) for a in bank[kind]])
        temporary.replace(destination/'bank.sqlite3')
    finally: temporary.unlink(missing_ok=True)
    print(json.dumps({'version': VERSION, 'digest': digest(bank),
        'counts': {kind: len(bank[kind]) for kind in ('skill', 'program', 'implementation', 'workflow')}}))


if __name__ == '__main__': build()
