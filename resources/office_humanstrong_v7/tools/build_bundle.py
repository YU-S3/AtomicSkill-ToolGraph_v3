"""Versioned V7 patch: binding preflight and authored guidance; preserve V6."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]/'src'))
from atomic_skillgraph.empirical.contracts import digest, program_digest, validate_program

VERSION = 'office.humanstrong.programmatic.v7.20261021'


def build():
    parent_root = ROOT.parent/'office_humanstrong_v6'
    parent = json.loads((parent_root/'bank/bank.json').read_text())
    bank = deepcopy(parent)
    bank.update(method_version=VERSION, parent_bank_digest=digest(parent), measured_performance=None,
        status='authored_reference_offline_validation_only',
        source_basis='V6 Val execution failures; authored development Bank, not an unseen evaluation.')
    specs = json.loads((ROOT/'skills.json').read_text())
    remap = {}
    for skill in bank['skill']:
        old = skill.pop('id')
        skill.update(specs[skill['title']], method_version=VERSION)
        skill['id'] = 'skill_'+digest(skill)
        remap[old] = skill['id']
    for program in bank['program']:
        path = (parent_root/program['source_file']).resolve()
        program['source_file'] = '../'+path.relative_to(ROOT.parent).as_posix()
        if path.name != 'table_calculate.py': continue
        old = program['id']
        source = path.read_text()
        marker = '        groups, evidence, units, windows = {}, [], set(), {}'
        assert source.count(marker) == 1
        source = source.replace(marker, "        bindings = validate_bindings(tables, inputs['bindings'])\n"+marker)
        start = source.index("        bindings = inputs['bindings']")
        end = source.index('        options = dict', start)
        source = source[:start]+(
            "        selected = {field: [cell for key in names for cell in groups[key]]\n"
            "                    for field, names in bindings.items()}\n")+source[end:]
        source = (ROOT/'programs/binding_contract.py').read_text()+'\n\n'+source
        (ROOT/'programs/table_calculate.py').write_text(source)
        program.update(source_file='programs/table_calculate.py', source=source, method_version=VERSION)
        props = program['input_schema']['properties']
        props['tables']['items']['properties']['key']['description'] = 'Unique selector ID referenced by bindings; not a row label.'
        props['bindings']['description'] = (
            'Map calculation operands to ordered tables[].key IDs. Every table key must be used; '
            'never use row labels, column names or literal numbers here. '
            'Example: tables=[{key:"base",...},{key:"new",...}], '
            'bindings={from_value:["base"],to_value:["new"]}. Rows stay in tables[].rows.')
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
            asset['id'] = kind+'_'+digest(asset)
    for program in bank['program']: validate_program(program)
    destination = ROOT/'bank'
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
    print(json.dumps({'version': VERSION, 'digest': digest(bank)}))


if __name__ == '__main__': build()
