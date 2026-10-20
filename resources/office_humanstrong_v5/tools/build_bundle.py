"""Reproducible V5 assets. V4 remains immutable and independently importable."""
from copy import deepcopy
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]/'src'))
from atomic_skillgraph.empirical.contracts import digest, program_digest, object_schema, validate_program

VERSION = 'office.humanstrong.programmatic.v5.20261020'


def build():
    parent = json.loads((ROOT.parent/'office_humanstrong_v4/bank/bank.json').read_text())
    bank = deepcopy(parent)
    bank.update(method_version=VERSION, parent_bank_digest=digest(parent), measured_performance=None,
        status='authored_reference_offline_validation_only',
        source_basis='Public failure traces exposed during V4 Test review; not an unseen-Test or naturally learned Bank.')
    changes = json.loads((ROOT/'guidance_changes.json').read_text())
    remap = {}
    for skill in bank['skill']:
        if skill['title'] not in changes: continue
        old = skill.pop('id')
        skill.update(guidance=changes[skill['title']], method_version=VERSION)
        skill['id'] = 'skill_'+digest(skill); remap[old] = skill['id']
    for program in bank['program']:
        path = (ROOT.parent/'office_humanstrong_v4'/program['source_file']).resolve()
        program['source_file'] = '../'+str(path.relative_to(ROOT.parent.resolve())).replace('\\', '/')
    text = {'type': 'string', 'minLength': 1}
    ref = object_schema({'result_id': text, 'path': {'type': 'array', 'items': {'type': ['string', 'integer']}}}, ['result_id', 'path'])
    display = object_schema({'style': {'enum': ['number', 'percent', 'unit_suffix']},
        'decimals': {'type': 'integer', 'minimum': 0, 'maximum': 18},
        'scale': {'enum': ['as_is', 'fraction_to_percent']}, 'suffix': text}, ['style', 'decimals', 'scale'])
    extractor = next(p for p in bank['program'] if p['source_file'].endswith('/table_extract.py'))
    calculator = next(p for p in bank['program'] if p['source_file'].endswith('/table_calculate.py'))
    additions = {'header_path': {'type': 'array', 'items': text, 'minItems': 1}, 'scope_quote': text}
    extractor['input_schema']['properties'].update(additions)
    extractor['output_schema']['properties']['cells']['items']['properties'].update(
        matched_row=text, header=text, column_path={'type': 'array', 'items': text},
        scope_quote={'type': ['string', 'null']})
    table_schema = calculator['input_schema']['properties']['tables']['items']
    table_schema['properties'].update(deepcopy(additions))
    schema = calculator['input_schema']
    schema['properties'].pop('answer_unit')
    schema['required'].remove('answer_unit')
    schema['properties']['format'] = display
    schema['required'].append('format')
    options = schema['properties']['calculation']
    options['required'].remove('decimals')
    options['properties'].update(ddof={'enum': [0, 1]}, observation={'type': ['number', 'string']},
                                 transform={'enum': ['none', 'first_difference']})
    options['properties']['method']['enum'].append('statistics')
    options['properties']['operation']['enum'] += ['coefficient_variation', 'z_score', 'mean_spread']
    # Arithmetic retains full precision; display rounding occurs only once.
    options['properties'].pop('decimals')
    calculator['output_schema']['properties']['format'] = {'type': 'object'}
    calculator['output_schema']['required'].append('format')
    metric = next(p for p in parent['program'] if p['source_file'].endswith('/series_metrics.py'))
    rules = {}
    for branch in metric['input_schema']['oneOf']:
        operation = branch['properties']['operation']
        for name in operation.get('enum', [operation.get('const')]):
            rules[name] = {'required': branch['required'], 'allowed': branch['propertyNames']['enum']}
    parts = ['METRIC_RULES = '+repr(rules)]
    for path, name in [(ROOT/'programs/table_extract.py', 'extract_table'),
                       (ROOT.parent/'office_humanstrong_v2/programs/series_metrics.py', 'calculate_metrics'),
                       (ROOT.parent/'office_humanstrong_v3/programs/polynomial_forecast.py', 'calculate_polynomial')]:
        parts.append(path.read_text().replace('def run(', f'def {name}(', 1))
    parts += [(ROOT/'programs'/name).read_text() for name in ('answer_format.py', 'statistics.py', 'table_calculate_entry.py')]
    (ROOT/'programs/table_calculate.py').write_text('\n\n'.join(parts))
    extractor['title'] = 'table_extract: literal rows with validated leaf headers; ambiguous hierarchy returns repair feedback'
    calculator['title'] = 'table_calculate: source rows → exact named arithmetic → explicit scalar format → final submission'
    for program, filename in [(extractor, 'table_extract.py'), (calculator, 'table_calculate.py')]:
        old = program['id']
        program.update(source_file='programs/'+filename, source=(ROOT/'programs'/filename).read_text(), method_version=VERSION)
        program['id'] = 'program_'+program_digest(program); remap[old] = program['id']
    item = object_schema({'value_ref': ref, 'format': display}, ['value_ref', 'format'])
    for kind, properties, title in [
        ('computed_scalar', {'value_ref': ref, 'format': display}, 'submit a numeric result leaf with explicit number/percent/unit format'),
        ('computed_list', {'items': {'type': 'array', 'items': item, 'minItems': 1, 'maxItems': 32},
                          'layout': {'enum': ['brackets', 'comma', 'lines']}}, 'submit ordered numeric result leaves with per-field formats')]:
        code = (ROOT/'programs/answer_format.py').read_text()+'\n\n'+(ROOT/f'programs/{kind}_entry.py').read_text()
        (ROOT/f'programs/{kind}.py').write_text(code)
        program = deepcopy(calculator)
        properties['submit_answer'] = {'const': True}
        program.update(title=kind+': '+title, source_file=f'programs/{kind}.py', source=code,
            input_schema=object_schema(properties, list(properties)),
            output_schema=object_schema({'answer': {'type': 'string'}, 'evidence': {'type': 'array'},
                                         'proof_scope': {'type': 'string'}}, ['answer', 'evidence', 'proof_scope']))
        program['id'] = 'program_'+program_digest(program)
        bank['program'].append(program)
        link = {'kind': 'implementation', 'skill_id': next(s['id'] for s in bank['skill'] if s['title'] == 'OA-Commit'),
            'program_id': program['id'], 'goal': title, 'origin': 'human_authored',
            'permission': 'author_trusted', 'method_version': VERSION}
        link['id'] = 'implementation_'+digest(link)
        bank['implementation'].append(link)
    def replace(value):
        if isinstance(value, dict): return {k: replace(v) for k, v in value.items()}
        if isinstance(value, list): return [replace(v) for v in value]
        return remap.get(value, value) if isinstance(value, str) else value
    for kind in ('implementation', 'workflow'):
        bank[kind] = [replace(asset) for asset in bank[kind]]
        for asset in bank[kind]:
            asset.pop('id'); asset['method_version'] = VERSION; asset['id'] = kind+'_'+digest(asset)
    for program in bank['program']: validate_program(program)
    destination = ROOT/'bank'; destination.mkdir(exist_ok=True)
    (destination/'bank.json').write_text(json.dumps(bank, ensure_ascii=False, indent=2)+'\n')
    with sqlite3.connect(':memory:') as db:
        db.execute('CREATE TABLE assets (id TEXT PRIMARY KEY, kind TEXT NOT NULL, json TEXT NOT NULL)')
        for kind in ('skill', 'program', 'implementation', 'workflow'):
            db.executemany('INSERT INTO assets VALUES (?,?,?)',
                [(a['id'], kind, json.dumps(a, ensure_ascii=False)) for a in bank[kind]])
        db.commit()
        # A fresh database avoids SQLite's existing-file change counter varying by rebuild.
        with NamedTemporaryFile(dir=destination, prefix='bank_', suffix='.tmp', delete=False) as temp:
            temporary = Path(temp.name)
        try:
            with closing(sqlite3.connect(temporary)) as target: db.backup(target)
            temporary.replace(destination/'bank.sqlite3')
        finally: temporary.unlink(missing_ok=True)
    print(json.dumps({'version': VERSION, 'digest': digest(bank),
                      'counts': {kind: len(bank[kind]) for kind in ('skill', 'program', 'implementation', 'workflow')}}))


if __name__ == '__main__': build()
