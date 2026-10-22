"""Small public-intent clauses and an explicit, bounded submission contract."""
from copy import deepcopy
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
KINDS = ['verbatim', 'phone', 'page', 'count']


def skills():
    catalog = json.loads((ROOT / 'bank/typed_v9.json').read_text())
    return [{'title': 'Document evidence', 'goal': 'Extract the requested source span',
        'guidance': catalog['core'], 'method_version': catalog['version'], 'principle_id': 'core'},
        *[{'title': rule['id'], 'goal': rule['id'], 'guidance': rule['text'],
           'method_version': catalog['version'], 'principle_id': rule['id'],
           'public_intent_trigger': rule['trigger'], 'priority': index}
          for index, rule in enumerate(catalog['principles'])]]


def select(assets, query):
    core = next(a for a in assets if a['principle_id'] == 'core')
    clauses = sorted((a for a in assets if a['principle_id'] != 'core'), key=lambda a: a['priority'])
    chosen = next((a for a in clauses if re.search(a['public_intent_trigger'], query, re.I)), None)
    return [core] + ([chosen] if chosen else [])


def program(environment, method, *, percentage=False):
    from atomic_skillgraph.empirical.contracts import program_digest
    filename = 'typed_submit_percent.py' if percentage else 'typed_submit.py'
    kinds = KINDS[1:] + (['percentage'] if percentage else [])
    asset = {'title': 'Submit an explicitly typed value', 'entry': 'run',
        'source_file': 'programs/' + filename, 'source': (ROOT / 'programs' / filename).read_text(),
        'environment': environment, 'method_version': method, 'allowed_tools': [], 'result_role': 'final_answer',
        'origin': 'human_authored_unqualified',
        'input_schema': {'type': 'object', 'properties': {'kind': {'enum': kinds},
            'value': {'type': 'string', 'minLength': 1, 'maxLength': 2048}},
            'required': ['kind', 'value'], 'additionalProperties': False},
        'output_schema': {'type': 'object', 'properties': {'answer': {'type': 'string', 'minLength': 1},
            'original': {'type': 'string'}, 'kind': {'enum': kinds}, 'format_applied': {'type': 'boolean'}},
            'required': ['answer', 'original', 'kind', 'format_applied'], 'additionalProperties': False}}
    asset['id'] = 'program_' + program_digest(asset)
    return asset


def contract(schema, *, can_inspect, percentage=False):
    schema = deepcopy(schema)
    for key in ('evidence_text', 'answer_kind'):
        schema['properties'].pop(key, None)
    schema['properties']['answer_kind'] = {'enum': KINDS + (['percentage'] if percentage else []),
        'description': 'verbatim for ordinary text, amounts, dates and identifiers; phone for a telephone number; page for a page number; count for the selected numerical count. Never infer a count from an ordinal.'}
    if percentage:
        schema['properties']['answer_kind']['description'] += ' percentage only for a selected value with one printed numeric percentage; preserve its unit and bounds.'
    schema['allOf'][1]['then']['required'] = ['answer', 'answer_kind', 'evidence_sources']
    if not can_inspect:
        schema['properties']['action']['enum'] = ['finish']
        schema['properties'].pop('tile', None)
        schema['allOf'] = schema['allOf'][1:]
    return schema


def instructions(can_inspect, *, percentage=False):
    return ('Use one answer_step call. For finish, choose the answer and its type; '
        'only the tool answer is submitted, with no later rewrite. '
        'phone removes spaces between digits; page removes paired dash ornaments; '
        'count removes N= from a number. Other text is unchanged. '
        'Use verbatim for all other fields. Select evidence_sources from the supplied handles. '
        'The original page is page; crop_N is a verified crop, row_N its row context. '
        'Formatting does not verify your reading. '
        + ('percentage removes spaces between a digit and %; it preserves all other characters and never converts units or verifies the value. ' if percentage else '')
        + ('If an uncertain character affects the answer, inspect its tile: columns A/B/C, rows 1/2/3/4. '
           'Match the detail to the original label and returned row. One inspection leaves room for typed submission. '
           if can_inspect else 'Answer from the images already supplied; no inspection remains. '))
