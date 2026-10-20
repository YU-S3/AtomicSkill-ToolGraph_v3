"""Conditional authored guidance and on-demand contracts on the native executor."""
from contextlib import contextmanager
from copy import deepcopy
from decimal import Decimal, InvalidOperation
import importlib.util
from pathlib import Path
import re

path = Path(__file__).resolve().parents[2]/'office_humanstrong_v5/integration/frontier.py'
spec = importlib.util.spec_from_file_location('office_v6_parent', path)
v5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v5)
base = v5.base
VERSION = 'office.humanstrong.programmatic.v6.20261021'
base.VERSION = VERSION
KINDS, native_import, input_protocol = base.KINDS, base.native_import, base.input_protocol
ANSWER_CONTRACT = v5.ANSWER_CONTRACT
base.selection.GLOBAL_RULE = (
    'Use supplied pages first; retrieve a specific missing fact only. Compute once the necessary '
    'operands, scope and periods are available. Native Python and Programs are alternatives; '
    'do not repeat a successful calculation to satisfy a phase. Program success verifies the named '
    'operation, not question semantics. Submit once all requested results are supported.')
base.selection.FINISH_ONLY_RULE = ANSWER_CONTRACT+' Preserve supported executed results; do not invent missing operands.'

SPECIAL = {
    'OA-Periods': r'\b(year|month|quarter|period|annual|weekly|daily|between|from|series|forecast)',
    'OA-Fiscal': r'\b(fiscal|calendar|fy|cy)\b|transition quarter',
    'OA-Treasury': r'treasury|debt|securit|tender|bid|matur|refunding|gold|silver|reserve|interest',
    'OA-Revision': r'revis|report|bulletin|published|as of|recorded|preliminary|adjusted|current dollar',
    'OA-Correlation': r'correlat|regress|ols|slope',
    'OA-Polynomial': r'polynomial|quadratic|cubic',
    'OA-Dispersion': r'variance|standard deviation|coefficient of variation|z.score|gini',
}


def work_status(context):
    from atomic_skillgraph.empirical.public_results import FILE_COMPUTE, PROGRAM_EXECUTION
    computed, failures = [], {}
    # Inputs/diagnostics may be numeric but are not proposed answer leaves.
    excluded = {'input', 'inputs', 'operand', 'operands', 'parameters', 'source', 'sources',
                'evidence', 'values', 'other_values', 'observations', 'x', 'y', 'count', 'n',
                'ddof', 'degree', 'periods', 'decimals', 'unit', 'source_line', 'start', 'end'}
    last_computed = -1
    for index, (rid, raw) in enumerate(context.results.items()):
        origin = context.origin(rid)
        if origin not in (FILE_COMPUTE, PROGRAM_EXECUTION): continue
        value = context.public_result(raw, origin=origin)
        if origin == PROGRAM_EXECUTION:
            producer = raw.get('program_id') or next(
                (m['name'] for m in context.memory.values() if m['result_id'] == rid), rid)
            if value.get('status') == 'needs_input':
                failures[producer] = (index, {'result_id': rid, 'detail': value.get('detail'),
                                              'recovery': value.get('recovery')})
            elif value.get('status') == 'ok':
                failures.pop(producer, None)
            outputs, path = value.get('outputs', {}), ['outputs']
            accepted = value.get('status') == 'ok'
        else:
            outputs, path = value.get('data', {}).get('computed', {}), ['data', 'computed']
            accepted = value.get('accepted')
        found = []
        def walk(item, route):
            if isinstance(item, dict):
                for key, child in item.items():
                    if key.casefold() not in excluded: walk(child, [*route, key])
            elif isinstance(item, list):
                for i, child in enumerate(item[:32]): walk(child, [*route, i])
            elif type(item) in (str, int, float):
                try:
                    if Decimal(str(item)).is_finite(): found.append({'result_id': rid, 'path': route})
                except InvalidOperation: pass
        if accepted: walk(outputs, path)
        if found:
            computed.extend(found)
            last_computed = index
    return {'numeric_result_refs': computed[-12:],
            'recent_program_refusals': [row for i, row in failures.values() if i > last_computed][-2:],
            'task_completion': 'not_inferred',
            'reference_scope': 'Candidate computed leaves only; match the question and reject inputs/diagnostics before submission.'}


class HumanBankView(base.HumanBankView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.schema_refs, self.schema_definitions = {}, {}
        self.latest_numeric_refs, self.latest_refusals = [], []
        self.has_pages = False

    def selected(self, task, phase=None, observed_progress=None):
        if self.arm == 'off': return []
        names = ['OA-Target', 'OA-Units']
        if task.inputs.get('source_pages') or self.observed.get('acquired_pages'): names.append('OA-Table')
        else: names.append('OA-Locate')
        if re.search(r'average|mean|sum|ratio|percent|change|difference|growth|standard deviation|variance|correlat|calculat|forecast', task.goal, re.I):
            names.append('OA-Stats')
        if self.latest_numeric_refs: names.append('OA-Commit')
        if self.latest_refusals: names.append('OA-Fallback')
        names += [name for name, pattern in SPECIAL.items() if re.search(pattern, task.goal, re.I)]
        by_name = {s['title']: s for s in self.bank.all('skill')}
        return [by_name[name] for name in names]

    def program_options(self, *args, **kwargs):
        cards = super().program_options(*args, **kwargs)
        goal = self.task.goal if self.task else ''
        active = {'table_calculate.py'} if self.has_pages else {'scoped_search.py'}
        if self.latest_numeric_refs:
            active = {'computed_scalar.py', 'computed_list.py'}
        else:
            if re.search(SPECIAL['OA-Polynomial'], goal, re.I): active.add('polynomial_forecast.py')
            if re.search(r'correlat|gini|ratio|percent|growth|difference|mean|average|sum', goal, re.I):
                active.add('series_metrics.py')
            if re.search(r'box.cox|troy|fine ounce|price tick|exchange.rate|currency conversion', goal, re.I):
                active.add('treasury_transform.py')
        # Every capability stays discoverable and callable; only likely contracts are inlined.
        selected = [c for c in cards if Path(self.bank.get(c['id'])['source_file']).name in active]
        schemas, self.schema_definitions = v5.schema_projection([c['input_schema'] for c in selected])
        packed = {c['id']: schema for c, schema in zip(selected, schemas)}
        for card in cards:
            schema = packed.get(card['id'])
            card.pop('input_schema')
            if schema is not None: card['input_schema'] = schema
            if card['id'] in self.schema_refs: card['schema_ref'] = self.schema_refs[card['id']]
        return cards

    def bind_context(self, context):
        self.has_pages = bool(self.task.inputs.get('source_pages')) or any(
            context.origin(rid) in ('native.text-page.v1', 'native.text-context.v3')
            for rid in context.results)
        for program in self.bank.all('program'):
            rid = context.register('program_schema:'+program['id'], program['input_schema'],
                origin='manual.program-schema.v1', verify_source=True)
            self.schema_refs[program['id']] = {'result_id': rid, 'path': []}


@contextmanager
def bridge():
    from atomic_skillgraph.empirical.system import EmpiricalSystem
    with base.bridge():
        original, original_finish = EmpiricalSystem.agent, EmpiricalSystem.prepare_finish
        def agent(s, stage, prompt, material, *args, **kwargs):
            if isinstance(s.bank, HumanBankView) and stage == 'runtime':
                prompt = ANSWER_CONTRACT+'\n'+prompt
                if s.bank.arm != 'off' and kwargs.get('decision_purpose') != 'finish_only':
                    material = deepcopy(material)
                    view = s.bank
                    view.bind_task(s.adapter.task)
                    view.bind_context(s.task_context)
                    material['answer_work'] = work_status(s.task_context)
                    view.latest_numeric_refs = material['answer_work']['numeric_result_refs']
                    view.latest_refusals = material['answer_work']['recent_program_refusals']
                    view.program_options()
                    material['defs'] = view.schema_definitions
                    prompt += ('\nProgram cards remain callable. Inline schema $ref expands through defs. '
                        'If a card has only schema_ref, use read_result on that ref to obtain its exact '
                        'input schema before calling; do not guess parameters. Contract lookup shares the tool budget.')
            return original(s, stage, prompt, material, *args, **kwargs)
        def finish(s, material, prompt, **kwargs):
            if isinstance(s.bank, HumanBankView): prompt = ANSWER_CONTRACT+'\n'+prompt
            return original_finish(s, material, prompt, **kwargs)
        EmpiricalSystem.agent, EmpiricalSystem.prepare_finish = agent, finish
        try: yield
        finally: EmpiricalSystem.agent, EmpiricalSystem.prepare_finish = original, original_finish
