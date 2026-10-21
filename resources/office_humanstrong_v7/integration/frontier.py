"""Conditional authored guidance and on-demand contracts on the native executor."""
from contextlib import contextmanager
from copy import deepcopy
from decimal import Decimal, InvalidOperation
import importlib.util
import json
from pathlib import Path
import re

path = Path(__file__).resolve().parents[2]/'office_humanstrong_v5/integration/frontier.py'
spec = importlib.util.spec_from_file_location('office_v7_parent', path)
v5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v5)
base = v5.base
VERSION = 'office.humanstrong.programmatic.v7.20261021'
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
                'ddof', 'degree', 'periods', 'decimals', 'unit', 'source_line', 'start', 'end',
                'operand_values', 'operand_refs', 'source_ref', 'line', 'offset', 'start_offset',
                'end_offset', 'anchor_offset', 'next_offset', 'size', 'len', 'length'}
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
            if not accepted and value.get('error'):
                failures['temporary_python'] = (index, {'result_id': rid, 'detail': value['error'],
                    'recovery': {'code': 'python_execution_failed'}})
        found = []
        def walk(item, route):
            if isinstance(item, dict):
                for key, child in item.items():
                    if key.casefold() not in excluded: walk(child, [*route, key])
            elif isinstance(item, list):
                # A filtered source line such as [line_number, text] is not a calculation.
                if any(isinstance(child, str) and not numeric(child) for child in item): return
                for i, child in enumerate(item[:32]): walk(child, [*route, i])
            elif type(item) in (str, int, float):
                if numeric(item): found.append({'result_id': rid, 'path': route})
        if accepted: walk(outputs, path)
        if found:
            computed.extend(found)
            last_computed = index
    return {'numeric_result_refs': computed[-12:],
            'recent_program_refusals': [row for i, row in failures.values() if i > last_computed][-2:],
            'task_completion': 'not_inferred',
            'reference_scope': 'Candidate computed leaves only; match the question and reject inputs/diagnostics before submission.'}


def numeric(value):
    try: return Decimal(str(value)).is_finite()
    except InvalidOperation: return False


def recent_scalar_result_refs(context, limit=6):
    if context is None: return []
    return [{**ref, 'display_value': context.resolve_public(ref), 'origin': context.origin(ref['result_id']),
             'proof_scope': 'candidate_computation_only_not_task_completion'}
            for ref in reversed(work_status(context)['numeric_result_refs'][-min(6, limit):])]


class HumanBankView(base.HumanBankView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.schema_refs = {}
        self.latest_numeric_refs, self.latest_refusals = [], []

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
        # Keep all capabilities discoverable, without paying for unused contracts every turn.
        for card in cards:
            card.pop('input_schema')
            if card['id'] in self.schema_refs: card['schema_ref'] = self.schema_refs[card['id']]
        return cards

    def bind_context(self, context):
        for program in self.bank.all('program'):
            # read_result reads public strings/lists, never dictionary roots.
            schema = json.dumps(program['input_schema'], ensure_ascii=False, separators=(',', ':'))
            rid = context.register('program_schema:'+program['id'], schema,
                origin='manual.program-schema.v1', verify_source=True)
            self.schema_refs[program['id']] = {'result_id': rid, 'path': []}


@contextmanager
def bridge():
    from atomic_skillgraph.empirical.system import EmpiricalSystem
    with base.bridge():
        original, original_finish = EmpiricalSystem.agent, EmpiricalSystem.prepare_finish
        original_refs = base.policy.recent_scalar_result_refs
        base.policy.recent_scalar_result_refs = recent_scalar_result_refs
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
                    prompt += ('\nProgram cards remain callable. Use read_result(schema_ref) for the exact '
                        'input contract of a Program you intend to use. The ref is JSON text; '
                        'read subsequent offsets if truncated. '
                        'Reuse that contract rather than rereading it; do not guess parameters. '
                        'Lookup shares the tool budget. Native Python and finish remain available.')
            return original(s, stage, prompt, material, *args, **kwargs)
        def finish(s, material, prompt, **kwargs):
            if isinstance(s.bank, HumanBankView): prompt = ANSWER_CONTRACT+'\n'+prompt
            return original_finish(s, material, prompt, **kwargs)
        EmpiricalSystem.agent, EmpiricalSystem.prepare_finish = agent, finish
        try: yield
        finally:
            EmpiricalSystem.agent, EmpiricalSystem.prepare_finish = original, original_finish
            base.policy.recent_scalar_result_refs = original_refs
