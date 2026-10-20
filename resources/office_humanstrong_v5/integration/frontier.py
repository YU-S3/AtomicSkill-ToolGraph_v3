"""V5 authored Bank only; the ordinary native Executor and budgets stay in control."""
from contextlib import contextmanager
from copy import deepcopy
from collections import Counter
import importlib.util
import json
from pathlib import Path

path = Path(__file__).resolve().parents[2]/'office_humanstrong_v2/integration/frontier.py'
spec = importlib.util.spec_from_file_location('office_v5_native_bridge', path)
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
VERSION = 'office.humanstrong.programmatic.v5.20261020'
base.VERSION = VERSION
previous = base.selection.callable_programs
base.selection.callable_programs = lambda *a, **kw: previous(*a, **kw)+[
    'polynomial_forecast.py', 'table_calculate.py', 'computed_scalar.py', 'computed_list.py']
KINDS, native_import, input_protocol = base.KINDS, base.native_import, base.input_protocol
ANSWER_CONTRACT = (
    'Return only the requested answer, without reasoning. Follow an explicit output template first. '
    'For quantities requested in a unit, return the number at that scale unless a suffix is requested. '
    'For percentages choose the public requested display (% or bare numeric) explicitly; do not change '
    'that choice after computation. Preserve sign, rounding, date form and list order.')
base.selection.FINISH_ONLY_RULE = ANSWER_CONTRACT+' Preserve executed results; retrieval alone is not computation.'
base.selection.GLOBAL_RULE = (
    'Use supplied pages first; search only a specific missing fact. All advertised Programs remain callable. '
    'Execute target arithmetic before finish-only. Program success proves the named operation, not question semantics. '
    'Use real result references, not evidence_state fields. Correct the reported needs_input field; no repeated guesses '
    'or compulsory verification. Submit promptly once supported.')


def schema_projection(schemas):
    """Lossless shared definitions in model materials; stored/native schemas stay exact."""
    encode = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'))
    counts, objects = Counter(), {}
    def scan(value):
        if isinstance(value, dict):
            key = encode(value)
            if len(key) > 100: counts[key] += 1; objects[key] = value
            for child in value.values(): scan(child)
        elif isinstance(value, list):
            for child in value: scan(child)
    scan(schemas)
    names = {key: 's'+str(i) for i, key in enumerate(sorted(k for k, n in counts.items() if n > 1))}
    def pack(value, root=False):
        if isinstance(value, dict):
            key = encode(value)
            if not root and key in names: return {'$ref': '#/defs/'+names[key]}
            return {k: pack(v) for k, v in value.items()}
        if isinstance(value, list): return [pack(v) for v in value]
        return value
    return pack(schemas), {name: pack(objects[key], True) for key, name in names.items()}


class HumanBankView(base.HumanBankView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        programs = self.bank.all('program')
        schemas, self.schema_definitions = schema_projection([p['input_schema'] for p in programs])
        self.model_schemas = {p['id']: schema for p, schema in zip(programs, schemas)}

    def program_options(self, *args, **kwargs):
        return [{**card, 'input_schema': self.model_schemas[card['id']]}
                for card in super().program_options(*args, **kwargs)]

    def selected(self, task, phase=None, observed_progress=None):
        if self.arm == 'off': return []
        names = ['OA-Target', 'OA-Table', 'OA-Units', 'OA-Stats']
        if getattr(self, 'latest_numeric_refs', []): names.append('OA-Commit')
        if not task.inputs.get('source_pages') and not self.observed.get('acquired_pages'):
            names.append('OA-Locate')
        if getattr(self, 'latest_refusals', []): names.append('OA-Fallback')
        relevant = base.selection.select_names(task.goal)
        # Specific computation/category instructions precede broad year/publication hints.
        names += [name for name in ('OA-Advanced', 'OA-Treasury', 'OA-Fiscal', 'OA-Revision', 'OA-Periods')
                  if name in relevant][:2]
        by_name = {skill['title']: skill for skill in self.bank.all('skill')}
        return [by_name[name] for name in names if name in by_name]


def work_status(context):
    from atomic_skillgraph.empirical.public_results import FILE_COMPUTE, PROGRAM_EXECUTION
    computed, refusals = [], []
    for rid, raw in context.results.items():
        origin = context.origin(rid)
        if origin not in (FILE_COMPUTE, PROGRAM_EXECUTION): continue
        value = context.public_result(raw, origin=origin)
        if origin == PROGRAM_EXECUTION and value.get('status') == 'needs_input':
            refusals.append({'result_id': rid, 'detail': value.get('detail'),
                             'recovery': value.get('recovery')})
        outputs = value.get('outputs', {}) if origin == PROGRAM_EXECUTION else value.get('data', {}).get('computed', {})
        # Record numeric leaves, not stdout/search success as a claim of task completion.
        def walk(item, path):
            if isinstance(item, dict):
                for key, child in item.items(): walk(child, [*path, key])
            elif isinstance(item, list):
                for index, child in enumerate(item[:32]): walk(child, [*path, index])
            elif type(item) in (str, int, float):
                from decimal import Decimal, InvalidOperation
                try:
                    if Decimal(str(item)).is_finite(): computed.append({'result_id': rid, 'path': path})
                except InvalidOperation: pass
        if (origin == PROGRAM_EXECUTION and value.get('status') == 'ok' and
                any(k in outputs for k in ('value', 'answer', 'prediction', 'calculation'))):
            walk(outputs, ['outputs'])
        elif origin == FILE_COMPUTE and value.get('accepted'):
            walk(outputs, ['data', 'computed'])
    return {'numeric_result_refs': computed[-12:], 'recent_program_refusals': refusals[-2:],
        'task_completion': 'not_inferred',
        'next_step': 'Bind supplied source rows and execute the requested calculation now; search only a specific missing fact.'
            if not computed else 'Check which executed result covers the question, then select its explicit final-format Program.'}


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
                    material['answer_work'] = work_status(s.task_context)
                    s.bank.latest_refusals = material['answer_work']['recent_program_refusals']
                    s.bank.latest_numeric_refs = material['answer_work']['numeric_result_refs']
                    material['defs'] = s.bank.schema_definitions
                    prompt += '\nProgram schema $ref expands through defs in this material; all fields and constraints apply.'
            return original(s, stage, prompt, material, *args, **kwargs)
        def finish(s, material, prompt, **kwargs):
            if isinstance(s.bank, HumanBankView): prompt = ANSWER_CONTRACT+'\n'+prompt
            return original_finish(s, material, prompt, **kwargs)
        EmpiricalSystem.agent, EmpiricalSystem.prepare_finish = agent, finish
        try: yield
        finally: EmpiricalSystem.agent, EmpiricalSystem.prepare_finish = original, original_finish
