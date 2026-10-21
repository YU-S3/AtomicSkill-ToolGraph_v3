"""Small conditional principles; native evidence and stopping for both arms."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('office_v8_native', ROOT.parent/'office_humanstrong_v2/integration/frontier.py')
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
VERSION = 'office.humanstrong.sparse.v8.20261021'
base.VERSION = VERSION
KINDS, native_import, input_protocol = base.KINDS, base.native_import, base.input_protocol
SPECS = json.loads((ROOT/'principles.json').read_text())
ANSWER_CONTRACT = (
    'Return only the requested answer, without reasoning. Follow an explicit output template first. '
    'For quantities requested in a unit, return the number at that scale unless a suffix is requested. '
    'For percentages choose the public requested display (% or bare numeric) explicitly; do not change '
    'that choice after computation. Preserve sign, rounding, date form and list order.')


def matching(goal):
    return [s for s in SPECS if re.search(s['when'], goal, re.I)][:2]


class HumanBankView(base.HumanBankView):
    def selected(self, task, *args, **kwargs):
        if self.arm == 'off': return []
        names = {s['title'] for s in matching(task.goal)}
        return [s for s in self.bank.all('skill') if s['title'] in names]

    def program_options(self, *args, **kwargs):
        if self.arm == 'off' or self.task is None: return []
        wanted = {s.get('operation') for s in matching(self.task.goal)}
        return [{'id': p['id'], 'state': 'callable_manual', 'goal': p['title'],
                 'input_schema': p['input_schema'], 'output_fields': ['value', 'answer', 'formula'],
                 'result_role': p['result_role']}
                for p in self.bank.all('program') if self.program_eligible(p)
                and p['operation'] in wanted and p['id'] not in kwargs.get('excluded', ())]


@contextmanager
def bridge():
    from atomic_skillgraph.empirical.system import EmpiricalSystem
    agent, finish = EmpiricalSystem.agent, EmpiricalSystem.prepare_finish

    def runtime(system, stage, prompt, material, name, schema, **kwargs):
        if not isinstance(system.bank, HumanBankView) or stage != 'runtime':
            return agent(system, stage, prompt, material, name, schema, **kwargs)
        prompt = ANSWER_CONTRACT+'\n'+prompt
        if kwargs.get('decision_purpose') != 'finish_only' and name != 'finish_answer':
            material = input_protocol.runtime_material(system, material)
            view = system.bank; view.bind_task(system.adapter.task)
            material = deepcopy(material)
            cards = view.program_options()
            material.setdefault('calls', {})['programs'] = cards
            if 'programs' in material: material['programs'] = cards
            skills = view.selected(view.task)
            if skills:
                material['human_frontier_skill_ids'] = [s['id'] for s in skills]
                prompt += '\n'+'\n'.join(s['guidance'] for s in skills)
            if cards:
                prompt += '\nNamed calculators are optional; use one when its exact formula fits. Native Python and finish remain available.'
        return agent(system, stage, prompt, material, name, schema, **kwargs)

    def prepare_finish(system, materials, prompt, *, allowed_result_ids=None):
        if isinstance(system.bank, HumanBankView):
            refs = input_protocol.seed_context(system)
            if allowed_result_ids is not None:
                allowed_result_ids = set(allowed_result_ids) | {r['result_id'] for r in refs}
            prompt = ANSWER_CONTRACT+'\n'+prompt
        # Same evidence packing on both arms. No numeric-leaf guessing or phase machine.
        return finish(system, materials, prompt, allowed_result_ids=allowed_result_ids)

    EmpiricalSystem.agent, EmpiricalSystem.prepare_finish = runtime, prepare_finish
    try: yield
    finally: EmpiricalSystem.agent, EmpiricalSystem.prepare_finish = agent, finish
