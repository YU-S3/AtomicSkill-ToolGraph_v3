"""Conditional short cards on the same native solver used by Bank-off."""
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('office_v10_sparse', ROOT.parent/'office_humanstrong_v8/integration/frontier.py')
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
VERSION = 'office.humanstrong.scoped.v10.20261021'
base.VERSION = base.base.VERSION = VERSION
SPECS = json.loads((ROOT/'principles.json').read_text())
ANSWER_CONTRACT = base.ANSWER_CONTRACT
KINDS, native_import, input_protocol = base.KINDS, base.native_import, base.input_protocol


def grouped_table(inputs):
    for page in inputs.get('source_pages', []):
        lines = page.get('text', '').splitlines()
        for i, line in enumerate(lines[:-1]):
            if not line.lstrip().startswith('|') or not re.search(r'\|\s*:?-{3,}', lines[i+1]): continue
            cells = [c.strip().lower() for c in line.strip().strip('|').split('|') if c.strip()]
            if len(cells) != len(set(cells)): return True
    return False


def matching(goal, inputs=None):
    price_goal = re.sub(r'\b(?:not|non)[ -]+inflation[ -]adjusted\b|\bnot adjusted for inflation\b|'
        r'\bwithout inflation adjustment\b', '', goal, flags=re.I)
    active = [s for s in SPECS if re.search(s['when'], price_goal if s['title']=='price_basis' else goal, re.I)]
    # A polynomial is not the linear statistic implemented by this calculator.
    if re.search(r'polynomial|degree\s*[2-9]', goal, re.I):
        active = [s for s in active if s['title'] not in ('series_scope', 'regression_coordinates')]
    if re.search(r'continuous|continuously', goal, re.I):
        active = [s for s in active if s['title'] != 'compound_growth']
    if inputs and grouped_table(inputs):
        active.insert(0, next(s for s in SPECS if s['title'] == 'table_groups'))
    return active[:2]


class HumanBankView(base.HumanBankView):
    def selected(self, task, *args, **kwargs):
        if self.arm == 'off': return []
        names = {s['title'] for s in matching(task.goal, task.inputs)}
        return [s for s in self.bank.all('skill') if s['title'] in names]

    def program_options(self, *args, **kwargs):
        if self.arm == 'off' or self.task is None: return []
        wanted = {s.get('operation') for s in matching(self.task.goal, self.task.inputs)}
        return [{'id':p['id'], 'state':'callable_manual', 'goal':p['title'],
                 'input_schema':p['input_schema'], 'output_fields':list(p['output_schema']['properties']),
                 'result_role':p['result_role']}
                for p in self.bank.all('program') if self.program_eligible(p)
                and p['operation'] in wanted and p['id'] not in kwargs.get('excluded', ())]


base.HumanBankView = HumanBankView
bridge = base.bridge
