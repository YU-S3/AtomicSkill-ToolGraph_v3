"""Specific actionable cards first; deterministic submission after acquired calculations."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('office_v11_parent',ROOT.parent/'office_humanstrong_v10/integration/frontier.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
VERSION='office.humanstrong.source-final.v11.20261021'
base.VERSION=base.base.VERSION=base.base.base.VERSION=VERSION
KINDS,native_import,input_protocol=base.KINDS,base.native_import,base.input_protocol
ANSWER_CONTRACT=base.ANSWER_CONTRACT
SPECS=json.loads((ROOT/'principles.json').read_text())


def matching(goal,inputs=None):
    active=[s for s in SPECS if re.search(s['when'],goal,re.I)]
    if re.search(r'polynomial|degree\s*[2-9]',goal,re.I):
        active=[s for s in active if s['title'] not in ('regression_coordinates','difference_value')]
    if re.search(r'percent|relative|ratio|growth|median|deviation',goal,re.I):
        active=[s for s in active if s['title']!='difference_value']
    if re.search(r'continuous|continuously',goal,re.I):
        active=[s for s in active if s['title']!='compound_growth']
    pages=(inputs or {}).get('source_pages',[])
    labels=[]
    for page in pages:
        labels.append({line.strip().strip('|').split('|')[0].strip().lower() for line in page.get('text','').splitlines()
            if line.startswith('|') and re.search(r'\d',line) and line.strip().strip('|').split('|')[0].strip()})
    if any(a&b for i,a in enumerate(labels) for b in labels[i+1:]):
        active.append(next(s for s in SPECS if s['title']=='revision_basis'))
    if re.search(r'\btotal\b',goal,re.I) and any(re.search(r'continued|payable in',p.get('text',''),re.I) for p in pages):
        active.append(next(s for s in SPECS if s['title']=='total_scope'))
    if inputs and base.grouped_table(inputs):
        active.append(next(s for s in SPECS if s['title']=='table_groups'))
    return active[:3]


class HumanBankView(base.HumanBankView):
    def selected(self,task,*args,**kwargs):
        if self.arm=='off':return []
        names={s['title'] for s in matching(task.goal,task.inputs)}
        return [s for s in self.bank.all('skill') if s['title'] in names]

    def program_options(self,*args,**kwargs):
        if self.arm=='off' or self.task is None:return []
        wanted={s.get('operation') for s in matching(self.task.goal,self.task.inputs)}
        return [self.card(p) for p in self.bank.all('program') if self.program_eligible(p)
            and p['operation'] in wanted and p['id'] not in kwargs.get('excluded',())]

    @staticmethod
    def card(p):
        return {'id':p['id'],'state':'callable_manual','goal':p['title'],'input_schema':p['input_schema'],
            'output_fields':list(p['output_schema']['properties']),'result_role':p['result_role']}


@contextmanager
def bridge():
    from atomic_skillgraph.empirical.system import EmpiricalSystem
    agent,finish=EmpiricalSystem.agent,EmpiricalSystem.prepare_finish

    def runtime(system,stage,prompt,material,name,schema,**kwargs):
        if not isinstance(system.bank,HumanBankView) or stage!='runtime':
            return agent(system,stage,prompt,material,name,schema,**kwargs)
        prompt=ANSWER_CONTRACT+'\n'+prompt
        if kwargs.get('decision_purpose')!='finish_only' and name!='finish_answer':
            material=deepcopy(input_protocol.runtime_material(system,material))
            view=system.bank;view.bind_task(system.adapter.task)
            cards=view.program_options();skills=view.selected(view.task)
            # A formatting helper is offered only after a real compute operation. No guessed value/leaf selection.
            from atomic_skillgraph.empirical.public_results import FILE_COMPUTE
            context=system.task_context
            computed=bool(context and any(context.origin(rid)==FILE_COMPUTE and raw.get('accepted') and
                raw.get('data',{}).get('computed') for rid,raw in context.results.items()))
            if view.arm!='off' and computed:
                p=next(p for p in view.bank.all('program') if p.get('operation')=='final_scalar')
                cards.append(view.card(p))
                skills.append(next(s for s in view.bank.all('skill') if s['title']=='final_display'))
            material.setdefault('calls',{})['programs']=cards
            if 'programs' in material:material['programs']=cards
            if skills:
                material['human_frontier_skill_ids']=[s['id'] for s in skills]
                prompt+='\n'+'\n'.join(s['guidance'] for s in skills)
            if cards:
                prompt+='\nFinal calculators take actual source/result references, not retyped numeric observations. Choose the exact requested operation and display; submit_answer=true ends the whole task after successful execution. For partial work or unsupported layouts use native Python; do not submit a partial result.'
        return agent(system,stage,prompt,material,name,schema,**kwargs)

    def prepare_finish(system,materials,prompt,*,allowed_result_ids=None):
        if isinstance(system.bank,HumanBankView):
            refs=input_protocol.seed_context(system)
            if allowed_result_ids is not None:allowed_result_ids=set(allowed_result_ids)|{r['result_id'] for r in refs}
            prompt=ANSWER_CONTRACT+'\n'+prompt
        return finish(system,materials,prompt,allowed_result_ids=allowed_result_ids)

    EmpiricalSystem.agent,EmpiricalSystem.prepare_finish=runtime,prepare_finish
    try:yield
    finally:EmpiricalSystem.agent,EmpiricalSystem.prepare_finish=agent,finish
