"""Native AtomicSkill selection; expose only the selected skills' implementations."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('office_v15_parent',ROOT.parent/'office_humanstrong_v14/integration/frontier.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
VERSION='office.humanstrong.operation-bound.v15.20261024'
KINDS,native_import,input_protocol=base.KINDS,base.native_import,base.input_protocol
native_import.__globals__['VERSION']=VERSION
CARDS=json.loads((ROOT/'cards.json').read_text())


def selected_titles(task):
    text='\n'.join(p.get('text','') for p in task.inputs.get('source_pages',[]))
    selected=[c for c in CARDS if re.search(c['when'],task.goal,re.I)
        and (not c.get('source_pattern') or re.search(c['source_pattern'],text,re.I))]
    removed={n for c in selected for n in c.get('replaces',[])}
    names=[c['title'] for c in selected]+[n for n in base.selected_titles(task) if n not in removed]
    return list(dict.fromkeys(names))[:6]


class HumanBankView(base.HumanBankView):
    def selected(self,task,*args,**kwargs):
        if self.arm=='off':return []
        by_title={s['title']:s for s in self.bank.all('skill')}
        return [by_title[n] for n in selected_titles(task) if n in by_title]

    def program_eligible(self,p):
        return bool(self.arm=='manual' and self.program_mode!='off' and p and p.get('id') in self.members
            and p.get('permission')=='author_trusted' and 'source' in p and p['environment']==self.bank.environment)

    def program_options(self,*args,**kwargs):
        if self.task is None:return []
        skills={s['id'] for s in self.selected(self.task)}
        ids={i['program_id'] for i in self.bank.all('implementation') if i['skill_id'] in skills}
        return [self.card(p) for p in self.bank.all('program') if p['id'] in ids
            and p['id'] not in kwargs.get('excluded',()) and self.program_eligible(p)]


@contextmanager
def bridge():
    from atomic_skillgraph.empirical.system import EmpiricalSystem
    agent,prepare=EmpiricalSystem.agent,EmpiricalSystem.prepare_finish
    def guidance(system,prompt):
        view=system.bank;view.bind_task(system.adapter.task);skills=view.selected(view.task)
        for skill in skills:
            if skill['guidance'] not in prompt:prompt+='\n'+skill['guidance']
        return prompt,skills
    def runtime(system,stage,prompt,material,name,schema,**kwargs):
        if not isinstance(system.bank,HumanBankView) or stage!='runtime':
            return agent(system,stage,prompt,material,name,schema,**kwargs)
        prompt,skills=guidance(system,prompt)
        final=kwargs.get('decision_purpose')=='finish_only' or name=='finish_answer'
        material=deepcopy(material if final else input_protocol.runtime_material(system,material))
        if not final:
            material['human_frontier_skill_ids']=[s['id'] for s in skills]
            refs=base.base.base.acquired_results(system.task_context);material['acquired_result_refs']=refs
            programs=system.bank.program_options()
            if not refs['numeric_refs']:
                scalar_ids={p['id'] for p in system.bank.bank.all('program') if p['operation']=='scalar_display'}
                programs=[p for p in programs if p['id'] not in scalar_ids]
            material.setdefault('calls',{})['programs']=programs
            if 'programs' in material:material['programs']=programs
            if programs:prompt+='\nThe linked Programs submit their final answer directly when submit_answer=true. Bind observed operands and requested units first; use the operation matching the question. Never submit an intermediate result. Native Python remains available for other operations.'
        return agent(system,stage,prompt,material,name,schema,**kwargs)
    def finish(system,materials,prompt,*,allowed_result_ids=None):
        if isinstance(system.bank,HumanBankView):
            prompt,_=guidance(system,prompt);refs=input_protocol.seed_context(system)
            if allowed_result_ids is not None:allowed_result_ids=set(allowed_result_ids)|{r['result_id'] for r in refs}
        return prepare(system,materials,prompt,allowed_result_ids=allowed_result_ids)
    EmpiricalSystem.agent,EmpiricalSystem.prepare_finish=runtime,finish
    try:yield
    finally:EmpiricalSystem.agent,EmpiricalSystem.prepare_finish=agent,prepare
