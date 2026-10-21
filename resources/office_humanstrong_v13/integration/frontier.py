"""Keep the short-rule control; isolate final precision and optional scalar submission."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('office_v13_parent',ROOT.parent/'office_humanstrong_v12/integration/frontier.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
VERSION='office.humanstrong.final-display.v13.20261022'
KINDS,native_import,input_protocol=base.KINDS,base.native_import,base.input_protocol
native_import.__globals__['VERSION']=VERSION
ANSWER_CONTRACT=base.ANSWER_CONTRACT
PRECISION=json.loads((ROOT/'precision.json').read_text())
PROFILES={'off','rules','precise','display'}


def precision_needed(goal):
    return bool(re.search(r'decimal places?|nearest (?:tenth|hundredth|thousandth)|round(?:ed|ing)?|percentage points',goal,re.I))


class HumanBankView(base.HumanBankView):
    def __init__(self,bank,bundle,arm='rules',**kwargs):
        if arm not in PROFILES:raise ValueError('Unknown final-display direction')
        super().__init__(bank,bundle,'off' if arm=='off' else 'rules',**kwargs)
        self.arm=arm
        from atomic_skillgraph.empirical.contracts import digest
        self.view_hash=digest([VERSION,bundle,arm,self.program_mode])

    def selected(self,task,*args,**kwargs):
        if self.arm=='off':return []
        names={s['title'] for s in base.matching(task.goal,task.inputs)}
        if self.arm in ('precise','display') and precision_needed(task.goal):names.add(PRECISION['title'])
        return [s for s in self.bank.all('skill') if s['title'] in names]

    def program_options(self,*args,**kwargs):return []

    def program_eligible(self,program):
        return self.arm=='display' and super().program_eligible(program)


@contextmanager
def bridge():
    from atomic_skillgraph.empirical.system import EmpiricalSystem
    agent,finish=EmpiricalSystem.agent,EmpiricalSystem.prepare_finish
    def finish_prompt(system,prompt):
        if (system.bank.arm in ('precise','display') and precision_needed(system.adapter.task.goal)
                and PRECISION['guidance'] not in prompt):prompt+='\n'+PRECISION['guidance']
        return prompt
    def runtime(system,stage,prompt,material,name,schema,**kwargs):
        if not isinstance(system.bank,HumanBankView) or stage!='runtime':
            return agent(system,stage,prompt,material,name,schema,**kwargs)
        view=system.bank;view.bind_task(system.adapter.task)
        prompt=ANSWER_CONTRACT+'\n'+prompt
        final=kwargs.get('decision_purpose')=='finish_only' or name=='finish_answer'
        if final:
            prompt=finish_prompt(system,prompt)
        else:
            material=deepcopy(input_protocol.runtime_material(system,material))
            refs=base.acquired_results(system.task_context);material['acquired_result_refs']=refs
            cards=[]
            if view.arm=='display' and refs['numeric_refs'] and precision_needed(view.task.goal):
                cards=[view.card(p) for p in view.bank.all('program')]
                prompt+='\nIf an acquired numeric result is the complete final quantity, use scalar_display with its exact value_ref and explicit input/output scale. Success submits directly. Otherwise use native tools or finish; never submit an intermediate value.'
            material.setdefault('calls',{})['programs']=cards
            if 'programs' in material:material['programs']=cards
            skills=view.selected(view.task)
            if skills:
                material['human_frontier_skill_ids']=[s['id'] for s in skills]
                prompt+='\n'+'\n'.join(s['guidance'] for s in skills)
        return agent(system,stage,prompt,material,name,schema,**kwargs)
    def prepare_finish(system,materials,prompt,*,allowed_result_ids=None):
        if isinstance(system.bank,HumanBankView):
            refs=input_protocol.seed_context(system)
            if allowed_result_ids is not None:allowed_result_ids=set(allowed_result_ids)|{r['result_id'] for r in refs}
            prompt=finish_prompt(system,ANSWER_CONTRACT+'\n'+prompt)
        return finish(system,materials,prompt,allowed_result_ids=allowed_result_ids)
    EmpiricalSystem.agent,EmpiricalSystem.prepare_finish=runtime,prepare_finish
    try:yield
    finally:EmpiricalSystem.agent,EmpiricalSystem.prepare_finish=agent,finish
