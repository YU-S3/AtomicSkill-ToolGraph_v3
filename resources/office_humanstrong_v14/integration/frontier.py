"""Native AtomicSkill guidance and one optional acquired-scalar Implementation."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('office_v14_parent',ROOT.parent/'office_humanstrong_v13/integration/frontier.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
VERSION='office.humanstrong.submission-facts.v14.20261023'
KINDS,native_import,input_protocol=base.KINDS,base.native_import,base.input_protocol
native_import.__globals__['VERSION']=VERSION
CARDS=json.loads((ROOT/'cards.json').read_text())


def selected_titles(task):
    text='\n'.join(p.get('text','') for p in task.inputs.get('source_pages',[]))
    matches=[c['title'] for c in CARDS if re.search(c['when'],task.goal,re.I)
             and (not c.get('source_pattern') or re.search(c['source_pattern'],text,re.I))]
    previous=[s['title'] for s in base.base.matching(task.goal,task.inputs)]
    if 'arc_change' in matches:previous=[n for n in previous if n not in ('normalized_midpoint','difference_value')]
    if 'percent_difference' in matches:previous=[n for n in previous if n not in ('normalized_midpoint','difference_value')]
    if 'median_absolute_deviation' in matches:previous=[n for n in previous if n not in ('median_deviation','absolute_deviation')]
    if 'complete_monthly_series' in matches:previous=[n for n in previous if n not in ('series_scope','table_groups')]
    if 'table_block_binding' in matches:previous=[n for n in previous if n not in ('total_scope','table_groups')]
    # Fact cards supplement their relevant operation, never outrank all of its checks.
    specific=[n for n in matches if n!='answer_only']
    return ['answer_only',*dict.fromkeys(specific+previous)][:6]


class HumanBankView(base.HumanBankView):
    def __init__(self,bank,bundle,arm='manual',**kwargs):
        if arm not in ('manual','off'):raise ValueError('Unknown V14 profile')
        super().__init__(bank,bundle,'display' if arm=='manual' else 'off',**kwargs)
        self.arm=arm

    def selected(self,task,*args,**kwargs):
        if self.arm=='off':return []
        by_title={s['title']:s for s in self.bank.all('skill')}
        return [by_title[n] for n in selected_titles(task) if n in by_title]

    def program_eligible(self,program):
        return bool(self.arm=='manual' and self.program_mode!='off' and program
            and program.get('id') in self.members and program.get('permission')=='author_trusted'
            and program.get('operation')=='scalar_display' and 'source' in program
            and program['environment']==self.bank.environment)


@contextmanager
def bridge():
    from atomic_skillgraph.empirical.system import EmpiricalSystem
    agent,prepare=EmpiricalSystem.agent,EmpiricalSystem.prepare_finish
    def guidance(system,prompt):
        view=system.bank;view.bind_task(system.adapter.task)
        skills=view.selected(view.task)
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
            refs=base.base.acquired_results(system.task_context);material['acquired_result_refs']=refs
            programs=[system.bank.card(p) for p in system.bank.bank.all('program')
                if system.bank.program_eligible(p)] if refs['numeric_refs'] else []
            material.setdefault('calls',{})['programs']=programs
            if programs:prompt+='\nIf an acquired numeric leaf is the complete final quantity, scalar_display can format and submit it directly with explicit scales. It does not verify your data/formula choice. Otherwise finish normally; never submit an intermediate value.'
        return agent(system,stage,prompt,material,name,schema,**kwargs)
    def finish(system,materials,prompt,*,allowed_result_ids=None):
        if isinstance(system.bank,HumanBankView):
            prompt,_=guidance(system,prompt)
            refs=input_protocol.seed_context(system)
            if allowed_result_ids is not None:allowed_result_ids=set(allowed_result_ids)|{r['result_id'] for r in refs}
        return prepare(system,materials,prompt,allowed_result_ids=allowed_result_ids)
    EmpiricalSystem.agent,EmpiricalSystem.prepare_finish=runtime,finish
    try:yield
    finally:EmpiricalSystem.agent,EmpiricalSystem.prepare_finish=agent,prepare
