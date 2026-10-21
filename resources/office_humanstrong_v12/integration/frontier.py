"""Ablate short rules, source calculators and typed submission on one native solver."""
from contextlib import contextmanager
from copy import deepcopy
from decimal import Decimal, InvalidOperation
import importlib.util
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('office_v12_parent',ROOT.parent/'office_humanstrong_v11/integration/frontier.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
VERSION='office.humanstrong.atomic-directions.v12.20261022'
base.VERSION=base.base.VERSION=base.base.base.VERSION=base.base.base.base.VERSION=VERSION
KINDS,native_import,input_protocol=base.KINDS,base.native_import,base.input_protocol
ANSWER_CONTRACT=base.ANSWER_CONTRACT
SPECS=json.loads((ROOT/'principles.json').read_text())
PROFILES={'off','rules','selective','guarded'}


def matching(goal,inputs=None):
    old={s['title'] for s in base.matching(goal,inputs)}
    active=[s for s in SPECS if (s['title'] in old or re.search(s['when'],goal,re.I)) and s['title']!='typed_submission']
    if re.search(r'polynomial|degree\s*[2-9]',goal,re.I):
        active=[s for s in active if s['title'] not in ('regression_coordinates','difference_value')]
    if re.search(r'percent|relative|ratio|growth|median|deviation',goal,re.I):
        active=[s for s in active if s['title']!='difference_value']
    if re.search(r'contin(?:u)?ous',goal,re.I):
        active=[s for s in active if s['title']!='compound_growth']
    if any(s['title']=='share_difference' for s in active):
        active=[s for s in active if s['title']!='difference_value']
    return active[:4]


def acquired_results(context):
    """Enumerate real public references. No inferred answers, units or repairs."""
    from atomic_skillgraph.empirical.public_results import FILE_COMPUTE,PROGRAM_EXECUTION
    scalars,quantities=[],[]
    if context is None:return {'numeric_refs':[],'quantity_refs':[]}
    def numeric(value):
        if type(value) not in (str,int,float):return False
        try:return Decimal(str(value)).is_finite()
        except InvalidOperation:return False
    def walk(value,rid,path):
        if isinstance(value,dict):
            if {'value','scale','unit'}<=set(value) and numeric(value['value']) and value['scale'] in (
                    'quantity','fraction','percent','percentage_points') and isinstance(value['unit'],str):
                quantities.append({'quantity_ref':{'result_id':rid,'path':path},
                    'value':value['value'],'scale':value['scale'],'unit':value['unit']})
            for key,child in value.items():walk(child,rid,[*path,key])
        elif isinstance(value,list):
            for index,child in enumerate(value[:40]):walk(child,rid,[*path,index])
        elif numeric(value):scalars.append({'value_ref':{'result_id':rid,'path':path},'value':value})
    for rid,raw in context.results.items():
        origin=context.origin(rid)
        value=context.public_result(raw,origin=origin)
        if origin==FILE_COMPUTE and value.get('accepted'):
            walk(value.get('data',{}).get('computed',{}),rid,['data','computed'])
        elif origin==PROGRAM_EXECUTION and value.get('status')=='ok':
            walk(value.get('outputs',{}),rid,['outputs'])
    return {'numeric_refs':scalars[-24:],'quantity_refs':quantities[-8:],
        'omitted_numeric_refs':max(0,len(scalars)-24),'answer_validity':'not_inferred'}


class HumanBankView(base.HumanBankView):
    def __init__(self,bank,bundle,arm='manual',**kwargs):
        arm='selective' if arm=='manual' else arm
        if arm not in PROFILES:raise ValueError('Unknown frozen Bank direction')
        super().__init__(bank,bundle,arm,**kwargs)

    def selected(self,task,*args,**kwargs):
        if self.arm=='off':return []
        names={s['title'] for s in matching(task.goal,task.inputs)}
        return [s for s in self.bank.all('skill') if s['title'] in names]

    def program_eligible(self,p):
        return self.arm not in ('off','rules') and super().program_eligible(p)

    def program_options(self,*args,**kwargs):
        if self.arm in ('off','rules') or self.task is None:return []
        wanted={s.get('operation') for s in matching(self.task.goal,self.task.inputs)}
        return [self.card(p) for p in self.bank.all('program') if self.program_eligible(p)
            and p['operation'] in wanted and p['id'] not in kwargs.get('excluded',())]


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
            # Mechanical public API references are identical on every arm, including off.
            refs=acquired_results(system.task_context);material['acquired_result_refs']=refs
            cards=view.program_options();skills=view.selected(view.task)
            if view.arm=='guarded':
                skills.append(next(s for s in view.bank.all('skill') if s['title']=='typed_submission'))
                if refs['quantity_refs']:
                    cards.append(view.card(next(p for p in view.bank.all('program') if p['operation']=='typed_scalar')))
            material.setdefault('calls',{})['programs']=cards
            if 'programs' in material:material['programs']=cards
            if skills:
                material['human_frontier_skill_ids']=[s['id'] for s in skills]
                prompt+='\n'+'\n'.join(s['guidance'] for s in skills)
            if cards:
                prompt+='\nUse a listed source calculator when its exact operation covers the whole answer; success submits directly. Copy provided source/result references exactly. Native Python and finish remain available for partial or unsupported work.'
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
