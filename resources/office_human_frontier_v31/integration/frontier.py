"""Isolated authored Office Bank view on the unchanged native tool-loop Executor."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path

from atomic_skillgraph.empirical import office_human_policy_v31 as policy
from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.contracts import digest, program_digest, RuntimeDecision
from atomic_skillgraph.empirical.checkpoint import save_json

VERSION = policy.VERSION
KINDS = ('skill', 'program', 'implementation', 'workflow')
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))


def native_import(bundle, destination, environment, office_method=None, seed=42):
    root = Path(destination)
    identity = {'bundle': digest(bundle), 'environment': environment, 'office_method': office_method, 'seed': seed}
    if (root/'manual_manifest.json').exists():
        if read(root/'manual_manifest.json')['identity'] != identity:
            raise ValueError('Changed Bank requires an independent output directory')
        return read(root/'imported_bank.json')
    if (root/'bank.sqlite3').exists():raise ValueError('Incomplete import; do not overwrite it')
    target = deepcopy(bundle);bank = Bank(root, seed=seed, environment=environment);remap = {}
    try:
        if office_method:bank.bind_office_method(office_method)
        for a in target['skill']:bank.put('skill', a)
        for p in target['program']:
            old=p['id'];p['environment']=deepcopy(environment);p.pop('local_validation', None)
            p['id']='program_'+program_digest(p);p.update(bank.put('program', p));remap[old]=p['id']
        for a in target['implementation']:
            a['program_id']=remap.get(a['program_id'], a['program_id']);bank.put('implementation', a)
        for a in target['workflow']:
            for n in a['nodes']:
                if n.get('program_id'):n['program_id']=remap.get(n['program_id'], n['program_id'])
            bank.put('workflow', a)
        bank.db.execute("INSERT OR REPLACE INTO metadata VALUES('manual_authority',?)", (VERSION,));bank.db.commit()
        save_json(root/'imported_bank.json', target)
        save_json(root/'manual_manifest.json', {'identity': identity, 'native_bank_digest': bank.digest(),
            'authority': 'author_trusted_not_learned', 'counts': {k: len(bank.all(k)) for k in KINDS}})
    finally:bank.close()
    return target


class HumanBankView:
    force_planner = False
    def __init__(self, bank, bundle, arm='manual', programs='auto', graph=None):
        if not bank.readonly or graph is not None:raise ValueError('This observation uses frozen Bank and native dynamic workflow')
        self.bank, self.bundle, self.arm, self.program_mode = bank, bundle, arm, programs
        self.state = {};self.observed = policy.progress(None, {});self.task = None
        self.view_hash = digest([VERSION, bundle, arm, programs])
        self.members = {a['id'] for k in KINDS for a in bank.all(k)}
    def __getattr__(self, name):return getattr(self.bank, name)
    def all(self, kind):return [] if self.arm=='off' else self.bank.all(kind)
    def get(self, key):return None if self.arm=='off' else self.bank.get(key)
    def bind_task(self, task):
        if self.task is None or self.task.task_id != task.task_id:
            self.state = {};self.observed = policy.progress(None, {})
        self.task = task
    def selected(self, task, phase=None, observed_progress=None):
        if self.arm=='off':return []
        phase = phase or (observed_progress or self.observed)['phase']
        names=policy.cards(phase)
        by_name={a['title']:a for a in self.bank.all('skill')}
        return [by_name[n] for n in names if n in by_name]
    def policy(self, task, phase=None):return '\n\n'.join(a['guidance'] for a in self.selected(task, phase))
    def retrieve(self, *a, **kw):return []
    def retrieve_guidance(self, *a, **kw):return []
    def planning_cards(self, *a, **kw):return []
    def select_guidance(self, *a, **kw):return {'selected': [], 'audit': []}
    def routes(self, node, **kw):return []
    def program_eligible(self, p):
        return bool(self.arm!='off' and self.program_mode!='off' and p and p.get('id') in self.members
                    and p.get('permission')=='author_trusted' and 'source' in p
                    and p['environment']==self.bank.environment)
    def program_options(self, *args, **kwargs):
        phase=self.observed['phase'];wanted=[]
        if phase=='locate':wanted.append('scoped_search.py')
        if phase=='extract' and self.observed['table_window_available']:wanted.append('table_extract.py')
        if phase=='compute':wanted.append('series_metrics.py')
        return [{'id':p['id'], 'state':'callable_manual', 'execution_permission':'author_trusted',
                 'goal':p['title'], 'input_schema':p['input_schema'], 'output_schema':p['output_schema'],
                 'result_role':p['result_role']} for p in self.bank.all('program')
                if self.program_eligible(p) and Path(p['source_file']).name in wanted
                and p['id'] not in kwargs.get('excluded', ())]


@contextmanager
def bridge():
    from atomic_skillgraph.empirical.system import EmpiricalSystem
    from atomic_skillgraph.agents.provider import OpenAICompatibleProvider
    from atomic_skillgraph.empirical.task_context import TaskContext
    original_agent, original_finish = EmpiricalSystem.agent, EmpiricalSystem.prepare_finish
    original_public=TaskContext.public_result
    def public_result(context,raw_result,*,origin=None):
        value=original_public(context,raw_result,origin=origin)
        if origin and origin.startswith('program.') and isinstance(value,dict):
            return {k:v for k,v in value.items() if k not in {'execution_receipt','worker_result','submission_preparation',
                'compatibility_contract_version','diagnostic','local_check','worker_status'}}
        return value
    def refresh(s, material=None):
        v=s.bank;v.bind_task(s.adapter.task)
        if s.checkpoint:v.state=deepcopy(s.checkpoint.state.get('human_office_evidence', v.state))
        remaining=(material or {}).get('recovery', {}).get('remaining_calls', 48)
        v.observed=policy.progress(s.task_context, v.state, remaining, task=v.task)
        return v
    def agent(s, stage, prompt, material, name, schema, **kwargs):
        if not isinstance(s.bank, HumanBankView) or s.bank.arm=='off':
            return original_agent(s, stage, prompt, material, name, schema, **kwargs)
        # The finish material was fitted by prepare_finish; never append after admission.
        if kwargs.get('decision_purpose')=='finish_only' or name=='finish_answer':
            return original_agent(s, stage, prompt, material, name, schema, **kwargs)
        if stage!='runtime':return original_agent(s, stage, prompt, material, name, schema, **kwargs)
        v=refresh(s, material);material=deepcopy(material)
        material.update(human_frontier_skill_ids=[a['id'] for a in v.selected(v.task)],
                        human_office_progress=v.observed, human_office_evidence=v.state,
                        recent_scalar_result_refs=policy.recent_scalar_result_refs(s.task_context), guidance=[])
        material.setdefault('calls', {})['programs']=v.program_options()
        if 'programs' in material:material['programs']=v.program_options()
        prompt=policy.GLOBAL_SUBMISSION_RULE+'\n\n'+v.policy(v.task)+'\n\n'+policy.NOTE+'\n\n'+prompt
        if name=='runtime_step':
            schema=deepcopy(schema);schema['properties']['evidence_state']=policy.STATE
        result=original_agent(s, stage, prompt, material, name, schema, **kwargs)
        actions=result.actions if isinstance(result, RuntimeDecision) else [result]
        for action in actions:
            if isinstance(action, dict) and 'evidence_state' in action:
                v.state=policy.validate_state(s.task_context, action['evidence_state'], v.state, task=v.task)
        if s.checkpoint:s.checkpoint.advance(s.checkpoint.state['stage'], human_office_evidence=deepcopy(v.state))
        return result
    def prepare_finish(s, materials, prompt, *, allowed_result_ids=None):
        if not isinstance(s.bank, HumanBankView) or s.bank.arm=='off':
            return original_finish(s, materials, prompt, allowed_result_ids=allowed_result_ids)
        v=refresh(s);provider=s.provider('runtime', 'finish_only')
        cap=s.resolve_call_settings('runtime','finish_only').get('max_completion_tokens',2048)
        maximum=min(s.config['runtime']['finish_input_max_bytes'], s._remaining('runtime','finish_only')-cap)
        if maximum<=0:raise ValueError('finish_unavailable:remaining_budget')
        def payload(value, completion):
            messages=[{'role':'system','content':prompt},
                      {'role':'user','content':json.dumps(value,ensure_ascii=False,allow_nan=False)}]
            wire=provider._build_payload(messages,[]) if isinstance(provider,OpenAICompatibleProvider) else {'messages':messages}
            wire[provider.config.token_limit_field if isinstance(provider,OpenAICompatibleProvider) else 'max_tokens']=completion
            return wire
        return policy.finish_evidence(s.task_context,materials,prompt,max_bytes=maximum,output_cap=cap,
            payload_builder=payload,state=v.state,policy=policy.GLOBAL_SUBMISSION_RULE+'\n'+v.policy(v.task,'finalize'),allowed_result_ids=allowed_result_ids,task=v.task)
    EmpiricalSystem.agent, EmpiricalSystem.prepare_finish = agent, prepare_finish
    TaskContext.public_result=public_result
    try:yield
    finally:
        EmpiricalSystem.agent, EmpiricalSystem.prepare_finish = original_agent, original_finish
        TaskContext.public_result=original_public
