"""P6 experimental execution permission requires local receipts and current public binding."""
from copy import deepcopy
import json
from pathlib import Path

from . import math_operation_contract_p6 as c
from .contracts import digest


class TrialView:
    def __init__(self, bank, manifest):
        self.bank=bank;self.manifest=json.loads(Path(manifest).read_text());self.router=self
        if not bank.readonly or self.manifest['version']!=c.VERSION or self.manifest['bank_digest']!=bank.digest():
            raise ValueError('P6 trial Bank identity differs')
        self.programs=self.manifest['program_hashes'];self.routes={};self.audit={}
        if set(self.programs)!={p['id'] for p in bank.all('program')}:raise ValueError('P6 trial Program membership changed')
        for p in bank.all('program'):
            if digest(p)!=self.programs[p['id']] or p['state']!='candidate' or p['method_version']!=c.VERSION:
                raise ValueError('P6 trial must not import human assets or forge usable state')
        for receipt in self.manifest['evidence_files']:
            path=Path(receipt['path']);content=path.read_bytes()
            import hashlib
            if hashlib.sha256(content).hexdigest()!=receipt['sha256']:raise ValueError('P6 generation/qualification evidence changed')

    def __getattr__(self, name):return getattr(self.bank,name)

    def bind(self, task, context=None, *, enabled=True):
        from dataclasses import asdict
        self.task=task;self.context=context;self.routes={};self.enabled=enabled
        for p in self.bank.all('program'):
            kind=p['entry_constraints']['operation_kind'];grounding=c.ready_bindings(asdict(task),kind)
            self.routes[p['id']]={'status':'ready' if grounding else 'requires_grounding',
                'reason':None if grounding else 'unbound_semantics','grounding':grounding}
        ready=[p for p in sorted(self.routes) if self.routes[p]['status']=='ready'][:2] if enabled else []
        self.visible=ready;skills=[]
        for p in ready:
            for sid in self.manifest['program_skills'][p]:
                a=self.bank.get(sid)
                if a and a not in skills:skills.append(a)
        self.audit={'version':c.VERSION,'candidate_visible':ready,'applicable':ready,'inputs_grounded':ready,
            'routes':deepcopy(self.routes),'qualification':'experimental_trial_only_not_proven_utility'}
        return skills[:2]

    def program_eligible(self, p):
        return bool(p and p['id'] in getattr(self,'visible',[]) and digest(p)==self.programs[p['id']])

    def program_options(self, query, **kwargs):
        result=[]
        for pid in getattr(self,'visible',[]):
            p=self.bank.get(pid);g=self.routes[pid]['grounding'];io=c.interface(p['entry_constraints']['operation_kind'])
            result.append({'id':pid,'state':'candidate','qualification':'local_algorithm_checked',
                'execution_permission':'experimental_trial_only','callable_now':True,'task_binding':'ready',
                'input_schema':p['input_schema'],'output_schema':p['output_schema'],'result_role':'intermediate',
                'verified_public_binding':g['grounded_task_inputs'],'binding_refs':g['public_source_refs'],
                'proof_scope':io['proof_scope'],'operation_semantics':io['semantics']})
        return result

    def validate_call(self, pid, args):
        if pid not in getattr(self,'visible',[]) or args!=self.routes[pid]['grounding']['grounded_task_inputs']:
            raise ValueError('P6 requires the exact Host-bound current public inputs')


def validate_config(config):
    e=config['experiment']
    if (e.get('math_program_evaluation_version')!=c.VERSION or e.get('math_program_method_version')!=c.VERSION
            or e.get('runtime_mode')!='frozen' or e.get('manual_reference_version') or not e.get('math_program_trial_manifest')):
        raise ValueError('Invalid P6 isolated trial evaluation identity')
    from .math_capability import validate_runtime_contract
    from .public_process import TRUNCATION_FINALIZATION_VERSION
    if config['runtime']['choice_bounded_finalization']!={'enabled':True,'version':c.FINALIZATION}:
        raise ValueError('P6 requires its bounded failure recovery contract')
    shared=deepcopy(config)
    shared['runtime']['choice_bounded_finalization']['version']=TRUNCATION_FINALIZATION_VERSION
    validate_runtime_contract(shared)


def usage_layers(execution, requests, local_reads):
    rows=[]
    for a in execution['attempts']:
        rid=a['result_id']
        rows.append({'program_id':a['program_id'],'worker_invoked':True,'worker_ok':a['status']=='ok',
            'result_exposed':any(r.get('response') and any(rid in m.get('content','') for m in r.get('messages',[]) if isinstance(m.get('content'),str)) for r in requests),
            'result_read':any(r.get('source_ref',{}).get('result_id')==rid for r in local_reads),
            'used_results_declared':a['outputs_consumed'],
            'downstream_handoff':any(ref.get('result_id')==rid for later in execution['attempts']+execution.get('temporary_executions',[])
                for ref in later.get('argument_refs',{}).values()),
            'task_correct':None,'qualification':'explicit_consumption_not_independent_utility_proof'})
    return rows
