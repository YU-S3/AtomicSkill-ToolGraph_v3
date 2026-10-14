"""Read-only evaluation of naturally generated, Docker-probed mathematical candidates."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from . import math_program_induction as m
from . import math_program_probe as probe
from .bank import Bank
from .contracts import digest, program_digest

VERSION='livemath.math-program.trace-val.v1'


def read(path):return json.loads(Path(path).read_text())


def verified_bundles(root):
    """Resolve every admitted candidate to original Extractor, Builder and Worker records."""
    root=Path(root);plan=read(root/'pilot_manifest.json');cfg=read(root/'config.json');result=[]
    for op in read(root/'selected_opportunities.json'):
        folder=root/'learning'/op['opportunity_id']
        if not (folder/'probe.json').exists():continue
        tested=read(folder/'probe.json')
        if not tested.get('probe_passed'):continue
        row=read(folder/'learning.json');bundle=row['bundle'];p=bundle['program'];a=bundle['atomic']
        proposal=m.validate_spec(row['proposal'],op)
        if a!=m.atomic(proposal,op) or not bundle['autonomously_generated'] or bundle['usable']:
            raise ValueError('Trial candidate has no matching natural Atomic source')
        rebuilt=m.program(p['source'],proposal['program_spec'],a,cfg['program_environment'],p['entry_constraints']['parent_program_id'])
        rebuilt['id']='program_'+program_digest(rebuilt)
        if rebuilt!=p or bundle['implementation']['skill_id']!=a['id'] or bundle['implementation']['program_id']!=p['id']:
            raise ValueError('Trial Program or Implementation changed')
        requests=read(folder/'requests.json')['requests']
        def received(name,value,purpose):
            return any(r.get('purpose')==purpose and r.get('response',{}).get('usage') and
                r.get('tool_choice')=={'type':'function','function':{'name':name}} and
                any(t['name']==name and t['arguments']==value for t in r['response'].get('tool_calls',[])) for r in requests)
        if not received('submit_math_program_spec',proposal,'math_program_spec') or not received('submit_program',{'source':p['source']},'math_program_build'):
            raise ValueError('Candidate has no exact saved real model generation response')
        expected=probe.probe_inputs(proposal['program_spec'])
        if len(tested['receipts'])!=3:raise ValueError('Missing Worker trials')
        for receipt,(role,arguments) in zip(tested['receipts'],expected):
            saved=read(folder/'worker'/(receipt['id']+'.json'))
            out=receipt['result'];execution=out.get('execution_receipt',{})
            valid=out['status'] in {'needs_input','not_found','blocked'} if role=='missing_premise' else (
                out['status']=='ok' and out.get('outputs')==probe.oracle(proposal['program_spec']['probe_kind'],arguments))
            if (receipt!=saved or receipt['receipt_hash']!=digest({k:v for k,v in receipt.items() if k!='receipt_hash'})
                    or receipt['role']!=role or receipt['arguments']!=arguments or receipt['program_digest']!=program_digest(p)
                    or receipt['rpc_calls'] or not receipt['passed'] or not valid or execution.get('exit_code')!=0
                    or execution.get('image_digest')!=m.IMAGE or '--network=none' not in execution.get('command',[])
                    or execution.get('source_sha256')!=hashlib.sha256(p['source'].encode()).hexdigest()
                    or execution.get('input_hash')!=digest(arguments)
                    or execution.get('result_hash')!=digest({k:v for k,v in out.items() if k!='execution_receipt'})):
                raise ValueError('Trial candidate has no matching real Docker evidence')
        result.append({'bundle':bundle,'probe':tested,'opportunity_id':op['opportunity_id'],
            'source_run_id':plan['run_id'],'qualification':'train_trial_only_not_usable'})
    return result


class TrialView:
    def __init__(self,bank,manifest):
        if not bank.readonly:raise ValueError('Candidate evaluation requires a read-only Bank')
        self.bank=bank;manifest=read(manifest)
        if manifest['version']!=VERSION or manifest['bank_digest']!=bank.digest():raise ValueError('Trial Bank identity changed')
        self.programs=manifest['program_hashes']
        if set(self.programs)!={p['id'] for p in bank.all('program')}:raise ValueError('Trial catalog changed')
        if any(digest(bank.get(k))!=v for k,v in self.programs.items()):raise ValueError('Trial Program content changed')
    def __getattr__(self,name):return getattr(self.bank,name)
    def program_eligible(self,p):return bool(p and p['id'] in self.programs and digest(p)==self.programs[p['id']])
    def retrieve_guidance(self,query,limit=3):
        words=self.words(query)
        return sorted(self.all('skill'),key=lambda a:(-len(words&self.words(a['goal']+' '+a['applicability'])),a['id']))[:limit]
    def guidance_retrieval_audit(self,query,assets):
        return {'policy':'fixed_top3_public_goal_overlap_then_id','qualification':'train_trial_only',
            'selected_ids':[a['id'] for a in assets]}
    def program_options(self,query,**kwargs):
        cards=Bank.program_options(self,query,**kwargs)
        for card in cards:
            p=self.get(card['id']);spec=p['entry_constraints']['spec'];a=self.get(p['entry_constraints']['atomic_skill_id'])
            card.update(state='train_trial_only',entry_constraints={k:a[k] for k in ('applicability','prerequisites','stop_condition','limitations')},
                operation_semantics=probe.semantics(spec['probe_kind']))
        return cards


def render(a):
    return {k:deepcopy(a[k]) for k in ('id','goal','applicability','prerequisites','operation','outputs','stop_condition','limitations')}


def validate_config(config):
    e=config['experiment']
    if (e.get('math_program_evaluation_version')!=VERSION or e.get('math_program_method_version')!=m.VERSION
            or e.get('runtime_mode')!='frozen' or e.get('manual_reference_version') or not e.get('math_program_trial_manifest')):
        raise ValueError('Invalid independently recorded mathematical candidate evaluation')
    from .math_capability import validate_runtime_contract
    validate_runtime_contract(config)
