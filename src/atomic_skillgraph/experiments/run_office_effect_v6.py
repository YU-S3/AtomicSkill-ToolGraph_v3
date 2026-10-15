"""Train evidence → short policy → matched utility gates. Old runs stay immutable."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from filelock import FileLock
import yaml

from ..core.errors import BudgetExhausted
from ..empirical.bank import Bank
from ..empirical.budget_governor import BudgetGovernor, input_token_bound, billing_blocked
from ..empirical.checkpoint import save_json
from ..empirical.contracts import PublicTask, digest
from ..empirical import office_effect_contract as c, office_effect_sources as sources
from ..empirical.office_effect_learning import learn, PolicyRequestStore
from ..empirical.office_effect_optimizer import evaluate_pairs
from ..empirical.office_operation_mining import read, sha
from ..empirical.office_trace_program_induction import RequestStore
from .formal_log import tree_identity, utc
from .run_empirical import code_identity, load_env, run
from .run_office_trace_v4_pilot import provider_for


def immutable(path, value):
    if path.exists():
        if read(path) != value: raise ValueError('Immutable receipt changed: '+str(path))
    else: save_json(path, value)


def prepare(root, v5, config, prior_effect=None):
    root, v5 = Path(root).resolve(), Path(v5).resolve()
    if root.exists(): raise ValueError('V6 requires an independent fresh directory')
    code = code_identity()
    if code['tracked_dirty'] or not code['git_sha']: raise ValueError('Commit the implementation before preparing paid identity')
    settings = yaml.safe_load(Path(config).read_text())
    if settings['method_version'] != c.VERSION or settings['seed'] != 42 or settings['max_candidates'] != 2:
        raise ValueError('Unsupported V6 identity')
    old = read(v5/'pilot_manifest.json'); records = read(v5/'source_records.json')
    parent = read(Path(old['v4_run'])/'pilot_manifest.json'); split = sources.partition(records)
    cfg = read(Path(parent['parent_run'])/'val/config.json')
    corpus = Path(old['corpus_root']); source = Path(old['source'])
    if any(p == root or p in root.parents or root in p.parents for p in (v5, source, corpus, Path(old['parent_bank']))):
        raise ValueError('V6 cannot overlap any protected source')
    canonical = read(read(source/'config.json')['manifest'])
    if {r['task']['task_id'] for r in records} != {t['task_id'] for t in canonical['tasks']}:
        raise ValueError('Original Train IDs differ from canonical authority')
    files = {str(Path(r['path']).resolve()):r['sha256'] for r in records}
    files.update(old['parent_files'])
    for p in [v5/'pilot_manifest.json',v5/'source_records.json',v5/'budget/usage.json',
              source/'config.json',source/'summary.json',source/'execution_manifest.json',Path(config).resolve(),
              *v5.glob('requests/**/*.json'), *v5.glob('probes/**/*.json')]: files[str(p.resolve())] = sha(p)
    for name, expected in files.items():
        if sha(name) != expected: raise ValueError('Historical source identity changed')
    if tree_identity(old['parent_bank']) != old['parent_frozen']: raise ValueError('V3 frozen changed')
    catalog = sources.build_catalog(records, split, old['corpus_identity']['sha256'])
    reserve = read(v5/'budget/sibling_reservations.json')
    ledger = read(v5/'budget/usage.json')
    if billing_blocked(ledger): raise ValueError('Unreconciled primary billing')
    prior=None
    if prior_effect:
        prior=Path(prior_effect).resolve(); prior_plan=read(prior/'manifest.json')
        if (prior_plan.get('prior_effect_run') or read(prior/'launch_status.json')['status'] not in {'failed','completed'}
                or read(prior/'partition.json')!=split or prior_plan['settings']!=settings
                or (prior/'feedback0').exists()):
            raise ValueError('Only the single stopped pre-pair first candidate can carry into this second attempt')
        previous=read(prior/'budget/usage.json')
        if billing_blocked(previous):raise ValueError('Prior V6 billing must be resolved')
        for key,row in ledger['attempts'].items():
            if previous['attempts'].get(key)!=row:raise ValueError('Prior V6 changed historical charges')
        ledger=deepcopy(previous)
        files.update({str(p):sha(p) for p in prior.rglob('*.json')})
    root.mkdir(parents=True)
    cfg['experiment'].update(office_effect_version=c.VERSION, output_dir=str(root), runtime_mode='frozen')
    cfg['data_dir'] = str(root/'empty_bank')
    bank = Bank(cfg['data_dir'], environment=cfg['program_environment'])
    try:
        bank.bind_office_method(cfg['experiment']['officeqa_method_version']); bank_hash = bank.digest()
    finally: bank.close()
    inherited = deepcopy(ledger)
    # Retain the full sibling allocation. Its used portion must not be charged a second time.
    limits = deepcopy(settings['cumulative_limits'])
    limits['token_limit'] -= sum(r['tokens'] for r in reserve)
    limits['request_limit'] -= sum(r['http'] for r in reserve)
    inherited['limits'] = limits
    outputs = {'partition.json':split,'induction.json':catalog,'source_records.json':records,
        'runtime_config.json':cfg,'budget/inherited.json':inherited,'budget/usage.json':inherited,
        'sibling_reservations.json':reserve}
    for name,value in outputs.items(): save_json(root/name,value)
    if prior:
        import shutil
        for p in (prior/'candidate0/requests').glob('notes_*.json'):
            target=root/'candidate0/requests'/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    datasets = Path(cfg['harness']['evaluator_records']).parent
    public = {}
    for phase in ('train','val','test'):
        p = datasets/(phase+'.json'); value = read(p)
        public[phase] = value['tasks'] if isinstance(value,dict) else value
        files[str(p)] = sha(p)
        evaluator = datasets/('evaluator_records_'+phase+'.json'); files[str(evaluator)] = sha(evaluator)
    if {digest(t) for t in public['train']} != {digest(r['task']) for r in records}:
        raise ValueError('Materialized Train public task identity changed')
    save_json(root/'public_tasks.json',public)
    manifest = {'method_version':c.VERSION,'material_version':c.MATERIAL,'runtime_version':c.RUNTIME,
        'selection_version':c.SELECTION,'created_at':utc(),'code':code,'settings':settings,
        'historical_files':files,'v5_run':str(v5),'prior_effect_run':str(prior) if prior else None,
        'candidate_attempts_before':1 if prior else 0,'parent_frozen':old['parent_frozen'],'parent_bank':old['parent_bank'],
        'corpus_root':str(corpus),'corpus_identity':old['corpus_identity'],'datasets':str(datasets),
        'empty_bank_digest':bank_hash,'empty_bank_identity':tree_identity(root/'empty_bank'),
        'inputs':{n:sha(root/n) for n in [*outputs,'public_tasks.json'] if n != 'budget/usage.json'},
        'inherited_tokens':sum(a.get('accounted_tokens',a['reserved_tokens']) for a in inherited['attempts'].values()),
        'inherited_http':len(inherited['attempts']),'effective_limits':limits,
        'exposure_history':'Val24 repeatedly developed; authored-reference Test exposure exists. Neither supplies learning inputs.',
        'paid_authorization':'pending'}
    save_json(root/'manifest.json',manifest); save_json(root/'manifest_identity.json',{'sha256':sha(root/'manifest.json')})
    return report(root)


def checked(root, corpus=False):
    plan = read(root/'manifest.json')
    if plan['method_version'] != c.VERSION or plan['code'] != code_identity(): raise ValueError('V6 source identity changed')
    if sha(root/'manifest.json') != read(root/'manifest_identity.json')['sha256']: raise ValueError('Manifest changed')
    for n,h in plan['inputs'].items():
        if sha(root/n) != h: raise ValueError('V6 input changed: '+n)
    for p,h in plan['historical_files'].items():
        if sha(p) != h: raise ValueError('Historical file changed: '+p)
    if tree_identity(root/'empty_bank') != plan['empty_bank_identity']: raise ValueError('Base Bank changed')
    if corpus and tree_identity(plan['corpus_root']) != plan['corpus_identity']: raise ValueError('Corpus changed')
    return plan, read(root/'runtime_config.json')


def accounting(root):
    plan = read(root/'manifest.json'); blocks = []
    authorization=read(root/'authorization.json') if (root/'authorization.json').exists() else {}
    isolated=authorization.get('isolated_siblings',{}) if authorization.get('manifest_sha256')==sha(root/'manifest.json') else {}
    for r in read(root/'sibling_reservations.json'):
        path = Path(r['run_root']); state = read(path/'budget.json')
        if sha(path/'plan.json') != r['plan_sha256']: blocks.append('sibling_identity_changed')
        exception=isolated.get(str(path))
        if exception and (exception.get('budget_sha256')!=sha(path/'budget.json') or exception.get('reserved_tokens')!=r['tokens']):
            blocks.append('isolated_sibling_changed:'+str(path))
        elif billing_blocked(state) and not exception: blocks.append('sibling_unknown_billing:'+str(path))
    state = read(root/'budget/usage.json')
    used = sum(a.get('accounted_tokens',a['reserved_tokens']) for a in state['attempts'].values())
    return {'blocks':blocks,'tokens':used,'http':len(state['attempts']),
            'remaining_tokens':state['limits']['token_limit']-used,
            'remaining_http':state['limits']['request_limit']-len(state['attempts']),
            'full_sibling_reservations':read(root/'sibling_reservations.json')}


def authorize(root, approval_path):
    """Apply a concrete human-approved finite allocation; never infer approval from a document."""
    plan,_=checked(root); approval=read(approval_path)
    if (approval.get('approved') is not True or not approval.get('user_authorization')
            or approval.get('method_version')!=c.VERSION
            or approval.get('manifest_sha256')!=sha(root/'manifest.json')):
        raise ValueError('A matching explicit V6 authorization receipt is required')
    phases=approval.get('approved_phases',[])
    if not phases or set(phases)-plan['settings']['phase_limits'].keys(): raise ValueError('Invalid approved phases')
    if 'test' in phases and not approval.get('explicit_test_authorization'):raise ValueError('Explicit Test approval required')
    state=read(root/'budget/usage.json')
    previous=read(root/'authorization.json') if (root/'authorization.json').exists() else None
    if previous and not set(previous['approved_phases'])<=set(phases):raise ValueError('Do not erase prior phase authority or spending')
    shift=approval.get('reallocate_sibling_http',0)
    reserve=read(root/'sibling_reservations.json')
    if type(shift) is not int or not 0<=shift<=sum(r['http'] for r in reserve):raise ValueError('Invalid HTTP reallocation')
    isolated={}
    for row in reserve:
        p=Path(row['run_root']); sibling=read(p/'budget.json')
        if billing_blocked(sibling) or shift:
            if not approval.get('isolate_stopped_siblings_under_full_reserve'):
                raise ValueError('Unsettled/paused sibling needs explicit isolation under its entire reservation')
            status=read(p/'launch_status.json')
            if status.get('status') not in {'failed','stopped','completed'}:raise ValueError('Sibling is not stopped')
            tasks=read(p/'tasks.json'); count=len(tasks['tasks'] if isinstance(tasks,dict) else tasks)
            sibling_cfg=read(p/'config.json'); calls=sibling_cfg['runtime']['global_action_budget']
            loop_bound=(2*max(16,4*calls+16)+5)*(sibling_cfg['llm']['max_retries']+1)*count
            if row['http']-shift<loop_bound:raise ValueError('Reallocation would consume the sibling full theoretical request allowance')
            isolated[str(p)]={'budget_sha256':sha(p/'budget.json'),'reserved_tokens':row['tokens'],
                'unknown_attempts':{k:digest(a) for k,a in sibling['attempts'].items() if a['status']!='metered'},
                'unknown_reserved_tokens':sum(a['reserved_tokens'] for a in sibling['attempts'].values() if a['status']!='metered'),
                'full_request_loop_bound':loop_bound,
                'rule':'No replay or fabricated usage. Entire sibling token ceiling remains reserved; any ledger change stops V6.'}
    limits={**plan['effective_limits'],'request_limit':plan['effective_limits']['request_limit']+shift}
    caps=[plan['settings']['phase_limits'][p] for p in phases]
    if (sum(c['tokens'] for c in caps)+plan['inherited_tokens']>limits['token_limit']
            or sum(c['http'] for c in caps)+plan['inherited_http']>limits['request_limit']):
        raise ValueError('Approved phases do not fit cumulative authority and sibling reservations')
    receipt={**approval,'phase_limits':plan['settings']['phase_limits'],'effective_limits':limits,'isolated_siblings':isolated,
        'prior_usage_sha256':sha(root/'budget/usage.json'),'full_sibling_token_reservations':sum(r['tokens'] for r in reserve)}
    immutable(root/'authorization_history'/(digest(receipt)+'.json'),receipt)
    save_json(root/'authorization.json',receipt)
    state['limits']=limits;save_json(root/'budget/usage.json',state)
    return report(root)


class PhaseGovernor(BudgetGovernor):
    def __init__(self, root, phase):
        self.root, self.phase = Path(root), phase
        self.plan = read(self.root/'manifest.json'); auth = read(self.root/'authorization.json')
        if (auth.get('manifest_sha256') != sha(self.root/'manifest.json') or not auth.get('user_authorization')
                or phase not in auth.get('approved_phases',[]) or auth.get('phase_limits') != self.plan['settings']['phase_limits']):
            raise ValueError('Matching phase authorization is required')
        check = read(self.root/'zero_http_check.json')
        if not check.get('passed') or check.get('manifest_sha256') != sha(self.root/'manifest.json'):
            raise ValueError('Matching production offline gate is required')
        if accounting(self.root)['blocks']: raise ValueError('Unresolved sibling accounting; preserve reservations')
        self.cap = self.plan['settings']['phase_limits'][phase]
        super().__init__(self.root/'budget/usage.json',**auth.get('effective_limits',self.plan['effective_limits']))

    def headroom(self, context):
        room = super().headroom(context)
        owned = [a for a in self.state['attempts'].values() if a['context'].get('office_effect_phase') == self.phase]
        room['tokens'] = min(room['tokens'],self.cap['tokens']-sum(a.get('accounted_tokens',a['reserved_tokens']) for a in owned))
        room['requests'] = min(room['requests'],self.cap['http']-len(owned))
        return room

    def admit(self, audit_id, payload, context):
        if accounting(self.root)['blocks']: raise BudgetExhausted('sibling_billing_blocked','Shared Office accounting requires reconciliation')
        with self.lock:
            context = {**context,'office_effect_phase':self.phase,'office_effect_run':self.root.name}
            room = self.headroom(context)
            reserve = input_token_bound(payload)+payload.get('max_tokens',payload.get('max_completion_tokens',0))
            if reserve > room['tokens'] or room['requests'] <= 0:
                raise BudgetExhausted('effect_phase_budget_exhausted','V6 stage or cumulative allowance exhausted')
            super().admit(audit_id,payload,context)


def induce(root, edit=False):
    plan,cfg = checked(root,corpus=True); governor = PhaseGovernor(root,'learning')
    folder = root/('candidate1' if edit else 'candidate0')
    parent = feedback = None
    if edit:
        if plan.get('candidate_attempts_before',0):raise ValueError('Both finite candidate opportunities have already been used')
        parent = read(root/'candidate0/learning.json')['candidate']
        feedback = read(root/'feedback0/pairs.json')
        if feedback['selection']['accepted']: raise ValueError('Positive first candidate needs no further edit')
        if not parent: raise ValueError('No candidate to edit')
        if {r['physical_key'] for r in feedback['pairs']} != set(read(root/'partition.json')['pilot']):
            raise ValueError('Only newly paired preregistered edit-feedback cases may reach the editor')
        feedback = deepcopy(feedback)
        for row in feedback['pairs']:
            for arm in ('off','on'):
                path = Path(row[arm]['trace_path']); trace = read(path)
                if digest(trace) != row[arm]['trace_sha256']: raise ValueError('New feedback trace changed')
                row[arm]['observed_experience'] = sources.learning_projection(
                    sources.material(trace,sha(path),plan['corpus_identity']['sha256']))
    client = PolicyRequestStore(folder/'requests',provider_for(cfg,'extractor',governor),phase='learning',
                          wire_max_bytes=plan['settings']['wire_max_bytes'])
    result = learn(folder,client,read(root/'induction.json'),parent=parent,feedback=feedback)
    immutable(root/'learner_memory'/('edit.json' if edit else 'induction.json'),result)
    return result


def arm_config(root, folder, cfg, task, candidate, enabled, bank_hash):
    config = deepcopy(cfg); view = {'method_version':c.VERSION,'candidate':candidate,'enabled':enabled,'bank_digest':bank_hash}
    immutable(folder/'view.json',view)
    config['experiment'].update(output_dir=str(folder),office_effect_manifest=str(folder/'view.json'))
    # Train-held-out episodes are evaluation, so their parent cap is the same 192K as Val/Test.
    config['harness']['evaluator_records'] = str(Path(cfg['harness']['evaluator_records']).parent/('evaluator_records_'+task.split+'.json'))
    return config


def episode(root, phase, folder, cfg, task, candidate, enabled, plan):
    config = arm_config(root,folder,cfg,task,candidate,enabled,plan['empty_bank_digest'])
    governor = PhaseGovernor(root,phase)
    summary = run(config,[task],folder,resume=(folder/'execution_manifest.json').exists(),readonly=True,budget_governor=governor)
    trace = summary['cases'][0]
    scope = str(folder)+':'+task.physical_key
    attempts = [a for a in governor.state['attempts'].values() if a['context'].get('parent_scope') == scope]
    unknown = any(a['status'] != 'metered' for a in attempts)
    def exposed(request):
        from ..empirical.model_view import expand_material
        messages = request.get('messages',[])
        material = expand_material(json.loads(messages[1]['content'])) if len(messages)>1 and isinstance(messages[1]['content'],str) else {}
        return material.get('active_guidance_ids',material.get('recovery',{}).get('active_guidance_ids',[]))
    exposure = [{'stage':r.get('stage'),'ids':exposed(r)} for r in trace['requests']]
    return {'score':trace['score'],'prediction':trace['execution'].get('prediction'),
        'empty':not bool(str(trace['execution'].get('prediction') or '').strip()),'complete':summary['complete'],
        'unknown_billing':unknown,'tokens':sum(a.get('accounted_tokens',a['reserved_tokens']) for a in attempts),
        'http':len(attempts),'tool_calls':sum(e.get('tool_call_consumed',False) for e in trace['tools']),
        'terminal_reason':trace['execution'].get('reason'),'exposure':exposure,
        'trace_path':str(folder/'traces'/(task.task_id+'.json')),'trace_sha256':digest(trace),
        'execution_manifest_sha256':sha(folder/'execution_manifest.json')}


def protocol_identity(path):
    value = read(path); cfg = value['config']
    for key in ('output_dir','office_effect_manifest'): cfg['experiment'].pop(key,None)
    return digest(value)


def read_candidate(root, index):
    folder = root/('candidate'+str(index)); result = read(folder/'learning.json')
    candidate = result['candidate']
    if not candidate: return None
    c.verify_candidate(candidate)
    path = folder/'requests'/('repair.json' if (folder/'requests/repair.json').exists() else 'proposal.json')
    response = read(path)
    if (digest(response) != result['response_sha256'] or PolicyRequestStore.decode_policy(response) != result['proposal']
            or response['payload_sha256'] != digest(response['payload'])):
        raise ValueError('Policy must match its natural paid model evidence')
    parent = read_candidate(root,0) if index else None
    if c.validate_candidate(result['proposal'],result['source_ids'],parent) != candidate:
        raise ValueError('Candidate differs from its admitted source and one-step edit')
    return candidate


def compare(root, phase, candidate_index=0):
    plan,cfg = checked(root,corpus=True); split = read(root/'partition.json'); public = read(root/'public_tasks.json')
    if phase == 'feedback':
        keys = split['pilot']; name = 'feedback'+str(candidate_index)
        candidate = read_candidate(root,candidate_index)
        budget_phase = 'revision' if candidate_index else 'feedback'
    else:
        lock = read(root/'selected.json'); candidate = lock['candidate']; name = phase; budget_phase = phase
        if candidate != read_candidate(root,lock['candidate_index']): raise ValueError('Selected policy changed')
        feedback_path = root/('feedback'+str(lock['candidate_index']))/'pairs.json'
        if sha(feedback_path) != lock['feedback_receipt_sha256'] or not read(feedback_path)['selection']['accepted']:
            raise ValueError('Selected policy lacks positive feedback gate')
        keys = split['train_gate'] if phase == 'gate' else [t['physical_key'] for t in public[phase]]
        if phase in {'val','test'}:
            frozen=read(root/'freeze.json')
            if (frozen['candidate_hash'] != digest(candidate) or frozen['candidate'] != candidate
                    or frozen['code'] != plan['code'] or frozen['gate_sha256'] != sha(root/'gate/pairs.json')
                    or frozen['base_bank_hash'] != plan['empty_bank_identity']):
                raise ValueError('Frozen evaluation identity changed')
        if phase == 'test':
            auth = read(root/'authorization.json')
            if not auth.get('explicit_test_authorization'): raise ValueError('Test needs explicit separate authorization')
            if not read(root/'val/pairs.json')['selection']['accepted']: raise ValueError('No positive Val utility gate')
    if candidate is None:
        save_json(root/name/'status.json',{'status':'completed_no_candidate'}); return report(root)
    tasks = {t['physical_key']:t for rows in public.values() for t in rows}
    pairs = []
    for i,key in enumerate(keys):
        task = PublicTask(**tasks[key]); arms = {}
        # Alternating order fixed before outcomes; revised candidate reuses same-protocol off receipt.
        for arm in (('off','on') if i % 2 == 0 else ('on','off')):
            folder = root/name/key/arm
            if phase == 'feedback' and candidate_index and arm == 'off':
                old = read(root/'feedback0/pairs.json'); arms[arm] = next(r['off'] for r in old['pairs'] if r['physical_key']==key)
            else: arms[arm] = episode(root,budget_phase,folder,cfg,task,candidate,arm=='on',plan)
        protocols = [protocol_identity(Path(arms[a]['trace_path']).parents[1]/'execution_manifest.json') for a in ('off','on')]
        pairs.append({'physical_key':key,'task_id':task.task_id,'same_protocol':protocols[0] == protocols[1],
            'bank_unchanged':tree_identity(root/'empty_bank')==plan['empty_bank_identity'],**arms})
        save_json(root/name/'progress.json',{'completed_pairs':len(pairs),'total_pairs':len(keys),'pairs':pairs})
    selection = evaluate_pairs(pairs,plan['settings']['max_losses'])
    value = {'candidate':candidate,'pairs':pairs,'selection':selection,'code':plan['code'],'partition_sha256':sha(root/'partition.json')}
    immutable(root/name/'pairs.json',value); immutable(root/'learner_memory'/(name+'.json'),value)
    if phase == 'feedback' and selection['accepted']:
        immutable(root/'selected.json',{'candidate':candidate,'candidate_index':candidate_index,'feedback_receipt_sha256':sha(root/name/'pairs.json')})
    return report(root)


def freeze(root):
    plan,_ = checked(root,corpus=True); gate = read(root/'gate/pairs.json'); selected = read(root/'selected.json')
    if not gate['selection']['accepted'] or gate['candidate'] != selected['candidate']: raise ValueError('Positive independent Train gate required')
    existing = read(root/'freeze.json') if (root/'freeze.json').exists() else {}
    immutable(root/'freeze.json',{'method_version':c.VERSION,'created_at':existing.get('created_at',utc()),'candidate':selected['candidate'],
        'candidate_hash':digest(selected['candidate']),'gate_sha256':sha(root/'gate/pairs.json'),
        'base_bank_hash':plan['empty_bank_identity'],'code':plan['code']})
    return report(root)


def report(root):
    plan = read(root/'manifest.json'); result = {'method_version':c.VERSION,'run_root':str(root),'code':plan['code'],
        'accounting':accounting(root),'candidate_count':0,'comparisons':{},'exposure_history':plan['exposure_history']}
    for folder in root.glob('candidate[01]'):
        if (folder/'learning.json').exists(): result['candidate_count'] += bool(read(folder/'learning.json')['candidate'])
    for phase in ('feedback0','feedback1','gate','val','test'):
        p = root/phase/'pairs.json'
        if p.exists(): result['comparisons'][phase] = read(p)['selection']
    result['status'] = ('utility_candidate_selected' if (root/'selected.json').exists() else
        'completed_no_positive_candidate' if result['comparisons'] else 'candidate_ready_for_feedback' if result['candidate_count'] else
        'completed_no_candidate' if (root/'candidate0/learning.json').exists() else 'prepared_zero_http')
    save_json(root/'report.json',result); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','check','authorize','pilot','induce','feedback','edit','revision','gate','freeze','val','test','report'])
    parser.add_argument('--output',required=True); parser.add_argument('--v5-run'); parser.add_argument('--config')
    parser.add_argument('--env-file')
    parser.add_argument('--approval')
    parser.add_argument('--prior-effect-run')
    args = parser.parse_args(); root = Path(args.output).resolve()
    if args.env_file: load_env(args.env_file)
    if args.stage == 'prepare': result = prepare(root,args.v5_run,args.config,args.prior_effect_run)
    else:
        with FileLock(str(root/'.v6.lock'),timeout=0):
            if args.stage == 'check':
                from .office_effect_offline import check
                result = check(root)
            elif args.stage=='authorize':result=authorize(root,args.approval)
            elif args.stage=='pilot':
                result=induce(root)
                if result['candidate']:
                    result=compare(root,'feedback')
                    if not (root/'selected.json').exists() and 'revision' in read(root/'authorization.json')['approved_phases']:
                        revised=induce(root,edit=True)
                        if revised['candidate']:result=compare(root,'feedback',1)
            elif args.stage in {'induce','edit'}: result = induce(root,args.stage=='edit')
            elif args.stage == 'revision': result = compare(root,'feedback',1)
            elif args.stage in {'feedback','gate','val','test'}: result = compare(root,args.stage)
            elif args.stage == 'freeze': result = freeze(root)
            else: result = report(root)
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__': main()
