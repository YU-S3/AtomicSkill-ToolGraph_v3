"""One declared post-Train capability batch under the formal campaign ledger."""
from copy import deepcopy
from pathlib import Path

import yaml

from ..core.errors import BudgetExhausted
from ..empirical.bank import Bank
from ..empirical.budget_governor import input_token_bound
from ..empirical.capability_batch import VERSION, PROMPT_REVISION, records
from ..empirical.contracts import digest
from . import run_choice_capability_batch as batch
from .canonical_manifest import sha256
from .formal_log import utc
from .run_empirical import code_identity, write_json

FORMAL_VERSION = 'choice-capability-batch.formal.v1'


def configure(base, spec, benchmark, seed, authority):
    declaration = spec.get('choice_batch')
    if not declaration: return base
    if benchmark != 'livemath' or seed != 42 or declaration.get('version') != FORMAL_VERSION:
        raise ValueError('The declared capability batch covers only LiveMath seed42')
    contract_path = Path(declaration['contract']).resolve()
    contract = yaml.safe_load(contract_path.read_text())
    if contract['schema'] != 'atomic.livemath.p2.selection-contract.v1' or contract['state'] != 'PREDECLARED_NOT_EXECUTED':
        raise ValueError('Unsupported capability selection contract')
    source = contract['source']
    if (source['run_seed'] != seed or source['split_seed'] != 42 or source['actual_model_request_label'] != base['llm']['model'] or
        source['generation_seed_supported'] != base['llm'].get('generation_seed_supported', False) or
        sha256(Path(authority)/benchmark/'train.json') != source['public_train_manifest_sha256']):
        raise ValueError('Formal model, seed or Train authority differs from the frozen selection contract')
    if contract['split']['size'] != {'candidate_source_train':48,'selection_screen':6,'selection_confirmation':6}:
        raise ValueError('Formal P2 requires the frozen 48/6/6 partition')
    if (contract['candidate']['max_candidates_to_selection'] != 2 or
        contract['candidate']['min_distinct_physical_train_source_cases'] != 2 or
        contract['paired_evaluation']['number_of_candidates_that_may_use_selection'] != 2):
        raise ValueError('Formal P2 retains exactly two declared two-source attempts')
    batch.policy(contract)
    budget = contract['budget_additional_authority_proposed_not_spent']
    if (budget['selection_retry_max'] != 0 or budget['unknown_billing'] != 'fail_closed' or
        budget['max_new_total_tokens'] != 1800000 or budget['max_new_physical_http_attempts'] != 80 or
        base['budget']['token_limit'] != budget['same_original_total_governor_limit_tokens'] or
        base['budget']['request_limit'] != budget['same_original_total_governor_limit_http']):
        raise ValueError('Formal P2 allowance or original cumulative budget changed')
    if spec.get('choice_bounded_finalization', False): raise ValueError('P2 keeps bounded finalization disabled')
    config = deepcopy(base)
    config['experiment'].update(choice_method_version=VERSION, choice_batch_version=FORMAL_VERSION,
        choice_batch_contract=contract, choice_batch_contract_sha256=sha256(contract_path),
        choice_selection_physical_keys=sorted(contract['split']['selected_physical_keys'].values()))
    for purpose in ('capability_induction','capability_routing'):
        config['llm'].setdefault('purpose_overrides', {})[purpose] = {
            'protocol':{'thinking_type':'disabled'},'max_completion_tokens':2048}
    return config


class BatchGovernor:
    """A bounded stage delegates every physical charge to the single campaign owner."""
    def __init__(self, governor, plan, root):
        self.owner, self.plan, self.root = governor, plan, Path(root)
        self.path, self.limits = governor.path, governor.limits
        self.prefix = batch.read(self.root/'campaign_budget_prefix.json')['attempts']
        if any(governor.state['attempts'].get(k) != v for k,v in self.prefix.items()):
            raise ValueError('Campaign billing prefix changed')
        self.save()

    @property
    def state(self): return self.owner.state

    def attempts(self): return {k:v for k,v in self.state['attempts'].items() if k not in self.prefix}

    def save(self):
        write_json(self.root/'budget.json', {'limits':{**self.plan['new_limits'],'finish_reserve':0},
            'attempts':self.attempts(),'unknown_billing':self.state['unknown_billing'],
            'campaign_budget_path':str(self.path)})

    def headroom(self, context):
        room = self.owner.headroom(context)
        used = sum(a.get('accounted_tokens',a['reserved_tokens']) for a in self.attempts().values())
        return {**room,'tokens':min(room['tokens'],self.plan['new_limits']['token_limit']-used),
            'requests':min(room['requests'],self.plan['new_limits']['request_limit']-len(self.attempts()))}

    def admit(self, audit_id, payload, context):
        room = self.headroom(context)
        reserved = input_token_bound(payload)+payload.get('max_tokens',payload.get('max_completion_tokens',0))
        if room['unknown_billing'] or room['requests'] < 1 or reserved > room['tokens']:
            raise BudgetExhausted('capability_batch_budget_exhausted','Post-Train P2 admission refused; no allowance refill')
        self.owner.admit(audit_id,payload,context);self.save()

    def complete(self, record):
        self.owner.complete(record);self.save()

    def __getattr__(self,name): return getattr(self.owner,name)


def prepare(root, config, governor, run_id):
    root = Path(root);output = root/'p2';spec = config['experiment']['choice_batch_contract']
    execution = batch.read(root/'train/execution_manifest.json');summary = batch.read(root/'train/summary.json')
    if not summary['complete'] or summary['tasks'] != 60 or execution['tasks'] != [t['task'] for t in summary['cases']]:
        raise ValueError('P2 requires all 60 fresh Train traces in their declared order')
    generation, stages = batch.partition(execution['tasks'],spec)
    bank = Bank(config['data_dir'],readonly=True,seed=42,environment=config['program_environment'])
    try:
        if bank.digest() != summary['knowledge_digest']: raise ValueError('Completed Train Bank changed before P2')
        raw = bank.train_cases()
    finally: bank.close()
    if [r['task'] for r in raw] != execution['tasks']: raise ValueError('Missing fresh Train evidence')
    by_key = {r['task']['physical_key']:r for r in raw}
    rows = [by_key[t['physical_key']] for t in generation]
    ledger = governor.state
    if ledger['unknown_billing'] or any(a['status'] != 'metered' for a in ledger['attempts'].values()):
        raise BudgetExhausted('unknown_billing','Unresolved billing before P2')
    tokens = sum(a['accounted_tokens'] for a in ledger['attempts'].values())
    budget = spec['budget_additional_authority_proposed_not_spent']
    limits = {'token_limit':budget['max_new_total_tokens'],'request_limit':budget['max_new_physical_http_attempts']}
    cfg = batch.configuration(config,output);cfg['data_dir'] = config['data_dir']
    for name,value in [('config.json',cfg),('source_cases.json',rows),('source_records.json',records(rows)),
                       ('campaign_budget_prefix.json',ledger)]: write_json(output/name,value)
    plan = {'schema':VERSION,'prompt_revision':PROMPT_REVISION,'scope_policy':batch.SCOPE_POLICY,
        'run_id':run_id+':p2','code':code_identity(),'prepared_at':utc(),'source_run':str(root),
        'source_run_id':run_id,'derivation_mode':FORMAL_VERSION,'initial_bank':'same fresh formal Train Bank; no historical assets',
        'settings':spec,'input_hashes':{str(root/n):sha256(root/n) for n in
            ('train/config.json','train/summary.json','train/execution_manifest.json')},
        'prepared_hashes':{n:sha256(output/n) for n in ('config.json','source_cases.json','source_records.json','campaign_budget_prefix.json')},
        'generation_tasks':generation,'selection_tasks':stages,
        'candidate_units':[[t['physical_key'] for t in generation[i*2:i*2+2]] for i in range(2)],
        'candidate_unit_order':'public task_id lexicographic; first bounded disjoint units; no outcome selection',
        'utility_policy':batch.policy(spec),'new_limits':limits,'inherited_tokens':tokens,'inherited_http':len(ledger['attempts']),
        'campaign_budget_prefix':{'tokens':tokens,'http':len(ledger['attempts'])},'cumulative_limits':ledger['limits'],
        'train_cases_hash':digest(raw),
        'source_train_tokens':summary['known_total_tokens'],
        'source_train_runtime_tokens':sum(a['usage']['total_tokens'] for t in summary['cases'] for r in t['requests']
            if r['stage']=='runtime' for a in r.get('http_attempts',[])),
        'status':'prepared_not_executed','new_model_http':0,'no_val_or_test_selection':True}
    write_json(output/'plan.json',plan);write_json(output/'plan_identity.json',{'plan_hash':digest(plan)})
    return plan


def finish_train(root, config, governor, log):
    root = Path(root);path = root/'p2/plan.json'
    plan = batch.read(path) if path.exists() else prepare(root,config,governor,log.manifest['run_id'])
    if plan['settings'] != config['experiment']['choice_batch_contract'] or plan['source_run_id'] != log.manifest['run_id']:
        raise ValueError('Formal P2 declaration or campaign identity changed')
    bank = Bank(config['data_dir'],readonly=True,seed=42,environment=config['program_environment'])
    try:
        if digest(bank.train_cases()) != plan['train_cases_hash']: raise ValueError('Prepared fresh Train evidence changed')
    finally:bank.close()
    if (root/'p2/result.json').exists():
        result = batch.read(root/'p2/result.json')
        if result['status'] in {'completed_selection','completed_no_candidate'}:
            batch.check(path)
            if result['plan_hash'] != digest(plan): raise ValueError('P2 result identity changed')
            frozen = batch.read(root/'train/frozen_bank/freeze.json')
            if result['frozen'] != frozen: raise ValueError('P2 published snapshot changed')
            return result
    result = batch.execute(path,None,governor=BatchGovernor(governor,plan,path.parent),observer=log,
        frozen_destination=root/'train/frozen_bank')
    if result['status'] not in {'completed_selection','completed_no_candidate'}:
        raise BudgetExhausted('capability_batch_stopped',result.get('reason',result['status']))
    return result


def carryforward(sources, limits):
    """Union metered physical attempts once; import costs, never a Bank."""
    attempts, provenance = {}, []
    for source in sources:
        path = Path(source).resolve()/'budget.json';ledger = batch.read(path)
        if ledger['unknown_billing'] or any(a['status'] != 'metered' for a in ledger['attempts'].values()):
            raise ValueError('Unresolved historical billing: '+str(path))
        for key,row in ledger['attempts'].items():
            if key in attempts and attempts[key] != row: raise ValueError('Historical physical charge changed: '+key)
            attempts[key] = row
        provenance.append({'source_budget':str(path),'sha256':sha256(path),'http':len(ledger['attempts']),
            'tokens':sum(a['accounted_tokens'] for a in ledger['attempts'].values())})
    tokens = sum(a['accounted_tokens'] for a in attempts.values())
    if tokens >= limits['token_limit'] or len(attempts) >= limits['request_limit']: raise ValueError('Original cumulative budget exhausted')
    return {'limits':{k:limits[k] for k in ('token_limit','finish_reserve','request_limit')},
        'attempts':attempts,'unknown_billing':False}, {
        'sources':provenance,'inherited_tokens':tokens,'inherited_http':len(attempts),
        'inherited_attempt_ids':sorted(attempts),'bank_inherited':False,'limits_unchanged':True}
