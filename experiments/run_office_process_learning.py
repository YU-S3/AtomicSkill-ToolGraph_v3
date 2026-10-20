"""Natural empty Bank -> all Train120 -> frozen Val24. No internal Train split."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from filelock import FileLock

from atomic_skillgraph.empirical import process_learning, public_pages
from atomic_skillgraph.empirical.budget_governor import BudgetGovernor, billing_blocked
from atomic_skillgraph.empirical.contracts import PublicTask, digest
from atomic_skillgraph.empirical.system import validate_config
from atomic_skillgraph.experiments.formal_log import FormalLog, tree_identity, utc
from atomic_skillgraph.experiments.run_empirical import code_identity, load_env, run, write_json

read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
TRAIN_TOKENS, EVAL_TOKENS = 400000, 216576
MAX_CUMULATIVE = 155000000


def resources(root):
    root = Path(root).resolve()
    manifest = read(root/'manifest.json')
    if manifest['version'] != public_pages.VERSION:
        raise ValueError('Expected the approved external public input version')
    for name, expected in manifest['files'].items():
        if sha(root/name) != expected:
            raise ValueError('Public input changed: ' + name)
    tasks = {split: [PublicTask(**r) for r in read(root/(split+'.json'))['tasks']]
             for split in ('train', 'val')}
    if [len(tasks[s]) for s in ('train', 'val')] != [120, 24]:
        raise ValueError('Formal learning uses all Train120 and Val24')
    if any(t.split != split for split, rows in tasks.items() for t in rows):
        raise ValueError('Split labels differ')
    train_ids = {t.physical_key for t in tasks['train']}
    val_ids = {t.physical_key for t in tasks['val']}
    if len(train_ids) != 120 or len(val_ids) != 24 or train_ids & val_ids:
        raise ValueError('Train and Val must be distinct canonical members')
    authority = Path(__file__).resolve().parents[1]/'data/main_experiment_v1/officeqa'
    for split, rows in tasks.items():
        canonical = read(authority/(split+'.json'))['tasks']
        if [t.task_id for t in rows] != [t['task_id'] for t in canonical]:
            raise ValueError('Public resource membership/order differs from canonical ' + split)
    return root, manifest, tasks


def configure(base, root, data, split):
    cfg = deepcopy(base)
    cfg['data_dir'] = str(root/'train'/('bank' if split == 'train' else 'frozen_bank'))
    cfg['manifest'] = str(data/(split+'.json'))
    for key in ('office_human_bank_version', 'office_effect_version', 'office_effect_manifest',
                'manual_reference_version', 'manual_reference_manifest', 'choice_selection_physical_keys'):
        cfg['experiment'].pop(key, None)
    cfg['experiment'].update(officeqa_method_version=process_learning.VERSION, benchmark='officeqa',
        shared_answer_contract='office.scalar-or-explicit-template.v1',
        runtime_mode='online' if split == 'train' else 'frozen', output_dir=str(root/split),
        input_alignment='legacy_val_internal_only' if split == 'val' else 'external_complete_public_prompt_and_evaluator')
    cfg['harness'].update(office_public_input_version=public_pages.VERSION, split=split,
                          evaluator_records=str(data/('evaluator_records_'+split+'.json')))
    for key in ('source_program_payload_max_bytes', 'builder_repair_payload_max_bytes'):
        cfg['learning'].pop(key, None)
    cfg['learning'].update(process_learning.POLICY)
    cfg['budget'].update(train_task_tokens=TRAIN_TOKENS, eval_task_tokens=EVAL_TOKENS,
                         train_solve_tokens=EVAL_TOKENS, train_learning_tokens=TRAIN_TOKENS-EVAL_TOKENS)
    cfg['llm']['runtime']['max_total_tokens_per_task'] = EVAL_TOKENS
    cfg['llm']['extractor'].update(max_total_tokens_per_task=TRAIN_TOKENS-EVAL_TOKENS,
        max_completion_tokens=8192, protocol={'thinking_type': 'disabled'})
    cfg['runtime'].update(global_action_budget=48, compact_request_json=True,
        evidence_handoff_version='public-evidence-spans.v1', finish_input_max_bytes=65536,
        dynamic_output_min_tokens=8192)
    cfg['llm']['purpose_overrides']['finish_only'] = {'protocol': {'thinking_type': 'disabled'},
                                                    'max_completion_tokens': 32768}
    return validate_config(cfg)


def prepare(root, base_config, resource_root, billing_parent):
    root, parent = Path(root).resolve(), Path(billing_parent).resolve()
    if root.exists():
        raise ValueError('Fresh natural learning requires a new run directory')
    code = code_identity()
    if code['tracked_dirty'] or not code['git_sha']:
        raise ValueError('Commit the implementation before paid preparation')
    data, manifest, tasks = resources(resource_root)
    if read(parent/'launch_status.json')['status'] != 'completed':
        raise ValueError('Billing parent must finish before a new allocation')
    ledger = read(parent/'budget.json')
    if billing_blocked(ledger):
        raise ValueError('Unsettled billing prevents allocation')
    used = sum(a.get('accounted_tokens', a['reserved_tokens']) for a in ledger['attempts'].values())
    cap = 120*TRAIN_TOKENS + 24*EVAL_TOKENS
    if used+cap > MAX_CUMULATIVE:
        raise ValueError(f'Full Train120 + Val24 needs cumulative {used+cap:,}; approved ceiling {MAX_CUMULATIVE:,}')
    base = read(base_config)
    seed = base['experiment']['seed']
    random.Random(seed).shuffle(tasks['train'])
    configs = {s: configure(base, root, data, s) for s in tasks}
    # Same runtime loop and retry formula as the frozen V4 runner, plus four learning decisions/task.
    requests = 144*(2*(48*4+16)+1)*(base['llm']['max_retries']+1)+120*4*(base['llm']['max_retries']+1)
    limits = {**ledger['limits'], 'token_limit': used+cap,
              'request_limit': len(ledger['attempts'])+requests, 'finish_reserve': 0}
    root.mkdir(parents=True)
    write_json(root/'budget_prefix.json', ledger)
    ledger['limits'] = limits
    write_json(root/'budget.json', ledger)
    for cfg in configs.values():
        cfg['budget'].update(limits)
    identity = {'method': process_learning.VERSION, 'source': code, 'seed': seed,
        'learning_members': [t.task_id for t in tasks['train']], 'train_selection_members': [],
        'validation_members': [t.task_id for t in tasks['val']], 'initial_bank': 'fresh_empty',
        'schedule': 'Train120 -> Frozen -> Val24 -> stop; no automatic Test',
        'resource_hash': sha(data/'manifest.json'),
        'val_policy': manifest.get('val_policy'), 'validation_use': 'existing local Val, internal only',
        'human_bank_loaded': False, 'test_used_for_learning': False}
    protected = {str(data/name): expected for name, expected in manifest['files'].items()}
    protected[str(data/'manifest.json')] = sha(data/'manifest.json')
    protected[str(parent/'budget.json')] = sha(parent/'budget.json')
    plan = {'identity': identity, 'configurations': configs,
            'tasks': {s: [t.__dict__ for t in rows] for s, rows in tasks.items()},
            'parent': str(parent), 'protected_files': protected, 'limits': limits,
            'new_token_cap': cap, 'created_at': utc()}
    write_json(root/'plan.json', plan)
    write_json(root/'plan_identity.json', {'sha256': sha(root/'plan.json')})
    return plan


def execute(root, env_file):
    root = Path(root).resolve()
    plan = read(root/'plan.json')
    if sha(root/'plan.json') != read(root/'plan_identity.json')['sha256']:
        raise ValueError('Run plan changed')
    if plan['identity']['source'] != code_identity():
        raise ValueError('Run source changed; use explicit versioned continuation')
    if any(sha(p) != h for p, h in plan['protected_files'].items()):
        raise ValueError('Protected inputs or parent billing changed')
    with ExitStack() as stack:
        # V4 repeats lock all ancestors. Retain the same convention.
        parent = Path(plan['parent'])
        inherited_locks = read(parent/'plan.json').get('locks', [])
        locks = {str(root/'.run.lock'), *inherited_locks}
        visited = set()
        while True:
            if parent.resolve() in visited:
                raise ValueError('Cyclic billing ancestry')
            visited.add(parent.resolve())
            locks.add(str(parent/'.run.lock'))
            prior = read(parent/'plan.json') if (parent/'plan.json').exists() else {}
            next_parent = prior.get('base_run') or prior.get('parent')
            if not next_parent or Path(next_parent).resolve() == parent.resolve():
                break
            parent = Path(next_parent)
        for path in sorted(locks):
            stack.enter_context(FileLock(path, timeout=0))
        if any(sha(p) != h for p, h in plan['protected_files'].items()):
            raise ValueError('Parent changed before lock acquisition')
        load_env(env_file)
        class Governor(BudgetGovernor):
            def admit(self, *args, **kwargs):
                parent_ledger = str(Path(plan['parent'])/'budget.json')
                if sha(parent_ledger) != plan['protected_files'][parent_ledger]:
                    raise ValueError('Parent ledger changed; refuse stale authority')
                return super().admit(*args, **kwargs)
        governor = Governor(root/'budget.json', **plan['limits'])
        log = FormalLog(root, plan['identity'], plan['configurations'],
                        resume=(root/'run_manifest.json').exists())
        results = {}
        try:
            for split, cfg in plan['configurations'].items():
                if (root/'STOP_AFTER_TASK').exists():
                    raise SystemExit(75)
                frozen = root/'train/frozen_bank'
                before = tree_identity(frozen) if split == 'val' else None
                result = run(cfg, [PublicTask(**r) for r in plan['tasks'][split]], root/split,
                    resume=(root/split/'execution_manifest.json').exists(), readonly=split == 'val',
                    budget_governor=governor, formal_log=log, order_offset=0 if split == 'train' else 120)
                results[split] = {k: result[k] for k in ('complete', 'tasks', 'successes', 'known_total_tokens')}
                if not result['complete']:
                    raise SystemExit(75)
                if split == 'val':
                    after = tree_identity(frozen)
                    if before != after:
                        raise ValueError('Frozen Bank changed during Val')
                    write_json(root/'validation.json', {'hash_before_val': before['sha256'],
                        'hash_after_val': after['sha256'], 'results': results,
                        'accuracy_target': '>65%', 'target_met_on_val': result['successes']/24 > .65,
                        'test_accuracy_claim': None, 'val_policy': plan['identity']['val_policy']})
            log.finish('completed_train_val')
        except BaseException:
            log.finish('stopped_or_failed')
            raise
    return results


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['prepare', 'run'])
    ap.add_argument('--root', required=True)
    ap.add_argument('--base-config')
    ap.add_argument('--resources')
    ap.add_argument('--billing-parent')
    ap.add_argument('--env-file', default='/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env')
    args = ap.parse_args()
    if args.action == 'prepare':
        if not all((args.base_config, args.resources, args.billing_parent)):
            ap.error('prepare requires --base-config, --resources and --billing-parent')
        prepare(args.root, args.base_config, args.resources, args.billing_parent)
    else:
        execute(args.root, args.env_file)
