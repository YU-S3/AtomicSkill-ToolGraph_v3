"""Fresh V6/NoSkill Val24 pairs; internal validation with the approved local Val."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import subprocess
import sys

from filelock import FileLock
import run_office_humanstrong_v1 as common
from run_office_humanstrong_v2 import load

PACKAGE = common.SOURCE/'resources/office_humanstrong_v6'
VERSION = 'office.humanstrong.v6.internal-val24-pairs.20261021'
TASK_CAP = 216576
CUMULATIVE_CAP = 190000000


def bind_runner():
    common.PACKAGE = PACKAGE
    common.frontier = load('office_v6_paired_frontier', PACKAGE/'integration/frontier.py')
    common.VERSION = VERSION


def settled_parent(base):
    plan = common.read(base/'plan.json')
    if common.read(base/'launch_status.json')['status'] != 'completed':
        raise ValueError('Billing parent must be completed')
    ledger = common.read(base/'budget.json')
    prefix = common.read(base/'budget_prefix.json')['attempts']
    if common.billing_blocked(ledger) or any(
            a['status'] != 'metered' for k, a in ledger['attempts'].items() if k not in prefix):
        raise ValueError('Parent has unsettled new requests')
    for episode in plan['episodes']:
        summary = common.read(Path(episode['path'])/'summary.json')
        if not summary['complete'] or summary['tasks'] != 1:
            raise ValueError('Incomplete parent episode')
    locks, seen, parent = set(), set(), base
    while True:
        if parent in seen:
            raise ValueError('Cyclic billing ancestry')
        seen.add(parent)
        locks.add(str(parent/'.run.lock'))
        previous = common.read(parent/'plan.json')
        locks.update(previous.get('locks', []))
        ancestor = previous.get('base_run') or previous.get('parent')
        if not ancestor:
            break
        parent = Path(ancestor).resolve()
    return plan, ledger, sorted(locks)


def prepare(root, base, resources, config_run):
    root, base, resources, config_run = [Path(p).resolve() for p in (root, base, resources, config_run)]
    if root.exists():
        raise ValueError('Use a fresh independent directory')
    code = common.code_identity()
    if code['tracked_dirty'] or not code['git_sha']:
        raise ValueError('Commit before preparing a paid run')
    parent, ledger, locks = settled_parent(base)
    used = sum(map(common.charge, ledger['attempts'].values()))
    configuration = load('office_v6_paired_config', PACKAGE/'integration/configuration.py')
    resources, manifest = configuration.verify_resources(resources)
    tasks = common.read(resources/'val.json')['tasks']
    canonical_path = common.SOURCE/'data/main_experiment_v1/officeqa/val.json'
    canonical = common.read(canonical_path)['tasks']
    if len(tasks) != 24 or any(t['split'] != 'val' for t in tasks):
        raise ValueError('All 24 Val tasks required')
    if [t['task_id'] for t in tasks] != [t['task_id'] for t in canonical]:
        raise ValueError('Val membership/order changed')
    if {t['task_id'] for t in tasks} & {
            t['task_id'] for split in ('train', 'test')
            for t in common.read(resources/(split+'.json'))['tasks']}:
        raise ValueError('Val overlaps Train/Test')
    token_cap = len(tasks)*2*TASK_CAP
    if used+token_cap > CUMULATIVE_CAP:
        raise ValueError('Insufficient cumulative authority')
    cfg = configuration.configure(common.read(config_run/'base_config.json'), resources, 'val')
    assert cfg['runtime']['global_action_budget'] == 48
    assert cfg['llm']['runtime']['max_total_tokens_per_task'] == TASK_CAP
    assert cfg['experiment']['office_human_bank_version'] == common.frontier.VERSION
    decisions = max(16, cfg['runtime']['global_action_budget']*4+16)
    http_cap = len(tasks)*2*(decisions*2+1)*(cfg['llm']['max_retries']+1)
    cfg['experiment'].update(runtime_mode='frozen', seed=42, output_dir=str(root), observation_version=VERSION)
    cfg['data_dir'] = str(root/'episodes')
    root.mkdir(parents=True)
    common.save_json(root/'budget_prefix.json', ledger)
    limits = {**ledger['limits'], 'token_limit': used+token_cap,
              'request_limit': len(ledger['attempts'])+http_cap, 'finish_reserve': 0}
    ledger['limits'] = limits
    common.save_json(root/'budget.json', ledger)
    cfg['budget'].update(token_limit=limits['token_limit'], request_limit=limits['request_limit'],
                         train_task_tokens=TASK_CAP, eval_task_tokens=TASK_CAP)
    common.write_json(root/'base_config.json', cfg)
    common.save_json(root/'tasks.json', {'tasks': tasks})
    bundle = common.read(PACKAGE/'bank/bank.json')
    episodes, rng = [], random.Random(42)
    for i, task in enumerate(tasks):
        arms = ['off', 'manual']
        rng.shuffle(arms)
        for arm in arms:
            out = root/'episodes'/f"{i+1:03}_{task['task_id'].split(':')[-1]}"/arm
            common.frontier.native_import(bundle, out/'bank', cfg['program_environment'],
                                          cfg['experiment']['officeqa_method_version'], 42)
            actual = deepcopy(cfg)
            actual['data_dir'], actual['experiment']['output_dir'] = str(out/'bank'), str(out)
            common.write_json(out/'config.json', actual)
            episodes.append({'task_id': task['task_id'], 'arm': arm, 'path': str(out),
                'config_hash': common.sha(out/'config.json'), 'bank_identity': common.tree_identity(out/'bank')})
    protected = [base/'budget.json', base/'plan.json', base/'launch_status.json',
        base/'comparison.json', base/'billing_result.json', config_run/'base_config.json',
        config_run/'plan.json', resources/'manifest.json', canonical_path, root/'tasks.json',
        root/'base_config.json', root/'budget_prefix.json']
    protected += [resources/name for name in manifest['files']]
    plan = {'created_at': common.utc(), 'code': code, 'method_version': VERSION,
        'package_identity': common.package_identity(), 'base_run': str(base),
        'inference_config_source': str(config_run), 'resources': str(resources),
        'authorization': '2026-10-21 user requested implementing the proposed Bank revision and launching related experiments. '
            'Use the previously approved cumulative 190M ceiling for one fresh V6 Bank/NoSkill Val24 pair. '
            'Local Val matched by ID was previously approved for internal checks. '
            'Retain all historical charges and unknown reservations. No Train split, Test or automatic repeats.',
        'selection': 'all 24 canonical Val members in canonical order; seed42 fixes paired arm order',
        'development_exposed': True, 'prior_scores_used_for_selection': False,
        'comparison_caveat': 'Internal local Val, with source hints and corpus retrieval but no supplied external page bodies. '
            'This does not establish external full-public-page Test parity or paper performance. Human-authored reference Bank. '
            'Both fresh arms share public input, scorer, native tools, task budget and output contract. '
            'Not a blind or naturally learned Ours result.',
        'input_alignment': manifest['val_policy'],
        'test_runs_authorized_in_this_plan': 0,
        'seed': 42, 'episodes': episodes,
        'corpus_identity': common.read(config_run/'plan.json')['corpus_identity'],
        'protected_files': {str(p): common.sha(p) for p in protected}, 'ledger_limits': limits,
        'authorized_cumulative_ceiling': CUMULATIVE_CAP, 'inherited_used_or_reserved': used,
        'new_token_cap': token_cap, 'new_http_cap': http_cap, 'fresh_paired_episodes': len(episodes),
        'http_cap_basis': {'decisions_per_episode': decisions, 'requests_per_decision_including_repair': 2,
            'final_requests': 1, 'physical_attempts_per_request': cfg['llm']['max_retries']+1},
        'asset_origin': 'human_authored_reference_not_learned_ours',
        'workflow_execution_mode': 'native_dynamic_tool_loop_with_explicit_final_program_submission',
        'shared_answer_contract': common.frontier.ANSWER_CONTRACT, 'locks': locks}
    common.save_json(root/'plan.json', plan)
    common.save_json(root/'plan_identity.json', {'sha256': common.sha(root/'plan.json')})
    common.save_json(root/'cost_carryforward_manifest.json', {k: plan[k] for k in (
        'base_run', 'authorization', 'inherited_used_or_reserved', 'authorized_cumulative_ceiling',
        'new_token_cap', 'new_http_cap')})
    common.save_json(root/'launch_status.json', {'status': 'prepared', 'created_at': common.utc()})
    print(json.dumps({'root': str(root), 'episodes': len(episodes), 'new_token_cap': token_cap,
                      'cumulative_max': used+token_cap, 'new_http_cap': http_cap}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['prepare', 'check', 'launch', 'run', 'status'])
    ap.add_argument('--root', required=True)
    ap.add_argument('--base-run', default='/home/yangchengyu/officeqa_humanstrong_v5_test102_pairs_seed42_20261020')
    ap.add_argument('--config-run', default='/home/yangchengyu/officeqa_humanstrong_v5_test102_pairs_seed42_20261020')
    ap.add_argument('--resources', default='/home/yangchengyu/officeqa_external_minimax_resources_20261020_v1')
    ap.add_argument('--env-file', default='/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env')
    args = ap.parse_args()
    root = Path(args.root).resolve()
    bind_runner()
    if args.action == 'prepare':
        prepare(root, args.base_run, args.resources, args.config_run)
    elif args.action == 'check':
        common.verify(root)
        print('Identity and billing precheck passed; zero HTTP.')
    elif args.action == 'status':
        print(json.dumps(common.status(root), ensure_ascii=False, indent=2))
    elif args.action == 'run':
        sys.exit(common.execute(root, args.env_file))
    else:
        common.verify(root)
        with FileLock(str(root/'.launch.lock'), timeout=0):
            if common.read(root/'launch_status.json')['status'] != 'prepared':
                raise ValueError('Already launched; inspect first')
            with (root/'run.log').open('x') as log:
                child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'run',
                    '--root', str(root), '--env-file', args.env_file], cwd=common.SOURCE,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            common.save_json(root/'launch_status.json', {'status': 'launching', 'pid': child.pid,
                                                        'started_at': common.utc()})
            (root/'run.pid').write_text(str(child.pid)+'\n')
            print(json.dumps({'pid': child.pid, 'root': str(root)}))
