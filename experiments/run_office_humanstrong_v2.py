"""Authorized Train12 pairs with frozen V2 Bank and external public page inputs."""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys

from filelock import FileLock
import run_office_humanstrong_v1 as common

PACKAGE = common.SOURCE/'resources/office_humanstrong_v2'
VERSION = 'office.humanstrong.v2.aligned-train12-pairs.20261020'
TOKEN_CAP, HTTP_CAP, CUMULATIVE_CAP = 4608000, 2400, 116000000


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def bind_runner():
    common.PACKAGE = PACKAGE
    common.frontier = load('office_v2_paired_frontier', PACKAGE/'integration/frontier.py')
    common.VERSION = VERSION


def prepare(root, base, resources):
    root, base, resources = Path(root).resolve(), Path(base).resolve(), Path(resources).resolve()
    if root.exists(): raise ValueError('Use a fresh independent directory')
    code = common.code_identity()
    if code['tracked_dirty'] or not code['git_sha']: raise ValueError('Commit before preparing a paid run')
    parent = common.read(base/'plan.json')
    if common.read(base/'launch_status.json')['status'] != 'completed': raise ValueError('Parent must be complete')
    ledger = common.read(base/'budget.json')
    if common.billing_blocked(ledger): raise ValueError('Resolve billing before a new run')
    used = sum(map(common.charge, ledger['attempts'].values()))
    if used+TOKEN_CAP > CUMULATIVE_CAP: raise ValueError('Insufficient existing cumulative authority')
    configuration = load('office_v2_paired_config', PACKAGE/'integration/configuration.py')
    resources, manifest = configuration.verify_resources(resources)
    tasks = common.read(resources/'train.json')['tasks'][:12]
    canonical_path = common.SOURCE/'data/main_experiment_v1/officeqa/train.json'
    canonical = common.read(canonical_path)['tasks'][:12]
    assert len(tasks) == 12 and all(t['split'] == 'train' for t in tasks)
    assert [t['task_id'] for t in tasks] == [t['task_id'] for t in canonical]
    assert not {t['task_id'] for t in tasks} & {t['task_id'] for t in common.read(resources/'test.json')['tasks']}
    cfg = configuration.configure(common.read(base/'base_config.json'), resources, 'train')
    assert cfg['runtime']['global_action_budget'] == 48
    assert cfg['llm']['runtime']['max_total_tokens_per_task'] == 192000
    cfg['experiment'].update(runtime_mode='frozen', seed=42, output_dir=str(root), observation_version=VERSION)
    cfg['data_dir'] = str(root/'episodes')
    root.mkdir(parents=True)
    common.save_json(root/'budget_prefix.json', ledger)
    limits = {**ledger['limits'], 'token_limit': used+TOKEN_CAP,
              'request_limit': len(ledger['attempts'])+HTTP_CAP, 'finish_reserve': 0}
    ledger['limits'] = limits
    common.save_json(root/'budget.json', ledger)
    cfg['budget'].update(token_limit=limits['token_limit'], request_limit=limits['request_limit'],
                         train_task_tokens=192000, eval_task_tokens=192000)
    common.write_json(root/'base_config.json', cfg)
    common.save_json(root/'tasks.json', {'tasks': tasks})
    bundle = common.read(PACKAGE/'bank/bank.json'); episodes = []; rng = random.Random(42)
    for i, task in enumerate(tasks):
        arms = ['off', 'manual']; rng.shuffle(arms)
        for arm in arms:
            out = root/'episodes'/f"{i+1:02}_{task['task_id'].split(':')[-1]}"/arm
            common.frontier.native_import(bundle, out/'bank', cfg['program_environment'],
                cfg['experiment']['officeqa_method_version'], 42)
            actual = deepcopy(cfg); actual['data_dir'] = str(out/'bank')
            actual['experiment']['output_dir'] = str(out)
            common.write_json(out/'config.json', actual)
            episodes.append({'task_id': task['task_id'], 'arm': arm, 'path': str(out),
                'config_hash': common.sha(out/'config.json'), 'bank_identity': common.tree_identity(out/'bank')})
    protected = [base/'budget.json', base/'plan.json', base/'launch_status.json',
        resources/'manifest.json', canonical_path, root/'tasks.json', root/'base_config.json', root/'budget_prefix.json']
    protected += [resources/name for name in manifest['files']]
    plan = {'created_at': common.utc(), 'code': code, 'method_version': VERSION,
        'package_identity': common.package_identity(), 'base_run': str(base), 'resources': str(resources),
        'authorization': '2026-10-20 user accepted the proposed fixed Train12 paired experiment: '
            'DeepSeek seed42, identical external public inputs and budgets, NoSkill versus HumanStrong V2; '
            'new <=4,608,000 tokens / 2,400 physical HTTP within existing 116M cumulative authority; '
            'frozen assets, no learning, no Val/Test or automatic repeat.',
        'selection': 'first 12 canonical Train members, no score filtering; paired order randomized once with seed42',
        'development_exposed': True,
        'comparison_caveat': 'These Train tasks were used in human Bank development. This is a paired development observation, not a blind accuracy estimate. Old full-corpus-only scores are not a matched control.',
        'seed': 42, 'episodes': episodes, 'corpus_identity': parent['corpus_identity'],
        'protected_files': {str(p): common.sha(p) for p in protected}, 'ledger_limits': limits,
        'original_cumulative_ceiling': CUMULATIVE_CAP, 'authorized_cumulative_ceiling': CUMULATIVE_CAP,
        'inherited_used_or_reserved': used, 'new_token_cap': TOKEN_CAP, 'new_http_cap': HTTP_CAP,
        'asset_origin': 'external_humanstrong_reference_not_learned_ours',
        'workflow_execution_mode': 'native_dynamic_tool_loop_no_automatic_graph_dispatch',
        'fresh_paired_episodes': 24, 'prior_scores_used_for_selection': False,
        'test_runs_authorized_in_this_plan': 0, 'locks': [str(base/'.run.lock'), *parent['locks']]}
    common.save_json(root/'plan.json', plan)
    common.save_json(root/'plan_identity.json', {'sha256': common.sha(root/'plan.json')})
    common.save_json(root/'cost_carryforward_manifest.json', {k: plan[k] for k in (
        'base_run', 'authorization', 'inherited_used_or_reserved', 'authorized_cumulative_ceiling', 'new_token_cap', 'new_http_cap')})
    common.save_json(root/'launch_status.json', {'status': 'prepared', 'created_at': common.utc()})
    print(json.dumps({'root': str(root), 'episodes': len(episodes), 'max_new_tokens': TOKEN_CAP,
                      'max_new_http': HTTP_CAP, 'cumulative_max': used+TOKEN_CAP}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['prepare', 'check', 'launch', 'run', 'status'])
    ap.add_argument('--root', required=True)
    ap.add_argument('--base-run', default='/home/yangchengyu/officeqa_evidence_handoff_train2_pairs_seed42_20261019')
    ap.add_argument('--resources', default='/home/yangchengyu/officeqa_external_minimax_resources_20261020_v1')
    ap.add_argument('--env-file', default='/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env')
    args = ap.parse_args(); root = Path(args.root).resolve(); bind_runner()
    if args.action == 'prepare': prepare(root, args.base_run, args.resources)
    elif args.action == 'check': common.verify(root); print('Offline identity precheck passed; zero HTTP.')
    elif args.action == 'status': print(json.dumps(common.status(root), ensure_ascii=False, indent=2))
    elif args.action == 'run': sys.exit(common.execute(root, args.env_file))
    else:
        common.verify(root)
        with FileLock(str(root/'.launch.lock'), timeout=0):
            if common.read(root/'launch_status.json')['status'] != 'prepared': raise ValueError('Already launched; inspect first')
            with (root/'run.log').open('x') as log:
                child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'run', '--root', str(root),
                    '--env-file', args.env_file], cwd=common.SOURCE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            common.save_json(root/'launch_status.json', {'status': 'launching', 'pid': child.pid, 'started_at': common.utc()})
            (root/'run.pid').write_text(str(child.pid)+'\n')
            print(json.dumps({'pid': child.pid, 'root': str(root)}))
