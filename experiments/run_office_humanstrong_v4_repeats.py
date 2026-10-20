"""Two sequential repeats using the unchanged V4 runtime and actual carryforward."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from filelock import FileLock

RUNTIME = Path('/home/yangchengyu/asg_office_humanstrong_v4_fixed_20261020')
PARENT = Path('/home/yangchengyu/officeqa_humanstrong_v4_test102_pairs_seed42_20261020')
RUNTIME_SHA = '7d93d1456eee3c859f9dc5e0758d51a24796ade2'
CEILING = 155_000_000


def bind(source):
    source = Path(source).resolve()
    sys.path[:0] = [str(source/'experiments'), str(source/'src')]
    os.chdir(source)
    import run_office_humanstrong_v4 as fixed
    fixed.bind_runner()
    if fixed.common.code_identity()['git_sha'] != RUNTIME_SHA:
        raise ValueError('Repeat requires the original V4 runtime commit')
    return fixed.common


def prepare(c, root, parent, repeat):
    root, parent = Path(root), Path(parent)
    if root.exists():
        raise ValueError('Repeat directory must be fresh')
    prior = c.read(parent/'plan.json')
    if c.read(parent/'launch_status.json')['status'] != 'completed':
        raise ValueError('Billing parent must have completed')
    ledger = c.read(parent/'budget.json')
    if c.billing_blocked(ledger):
        raise ValueError('Billing parent has unresolved charges')
    original = c.read(PARENT/'plan.json')
    tasks = c.read(PARENT/'tasks.json')
    rows = [e for e in original['episodes'] if e['arm'] == 'manual']
    if len(rows) != 102 or [e['task_id'] for e in rows] != [t['task_id'] for t in tasks['tasks']]:
        raise ValueError('Repeat must contain the exact canonical Test102')
    used = sum(map(c.charge, ledger['attempts'].values()))
    cap = 102*216576
    if used+cap > CEILING:
        raise ValueError('Cumulative authority insufficient; do not launch this repeat')
    cfg = c.read(PARENT/'base_config.json')
    decisions = max(16, cfg['runtime']['global_action_budget']*4+16)
    http = 102*(decisions*2+1)*(cfg['llm']['max_retries']+1)
    assert cfg['llm']['model'] == 'deepseek-v4-flash'
    assert cfg['llm']['runtime']['max_total_tokens_per_task'] == 216576
    root.mkdir(parents=True)
    c.save_json(root/'budget_prefix.json', ledger)
    limits = {**ledger['limits'], 'token_limit': used+cap,
              'request_limit': len(ledger['attempts'])+http, 'finish_reserve': 0}
    ledger['limits'] = limits
    c.save_json(root/'budget.json', ledger)
    cfg['data_dir'] = str(root/'episodes')
    cfg['experiment']['output_dir'] = str(root)
    cfg['budget'].update(token_limit=limits['token_limit'], request_limit=limits['request_limit'])
    c.write_json(root/'base_config.json', cfg)
    c.save_json(root/'tasks.json', tasks)
    episodes = []
    for old in rows:
        previous = Path(old['path'])
        target = root/'episodes'/previous.parent.name/'manual'
        shutil.copytree(previous/'bank', target/'bank')
        actual = deepcopy(cfg)
        actual['data_dir'] = str(target/'bank')
        actual['experiment']['output_dir'] = str(target)
        c.write_json(target/'config.json', actual)
        identity = c.tree_identity(target/'bank')
        if identity != old['bank_identity']:
            raise ValueError('Imported Bank changed')
        episodes.append({'task_id': old['task_id'], 'arm': 'manual', 'path': str(target),
                         'config_hash': c.sha(target/'config.json'), 'bank_identity': identity})
    protected = {p: h for p, h in original['protected_files'].items()
                 if Path(p).is_relative_to(Path(original['resources']))}
    for path in (parent/'budget.json', parent/'plan.json', parent/'launch_status.json',
                 PARENT/'plan.json', PARENT/'tasks.json', root/'tasks.json',
                 root/'base_config.json', root/'budget_prefix.json', Path(__file__).resolve()):
        protected[str(path)] = c.sha(path)
    plan = {**original, 'created_at': c.utc(), 'code': c.code_identity(),
        'base_run': str(parent), 'episodes': episodes, 'protected_files': protected,
        'ledger_limits': limits, 'inherited_used_or_reserved': used,
        'new_token_cap': cap, 'new_http_cap': http, 'test_runs_authorized_in_this_plan': 1,
        'fresh_paired_episodes': 0, 'repeat_index': repeat,
        'selection': 'All canonical Test102 in original order, manual Bank only; same seed42 and inference settings.',
        'authorization': '2026-10-20 user requested two additional DeepSeek Test repeats of the current human Bank. '
                         'Run sequentially; each reserves its full task caps only after the preceding ledger settles. '
                         'Retain the approved 155M cumulative ceiling and all historical charges/reservations.',
        'controller_sha256': c.sha(Path(__file__)),
        'locks': list(dict.fromkeys([str(parent/'.run.lock'), *prior['locks']]))}
    c.save_json(root/'plan.json', plan)
    c.save_json(root/'plan_identity.json', {'sha256': c.sha(root/'plan.json')})
    c.save_json(root/'cost_carryforward_manifest.json', {k: plan[k] for k in (
        'authorization', 'base_run', 'inherited_used_or_reserved',
        'authorized_cumulative_ceiling', 'new_token_cap', 'new_http_cap', 'repeat_index')})
    c.save_json(root/'launch_status.json', {'status': 'prepared', 'created_at': c.utc()})
    c.verify(root)
    return plan


def run(c, campaign, env):
    identity = c.read(campaign/'campaign.json')
    if identity['controller_sha256'] != c.sha(Path(__file__)):
        raise ValueError('Controller changed after launch')
    with FileLock(str(campaign/'.campaign.lock'), timeout=0):
        parent = PARENT
        for repeat in (2, 3):
            root = campaign/f'repeat_{repeat}'
            c.save_json(campaign/'status.json', {'status': 'running', 'repeat': repeat,
                'root': str(root), 'pid': os.getpid(), 'updated_at': c.utc()})
            try:
                if not root.exists():
                    prepare(c, root, parent, repeat)
                if c.read(root/'launch_status.json')['status'] != 'completed':
                    code = c.execute(root, env)
                    if code:
                        raise RuntimeError(f'Repeat stopped with exit code {code}; inspect before any continuation')
            except Exception as exc:
                c.save_json(campaign/'status.json', {'status': 'stopped', 'repeat': repeat,
                    'root': str(root), 'error': str(exc), 'updated_at': c.utc()})
                raise
            parent = root
        c.save_json(campaign/'status.json', {'status': 'completed', 'updated_at': c.utc(),
            'repeats': {str(i): c.status(campaign/f'repeat_{i}') for i in (2, 3)}})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['check', 'launch', 'run', 'status'])
    ap.add_argument('--root', required=True)
    ap.add_argument('--runtime-source', default=str(RUNTIME))
    ap.add_argument('--env-file', default='/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env')
    args = ap.parse_args()
    campaign = Path(args.root).resolve()
    c = bind(args.runtime_source)
    if args.action == 'check':
        assert c.read(PARENT/'launch_status.json')['status'] == 'completed'
        assert not c.billing_blocked(c.read(PARENT/'budget.json'))
        print(json.dumps({'runtime': c.code_identity(), 'parent_used': sum(map(c.charge,
            c.read(PARENT/'budget.json')['attempts'].values())), 'cap_each': 102*216576,
            'cumulative_ceiling': CEILING, 'actual_http': 0}))
    elif args.action == 'status':
        print(json.dumps(c.read(campaign/'status.json'), ensure_ascii=False, indent=2))
        for repeat in (2, 3):
            path = campaign/f'repeat_{repeat}'
            if (path/'plan.json').exists():
                print(json.dumps({'repeat': repeat, **{k: v for k, v in c.status(path).items()
                                                      if k != 'results'}}, ensure_ascii=False, indent=2))
    elif args.action == 'run':
        run(c, campaign, args.env_file)
    else:
        if campaign.exists():
            raise ValueError('Use a fresh campaign; never double launch')
        campaign.mkdir(parents=True)
        c.save_json(campaign/'campaign.json', {'controller_sha256': c.sha(Path(__file__)),
            'runtime_source': str(Path.cwd()), 'runtime_sha': RUNTIME_SHA,
            'repeat_indices': [2, 3], 'cumulative_ceiling': CEILING, 'created_at': c.utc()})
        with (campaign/'controller.log').open('x') as log:
            proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'run',
                '--root', str(campaign), '--runtime-source', str(Path.cwd()), '--env-file', args.env_file],
                cwd=Path.cwd(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'pid': proc.pid, 'root': str(campaign)}))
