"""Bounded paired engineering observation; reuse the frozen Bank and billing runner."""
import argparse
from copy import deepcopy
from pathlib import Path
import random
import subprocess
import sys

from filelock import FileLock
from atomic_skillgraph.empirical.evidence_handoff import VERSION as HANDOFF
import run_office_humanstrong_v1 as common

VERSION = 'office.public-evidence-handoff.train2-pairs.v1.20261019'
TOKEN_CAP, HTTP_CAP, CUMULATIVE_CAP = 768000, 400, 116000000


def configure(cfg):
    cfg = deepcopy(cfg)
    cfg['runtime'].update(evidence_handoff_version=HANDOFF, finish_input_max_bytes=65536,
                          dynamic_output_min_tokens=8192)
    cfg['llm']['purpose_overrides']['finish_only'] = {
        'protocol': {'thinking_type': 'enabled'}, 'max_completion_tokens': 8192}
    return common.validate_config(cfg)


def prepare(root, base, resources):
    root, base, resources = Path(root), Path(base).resolve(), Path(resources).resolve()
    if root.exists(): raise ValueError('Use a fresh independent observation directory')
    code = common.code_identity()
    if code['tracked_dirty'] or not code['git_sha']: raise ValueError('Commit before preparing a paid run')
    parent = common.read(base/'plan.json')
    if common.read(base/'launch_status.json')['status'] != 'completed': raise ValueError('Parent must be complete')
    ledger = common.read(base/'budget.json')
    if common.billing_blocked(ledger): raise ValueError('Resolve billing before a new run')
    used = sum(map(common.charge, ledger['attempts'].values()))
    if used+TOKEN_CAP > CUMULATIVE_CAP: raise ValueError('Insufficient existing cumulative authority')
    tasks = common.read(resources/'train.json')['tasks'][:2]
    canonical = common.read(common.SOURCE/'data/main_experiment_v1/officeqa/train.json')['tasks'][:2]
    assert [t['task_id'] for t in tasks] == [t['task_id'] for t in canonical]
    assert all(t['split'] == 'train' for t in tasks)
    assert not {t['task_id'] for t in tasks}&{t['task_id'] for t in common.read(resources/'test.json')['tasks']}
    assert common.package_identity() == parent['package_identity']
    cfg = configure(common.read(base/'base_config.json'))
    assert cfg['runtime']['global_action_budget'] == 48
    assert cfg['llm']['runtime']['max_total_tokens_per_task'] == 192000
    cfg['harness'].update(evaluator_records=str(resources/'evaluator_records_train.json'), split='train')
    cfg['manifest'] = str(resources/'train.json')
    cfg['experiment'].update(runtime_mode='frozen', seed=42, output_dir=str(root), observation_version=VERSION)
    cfg['data_dir'] = str(root/'episodes')
    root.mkdir(parents=True)
    common.save_json(root/'budget_prefix.json', ledger)
    limits = {**ledger['limits'], 'token_limit':used+TOKEN_CAP,
              'request_limit':len(ledger['attempts'])+HTTP_CAP, 'finish_reserve':0}
    ledger['limits'] = limits
    common.save_json(root/'budget.json', ledger)
    cfg['budget'].update(token_limit=limits['token_limit'], request_limit=limits['request_limit'],
                         train_task_tokens=192000, eval_task_tokens=192000)
    common.write_json(root/'base_config.json', cfg)
    common.save_json(root/'tasks.json', {'tasks':tasks})
    bundle = common.read(common.PACKAGE/'bank/bank.json'); episodes = []; rng = random.Random(42)
    for i, task in enumerate(tasks):
        arms = ['off', 'manual']; rng.shuffle(arms)
        for arm in arms:
            out = root/'episodes'/f"{i+1:02}_{task['task_id'].split(':')[-1]}"/arm
            common.frontier.native_import(bundle, out/'bank', cfg['program_environment'],
                cfg['experiment']['officeqa_method_version'], 42)
            actual = deepcopy(cfg); actual['data_dir'] = str(out/'bank')
            actual['experiment']['output_dir'] = str(out)
            common.write_json(out/'config.json', actual)
            episodes.append({'task_id':task['task_id'], 'arm':arm, 'path':str(out),
                'config_hash':common.sha(out/'config.json'), 'bank_identity':common.tree_identity(out/'bank')})
    protected = [base/'budget.json', base/'plan.json', base/'launch_status.json',
        resources/'train.json', resources/'test.json', resources/'evaluator_records_train.json',
        common.SOURCE/'data/main_experiment_v1/officeqa/train.json', root/'tasks.json',
        root/'base_config.json', root/'budget_prefix.json']
    plan = {'created_at':common.utc(), 'code':code, 'method_version':VERSION,
        'package_identity':common.package_identity(), 'base_run':str(base), 'resources':str(resources),
        'authorization':'User requested the evidence/finalization fixes after authorizing corresponding paid experiments. '
            'Agent allocates one bounded two-Train paired engineering check within the existing 116M ceiling; '
            'no new Val/Test, no learning, no Bank editing or automatic repeat.',
        'selection':'first two canonical Train members, no score filtering; arm order randomized once with seed42',
        'development_exposed':True, 'comparison_caveat':'These Train tasks were already used in Bank development; this is engineering validation, not a blind accuracy estimate.',
        'seed':42, 'episodes':episodes, 'corpus_identity':parent['corpus_identity'],
        'protected_files':{str(p):common.sha(p) for p in protected}, 'ledger_limits':limits,
        'original_cumulative_ceiling':CUMULATIVE_CAP, 'authorized_cumulative_ceiling':CUMULATIVE_CAP,
        'inherited_used_or_reserved':used, 'new_token_cap':TOKEN_CAP, 'new_http_cap':HTTP_CAP,
        'asset_origin':'external_humanstrong_reference_not_learned_ours',
        'workflow_execution_mode':'native_dynamic_tool_loop_no_automatic_graph_dispatch',
        'test_runs_authorized_in_this_plan':0, 'locks':[str(base/'.run.lock'), *parent['locks']]}
    common.save_json(root/'plan.json', plan)
    common.save_json(root/'plan_identity.json', {'sha256':common.sha(root/'plan.json')})
    common.save_json(root/'cost_carryforward_manifest.json', {k:plan[k] for k in (
        'base_run','authorization','inherited_used_or_reserved','authorized_cumulative_ceiling','new_token_cap','new_http_cap')})
    common.save_json(root/'launch_status.json', {'status':'prepared','created_at':common.utc()})
    print({'root':str(root),'episodes':len(episodes),'max_new_tokens':TOKEN_CAP,'cumulative_max':used+TOKEN_CAP})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['prepare','check','launch','run','status'])
    ap.add_argument('--root', required=True)
    ap.add_argument('--base-run', default='/home/yangchengyu/officeqa_humanstrong_test102_seed42_20261019')
    ap.add_argument('--resources', default='/home/yangchengyu/main_experiment_v1_resources_cf4_r3_20261009_v1/officeqa')
    ap.add_argument('--env-file', default='/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env')
    args = ap.parse_args(); root = Path(args.root).resolve(); common.VERSION = VERSION
    if args.action == 'prepare': prepare(root, args.base_run, args.resources)
    elif args.action == 'check': common.verify(root); print('Offline identity precheck passed; zero HTTP.')
    elif args.action == 'status': print(common.status(root))
    elif args.action == 'run': sys.exit(common.execute(root, args.env_file))
    else:
        common.verify(root)
        with FileLock(str(root/'.launch.lock'), timeout=0):
            if common.read(root/'launch_status.json')['status'] != 'prepared': raise ValueError('Already launched; inspect first')
            with (root/'run.log').open('x') as log:
                child = subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'run','--root',str(root),
                    '--env-file',args.env_file], cwd=common.SOURCE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            common.save_json(root/'launch_status.json', {'status':'launching','pid':child.pid,'started_at':common.utc()})
            (root/'run.pid').write_text(str(child.pid)+'\n')
            print({'pid':child.pid,'root':str(root)})
