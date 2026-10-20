"""Frozen paired stages with one cumulative ledger; no changes to the main repository."""
import argparse
from copy import deepcopy
import fcntl
import importlib.util
import json
from pathlib import Path
import random
import shutil
import sqlite3
import sys
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
import runtime as r
from provider_outcomes import refusal_guard, recover_saved_refusals, POLICY

CAMPAIGN = Path('/home/yangchengyu/searchqa_glm53_v2_v7_test1400_seed42_refusal_resume_20261020')
PARENT = Path('/home/yangchengyu/searchqa_humanstrong_v6_test1400_seed42_20261019_accounting')


def module(root, arm, kind):
    package = root / arm / 'input'
    if kind == 'v2':
        return r.bridge(package)
    path = package / 'integration.py'
    spec = importlib.util.spec_from_file_location('accuracy_' + r.sha(path), path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def completed(root, arm):
    path = root / arm / 'test/run.sqlite3'
    if not path.exists():
        return {}
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
        return {key: json.loads(value) for key, value in db.execute("SELECT id,result FROM tasks WHERE status='completed'")}


def prepare(stage, spec_path):
    root = CAMPAIGN / stage
    assert not root.exists(), 'Never overwrite an existing stage'
    spec = r.read(spec_path)
    tasks = r.read(spec['tasks'])['tasks']
    canonical = {t['task_id']: t for t in r.read(r.DATASETS / 'test.json')['tasks']}
    assert len(tasks) == len({t['task_id'] for t in tasks})
    assert all(t == canonical[t['task_id']] for t in tasks)
    code = r.code_identity()
    assert code == r.read(HERE / 'campaign_plan.json')['source']
    ledger = r.read(CAMPAIGN / 'budget.json')
    r.write_json(root / 'tasks.json', {'tasks': tasks})
    r.write_json(root / 'budget_at_start.json', ledger)
    rng = random.Random(42)
    arms = spec['arms']
    r.write_json(root / 'schedule.json', [{'task_id': task['task_id'], 'arm': arm}
        for task in tasks for arm in rng.sample(list(arms), len(arms))])
    inputs, frozen = {}, {}
    for arm, item in arms.items():
        package = Path(item['package'])
        for name, expected in r.read(package / 'FILE_SHA256.json').items():
            assert r.sha(package / name) == expected, name
        shutil.copytree(package, root / arm / 'input', ignore=shutil.ignore_patterns('__pycache__'))
        mod = module(root, arm, item['kind'])
        cfg = deepcopy(r.read(spec['base_config']))
        cfg['experiment'].update(output_dir=str(root / arm / 'test'), asset_origin='human_authored_post_test_development')
        cfg['data_dir'] = str(root / arm / 'frozen_bank')
        cfg['budget'].update(ledger['limits'])
        cfg = r.validate_config(cfg)
        assert cfg['budget']['eval_task_tokens'] == cfg['llm']['runtime']['max_total_tokens_per_task'] == 128000
        if item.get('frozen_source'):
            shutil.copytree(item['frozen_source'], root / arm / 'frozen_bank')
        else:
            mod.native_import(r.read(root / arm / 'input/bank/bank.json'), root / arm / 'frozen_bank', cfg['program_environment'])
        r.write_json(root / arm / 'config.json', cfg)
        inputs[arm] = r.tree_identity(root / arm / 'input')
        frozen[arm] = r.tree_identity(root / arm / 'frozen_bank')
    files = [root / name for name in ('tasks.json', 'schedule.json', 'budget_at_start.json')]
    files += [root / arm / 'config.json' for arm in arms]
    files += [Path(__file__), Path(r.__file__), Path(spec_path), HERE / 'campaign_plan.json', HERE / 'provider_outcomes.py',
              r.DATASETS / 'evaluator_records_test.json', CAMPAIGN / 'authorization.json',
              CAMPAIGN / 'cost_carryforward_manifest.json']
    plan = {'code': code, 'created_at': r.utc(), 'tasks': len(tasks), 'arms': arms,
            'input_identity': inputs, 'frozen': frozen, 'files': {str(f): r.sha(f) for f in files},
            'purpose': spec['purpose'], 'limits': ledger['limits'], 'post_test_development': True,
            'replay_sources': spec.get('replay_sources', {}), 'refusal_policy': POLICY}
    r.write_json(root / 'plan.json', plan)
    r.write_json(root / 'plan_identity.json', {'sha256': r.digest(plan)})
    return check(stage)


def check(stage):
    root = CAMPAIGN / stage
    plan = r.read(root / 'plan.json')
    assert r.digest(plan) == r.read(root / 'plan_identity.json')['sha256']
    assert plan['code'] == r.code_identity()
    for path, expected in plan['files'].items():
        assert r.sha(path) == expected, path
    for arm in plan['arms']:
        assert r.tree_identity(root / arm / 'input') == plan['input_identity'][arm]
        assert r.tree_identity(root / arm / 'frozen_bank') == plan['frozen'][arm]
    ledger, inherited = r.read(CAMPAIGN / 'budget.json'), r.read(CAMPAIGN / 'inherited_budget.json')
    assert ledger['limits'] == plan['limits'] == inherited['limits']
    assert all(ledger['attempts'].get(k) == v for k, v in inherited['attempts'].items())
    return {'passed': True, 'stage': stage, 'tasks': plan['tasks'], 'arms': list(plan['arms'])}


def status(stage):
    root = CAMPAIGN / stage
    plan = r.read(root / 'plan.json')
    summaries = {}
    for arm in plan['arms']:
        rows = completed(root, arm)
        calls = [a for row in rows.values() for a in row['execution'].get('attempts', [])]
        summaries[arm] = {'completed': len(rows), 'total': plan['tasks'],
            'correct': sum(bool(row['score']['hard']) for row in rows.values()),
            'tokens': sum(u['total_tokens'] for row in rows.values() for u in row['usage']),
            'native_graphs': sum(bool(row['execution'].get('all_authored_nodes_completed')) for row in rows.values()),
            'program_calls': len(calls), 'program_ok': sum(a['status'] == 'ok' for a in calls)}
    ledger, inherited = r.read(CAMPAIGN / 'budget.json'), r.read(CAMPAIGN / 'inherited_budget.json')
    added = [v for k, v in ledger['attempts'].items() if k not in inherited['attempts']]
    carry = r.read(CAMPAIGN / 'cost_carryforward_manifest.json')
    for arm, prefix in carry.get('parent_completed', {}).items():
        summaries[arm]['new_completed'] = summaries[arm]['completed']
        summaries[arm]['inherited_completed'] = prefix['completed']
        for field in ('completed', 'total', 'correct', 'tokens', 'native_graphs', 'program_calls', 'program_ok'):
            summaries[arm][field] += prefix[field]
    unknown = [a for a in ledger['attempts'].values() if a['status'] != 'metered']
    from atomic_skillgraph.empirical.budget_governor import billing_blocked
    return {'stage': stage, 'arms': summaries, 'campaign_tokens': carry.get('prior_searchqa_tokens', 0) + sum(a.get('accounted_tokens', 0) for a in added),
            'campaign_http': carry.get('prior_searchqa_http', 0) + len(added),
            'inflight_or_unmetered': len(unknown),
            'unknown_reserved_tokens': sum(a['reserved_tokens'] for a in unknown),
            'billing_blocked': billing_blocked(ledger),
            'refusals': sum(row['execution']['reason'] == 'provider_content_filter'
                            for arm in plan['arms'] for row in completed(root, arm).values()),
            'status': r.read(root / 'result.json')['status'] if (root / 'result.json').exists() else 'prepared', 'at': r.utc()}


def execute(stage):
    root = CAMPAIGN / stage
    with (CAMPAIGN / 'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check(stage)
        assert r.read(CAMPAIGN / 'authorization.json')['status'] == 'approved'
        receipt = r.read(HERE / 'offline_acceptance.json')
        assert receipt['passed'] and receipt['driver_sha256'] == r.sha(__file__)
        if (root / 'result.json').exists() and r.read(root / 'result.json')['status'] == 'completed':
            return status(stage)
        ledger = r.read(CAMPAIGN / 'budget.json')
        governor = r.BudgetGovernor(CAMPAIGN / 'budget.json', **ledger['limits'])
        if governor.billing_blocked:
            recover_saved_refusals(governor, root.glob('*/test/requests/*.json'))
        assert not governor.billing_blocked, 'Unresolved billing blocks resume'
        r.load_env('/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env')
        plan = r.read(root / 'plan.json')
        arms = plan['arms']
        tasks = [r.PublicTask(**t) for t in r.read(root / 'tasks.json')['tasks']]
        offset = plan.get('order_offset', 0)
        metadata = {t.task_id: {'task_order_index': i + offset, 'campaign_stage': stage} for i, t in enumerate(tasks)}
        configs = {a: r.read(root / a / 'config.json') for a in arms}
        bundles = {a: r.read(root / a / 'frozen_bank/imported_bank.json') for a in arms}
        modules = {a: module(root, a, item['kind']) for a, item in arms.items()}
        logs = {a: r.FormalLog(root / a, {'arm': a, 'plan_hash': r.digest(plan), 'source_commit': plan['code']['git_sha'],
             'run_seed': 42, 'provenance': plan['purpose']}, configs[a], resume=(root / a / 'run_manifest.json').exists()) for a in arms}
        state, error = 'stopped', None
        r.write_json(root / 'result.json', {'status': 'running', 'started_at': r.utc()})
        try:
            for item in r.read(root / 'schedule.json'):
                arm = item['arm']
                if item['task_id'] in completed(root, arm):
                    continue
                if governor.billing_blocked:
                    raise RuntimeError('Unresolved billing before next task')
                if (CAMPAIGN / 'STOP_AFTER_TASK').exists() or (root / 'STOP_AFTER_TASK').exists():
                    raise SystemExit(75)
                r.write_json(root / 'current.json', {**item, 'started_at': r.utc()})
                mod = modules[arm]
                factory = (lambda bank: mod.HumanBankView(bank, bundles[arm], 'manual', 'auto', None)) if arms[arm]['kind'] == 'v2' else (lambda bank: mod.HumanBankView(bank, bundles[arm]))
                bridge = mod.bridge(argparse.Namespace()) if arms[arm]['kind'] == 'v2' else mod.bridge()
                with bridge, refusal_guard(governor, plan['replay_sources'].get(arm, {})):
                    r.run(configs[arm], tasks, root / arm / 'test', readonly=True,
                          resume=(root / arm / 'test/execution_manifest.json').exists(), max_new_tasks=1,
                          bank_view_factory=factory, budget_governor=governor, formal_log=logs[arm],
                          task_metadata=metadata, order_offset=offset, freeze_on_complete=False)
                assert not governor.billing_blocked and item['task_id'] in completed(root, arm)
                progress = status(stage)
                r.write_json(root / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
            assert all(len(completed(root, a)) == plan['tasks'] for a in arms)
            state = 'completed'
        except BaseException as exc:
            error = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc(), 'at': r.utc()}
            raise
        finally:
            for log in logs.values():
                log.finish(state)
            audit = {a: r.tree_identity(root / a / 'frozen_bank') == plan['frozen'][a] for a in arms}
            r.write_json(root / 'freeze_audit.json', audit)
            r.write_json(root / 'result.json', {**status(stage), 'status': state, 'error': error})
            assert all(audit.values()), 'Frozen changed'
        return status(stage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'check', 'status', 'run'))
    parser.add_argument('--stage', default='diagnostic48')
    parser.add_argument('--spec')
    args = parser.parse_args()
    result = prepare(args.stage, args.spec) if args.action == 'prepare' else {'check': check, 'status': status, 'run': execute}[args.action](args.stage)
    print(json.dumps(result, ensure_ascii=False))
