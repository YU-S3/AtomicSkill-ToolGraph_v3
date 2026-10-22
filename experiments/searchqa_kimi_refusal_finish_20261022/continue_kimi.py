"""Versioned final-ten continuation; preserve the original 1390-task prefix."""
import argparse
from copy import deepcopy
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'searchqa_glm53_refusal_resume_20261020'
sys.path.insert(0, str(PRIOR))
import campaign as c

spec = importlib.util.spec_from_file_location('kimi_refusal_v2', HERE / 'provider_outcomes.py')
outcomes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(outcomes)
c.HERE = HERE
c.refusal_guard = outcomes.refusal_guard
c.recover_saved_refusals = outcomes.recover_saved_refusals
c.POLICY = outcomes.POLICY
r = c.r
PARENT = Path('/home/yangchengyu/searchqa_v7_crossmodel_test1400_seed42_20261021/kimi_k26')
ROOT = Path('/home/yangchengyu/searchqa_v7_kimi_test1400_refusal_finish_20261022')
STAGE, ARM = 'fulltest1400', 'v7_text_compiler'
c.CAMPAIGN = ROOT


def alive(root):
    pid = (root / 'pid').read_text().strip() if (root / 'pid').exists() else None
    path = Path('/proc') / str(pid) / 'cmdline'
    return bool(pid and path.exists() and path.read_bytes())


def specs():
    assert not ROOT.exists() and not alive(PARENT)
    c.CAMPAIGN = PARENT
    c.check(STAGE)
    c.CAMPAIGN = ROOT
    parent_stage = PARENT / STAGE
    assert r.read(parent_stage / 'result.json')['status'] == 'stopped'
    rows = c.completed(parent_stage, ARM)
    tasks = r.read(parent_stage / 'tasks.json')['tasks']
    assert len(rows) == 1390 and set(rows) == {t['task_id'] for t in tasks[:1390]}
    remaining = tasks[1390:]
    assert len(remaining) == 10
    failed_id = remaining[0]['task_id']
    path = parent_stage / ARM / 'test/requests' / (failed_id + '_attempt1.json')
    audit = r.read(path)
    attempt = outcomes.last_refusal(audit, failed_id)
    assert attempt and not audit['usage'] and len(audit['requests']) == 1
    assert outcomes.refusal_evidence(attempt)['source'] == 'http_error'
    ledger = r.read(PARENT / 'budget.json')
    unknown = {k for k, v in ledger['attempts'].items() if v['status'] != 'metered'}
    assert unknown == {attempt['request_id']}
    calls = [a for row in rows.values() for a in row['execution']['attempts']]
    prefix = {'completed': 1390, 'total': 1390,
              'correct': sum(bool(row['score']['hard']) for row in rows.values()),
              'tokens': sum(u['total_tokens'] for row in rows.values() for u in row['usage']),
              'native_graphs': sum(bool(row['execution']['all_authored_nodes_completed']) for row in rows.values()),
              'program_calls': len(calls), 'program_ok': sum(a['status'] == 'ok' for a in calls)}
    r.write_json(HERE / 'parent_snapshot.json', {
        'root': str(PARENT), 'budget_sha256': r.sha(PARENT / 'budget.json'),
        'result_digest': r.digest(rows), 'parent_completed': {ARM: prefix},
        'order_offset': 1390, 'refused_attempt': attempt,
        'glm_result_sha256': r.sha(PARENT.parent / 'glm53_flash' / STAGE / 'result.json')})
    r.write_json(HERE / 'remaining_tasks.json', {'tasks': remaining})
    r.write_json(HERE / 'base_config.json', r.read(parent_stage / ARM / 'config.json'))
    r.write_json(HERE / 'campaign_plan.json', {
        'source': r.code_identity(), 'refusal_policy': outcomes.POLICY,
        'guard_sha256': r.sha(outcomes.__file__), 'core_source_unchanged': True,
        'authorization': '完成kimi的实验'})
    r.write_json(HERE / 'spec.json', {
        'tasks': str(HERE / 'remaining_tasks.json'), 'base_config': str(HERE / 'base_config.json'),
        'arms': {ARM: {'package': str(parent_stage / ARM / 'input'),
                      'frozen_source': str(parent_stage / ARM / 'frozen_bank'), 'kind': 'native'}},
        'replay_sources': {ARM: {failed_id: {'path': str(path), 'audit_digest': r.digest(audit),
            'source_plan_sha256': r.sha(parent_stage / 'plan.json'), 'no_new_http': True}}},
        'purpose': 'Complete unchanged V7 Kimi Test1400: preserve 1390 completed tasks; '
                   'score one saved explicit HTTP400 content_filter refusal without resending; '
                   'run the remaining nine canonical tasks with unchanged model and budget.'})
    return {'specs_ready': True, 'remaining': 10, 'replay_without_http': 1}


def prepare():
    assert not ROOT.exists()
    acceptance = r.read(HERE / 'offline_acceptance.json')
    assert acceptance['passed'] and acceptance['driver_sha256'] == r.sha(c.__file__)
    assert acceptance['launcher_sha256'] == r.sha(__file__)
    assert acceptance['guard_sha256'] == r.sha(outcomes.__file__)
    snap = r.read(HERE / 'parent_snapshot.json')
    assert snap['budget_sha256'] == r.sha(PARENT / 'budget.json')
    assert snap['result_digest'] == r.digest(c.completed(PARENT / STAGE, ARM))
    ledger = r.read(PARENT / 'budget.json')
    for name in ('budget.json', 'inherited_budget.json'):
        r.write_json(ROOT / name, ledger)
    governor = r.BudgetGovernor(ROOT / 'budget.json', **ledger['limits'])
    outcomes.retain_reservation(governor, snap['refused_attempt'])
    assert not governor.billing_blocked
    r.write_json(ROOT / 'authorization.json', {**r.read(PARENT / 'authorization.json'),
        'continuation_authorization': '完成kimi的实验', 'refusal_policy': outcomes.POLICY, 'at': r.utc()})
    r.write_json(ROOT / 'cost_carryforward_manifest.json', {
        'parent': str(PARENT), 'parent_budget_sha256': snap['budget_sha256'],
        'parent_completed': snap['parent_completed'],
        'prior_searchqa_tokens': sum(a.get('accounted_tokens', 0) for a in ledger['attempts'].values()),
        'prior_searchqa_http': len(ledger['attempts']), 'limits': ledger['limits'], 'no_limit_increase': True,
        'unknown_reservation_preserved': deepcopy(governor.state['inherited_reservation_recovery']),
        'accounting': 'Inherit physical request IDs once. Preserve 77429 unknown tokens as a reservation, never as measured usage.'})
    c.prepare(STAGE, HERE / 'spec.json')
    stage = ROOT / STAGE
    plan, parent_plan = r.read(stage / 'plan.json'), r.read(PARENT / STAGE / 'plan.json')
    assert plan['input_identity'] == parent_plan['input_identity'] and plan['frozen'] == parent_plan['frozen']
    cfg, old = r.read(stage / ARM / 'config.json'), r.read(PARENT / STAGE / ARM / 'config.json')
    for field in ('llm', 'runtime', 'budget', 'program_environment'):
        assert cfg[field] == old[field]
    plan['order_offset'] = 1390
    plan['execution_adapter'] = {'version': outcomes.VERSION, 'sha256': r.sha(outcomes.__file__),
                                 'core_source_unchanged': True}
    locked = [Path(__file__), HERE / 'check_continuation.py', HERE / 'offline_acceptance.json',
              HERE / 'parent_snapshot.json', HERE / 'remaining_tasks.json',
              PARENT / 'budget.json', PARENT / STAGE / 'plan.json']
    locked += [Path(v['path']) for v in plan['replay_sources'][ARM].values()]
    plan['files'].update({str(p): r.sha(p) for p in locked})
    r.write_json(stage / 'plan.json', plan)
    r.write_json(stage / 'plan_identity.json', {'sha256': r.digest(plan)})
    r.write_json(ROOT / 'continuation_manifest.json', {
        'parent': str(PARENT), 'parent_plan_sha256': r.sha(PARENT / STAGE / 'plan.json'),
        'child_plan_sha256': r.sha(stage / 'plan.json'), 'source': plan['code'],
        'execution_adapter': plan['execution_adapter'], 'parent_completed': 1390,
        'method_model_bank_input_unchanged': True, 'replay_sources': plan['replay_sources'], 'at': r.utc()})
    return c.check(STAGE)


def start(resume=False):
    with (ROOT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        c.check(STAGE)
        assert not alive(ROOT), 'Already running'
        assert not (ROOT / 'pid').exists() or resume
        assert not (ROOT / STAGE / 'result.json').exists() or resume
        assert not (ROOT / 'STOP_AFTER_TASK').exists()
        cmd = [sys.executable, '-B', '-u', str(Path(__file__)), 'run']
        with (ROOT / 'run.log').open('ab') as log:
            child = subprocess.Popen(cmd, cwd=r.SOURCE, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (ROOT / 'pid').write_text(str(child.pid) + '\n')
        launch = {'pid': child.pid, 'at': r.utc(), 'command': cmd, 'resume': resume}
        r.write_json(ROOT / ('resume_launch.json' if resume else 'launch.json'), launch)
        time.sleep(2)
        assert child.poll() is None, 'Inspect run.log'
        return launch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('specs', 'prepare', 'start', 'run', 'status', 'check'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    result = (specs() if args.action == 'specs' else prepare() if args.action == 'prepare'
              else start(args.resume) if args.action == 'start' else c.execute(STAGE) if args.action == 'run'
              else c.check(STAGE) if args.action == 'check' else c.status(STAGE))
    print(json.dumps(result, ensure_ascii=False), flush=True)
