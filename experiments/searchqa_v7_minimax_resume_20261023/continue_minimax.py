"""Resume one saved MiniMax overload, keeping the original test prefix and ledger."""
import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'searchqa_v7_qwen_minimax_test1400_20261022'
sys.path.insert(0, str(PRIOR))
import campaign as c
from atomic_skillgraph.empirical.budget_governor import reserved_recovery

r = c.r
PARENT = Path('/home/yangchengyu/searchqa_v7_qwen_minimax_test1400_seed42_20261022/minimax_m3')
ROOT = Path('/home/yangchengyu/searchqa_v7_minimax_test1400_resume_20261023')
STAGE, ARM = 'fulltest1400', 'v7_text_compiler'
c.HERE, c.CAMPAIGN = HERE, ROOT
AUTHORIZATION = '用户于2026-10-23要求：minimax能续跑吗。恢复已中断的Test；保留历史费用预留，不扩大累计预算。'


def alive(root):
    pid = (root / 'pid').read_text().strip() if (root / 'pid').exists() else None
    p = Path('/proc') / str(pid) / 'cmdline'
    return bool(pid and p.exists() and p.read_bytes())


def recover_ledger(ledger, attempt):
    """One explicit 529 recovery, never a blanket waiver of later unknown billing."""
    result = deepcopy(ledger)
    request_id = attempt['request_id']
    entry = result['attempts'][request_id]
    diagnostic = attempt.get('response_diagnostic', {})
    assert attempt['http_status'] == 529 and diagnostic.get('error', {}).get('type') == 'overloaded_error'
    assert attempt['outcome'] == 'error' and attempt.get('raw_usage') is None
    assert entry['status'] == 'unknown_billing' and 'accounted_tokens' not in entry
    assert entry['reserved_tokens'] > 0 and entry['context'] == attempt['request_context']
    assert {k for k,v in result['attempts'].items() if v['status'] != 'metered'} == {request_id}
    receipt = result.setdefault('inherited_reservation_recovery', {
        'schema': 'billing-reservation-recovery.v1', 'authorization': AUTHORIZATION, 'attempts': {}})
    assert request_id not in receipt['attempts']
    receipt['attempts'][request_id] = {'entry_hash': r.digest(entry), 'reserved_tokens': entry['reserved_tokens']}
    assert reserved_recovery(result) == {request_id}
    assert result['attempts'] == ledger['attempts'] and result['limits'] == ledger['limits']
    return result


def specs():
    assert not ROOT.exists() and not alive(PARENT)
    c.CAMPAIGN = PARENT
    c.check(STAGE)
    c.CAMPAIGN = ROOT
    stage = PARENT / STAGE
    assert r.read(stage / 'result.json')['status'] == 'stopped'
    rows, tasks = c.completed(stage, ARM), r.read(stage / 'tasks.json')['tasks']
    assert len(rows) == 254 and set(rows) == {t['task_id'] for t in tasks[:254]}
    remaining = tasks[254:]
    failed = remaining[0]['task_id']
    path = stage / ARM / 'test/requests' / (failed + '_attempt1.json')
    audit = r.read(path)
    attempts = [a for q in audit['requests'] for a in q.get('http_attempts', [])]
    assert len(attempts) == 1 and not audit['usage']
    attempt = attempts[0]
    assert attempt['request_context']['parent_task_id'] == failed
    recover_ledger(r.read(PARENT / 'budget.json'), attempt)
    calls = [a for row in rows.values() for a in row['execution'].get('attempts', [])]
    prefix = {'completed': 254, 'total': 254,
              'correct': sum(bool(row['score']['hard']) for row in rows.values()),
              'tokens': sum(u['total_tokens'] for row in rows.values() for u in row['usage']),
              'native_graphs': sum(bool(row['execution'].get('all_authored_nodes_completed')) for row in rows.values()),
              'program_calls': len(calls), 'program_ok': sum(a['status'] == 'ok' for a in calls)}
    r.write_json(HERE / 'parent_snapshot.json', {
        'root': str(PARENT), 'budget_sha256': r.sha(PARENT / 'budget.json'),
        'result_digest': r.digest(rows), 'parent_completed': {ARM: prefix}, 'failed_attempt': attempt,
        'failed_audit_path': str(path), 'failed_audit_sha256': r.sha(path),
        'qwen_result_sha256': r.sha(PARENT.parent / 'qwen38_flash' / STAGE / 'result.json')})
    r.write_json(HERE / 'remaining_tasks.json', {'tasks': remaining})
    r.write_json(HERE / 'base_config.json', r.read(stage / ARM / 'config.json'))
    shutil.copyfile(PRIOR / 'provider_outcomes.py', HERE / 'provider_outcomes.py')
    r.write_json(HERE / 'campaign_plan.json', {'source': r.code_identity(), 'authorization': AUTHORIZATION,
        'core_source_unchanged': True, 'only_one_saved_overload_recovery': True})
    r.write_json(HERE / 'spec.json', {
        'tasks': str(HERE / 'remaining_tasks.json'), 'base_config': str(HERE / 'base_config.json'),
        'arms': {ARM: {'package': str(stage / ARM / 'input'), 'kind': 'native',
                      'frozen_source': str(stage / ARM / 'frozen_bank')}},
        'purpose': 'Continue MiniMax V7 Test1400 after saved HTTP529 overloaded_error. '
                   'Keep 254 completed tasks and original physical charges/reservation; '
                   'one authorized new execution of interrupted task, then remaining canonical tasks. '
                   'Same model, prompt, Bank, method, scoring and configured per-execution limits; no cumulative increase.'})
    return {'ready': True, 'preserved': 254, 'remaining': len(remaining), 'failed_task': failed}


def prepare():
    assert not ROOT.exists()
    acceptance = r.read(HERE / 'offline_acceptance.json')
    assert acceptance['passed'] and acceptance['launcher_sha256'] == r.sha(__file__)
    assert acceptance['driver_sha256'] == r.sha(c.__file__)
    snap = r.read(HERE / 'parent_snapshot.json')
    assert snap['budget_sha256'] == r.sha(PARENT / 'budget.json')
    assert snap['result_digest'] == r.digest(c.completed(PARENT / STAGE, ARM))
    ledger = r.read(PARENT / 'budget.json')
    r.write_json(ROOT / 'inherited_budget.json', ledger)
    recovered = recover_ledger(ledger, snap['failed_attempt'])
    r.write_json(ROOT / 'budget.json', recovered)
    governor = r.BudgetGovernor(ROOT / 'budget.json', **ledger['limits'])
    assert not governor.billing_blocked
    r.write_json(ROOT / 'authorization.json', {**r.read(PARENT / 'authorization.json'),
        'continuation_authorization': AUTHORIZATION, 'at': r.utc(),
        'retry_scope': 'One new execution of the saved overloaded task in a new run scope; '
                       'its old unknown reservation remains charged to the shared cumulative limit.'})
    r.write_json(ROOT / 'cost_carryforward_manifest.json', {
        'parent': str(PARENT), 'parent_budget_sha256': snap['budget_sha256'],
        'parent_completed': snap['parent_completed'],
        'prior_searchqa_tokens': sum(a.get('accounted_tokens', 0) for a in ledger['attempts'].values()),
        'prior_searchqa_http': len(ledger['attempts']), 'limits': ledger['limits'], 'no_limit_increase': True,
        'unknown_reservation_preserved': recovered['inherited_reservation_recovery'],
        'accounting': 'Inherit every physical request ID once. Keep 77459 tokens as unknown reservation, not measured usage.'})
    c.prepare(STAGE, HERE / 'spec.json')
    stage = ROOT / STAGE
    plan, parent_plan = r.read(stage / 'plan.json'), r.read(PARENT / STAGE / 'plan.json')
    assert plan['input_identity'] == parent_plan['input_identity'] and plan['frozen'] == parent_plan['frozen']
    cfg, old = r.read(stage / ARM / 'config.json'), r.read(PARENT / STAGE / ARM / 'config.json')
    for field in ('llm', 'runtime', 'budget', 'program_environment'):
        assert cfg[field] == old[field]
    plan['order_offset'] = 254
    plan['recovery'] = {'version': 'explicit-single-overload-continuation.v1',
        'original_request_id': snap['failed_attempt']['request_id'],
        'old_request_scope': snap['failed_attempt']['request_context']['parent_scope'],
        'old_reservation_charged_to_cumulative_ledger': True, 'max_recovery_attempts_for_saved_failure': 1}
    locked = [Path(__file__), HERE / 'check_continuation.py', HERE / 'offline_acceptance.json',
              HERE / 'parent_snapshot.json', HERE / 'remaining_tasks.json',
              PARENT / 'budget.json', PARENT / STAGE / 'plan.json', Path(snap['failed_audit_path']),
              PARENT.parent / 'qwen38_flash' / STAGE / 'result.json']
    plan['files'].update({str(p): r.sha(p) for p in locked})
    r.write_json(stage / 'plan.json', plan)
    r.write_json(stage / 'plan_identity.json', {'sha256': r.digest(plan)})
    r.write_json(ROOT / 'continuation_manifest.json', {'parent': str(PARENT),
        'parent_plan_sha256': r.sha(PARENT / STAGE / 'plan.json'), 'child_plan_sha256': r.sha(stage / 'plan.json'),
        'source': plan['code'], 'recovery': plan['recovery'], 'parent_completed': 254,
        'method_model_bank_input_unchanged': True, 'at': r.utc()})
    return c.check(STAGE)


def start():
    with (ROOT / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        c.check(STAGE)
        assert not alive(ROOT) and not alive(PARENT)
        assert not (ROOT / 'pid').exists() and not (ROOT / STAGE / 'result.json').exists()
        assert not (ROOT / 'STOP_AFTER_TASK').exists()
        cmd = [sys.executable, '-B', '-u', str(Path(__file__)), 'run']
        with (ROOT / 'run.log').open('ab') as log:
            child = subprocess.Popen(cmd, cwd=r.SOURCE, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (ROOT / 'pid').write_text(str(child.pid) + '\n')
        receipt = {'pid': child.pid, 'at': r.utc(), 'command': cmd}
        r.write_json(ROOT / 'launch.json', receipt)
        time.sleep(2)
        assert child.poll() is None, 'Inspect run.log'
        return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('specs', 'prepare', 'start', 'run', 'status', 'check'))
    args = parser.parse_args()
    value = (specs() if args.action == 'specs' else prepare() if args.action == 'prepare'
             else start() if args.action == 'start' else c.execute(STAGE) if args.action == 'run'
             else c.check(STAGE) if args.action == 'check' else c.status(STAGE))
    print(json.dumps(value, ensure_ascii=False), flush=True)
