"""Explicit child continuation: preserve 228 results and consume the saved refusal once."""
import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
import campaign as c
from infrastructure_guard import DockerUnavailable, docker_ready
from provider_outcomes import last_refusal, retain_reservation
from provider_retry import POLICY as RETRY_POLICY

r = c.r
PARENT = Path('/home/yangchengyu/searchqa_minimax_r2_test1400_seed42_20261023')
ROOT = Path('/home/yangchengyu/searchqa_minimax_r2_test1400_resume_20261024')
STAGE, ARM = 'fulltest1400', 'r2'
c.CAMPAIGN = ROOT
AUTHORIZATION = '补齐修复然后续跑，补齐相关重试机制让其能自己跑完。'


def specs():
    assert not ROOT.exists() and not (HERE / 'spec.json').exists()
    prior = r.read(PARENT / STAGE / 'plan.json')
    assert r.digest(prior) == r.read(PARENT / STAGE / 'plan_identity.json')['sha256']
    for path, sha in prior['files'].items():
        assert r.sha(path) == sha, path
    for kind, sub in [('frozen', 'frozen_bank'), ('input_identity', 'input')]:
        assert r.tree_identity(PARENT / STAGE / ARM / sub) == prior[kind][ARM]
    rows = c.completed(PARENT / STAGE, ARM)
    tasks = r.read(r.DATASETS / 'test.json')['tasks']
    assert len(rows) == 228 and set(rows) == {t['task_id'] for t in tasks[:228]}
    current = r.read(PARENT / STAGE / 'current.json')['task_id']
    assert current == tasks[228]['task_id']
    audit_path = PARENT / STAGE / ARM / 'test/requests' / (current + '_attempt1.json')
    audit = r.read(audit_path)
    failed = last_refusal(audit, current)
    assert failed is not None
    summary = r.read(PARENT / STAGE / 'result.json')['arms'][ARM]
    assert summary['completed'] == 228 and summary['correct'] == 199
    summary['total'] = 228
    r.write_json(HERE / 'parent_snapshot.json', {'parent_plan_sha256': r.sha(PARENT / STAGE / 'plan.json'),
        'budget_sha256': r.sha(PARENT / 'budget.json'), 'completed_digest': r.digest(rows),
        'parent_completed': {ARM: summary}, 'failed_attempt': failed, 'failed_audit_path': str(audit_path)})
    r.write_json(HERE / 'remaining_tasks.json', {'tasks': tasks[228:]})
    cfg = r.read(PARENT / STAGE / ARM / 'config.json')
    r.write_json(HERE / 'base_config.json', cfg)
    r.write_json(HERE / 'spec.json', {'tasks': str(HERE / 'remaining_tasks.json'),
        'base_config': str(HERE / 'base_config.json'), 'arms': {ARM: {'kind': 'native',
            'package': str(PARENT / STAGE / ARM / 'input'), 'frozen_source': str(PARENT / STAGE / ARM / 'frozen_bank')}},
        'replay_sources': {ARM: {current: {'path': str(audit_path), 'audit_digest': r.digest(audit)}}},
        'purpose': 'Continue MiniMax R2 full Test1400 after 228 tasks. Score the saved provider refusal without another HTTP request. Same Bank/input/model/scorer. Bounded transient retries, conservative unknown billing, no limit increase.'})
    return {'passed': True, 'retained': 228, 'remaining': 1172, 'saved_refusal': current}


def prepare():
    assert not ROOT.exists()
    receipt = r.read(HERE / 'offline_acceptance.json')
    assert receipt['passed'] and receipt['real_http'] == 0
    for name, sha in receipt['files'].items():
        assert r.sha(HERE / name) == sha
    snap = r.read(HERE / 'parent_snapshot.json')
    assert snap['budget_sha256'] == r.sha(PARENT / 'budget.json')
    assert snap['completed_digest'] == r.digest(c.completed(PARENT / STAGE, ARM))
    ledger = r.read(PARENT / 'budget.json')
    for name in ['parent_budget.json', 'budget.json', 'inherited_budget.json']:
        r.write_json(ROOT / name, ledger)
    governor = r.BudgetGovernor(ROOT / 'budget.json', **ledger['limits'])
    retain_reservation(governor, snap['failed_attempt'])
    assert not governor.billing_blocked
    r.write_json(ROOT / 'authorization.json', {'status': 'approved', 'user_request': AUTHORIZATION,
        'at': r.utc(), 'limits': ledger['limits'], 'retry_policy': RETRY_POLICY,
        'scope': 'Continue the same 1400-task experiment; retain unknown costs; no budget increase.'})
    r.write_json(ROOT / 'cost_carryforward_manifest.json', {
        'parent': str(PARENT), 'parent_budget_sha256': snap['budget_sha256'],
        'parent_completed': snap['parent_completed'],
        'prior_searchqa_tokens': sum(a.get('accounted_tokens', 0) for a in ledger['attempts'].values()),
        'prior_searchqa_http': len(ledger['attempts']), 'limits': ledger['limits'], 'no_limit_increase': True,
        'old_unknown_reservations_retained': True, 'old_excluded_200_effective_cost_stays_excluded': True})
    c.prepare(STAGE, HERE / 'spec.json')
    planpath = ROOT / STAGE / 'plan.json'
    plan, parent_plan = r.read(planpath), r.read(PARENT / STAGE / 'plan.json')
    assert plan['frozen'] == parent_plan['frozen'] and plan['input_identity'] == parent_plan['input_identity']
    cfg = r.read(ROOT / STAGE / ARM / 'config.json')
    old_cfg = r.read(PARENT / STAGE / ARM / 'config.json')
    for key in ['llm', 'runtime', 'budget', 'program_environment']:
        assert cfg[key] == old_cfg[key]
    plan.update(order_offset=228, retry_policy=RETRY_POLICY)
    locked = [HERE / name for name in receipt['files']]
    locked += [HERE / 'offline_acceptance.json', HERE / 'parent_snapshot.json', HERE / 'remaining_tasks.json',
               PARENT / 'budget.json', PARENT / STAGE / 'plan.json', Path(snap['failed_audit_path'])]
    plan['files'].update({str(p): r.sha(p) for p in locked})
    r.write_json(planpath, plan)
    r.write_json(ROOT / STAGE / 'plan_identity.json', {'sha256': r.digest(plan)})
    r.write_json(ROOT / 'continuation_manifest.json', {'parent': str(PARENT), 'at': r.utc(),
        'parent_plan_sha256': snap['parent_plan_sha256'], 'child_plan_sha256': r.sha(planpath),
        'source': plan['code'], 'parent_completed': 228, 'saved_refusal_replayed_without_http': True,
        'normal_model_parameters_and_bank_unchanged': True, 'retry_policy': RETRY_POLICY})
    return c.check(STAGE)


def status():
    s = c.status(STAGE)
    ledger, inherited = r.read(ROOT / 'budget.json'), r.read(ROOT / 'inherited_budget.json')
    added = [a for k, a in ledger['attempts'].items() if k not in inherited['attempts']]
    pid = ROOT / 'pid'
    s.update(alive=pid.exists() and Path('/proc', pid.read_text().strip()).exists(),
             continuation_tokens=sum(a.get('accounted_tokens', 0) for a in added),
             continuation_http=len(added), transient_retry_failures=sum(len(v['failures']) for v in ledger.get('provider_retries', {}).values()),
             note='admitted is in-flight; authorized unknown reservations are retained, not measured usage')
    if (ROOT / 'supervisor.json').exists():
        s['supervisor'] = r.read(ROOT / 'supervisor.json')
    return s


def run():
    # Only Docker availability gets controller-level retries; all other errors retain their own bounded policy.
    with (ROOT / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for recovery in range(4):
            try:
                r.write_json(ROOT / 'supervisor.json', {'state': 'running', 'docker_recovery': recovery, 'at': r.utc()})
                result = c.execute(STAGE)
                break
            except DockerUnavailable:
                if recovery == 3:
                    raise
                image = r.read(ROOT / STAGE / ARM / 'config.json')['program_environment']['image_digest']
                r.write_json(ROOT / 'supervisor.json', {'state': 'waiting_for_docker', 'docker_recovery': recovery + 1, 'at': r.utc()})
                for _ in range(6):
                    if (ROOT / 'STOP_AFTER_TASK').exists():
                        raise SystemExit(75)
                    time.sleep(30)
                    try:
                        docker_ready(image)
                        break
                    except DockerUnavailable:
                        continue
                else:
                    raise
        rows = c.completed(PARENT / STAGE, ARM)
        new = c.completed(ROOT / STAGE, ARM)
        assert not set(rows) & set(new)
        rows.update(new)
        assert len(rows) == 1400
        ledger = r.read(ROOT / 'budget.json')
        baseline = r.read(PARENT / 'inherited_budget.json')['attempts']
        attempts = [a for k, a in ledger['attempts'].items() if k not in baseline]
        out = {**result, 'at': r.utc(), 'em': sum(bool(v['score']['hard']) for v in rows.values()) / 1400,
               'f1': sum(v['score']['soft'] for v in rows.values()) / 1400,
               'test_measured_tokens': sum(a.get('accounted_tokens', 0) for a in attempts),
               'test_http': len(attempts), 'test_unknown_reserved_tokens': sum(a['reserved_tokens'] for a in attempts if a['status'] != 'metered'),
               'post_test_development': True, 'freeze_audit': r.read(ROOT / STAGE / 'freeze_audit.json')}
        r.write_json(ROOT / 'completion_report.json', out)
        r.write_json(HERE / 'completion_report.json', out)
        r.write_json(ROOT / 'supervisor.json', {'state': 'completed', 'at': r.utc()})
        return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['specs', 'prepare', 'check', 'status', 'run'])
    action = parser.parse_args().action
    result = {'specs': specs, 'prepare': prepare, 'check': lambda: c.check(STAGE), 'status': status, 'run': run}[action]()
    print(json.dumps(result, ensure_ascii=False))
