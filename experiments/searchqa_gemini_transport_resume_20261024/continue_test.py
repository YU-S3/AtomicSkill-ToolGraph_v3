"""Explicit Gemini child: retain 18 results, two 504 attempts and the original total ceiling."""
import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import time

import campaign as c
from infrastructure_guard import DockerUnavailable, docker_ready
from provider_outcomes import retain_reservation
from provider_retry import POLICY, transient

r, HERE = c.r, c.HERE
PARENT = Path('/home/yangchengyu/searchqa_target_accuracy_r2_20261024/gemini')
ROOT = Path('/home/yangchengyu/searchqa_gemini_transport_resume_20261024')
STAGE, ARM = 'fulltest1400', 'gepa'
c.CAMPAIGN = ROOT
CODE_FILES = ['campaign.py', 'runtime.py', 'transport_budget.py', 'provider_retry.py',
              'provider_outcomes.py', 'infrastructure_guard.py', 'continue_test.py', 'check_offline.py']
AUTHORIZATION = '修复gemini的问题，续跑，并确保重试等机制能保证其正常跑完'


def specs():
    assert not ROOT.exists() and not (HERE / 'spec.json').exists()
    prior = r.read(PARENT / STAGE / 'plan.json')
    assert r.digest(prior) == r.read(PARENT / STAGE / 'plan_identity.json')['sha256']
    for path, sha in prior['files'].items():
        assert r.sha(path) == sha, path
    for kind, sub in [('frozen', 'frozen_bank'), ('input_identity', 'input')]:
        assert r.tree_identity(PARENT / STAGE / ARM / sub) == prior[kind][ARM]
    tasks = r.read(r.DATASETS / 'test.json')['tasks']
    rows = c.completed(PARENT / STAGE, ARM)
    assert len(rows) == 18 and set(rows) == {t['task_id'] for t in tasks[:18]}
    current = r.read(PARENT / STAGE / 'current.json')['task_id']
    assert current == tasks[18]['task_id']
    audit_path = PARENT / STAGE / ARM / 'test/requests' / (current + '_attempt1.json')
    audit = r.read(audit_path)
    attempts = [a for req in audit['requests'] for a in req['http_attempts']]
    assert len(attempts) == 2 and all(a['http_status'] == 504 and transient(a, RuntimeError()) for a in attempts)
    ledger = r.read(PARENT / 'budget.json')
    retry_key, retry = next((key, value) for key, value in ledger['provider_retries'].items()
                            if value['context']['parent_task_id'] == current)
    assert retry['failures'] == [a['request_id'] for a in attempts]
    assert retry['stopped'] == 'Retry cannot fit the remaining task/global budget'
    payload = {**attempts[0]['final_payload_audit']['serialized_parameters'],
               'messages': attempts[0]['final_payload_audit']['messages']}
    assert not attempts[0]['final_payload_audit']['tools']
    summary = deepcopy(r.read(PARENT / STAGE / 'result.json')['arms'][ARM])
    assert summary['completed'] == 18 and summary['correct'] == 17
    summary['total'] = 18
    binding = {k: retry['context'][k] for k in ('parent_scope', 'logical_decision_id', 'purpose')}
    binding.update(retry_key=retry_key, payload_digest=r.digest(payload), initial_failures=retry['failures'])
    r.write_json(HERE / 'parent_snapshot.json', {'parent_plan_sha256': r.sha(PARENT / STAGE / 'plan.json'),
        'budget_sha256': r.sha(PARENT / 'budget.json'), 'completed_digest': r.digest(rows),
        'parent_completed': {ARM: summary}, 'attempts': attempts, 'failed_audit_path': str(audit_path),
        'task_id': current, 'binding': binding, 'retry_entry': retry})
    r.write_json(HERE / 'remaining_tasks.json', {'tasks': tasks[18:]})
    r.write_json(HERE / 'base_config.json', r.read(PARENT / STAGE / ARM / 'config.json'))
    r.write_json(HERE / 'spec.json', {'tasks': str(HERE / 'remaining_tasks.json'),
        'base_config': str(HERE / 'base_config.json'), 'arms': {ARM: {'kind': 'native',
            'package': str(PARENT / STAGE / ARM / 'input'), 'frozen_source': str(PARENT / STAGE / ARM / 'frozen_bank')}},
        'purpose': 'Continue the same Gemini Test1400 after 18 completed tasks. Verified transport failures remain globally charged but do not consume productive task tokens. Same Bank/input/model/scorer; original retry count and global limits retained.'})
    return {'retained': 18, 'remaining': 1382, 'prior_failures': 2, 'remaining_attempts': 2}


def migrate(governor, snapshot):
    """Lift only the proven local-budget stop; never reset attempts or their charges."""
    binding = snapshot['binding']
    assert governor.state['provider_retries'][binding['retry_key']] == snapshot['retry_entry']
    for attempt in snapshot['attempts']:
        assert transient(attempt, RuntimeError())
        retain_reservation(governor, attempt, 'bounded_transient_retry')
        governor.transport_failure(attempt, snapshot['failed_audit_path'])
    entry = governor.state['provider_retries'][binding['retry_key']]
    entry['previous_policy'] = deepcopy(entry['policy'])
    entry['policy'] = deepcopy(POLICY)
    entry['retry_required_tokens'] = governor.state['attempts'][entry['failures'][0]]['reserved_tokens']
    entry['migration'] = {'previous_stop': entry.pop('stopped'), 'previous_retry_cap': entry.pop('retry_completion_cap'),
                        'reason': 'Verified transport charges separated from productive task cap; global costs unchanged.',
                        'at': r.utc()}
    governor.state.setdefault('transport_continuations', {})[snapshot['task_id']] = binding
    r.write_json(governor.path, governor.state)


def prepare():
    assert not ROOT.exists()
    receipt, snap = r.read(HERE / 'offline_acceptance.json'), r.read(HERE / 'parent_snapshot.json')
    assert receipt['passed'] and receipt['real_http'] == 0
    for name, sha in receipt['files'].items():
        assert r.sha(HERE / name) == sha, name
    assert snap['budget_sha256'] == r.sha(PARENT / 'budget.json')
    assert snap['completed_digest'] == r.digest(c.completed(PARENT / STAGE, ARM))
    ledger = r.read(PARENT / 'budget.json')
    for name in ('parent_budget.json', 'inherited_budget.json', 'budget.json'):
        r.write_json(ROOT / name, ledger)
    governor = r.BudgetGovernor(ROOT / 'budget.json', **ledger['limits'])
    migrate(governor, snap)
    assert not governor.billing_blocked
    r.write_json(ROOT / 'authorization.json', {'status': 'approved', 'user_request': AUTHORIZATION,
        'at': r.utc(), 'limits': ledger['limits'], 'policy': POLICY, 'no_limit_increase': True})
    r.write_json(ROOT / 'cost_carryforward_manifest.json', {'parent': str(PARENT),
        'parent_budget_sha256': snap['budget_sha256'], 'parent_completed': snap['parent_completed'],
        'prior_searchqa_tokens': sum(a.get('accounted_tokens', 0) for a in ledger['attempts'].values()),
        'prior_searchqa_http': len(ledger['attempts']), 'limits': ledger['limits'],
        'no_limit_increase': True, 'all_unknown_reservations_retained': True})
    c.prepare(STAGE, HERE / 'spec.json')
    path = ROOT / STAGE / 'plan.json'
    plan, old = r.read(path), r.read(PARENT / STAGE / 'plan.json')
    assert plan['frozen'] == old['frozen'] and plan['input_identity'] == old['input_identity']
    cfg, old_cfg = r.read(ROOT / STAGE / ARM / 'config.json'), r.read(HERE / 'base_config.json')
    for key in ('llm', 'runtime', 'budget', 'program_environment'):
        assert cfg[key] == old_cfg[key], key
    plan.update(order_offset=18, retry_policy=POLICY)
    locked = [HERE / name for name in CODE_FILES + ['offline_acceptance.json', 'parent_snapshot.json', 'remaining_tasks.json']]
    locked += [PARENT / 'budget.json', PARENT / STAGE / 'plan.json', Path(snap['failed_audit_path'])]
    plan['files'].update({str(p): r.sha(p) for p in locked})
    r.write_json(path, plan)
    r.write_json(ROOT / STAGE / 'plan_identity.json', {'sha256': r.digest(plan)})
    r.write_json(ROOT / 'continuation_manifest.json', {'parent': str(PARENT), 'at': r.utc(),
        'parent_completed': 18, 'parent_plan_sha256': snap['parent_plan_sha256'],
        'child_plan_sha256': r.sha(path), 'source': plan['code'], 'retry_policy': POLICY,
        'same_bank_input_model_and_scorer': True, 'prior_failures': 2, 'remaining_attempts_for_interrupted_request': 2})
    return c.check(STAGE)


def status():
    result = c.status(STAGE)
    ledger, inherited = r.read(ROOT / 'budget.json'), r.read(ROOT / 'inherited_budget.json')
    added = [a for k, a in ledger['attempts'].items() if k not in inherited['attempts']]
    pid = ROOT / 'pid'
    result.update(alive=pid.exists() and Path('/proc', pid.read_text().strip()).exists(),
        continuation_tokens=sum(a.get('accounted_tokens', 0) for a in added), continuation_http=len(added),
        remaining_admission_tokens=ledger['limits']['token_limit'] - sum(a.get('accounted_tokens', a['reserved_tokens']) for a in ledger['attempts'].values()),
        remaining_http=ledger['limits']['request_limit'] - len(ledger['attempts']),
        transport_failure_count=len(ledger.get('transport_failures', {})))
    if (ROOT / 'supervisor.json').exists():
        result['supervisor'] = r.read(ROOT / 'supervisor.json')
    if (ROOT / STAGE / 'current.json').exists():
        result['current'] = r.read(ROOT / STAGE / 'current.json')
    return result


def run():
    with (ROOT / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        supervisor = r.read(ROOT / 'supervisor.json') if (ROOT / 'supervisor.json').exists() else {}
        recovery = supervisor.get('docker_recoveries', 0)
        try:
            while True:
                try:
                    r.write_json(ROOT / 'supervisor.json', {'state': 'running', 'docker_recoveries': recovery, 'at': r.utc()})
                    result = c.execute(STAGE)
                    break
                except DockerUnavailable:
                    if recovery >= 3:
                        raise
                    recovery += 1
                    r.write_json(ROOT / 'supervisor.json', {'state': 'waiting_for_docker', 'docker_recoveries': recovery, 'at': r.utc()})
                    image = r.read(ROOT / STAGE / ARM / 'config.json')['program_environment']['image_digest']
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
            old, new = c.completed(PARENT / STAGE, ARM), c.completed(ROOT / STAGE, ARM)
            assert not set(old) & set(new)
            rows = {**old, **new}
            assert set(rows) == {t['task_id'] for t in r.read(r.DATASETS / 'test.json')['tasks']}
            ledger = r.read(ROOT / 'budget.json')
            # Include failed Test attempts as well as successful Test usage; probes stay in cumulative accounting.
            attempts = [a for a in ledger['attempts'].values()
                        if any(str(p / STAGE) in a['context'].get('parent_scope', '') for p in (PARENT, ROOT))]
            report = {**result, 'em': sum(bool(v['score']['hard']) for v in rows.values()) / 1400,
                'f1': sum(v['score']['soft'] for v in rows.values()) / 1400,
                'test_tokens': sum(a.get('accounted_tokens', 0) for a in attempts), 'test_http': len(attempts),
                'test_unknown_reserved_tokens': sum(a['reserved_tokens'] for a in attempts if a['status'] != 'metered'),
                'target': .88, 'target_met': sum(bool(v['score']['hard']) for v in rows.values()) >= 1232,
                'freeze_audit': r.read(ROOT / STAGE / 'freeze_audit.json'), 'post_test_development': True, 'at': r.utc()}
            r.write_json(ROOT / 'completion_report.json', report)
            r.write_json(HERE / 'completion_report.json', report)
            r.write_json(ROOT / 'supervisor.json', {'state': 'completed', 'docker_recoveries': recovery, 'at': r.utc()})
            return report
        except BaseException as exc:
            r.write_json(ROOT / 'supervisor.json', {'state': 'stopped', 'docker_recoveries': recovery,
                'error': {'type': type(exc).__name__, 'message': str(exc)}, 'at': r.utc()})
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['specs', 'prepare', 'check', 'status', 'run'])
    result = {'specs': specs, 'prepare': prepare, 'check': lambda: c.check(STAGE), 'status': status, 'run': run}[parser.parse_args().action]()
    print(json.dumps(result, ensure_ascii=False))
