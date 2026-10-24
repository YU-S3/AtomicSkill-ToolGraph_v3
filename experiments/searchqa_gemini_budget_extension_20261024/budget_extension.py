"""Authorized +500K continuation, reusing the already tested frozen Gemini runtime."""
import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
RUNTIME = Path('/mnt/d/T3S_exp/deliveries/searchqa_gemini_transport_resume_20261024')
sys.path.insert(0, str(RUNTIME))
import continue_test as previous
from infrastructure_guard import DockerUnavailable, docker_ready

c, r = previous.c, previous.r
PARENT = previous.ROOT
ANCESTORS = [previous.PARENT, PARENT]
ROOT = Path('/home/yangchengyu/searchqa_gemini_budget_extension_20261024')
STAGE, ARM = previous.STAGE, previous.ARM
c.CAMPAIGN = ROOT
AUTHORIZATION = '批准追加50万 tokens，继续跑完'


def prefix():
    rows = {}
    for root in ANCESTORS:
        added = c.completed(root / STAGE, ARM)
        assert not set(rows) & set(added)
        rows.update(added)
    return rows


def prepare():
    assert not ROOT.exists()
    receipt = r.read(HERE / 'offline_acceptance.json')
    assert receipt['passed'] and receipt['real_http'] == 0 and receipt['extension_sha256'] == r.sha(__file__)
    for name, sha in receipt['runtime_files'].items():
        assert r.sha(RUNTIME / name) == sha
    with (PARENT / 'supervisor.lock').open('a') as supervisor, (PARENT / 'execution.lock').open('a') as execution:
        fcntl.flock(supervisor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(execution, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert (PARENT / 'STOP_AFTER_TASK').exists()
        c.CAMPAIGN = PARENT
        try:
            c.check(STAGE)
            summary = deepcopy(c.status(STAGE)['arms'][ARM])
        finally:
            c.CAMPAIGN = ROOT
        rows, tasks = prefix(), r.read(r.DATASETS / 'test.json')['tasks']
        count = len(rows)
        assert 0 < count < 1400 and set(rows) == {t['task_id'] for t in tasks[:count]}
        assert summary['completed'] == count
        summary['total'] = count
        old = r.read(PARENT / 'budget.json')
        assert old['limits'] == {'token_limit': 10272289, 'finish_reserve': 0, 'request_limit': 2257}
        assert not any(a['status'] == 'admitted' for a in old['attempts'].values())
        assert not any(a.get('next_retry') for a in old.get('provider_retries', {}).values())
        ledger = deepcopy(old)
        ledger['limits']['token_limit'] += 500000
        authority = {'status': 'approved', 'user_request': AUTHORIZATION, 'at': r.utc(),
            'old_limits': old['limits'], 'new_limits': ledger['limits'], 'additional_token_limit': 500000,
            'additional_http_limit': 0, 'parent': str(PARENT), 'parent_budget_sha256': r.sha(PARENT / 'budget.json'),
            'retained_attempts': len(old['attempts']), 'retained_attempts_digest': r.digest(old['attempts']),
            'retained_completed': count, 'retained_results_digest': r.digest(rows)}
        r.write_json(ROOT / 'parent_budget.json', old)
        for name in ('budget.json', 'inherited_budget.json'):
            r.write_json(ROOT / name, ledger)
        assert not r.BudgetGovernor(ROOT / 'budget.json', **ledger['limits']).billing_blocked
        r.write_json(ROOT / 'authorization.json', authority)
        r.write_json(ROOT / 'budget_extension_manifest.json', authority)
        r.write_json(ROOT / 'cost_carryforward_manifest.json', {'parent': str(PARENT), 'ancestors': list(map(str, ANCESTORS)),
            'parent_completed': {ARM: summary}, 'prior_searchqa_tokens': sum(a.get('accounted_tokens', 0) for a in old['attempts'].values()),
            'prior_searchqa_http': len(old['attempts']), 'all_prior_costs_and_unknown_reservations_retained': True,
            'authorization': str(ROOT / 'budget_extension_manifest.json')})
        r.write_json(HERE / 'remaining_tasks.json', {'tasks': tasks[count:]})
        r.write_json(HERE / 'spec.json', {'tasks': str(HERE / 'remaining_tasks.json'),
            'base_config': str(PARENT / STAGE / ARM / 'config.json'),
            'arms': {ARM: {'kind': 'native', 'package': str(PARENT / STAGE / ARM / 'input'),
                           'frozen_source': str(PARENT / STAGE / ARM / 'frozen_bank')}},
            'purpose': 'Authorized +500000 cumulative tokens. Continue remaining canonical Test tasks with identical Bank/model/input/scorer and all prior results, retries and charges.'})
        c.prepare(STAGE, HERE / 'spec.json')
        planpath = ROOT / STAGE / 'plan.json'
        plan, prior = r.read(planpath), r.read(PARENT / STAGE / 'plan.json')
        assert plan['input_identity'] == prior['input_identity'] and plan['frozen'] == prior['frozen']
        cfg, old_cfg = r.read(ROOT / STAGE / ARM / 'config.json'), r.read(PARENT / STAGE / ARM / 'config.json')
        expected = deepcopy(old_cfg)
        expected['budget']['token_limit'] += 500000
        expected['data_dir'] = cfg['data_dir']
        expected['experiment']['output_dir'] = cfg['experiment']['output_dir']
        assert cfg == expected, 'Only run paths and authorized cumulative token limit may change'
        locked = [Path(__file__), HERE / 'check_extension.py', HERE / 'offline_acceptance.json', HERE / 'remaining_tasks.json',
                  ROOT / 'budget_extension_manifest.json', ROOT / 'parent_budget.json', PARENT / 'budget.json', PARENT / STAGE / 'plan.json']
        locked += [RUNTIME / name for name in receipt['runtime_files']]
        plan['files'].update({str(p): r.sha(p) for p in locked})
        plan['order_offset'] = count
        r.write_json(planpath, plan)
        r.write_json(ROOT / STAGE / 'plan_identity.json', {'sha256': r.digest(plan)})
        r.write_json(ROOT / 'continuation_manifest.json', {**authority, 'child': str(ROOT),
            'child_plan_sha256': r.sha(planpath), 'remaining_tasks': len(tasks) - count,
            'same_bank_model_inputs_scorer': True, 'runtime_unchanged': str(RUNTIME)})
        r.write_json(ROOT / 'supervisor.json', {'state': 'prepared', 'docker_recoveries': r.read(PARENT / 'supervisor.json').get('docker_recoveries', 0)})
    return c.check(STAGE)


def status():
    value = c.status(STAGE)
    ledger = r.read(ROOT / 'budget.json')
    value.update(limits=ledger['limits'], remaining_admission_tokens=ledger['limits']['token_limit'] - sum(
        a.get('accounted_tokens', a['reserved_tokens']) for a in ledger['attempts'].values()),
        supervisor=r.read(ROOT / 'supervisor.json'),
        billing_states={state: sum(a['status'] == state for a in ledger['attempts'].values()) for state in ('admitted', 'metered', 'unknown_billing')})
    return value


def run():
    with (ROOT / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        recovery = r.read(ROOT / 'supervisor.json').get('docker_recoveries', 0)
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
            old, new = prefix(), c.completed(ROOT / STAGE, ARM)
            assert not set(old) & set(new)
            rows = {**old, **new}
            assert set(rows) == {t['task_id'] for t in r.read(r.DATASETS / 'test.json')['tasks']}
            ledger = r.read(ROOT / 'budget.json')
            attempts = [a for a in ledger['attempts'].values() if any(str(p / STAGE) in a['context'].get('parent_scope', '') for p in ANCESTORS + [ROOT])]
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
    parser.add_argument('action', choices=['prepare', 'run', 'status', 'check'])
    print(json.dumps({'prepare': prepare, 'run': run, 'status': status, 'check': lambda: c.check(STAGE)}[parser.parse_args().action](), ensure_ascii=False))
