"""One fresh full SearchQA Test per configured model, using the exact Gemini 89% Bank."""
import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import campaign as c
from infrastructure_guard import DockerUnavailable, docker_ready
from provider_retry import POLICY

r, HERE = c.r, c.HERE
ROOT = Path('/home/yangchengyu/searchqa_gepa_deepseek_kimi_test1400_20261024')
BANK_RUN = Path('/home/yangchengyu/searchqa_gemini_budget_extension_20261024')
BANK = BANK_RUN / 'fulltest1400/gepa'
STAGE, ARM = 'fulltest1400', 'gepa'
AUTHORIZATION = '同样的高准确率bank启动一轮deepseek、一轮kimi的test'
MODELS = {
    'deepseek': {'parent': '/home/yangchengyu/searchqa_v7_repeat2_test1400_seed42_20261021',
                'config_arm': 'v7_repeat_1', 'model': 'deepseek-v4-flash'},
    'kimi': {'parent': '/home/yangchengyu/searchqa_v7_kimi_test1400_refusal_finish_20261022',
             'config_arm': 'v7_text_compiler', 'model': 'kimi-k2.6'}}
ALLOCATION = {'tokens': 10000000, 'http': 2000}
CODE_FILES = ['campaign.py', 'runtime.py', 'transport_budget.py', 'provider_retry.py', 'provider_outcomes.py',
              'infrastructure_guard.py', 'run_models.py', 'check_models.py']


def setup():
    assert not (HERE / 'campaign_plan.json').exists()
    bank_plan = r.read(BANK_RUN / STAGE / 'plan.json')
    assert r.read(BANK_RUN / 'completion_report.json')['em'] == .89
    assert r.tree_identity(BANK / 'input') == bank_plan['input_identity'][ARM]
    assert r.tree_identity(BANK / 'frozen_bank') == bank_plan['frozen'][ARM]
    r.write_json(HERE / 'campaign_plan.json', {'source': r.code_identity(), 'authorization': AUTHORIZATION,
        'models': MODELS, 'allocation_per_model': ALLOCATION, 'bank_origin': str(BANK),
        'bank_input_identity': bank_plan['input_identity'][ARM], 'frozen_identity': bank_plan['frozen'][ARM],
        'canonical_tasks': str(r.DATASETS / 'test.json'), 'task_sha256': r.sha(r.DATASETS / 'test.json'),
        'scope': 'Exactly one fresh Test1400 per model; parallel independent ledgers; no training, candidate selection, extra repeat or automatic budget expansion.',
        'post_test_development': True, 'retry_policy': POLICY})
    for model, info in MODELS.items():
        parent = Path(info['parent'])
        prior = parent / STAGE / info['config_arm'] / 'config.json'
        cfg = r.read(prior)
        assert cfg['llm']['model'] == info['model']
        assert cfg['program_environment'] == r.read(BANK / 'config.json')['program_environment']
        old_retry = cfg['llm']['max_retries']
        cfg['llm']['max_retries'] = 0  # One durable retry loop, never nested retries.
        cfg['experiment']['authored_bank_runtime'] = r.read(BANK / 'frozen_bank/imported_bank.json')['method_version']
        r.write_json(HERE / model / 'base_config.json', cfg)
        r.write_json(HERE / model / 'config_provenance.json', {'source': str(prior), 'sha256': r.sha(prior),
            'normal_generation_parameters_unchanged': True, 'previous_provider_max_retries': old_retry,
            'new_provider_max_retries': 0, 'durable_outer_retry_policy': POLICY,
            'bank_runtime_changed_to_requested_gepa_bank': True})
        r.write_json(HERE / model / 'spec.json', {'tasks': str(r.DATASETS / 'test.json'),
            'base_config': str(HERE / model / 'base_config.json'),
            'arms': {ARM: {'kind': 'native', 'package': str(BANK / 'input'), 'frozen_source': str(BANK / 'frozen_bank')}},
            'purpose': 'One new full Test1400 with the exact frozen authored Bank that scored Gemini 1246/1400. Same canonical public inputs and official scorer. Configured model parameters; post-Test development evaluation.'})
    return r.read(HERE / 'campaign_plan.json')


def prepare(model):
    info = MODELS[model]
    root, parent = ROOT / model, Path(info['parent'])
    c.CAMPAIGN = root
    assert not root.exists()
    receipt = r.read(HERE / 'offline_acceptance.json')
    assert receipt['passed'] and receipt['real_http'] == 0
    assert all(r.sha(HERE / name) == sha for name, sha in receipt['files'].items())
    assert r.read(parent / STAGE / 'result.json')['status'] == 'completed'
    old = r.read(parent / 'budget.json')
    assert not any(a['status'] == 'admitted' for a in old['attempts'].values())
    charge = sum(a.get('accounted_tokens', a['reserved_tokens']) for a in old['attempts'].values())
    state = deepcopy(old)
    state['limits'] = {'token_limit': charge + ALLOCATION['tokens'], 'request_limit': len(old['attempts']) + ALLOCATION['http'], 'finish_reserve': 0}
    carry = {'parent': str(parent), 'parent_budget_sha256': r.sha(parent / 'budget.json'),
        'prior_searchqa_tokens': sum(a.get('accounted_tokens', 0) for a in old['attempts'].values()),
        'prior_searchqa_http': len(old['attempts']), 'prior_admission_charge': charge,
        'new_allocation': ALLOCATION, 'physical_entries_unchanged': True, 'all_unknown_reservations_retained': True,
        'no_inherited_task_results': True, 'no_automatic_budget_expansion': True}
    if model == 'deepseek':
        prior = r.read(parent / 'cost_carryforward_manifest.json')
        other = prior['concurrent_sibling']['allocation_reserved_in_full']
        reserve = prior['r3_remaining_reservation']
        envelope = {'tokens': state['limits']['token_limit'] + other['tokens'] + reserve['tokens'],
                    'http': state['limits']['request_limit'] + other['http'] + reserve['http']}
        authority = prior['original_authority_limits']
        assert envelope['tokens'] <= authority['token_limit'] and envelope['http'] <= authority['request_limit']
        carry.update(original_authority_limits=authority, authority_envelope=envelope,
                     concurrent_sibling=prior['concurrent_sibling'], r3_remaining_reservation=reserve)
    r.write_json(root / 'parent_budget.json', old)
    for name in ('budget.json', 'inherited_budget.json'):
        r.write_json(root / name, state)
    assert not r.BudgetGovernor(root / 'budget.json', **state['limits']).billing_blocked
    r.write_json(root / 'authorization.json', {'status': 'approved', 'at': r.utc(), 'user_request': AUTHORIZATION,
        'model': info['model'], 'scope': 'One fresh full Test1400 with the requested identical Bank, existing model generation settings and bounded retries.',
        'new_allocation': ALLOCATION, 'limits': state['limits'], 'retry_policy': POLICY})
    r.write_json(root / 'cost_carryforward_manifest.json', carry)
    c.prepare(STAGE, HERE / model / 'spec.json')
    path = root / STAGE / 'plan.json'
    plan = r.read(path)
    bank_plan = r.read(HERE / 'campaign_plan.json')
    assert plan['frozen'][ARM] == bank_plan['frozen_identity']
    assert plan['input_identity'][ARM] == bank_plan['bank_input_identity']
    files = [HERE / name for name in CODE_FILES] + [HERE / 'offline_acceptance.json',
        HERE / model / 'config_provenance.json', parent / 'budget.json', root / 'parent_budget.json', BANK_RUN / 'completion_report.json']
    plan['files'].update({str(p): r.sha(p) for p in files})
    r.write_json(path, plan)
    r.write_json(root / STAGE / 'plan_identity.json', {'sha256': r.digest(plan)})
    r.write_json(root / 'supervisor.json', {'state': 'prepared', 'docker_recoveries': 0})
    return c.check(STAGE)


def status(model):
    root = ROOT / model
    c.CAMPAIGN = root
    value = c.status(STAGE)
    ledger, old = r.read(root / 'budget.json'), r.read(root / 'inherited_budget.json')
    added = [v for k, v in ledger['attempts'].items() if k not in old['attempts']]
    value.update(model=MODELS[model]['model'], new_measured_tokens=sum(a.get('accounted_tokens', 0) for a in added),
        new_http=len(added), new_error_attempts=sum(a.get('outcome') == 'error' for a in added),
        new_unknown_reserved_tokens=sum(a['reserved_tokens'] for a in added if a['status'] == 'unknown_billing'),
        inflight=sum(a['status'] == 'admitted' for a in added),
        remaining_admission_tokens=ledger['limits']['token_limit'] - sum(a.get('accounted_tokens', a['reserved_tokens']) for a in ledger['attempts'].values()),
        supervisor=r.read(root / 'supervisor.json'))
    return value


def run(model):
    root = ROOT / model
    c.CAMPAIGN = root
    with (root / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        recovery = r.read(root / 'supervisor.json').get('docker_recoveries', 0)
        try:
            while True:
                try:
                    r.write_json(root / 'supervisor.json', {'state': 'running', 'docker_recoveries': recovery, 'at': r.utc()})
                    c.execute(STAGE)
                    break
                except DockerUnavailable:
                    if recovery >= 3:
                        raise
                    recovery += 1
                    r.write_json(root / 'supervisor.json', {'state': 'waiting_for_docker', 'docker_recoveries': recovery, 'at': r.utc()})
                    cfg = r.read(root / STAGE / ARM / 'config.json')
                    for _ in range(6):
                        if (root / 'STOP_AFTER_TASK').exists():
                            raise SystemExit(75)
                        time.sleep(30)
                        try:
                            docker_ready(cfg['program_environment']['image_digest'])
                            break
                        except DockerUnavailable:
                            continue
                    else:
                        raise
            rows = c.completed(root / STAGE, ARM)
            assert set(rows) == {t['task_id'] for t in r.read(r.DATASETS / 'test.json')['tasks']}
            result = {**status(model), 'em': sum(bool(row['score']['hard']) for row in rows.values()) / 1400,
                'f1': sum(row['score']['soft'] for row in rows.values()) / 1400,
                'freeze_audit': r.read(root / STAGE / 'freeze_audit.json'), 'post_test_development': True}
            r.write_json(root / 'completion_report.json', result)
            r.write_json(HERE / model / 'completion_report.json', result)
            r.write_json(root / 'supervisor.json', {'state': 'completed', 'docker_recoveries': recovery, 'at': r.utc()})
            return result
        except BaseException as exc:
            r.write_json(root / 'supervisor.json', {'state': 'stopped', 'docker_recoveries': recovery,
                'error': {'type': type(exc).__name__, 'message': str(exc)}, 'at': r.utc()})
            raise


def start(model):
    root = ROOT / model
    pid = root / 'pid'
    if pid.exists() and Path('/proc', pid.read_text().strip()).exists():
        raise RuntimeError('Process already running')
    assert r.read(root / 'supervisor.json')['state'] == 'prepared'
    with (root / 'run.log').open('ab', buffering=0) as log:
        process = subprocess.Popen([sys.executable, '-B', str(Path(__file__)), 'run', '--model', model],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    pid.write_text(str(process.pid))
    return {'model': model, 'pid': process.pid, 'run': str(root)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['setup', 'prepare', 'start', 'run', 'status'])
    parser.add_argument('--model', choices=list(MODELS))
    args = parser.parse_args()
    if args.action == 'setup':
        result = setup()
    elif args.action == 'status' and not args.model:
        result = {name: status(name) for name in MODELS}
    else:
        assert args.model
        result = {'prepare': prepare, 'start': start, 'run': run, 'status': status}[args.action](args.model)
    print(json.dumps(result, ensure_ascii=False))
