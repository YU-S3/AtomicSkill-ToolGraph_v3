"""Replace the fixed Docker-affected 200 tasks, then finish the canonical test."""
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
sys.path.insert(0, str(HERE))
import campaign as c
from atomic_skillgraph.empirical.budget_governor import reserved_recovery, billing_blocked
from infrastructure_guard import VERSION, docker_ready

r = c.r
FIRST = Path('/home/yangchengyu/searchqa_v7_qwen_minimax_test1400_seed42_20261022/minimax_m3')
PARENT = Path('/home/yangchengyu/searchqa_v7_minimax_test1400_resume_20261023')
ROOT = Path('/home/yangchengyu/searchqa_v7_minimax_docker_replacement_20261023')
OLD_STAGE, ARM = 'fulltest1400', 'v7_text_compiler'
STAGES = ('replacement200', 'remaining303')
c.CAMPAIGN = ROOT
AUTHORIZATION = '用户要求：旧成本不计，重跑Docker故障的200题；此前已授权完成MiniMax完整Test1400。'


def parent_rows():
    a, b = c.completed(FIRST / OLD_STAGE, ARM), c.completed(PARENT / OLD_STAGE, ARM)
    assert not set(a) & set(b)
    return {**a, **b}


def metrics(rows):
    values = list(rows.values())
    return {'tasks': len(values), 'correct': sum(bool(v['score']['hard']) for v in values),
            'em': sum(bool(v['score']['hard']) for v in values) / len(values) if values else None,
            'f1': sum(v['score']['soft'] for v in values) / len(values) if values else None,
            'tokens': sum(u['total_tokens'] for v in values for u in v['usage']),
            'http': sum(len(q.get('http_attempts', [])) for v in values for q in v['requests'])}


def recover_ledger(ledger, attempt):
    value = deepcopy(ledger)
    key = attempt['request_id']
    old_allowed = reserved_recovery(value)
    unresolved = {k for k,v in value['attempts'].items() if v['status'] != 'metered'} - old_allowed
    assert unresolved == {key}
    entry = value['attempts'][key]
    stream = attempt.get('response_diagnostic', {}).get('stream', {})
    assert attempt['outcome'] == 'error' and attempt.get('raw_usage') is None
    assert stream.get('completed') is False and stream.get('content_chars') == 0
    assert stream.get('response_model') == 'MiniMax-M3' and stream.get('events', 0) > 0
    assert entry['status'] == 'unknown_billing' and 'accounted_tokens' not in entry
    assert entry['context'] == attempt['request_context']
    receipt = value['inherited_reservation_recovery']
    receipt['authorization'] = {'prior': receipt['authorization'], 'current': AUTHORIZATION,
        'scope': 'One new execution of this interrupted, unscored task; no blanket future retry.'}
    receipt['attempts'][key] = {'entry_hash': r.digest(entry), 'reserved_tokens': entry['reserved_tokens']}
    assert reserved_recovery(value) == old_allowed | {key}
    assert not billing_blocked(value)
    assert value['attempts'] == ledger['attempts'] and value['limits'] == ledger['limits']
    return value


def specs():
    assert not ROOT.exists()
    tasks = r.read(FIRST / OLD_STAGE / 'tasks.json')['tasks']
    rows = parent_rows()
    assert len(rows) == 1097 and set(rows) == {t['task_id'] for t in tasks[:1097]}
    replaced = tasks[517:717]
    excluded = {t['task_id'] for t in replaced}
    failed_program_ids = {tid for tid,row in rows.items()
        if any(a['status'] == 'execution_error' for a in row['execution'].get('attempts', []))}
    assert failed_program_ids == excluded and len(excluded) == 200
    diagnostics = []
    for task in replaced:
        row = rows[task['task_id']]
        found = []
        for q in row['requests']:
            for message in q['messages']:
                if message['role'] != 'user': continue
                try: material = json.loads(message['content'])
                except (TypeError, ValueError): continue
                for item in material.get('memory', {}).get('recent_results', []):
                    text = item.get('diagnostic', '')
                    if 'Cannot connect to the Docker daemon' in text: found.append(text)
        assert found, 'Every excluded task must have actual Docker evidence'
        diagnostics.append({'task_id': task['task_id'], 'physical_key': task['physical_key'],
                            'diagnostic': found[0], 'old_result_hash': r.digest(row)})
    kept = {k:v for k,v in rows.items() if k not in excluded}
    path = PARENT / OLD_STAGE / ARM / 'test/requests' / (tasks[1097]['task_id'] + '_attempt1.json')
    audit = r.read(path)
    attempts = [a for q in audit['requests'] for a in q.get('http_attempts', [])]
    assert len(attempts) == 1 and not audit['usage']
    ledger = r.read(PARENT / 'budget.json')
    recover_ledger(ledger, attempts[0])
    for p in (FIRST, PARENT):
        assert r.read(p / OLD_STAGE / 'result.json')['status'] == 'stopped'
        oldplan = r.read(p / OLD_STAGE / 'plan.json')
        assert oldplan['code'] == r.code_identity()
        assert r.tree_identity(p / OLD_STAGE / ARM / 'input') == oldplan['input_identity'][ARM]
        assert r.tree_identity(p / OLD_STAGE / ARM / 'frozen_bank') == oldplan['frozen'][ARM]
    r.write_json(HERE / 'replacement_manifest.json', {
        'reason': 'User-authorized replacement of all and only Docker-affected tasks, independent of scores.',
        'canonical_range_one_based': [518, 717], 'excluded': diagnostics,
        'old_excluded': metrics({k:rows[k] for k in excluded}), 'kept': metrics(kept),
        'historical': metrics(rows), 'parents': [str(FIRST), str(PARENT)],
        'parent_rows_hash': r.digest(rows), 'parent_budget_sha256': r.sha(PARENT / 'budget.json'),
        'first_budget_sha256': r.sha(FIRST / 'budget.json'),
        'failed_audit_path': str(path), 'failed_audit_sha256': r.sha(path), 'failed_attempt': attempts[0],
        'qwen_result_sha256': r.sha(FIRST.parent / 'qwen38_flash' / OLD_STAGE / 'result.json'),
        'accounting': 'Exclude the old 200 rows and all their 1659185 measured tokens / 439 HTTP from effective results. Preserve physical historical ledger and reservations unchanged.'})
    r.write_json(HERE / 'base_config.json', r.read(PARENT / OLD_STAGE / ARM / 'config.json'))
    r.write_json(HERE / 'campaign_plan.json', {'source': r.code_identity(), 'authorization': AUTHORIZATION,
        'guard_version': VERSION, 'method_bank_model_unchanged': True, 'cumulative_limits': ledger['limits']})
    for stage, items in zip(STAGES, [replaced, tasks[1097:]]):
        r.write_json(HERE / (stage + '_tasks.json'), {'tasks': items})
        r.write_json(HERE / (stage + '_spec.json'), {
            'tasks': str(HERE / (stage + '_tasks.json')), 'base_config': str(HERE / 'base_config.json'),
            'arms': {ARM: {'kind': 'native', 'package': str(PARENT / OLD_STAGE / ARM / 'input'),
                          'frozen_source': str(PARENT / OLD_STAGE / ARM / 'frozen_bank')}},
            'purpose': 'Authorized ' + stage + ': same frozen V7/model/input/scorer. '
                       'Stop on Docker availability failure before model repair; separate excluded old costs from physical billing.'})
    return {'replacement': 200, 'remaining': 303, 'kept': metrics(kept),
            'excluded_cost': metrics({k:rows[k] for k in excluded})}


def prepare():
    assert not ROOT.exists()
    receipt = r.read(HERE / 'offline_acceptance.json')
    assert receipt['passed'] and receipt['launcher_sha256'] == r.sha(__file__)
    assert receipt['driver_sha256'] == r.sha(c.__file__)
    manifest = r.read(HERE / 'replacement_manifest.json')
    assert r.digest(parent_rows()) == manifest['parent_rows_hash']
    assert r.sha(PARENT / 'budget.json') == manifest['parent_budget_sha256']
    ledger = r.read(PARENT / 'budget.json')
    r.write_json(ROOT / 'inherited_budget.json', ledger)
    r.write_json(ROOT / 'budget.json', recover_ledger(ledger, manifest['failed_attempt']))
    r.write_json(ROOT / 'authorization.json', {'status': 'approved', 'at': r.utc(), 'user': AUTHORIZATION,
        'limits': ledger['limits'], 'scope': 'Replace fixed 200; finish remaining 303; no automatic additional retries.'})
    r.write_json(ROOT / 'cost_carryforward_manifest.json', {
        'parent': str(PARENT), 'parent_budget_sha256': manifest['parent_budget_sha256'],
        'prior_searchqa_tokens': sum(v.get('accounted_tokens',0) for v in ledger['attempts'].values()),
        'prior_searchqa_http': len(ledger['attempts']), 'limits': ledger['limits'],
        'effective_report_exclusion': manifest['old_excluded'], 'no_limit_increase': True,
        'unknown_reserved_tokens': sum(v['reserved_tokens'] for v in ledger['attempts'].values() if v['status']!='metered')})
    for stage, offset in zip(STAGES, [517, 1097]):
        c.prepare(stage, HERE / (stage + '_spec.json'))
        path = ROOT / stage
        plan = r.read(path / 'plan.json')
        oldplan = r.read(PARENT / OLD_STAGE / 'plan.json')
        assert plan['frozen'] == oldplan['frozen'] and plan['input_identity'] == oldplan['input_identity']
        cfg,old = r.read(path / ARM / 'config.json'),r.read(PARENT / OLD_STAGE / ARM / 'config.json')
        assert all(cfg[k] == old[k] for k in ('llm','runtime','budget','program_environment'))
        plan.update(order_offset=offset, guard=VERSION)
        locked = [Path(__file__), HERE/'infrastructure_guard.py', HERE/'offline_acceptance.json',
                  HERE/'check_replacement.py', HERE/'replacement_manifest.json',
                  PARENT/'budget.json', FIRST/'budget.json', Path(manifest['failed_audit_path']),
                  FIRST.parent/'qwen38_flash'/OLD_STAGE/'result.json']
        plan['files'].update({str(p):r.sha(p) for p in locked})
        r.write_json(path/'plan.json',plan)
        r.write_json(path/'plan_identity.json',{'sha256':r.digest(plan)})
        c.check(stage)
    return {'passed':True,'stages':list(STAGES),'source':r.code_identity()}


def status():
    manifest = r.read(HERE/'replacement_manifest.json')
    excluded = {v['task_id'] for v in manifest['excluded']}
    combined = {k:v for k,v in parent_rows().items() if k not in excluded}
    stages = {}
    for stage in STAGES:
        rows = c.completed(ROOT/stage,ARM)
        assert not set(rows) & set(combined)
        combined.update(rows)
        p=ROOT/stage/'result.json'
        stages[stage]={'metrics':metrics(rows),'status':r.read(p)['status'] if p.exists() else 'prepared',
                       'error':r.read(p).get('error') if p.exists() else None}
    ledger=r.read(ROOT/'budget.json')
    return {'at':r.utc(),'stages':stages,'effective':metrics(combined),'expected_tasks':1400,
        'excluded_old_cost':manifest['old_excluded'],
        'physical_metered_tokens':sum(v.get('accounted_tokens',0) for v in ledger['attempts'].values()),
        'physical_http':len(ledger['attempts']),
        'unknown_reserved_tokens':sum(v['reserved_tokens'] for v in ledger['attempts'].values() if v['status']=='unknown_billing'),
        'alive':alive(ROOT)}


def alive(root):
    pid=(root/'pid').read_text().strip() if (root/'pid').exists() else ''
    p=Path('/proc')/pid/'cmdline'
    return bool(pid and p.exists() and p.read_bytes())


def execute():
    for stage in STAGES:
        c.execute(stage)
    final=status()
    assert final['effective']['tasks']==1400
    r.write_json(ROOT/'effective_summary.json',final)
    return final


def start():
    with (ROOT/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not alive(ROOT) and not alive(PARENT)
        assert not (ROOT/'pid').exists()
        for stage in STAGES:c.check(stage)
        docker_ready(r.read(HERE/'base_config.json')['program_environment']['image_digest'])
        cmd=[sys.executable,'-B','-u',str(Path(__file__)),'run']
        with (ROOT/'run.log').open('ab') as log:
            process=subprocess.Popen(cmd,cwd=r.SOURCE,stdin=subprocess.DEVNULL,stdout=log,
                                     stderr=subprocess.STDOUT,start_new_session=True)
        (ROOT/'pid').write_text(str(process.pid)+'\n')
        launch={'pid':process.pid,'at':r.utc(),'command':cmd}
        r.write_json(ROOT/'launch.json',launch)
        time.sleep(2)
        assert process.poll() is None,'Inspect run.log'
        return launch


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('specs','prepare','start','run','status'))
    args=parser.parse_args()
    value={'specs':specs,'prepare':prepare,'start':start,'run':execute,'status':status}[args.action]()
    print(json.dumps(value,ensure_ascii=False),flush=True)
