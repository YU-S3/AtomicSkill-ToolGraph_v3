"""Independent full Test1400 with the unchanged MiniMax R2 Bank and explicit carryforward."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
import campaign as c

r = c.r
ROOT = Path('/home/yangchengyu/searchqa_minimax_r2_test1400_seed42_20261023')
PARENT = Path('/home/yangchengyu/searchqa_minimax_accuracy_r2_20261023')
STAGE = 'fulltest1400'
c.CAMPAIGN = ROOT
NEW_TOKENS = 6_000_000
NEW_HTTP = 1600


def prepare():
    assert not ROOT.exists(), 'Preserve existing runs; use run to resume'
    receipt = r.read(HERE / 'offline_acceptance.json')
    assert receipt['passed'] and receipt['real_http'] == 0
    assert receipt['driver_sha256'] == r.sha(c.__file__)
    assert receipt['controller_sha256'] == r.sha(__file__)
    assert receipt['source'] == r.code_identity()
    assert receipt['package'] == r.tree_identity(HERE / 'packages/minimax_r2')
    c.CAMPAIGN = PARENT
    try:
        c.check('followup96')
    finally:
        c.CAMPAIGN = ROOT
    assert r.read(PARENT / 'followup96/result.json')['status'] == 'completed'
    original = r.read(PARENT / 'budget.json')
    measured = sum(a.get('accounted_tokens', 0) for a in original['attempts'].values())
    charged = sum(a.get('accounted_tokens', a['reserved_tokens']) for a in original['attempts'].values())
    assert measured == 7666801 and charged == 7821713 and len(original['attempts']) == 2195
    from atomic_skillgraph.empirical.budget_governor import billing_blocked
    assert not billing_blocked(original)
    ledger = deepcopy(original)
    ledger['limits'].update(token_limit=charged + NEW_TOKENS,
                            request_limit=len(original['attempts']) + NEW_HTTP)
    r.write_json(ROOT / 'parent_budget.json', original)
    for name in ('budget.json', 'inherited_budget.json'):
        r.write_json(ROOT / name, ledger)
    r.write_json(ROOT / 'authorization.json', {
        'status': 'approved', 'at': r.utc(), 'user_request': '能否启动test？可以就启动',
        'context': 'User requested a full test after being informed the old remaining allowance cannot cover Test1400.',
        'scope': 'One independent MiniMax-M3 R2 canonical Test1400; prior paid results retained.',
        'operator_set_stop_limits': {'new_tokens': NEW_TOKENS, 'new_http': NEW_HTTP},
        'limits': ledger['limits'], 'automatic_budget_expansion': False})
    r.write_json(ROOT / 'cost_carryforward_manifest.json', {
        'parent': str(PARENT), 'parent_budget_sha256': r.sha(PARENT / 'budget.json'),
        'prior_searchqa_tokens': measured, 'prior_searchqa_http': len(original['attempts']),
        'unknown_reserved_tokens': charged - measured, 'old_entries_preserved': True,
        'old_excluded_200_effective_cost_stays_excluded': True,
        'prior_limits': original['limits'], 'new_limits': ledger['limits'],
        'incremental_token_limit': NEW_TOKENS, 'incremental_http_limit': NEW_HTTP,
        'prior_task_results_reused': False})
    tasks = r.read(r.DATASETS / 'test.json')['tasks']
    assert len(tasks) == len({t['task_id'] for t in tasks}) == 1400
    spec = {'tasks': str(r.DATASETS / 'test.json'), 'base_config': str(HERE / 'base_config.json'),
            'arms': {'r2': {'kind': 'native', 'package': str(HERE / 'packages/minimax_r2'),
                            'frozen_source': str(PARENT / 'followup96/r2/frozen_bank')}},
            'purpose': 'Complete canonical Test1400 in its original order, seed42. Unchanged R2 Bank and MiniMax settings from followup96. Human-authored post-Test development; not blind generalization. No task-result reuse or automatic subsequent runs.'}
    r.write_json(HERE / 'spec.json', spec)
    c.prepare(STAGE, HERE / 'spec.json')
    cfgpath = ROOT / STAGE / 'r2/config.json'
    cfg = r.read(cfgpath)
    cfg['experiment']['authored_bank_runtime'] = r.read(ROOT / STAGE / 'r2/frozen_bank/imported_bank.json')['method_version']
    r.write_json(cfgpath, cfg)
    planpath = ROOT / STAGE / 'plan.json'
    plan = r.read(planpath)
    for p in (cfgpath, Path(__file__), HERE / 'infrastructure_guard.py', HERE / 'offline_acceptance.json',
              HERE / 'check_offline.py', PARENT / 'budget.json', PARENT / 'followup96/plan.json'):
        plan['files'][str(p)] = r.sha(p)
    parent_plan = r.read(PARENT / 'followup96/plan.json')
    assert plan['frozen']['r2'] == parent_plan['frozen']['r2']
    assert plan['input_identity']['r2'] == parent_plan['input_identity']['r2']
    r.write_json(planpath, plan)
    r.write_json(ROOT / STAGE / 'plan_identity.json', {'sha256': r.digest(plan)})
    result = c.check(STAGE)
    r.write_json(HERE / 'preflight.json', {**result, 'at': r.utc(), 'source': plan['code'],
        'same_bank_as_followup96': True, 'canonical_tasks_sha256': r.sha(ROOT / STAGE / 'tasks.json'),
        'bank_input_sha256': plan['input_identity']['r2']['sha256'],
        'frozen_sha256': plan['frozen']['r2']['sha256'], 'new_tokens': NEW_TOKENS, 'new_http': NEW_HTTP,
        'cumulative_limits': ledger['limits'], 'parent_physical_tokens': measured,
        'retained_unknown_reservation': charged - measured})
    return result


def status():
    s = c.status(STAGE)
    ledger, inherited = r.read(ROOT / 'budget.json'), r.read(ROOT / 'inherited_budget.json')
    added = [a for k, a in ledger['attempts'].items() if k not in inherited['attempts']]
    s.update(new_tokens=sum(a.get('accounted_tokens', 0) for a in added), new_http=len(added),
             new_unmetered_states=[a['status'] for a in added if a['status'] != 'metered'])
    pidfile = ROOT / 'pid'
    s['alive'] = pidfile.exists() and Path('/proc', pidfile.read_text().strip()).exists()
    for name in ('current', 'result'):
        path = ROOT / STAGE / (name + '.json')
        if path.exists():
            s[name] = r.read(path) if name == 'current' else r.read(path).get('error')
    return s


def run():
    # The shared execution lock in campaign.execute prevents duplicate paid runs.
    result = c.execute(STAGE)
    rows = c.completed(ROOT / STAGE, 'r2')
    assert len(rows) == 1400
    ledger, inherited = r.read(ROOT / 'budget.json'), r.read(ROOT / 'inherited_budget.json')
    added = [a for k, a in ledger['attempts'].items() if k not in inherited['attempts']]
    assert all(a['status'] == 'metered' for a in added)
    tokens = sum(u['total_tokens'] for row in rows.values() for u in row['usage'])
    assert tokens == sum(a['accounted_tokens'] for a in added)
    out = {**result, 'em': sum(bool(v['score']['hard']) for v in rows.values()) / 1400,
           'f1': sum(v['score']['soft'] for v in rows.values()) / 1400,
           'new_tokens': tokens, 'new_http': len(added), 'at': r.utc(),
           'post_test_development': True, 'freeze_audit': r.read(ROOT / STAGE / 'freeze_audit.json')}
    r.write_json(ROOT / 'completion_report.json', out)
    r.write_json(HERE / 'completion_report.json', out)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'check', 'status', 'run'])
    action = parser.parse_args().action
    value = {'prepare': prepare, 'check': lambda: c.check(STAGE), 'status': status, 'run': run}[action]()
    print(json.dumps(value, ensure_ascii=False))
