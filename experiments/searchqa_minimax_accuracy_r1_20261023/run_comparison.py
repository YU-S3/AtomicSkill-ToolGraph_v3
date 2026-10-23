"""Fixed uniform paired comparison, using remaining MiniMax cumulative allowance."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import campaign as c

r = c.r
ROOT = Path('/home/yangchengyu/searchqa_minimax_accuracy_r1_20261023')
PARENT = Path('/home/yangchengyu/searchqa_v7_minimax_docker_replacement_20261023')
BASELINE = PARENT / 'remaining303/v7_text_compiler'
STAGE = 'paired128'
c.CAMPAIGN = ROOT
AUTHORIZATION = 'User requested MiniMax accuracy optimization, including specific skills and sourced fact cards; earlier paid Bank iteration authorization remains in effect. This comparison uses only the existing MiniMax cumulative allowance.'


def arms():
    return {'v7': {'kind': 'native', 'package': str(BASELINE / 'input'), 'frozen_source': str(BASELINE / 'frozen_bank')},
            'r1': {'kind': 'native', 'package': str(HERE / 'packages/minimax_r1')}}


def specs():
    assert not (HERE / 'spec.json').exists()
    tasks = r.read(r.DATASETS / 'test.json')['tasks']
    selected = set(random.Random(20261023).sample(range(len(tasks)), 128))
    r.write_json(HERE / 'tasks128_public.json', {'tasks': [t for i,t in enumerate(tasks) if i in selected]})
    r.write_json(HERE / 'selection.json', {'seed': 20261023, 'selection': 'uniform without replacement; no correctness filter; canonical order',
        'indices_zero_based': sorted(selected), 'input_sha256': r.sha(r.DATASETS / 'test.json'),
        'post_test_development': True, 'normal_http': 256, 'new_http_limit': 300,
        'selection_rule_frozen_before_paid_comparison': True,
        'decision': 'Report EM/F1, paired gains/losses, tokens and failures; no automatic Test1400 or automatic promotion.'})
    r.write_json(HERE / 'campaign_plan.json', {'source': r.code_identity(), 'authorization': AUTHORIZATION})
    r.write_json(HERE / 'spec.json', {'tasks': str(HERE / 'tasks128_public.json'), 'base_config': str(HERE / 'base_config.json'),
        'arms': arms(), 'purpose': 'Post-Test development: fixed uniform 128 questions, contemporary V7/R1 paired comparison with identical MiniMax model/public inputs/scorer. No automatic full Test.'})
    return {'tasks': 128, 'arms': list(arms()), 'source': r.code_identity()}


def prepare():
    assert not ROOT.exists()
    check = r.read(HERE / 'offline_acceptance.json')
    assert check['passed'] and check['controller_sha256'] == r.sha(__file__)
    assert check['driver_sha256'] == r.sha(c.__file__)
    ledger = r.read(PARENT / 'budget.json')
    measured = sum(x.get('accounted_tokens',0) for x in ledger['attempts'].values())
    reserved = sum(x['reserved_tokens'] for x in ledger['attempts'].values() if x['status'] != 'metered')
    assert measured == 6494941 and reserved == 154912 and len(ledger['attempts']) == 1842
    r.write_json(ROOT / 'parent_budget.json', ledger)
    ledger = deepcopy(ledger)
    ledger['limits']['request_limit'] = min(ledger['limits']['request_limit'], len(ledger['attempts']) + 300)
    for name in ['budget.json', 'inherited_budget.json']:
        r.write_json(ROOT / name, ledger)
    r.write_json(ROOT / 'authorization.json', {'status': 'approved', 'user': AUTHORIZATION, 'at': r.utc(),
        'scope': 'One paired128 campaign; no new full Test or increase of prior 8M cumulative token cap', 'limits': ledger['limits']})
    r.write_json(ROOT / 'cost_carryforward_manifest.json', {'parent': str(PARENT), 'parent_budget_sha256': r.sha(PARENT / 'budget.json'),
        'prior_searchqa_tokens': measured, 'prior_searchqa_http': len(ledger['attempts']),
        'unknown_reserved_tokens': reserved, 'token_headroom_at_start': ledger['limits']['token_limit'] - measured - reserved,
        'old_excluded_200_effective_cost_stays_excluded': True, 'physical_ledger_entries_preserved': True})
    c.prepare(STAGE, HERE / 'spec.json')
    plan_path = ROOT / STAGE / 'plan.json'
    plan = r.read(plan_path)
    for arm in arms():
        path = ROOT / STAGE / arm / 'config.json'
        config = r.read(path)
        config['experiment']['authored_bank_runtime'] = r.read(ROOT / STAGE / arm / 'frozen_bank/imported_bank.json')['method_version']
        r.write_json(path, config)
        plan['files'][str(path)] = r.sha(path)
    for path in [Path(__file__), HERE / 'infrastructure_guard.py', HERE / 'offline_acceptance.json', HERE / 'selection.json', HERE / 'check_offline.py']:
        plan['files'][str(path)] = r.sha(path)
    r.write_json(plan_path, plan)
    r.write_json(ROOT / STAGE / 'plan_identity.json', {'sha256': r.digest(plan)})
    return c.check(STAGE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['specs','prepare','run','status','check'])
    action = parser.parse_args().action
    value = specs() if action == 'specs' else prepare() if action == 'prepare' else {'run': c.execute, 'status': c.status, 'check': c.check}[action](STAGE)
    print(json.dumps(value, ensure_ascii=False))
