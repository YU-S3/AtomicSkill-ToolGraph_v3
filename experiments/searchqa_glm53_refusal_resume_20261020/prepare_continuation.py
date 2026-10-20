"""Continue only unfinished canonical tasks; retain the parent run verbatim."""
from copy import deepcopy
from pathlib import Path
import json
import os
import shutil
import sys
import campaign as c
from provider_outcomes import last_refusal, retain_reservation, POLICY

r, HERE, ROOT = c.r, c.HERE, c.CAMPAIGN
PARENT = Path('/home/yangchengyu/searchqa_glm53_v2_v7_test1400_seed42_20261020')
STAGE = 'fulltest1400'


def make_specs():
    parent_stage = PARENT / STAGE
    old = r.read(parent_stage / 'plan.json')
    assert r.read(parent_stage / 'result.json')['status'] == 'stopped'
    try:
        os.kill(int((parent_stage / 'pid').read_text()), 0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError('Parent process exists')
    assert r.digest(old) == r.read(parent_stage / 'plan_identity.json')['sha256']
    for name, expected in old['files'].items():
        assert r.sha(name) == expected, name
    rows = {a: c.completed(parent_stage, a) for a in old['arms']}
    tasks = r.read(parent_stage / 'tasks.json')['tasks']
    count = len(next(iter(rows.values())))
    assert count == 99 and all(list(v) == [t['task_id'] for t in tasks[:count]] for v in rows.values())
    remaining = tasks[count:]
    r.write_json(HERE / 'remaining_tasks.json', {'tasks': remaining})
    base = r.read(parent_stage / 'v7_text_compiler/config.json')
    r.write_json(HERE / 'base_config.json', base)
    r.write_json(HERE / 'campaign_plan.json', {'source': r.code_identity(),
        'allocation': r.read(PARENT / 'authorization.json')['allocation'], 'policy': POLICY})
    replay = {}
    requests = {}
    for arm in old['arms']:
        assert r.tree_identity(parent_stage / arm / 'input') == old['input_identity'][arm]
        assert r.tree_identity(parent_stage / arm / 'frozen_bank') == old['frozen'][arm]
        for path in (parent_stage / arm / 'test/requests').glob('*.json'):
            audit = r.read(path)
            attempt = last_refusal(audit, remaining[0]['task_id'])
            if attempt:
                assert not audit['usage'], 'Use explicit metered usage transfer if needed'
                replay[arm] = {remaining[0]['task_id']: {'path': str(path), 'audit_digest': r.digest(audit),
                    'source_plan_sha256': r.sha(parent_stage / 'plan.json'), 'no_new_http': True}}
                requests[attempt['request_id']] = attempt
    assert len(requests) == 1
    arms = {a: {'package': str(parent_stage / a / 'input'),
                'frozen_source': str(parent_stage / a / 'frozen_bank'), 'kind': item['kind']}
            for a, item in old['arms'].items()}
    r.write_json(HERE / 'spec.json', {'tasks': str(HERE / 'remaining_tasks.json'),
        'base_config': str(HERE / 'base_config.json'), 'arms': arms, 'replay_sources': replay,
        'purpose': 'Versioned refusal handling continuation, same canonical inputs and immutable Banks; '
                   'parent 99+99 preserved, explicit refusal scored once without resending.'})
    r.write_json(HERE / 'parent_snapshot.json', {
        'root': str(PARENT), 'budget_sha256': r.sha(PARENT / 'budget.json'),
        'parent_completed': {a: {'completed': len(v), 'total': len(v),
            'correct': sum(bool(t['score']['hard']) for t in v.values()),
            'tokens': sum(u['total_tokens'] for t in v.values() for u in t['usage']),
            'native_graphs': sum(bool(t['execution'].get('all_authored_nodes_completed')) for t in v.values()),
            'program_calls': sum(len(t['execution'].get('attempts', [])) for t in v.values()),
            'program_ok': sum(p['status'] == 'ok' for t in v.values() for p in t['execution'].get('attempts', [])),
            'result_digest': r.digest(v)} for a, v in rows.items()},
        'refused_attempts': requests, 'order_offset': count})
    print({'specs': True, 'new_tasks_per_arm': len(remaining), 'source': r.code_identity()})


def prepare():
    assert not ROOT.exists()
    receipt = r.read(HERE / 'offline_acceptance.json')
    assert receipt['passed'] and receipt['driver_sha256'] == r.sha(c.__file__)
    snap = r.read(HERE / 'parent_snapshot.json')
    assert snap['budget_sha256'] == r.sha(PARENT / 'budget.json')
    ledger = r.read(PARENT / 'budget.json')
    for name in ('budget.json', 'inherited_budget.json'):
        r.write_json(ROOT / name, ledger)
    governor = r.BudgetGovernor(ROOT / 'budget.json', **ledger['limits'])
    for attempt in snap['refused_attempts'].values():
        retain_reservation(governor, attempt)
    assert not governor.billing_blocked
    r.write_json(ROOT / 'authorization.json', {**r.read(PARENT / 'authorization.json'),
        'continuation_authorization': '本地发送请求有问题需要修改？别的也需要修改好做续跑',
        'refusal_policy': POLICY, 'at': r.utc()})
    old_prefix = r.read(PARENT / 'inherited_budget.json')
    new_rows = [v for k, v in ledger['attempts'].items() if k not in old_prefix['attempts']]
    carry = {'parent': str(PARENT), 'parent_budget_sha256': snap['budget_sha256'],
        'parent_completed': snap['parent_completed'], 'prior_searchqa_tokens': sum(a.get('accounted_tokens', 0) for a in new_rows),
        'prior_searchqa_http': len(new_rows), 'limits': ledger['limits'], 'no_limit_increase': True,
        'unknown_reservation_preserved': deepcopy(governor.state['inherited_reservation_recovery']),
        'accounting': 'Physical IDs inherited once; 77540 unknown tokens reserved, not actual measured usage.'}
    r.write_json(ROOT / 'cost_carryforward_manifest.json', carry)
    c.prepare(STAGE, HERE / 'spec.json')
    root = ROOT / STAGE
    plan = r.read(root / 'plan.json')
    parent_plan = r.read(PARENT / STAGE / 'plan.json')
    assert plan['input_identity'] == parent_plan['input_identity']
    assert plan['frozen'] == parent_plan['frozen']
    remaining_ids = {t['task_id'] for t in r.read(root / 'tasks.json')['tasks']}
    schedule = [v for v in r.read(PARENT / STAGE / 'schedule.json') if v['task_id'] in remaining_ids]
    r.write_json(root / 'schedule.json', schedule)
    plan['order_offset'] = snap['order_offset']
    for arm in plan['arms']:
        path = root / arm / 'config.json'
        cfg = r.read(path)
        parent_cfg = r.read(PARENT / STAGE / arm / 'config.json')
        cfg['experiment']['authored_bank_runtime'] = parent_cfg['experiment']['authored_bank_runtime']
        r.write_json(path, cfg)
        for field in ('llm', 'runtime', 'budget'):
            assert cfg[field] == parent_cfg[field]
    files = [root / 'schedule.json', *[root / a / 'config.json' for a in plan['arms']]]
    files += [HERE / name for name in ('source_provenance.json', 'offline_acceptance.json', 'parent_snapshot.json',
        'remaining_tasks.json', 'prepare_continuation.py', 'start_stage.py', 'monitor.sh')]
    files += [PARENT / 'budget.json', PARENT / STAGE / 'plan.json']
    for arm, sources in plan['replay_sources'].items():
        files += [Path(v['path']) for v in sources.values()]
    plan['files'].update({str(p): r.sha(p) for p in files})
    r.write_json(root / 'plan.json', plan)
    r.write_json(root / 'plan_identity.json', {'sha256': r.digest(plan)})
    r.write_json(ROOT / 'continuation_manifest.json', {
        'parent': str(PARENT), 'parent_source': parent_plan['code'], 'child_source': plan['code'],
        'parent_plan': r.sha(PARENT / STAGE / 'plan.json'), 'child_plan': r.sha(root / 'plan.json'),
        'completed_prefix': snap['parent_completed'], 'replayed_refusals': plan['replay_sources'],
        'method_and_model_unchanged': True, 'at': r.utc()})
    print(c.check(STAGE))


if __name__ == '__main__':
    make_specs() if sys.argv[1] == 'specs' else prepare()
