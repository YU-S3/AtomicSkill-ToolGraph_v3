"""Independent Kimi/GLM evaluations of the unchanged V7 Bank and Test1400."""
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
PRIOR = HERE.parent / 'searchqa_glm53_refusal_resume_20261020'
sys.path.insert(0, str(PRIOR))
import campaign as c
import provider_outcomes
from atomic_skillgraph.empirical.budget_governor import billing_blocked

r = c.r
c.HERE = HERE
ROOT = Path('/home/yangchengyu/searchqa_v7_crossmodel_test1400_seed42_20261021')
REFERENCE = Path('/home/yangchengyu/searchqa_v2_v7_fulltest1400_seed42_20261020/fulltest1400')
GLM_PARENT = Path('/home/yangchengyu/searchqa_glm53_v2_v7_test1400_seed42_refusal_resume_20261020')
KIMI_CONFIG = Path('/home/yangchengyu/livemath_predicate_discovery_kimi_seed42_20261020/config.json')
STAGE, ARM = 'fulltest1400', 'v7_text_compiler'
MODELS = ('kimi_k26', 'glm53_flash')
ALLOCATION = {'tokens': 8000000, 'http': 3000}


def select(model):
    assert model in MODELS
    c.CAMPAIGN = ROOT / model
    return c.CAMPAIGN


def config(model):
    cfg = r.read(PRIOR / 'base_config.json')
    if model == 'kimi_k26':
        configured = r.read(KIMI_CONFIG)
        cfg['llm'].update(deepcopy(configured['llm']))
        # Kimi has no effort control; remove inherited GLM settings at every level.
        def remove_effort(value):
            if isinstance(value, dict):
                value.pop('reasoning_effort', None)
                for child in value.values():
                    remove_effort(child)
        remove_effort(cfg['llm'])
        cfg['llm']['capability_profile'] = deepcopy(configured['llm']['capability_profile'])
        cfg['llm']['purpose_protocols'] = {
            purpose: {'thinking_type': 'enabled'} for purpose in ('runtime', 'finish', 'learning')}
        limits = configured['effect_p7']['runtime']
        assert limits['completion'] == 65536 and limits['parent_tokens'] == 128000
        assert limits['finish_completion'] == 16384 and limits['timeout'] == 600
    cfg['llm'].update(display_name=cfg['llm']['model'], max_retries=0)
    cfg['experiment']['authored_bank_runtime'] = 'human.searchqa.accuracy.v7.text_compiler'
    return r.validate_config(cfg)


def specs():
    assert not ROOT.exists(), 'Preserve existing runs'
    source = r.code_identity()
    assert source['git_sha'] == 'a4b96eedda9dc920afe4b7105b9e37f93cc931e8'
    assert not source['tracked_dirty']
    reference = r.read(REFERENCE / 'plan.json')
    assert r.read(REFERENCE / 'result.json')['status'] == 'completed'
    assert r.digest(reference) == r.read(REFERENCE / 'plan_identity.json')['sha256']
    assert r.tree_identity(REFERENCE / ARM / 'input') == reference['input_identity'][ARM]
    assert r.tree_identity(REFERENCE / ARM / 'frozen_bank') == reference['frozen'][ARM]
    shutil.copyfile(PRIOR / 'provider_outcomes.py', HERE / 'provider_outcomes.py')
    r.write_json(HERE / 'campaign_plan.json', {
        'source': source, 'allocation_per_model': ALLOCATION,
        'authorization': '启动同样的bank做kimi-k2.6和GLM-5.3-flash的各一遍完整test',
        'bank': str(REFERENCE / ARM), 'bank_unchanged': True,
        'source_note': 'Reuse tested provider-compatible SearchQA release; no method or source edits.',
        'glm_is_new_independent_repetition': True})
    for model in MODELS:
        cfg = config(model)
        r.write_json(HERE / (model + '_config.json'), cfg)
        r.write_json(HERE / (model + '_spec.json'), {
            'tasks': str(r.DATASETS / 'test.json'), 'base_config': str(HERE / (model + '_config.json')),
            'arms': {ARM: {'package': str(REFERENCE / ARM / 'input'), 'kind': 'native',
                           'frozen_source': str(REFERENCE / ARM / 'frozen_bank')}},
            'purpose': 'One fresh full canonical Test1400 with configured ' + cfg['llm']['model'] +
                       '; identical V7 Bank, prompt, public context, workflow and official scorer. No training or outcome-based edits.'})
    print(json.dumps({'specs_ready': True, 'source': source}))


def prepare(model):
    root = select(model)
    assert not root.exists(), 'Never overwrite an existing evaluation'
    receipt = r.read(HERE / 'offline_acceptance.json')
    assert receipt['passed'] and receipt['driver_sha256'] == r.sha(c.__file__)
    assert receipt['launcher_sha256'] == r.sha(__file__)
    if model == 'glm53_flash':
        assert r.read(GLM_PARENT / STAGE / 'result.json')['status'] == 'completed'
        inherited = r.read(GLM_PARENT / 'budget.json')
        assert not billing_blocked(inherited)
        parent = {'path': str(GLM_PARENT), 'budget_sha256': r.sha(GLM_PARENT / 'budget.json'),
                  'limits': inherited['limits']}
    else:
        inherited = {'attempts': {}, 'unknown_billing': False}
        parent = None
    charge = sum(a.get('accounted_tokens', a['reserved_tokens']) for a in inherited['attempts'].values())
    limits = {'token_limit': charge + ALLOCATION['tokens'],
              'request_limit': len(inherited['attempts']) + ALLOCATION['http'], 'finish_reserve': 0}
    if parent:
        assert limits['token_limit'] <= parent['limits']['token_limit']
        assert limits['request_limit'] <= parent['limits']['request_limit']
    inherited['limits'] = limits
    for name in ('budget.json', 'inherited_budget.json'):
        r.write_json(root / name, inherited)
    r.write_json(root / 'authorization.json', {
        'status': 'approved', 'at': r.utc(), 'allocation': ALLOCATION,
        'user_instruction': r.read(HERE / 'campaign_plan.json')['authorization'],
        'scope': 'One full fresh Test1400, unchanged V7, configured model; no automatic budget expansion.'})
    r.write_json(root / 'cost_carryforward_manifest.json', {
        'parent': parent, 'historical_charged_tokens': charge,
        'historical_http': len(inherited['attempts']), 'allocation': ALLOCATION, 'limits': limits,
        'accounting': 'Separate model ledger. GLM inherits its completed SearchQA lineage once, including unknown-usage reservations. Kimi starts its first SearchQA ledger; unrelated LiveMath/DocVQA ledgers remain separate and unchanged.'})
    c.prepare(STAGE, HERE / (model + '_spec.json'))
    stage = root / STAGE
    plan = r.read(stage / 'plan.json')
    cfg = r.read(stage / ARM / 'config.json')
    refcfg = r.read(REFERENCE / ARM / 'config.json')
    assert cfg['runtime'] == refcfg['runtime'] and cfg['program_environment'] == refcfg['program_environment']
    reference = r.read(REFERENCE / 'plan.json')
    assert plan['input_identity'][ARM] == reference['input_identity'][ARM]
    assert plan['frozen'][ARM] == reference['frozen'][ARM]
    assert r.read(stage / 'tasks.json') == r.read(REFERENCE / 'tasks.json')
    canonical = Path(cfg['manifest'])
    r.verify(canonical.parent.parent)
    public = r.read(r.DATASETS / 'test.json')
    assert len(public['tasks']) == len({t['task_id'] for t in public['tasks']}) == 1400
    assert [t['task_id'] for t in public['tasks']] == [t['task_id'] for t in r.read(canonical)['tasks']]
    assert public['split_sha256'] == r.sha(canonical)
    assert public['authority_sha256'] == r.sha(canonical.parent.parent / 'manifest.json')
    locked = [Path(__file__), HERE / 'check_models.py', HERE / 'offline_acceptance.json',
              HERE / (model + '_config.json'), Path(provider_outcomes.__file__),
              PRIOR / 'source_provenance.json', REFERENCE / 'plan.json', canonical,
              canonical.parent.parent / 'manifest.json', r.DATASETS / 'test.json']
    if parent:
        locked.append(GLM_PARENT / 'budget.json')
    else:
        locked.append(KIMI_CONFIG)
    plan['files'].update({str(p): r.sha(p) for p in locked})
    plan.update(canonical_order=True, fresh_independent_run=True, reference=str(REFERENCE / ARM),
                source_provenance=r.read(PRIOR / 'source_provenance.json'), generation_seed_supported=False)
    r.write_json(stage / 'plan.json', plan)
    r.write_json(stage / 'plan_identity.json', {'sha256': r.digest(plan)})
    checked = c.check(STAGE)
    r.write_json(root / 'preflight.json', {**checked, 'at': r.utc(), 'real_http': 0,
        'same_v7_bank': True, 'same_public_input_order': True, 'same_runtime_and_scorer': True,
        'no_completed_tasks_inherited': not (stage / ARM / 'test').exists(), 'config': cfg['llm']})
    return checked


def start(model, resume=False):
    root = select(model)
    with (root / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        c.check(STAGE)
        if (root / 'pid').exists():
            proc = Path('/proc') / (root / 'pid').read_text().strip() / 'cmdline'
            assert not proc.exists() or str(Path(__file__)) not in proc.read_text(), 'Already running'
            assert resume, 'Explicit resume required'
        assert not (root / STAGE / 'result.json').exists() or resume
        assert not (root / 'STOP_AFTER_TASK').exists() and not (root / STAGE / 'STOP_AFTER_TASK').exists()
        cmd = [sys.executable, '-B', '-u', str(Path(__file__)), 'run', '--model', model]
        with (root / 'run.log').open('ab') as log:
            child = subprocess.Popen(cmd, cwd=r.SOURCE, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (root / 'pid').write_text(str(child.pid) + '\n')
        launch = {'pid': child.pid, 'at': r.utc(), 'command': cmd, 'resume': resume,
                  'plan_sha256': r.read(root / STAGE / 'plan_identity.json')['sha256']}
        r.write_json(root / ('resume_launch.json' if resume else 'launch.json'), launch)
        time.sleep(2)
        assert child.poll() is None, 'Inspect run.log'
        return launch


def monitor(model):
    root = select(model)
    value = c.status(STAGE)
    pid = (root / 'pid').read_text().strip() if (root / 'pid').exists() else None
    proc = Path('/proc') / str(pid) / 'cmdline'
    value.update(model=model, pid=pid, process_alive=bool(pid and proc.exists() and str(Path(__file__)) in proc.read_text()))
    current = root / STAGE / 'current.json'
    result = root / STAGE / 'result.json'
    value['current'] = r.read(current) if current.exists() else None
    value['error'] = r.read(result).get('error') if result.exists() else None
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('specs', 'prepare', 'check', 'start', 'run', 'status'))
    parser.add_argument('--model', choices=(*MODELS, 'all'), default='all')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.action == 'specs':
        specs()
    else:
        assert args.action != 'run' or args.model != 'all', 'Run each model in its own process'
        for model in MODELS if args.model == 'all' else (args.model,):
            select(model)
            result = (prepare(model) if args.action == 'prepare' else start(model, args.resume)
                      if args.action == 'start' else monitor(model) if args.action == 'status'
                      else c.check(STAGE) if args.action == 'check' else c.execute(STAGE))
            print(json.dumps({'model': model, 'result': result}, ensure_ascii=False), flush=True)
