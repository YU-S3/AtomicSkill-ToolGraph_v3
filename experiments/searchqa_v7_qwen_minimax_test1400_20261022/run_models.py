"""One independent canonical SearchQA Test1400 per configured model, unchanged V7."""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import campaign as c

r = c.r
ROOT = Path('/home/yangchengyu/searchqa_v7_qwen_minimax_test1400_seed42_20261022')
REFERENCE = Path('/home/yangchengyu/searchqa_v2_v7_fulltest1400_seed42_20261020/fulltest1400')
PRIOR = HERE.parent / 'searchqa_glm53_refusal_resume_20261020'
CONFIGURED = {
    'qwen38_flash': HERE.parent / 'qwen_body_budget_fresh_seed42_20261022/config.json',
    'minimax_m3': HERE.parent / 'minimax_qwen_configuration_20261022/minimax_m3.yaml',
}
STAGE, ARM = 'fulltest1400', 'v7_text_compiler'
MODELS = tuple(CONFIGURED)
ALLOCATION = {'tokens': 8000000, 'http': 3000}


def select(model):
    assert model in MODELS
    c.CAMPAIGN = ROOT / model
    return c.CAMPAIGN


def config(model):
    cfg = r.read(PRIOR / 'base_config.json')
    configured = yaml.safe_load(CONFIGURED[model].read_text())
    limits = configured.get('effect_p7', configured)['runtime']
    assert limits['completion'] == 65536 and limits['finish_completion'] == 16384
    assert limits['timeout'] == 600
    parent = limits['parent_tokens']
    # Rebuild the provider section so no old model's parameters survive.
    cfg['llm'] = {**deepcopy(configured['llm']), 'provider': 'openai_compatible',
        'display_name': configured['llm']['model'], 'input_modalities': ['text'],
        'max_retries': 0, 'protocol': {'thinking_type': 'enabled'},
        'runtime': {'max_completion_tokens': 65536, 'request_timeout_seconds': 600,
                    'max_total_tokens_per_task': parent},
        'purpose_overrides': {
            'finish_only': {'protocol': {'thinking_type': 'enabled'}, 'max_completion_tokens': 16384},
            'learning_structure_repair': {'protocol': {'thinking_type': 'disabled'}, 'max_completion_tokens': 2048}}}
    if model == 'qwen38_flash':
        cfg['llm']['reasoning_effort'] = 'high'
        assert cfg['llm']['thinking_budget'] == 131072 and parent == 400000
    else:
        assert parent == 128000 and 'thinking_budget' not in cfg['llm']
    cfg['budget'].update(eval_task_tokens=parent, train_task_tokens=parent)
    cfg['experiment']['authored_bank_runtime'] = 'human.searchqa.accuracy.v7.text_compiler'
    return r.validate_config(cfg)


def specs():
    assert not ROOT.exists(), 'Preserve existing runs'
    source = r.code_identity()
    provenance = r.read(HERE / 'source_provenance.json')
    assert source['git_sha'] == provenance['source_commit'] and not source['tracked_dirty']
    ref = r.read(REFERENCE / 'plan.json')
    assert r.read(REFERENCE / 'result.json')['status'] == 'completed'
    assert r.digest(ref) == r.read(REFERENCE / 'plan_identity.json')['sha256']
    assert r.tree_identity(REFERENCE / ARM / 'input') == ref['input_identity'][ARM]
    assert r.tree_identity(REFERENCE / ARM / 'frozen_bank') == ref['frozen'][ARM]
    r.load_env('/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env')
    keys = {model: bool(os.getenv(config(model)['llm']['api_key_env'])) for model in MODELS}
    assert all(keys.values()), 'Configured API key is missing'
    r.write_json(HERE / 'campaign_plan.json', {
        'source': source, 'allocation_per_model': ALLOCATION,
        'authorization': '检查配置，用qwen-3.8-flash、minimax-M3加bank分别运行一次test',
        'bank': str(REFERENCE / ARM), 'bank_unchanged': True,
        'configured_keys_present': keys, 'source_provenance': provenance,
        'scope': 'Two fresh independent Test1400 runs; no Train, no Bank edits, no automatic budget expansion.'})
    for model in MODELS:
        cfg = config(model)
        r.write_json(HERE / (model + '_config.json'), cfg)
        r.write_json(HERE / (model + '_spec.json'), {
            'tasks': str(r.DATASETS / 'test.json'), 'base_config': str(HERE / (model + '_config.json')),
            'arms': {ARM: {'package': str(REFERENCE / ARM / 'input'), 'kind': 'native',
                           'frozen_source': str(REFERENCE / ARM / 'frozen_bank')}},
            'purpose': 'One fresh full canonical Test1400 using ' + cfg['llm']['model'] +
                       '; unchanged V7 Bank, prompt, workflow, context, official scorer and canonical order.'})
    return {'specs_ready': True, 'source': source, 'keys_present': keys}


def prepare(model):
    root = select(model)
    assert not root.exists(), 'Never overwrite an existing evaluation'
    receipt = r.read(HERE / 'offline_acceptance.json')
    assert receipt['passed'] and receipt['driver_sha256'] == r.sha(c.__file__)
    assert receipt['launcher_sha256'] == r.sha(__file__)
    limits = {'token_limit': ALLOCATION['tokens'], 'request_limit': ALLOCATION['http'], 'finish_reserve': 0}
    for name in ('budget.json', 'inherited_budget.json'):
        r.write_json(root / name, {'limits': limits, 'attempts': {}, 'unknown_billing': False})
    r.write_json(root / 'authorization.json', {
        'status': 'approved', 'at': r.utc(), 'allocation': ALLOCATION,
        'user_instruction': r.read(HERE / 'campaign_plan.json')['authorization'],
        'scope': 'One complete SearchQA Test1400 with unchanged V7, no extra trial or training.'})
    r.write_json(root / 'cost_carryforward_manifest.json', {
        'parent': None, 'historical_charged_tokens': 0, 'historical_http': 0,
        'allocation': ALLOCATION, 'limits': limits,
        'accounting': 'First SearchQA ledger for this model. Existing provider probes and LiveMath ledgers remain separate and unchanged; their costs are not represented as zero or included twice.'})
    c.prepare(STAGE, HERE / (model + '_spec.json'))
    stage = root / STAGE
    plan = r.read(stage / 'plan.json')
    cfg, refcfg = r.read(stage / ARM / 'config.json'), r.read(REFERENCE / ARM / 'config.json')
    assert cfg['runtime'] == refcfg['runtime'] and cfg['program_environment'] == refcfg['program_environment']
    ref = r.read(REFERENCE / 'plan.json')
    assert plan['input_identity'][ARM] == ref['input_identity'][ARM]
    assert plan['frozen'][ARM] == ref['frozen'][ARM]
    assert r.read(stage / 'tasks.json') == r.read(REFERENCE / 'tasks.json')
    canonical = Path(cfg['manifest'])
    r.verify(canonical.parent.parent)
    public = r.read(r.DATASETS / 'test.json')
    assert len(public['tasks']) == len({t['task_id'] for t in public['tasks']}) == 1400
    assert [t['task_id'] for t in public['tasks']] == [t['task_id'] for t in r.read(canonical)['tasks']]
    assert public['split_sha256'] == r.sha(canonical)
    assert public['authority_sha256'] == r.sha(canonical.parent.parent / 'manifest.json')
    locked = [Path(__file__), HERE / 'check_models.py', HERE / 'offline_acceptance.json',
              HERE / (model + '_config.json'), HERE / 'source_provenance.json',
              CONFIGURED[model], REFERENCE / 'plan.json', canonical,
              canonical.parent.parent / 'manifest.json', r.DATASETS / 'test.json']
    plan['files'].update({str(p): r.sha(p) for p in locked})
    plan.update(canonical_order=True, fresh_independent_run=True, reference=str(REFERENCE / ARM),
                source_provenance=r.read(HERE / 'source_provenance.json'),
                generation_seed_supported=cfg['llm']['generation_seed_supported'])
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
    current, result = root / STAGE / 'current.json', root / STAGE / 'result.json'
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
        print(json.dumps(specs(), ensure_ascii=False))
    else:
        assert args.action != 'run' or args.model != 'all', 'One process per model'
        for model in MODELS if args.model == 'all' else (args.model,):
            select(model)
            result = (prepare(model) if args.action == 'prepare' else start(model, args.resume)
                      if args.action == 'start' else monitor(model) if args.action == 'status'
                      else c.check(STAGE) if args.action == 'check' else c.execute(STAGE))
            print(json.dumps({'model': model, 'result': result}, ensure_ascii=False), flush=True)
