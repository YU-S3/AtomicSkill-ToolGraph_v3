"""One fresh canonical SearchQA Test1400 with the configured Gemini and frozen V7."""
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
from infrastructure_guard import docker_ready

r = c.r
ROOT = Path('/home/yangchengyu/searchqa_v7_gemini38_test1400_seed42_20261023')
REFERENCE = Path('/home/yangchengyu/searchqa_v2_v7_fulltest1400_seed42_20261020/fulltest1400')
CONFIGURED = HERE.parent/'rootflow_gemini38_configuration_20261023/rootflow_gemini38_flash.yaml'
PRIOR_CONFIG = HERE.parent/'searchqa_v7_minimax_docker_replacement_20261023/base_config.json'
STAGE, ARM = 'fulltest1400', 'v7_text_compiler'
ALLOCATION = {'tokens': 8000000, 'http': 3000}
AUTHORIZATION = '用户：启动配置好的gemini-3.8-flash的test；按当前会话同一V7 Bank运行一次SearchQA完整Test1400。'
c.CAMPAIGN = ROOT


def config():
    cfg = r.read(PRIOR_CONFIG)
    configured = yaml.safe_load(CONFIGURED.read_text())
    limits = configured['runtime']
    assert limits['completion'] == 65536 and limits['finish_completion'] == 16384
    assert limits['timeout'] == 600 and limits['parent_tokens'] == 128000
    cfg['llm'] = {**deepcopy(configured['llm']), 'provider':'openai_compatible',
        'display_name':configured['llm']['model'], 'input_modalities':['text'], 'max_retries':0,
        'reasoning_effort':'high', 'protocol':{'thinking_type':'enabled'},
        'runtime':{'max_completion_tokens':limits['completion'], 'request_timeout_seconds':limits['timeout'],
                   'max_total_tokens_per_task':limits['parent_tokens']},
        'purpose_overrides':{'finish_only':{'protocol':{'thinking_type':'enabled'},
            'reasoning_effort':'high','max_completion_tokens':limits['finish_completion']}}}
    cfg['budget'].update(eval_task_tokens=limits['parent_tokens'], train_task_tokens=limits['parent_tokens'])
    cfg['experiment']['authored_bank_runtime'] = 'human.searchqa.accuracy.v7.text_compiler'
    return r.validate_config(cfg)


def specs():
    assert not ROOT.exists(), 'Preserve existing runs'
    source = r.code_identity()
    provenance = r.read(HERE/'source_provenance.json')
    assert source['git_sha'] == provenance['source_commit'] and not source['tracked_dirty']
    ref = r.read(REFERENCE/'plan.json')
    assert r.read(REFERENCE/'result.json')['status'] == 'completed'
    assert r.digest(ref) == r.read(REFERENCE/'plan_identity.json')['sha256']
    assert r.tree_identity(REFERENCE/ARM/'input') == ref['input_identity'][ARM]
    assert r.tree_identity(REFERENCE/ARM/'frozen_bank') == ref['frozen'][ARM]
    cfg = config()
    r.load_env('/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env')
    assert os.getenv(cfg['llm']['api_key_env']), 'Configured API key is missing'
    r.write_json(HERE/'campaign_plan.json', {'source':source,'allocation':ALLOCATION,
        'authorization':AUTHORIZATION,'allocation_basis':'Conservative 8M/3000 cap for this single authorized SearchQA run; no automatic expansion.',
        'bank':str(REFERENCE/ARM),'bank_unchanged':True,'configured_key_present':True,
        'source_provenance':provenance,'scope':'One fresh independent Test1400; no Train or Bank edits.'})
    r.write_json(HERE/'base_config.json',cfg)
    r.write_json(HERE/'spec.json', {'tasks':str(r.DATASETS/'test.json'),'base_config':str(HERE/'base_config.json'),
        'arms':{ARM:{'package':str(REFERENCE/ARM/'input'),'kind':'native','frozen_source':str(REFERENCE/ARM/'frozen_bank')}},
        'purpose':'One full canonical Test1400 using configured RootFlow gemini-3.8-flash; unchanged V7 Bank, messages, workflow, public context and official scorer.'})
    return {'specs_ready':True,'source':source,'key_present':True,'llm':cfg['llm']}


def prepare():
    assert not ROOT.exists(), 'Never overwrite an existing evaluation'
    receipt = r.read(HERE/'offline_acceptance.json')
    assert receipt['passed'] and receipt['driver_sha256'] == r.sha(c.__file__)
    assert receipt['launcher_sha256'] == r.sha(__file__)
    assert receipt['guard_sha256'] == r.sha(HERE/'infrastructure_guard.py')
    limits = {'token_limit':ALLOCATION['tokens'],'request_limit':ALLOCATION['http'],'finish_reserve':0}
    for name in ('budget.json','inherited_budget.json'):
        r.write_json(ROOT/name, {'limits':limits,'attempts':{},'unknown_billing':False})
    r.write_json(ROOT/'authorization.json', {'status':'approved','at':r.utc(),'user_instruction':AUTHORIZATION,
        'scope':'One complete SearchQA Test1400 with unchanged V7; no extra trial or training.',
        'execution_cap':ALLOCATION,'cap_basis':'Execution safeguard selected for the authorized full test, not an authorization copied from another benchmark.'})
    r.write_json(ROOT/'cost_carryforward_manifest.json', {'parent':None,'prior_searchqa_tokens':0,'prior_searchqa_http':0,
        'limits':limits,'accounting':'First SearchQA Gemini run; interface probes and LiveMath costs remain in their separate original ledgers, neither erased nor billed twice.'})
    c.prepare(STAGE,HERE/'spec.json')
    stage = ROOT/STAGE
    plan, ref = r.read(stage/'plan.json'),r.read(REFERENCE/'plan.json')
    cfg, refcfg = r.read(stage/ARM/'config.json'),r.read(REFERENCE/ARM/'config.json')
    assert cfg['runtime'] == refcfg['runtime'] and cfg['program_environment'] == refcfg['program_environment']
    assert plan['input_identity'][ARM] == ref['input_identity'][ARM]
    assert plan['frozen'][ARM] == ref['frozen'][ARM]
    assert r.read(stage/'tasks.json') == r.read(REFERENCE/'tasks.json')
    canonical = Path(cfg['manifest'])
    r.verify(canonical.parent.parent)
    public = r.read(r.DATASETS/'test.json')
    assert len(public['tasks']) == len({t['task_id'] for t in public['tasks']}) == 1400
    assert [t['task_id'] for t in public['tasks']] == [t['task_id'] for t in r.read(canonical)['tasks']]
    assert public['split_sha256'] == r.sha(canonical)
    assert public['authority_sha256'] == r.sha(canonical.parent.parent/'manifest.json')
    files = [Path(__file__),HERE/'check_gemini.py',HERE/'offline_acceptance.json',HERE/'infrastructure_guard.py',
             HERE/'base_config.json',HERE/'source_provenance.json',CONFIGURED,REFERENCE/'plan.json',
             canonical,canonical.parent.parent/'manifest.json',r.DATASETS/'test.json']
    plan['files'].update({str(p):r.sha(p) for p in files})
    plan.update(canonical_order=True,fresh_independent_run=True,generation_seed_supported=False,
                reference=str(REFERENCE/ARM),source_provenance=r.read(HERE/'source_provenance.json'))
    r.write_json(stage/'plan.json',plan)
    r.write_json(stage/'plan_identity.json',{'sha256':r.digest(plan)})
    checked = c.check(STAGE)
    r.write_json(ROOT/'preflight.json',{**checked,'at':r.utc(),'real_http':0,'same_v7_bank':True,
        'same_public_input_order':True,'same_runtime_and_scorer':True,'no_inherited_results':True,'config':cfg['llm']})
    return checked


def status():
    value = c.status(STAGE)
    pid = (ROOT/'pid').read_text().strip() if (ROOT/'pid').exists() else None
    proc = Path('/proc')/str(pid)/'cmdline'
    value.update(pid=pid,process_alive=bool(pid and proc.exists() and str(Path(__file__)) in proc.read_text()))
    current, result = ROOT/STAGE/'current.json',ROOT/STAGE/'result.json'
    value['current'] = r.read(current) if current.exists() else None
    value['error'] = r.read(result).get('error') if result.exists() else None
    return value


def start(resume=False):
    with (ROOT/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        c.check(STAGE)
        assert not status()['process_alive'], 'Already running'
        assert not (ROOT/'pid').exists() or resume, 'Explicit resume required'
        assert not (ROOT/STAGE/'result.json').exists() or resume
        assert not (ROOT/'STOP_AFTER_TASK').exists() and not (ROOT/STAGE/'STOP_AFTER_TASK').exists()
        docker_ready(r.read(HERE/'base_config.json')['program_environment']['image_digest'])
        cmd = [sys.executable,'-B','-u',str(Path(__file__)),'run']
        with (ROOT/'run.log').open('ab') as log:
            child = subprocess.Popen(cmd,cwd=r.SOURCE,stdin=subprocess.DEVNULL,stdout=log,
                                     stderr=subprocess.STDOUT,start_new_session=True)
        (ROOT/'pid').write_text(str(child.pid)+'\n')
        launch = {'pid':child.pid,'at':r.utc(),'command':cmd,'resume':resume,
                  'plan_sha256':r.read(ROOT/STAGE/'plan_identity.json')['sha256']}
        r.write_json(ROOT/('resume_launch.json' if resume else 'launch.json'),launch)
        time.sleep(2)
        assert child.poll() is None, 'Inspect run.log'
        return launch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('specs','prepare','check','start','run','status'))
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    value = {'specs':specs,'prepare':prepare,'check':lambda:c.check(STAGE),'start':lambda:start(args.resume),
             'run':lambda:c.execute(STAGE),'status':status}[args.action]()
    print(json.dumps(value,ensure_ascii=False),flush=True)
