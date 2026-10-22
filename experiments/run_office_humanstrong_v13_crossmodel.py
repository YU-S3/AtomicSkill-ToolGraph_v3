"""One frozen rules-bank Test102 per model, interleaved under one cumulative ledger."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from filelock import FileLock
import run_office_humanstrong_v13 as fixed

common = fixed.common
PARENT = Path('/home/yangchengyu/officeqa_humanstrong_v13_rules_test102_repeat2_seed42_20261022')
MODELS = ('kimi_k26', 'glm53_flash')
MODEL_CONFIGS = common.SOURCE/'experiments/searchqa_v7_crossmodel_test1400_20261021'
CEILING, TASK_CAP = 262000000, 216576
PROVIDER_SOURCE = {'path':'/home/yangchengyu/asg_searchqa_glm53_refusal_release_20261020',
    'commit':'a4b96eedda9dc920afe4b7105b9e37f93cc931e8',
    'sha256':'256c926bf9d696fa890adc84dc66c9a5db54c27b5c90ee11ea507ceec196c958'}


def model_config(original, model):
    cfg = deepcopy(original)
    transport = common.read(MODEL_CONFIGS/(model+'_config.json'))['llm']
    for key in ('provider','dialect','base_url','model','api_key_env','capability_profile','stream'):
        cfg['llm'][key] = deepcopy(transport[key])
    cfg['llm'].update(display_name=transport['model'], max_retries=0)
    cfg['llm']['runtime']['request_timeout_seconds'] = 600
    if model == 'kimi_k26':
        def remove_effort(value):
            if isinstance(value, dict):
                value.pop('reasoning_effort', None)
                for key, child in value.items():
                    if key != 'capability_profile': remove_effort(child)
        remove_effort(cfg['llm'])
    else:
        # The configured GLM dialect requires thinking even during finalization.
        cfg['llm']['purpose_overrides']['finish_only']['protocol']['thinking_type'] = 'enabled'
    assert cfg['llm']['runtime']['max_completion_tokens'] == 32768
    assert cfg['llm']['purpose_overrides']['finish_only']['max_completion_tokens'] == 32768
    assert cfg['llm']['runtime']['max_total_tokens_per_task'] == TASK_CAP
    return common.validate_config(cfg)


def prepare(root):
    if root.exists(): raise ValueError('Use a fresh campaign; never overwrite')
    code = common.code_identity()
    if code['tracked_dirty'] or not code['git_sha']: raise ValueError('Commit before preparing')
    original, ledger, locks = fixed.paired.settled_parent(PARENT)
    assert original['code']['git_sha'] == '52acb999aff031087f3aa0053c94caf1797ea9f7'
    assert original['package_identity'] == common.package_identity()
    rows = original['episodes']
    tasks = common.read(PARENT/'tasks.json')
    assert len(rows) == 102 and {e['arm'] for e in rows} == {'rules'}
    assert [e['task_id'] for e in rows] == [t['task_id'] for t in tasks['tasks']]
    assert all(common.sha(p) == sha for p,sha in original['protected_files'].items())
    used = sum(map(common.charge, ledger['attempts'].values()))
    cap = len(rows)*len(MODELS)*TASK_CAP
    if used+cap > CEILING: raise ValueError('Cumulative authority insufficient')
    base = common.read(PARENT/'base_config.json')
    configs = {m:model_config(base,m) for m in MODELS}
    decisions = max(16,base['runtime']['global_action_budget']*4+16)
    http_cap = len(rows)*len(MODELS)*(decisions*2+1)
    root.mkdir(parents=True)
    common.save_json(root/'budget_prefix.json',ledger)
    limits = {**ledger['limits'],'token_limit':used+cap,
              'request_limit':len(ledger['attempts'])+http_cap,'finish_reserve':0}
    ledger['limits'] = limits
    common.save_json(root/'budget.json',ledger)
    common.save_json(root/'tasks.json',tasks)
    for model,cfg in configs.items():
        cfg['data_dir'] = str(root/model/'episodes')
        cfg['experiment']['output_dir'] = str(root/model)
        cfg['budget'].update(token_limit=limits['token_limit'],request_limit=limits['request_limit'])
        common.write_json(root/(model+'_config.json'),cfg)
    common.write_json(root/'base_config.json',configs[MODELS[0]])
    episodes = []
    for old in rows:
        previous = Path(old['path'])
        for model in MODELS:
            target = root/model/'episodes'/previous.parent.name/'rules'
            shutil.copytree(previous/'bank',target/'bank')
            actual = deepcopy(configs[model])
            actual['data_dir'] = str(target/'bank')
            actual['experiment']['output_dir'] = str(target)
            common.write_json(target/'config.json',actual)
            identity = common.tree_identity(target/'bank')
            assert identity == old['bank_identity']
            episodes.append({'task_id':old['task_id'],'arm':'rules','model_key':model,
                'path':str(target),'config_hash':common.sha(target/'config.json'),'bank_identity':identity})
    protected = dict(original['protected_files'])
    for path in (PARENT/'budget.json',PARENT/'plan.json',PARENT/'launch_status.json',
                 PARENT/'comparison.json',PARENT/'billing_result.json',root/'tasks.json',
                 root/'base_config.json',root/'budget_prefix.json',Path(__file__).resolve(),
                 *[MODEL_CONFIGS/(m+'_config.json') for m in MODELS],
                 *[root/(m+'_config.json') for m in MODELS]):
        protected[str(path)] = common.sha(path)
    plan = {**original,'created_at':common.utc(),'code':code,'base_run':str(PARENT),
        'inference_config_source':str(PARENT),'episodes':episodes,'protected_files':protected,
        'ledger_limits':limits,'inherited_used_or_reserved':used,'new_token_cap':cap,'new_http_cap':http_cap,
        'authorized_cumulative_ceiling':CEILING,'test_runs_authorized_in_this_plan':2,
        'repeat_index':None,'models':list(MODELS),'comparison_arms':['rules'],'fresh_paired_episodes':0,
        'selection':'All canonical Test102 in original order for each model; alternate models at task boundaries under one ledger.',
        'selection_receipt':{'rule':'Exact frozen V13 rules profile, without outcome filtering.',
                             'original_profile_selection':original['selection_receipt']},
        'authorization':'2026-10-22 user requested the same Bank on kimi-k2.6 and glm-5.3-flash. '
            'One full Test102 per model within existing cumulative 262M; no added off arm, learning or Bank edits. '
            'Preserve all historical costs and unknown reservations. No automatic repeats.',
        'comparison_caveat':'Human-authored rules and deterministic selector, unchanged from DeepSeek. '
            'Development-exposed Test; not a Planner-selected or naturally learned Ours result.',
        'provider_source':PROVIDER_SOURCE,'runtime_changes':['tested provider dialect/stream adapter',
            'pass stream setting to provider','omit unsupported effort for Kimi; GLM finish thinking enabled'],
        'scheduler':'single process, alternating model per task, no concurrent ledger writers','locks':locks}
    common.save_json(root/'plan.json',plan)
    common.save_json(root/'plan_identity.json',{'sha256':common.sha(root/'plan.json')})
    common.save_json(root/'cost_carryforward_manifest.json',{k:plan[k] for k in (
        'authorization','base_run','inherited_used_or_reserved','authorized_cumulative_ceiling',
        'new_token_cap','new_http_cap','models','scheduler')})
    common.save_json(root/'launch_status.json',{'status':'prepared','created_at':common.utc()})
    common.verify(root)
    print(json.dumps({'root':str(root),'episodes':len(episodes),'new_token_cap':cap,
                      'cumulative_max':used+cap,'source':code}))


def status(root, details=False):
    result = common.status(root)
    plan = common.read(root/'plan.json')
    metrics = {m:{'completed':0,'correct':0,'tokens':0,'http':0} for m in MODELS}
    for e in plan['episodes']:
        path = Path(e['path'])/'summary.json'
        if not path.exists(): continue
        summary = common.read(path)
        if not summary['complete']: continue
        m = metrics[e['model_key']]
        m['completed'] += 1
        m['correct'] += summary['successes']
        m['tokens'] += summary['total_tokens']
        m['http'] += sum(u.get('call_count',0) for c in summary['cases'] for u in c['usage'])
    if result['launch'].get('status') == 'running':
        result['launch']['model_key'] = plan['episodes'][result['launch']['episode_index']]['model_key']
    result['models'] = metrics
    if details:
        completed = [e for e in plan['episodes'] if (Path(e['path'])/'summary.json').exists()]
        for row, episode in zip(result['results'], completed): row['model_key'] = episode['model_key']
    else: result.pop('results')
    return result


def execute(root, env):
    code = common.execute(root, env)
    result = status(root, details=True)
    common.save_json(root/'model_results.json', result)
    if code == 0: common.save_json(root/'comparison.json', result)
    return code


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=('prepare','check','launch','run','status'))
    ap.add_argument('--root',required=True)
    ap.add_argument('--env-file',default='/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env')
    args=ap.parse_args();root=Path(args.root).resolve();fixed.bind_runner(True)
    if args.action=='prepare':prepare(root)
    elif args.action=='check':common.verify(root);print('Frozen source, both model configs, all inputs and Bank snapshots verified; zero HTTP.')
    elif args.action=='status':print(json.dumps(status(root),ensure_ascii=False,indent=2))
    elif args.action=='run':sys.exit(execute(root,args.env_file))
    else:
        common.verify(root)
        with FileLock(str(root/'.launch.lock'),timeout=0):
            if common.read(root/'launch_status.json')['status']!='prepared':raise ValueError('Already launched')
            with (root/'run.log').open('x') as log:
                child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'run','--root',str(root),
                    '--env-file',args.env_file],cwd=common.SOURCE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            common.save_json(root/'launch_status.json',{'status':'launching','pid':child.pid,'started_at':common.utc()})
            (root/'run.pid').write_text(str(child.pid)+'\n')
            print(json.dumps({'pid':child.pid,'root':str(root)}))
