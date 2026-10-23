"""Finite three-model Bank experiment; all choices and costs remain auditable."""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
import campaign as c
from infrastructure_guard import DockerUnavailable, docker_ready
from provider_outcomes import retain_reservation
r=c.r
ROOT=Path('/home/yangchengyu/searchqa_target_accuracy_r2_20261024')
PREFIX=Path('/home/yangchengyu/searchqa_target_accuracy_20261024')
MODELS={
 'minimax':{'parent':'/home/yangchengyu/searchqa_minimax_r2_test1400_resume_20261024','arm':'r2','tokens':8000000,'requests':2000,'target':.82},
 'glm':{'parent':'/home/yangchengyu/searchqa_v7_crossmodel_test1400_seed42_20261021/glm53_flash','arm':'v7_text_compiler','tokens':8000000,'requests':2000,'target':.84},
 'gemini':{'parent':'/home/yangchengyu/searchqa_v7_gemini38_test1400_seed42_20261023','arm':'v7_text_compiler','tokens':10000000,'requests':2200,'target':.88}}
CODE_FILES=['campaign.py','runtime.py','provider_outcomes.py','provider_retry.py','infrastructure_guard.py','run_campaign.py','check_offline.py','build_reference.py']
AUTH='用户要求提高 MiniMax≥82%、Gemini≥88%、GLM≥84%，允许参考事实卡片、技能/Program迭代；沿用本对话真实付费实验授权。'

def setup():
    assert not (HERE/'campaign_plan.json').exists()
    tasks=r.read(r.DATASETS/'test.json')['tasks']
    chosen=sorted(random.Random(20261024).sample(range(1400),96))
    r.write_json(HERE/'probe96_public.json',{'tasks':[tasks[i] for i in chosen]})
    r.write_json(HERE/'selection.json',{'seed':20261024,'indices_zero_based':chosen,'rule':'Uniform without replacement, no correctness filtering; canonical order.',
        'post_test_development':True,'selection':'Compare baseline/scope/full-GEPA-transfer on 96. Choose the new candidate with more correct answers, then fewer tokens. Run exactly one fresh full1400 per model. Earlier sparse-card comparison stopped at boundaries; all costs retained. This is not automatic deployment or a claim of unseen Test performance.'})
    r.write_json(HERE/'campaign_plan.json',{'source':r.code_identity(),'authorization':AUTH,'models':MODELS,
        'whole_iteration_token_cap':26000000,'whole_iteration_http_cap':6200,'limits_unchanged_from_prefix':True,'phases':['paired probe96 (3 distinct strategy arms)','one selected full1400 per model'],
        'no_automatic_extra_iteration':True})
    for model,info in MODELS.items():
        parent=Path(info['parent']); baseline=parent/'fulltest1400'/info['arm']
        cfg=r.read(baseline/'config.json')
        assert cfg['llm']['max_retries']==0
        r.write_json(HERE/model/'base_config.json',cfg)
        arms={'baseline':{'kind':'native','package':str(baseline/'input'),'frozen_source':str(baseline/'frozen_bank')}}
        arms.update({a:{'kind':'native','package':str(HERE/'packages'/a)} for a in ('scope','gepa')})
        r.write_json(HERE/model/'probe_spec.json',{'tasks':str(HERE/'probe96_public.json'),'base_config':str(HERE/model/'base_config.json'),
            'arms':arms,'purpose':'User-authorized post-Test authored development; fixed 96 paired comparison with unchanged public inputs, provider parameters and scorer. Candidate choice precedes new full Test1400.'})
    return r.read(HERE/'campaign_plan.json')

def lock_plan(model,stage):
    root=ROOT/model/stage;p=root/'plan.json';plan=r.read(p)
    for arm in plan['arms']:
        path=root/arm/'config.json';cfg=r.read(path)
        cfg['experiment']['authored_bank_runtime']=r.read(root/arm/'frozen_bank/imported_bank.json')['method_version']
        r.write_json(path,cfg);plan['files'][str(path)]=r.sha(path)
    files=[HERE/n for n in CODE_FILES]+[HERE/'offline_acceptance.json',HERE/'selection.json']
    plan['files'].update({str(p):r.sha(p) for p in files})
    r.write_json(p,plan);r.write_json(root/'plan_identity.json',{'sha256':r.digest(plan)})
    return c.check(stage)

def prepare(model):
    info=MODELS[model];root=ROOT/model;parent=PREFIX/model;c.CAMPAIGN=root
    assert not root.exists()
    receipt=r.read(HERE/'offline_acceptance.json');assert receipt['passed'] and receipt['real_http']==0
    assert all(r.sha(HERE/n)==s for n,s in receipt['files'].items())
    assert r.read(parent/'supervisor.json')['state']=='stopped'
    old=r.read(parent/'budget.json');state=deepcopy(old)
    charge=sum(a.get('accounted_tokens',a['reserved_tokens']) for a in old['attempts'].values())
    measured=sum(a.get('accounted_tokens',0) for a in old['attempts'].values())
    r.write_json(root/'parent_budget.json',old)
    r.write_json(root/'budget.json',state)
    governor=r.BudgetGovernor(root/'budget.json',**state['limits'])
    if governor.billing_blocked:
        # New experiment inherits old unknown usage at full reservation, without replaying that old task.
        recovered=state.get('inherited_reservation_recovery',{}).get('attempts',{})
        unknown={k for k,a in state['attempts'].items() if a['status']!='metered' and k not in recovered}
        for path in (parent/'fulltest1400'/info['arm']/'test/requests').glob('*.json'):
            for req in r.read(path).get('requests',[]):
                for attempt in req.get('http_attempts',[]):
                    if attempt['request_id'] in unknown:
                        assert model=='gemini' and attempt['http_status']==429 and attempt.get('usage') is None
                        retain_reservation(governor,attempt,'new_authorized_bank_experiment_preserves_prior_429_reservation')
        assert not governor.billing_blocked
    r.write_json(root/'inherited_budget.json',governor.state)
    r.write_json(root/'authorization.json',{'status':'approved','user_request':AUTH,'at':r.utc(),'model':model,
        'limits':state['limits'],'remaining_token_cap':state['limits']['token_limit']-charge,'remaining_http_cap':state['limits']['request_limit']-len(old['attempts']),
        'scope':'Revised comparison within the SAME iteration caps, then one new full1400. Preserve previous prefix and all costs. No additional full run or iterative spending loop.'})
    r.write_json(root/'cost_carryforward_manifest.json',{'parent':str(parent),'parent_budget_sha256':r.sha(parent/'budget.json'),
        'prior_searchqa_tokens':measured,'prior_searchqa_http':len(old['attempts']),'prior_admission_charge':charge,
        'unknown_reservations_retained':charge-measured,'physical_entries_unchanged':True,'new_limit_basis':'unchanged from prior iteration',
        'stopped_prefix':r.read(parent/'probe96/result.json')})
    c.prepare('probe96',HERE/model/'probe_spec.json')
    return lock_plan(model,'probe96')

def run_stage(model,stage):
    root=ROOT/model
    for recovery in range(4):
        try:
            r.write_json(root/'supervisor.json',{'state':'running','stage':stage,'docker_recovery':recovery,'at':r.utc()})
            return c.execute(stage)
        except DockerUnavailable:
            if recovery==3:raise
            cfg=r.read(HERE/model/'base_config.json')
            r.write_json(root/'supervisor.json',{'state':'waiting_for_docker','stage':stage,'at':r.utc()})
            for _ in range(6):
                if (root/'STOP_AFTER_TASK').exists():raise SystemExit(75)
                time.sleep(30)
                try:docker_ready(cfg['program_environment']['image_digest']);break
                except DockerUnavailable:pass

def report(model):
    root=ROOT/model;c.CAMPAIGN=root
    stages={stage:c.status(stage) for stage in ('probe96','fulltest1400') if (root/stage/'plan.json').exists()}
    ledger=r.read(root/'budget.json');old=r.read(root/'inherited_budget.json')
    added=[a for k,a in ledger['attempts'].items() if k not in old['attempts']]
    out={'model':model,'target':MODELS[model]['target'],'stages':stages,
        'new_measured_tokens':sum(a.get('accounted_tokens',0) for a in added),'new_http':len(added),
        'new_unknown_or_inflight_reservations':sum(a['reserved_tokens'] for a in added if a['status']!='metered'),
        'alive':(root/'pid').exists() and Path('/proc', (root/'pid').read_text().strip()).exists(),'at':r.utc()}
    if (root/'selection_result.json').exists():out['selection']=r.read(root/'selection_result.json')
    if 'fulltest1400' in stages and stages['fulltest1400']['status']=='completed':
        a=next(iter(stages['fulltest1400']['arms'].values()));out.update(em=a['correct']/1400,target_met=a['correct']/1400>=MODELS[model]['target'])
    return out

def run(model):
    root=ROOT/model;c.CAMPAIGN=root
    with (root/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            run_stage(model,'probe96')
            root_probe=root/'probe96'
            stats=c.status('probe96')['arms']
            chosen=min(('scope','gepa'),key=lambda a:(-stats[a]['correct'],stats[a]['tokens'],a))
            decision={'candidate':chosen,'probe':stats,'rule':'most correct new candidate, then fewer tokens','frozen_at':r.utc(),
                'probe_does_not_establish_full_test_accuracy':True}
            if (root/'selection_result.json').exists():assert r.read(root/'selection_result.json')['candidate']==chosen
            else:r.write_json(root/'selection_result.json',decision)
            if not (root/'fulltest1400').exists():
                spec={'tasks':str(r.DATASETS/'test.json'),'base_config':str(HERE/model/'base_config.json'),
                    'arms':{chosen:{'kind':'native','package':str(root_probe/chosen/'input'),'frozen_source':str(root_probe/chosen/'frozen_bank')}},
                    'purpose':'One fresh full Test1400 with preselected authored candidate. Post-Test development; official scores and all refusals retained. No automatic repeat or deployment.'}
                r.write_json(root/'fulltest_spec.json',spec);c.prepare('fulltest1400',root/'fulltest_spec.json');lock_plan(model,'fulltest1400')
            run_stage(model,'fulltest1400')
            result=report(model);r.write_json(root/'completion_report.json',result)
            r.write_json(root/'supervisor.json',{'state':'completed','at':r.utc()})
            return result
        except BaseException as error:
            r.write_json(root/'supervisor.json',{'state':'stopped','error_type':type(error).__name__,'message':str(error),'at':r.utc()})
            raise

def start(model):
    root=ROOT/model
    if (root/'pid').exists() and Path('/proc',(root/'pid').read_text().strip()).exists():raise RuntimeError('Process already running')
    with (root/'run.log').open('a') as log:
        proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'run','--model',model],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=str(HERE))
    (root/'pid').write_text(str(proc.pid)+'\n');return {'model':model,'pid':proc.pid,'root':str(root)}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['setup','prepare','run','start','status']);p.add_argument('--model',choices=list(MODELS));a=p.parse_args()
    result=setup() if a.action=='setup' else {'prepare':prepare,'run':run,'start':start,'status':report}[a.action](a.model)
    print(json.dumps(result,ensure_ascii=False))
