"""One bounded four-arm diagnosis: native control, rules, calculators, typed submission."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from filelock import FileLock
import run_office_humanstrong_v6 as paired

common=paired.common
PARENT='/home/yangchengyu/officeqa_humanstrong_v11_train62_pairs_seed42_20261021'
RESOURCES='/home/yangchengyu/officeqa_external_minimax_resources_20261020_v1'
ARMS=('off','rules','selective','guarded')
# Developer diagnosis only. The Bank matcher never reads this manifest or any task ID.
CASES={'UID0138':'source revision gain','UID0060':'CAGR display gain','UID0239':'continuous-change gain',
       'UID0061':'fraction/percent regression','UID0062':'revision rule not acted on','UID0166':'unit display ambiguity',
       'UID0208':'long polynomial series','UID0074':'fitting-scope refusal',
       'UID0098':'correct total/trim control','UID0146':'correct ratio control',
       'UID0115':'correct mean control','UID0181':'correct result with unnecessary reference retry'}


def bind_runner():
    paired.PACKAGE=common.SOURCE/'resources/office_humanstrong_v12'
    paired.VERSION='office.humanstrong.v12.direction-search.20261022'
    paired.CUMULATIVE_CAP=262000000
    paired.AUTHORIZATION=('2026-10-22 user requested implementing the plan and comparing multiple short-rule/Program '
        'directions to improve accuracy while preserving correct answers. This plan fixes one 12-task, four-arm '
        'development diagnosis (48 fresh executions), under the previously approved cumulative 262M ceiling. '
        'Preserve all historical billing and unknown reservations. No Ours Train partition, automatic repetition or Test in this plan.')
    paired.bind_runner()


def selection(tasks):
    chosen=[t for t in tasks if t['task_id'].split(':')[-1] in CASES]
    if len(chosen)!=12:raise ValueError('Development diagnostic membership changed')
    return chosen,{'rule':'12 fixed development Train cases in canonical order, covering observed wins, regressions, '
        'unresolved failures and previously correct controls; no Ours training split.',
        'prior_scores_used_for_selection':True,'development_exposed':True,'case_reasons':CASES,
        'decision':'Compare current off on every case. Select by correctness, then fewer regressions, then cost. '
        'Do not claim this selected sample estimates general accuracy. No Test until a separately frozen chosen profile.',
        'full_test_gate':{'all_48_completed_and_metered':True,'positive_net_vs_current_off':True,
                          'no_losses_vs_current_off':True,'no_new_infrastructure_error':True,
                          'ranking':['correct_count_descending','tokens_ascending'],
                          'review':'Inspect actual source/scale decisions and refusals before selecting. '
                                   'A failed gate calls for a separately versioned bounded revision, not automatic Test.'}}


def prepare(root):
    return paired.prepare(root,PARENT,RESOURCES,PARENT,split='train',selector=selection,comparison_arms=ARMS)


def analysis(root):
    state=common.status(root);tasks={t['task_id']:t for t in common.read(root/'tasks.json')['tasks']}
    by_id={};metrics={a:{'completed':0,'correct':0,'tokens':0,'http':0,'program_calls':0,'refusals':0,'submitted':0} for a in ARMS}
    for e in common.read(root/'plan.json')['episodes']:
        path=Path(e['path'])/'summary.json'
        if not path.exists():continue
        summary=common.read(path);c=summary['cases'][0];attempts=c['execution']['attempts'];arm=e['arm']
        row={'correct':bool(c['score']['hard']),'prediction':c['execution']['prediction'],'tokens':summary['total_tokens'],
             'reason':c['execution']['reason'],'program_calls':len(attempts),
             'refusals':sum(a['status']!='ok' for a in attempts),'submitted':any(a.get('submission_by_program') for a in attempts),
             'http':sum(u.get('call_count',0) for u in c['usage']),'summary':str(path)}
        by_id.setdefault(e['task_id'],{})[arm]=row
        metrics[arm]['completed']+=1
        for k in ('correct','tokens','http','program_calls','refusals','submitted'):metrics[arm][k]+=row[k]
    comparisons={}
    for arm in ARMS[1:]:
        paired_rows={uid:rows for uid,rows in by_id.items() if arm in rows and 'off' in rows}
        wins=[uid for uid,rows in paired_rows.items() if rows[arm]['correct'] and not rows['off']['correct']]
        losses=[uid for uid,rows in paired_rows.items() if rows['off']['correct'] and not rows[arm]['correct']]
        comparisons[arm]={'pairs':len(paired_rows),'wins':wins,'losses':losses,'net':len(wins)-len(losses)}
    return {'status':state['launch'],'metrics':metrics,'vs_off':comparisons,'cases':by_id,
        'new_tokens_used_or_reserved':state['new_tokens_used_or_reserved'],'new_http':state['new_http'],
        'cumulative_tokens_used_or_reserved':state['cumulative_tokens_used_or_reserved'],'new_unsettled':state['new_unsettled']}


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['prepare','check','launch','run','status','analyze'])
    ap.add_argument('--root',required=True);ap.add_argument('--env-file',default='/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env')
    args=ap.parse_args();root=Path(args.root).resolve();bind_runner()
    if args.action=='prepare':prepare(root)
    elif args.action=='check':common.verify(root);print('Frozen identities and inherited costs verified; zero model HTTP.')
    elif args.action in ('status','analyze'):
        result=analysis(root)
        if args.action=='analyze':common.save_json(root/'direction_comparison.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='cases'},ensure_ascii=False,indent=2))
    elif args.action=='run':sys.exit(common.execute(root,args.env_file))
    else:
        common.verify(root)
        with FileLock(str(root/'.launch.lock'),timeout=0):
            if common.read(root/'launch_status.json')['status']!='prepared':raise ValueError('Already launched')
            with (root/'run.log').open('x') as log:
                child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'run','--root',str(root),
                    '--env-file',args.env_file],cwd=common.SOURCE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            common.save_json(root/'launch_status.json',{'status':'launching','pid':child.pid,'started_at':common.utc()})
            (root/'run.pid').write_text(str(child.pid)+'\n');print(json.dumps({'pid':child.pid,'root':str(root)}))
