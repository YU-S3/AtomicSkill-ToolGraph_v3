"""Host-only paired statistics and review; gold is never loaded by the Bank."""
from collections import Counter
import json
import math
from pathlib import Path
import random
import run_comparison as run
from atomic_skillgraph.harness.scorers.searchqa import evaluate, extract_answer

c,r,HERE,ROOT,STAGE=run.c,run.r,run.HERE,run.ROOT,run.STAGE


def metric(rows):
    vals=list(rows.values())
    return {'tasks':len(vals),'correct':sum(bool(v['score']['hard']) for v in vals),
        'em':sum(bool(v['score']['hard']) for v in vals)/len(vals) if vals else None,
        'f1':sum(v['score']['soft'] for v in vals)/len(vals) if vals else None,
        'tokens':sum(u['total_tokens'] for v in vals for u in v['usage']),
        'http':sum(len(q.get('http_attempts',[])) for v in vals for q in v['requests']),
        'program_calls':sum(len(v['execution'].get('attempts',[])) for v in vals),
        'program_errors':sum(a['status']!='ok' for v in vals for a in v['execution'].get('attempts',[]))}


def analyze():
    c.check(STAGE)
    status=c.status(STAGE)
    rows={a:c.completed(ROOT/STAGE,a) for a in ['v7','r1']}
    tasks=r.read(HERE/'tasks128_public.json')['tasks']
    gold=r.read(r.DATASETS/'evaluator_records_test.json')
    old={v['task_id']:v for v in r.read(HERE/'test1400_host_only.json')}
    common=[t['task_id'] for t in tasks if all(t['task_id'] in rows[a] for a in rows)]
    deltas=[];changes=[];surface=[]
    for tid in common:
        a,b=rows['v7'][tid],rows['r1'][tid]
        delta=int(bool(b['score']['hard']))-int(bool(a['score']['hard']));deltas.append(delta)
        item={'task_id':tid,'question':a['task']['goal'],'context':a['task']['inputs']['context'],
            'gold':gold[tid]['answers'],'v7_prediction':a['execution']['prediction'],'r1_prediction':b['execution']['prediction'],
            'v7_score':a['score'],'r1_score':b['score'],'delta':delta,
            'historical_v7_prediction':old[tid]['prediction'],'historical_v7_correct':old[tid]['correct']}
        if delta or a['execution']['prediction']!=b['execution']['prediction']: changes.append(item)
        raw=b['requests'][0]['response']['content']
        score=evaluate(raw,gold[tid]['answers'])
        if extract_answer(raw)!=extract_answer(b['execution']['prediction']):
            surface.append({'task_id':tid,'raw_answer':raw,'final':b['execution']['prediction'],
                'raw_em':score['em'],'final_em':b['score']['hard']})
    gained=deltas.count(1);lost=deltas.count(-1);discordant=gained+lost
    pvalue=min(1.,2*sum(math.comb(discordant,i) for i in range(min(gained,lost)+1))/2**discordant) if discordant else 1.
    rng=random.Random(20261023)
    boot=sorted(sum(rng.choices(deltas,k=len(deltas)))/len(deltas) for _ in range(4000)) if deltas else []
    inherited=r.read(ROOT/'inherited_budget.json');ledger=r.read(ROOT/'budget.json')
    added=[v for k,v in ledger['attempts'].items() if k not in inherited['attempts']]
    stats={a:metric({tid:rows[a][tid] for tid in common}) for a in rows}
    report={'at':r.utc(),'status':status['status'],'completed':{a:len(rows[a]) for a in rows},
        'common_tasks':len(common),'metrics_same_completed_tasks':stats,'gained':gained,'lost':lost,
        'net':gained-lost,'paired_mcnemar_exact_two_sided_p':pvalue,
        'paired_bootstrap95_em_delta':[boot[100],boot[3899]] if boot else None,
        'r1_v7_token_ratio':stats['r1']['tokens']/stats['v7']['tokens'] if stats['v7']['tokens'] else None,
        'measured_new_tokens':sum(x.get('accounted_tokens',0) for x in added),'new_physical_http':len(added),
        'new_attempt_statuses':dict(Counter(v['status'] for v in added)),
        'old_reservations_unchanged':all(ledger['attempts'][k]==v for k,v in inherited['attempts'].items()),
        'remaining_token_admission':ledger['limits']['token_limit']-sum(v.get('accounted_tokens',v['reserved_tokens']) for v in ledger['attempts'].values()),
        'full_test_started':False,'post_test_development':True,
        'historical_subset_correct':sum(old[tid]['correct'] for tid in common)}
    r.write_json(HERE/'paired_results.json',report)
    r.write_json(HERE/'paired_changes_host_only.json',changes)
    r.write_json(HERE/'actual_surface_changes_host_only.json',surface)
    return report


if __name__=='__main__': print(json.dumps(analyze(),ensure_ascii=False))
