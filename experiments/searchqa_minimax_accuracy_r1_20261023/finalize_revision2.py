"""Compare R2 with the same 96 saved V7/R1 tasks; explicitly sequential, not a new paired control."""
from collections import Counter
import json
import math
import zipfile
import analyze
import run_revision2 as x

r,c,HERE,ROOT,STAGE=x.r,x.c,x.HERE,x.ROOT,x.STAGE


def finalize():
    c.CAMPAIGN=ROOT;c.check(STAGE)
    assert r.read(ROOT/STAGE/'result.json')['status']=='completed'
    ids=[t['task_id'] for t in r.read(HERE/'tasks96_public.json')['tasks']]
    rows={a:c.completed(x.PARENT/x.base.STAGE,a) for a in ['v7','r1']}
    rows['r2']=c.completed(ROOT/STAGE,'r2')
    assert len(rows['r2'])==96 and all(t in rows[a] for a in rows for t in ids)
    rows={a:{t:values[t] for t in ids} for a,values in rows.items()}
    stats={a:analyze.metric(v) for a,v in rows.items()}
    inherited=r.read(ROOT/'inherited_budget.json');ledger=r.read(ROOT/'budget.json')
    assert all(ledger['attempts'][k]==v for k,v in inherited['attempts'].items())
    added=[v for k,v in ledger['attempts'].items() if k not in inherited['attempts']]
    assert all(v['status']=='metered' for v in added)
    tokens=sum(v['accounted_tokens'] for v in added)
    assert tokens==stats['r2']['tokens'] and len(added)==stats['r2']['http']
    paired={};changes=[]
    for a in ['r1','r2']:
        gain=sum(not rows['v7'][t]['score']['hard'] and rows[a][t]['score']['hard'] for t in ids)
        loss=sum(rows['v7'][t]['score']['hard'] and not rows[a][t]['score']['hard'] for t in ids)
        n=gain+loss
        paired[a]={'gained':gain,'lost':loss,'net':gain-loss,'p':min(1.,2*sum(math.comb(n,k) for k in range(min(gain,loss)+1))/2**n) if n else 1.,
                   'token_ratio':stats[a]['tokens']/stats['v7']['tokens']}
    for tid in ids:
        v={a:rows[a][tid] for a in rows}
        changes.append({'task_id':tid,'question':v['v7']['task']['goal'],
            'predictions':{a:b['execution']['prediction'] for a,b in v.items()},'scores':{a:b['score'] for a,b in v.items()}})
    value={'at':r.utc(),'status':'completed','tasks':96,'metrics':stats,'versus_v7':paired,
           'r2_new_tokens':tokens,'r2_new_http':len(added),'prior_entries_unchanged':True,
           'full_test_started':False,'sequential_controls_reused':True,'post_test_development':True,
           'cumulative_physical_tokens':sum(v.get('accounted_tokens',0) for v in ledger['attempts'].values()),
           'retained_unmetered_reservations':sum(v['reserved_tokens'] for v in ledger['attempts'].values() if v['status']!='metered')}
    r.write_json(HERE/'r2_results.json',value);r.write_json(HERE/'r2_changes_host_only.json',changes)
    text='# MiniMax R1/R2 后续审查\n\n以下只比较同一96题；V7/R1来自此前128题对照，R2在其后运行，没有额外重跑对照模型。选择按seed20261024从固定128题再均匀抽96题，未按结果筛选。已接触旧Test，属于事后开发；不是完整1400题结果。\n\n|版本|正确数|EM|F1|Tokens|HTTP|完整原生图|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for a,s in stats.items():text+=f"|{a}|{s['correct']}/96|{s['em']:.2%}|{s['f1']:.2%}|{s['tokens']:,}|{s['http']}|{s['native_graphs_completed']}/96|\n"
    for a,s in paired.items():text+=f"\n{a}相对V7改对{s['gained']}、改错{s['lost']}、净增{s['net']}；token比{s['token_ratio']:.3f}；精确配对p={s['p']:.4f}。\n"
    text+='\nR2仅补回标题/引文缺词约束、说明Jeopardy题型、在完整context后重复原题；事实卡片、Program、模型参数和评分器均未改变。R1的长生成/截断失败及全部费用保留，不用R2覆盖。\n'
    text+=f"\nR2新增{tokens:,} tokens/{len(added)} HTTP。原累计8M额度保持不变；历史未知预留{value['retained_unmetered_reservations']:,} tokens原样保留。尚不能据此保证完整Test达到82%或85%，也未自动启动完整Test。\n"
    (HERE/'R1_R2后续审查报告.md').write_text(text)
    dest=HERE.parent/'SearchQA_MiniMax_R1_R2_审查_20261023.zip'
    assert not dest.exists()
    with zipfile.ZipFile(dest,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        manifest={}
        for base,prefix in [(HERE,'delivery'),(x.PARENT,'paired128'),(ROOT,'followup96')]:
            for p in sorted(base.rglob('*')):
                if not p.is_file() or '__pycache__' in p.parts or p.name in ['runner.log','pid','.lock','budget.json.lock']:continue
                name=prefix+'/'+str(p.relative_to(base));z.write(p,name);manifest[name]=r.sha(p)
        z.writestr('SHA256SUMS.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(dest) as z:assert z.testzip() is None
    r.write_json(HERE/'r2_delivery_artifacts.json',{'path':str(dest),'sha256':r.sha(dest),'bytes':dest.stat().st_size,'results':value})


if __name__=='__main__':finalize()
