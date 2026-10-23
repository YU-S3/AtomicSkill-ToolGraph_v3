"""Host-only comparisons. This file and outputs are never imported by a Bank."""
import json
import sqlite3
from pathlib import Path
import sys
sys.path.insert(0, '/home/yangchengyu/asg_searchqa_gemini38_release_20261023/src')
from atomic_skillgraph.harness.scorers.searchqa import extract_answer, normalize_answer

HERE=Path(__file__).resolve().parent
DATA=Path('/home/yangchengyu/main_experiment_v1_resources_cf4_r3_20261009_v1/searchqa')
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def results(paths):
    out={}
    for root in paths:
        p=Path(root)/'test/run.sqlite3'
        with sqlite3.connect(p.as_uri()+'?mode=ro',uri=True) as db:
            for tid,raw in db.execute("SELECT id,result FROM tasks WHERE status='completed'"):
                assert tid not in out
                v=json.loads(raw)
                out[tid]={'prediction':extract_answer(v['execution'].get('prediction') or ''),
                    'em':float(v['score']['hard']),'f1':v['score']['soft'],
                    'tokens':sum(u['total_tokens'] for u in v['usage'])}
    return out

def main():
    models={
        'minimax_r2':results([
            '/home/yangchengyu/searchqa_minimax_r2_test1400_seed42_20261023/fulltest1400/r2',
            '/home/yangchengyu/searchqa_minimax_r2_test1400_resume_20261024/fulltest1400/r2']),
        'glm_v7':results(['/home/yangchengyu/searchqa_v7_crossmodel_test1400_seed42_20261021/glm53_flash/fulltest1400/v7_text_compiler']),
    }
    reference=read(HERE.parent/'searchqa_gemini_gepa_audit_20261020/predictions_host_only.json')
    for key,value in reference.items():models[key]={'searchqa:'+tid:v for tid,v in value.items()}
    tasks=read(DATA/'test.json')['tasks'];gold=read(DATA/'evaluator_records_test.json')
    rows=[]
    for t in tasks:
        tid=t['task_id']
        rows.append({'task_id':tid,'question':t['goal'],'gold':gold[tid]['answers'],
                     'results':{m:v[tid] for m,v in models.items()}})
    write(HERE/'comparisons_host_only.json',rows)
    write(HERE/'source_identity.json',{'results':{m:{'count':len(v),'correct':sum(x['em'] for x in v.values()),'tokens':sum(x['tokens'] for x in v.values())} for m,v in models.items()},'canonical_data':str(DATA),'post_test_development':True})
    errors={}
    for model in ('minimax_r2','glm_v7'):
        errors[model]=[]
        for row in rows:
            own=row['results'][model]
            if own['em']:continue
            refs={m:v['prediction'] for m,v in row['results'].items() if v['em'] and m!=model}
            norm=normalize_answer(own['prediction'])
            norms=[normalize_answer(a) for a in row['gold']]
            errors[model].append({**row,'correct_references':refs,
                'word_containment':any(norm and g and (set(norm.split())<=set(g.split()) or set(g.split())<=set(norm.split())) for g in norms)})
        print(model,len(errors[model]),'with reference',sum(bool(x['correct_references']) for x in errors[model]),'containment',sum(x['word_containment'] for x in errors[model]))
    write(HERE/'errors_host_only.json',errors)
    for model in errors:
        print('\nMODEL',model)
        for e in errors[model]:
            if e['correct_references']:
                print(json.dumps({'q':e['question'],'own':e['results'][model]['prediction'],'gold':e['gold'],'ref':next(iter(e['correct_references'].values()))},ensure_ascii=False))

if __name__=='__main__':main()
