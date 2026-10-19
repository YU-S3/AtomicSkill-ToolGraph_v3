"""Read archived public inputs verbatim; keep evaluator answers in separate files."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parents[1]/'src'))
from atomic_skillgraph.empirical.contracts import digest

VERSION='office.external-minimax.public-pages.v1'
sha=lambda data:hashlib.sha256(data).hexdigest()


def public_input(record):
    prompt=record['target_user_prompt']
    prefix='## Question\n'+record['question']
    if not prompt.startswith(prefix):raise ValueError('Question and recorded public prompt disagree')
    body=prompt.split('\n## Oracle Parsed Pages\n',1)[1].split('\n## Candidate Files\n',1)[0]
    pages=[]
    pattern=r'^### (?P<document>[^\n]+?) page (?P<page>\d+)\nSource URL: (?P<url>[^\n]+)\n\n(?P<text>.*?)(?=^### |\Z)'
    for match in re.finditer(pattern,body,re.M|re.S):
        p=match.groupdict();text=p.pop('text')
        pages.append({'document':p['document'],'page':p['page'],'source_url':p['url'],
                      'text':text,'text_digest':digest(text)})
    if not pages or len(pages)!=len(re.findall(r'^### ',body,re.M)):
        raise ValueError('Unexpected public source-page structure')
    paths=prompt.split('\n## Candidate Files\n',1)[1].split('\n## Source Hints\n',1)[0]
    hints=prompt.split('\n## Source Hints\n',1)[1]
    original=[line[2:] for line in paths.splitlines() if line.startswith('- ')]
    candidates=[PurePosixPath(p).name for p in original]
    if len(set(candidates))!=len(candidates):raise ValueError('Candidate path mapping collides')
    inputs={'office_public_input_version':VERSION,'source_pages':pages,'candidate_files':candidates,
        'source_hints':[line[2:] for line in hints.splitlines() if line.startswith('- ')],
        'evidence_policy':'Provided source pages are primary public evidence; candidate filenames are hints, not a claim that other authorized files are forbidden.'}
    return inputs,{'original_prompt_sha256':sha(prompt.encode()),'candidate_path_mapping':dict(zip(original,candidates))}


def archive_records(path,seed):
    members={}
    with zipfile.ZipFile(path) as z:
        prefix=f'Trace2Skill_MiniMaxM3_seed{seed}/'
        def read(relative):
            raw=z.read(prefix+relative);members[relative]=sha(raw);return json.loads(raw)
        splits={s:read(f'reference/split_manifests/officeqa/{s}/ids.json') for s in ('train','val','test')}
        items={s:read(f'reference/split_manifests/officeqa/{s}/items.json') for s in splits}
        # Exact archived outputs are the source; only allowlisted public fields are exported.
        train=[read(f'runs/officeqa/seed{seed}/native/Trace2Skill/trajectories/{uid}/result.json') for uid in splits['train']]
        test=read(f'runs/officeqa/seed{seed}/test_results.json')['results']
        rows={r['id']:r for r in train+test}
        if set(rows)!=set(splits['train']+splits['test']):raise ValueError('Source split membership differs')
    return splits,items,rows,members


def materialize(archives,local_resources,output):
    output=Path(output)
    if output.exists():raise ValueError('Use a new independent resource directory')
    snapshots=[archive_records(p,s) for p,s in zip(archives,(42,43,44))]
    splits,items,rows,_=snapshots[0]
    for split in splits:
        if len(splits[split])!=len(set(splits[split])):raise ValueError('Duplicate ID')
    if sum(map(len,splits.values()))!=len(set(sum(splits.values(),[]))):raise ValueError('Overlapping splits')
    for other in snapshots[1:]:
        if other[0]!=splits:raise ValueError('External seeds have different split identities')
        for uid,r in rows.items():
            for key in ('question','ground_truth','target_user_prompt'):
                if r[key]!=other[2][uid][key]:raise ValueError(f'Cross-seed source differs: {uid} {key}')
    legacy=Path(local_resources)
    local_tasks={t['task_id'].split(':')[-1]:t for t in json.loads((legacy/'val.json').read_text())['tasks']}
    local_eval=json.loads((legacy/'evaluator_records_val.json').read_text())
    files={};provenance={}
    for split,ids in splits.items():
        tasks=[];answers={}
        for uid in ids:
            if split=='val':
                old=local_tasks[uid];item=next(x for x in items[split] if x.get('uid',x.get('id'))==uid)
                listed=lambda key:[x.strip() for x in item.get(key,'').splitlines() if x.strip()]
                goal=old['goal'];answer=local_eval[old['task_id']]['answer']
                inputs={'office_public_input_version':VERSION,'candidate_files':listed('source_files'),
                    'source_hints':listed('source_docs'),'source_pages':[],
                    'data_status':'legacy_local_val_by_external_uid_internal_only_no_external_pages'}
                origin={'data_status':inputs['data_status'],'legacy_task_digest':digest(old)}
            else:
                record=rows[uid];goal=record['question'];answer=record['ground_truth']
                inputs,origin=public_input(record)
                origin['data_status']='external_complete_public_prompt_and_evaluator_answer'
            task={'task_id':'officeqa:'+uid,'physical_key':digest([VERSION,uid,goal,inputs]),
                  'goal':goal,'inputs':inputs,'split':split}
            tasks.append(task);answers[task['task_id']]={'id':uid,'answer':answer}
            provenance[uid]=origin
        files[split+'.json']={'schema':'empirical.tasks.v1','benchmark':'officeqa','input_version':VERSION,
                             'split_seed':42,'tasks':tasks}
        files['evaluator_records_'+split+'.json']=answers
    files['input_provenance.json']=provenance
    output.mkdir(parents=True)
    for name,value in files.items():(output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    manifest={'version':VERSION,'source_seeds':[42,43,44],'cross_seed_inputs_and_answers_identical':True,
        'counts':{s:len(ids) for s,ids in splits.items()},'external_complete_splits':['train','test'],
        'val_policy':'User approved existing local Val matched by external IDs, internal checks only; not external-page parity or paper result.',
        'corpus_policy':'Candidate path basenames mapped to existing authorized corpus. Complete external transformed files were not supplied; corpus byte parity is unverified.',
        'source_members':{str(p):snapshot[3] for p,snapshot in zip(archives,snapshots)},
        'legacy_val_files':{str(legacy/name):sha((legacy/name).read_bytes()) for name in ('val.json','evaluator_records_val.json')},
        'files':{name:sha((output/name).read_bytes()) for name in files}}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'resources':str(output),'counts':manifest['counts'],'manifest_sha256':sha((output/'manifest.json').read_bytes()),'model_http':0}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--archives',nargs=3,required=True)
    ap.add_argument('--local-resources',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();materialize(args.archives,args.local_resources,args.output)
