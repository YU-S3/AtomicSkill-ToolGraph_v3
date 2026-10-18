"""No-HTTP preflight of frozen DocVQA inputs and the configured image transport."""
import argparse
import json
from pathlib import Path
import subprocess

import yaml

from atomic_skillgraph.agents.image_input import measure
from atomic_skillgraph.experiments.canonical_manifest import ordered_train, sha256, verify
from atomic_skillgraph.experiments.run_empirical import code_identity
from atomic_skillgraph.experiments.run_formal import model_settings, resolved_config
from atomic_skillgraph.harness.benchmarks import image_content_parts


def check(datasets):
    spec=yaml.safe_load(Path('configs/docvqa_deepseek_v41.yaml').read_text())
    authority=Path(spec['authority']).resolve();datasets=Path(datasets).resolve()
    manifest=verify(authority)
    model=json.loads(Path(spec['model_lock']).read_text())['formal_models'][0]
    base=model_settings(yaml.safe_load(Path(spec['base_config']).read_text()),model)
    profile=json.loads(Path(spec['benchmark_profiles']).read_text())['profiles']['docvqa']
    materialization=json.loads((datasets/'materialization.json').read_text())
    assert materialization['authority_sha256']==sha256(authority/'manifest.json')
    for name,expected in materialization['files_sha256'].items():
        assert sha256(datasets/name)==expected, name
    for name,expected in materialization.get('external_files_sha256',{}).items():
        assert sha256(name)==expected, name
    code=code_identity()
    assert code['git_sha'] and not code['tracked_dirty'], 'Commit clean source before launch'
    orders={};counts={};images=[];seen=set()
    for split,count in zip(('train','val','test','reserve'),(180,22,201,131)):
        rows=json.loads((datasets/'docvqa'/f'{split}.json').read_text())['tasks']
        canonical=json.loads((authority/'docvqa'/f'{split}.json').read_text())['tasks']
        ids=[r['task_id'] for r in rows]
        assert ids==[r['task_id'] for r in canonical] and len(ids)==count
        assert not seen.intersection(ids);seen.update(ids);counts[split]=count
        if split=='train':
            # ordered_train takes objects exposing task_id; the actual task objects
            # are also used by the production campaign.
            from atomic_skillgraph.empirical.contracts import PublicTask
            tasks=[PublicTask(**r) for r in rows]
            orders={str(seed):[t.task_id for t in ordered_train(tasks,seed)] for seed in (42,43,44)}
        for row in rows:
            assert set(row['inputs'])=={'images'}, 'Unexpected model-visible DocVQA fields'
            assert len(row['inputs']['images'])==1
            payload={'messages':[{'role':'user','content':[{'type':'text','text':row['goal']},
                *image_content_parts(row['inputs']['images'])]}]}
            images.extend(measure(payload,model['image_input'])['images'])
    assert len({tuple(v) for v in orders.values()})==3
    assert all(set(v)==set(orders['42']) for v in orders.values())
    limits=base['budget']
    assert limits['token_limit']==180*limits['train_task_tokens']+223*limits['eval_task_tokens']
    assert limits['train_task_tokens']==limits['train_solve_tokens']+limits['train_learning_tokens']
    assert limits['request_limit']==(180*13+223*9)*(base['llm']['max_retries']+1)
    for split in ('train','val','test'):
        resolved_config(base,profile,'docvqa',42,split,Path('/tmp/docvqa-preflight-only'),datasets,authority,None)
    subprocess.run(['docker','run','--rm','--network','none','--read-only',profile['image_digest'],
        'python','-c','from PIL import Image; print("locked-worker-ready")'],check=True,timeout=90)
    return {'source':code,'model':model,'counts':counts,'split_hashes':manifest['benchmarks']['docvqa']['splits'],
        'authority_sha256':sha256(authority/'manifest.json'),'materialization_sha256':sha256(datasets/'materialization.json'),
        'images_checked':len(images),'maximum_image_bytes':max(i['bytes'] for i in images),
        'maximum_dimension':max(max(i['width'],i['height']) for i in images),'budgets_per_seed':limits,
        'train_order_prefixes':{k:v[:2] for k,v in orders.items()},'status':'passed','model_requests':0}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',required=True)
    parser.add_argument('--output')
    args=parser.parse_args()
    result=check(args.datasets)
    text=json.dumps(result,ensure_ascii=False,indent=2)
    if args.output:Path(args.output).write_text(text+'\n')
    print(text)
