"""Ours-only shared empirical runner and fixed 2 Train + 1 Val adapter smoke."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import yaml

from atomic_skillgraph.empirical.contracts import PublicTask
from atomic_skillgraph.harness.registry import create_simple_harness
from .run_empirical import load_env, run, write_json


def load_tasks(path):
    return [PublicTask(**row) for row in json.loads(Path(path).read_text())['tasks']]


def run_smoke(config, dataset_root, output, benchmarks):
    dataset_root, output = Path(dataset_root).resolve(), Path(output).resolve()
    report = {'profile':config['mechanism_profile'],'method':'ours','benchmarks':{}}
    for benchmark in benchmarks:
        current = deepcopy(config)
        current['harness'] = {'adapter':benchmark,
            'evaluator_records':str(dataset_root/benchmark/'evaluator_records.json')}
        current['benchmark_profile'] = benchmark + '.skillopt.v1'
        modalities = {'text','image'} if benchmark=='docvqa' else {'text','files'} if benchmark in {'spreadsheet','officeqa'} else {'text'}
        if 'image' in modalities and 'image' not in current['llm'].get('input_modalities',['text']):
            report['benchmarks'][benchmark] = {'status':'unsupported','reason':'Locked model has no image capability', 'score':None}
            write_json(output/'adapter_smoke.json',report)
            continue
        if benchmark=='officeqa':
            current['harness']['corpus_root']=config['harness'].get('corpus_root','')
        current['runtime']['global_action_budget'] = {'officeqa':24,'spreadsheet':30}.get(benchmark,1)
        current['runtime'].pop('environment_step_budget',None)
        if benchmark=='spreadsheet':
            current['program_worker'].update(wall_timeout_seconds=120,memory_limit_mb=2048)
        summaries={}
        for split, count in [('train',2),('val',1)]:
            settings=deepcopy(current)
            settings['manifest'] = str(dataset_root/benchmark/(split+'.json'))
            settings['data_dir']=str(output/benchmark/'train'/('bank' if split=='train' else 'frozen_bank'))
            settings['experiment'].update(runtime_mode='online' if split=='train' else 'frozen',output_dir=str(output/benchmark/split),benchmark=benchmark)
            adapter=create_simple_harness(settings)
            tasks=load_tasks(dataset_root/benchmark/(split+'.json'))[:count]
            summary=run(settings,tasks,output/benchmark/split,readonly=split!='train',adapter=adapter,
                adapter_factory=lambda settings=settings:create_simple_harness(settings))
            summaries[split]={key:summary[key] for key in ['tasks','successes','total_tokens','knowledge_digest','complete']}
        report['benchmarks'][benchmark]={'status':'completed','runs':summaries}
        write_json(output/'adapter_smoke.json',report)
    report['implementation_ready']=all(row['status'] == 'completed' for row in report['benchmarks'].values())
    report['mechanism_demonstrated']=None
    report['measured_effect']=None
    write_json(output/'multibench_release.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/default.yaml')
    parser.add_argument('--datasets',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--corpus-root',required=True)
    parser.add_argument('--env-file')
    parser.add_argument('--benchmarks',nargs='+',default=['searchqa','officeqa','docvqa','livemath','spreadsheet'])
    args=parser.parse_args()
    if args.env_file: load_env(args.env_file)
    config=yaml.safe_load(Path(args.config).read_text())
    config['harness']['corpus_root']=args.corpus_root
    print(json.dumps(run_smoke(config,args.datasets,args.output,args.benchmarks)),flush=True)


if __name__=='__main__': main()
