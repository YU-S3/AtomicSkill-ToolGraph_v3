"""Import this Bank and freeze aligned split configs; performs no model requests."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parents[1]/'src'))


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def prepare(base_config,resources,output):
    root=Path(output).resolve()
    if root.exists():raise ValueError('Use a new independent runtime directory')
    config=load('office_v2_config',ROOT/'integration/configuration.py')
    frontier=load('office_v2_frontier',ROOT/'integration/frontier.py')
    base=json.loads(Path(base_config).read_text())
    def reject_secrets(value):
        if isinstance(value,dict):
            for k,v in value.items():
                if k.lower() in {'api_key','access_token','authorization'} and v:
                    raise ValueError('Use environment key names, not literal credentials in the config')
                reject_secrets(v)
        elif isinstance(value,list):
            for v in value:reject_secrets(v)
    reject_secrets(base)
    configs={split:config.configure(base,resources,split) for split in ('train','val','test')}
    cfg=configs['train'];bundle=json.loads((ROOT/'bank/bank.json').read_text())
    frontier.native_import(bundle,root/'bank',cfg['program_environment'],cfg['experiment'].get('officeqa_method_version'),42)
    for split,cfg in configs.items():
        cfg['data_dir']=str(root/'bank');cfg['experiment'].update(output_dir=str(root/split),runtime_mode='frozen',seed=42)
        (root/(split+'_config.json')).write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
    files={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
           for p in root.rglob('*') if p.is_file()}
    binding={'bank_version':frontier.VERSION,'input_manifest':str(Path(resources).resolve()/'manifest.json'),
        'runtime_bridge':str(ROOT/'integration/frontier.py'),'evaluation_entry':str(ROOT/'integration/evaluate.py'),
        'default_arm':'manual','model_http':0,'status':'prepared_only_not_a_paid_run',
        'billing':'Execution requires an explicit cumulative BudgetGovernor from the campaign ledger; this preparation allocates no new budget.',
        'files_sha256':files}
    (root/'runtime_binding.json').write_text(json.dumps(binding,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'runtime':str(root),'status':binding['status'],'model_http':0}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base-config',required=True);p.add_argument('--resources',required=True)
    p.add_argument('--output',required=True);a=p.parse_args();prepare(a.base_config,a.resources,a.output)
