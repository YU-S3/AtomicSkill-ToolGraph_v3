"""Optional *real Docker* smoke on three pure Programs; zero model HTTP."""
import argparse,json,sys
from pathlib import Path

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--repo-root',required=True)
    a.add_argument('--bank-dir',required=True)
    a.add_argument('--config',required=True)
    args=a.parse_args()
    sys.path.insert(0,str(Path(args.repo_root).resolve()/'src'))
    from atomic_skillgraph.empirical.bank import Bank
    from atomic_skillgraph.empirical.contracts import PublicTask
    from atomic_skillgraph.empirical.task_context import TaskContext
    from atomic_skillgraph.empirical.program_worker import ProgramWorker
    from atomic_skillgraph.empirical.math_program_probe import PureBroker
    from atomic_skillgraph.harness.benchmarks import AnswerAdapter
    cfg=json.loads(Path(args.config).read_text())
    bank=Bank(args.bank_dir,readonly=True,seed=cfg['experiment'].get('seed',42),environment=cfg['program_environment'])
    cases={
        'period_ledger.py':{'required_periods':['FY2001'],'observations':[
            {'period':'FY2001','value':'12','unit':'millions','source_ref':{'result_id':'synthetic-smoke','path':['data','text']}}]},
        'treasury_transform.py':{'operation':'price_32nds','quote':'99.27','quote_basis':'thirty_seconds','decimals':5},
        'series_metrics.py':{'operation':'difference','from_value':'9','to_value':'14','absolute':False,'unit':'million','decimals':0}}
    try:
        adapter=AnswerAdapter('officeqa',{'worker-smoke':{'answer':'0'}})
        adapter.reset(PublicTask('worker-smoke','worker-smoke','synthetic internal Program smoke'))
        worker=ProgramWorker(cfg.get('program_worker',{}));reports=[]
        for p in bank.all('program'):
            name=Path(p['source_file']).name
            if name not in cases:continue
            broker=PureBroker(adapter,1,context=TaskContext({}))
            result=worker.execute(p,cases[name],broker)
            reports.append({'program':name,'status':result['status'],'output_contract_status':result.get('output_contract_status'),
                'detail':result.get('detail'),'outputs':result.get('outputs')})
            if result['status']!='ok':raise RuntimeError('Docker Program smoke failed: '+name+' '+str(result.get('detail')))
        assert len(reports)==3
        print(json.dumps({'docker_smoke':'passed','model_http':0,'reports':reports},indent=2))
    finally:bank.close()
if __name__=='__main__':main()
