"""Actual native Bank + locked Docker smoke (zero paid model HTTP)."""
from pathlib import Path
import argparse,sys,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser()
 p.add_argument('--repo-root',required=True);p.add_argument('--bank-root',required=True)
 a=p.parse_args();sys.path.insert(0,str(Path(a.repo_root).resolve()/'src'));sys.path.insert(0,str(ROOT))
 from atomic_skillgraph.empirical.bank import Bank
 from atomic_skillgraph.empirical.contracts import PublicTask
 from atomic_skillgraph.harness.benchmarks import AnswerAdapter
 from integration.native_programmatic_runtime import docker_worker,run_programmatic_graph
 choices=[{'label':l,'text':('One of the remaining options is correct, but a stronger result can be proven.' if l=='D' else 'A mathematical condition')} for l in 'ABCDE']
 task=PublicTask('manual_fixture','manual_fixture_hash','Which public option is the meta option?',{'choices':choices},'train')
 bank=Bank(a.bank_root,readonly=True,seed=42)
 try:
  adapter=AnswerAdapter('livemath',{'manual_fixture':{}});adapter.reset(task)
  result=run_programmatic_graph(task,bank,variant='benchmark_prior',run_worker=docker_worker(adapter))
  if result['status']!='ok' or result['answer']!='D' or len(result['program_nodes'])!=2:
   raise SystemExit('Real Docker handoff did not yield expected public label')
  print(json.dumps({'status':'docker_worker_smoke_ok','graph':result['graph'],
                    'nodes':result['program_nodes'],'paid_model_requests':0},ensure_ascii=False,indent=2))
 finally:bank.close()
if __name__=='__main__':main()
