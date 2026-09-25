"""Finite real-JVM load probe; no policy/model calls or official-result claims."""
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from atomic_skillgraph.harness.scienceworld import ScienceWorldAdapter
from atomic_skillgraph.core.serialization import atomic_write_json
from experiments.scienceworld_manifest import load,task_from_entry


def run(output, workers):
    manifest=load(Path(__file__).resolve().parents[3]/'data/scienceworld_manifests/train_120.json')
    barrier=threading.Barrier(workers,timeout=180)
    def one(index):
        harness=ScienceWorldAdapter()
        try:
            entry=manifest['tasks'][index%2]
            harness.reset(task_from_entry(entry,''))
            barrier.wait()
            return {'worker':index,'task_id':entry['task_id'],
                'revision':harness.validator_channel().revision,'loaded':True}
        except BaseException:
            barrier.abort()
            raise
        finally:
            harness._close_backend()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows=list(pool.map(one,range(workers)))
    memory={line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines()}
    atomic_write_json(output,{'passed':True,'simultaneously_loaded_jvms':workers,'rows':rows,
        'memory_after':memory,'not_a_benchmark_result':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,choices=range(1,25),default=18)
    a=p.parse_args();run(a.output,a.workers)
