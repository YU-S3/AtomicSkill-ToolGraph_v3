"""Readonly authored-reference evaluation, independent of all learned Train lanes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from atomic_skillgraph.core.serialization import atomic_write_json
from atomic_skillgraph.deployment.train_bank_compiler import bank_digest
from .scienceworld_config import make_config
from .scienceworld_campaign import completed


def run(root,bank,seed,resume=False):
    import fcntl
    root,bank=Path(root).resolve(),Path(bank).resolve()
    metadata=json.loads((bank/'freeze_manifest.json').read_text())
    if metadata.get('experiment_kind')!='authored_reference':
        raise ValueError('This launcher only accepts the separately published authored reference')
    digest=metadata['knowledge_digest']
    if bank_digest(bank)!=digest:raise ValueError('Authored Frozen digest changed')
    lane=root/f'seed{seed}';lane.mkdir(parents=True,exist_ok=True)
    with (lane/'.lane.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for repeat in range(1,4 if seed==42 else 2):
            stop=lane/'STOP_AFTER_TASK'
            if stop.exists():return 75
            config=make_config(root,seed=seed,phase='test',repetition=repeat,frozen=bank,authored_reference=True)
            if completed(config):
                if not resume:raise FileExistsError('Use --resume for existing reference evaluation')
                continue
            output=lane/f'test_repeat{repeat}'
            command=[sys.executable,'-m','experiments.run_scienceworld','--config',str(config),'--stop-file',str(stop)]
            if (output/'attempt_history').exists():
                if not resume:raise FileExistsError(output)
                command+=['--resume']
            with (lane/f'test_repeat{repeat}.log').open('a' if resume else 'x') as log:
                code=subprocess.call(command,stdout=log,stderr=subprocess.STDOUT)
            if code:return code
            if not completed(config) or bank_digest(bank)!=digest:
                raise RuntimeError('Reference Test completion/digest validation failed')
        atomic_write_json(lane/'completion.json',{'completed':True,'experiment_kind':'authored_reference',
            'seed':seed,'test_repetitions':3 if seed==42 else 1,'frozen_digest':digest})
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--bank',required=True)
    p.add_argument('--seeds',nargs='+',type=int,choices=(42,43,44),default=[42,43,44])
    p.add_argument('--workers',type=int,choices=(1,2,3),default=3);p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if len(set(a.seeds))!=len(a.seeds):p.error('Duplicate seed')
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        codes=list(pool.map(lambda seed:run(a.root,a.bank,seed,a.resume),a.seeds))
    raise SystemExit(max(codes))
