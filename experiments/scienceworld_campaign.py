"""Independent seed lanes; Train-only compile, readonly Dev, identical Test bank.

No performance/asset-count gate. Subprocess isolation bounds concurrent JVMs and
provider requests. Stop files are checked at task boundaries, never by killing a
paid request. Existing output requires explicit --resume.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

from atomic_skillgraph.core.serialization import atomic_write_json
from atomic_skillgraph.deployment.train_bank_compiler import bank_digest, compile_train
from atomic_skillgraph.system import load_config
from .scienceworld_config import make_config


def completed(config):
    cfg = load_config(config)
    output = Path(cfg['experiment']['output_dir'])
    database = Path(cfg['data_dir']) / 'state.sqlite3' if cfg['experiment']['phase']=='train' else output/'run_state.sqlite3'
    if not database.exists():
        return False
    with sqlite3.connect(f'file:{database.as_posix()}?mode=ro', uri=True) as db:
        row = db.execute('SELECT state FROM run_manifests WHERE run_id=?', (cfg['experiment']['name'],)).fetchone()
    if not row or row[0] != 'completed':
        return False
    summary = json.loads((output/'summary.json').read_text())
    if summary['completed'] != summary['expected']:
        raise RuntimeError('Completed run has inconsistent task accounting')
    if bank_digest(cfg['data_dir']) != summary['knowledge_digest']:
        raise RuntimeError('Completed run bank changed')
    return True


def run_lane(root, seed, *, resume=False, diagnostic=False):
    import fcntl
    lane=Path(root).expanduser().resolve()/f'seed{seed}'
    lane.mkdir(parents=True,exist_ok=True)
    with (lane/'.lane.lock').open('a+') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Seed {seed} already has a running supervisor') from None
        lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
        return _run_lane(root,seed,resume=resume,diagnostic=diagnostic)


def _run_lane(root, seed, *, resume=False, diagnostic=False):
    root = Path(root).resolve()
    lane = root/f'seed{seed}'
    lane.mkdir(parents=True, exist_ok=True)
    stop = lane/'STOP_AFTER_TASK'
    def phase(config):
        if stop.exists():
            return 75
        if completed(config):
            if not resume: raise FileExistsError('Existing phase requires --resume')
            return 0
        cfg = load_config(config)
        output = Path(cfg['experiment']['output_dir'])
        started = (output/'attempt_history').exists()
        command = [sys.executable,'-m','experiments.run_scienceworld','--config',str(config),'--stop-file',str(stop)]
        if started:
            if not resume: raise FileExistsError(output)
            command += ['--resume']
        log = lane/(output.name+'.log')
        with log.open('a' if resume else 'x') as handle:
            code = subprocess.call(command,stdout=handle,stderr=subprocess.STDOUT)
        if code == 0 and not completed(config):
            raise RuntimeError('Phase returned without a completion marker')
        return code
    train = make_config(root, seed=seed, diagnostic=diagnostic)
    code = phase(train)
    if code: return code
    if stop.exists(): return 75
    compiled = lane/'compiled'
    marker = compiled/'compiler_manifest.json'
    if compiled.exists():
        if not resume or not marker.is_file():
            raise RuntimeError('Partial/existing compiler output must be inspected; it is never silently replaced')
        report = json.loads(marker.read_text())
        if not report['completed'] or report['source_train_bank_digest'] != bank_digest(load_config(train)['data_dir']):
            raise RuntimeError('Compiler source changed')
    else:
        report = compile_train(train, compiled)
    frozen = compiled/'data_v3'
    digest = report['compiled_bank_digest']
    if bank_digest(frozen)!=digest: raise RuntimeError('Compiled Frozen digest changed')
    dev = make_config(root,seed=seed,phase='dev',frozen=frozen,diagnostic=diagnostic)
    code = phase(dev)
    if code: return code
    if bank_digest(frozen)!=digest: raise RuntimeError('Dev mutated Frozen bank')
    # Diagnostic lane is the compiler mini-chain; it never starts formal Test90.
    repeats = 0 if diagnostic else (3 if seed==42 else 1)
    for repeat in range(1,repeats+1):
        config = make_config(root,seed=seed,phase='test',repetition=repeat,frozen=frozen)
        code = phase(config)
        if code: return code
        if bank_digest(frozen)!=digest: raise RuntimeError('Test mutated Frozen bank')
    atomic_write_json(lane/'completion.json',{'completed':True,'seed':seed,'diagnostic':diagnostic,
        'frozen_digest':digest,'test_repetitions':repeats,'dev_is_readonly':True})
    return 0


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--seeds',type=int,nargs='+',choices=(42,43,44),default=[42,43,44])
    p.add_argument('--workers',type=int,choices=(1,2,3),default=1)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--diagnostic',action='store_true')
    args=p.parse_args()
    if len(set(args.seeds))!=len(args.seeds): p.error('Duplicate seed lane')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes=list(pool.map(lambda seed:run_lane(args.root,seed,resume=args.resume,diagnostic=args.diagnostic),args.seeds))
    return max(codes)

if __name__=='__main__':
    raise SystemExit(main())
