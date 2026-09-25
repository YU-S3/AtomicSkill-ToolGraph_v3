"""Run existing baseline lanes with bounded, explicit lane concurrency."""
import argparse
import concurrent.futures
from pathlib import Path
import subprocess
import sys
import os
import json
import shlex
from .run import METHODS


def credentials(path=None):
    if path:
        for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
            line=line.strip().removeprefix('export ')
            name,sep,value=line.partition('=')
            if sep and name.strip()=='MODEL_API_KEY' and not os.environ.get('MODEL_API_KEY'):
                words=shlex.split(value,comments=True)
                if len(words)!=1: raise ValueError('Invalid MODEL_API_KEY assignment in env file')
                os.environ['MODEL_API_KEY']=words[0]
    if not os.environ.get('MODEL_API_KEY','').strip():
        raise ValueError('MODEL_API_KEY is missing. Supply --env-file /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env; no lanes were started.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,43,44],choices=[42,43,44])
    parser.add_argument('--methods',nargs='+',default=list(METHODS),choices=METHODS)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--workers',type=int,default=1,choices=range(1,16))
    parser.add_argument('--episode-workers',type=int,default=1,choices=range(1,9))
    parser.add_argument('--provider-slots',type=int,default=12,choices=range(1,33))
    parser.add_argument('--environment-slots',type=int,default=18,choices=range(1,25))
    parser.add_argument('--env-file')
    args=parser.parse_args()
    if len(set(args.seeds))!=len(args.seeds) or len(set(args.methods))!=len(args.methods):
        parser.error('Duplicate method/seed lane')
    credentials(args.env_file)
    root=Path(args.root).expanduser().resolve()
    root.mkdir(parents=True,exist_ok=True)
    policy={'episode_workers':args.episode_workers,'provider_slots':args.provider_slots,
            'environment_slots':args.environment_slots,'lane_workers':args.workers}
    receipt=root/'parallel_policy.json'
    if receipt.exists() and json.loads(receipt.read_text())!=policy:
        raise ValueError('Parallel policy changed; use the original settings to resume')
    from atomic_skillgraph.core.serialization import atomic_write_json
    atomic_write_json(receipt,policy)
    os.environ.update(SW_GATE_ROOT=str(root/'gates'),SW_EPISODE_WORKERS=str(args.episode_workers),
        SW_PROVIDER_SLOTS=str(args.provider_slots),SW_ENV_SLOTS=str(args.environment_slots),
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    from .parallel import provider_gate
    provider_gate()  # Validate the cross-process identity before creating lanes.
    def lane(pair):
        import fcntl
        method,seed=pair
        command=[sys.executable,'-m','experiments.baselines.scienceworld.run','--root',str(root),
                 '--method',method,'--seed',str(seed)]
        if args.smoke:command.append('--smoke')
        if args.resume:command.append('--resume')
        log=root/f'{method}_seed{seed}.log'
        print(f'Starting {method} seed{seed}; log={log}',flush=True)
        with (root/f'.{method}_seed{seed}.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                with log.open('a' if args.resume else 'x') as handle:
                    subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,check=True)
            except subprocess.CalledProcessError as exc:
                print(f'FAILED {method} seed{seed}: exit={exc.returncode}; details={log}',flush=True)
                return {'method':method,'seed':seed,'passed':False,'log':str(log),'exit_code':exc.returncode}
        print(f'Completed {method} seed{seed}',flush=True)
        return {'method':method,'seed':seed,'passed':True,'log':str(log)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results=list(pool.map(lane,[(m,s) for m in args.methods for s in args.seeds]))
    atomic_write_json(root/'campaign_results.json',results)
    return 0 if all(r['passed'] for r in results) else 1


if __name__=='__main__':raise SystemExit(main())
