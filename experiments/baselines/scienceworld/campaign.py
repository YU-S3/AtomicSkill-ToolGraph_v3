"""Run existing baseline lanes with bounded, explicit lane concurrency."""
import argparse
import concurrent.futures
from pathlib import Path
import subprocess
import sys
from .run import METHODS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,43,44],choices=[42,43,44])
    parser.add_argument('--methods',nargs='+',default=list(METHODS),choices=METHODS)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--workers',type=int,default=1,choices=[1,2,3])
    args=parser.parse_args()
    if len(set(args.seeds))!=len(args.seeds) or len(set(args.methods))!=len(args.methods):
        parser.error('Duplicate method/seed lane')
    root=Path(args.root).expanduser().resolve()
    root.mkdir(parents=True,exist_ok=True)
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
            with log.open('a' if args.resume else 'x') as handle:
                subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,check=True)
        print(f'Completed {method} seed{seed}',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lane,[(m,s) for m in args.methods for s in args.seeds]))


if __name__=='__main__':main()
