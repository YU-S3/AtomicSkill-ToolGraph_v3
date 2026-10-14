"""P4 original Train-trace learning and new Val selection; Test requires separate review."""
import argparse
import json
from pathlib import Path

from ..empirical.math_capability import P4
from . import run_math_p3 as runner


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='stage',required=True)
    p=sub.add_parser('prepare')
    for key in ('source','ledger-run','datasets','output'):p.add_argument('--'+key,required=True)
    for stage in ('check','status','execute','test'):
        p=sub.add_parser(stage);p.add_argument('--output',required=True)
        if stage in {'execute','test'}:p.add_argument('--env-file',required=True)
        if stage=='test':p.add_argument('--reviewed-p4-val',action='store_true')
    args=parser.parse_args()
    if args.stage=='prepare':result=runner.prepare(args.source,args.ledger_run,args.datasets,args.output,version=P4)
    else:
        if runner.batch.read(Path(args.output)/'plan.json')['schema']!=P4:raise ValueError('P4 requires its own plan, not old --resume')
        result=runner.status(args.output) if args.stage=='status' else runner.check(args.output) if args.stage=='check' else runner.execute(
            args.output,args.env_file,test=args.stage=='test',reviewed=getattr(args,'reviewed_p4_val',False))
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
