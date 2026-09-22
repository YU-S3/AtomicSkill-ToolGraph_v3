"""Explicit preparation and verification of the authored-bank release."""
import argparse
import json
from pathlib import Path
from atomic_skillgraph.system import load_config
from atomic_skillgraph.deployment.release_protocol import ReleaseSpec,PreparedRelease,FrozenRelease
from atomic_skillgraph.deployment.bank_release import prepare_release,verify_release,freeze_release,make_configs

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    prepare=sub.add_parser('prepare')
    prepare.add_argument('--input-root',type=Path,required=True)
    prepare.add_argument('--output-root',type=Path,required=True)
    prepare.add_argument('--seeds',nargs='+',type=int,choices=(42,43,44),required=True)
    prepare.add_argument('--base-config-template',required=True)
    prepare.add_argument('--resume',action='store_true')
    for name in ('verify','freeze','make-configs'):
        command=sub.add_parser(name)
        command.add_argument('--output-root',type=Path,required=True)
        command.add_argument('--seeds',nargs='+',type=int,choices=(42,43,44),required=True)
        if name=='make-configs':
            command.add_argument('--profile',choices=('current','lean'),required=True)
            command.add_argument('--repeats',nargs='+',required=True)
    args=parser.parse_args(argv)
    args.output_root=args.output_root.expanduser().resolve()
    if args.command=='make-configs':
        repeats={int(k):int(v) for k,v in (r.split(':') for r in args.repeats)}
        if repeats!={42:3,43:1,44:1} or set(args.seeds)!={42,43,44}:parser.error('release matrix must be 42:3 43:1 44:1')
        releases=[]
        for seed in args.seeds:
            frozen=args.output_root/f'seed{seed}'/'frozen'
            manifest=json.loads((frozen/'release_manifest.json').read_text())
            releases.append(FrozenRelease(frozen,seed,manifest['knowledge_digest']))
        print(json.dumps(make_configs(releases,args.profile,repeats)),flush=True)
        return 0
    for seed in args.seeds:
        prepared=PreparedRelease(args.output_root/f'seed{seed}',seed)
        if args.command=='prepare':
            prepared=prepare_release(ReleaseSpec(seed,args.input_root/f'seed{seed}_edited_bank.zip',
                prepared.root,load_config(args.base_config_template.format(seed=seed)),args.resume))
        else:
            checks=verify_release(prepared)
            if args.command=='freeze':freeze_release(prepared,checks)
        print(json.dumps({'seed':seed,'command':args.command,'root':str(prepared.root)}),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
