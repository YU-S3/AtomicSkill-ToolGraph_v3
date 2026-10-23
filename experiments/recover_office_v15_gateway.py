"""One reviewed V15 gateway continuation, preserving all task and campaign caps."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from filelock import FileLock
import run_office_humanstrong_v15 as v15
import recover_office_crossmodel_stream as recovery

BASE=Path('/home/yangchengyu/officeqa_humanstrong_v15_six_model_probe12_seed42_20261024')
AUTHORIZATION=(' Under the standing user authorization to repair failures and continue the specified paid experiments, '
    'resume the reviewed 2026-10-24 V15 gateway interruption once. Preserve the full unknown reservation and '
    'all prior costs, results and completed actions. Keep the original per-task caps, 72 executions, '
    '17,794,560 observation allowance and cumulative 270M ceiling. No automatic repeated recovery.')


def bind():
    v15.bind_runner()
    recovery.runner=SimpleNamespace(common=v15.common,MODELS=v15.MODELS,CEILING=270000000,
        execute=v15.previous.previous.execute,status=v15.previous.previous.status)
    recovery.c=v15.common


def prepare(root):
    return recovery.prepare(root,BASE,failure_kind='gateway_timeout',authorization=AUTHORIZATION,
        source_changes=['Restore finished sandbox computation only when explicit receipts prove no published or deleted files; replay no completed tool actions.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','check','launch','run','status'));parser.add_argument('--root',required=True)
    parser.add_argument('--env-file',default='/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env')
    args=parser.parse_args();bind();c=v15.common;root=Path(args.root).resolve()
    if args.action=='prepare':prepare(root)
    elif args.action=='check':c.verify(root);print('Continuation identity, unchanged Bank and bounded ledger verified; zero HTTP.')
    elif args.action=='status':print(json.dumps(recovery.runner.status(root),ensure_ascii=False,indent=2))
    elif args.action=='run':sys.exit(recovery.runner.execute(root,args.env_file))
    else:
        c.verify(root)
        with FileLock(str(root/'.launch.lock'),timeout=0):
            if c.read(root/'launch_status.json')['status']!='prepared':raise ValueError('Already launched')
            with (root/'run.log').open('x') as log:
                child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'run','--root',str(root),'--env-file',args.env_file],
                    cwd=c.SOURCE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            c.save_json(root/'launch_status.json',{'status':'launching','pid':child.pid,'started_at':c.utc()})
            print(json.dumps({'pid':child.pid,'root':str(root)}))
