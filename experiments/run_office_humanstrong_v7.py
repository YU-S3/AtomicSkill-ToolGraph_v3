"""One authorized V7/NoSkill Val24 observation, reusing the frozen paired runner."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from filelock import FileLock
import run_office_humanstrong_v6 as paired

common = paired.common
PARENT = '/home/yangchengyu/officeqa_humanstrong_v6_val24_pairs_seed42_20261021'


def bind_runner():
    paired.PACKAGE = common.SOURCE/'resources/office_humanstrong_v7'
    paired.VERSION = 'office.humanstrong.v7.internal-val24-pairs.20261021'
    paired.AUTHORIZATION = (
        '2026-10-21 user explicitly requested the proposed real V7 Bank/NoSkill Val24 paired acceptance. '
        'One fresh observation, identical DeepSeek, public tasks, tools, per-task budget and finalization settings. '
        'Continue the approved cumulative 190M ceiling; carry forward all V6 and ancestral charges and unknown reservations. '
        'All canonical Val24, previously approved local Val for internal development checks. '
        'No Train split, Test, learning or automatic repeats.')
    paired.bind_runner()


def prepare(root, base=PARENT, resources='/home/yangchengyu/officeqa_external_minimax_resources_20261020_v1', config_run=PARENT):
    return paired.prepare(root, base, resources, config_run)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['prepare', 'check', 'launch', 'run', 'status'])
    ap.add_argument('--root', required=True)
    ap.add_argument('--base-run', default=PARENT)
    ap.add_argument('--config-run', default=PARENT)
    ap.add_argument('--resources', default='/home/yangchengyu/officeqa_external_minimax_resources_20261020_v1')
    ap.add_argument('--env-file', default='/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env')
    args = ap.parse_args()
    root = Path(args.root).resolve()
    bind_runner()
    if args.action == 'prepare':
        prepare(root, args.base_run, args.resources, args.config_run)
    elif args.action == 'check':
        common.verify(root)
        print('Identity and billing precheck passed; zero HTTP.')
    elif args.action == 'status':
        print(json.dumps(common.status(root), ensure_ascii=False, indent=2))
    elif args.action == 'run':
        sys.exit(common.execute(root, args.env_file))
    else:
        common.verify(root)
        with FileLock(str(root/'.launch.lock'), timeout=0):
            if common.read(root/'launch_status.json')['status'] != 'prepared':
                raise ValueError('Already launched; inspect first')
            with (root/'run.log').open('x') as log:
                child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'run',
                    '--root', str(root), '--env-file', args.env_file], cwd=common.SOURCE,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            common.save_json(root/'launch_status.json', {'status': 'launching', 'pid': child.pid,
                                                        'started_at': common.utc()})
            (root/'run.pid').write_text(str(child.pid)+'\n')
            print(json.dumps({'pid': child.pid, 'root': str(root)}))
