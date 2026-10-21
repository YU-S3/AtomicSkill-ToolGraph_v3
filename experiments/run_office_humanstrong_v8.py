"""One sparse Bank/NoSkill development comparison on applicable Test questions."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from filelock import FileLock
import run_office_humanstrong_v6 as paired

common = paired.common
PARENT = '/home/yangchengyu/officeqa_humanstrong_v7_val24_pairs_seed42_20261021'
RESOURCES = '/home/yangchengyu/officeqa_external_minimax_resources_20261020_v1'


def bind_runner():
    paired.PACKAGE = common.SOURCE/'resources/office_humanstrong_v8'
    paired.VERSION = 'office.humanstrong.v8.targeted-test-pairs.20261021'
    paired.AUTHORIZATION = (
        '2026-10-21 user requested short independent principles, proportionate Programs and real tests on '
        'applicable Test questions, retaining beneficial rules and removing harmful coverage. '
        'One fresh paired development observation: up to two public-text-selected questions per principle family '
        'plus four uncovered regression controls. No gold or previous score filtering. '
        'Carry forward settled V7 and all ancestral costs within the existing 190M cumulative ceiling. '
        'No training, full Test rerun or automatic repetition. This is development-exposed selection, not final evaluation.')
    paired.bind_runner()


def select_tasks(tasks):
    rank = lambda t: hashlib.sha256(('42:'+t['task_id']).encode()).hexdigest()
    chosen, groups = [], []
    for spec in common.frontier.SPECS:
        eligible = [t for t in sorted(tasks, key=rank)
                    if t not in chosen and spec['title'] in [s['title'] for s in common.frontier.matching(t['goal'])]]
        if not eligible: raise ValueError('No public matches: '+spec['title'])
        batch = eligible[:2]; chosen.extend(batch)
        groups.append({'principle': spec['title'], 'task_ids': [t['task_id'] for t in batch]})
    controls = [t for t in sorted(tasks, key=rank) if not common.frontier.matching(t['goal'])][:4]
    if len(controls) != 4: raise ValueError('Four uncovered controls required')
    chosen.extend(controls)
    selected = [t for t in tasks if t in chosen]
    return selected, {
        'rule': 'Up to two per principle in principle order, ranked by SHA256(42:task_id), without duplicates; '
                'four zero-principle controls by the same rank; execution retains canonical task order.',
        'groups': groups, 'control_ids': [t['task_id'] for t in controls],
        'selected_principles': {t['task_id']: [s['title'] for s in common.frontier.matching(t['goal'])] for t in selected},
        'retention_rule': 'A family is provisional only when paired wins exceed paired losses; '
                          'ties are inconclusive and are not promoted. Remove a family with more losses than wins. '
                          'Report controls and all individual losses. No claim of independent Test accuracy.',
        'overall_gate': 'Bank must exceed paired NoSkill correct count, with no net loss on uncovered controls.'}


def prepare(root, base=PARENT, resources=RESOURCES, config_run=PARENT):
    return paired.prepare(root, base, resources, config_run, split='test', selector=select_tasks)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['prepare', 'check', 'launch', 'run', 'status'])
    ap.add_argument('--root', required=True)
    ap.add_argument('--base-run', default=PARENT)
    ap.add_argument('--config-run', default=PARENT)
    ap.add_argument('--resources', default=RESOURCES)
    ap.add_argument('--env-file', default='/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env')
    args = ap.parse_args(); root = Path(args.root).resolve(); bind_runner()
    if args.action == 'prepare': prepare(root, args.base_run, args.resources, args.config_run)
    elif args.action == 'check': common.verify(root); print('Identity/billing check passed; zero HTTP.')
    elif args.action == 'status': print(json.dumps(common.status(root), ensure_ascii=False, indent=2))
    elif args.action == 'run': sys.exit(common.execute(root, args.env_file))
    else:
        common.verify(root)
        with FileLock(str(root/'.launch.lock'), timeout=0):
            if common.read(root/'launch_status.json')['status'] != 'prepared': raise ValueError('Already launched')
            with (root/'run.log').open('x') as log:
                child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'run',
                    '--root', str(root), '--env-file', args.env_file], cwd=common.SOURCE,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            common.save_json(root/'launch_status.json', {'status': 'launching', 'pid': child.pid, 'started_at': common.utc()})
            (root/'run.pid').write_text(str(child.pid)+'\n')
            print(json.dumps({'pid': child.pid, 'root': str(root)}))
