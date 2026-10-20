"""Launch a frozen authorized stage once; resume requires explicit --resume."""
import argparse
import fcntl
import os
import subprocess
import sys
import time
import campaign as c

parser = argparse.ArgumentParser()
parser.add_argument('stage')
parser.add_argument('--resume', action='store_true')
args = parser.parse_args()
root = c.CAMPAIGN / args.stage
with (c.CAMPAIGN / 'launch.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    c.check(args.stage)
    assert c.r.read(c.HERE / 'offline_acceptance.json')['driver_sha256'] == c.r.sha(c.__file__)
    if (root / 'pid').exists():
        try:
            os.kill(int((root / 'pid').read_text()), 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('Process still exists; inspect before relaunch')
        assert args.resume
    assert not (root / 'result.json').exists() or args.resume
    cmd = [sys.executable, '-B', '-u', str(c.HERE / 'campaign.py'), 'run', '--stage', args.stage]
    with (root / 'run.log').open('ab') as log:
        child = subprocess.Popen(cmd, cwd=c.r.SOURCE, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (root / 'pid').write_text(str(child.pid) + '\n')
    c.r.write_json(root / ('resume_launch.json' if args.resume else 'launch.json'), {'pid': child.pid, 'at': c.r.utc(), 'command': cmd,
        'plan_sha256': c.r.read(root / 'plan_identity.json')['sha256'], 'driver_sha256': c.r.sha(c.__file__)})
    time.sleep(2)
    assert child.poll() is None, 'Inspect run.log'
    print({'started': args.stage, 'pid': child.pid, 'root': str(root)})
