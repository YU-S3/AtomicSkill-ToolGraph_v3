#!/usr/bin/env bash
set -euo pipefail
/home/yangchengyu/asg_alfworld_venv/bin/python -B - <<'PY'
import json, os, sys
sys.path.insert(0, '/mnt/d/T3S_exp/deliveries/searchqa_glm53_refusal_resume_20261020')
import campaign as c
root = c.CAMPAIGN / 'fulltest1400'
value = c.status('fulltest1400')
pid = int((root / 'pid').read_text()) if (root / 'pid').exists() else None
try:
    if pid is None: raise ProcessLookupError()
    os.kill(pid, 0)
    value['process_alive'] = True
except ProcessLookupError:
    value['process_alive'] = False
value['pid'] = pid
value['current'] = c.r.read(root / 'current.json') if (root / 'current.json').exists() else None
value['error'] = c.r.read(root / 'result.json').get('error') if (root / 'result.json').exists() else None
print(json.dumps(value, ensure_ascii=False, indent=2))
PY
