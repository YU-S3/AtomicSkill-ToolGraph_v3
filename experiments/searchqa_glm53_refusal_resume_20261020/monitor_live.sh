#!/usr/bin/env bash
set -euo pipefail
/home/yangchengyu/asg_alfworld_venv/bin/python -B - <<'PY'
import json, os, sys
sys.path.insert(0, '/mnt/d/T3S_exp/deliveries/searchqa_glm53_refusal_resume_20261020')
import campaign as c
root = c.CAMPAIGN / 'fulltest1400'
value = c.status('fulltest1400')
ledger = c.r.read(c.CAMPAIGN / 'budget.json')
allowed = ledger.get('inherited_reservation_recovery', {}).get('attempts', {})
pid = int((root / 'pid').read_text())
try:
    os.kill(pid, 0)
    alive = True
except ProcessLookupError:
    alive = False
pending = [a for a in ledger['attempts'].values() if a['status'] == 'admitted']
unresolved = [a for k, a in ledger['attempts'].items()
              if a['status'] == 'unknown_billing' and k not in allowed]
reserved = sum(a['reserved_tokens'] for k, a in ledger['attempts'].items() if k in allowed)
print('status:', value['status'], '| process_alive:', alive, '| pid:', pid)
for arm, row in value['arms'].items():
    print(arm, f"{row['completed']}/{row['total']}", 'correct:', row['correct'],
          'known_tokens:', row['tokens'], 'program_ok:', row['program_ok'])
print('SearchQA known_tokens:', value['campaign_tokens'], '| HTTP:', value['campaign_http'])
print('active_http:', len(pending) if alive else 0,
      '| unresolved_failures:', len(unresolved) + (len(pending) if not alive else 0),
      '| terminal_refusal_unknown_reserved_tokens:', reserved)
print('refusals:', value['refusals'])
print('current:', json.dumps(c.r.read(root / 'current.json'), ensure_ascii=False))
result = c.r.read(root / 'result.json')
if result.get('error'): print('error:', json.dumps(result['error'], ensure_ascii=False))
PY
