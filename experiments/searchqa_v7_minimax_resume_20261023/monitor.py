"""Read-only combined progress for the completed Qwen run and MiniMax continuation."""
from collections import Counter
import continue_minimax as run

r = run.r
qwen = r.read(run.PARENT.parent / 'qwen38_flash' / run.STAGE / 'result.json')
q = qwen['arms'][run.ARM]
print(f"Qwen: {qwen['status']} {q['completed']}/1400 correct={q['correct']} EM={q['correct']/q['completed']:.2%} tokens={qwen['campaign_tokens']:,}")
v = run.c.status(run.STAGE)
p = v['arms'][run.ARM]
ledger = r.read(run.ROOT / 'budget.json')
inherited = r.read(run.ROOT / 'inherited_budget.json')
added = [x for k,x in ledger['attempts'].items() if k not in inherited['attempts']]
print(f"MiniMax: {v['status']} alive={run.alive(run.ROOT)} {p['completed']}/1400 "
      f"correct={p['correct']} EM={p['correct']/p['completed']:.2%} tokens={v['campaign_tokens']:,} "
      f"HTTP={v['campaign_http']} Program={p['program_ok']}/{p['program_calls']}")
print(f"  Preserved completed={p['inherited_completed']}; new completed={p['new_completed']}; new request states={dict(Counter(x['status'] for x in added))}")
print('  Historical unknown reservation=77,459 tokens (not measured usage); cumulative limit=8,000,000 tokens / 3,000 HTTP')
current = run.ROOT / run.STAGE / 'current.json'
if current.exists(): print('  current:', r.read(current))
result = run.ROOT / run.STAGE / 'result.json'
if result.exists() and r.read(result).get('error'): print('  error:', r.read(result)['error'])
print('UTC:', r.utc())
