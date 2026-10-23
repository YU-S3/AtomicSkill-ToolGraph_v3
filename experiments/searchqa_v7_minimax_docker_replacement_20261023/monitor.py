"""Read-only effective results and physical ledger; does not send model requests."""
from collections import Counter
import run_replacement as run

r = run.r
v = run.status()
q = r.read(run.FIRST.parent/'qwen38_flash'/run.OLD_STAGE/'result.json')
a = q['arms'][run.ARM]
print(f"Qwen: {q['status']} {a['completed']}/1400 correct={a['correct']} EM={a['correct']/a['completed']:.2%} tokens={q['campaign_tokens']:,}")
e = v['effective']
print(f"MiniMax effective: {e['tasks']}/1400 correct={e['correct']} EM={e['em']:.2%} F1={e['f1']:.2%} tokens={e['tokens']:,} HTTP={e['http']} alive={v['alive']}")
for stage, data in v['stages'].items():
    m = data['metrics']
    expected = 200 if stage == 'replacement200' else 303
    em = f"{m['em']:.2%}" if m['tasks'] else '-'
    print(f"  {stage}: {data['status']} {m['tasks']}/{expected} correct={m['correct']} EM={em} tokens={m['tokens']:,}")
    if data['error']: print('    error:', data['error'])
    current = run.ROOT/stage/'current.json'
    if current.exists(): print('    current:', r.read(current))
ledger = r.read(run.ROOT/'budget.json')
inherited = r.read(run.ROOT/'inherited_budget.json')
states = Counter(x['status'] for k,x in ledger['attempts'].items() if k not in inherited['attempts'])
print('New HTTP states:', dict(states))
print('Excluded old 200: 1,659,185 measured tokens / 439 HTTP; immutable physical records retained.')
print(f"Physical ledger: {v['physical_metered_tokens']:,} measured tokens / {v['physical_http']} HTTP; unknown reservation={v['unknown_reserved_tokens']:,} tokens")
print('UTC:', v['at'])
