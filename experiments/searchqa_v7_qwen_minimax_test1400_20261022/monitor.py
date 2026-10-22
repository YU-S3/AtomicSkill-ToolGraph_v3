"""Compact read-only progress; distinguish active requests from unknown billing."""
from collections import Counter
import run_models as run

for model in run.MODELS:
    value = run.monitor(model)
    root, r = run.c.CAMPAIGN, run.r
    row = value['arms'][run.ARM]
    ledger, inherited = r.read(root / 'budget.json'), r.read(root / 'inherited_budget.json')
    added = [v for k, v in ledger['attempts'].items() if k not in inherited['attempts']]
    counts = Counter(v['status'] for v in added)
    n = row['completed']
    em = f"{row['correct']/n:.2%}" if n else '-'
    print(f"{model}: {value['status']} alive={value['process_alive']} {n}/1400 "
          f"correct={row['correct']} EM={em} tokens={value['campaign_tokens']:,} "
          f"HTTP={value['campaign_http']} Program={row['program_ok']}/{row['program_calls']}")
    print(f"  new request states={dict(counts)}; inherited unknown usage entries="
          f"{sum(v['status']=='unknown_billing' for v in inherited['attempts'].values())}")
    if value['current']:
        print('  current:', value['current'])
    if value['error']:
        print('  error:', value['error'])
print('UTC:', r.utc())
