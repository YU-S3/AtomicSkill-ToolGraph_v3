"""Read-only status for the bounded three-arm V8 experiment."""
from pathlib import Path
import run_experiment as run

s=run.status()
pid=int((run.ROOT/'pid').read_text()) if (run.ROOT/'pid').exists() else None
proc=Path('/proc')/str(pid)/'cmdline'
alive=bool(pid and proc.exists() and str(run.HERE/'run_experiment.py') in proc.read_text())
print(f"SearchQA V8 | {s['at']} | state={s['status']} | process_alive={alive} | pid={pid}")
for arm,row in s['arms'].items():
    n=row['completed']
    em=f"{row['correct']/n:.2%}" if n else '-'
    print(f"{arm:20} {n:3}/{row['total']}  correct={row['correct']:3}  EM={em:>7}"
          f"  tokens={row['tokens']:,}  program_ok={row['program_ok']}"
          f"  fields_delivered={row['fields_delivered']}  matched_fields={row['matched_fields']}")
print(f"New measured tokens={s['campaign_tokens']:,}/4,000,000; HTTP={s['campaign_http']}/720;"
      f" in_flight_or_unmetered={s['inflight_or_unmetered']}")
current=run.ROOT/run.STAGE/'current.json'
if current.exists():print('Current:',run.r.read(current))
result=run.ROOT/run.STAGE/'result.json'
if result.exists() and run.r.read(result).get('error'):print('STOP ERROR:',run.r.read(result)['error'])
