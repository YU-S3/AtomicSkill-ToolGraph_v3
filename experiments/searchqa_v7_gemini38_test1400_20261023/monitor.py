"""Read-only Gemini SearchQA progress; no model requests."""
from collections import Counter
import run_gemini as run

value = run.status()
arm = value['arms'][run.ARM]
em = f"{arm['correct']/arm['completed']:.2%}" if arm['completed'] else '-'
print(f"Gemini-3.8-flash + V7: {value['status']} alive={value['process_alive']}")
print(f"Test {arm['completed']}/1400 correct={arm['correct']} EM={em}")
print(f"Tokens={value['campaign_tokens']:,} HTTP={value['campaign_http']} Program={arm['program_ok']}/{arm['program_calls']}")
ledger = run.r.read(run.ROOT/'budget.json')
print('Request states:',dict(Counter(a['status'] for a in ledger['attempts'].values())))
print('In-flight / unknown reservations:',value['unknown_reserved_tokens'],'tokens')
if value['current']:print('Current:',value['current'])
if value['error']:print('Error:',value['error'])
print('UTC:',value['at'])
