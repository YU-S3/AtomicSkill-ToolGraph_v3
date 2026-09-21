# Seed43 final report recovery (2026-09-21)

## Confirmed cause

Seed43 completed all 120 training tasks (109 won, 11 unsuccessful), then stopped
in `validate_formal_usage -> trace_to_row -> _r4_learning_metrics`. The execution
layer legitimately records `budget_exhausted`, but the reporter's final-outcome
allowlist omitted it. The original train45 Trace
`trace_b3fe5404b5f84853ab2c905f7fce7248` contains five such records: one request
boundary and four before-session boundaries. This is not a provider/network
failure and does not require replaying the completed training tasks.

## Fix and validation

- Add only `budget_exhausted` to the report allowlist; retain all record/type/count
  reconciliation checks. Do not classify it as success, no_tool or aborted.
- Extend the real budget-retention regression through `trace_to_row`, asserting
  accepted budget outcomes and continued rejection of unknown outcomes.
- Full suite: **1550 passed in 55.71s**; focused learning/budget suite: 33 passed.
- Seed43 real-data check: **120 completed tasks, 143 Trace files**, all accepted
  by the corrected `validate_formal_usage`; no API request or training started.
- Shell syntax and diff whitespace checks passed.

## Exact historical execution boundary

Do not edit or pull into the historical seed43 checkout. The new standalone
`scripts/finalize_budget_report.py` checks its complete source inventory against
the original execution manifest, configuration identity, all tasks completed,
final-maintenance evidence, no pending checkpoint/attempt, and the existing
provider capability gate. It permits an AST-verified overlay of exactly one
report function with exactly the outcome-set addition above. All other source
changes are rejected. Historical hash checks are not disabled or spoofed.

This is a **new reporting implementation over unchanged historical execution**,
not a claim that the old reporter produced the repaired report. Before finalizing,
the full run is copied to a timestamped sibling backup; a
`report_only_recovery.json` receipt records the original code/report hashes,
corrected report and wrapper hashes, and all source Trace hashes. The original
runner then executes its normal completed-task resume/final maintenance/report/
freeze path. Existing tasks are completed and not re-executed; normal final
maintenance may run according to the original runner's rules. Original Trace
hashes are checked again after successful finalization. No original manifest,
provider certificate or task outcome is rewritten to conceal a code change.

`scripts/resume_seed43_report.sh` holds the existing lock under `runs/`, performs
this finalization when no frozen snapshot exists, and starts the original
seed43 frozen-test runner only on success. A later invocation resumes an existing
test rather than creating a duplicate. The script touches neither seed42 nor
seed44. Short usage below does not itself imply that a formal process was started
by the assistant.

```bash
bash /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/scripts/resume_seed43_report.sh
```

Logs under `/home/yangchengyu/asg_r1021_parallel_18EnMl/seed43/`:
`report_recovery_launcher.log`, `train_report_resume.log`, then `test.log`.
The earlier user-requested exclusion of unpersisted crash usage remains documented
in `manual_interruption_recovery.json`; this report fix does not delete any usage.
