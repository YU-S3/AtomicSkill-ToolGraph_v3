# B4 offline replay recovery

The three formal seed lanes had all 134 Test checkpoints, but two report
generators failed while resetting ALFWorld. Replaying the first saved task
from three seeds concurrently reproduced `pop from empty list` inside
TatSu's mutable parser stacks, called by the process-global TextWorld PDDL
`_PARSER`. The same three real replays pass after protecting the entire
StrictTaskEvaluator environment lifetime with a process-local lock. This
does not serialize model workers or alter any ALFWorld action/validator rule.

Completed experiments can regenerate reports without invoking an agent:

```bash
PYTHONPATH="$PWD/src:$PWD" /home/yangchengyu/asg_alfworld_venv/bin/python \
  -m experiments.baselines.b4_embodiskill.repair_reports \
  --campaign /absolute/path/to/existing/b4_campaign
```

The command requires all planned Train, revision, Validation and Test
checkpoints and validates task identity, physical gamefile hashes, attempt
results, state digests, selected best snapshot and Frozen integrity. Missing
evidence stops report repair; there is no fallback that starts training or
generates actions. The report-only controller refuses worker operations.

Two independent replay processes can repair seed reports concurrently.
Already completed reports are retained. New reports use the existing
report-generation functions and still require replayed won/action count to
match the recorded episode. No model calls or API key are needed. Failure
reports are backed up in `report_recovery/<timestamp>`; provider evidence,
checkpoints, selection and Frozen files are fingerprinted before and after.
`paper_report.json`, `campaign_summary.json` and `REPORT.md` are finalized
only after all three seed reports and the original-evidence check pass.

When another experiment is already running, use an isolated checkout for
this repair. Do not change that running experiment's source commit or code
hash. The new report metadata separately records its reporting commit/hash;
the original experimental provenance remains intact.
