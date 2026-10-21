# OfficeQA HumanStrong V13: final-display directions

V12's completed 12-case diagnosis scored off 5/12, short rules 9/12, source calculators 7/12, and typed submission 6/12. The short-rule arm gained four with no concurrent paired losses. Three gains were final-display differences; one selected the continued-panel borrowing total instead of the public-debt-securities component. Both arms used the same log-trimming formula on that task. These are development-selected cases, not general accuracy estimates.

This second bounded comparison keeps that short-rule arm intact. It compares:

| Profile | Active rules | Program |
| --- | --- | --- |
| off | none | none |
| rules | the V12 short rules | none |
| precise | same rules plus one final-display atom | none |
| display | same as precise | one scalar formatter, only after a real numeric result exists |

The final-display atom is included before finish-only capacity planning, as well as in the normal solve prompt. It preserves requested trailing zeros and separates numeric scale from unrequested word suffixes. Explicit output templates take priority.

The scalar Program reads an actual numeric result reference, accepts explicit source/output scales, converts at most once, rounds, and submits directly. It does not require the producer to emit a new object format. Its receipt distinguishes caller-declared units from verified numeric conversion; it does not certify data selection or formula semantics. The broad source-calculator menu is absent from this Bank.

All four profiles keep the same approved public inputs, DeepSeek configuration, native tools, 216,576-token task limit, 48-call limit, scorer, and mechanical result-reference directory. The selection remains the same 12 development Train tasks. This is human Bank development testing; no extra Ours Train subdivision or natural learning is introduced.

The second round is capped at 48 executions / 10,395,648 tokens. Completed V12 cumulative use and reservations are 175,570,015 tokens; the campaign ceiling remains 262M. Historical charges and unknown reservations carry forward unchanged. There is no automatic repeat.

Selection requires positive gain and no losses against concurrent off. A new precise/display profile must also preserve every success of the concurrent short-rule arm. Qualifying profiles rank by correct count, then token cost; short rules remain eligible. After manual trace review, the chosen profile can be frozen for one complete Test102/off pair. Test must not edit the Bank or select per-task arms.

Build with `PYTHONPATH=src python resources/office_humanstrong_v13/tools/build_bundle.py`. The runner is `experiments/run_office_humanstrong_v13.py`.
