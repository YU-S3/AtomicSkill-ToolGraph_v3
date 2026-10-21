# OfficeQA HumanStrong V12 direction comparison

A frozen human-authored development Bank with three explicit profiles. This is not a partition of Ours training and does not train on Test labels.

| Profile | Short rules | Source calculators | Typed final helper |
| --- | --- | --- | --- |
| off | no | no | no |
| rules | yes | no | no |
| selective | same | only matching operations | no |
| guarded | same plus typed-output instructions | same | only after an actual typed quantity exists |

All arms share public inputs, native tools, model settings, task limits, scoring and a mechanical directory of exact public numeric result references. The directory does not infer correct answers or units. It fixes the need to guess `data/computed` paths. Bank-off has no active Bank rules or Programs.

The 20 cards contain 3,341 guidance characters. At most four matching initial cards are selected from public goals/source structure (at most 664 characters on the current public Train/Test resources). Guarded additionally receives a 254-character typed-output card. Log trimming, median deviation and Theil T are separate atoms; no task ID or answer is used for routing.

The share-difference Program reads actual source cells and computes `(first-second)/total`; fraction and percent output scales are explicit. The typed helper reads an acquired object's `value`, `scale` and `unit`, performs an allowed conversion once and rounds at final display. It does not guess these fields or prove that the producer chose the right semantic interpretation. Existing source calculators remain frozen and reused.

Build: `PYTHONPATH=src python resources/office_humanstrong_v12/tools/build_bundle.py`.

The first paid plan compares all four profiles on 12 fixed development Train cases (48 fresh executions), selected transparently from V11 wins, regressions, unresolved failures and correct controls. IDs live only in the experiment selection manifest. Maximum 10,395,648 new tokens, inherited usage/reservations 173,708,915; the approved cumulative ceiling remains 262M. The plan includes no automatic repeats or Test execution.

Judge each direction against the concurrent off arm using wins, regressions, actual Program execution and total cost. Accuracy on this selected diagnosis is not a general benchmark estimate. The final chosen profile must be frozen before any full Test102 comparison. Original runs, Banks, paid records and unknown reservations are preserved.
