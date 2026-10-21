# OfficeQA HumanStrong V11

Frozen, human-authored development Bank. Candidate status until the real paired run is reviewed; it is not a naturally learned Ours result.

The stopped V10 Train prefix had 62 complete pairs: Bank and Bank-off both scored 44/62. V11 targets observed selection and submission failures without changing public inputs, scoring, the model or the paired inference budget.

## Assets and execution

- 16 short cards (3,452 guidance characters total, longest 234); at most three initial cards. Across the full public Train/Test resources, their combined initial guidance is at most 686 characters. A final-display card is offered only after a real native computation.
- Eight final-answer Program profiles: CAGR, annual log growth, symmetric difference, subtraction, polynomial forecast, explicit-period statistics, absolute deviation and formatting an acquired scalar. Thirteen Implementation links, no Composite/Workflow assets.
- Calculators read acquired result references or exact Markdown table cells through `read_result`. Selectors preserve blank columns, distinguish repeated date groups and record actual quoted rows. No retyped observation values are accepted by the source-bound calculators.
- The caller chooses task semantics, observation periods, source columns and explicit display. Source checks prove literal cell fidelity, not semantic correctness. Unsupported or ambiguous inputs return `needs_input`; native computation remains available.
- A successful full-task Program uses the existing `final_answer` runtime contract to submit directly, without a subsequent model rewrite. Program calls and source reads retain the normal accounting and trace path.
- Bank-off receives zero active assets, guidance cards or Programs. Both arms keep the same public source pages, native tools, answer contract, model settings and task limits.

The revision card compares overlapping publications, respecting an explicit as-of date before preferring a later matching retrospective value. The total card distinguishes continued totals from subcomponents. Neither uses task IDs, answers or scores. Broad accuracy improvements are hypotheses to test, not facts established by the offline arithmetic fixtures.

## Build and validation

Run `PYTHONPATH=src python resources/office_humanstrong_v11/tools/build_bundle.py` in the project environment. The bundle stores content-derived IDs and Program source; legacy assets remain unchanged. Tests in `tests/test_office_humanstrong_v11.py` include source selector refusals, actual saved public pages through Docker, direct submission, both-arm request isolation, complete public-task admission and paused-parent cost carryforward.

The fresh paid test uses all 62 completed pairs in the previously stopped chronological Train prefix. All 44 old Bank-off successes and all 18 failures are included; no score-based selection and no Ours Train partition. This prefix was inspected during Bank development. No automatic Test or further repeated evaluation is launched.

The fixed cumulative authority is 262,000,000 tokens. Inherited usage/reservations: 169,104,311. The new 124 executions have a theoretical cap of 26,855,424 tokens, within that authority. Exact code, Bank hashes, per-arm settings, costs and terminal status are frozen by `experiments/run_office_humanstrong_v11.py`.
