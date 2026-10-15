# LiveMath P7 text-policy experiment

Method: `livemath.effect-evolution.p7`. This is a fresh derived experiment; it
does not resume P4/P5/P6 or import their published assets. Existing Test100 has
development exposure and is never described as a new blind holdout.

The user confirmed public-only Train28/feedback12/protected20, up to three
initial candidates (one global plus two local), two editor rounds of at most
two edits, net gain >=2, regressions <=1 and no increase in empty answers.
The first implementation deliberately disables Programs and Workflow. Their
existing qualifications and runners remain unchanged. Runtime selects at most
one global and one local text policy without an LLM routing call.

The actual DeepSeek-V4-Flash tokenizer is locked by upstream revision and file
SHA in the configuration. Install `.[p7]` and materialize that public tokenizer
at the configured path before preparing. The 600-token limit measures injected
cards, not Provider reasoning. HTTP admission still uses the existing
conservative UTF-8 wire-byte bound.

Every Runtime arm uses the same public question, choices, prompt, model,
65,536 completion cap and 600-second timeout. Invalid/empty answers receive at
most one low-effort 16,384-token finish with a hashed 16-KiB reasoning tail.
There are no Runtime tools, actions or program catalogues. Successful finish
is not budget-censored. The unmodified official scorer evaluates predictions.

## Commands

Run from the **committed, clean Linux checkout** used by `prepare`:

```bash
export PYTHONPATH="$PWD/src"
PY=/home/yangchengyu/asg_alfworld_venv/bin/python
OUT=/home/yangchengyu/livemath_effect_p7_seed42_20261015
"$PY" -m atomic_skillgraph.experiments.run_math_effect_p7 prepare \
  --config configs/livemath_effect_p7_seed42.yaml \
  --source-run /home/yangchengyu/livemath_p2_formal_seed42_20261013/livemath/seed42 \
  --datasets /home/yangchengyu/main_experiment_v1_resources_cf4_r3_20261009_v1 \
  --output "$OUT"
"$PY" -m atomic_skillgraph.experiments.run_math_effect_p7 check --output "$OUT" --zero-http
"$PY" -m atomic_skillgraph.experiments.run_math_effect_p7 report --output "$OUT"
```

The phase commands are `learn`, `edit`, `gate`, `freeze`, `val`, `review`,
`test`. Execute stages explicitly in that order; a negative result ends this
method version. `edit` evaluates initial candidates and up to two limited
revisions, reusing the same paid NoSkill12 receipts. It never borrows another
phase's budget. Gate locks the selected policy before any of its responses;
afterward editing is forbidden, including after failure.

The proposed paid allowances (not authorization) are:

| Phase | Tokens | Maximum physical HTTP |
|---|---:|---:|
| Inducer + two Editors | 700,000 | 3 |
| Feedback12 | 2,400,000 | 192 |
| Gate20 | 1,600,000 | 80 |
| Val17 paired | 1,500,000 | 68 |
| Test100 paired | 12,000,000 | 400 |

There are zero HTTP retries and no automatic formatting repairs. HTTP bounds
are derived from the declared loops: feedback is 12 NoSkill plus at most
7 x 12 Skill solutions, each main + at most one finish. Stage tokens are hard
limits, not a guarantee of completing all pairs. Test worst-case is 25.6M
tokens; 12M requires observed-cost preflight and separate review approval.

`prepare` does not write authorization. Once explicitly approved by the user,
`paid_authorization.json` must contain `plan_hash`, the original
`human_instruction`, and `phases` mapping approved phase names to the exact
limits above. `test_authorization.json` additionally contains the exact
`review_identity` printed in `review.json` and Test `limits`. Test requires a
positive completed Val review and a matching Bank/model/plan/task list.
These files are audit records of human authorization, not tools for granting
permission to oneself. Repeated stages use saved physical responses; an
interrupted request lacking a committed response blocks replay.

Snapshots of historical billing are merged by physical request ID. A new
directory cannot reset costs or resolve an unknown bill. If another campaign
is still changing its ledger, prepare/check may be inspected offline, but a
fresh final identity must be prepared from settled ledgers before paid use.

## Audit and interpretation

`plan.json`, `partition.json`, `source_catalog.json`, `proposal_material.json`
and `zero_http_check.json` show isolation and actual wire capacity. Historical
trace windows retain byte boundaries and full hashes. Complete public questions
and choices are never trimmed. Initial learning only sees 28 sources; Editor
only sees newly produced feedback12 responses and correctness, not evaluator
gold fields. No protected historical response enters either request.

`learning/*/request/` stores final payload, raw physical usage and full response;
`candidate_versions/` and `memory/` preserve parents and rejected revisions.
`feedback`, `gate`, `val` and (only if approved) `test` retain each off/on trace,
immutable response hashes, official scores and exact paired conversion counts.
`freeze_identity.json` seals the selected policy and utility receipts. The Bank
validates P7 qualification during put and freeze and excludes candidates.

`report.json/md`, `usage_by_phase.json`, candidate/memory JSONL and Val review
summarize provenance, costs, Wilson intervals, paired-bootstrap intervals and
public meta-option groups. Dollar costs remain null without a frozen price
table. Real source and transport-interception tests demonstrate engineering
behavior; all simulated candidate/gain results remain test fixtures and never
constitute P7 learning or accuracy evidence.
