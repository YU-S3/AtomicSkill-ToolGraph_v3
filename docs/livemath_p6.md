# LiveMath P6

`livemath.math-capability-program.p6` is an isolated Train-trace derivation.
It reads original Train60 and the fifteen pre-Val P4 source hypotheses. It never
imports P4/P5 published assets, selection outcomes, manual assets, or gold.

The operation registry owns the model schemas, diagnostics, pure Worker ABI,
public parsers and independent local oracles. Keyword matches create unresolved
demands only. The initial parsers support literal formal quantifier syntax and
explicitly labelled finite JSON relations/rational pairs. This is deliberately
narrow: mathematical prose is not converted to invented finite examples.

The evidence ladder separates synthetic algorithm checks, real task binding,
independent Train transfer, explicit consumption and formal utility. Every new
Program remains `candidate`; the read-only trial view grants separate experimental
execution permission only when its current public arguments can be bound.
At most two Programs and two related Skills are shown. Nothing is automatically
promoted to usable or sent to Val/Test.

The fixed scheduler groups at most two physical sources per unit, up to eight
units, with one shared repair across Extractor, Builder and local validation.
An independent transfer task is reserved before generation when at least three
sources support an operation. Insufficient coverage is an explicit negative
result, not a reason to synthesize another source or change the schedule.

## Offline preparation

Use a clean Linux checkout of the committed revision:

```bash
export PYTHONPATH="$PWD/src"
PY=/home/yangchengyu/asg_alfworld_venv/bin/python
OUT=/home/yangchengyu/livemath_p6_seed42_preflight_20261015
"$PY" -m atomic_skillgraph.experiments.run_math_program_p6 prepare --output "$OUT"
"$PY" -m atomic_skillgraph.experiments.run_math_program_p6 check --output "$OUT"
"$PY" -m atomic_skillgraph.experiments.run_math_program_p6 report --output "$OUT"
```

`check` builds production Provider payloads and exercises actual Governor admission
against disposable ledger copies; it sends no HTTP. It also checks source hashes,
the empty initial Bank, the locked Docker image, and the historical accounting
prefix. Parallel manual V2 costs are covered once by their full authorized cap.

`execute --phase learning|smoke|all` is available only after separate paid approval.
The document's 1M/100 learning and 400K/30 smoke limits are proposals, not current
authorization. Val requires a later fixed protocol and budget. A source/schema
change requires a new prepared identity; ordinary resume cannot disguise it.

Completed responses and Worker receipts are reused. A sent HTTP or started Worker
without its final receipt stops recovery. A cached model rejection without its
diagnostic receipt also stops rather than reconstructing different repair feedback.
Trial export is immutable; a partial export without a manifest requires explicit
recovery. These stops preserve the original evidence and charges.

## Evidence

- `public_demands.json`, `units.json`, `source_catalog.json`: public coverage and fixed sources.
- `plan.json`, `allocation_manifest.json`, `zero_http_check.json`: source identity, budgets and gate.
- `learning/*/{learning,requests}.json`, `learning/*/worker/`: natural responses and separate local/transfer receipts, only after execution.
- `trial_manifest.json`: deduplicated implementations, complete source lineage and candidate-only permission.
- `smoke/{off,program}/`: paired Train-only consumption evidence, only when a fixed independent task binds an actual generated Program.
- `learning_report.json`: real negative outcomes, actual charges and no automatic promotion.

`tests/test_math_program_p6_contract.py` and `tests/test_math_program_p6_transfer.py`
use intercepted model responses and actual Docker. Their synthetic tasks demonstrate
protocol behavior; they are never counted as naturally learned assets or real Train transfer.
`math_program_p6_audit.replay` separately replays all twenty real P5 final failures,
preserves all 51 locally passed source lineages and repeats the original pairwise
comparison failure without changing historical records.
