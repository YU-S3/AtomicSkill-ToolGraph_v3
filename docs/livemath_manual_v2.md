# LiveMath authored reference V2

Method: `manual-reference.graph-test.v2`. This is an external human-authored control, never an Ours initial Bank. The previously viewed Test100 is a developmental retest, not a blind evaluation.

## Compare before changing

V1 already had locked Docker qualification, real `call_workflow`, ordered ValueStore dataflow, node recovery, and a shared truth-table Program for LM03/LM09. Those mechanisms and the six original Python sources are retained. V2 does not transfer Ours learning receipts or claim `usable` state.

The V1 recorded failures motivate three changes: conditional public-input routing instead of full-catalog injection; disjoint action schemas instead of permissive fields; and bounded continuation/finalization instead of another 65,536-token request after local failure. V1 results and costs remain unchanged.

## Frozen behavior

- Main: enabled/high, 65,536 completion, 600 seconds; parent limit 128,000 tokens.
- One structural feedback: at most 4,096 completion; later valid local continuation at most 16,384.
- One finalizer: enabled/low, 16,384 completion, no transport retry, no tools. It preserves the full public question/options and an explicitly bounded reasoning tail. Length, schema rejection, tool failure, admission failure and actual parent exhaustion are recorded separately.
- PublicTask goal and choices are the only routing text. Exact closed fractions, gcd/remainder, explicit small rational matrices, or explicit typed public operands can bind local Programs. Unsupported structures remain unbound. Referenced results must resolve to the same grounded operands; quotes are not truth certificates.
- At most two procedure/conditional-knowledge cards in total, two Program cards, one executable Workflow. No relevance means zero injection. Original graphs remain stored offline; only grounded V2 graphs are routed. Original final-answer Programs cannot route without independent all-choice certificates, which this release does not synthesize.
- All imported Programs remain candidates in generic Bank state. The control view separately checks the exact Program hash, locked image and positive/variation/rejection Docker receipts.

## Assets and coverage

Build with `python -m atomic_skillgraph.experiments.build_manual_math_v2 --original <V1-package> --output <new-V2-package>`.

The independent package contains 14 Skills, 8 Programs, 9 Implementations and 10 Workflows (five preserved, five V2 local-operation graphs). New Programs perform bounded integer gcd/remainder and exact rational determinants of matrices of dimension at most four. They return intermediate local facts, not general proofs.

Three conditional theorem entries cite fixed mathlib4 commit `f0b4c8150aa0dbc758aff1291410c5ae335aff3a`: Banach contraction, first Borel–Cantelli and Lagrange's subgroup theorem. Downloaded source files and Apache license are retained with hashes. `source_checked` means the declaration and surrounding hypotheses were inspected; no local Lean build or independent expert sign-off is asserted. Runtime always lists task assumptions as unverified and coverage as insufficient until mathematical reasoning establishes applicability. No PDE or representation-theory coverage is claimed.

## Evaluation and accounting

`run_manual_math_v2 prepare` imports the separate package, reruns 24 actual Docker probes, freezes the canonical task lists, source identity, configs, manifest and Bank, and unions historical ledgers by physical attempt ID. The authorized unresolved historical reservation remains once; no usage is invented.

The proposed small development comparison uses the first three canonical Val tasks off/on. Both arms have identical Runtime parameters, scorer and public inputs; only the reference view is enabled/disabled. The fixed development ceiling is 768,000 tokens / 66 HTTP, Test ceiling 13,000,000 / 5,000, both within the existing cumulative 45,000,000 / 9,165. These are ceilings, not expected charges. No performance threshold selects or rewrites assets. Test follows completion of the development comparison and is read-only.

Commands: `prepare --package ... --preflight ... --source-plan ... --ledgers <historical budget.json paths> --output ...`; `check --output ...`; `execute --output ... --env-file ... --phase development|test|all`; `status --output ...`.

The zero-HTTP suite intercepts the production Provider, replays the actual ten V1 failure ToolCalls/raw usage, checks public Train60/Val17 routing, exact schema negatives, real Docker Program/Workflow consumption, bounded finalization, paid-response recovery, and 3+3+100 control flow. Synthetic final answers in replay are protocol fixtures, not corrected historical scores or evidence of mathematical improvement.
