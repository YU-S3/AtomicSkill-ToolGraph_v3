# ScienceWorld v2.1 final engineering acceptance

Date: 2026-09-25. Final-round base: main 31f4fe3; baseline 3f883b5.

## Current scope and release gates

**FORMAL LEARNED Ours REQUIRED:** F1 Train-only Compiler + readonly mini-chain;
F2 real RuntimeTool/persistent-reuse integration; F3 fixed 10-family diagnostic
through final maintenance/accounting/completion; F4 lane/resume/baseline checks.
**ScienceWorld learned formal = GO.** F1–F4 engineering acceptance is complete,
including the explicitly documented report-only recovery below. No formal
Train120/Test90 has been launched. This is not a perfect-score/performance claim.

### Final-round evidence (not formal performance results)

- F1: `/home/yangchengyu/sw_final_f3_20260925/seed42/compiled/`.
  Train-only Compiler completed, preserving raw Train digest
  `a53e261276870dcf20b59a1c1de6afa295a6e2c19006118f76f8c245ecfc9240`;
  compiled digest `e3017a386994ff946822f7d325c24f5304f05fce15d869386b5b31a241a52431`.
  Real readonly Dev returned score 100, no infrastructure failure, unchanged
  Frozen digest, then paused at the next task boundary. The mini-bank has no
  Active executable route yet; Candidate/history assets are preserved rather
  than promoted to satisfy a performance/asset-count gate. Independent nonempty
  Compiler tests also cover exact Tool/Implementation aliases and route checks.
- Main full regression: **1917 passed**, 115.06 s;
  `/home/yangchengyu/sw_final_main_regression5.log`.
- Baseline full regression: **898 passed, 4 skipped**, 25.60 s;
  `/home/yangchengyu/sw_final_baseline_regression5.log`.
- F2: `/home/yangchengyu/sw_final_f2_20260925_r5/result.json`.
  Three real R0/Builder/Tool/R1 trials supplied committed observations. Ordinary
  maintenance admitted/promoted the persistent A/I/T; a fourth real episode used
  the stored route for three actions, with zero provider requests inside the Tool
  and zero new Builder calls. Controlled decisions are explicitly fixtures, not
  evidence of spontaneous model automation. The fourth finite fixture stops after
  reuse, official score 0; no official success is claimed.
- F3: `/home/yangchengyu/sw_final_f3_20260925/seed42/train/summary.json`.
  All ten fixed tasks returned: scores -100, 100, 77, 100, 100, 100, -100, 66, 100,
  100 (mean 54.3, 6/10 perfect). Episode infra failures 0, unknown provider usage
  requests 0, token mismatch 0; final maintenance queue 0. Ordinary low scores are
  recorded without changing the policy or adding performance gates.
  Final reporting exposed a misnamed `artifact_lifecycle_after` result key.
  The writer now uses the common `artifact_lifecycle` protocol, with a regression
  through the actual report reader. This diagnostic was finalized from its already
  authenticated persisted snapshots/attempt receipts; original result JSON is
  backed up, original immutable Traces and source code identity are retained, no
  episodes/API calls were repeated, and the knowledge digest is unchanged.
  The exited process's in-memory event count is **null**, not fabricated; 179
  persisted events in 12 task/maintenance Traces were reconciled. See
  `report_recovery_receipt.json` and `report_recovery_original_results.json` beside
  the summary. The fixed diagnostic began before the isolated bounded-count
  identity/numeric-narrowing repair; F2 exercises that repaired chain and the full
  regression covers the final code.
- F4 baseline fixed smoke: `/home/yangchengyu/sw_final_f4_20260925/`.
  All five methods completed three Test episodes, infra failures/unknown request
  usage 0. B3/B4/B5 completed their real minimal learning/selection/freeze paths.
  Completed-run resume was checked without rerunning episodes.
- Authored reference: `/home/yangchengyu/sw_authored_final_20260925_r3/`.
  108 registered assets, **76 qualified Active** (25 A + 25 I + 25 T + G03), 32
  retained Draft. G03 passed the production partial Composite/DataFlow workflow
  on two independent Train variations, one bootstrap request and zero downstream
  requests. Partial capability is not official complete-contract/P0 authority.
  Frozen digest: `575877e4ad9d5c11a0d2f08cbaa9df5bfe64712c1ef6c3a725537eb0b039fa3a`.
  Real readonly Dev acceptance completed one episode (official score 71, no infra
  failure), then paused at the task boundary; digest unchanged. Evidence:
  `/home/yangchengyu/sw_authored_final_readonly_check_r3/seed42/dev/`.

Commands, including the **separate authored Bank Test-only command**, are in
`SCIENCEWORLD_FINAL_RUN.md`. Formal commands are not executed during delivery.
The consolidated evidence check is
`/home/yangchengyu/sw_final_f3_20260925/final_acceptance.json`
(`formal_learned_go=true`, `formal_train120_test90_started=false`).

**AUTHORED REFERENCE OPTIONAL/PARALLEL:** separate A01–A30 and G01–G18 inventory,
qualification and Frozen publication, plus an independent readonly Test launcher.
This is an additional delivery, not a learned Train input or a learned release gate.
Partial capability workflows never claim official score-100 contract coverage or P0.

The historical checkpoint below records earlier verification, not current blockers.

## Completed and verified in this checkpoint

- One public collection-source registry feeds Native schema, ToolBuilder help,
  static source checks and runtime source enumeration. The five opcodes remain unchanged.
- Exact current input/local references in public action-catalog filters; no fuzzy
  action resolution and no lookup outside the current scope.
- Explicit benchmark/adapter identity, with compatibility for historical ALFWorld configs.
- Authored A01–A30 drafts (30 Atomic/Implementation/Tool triples), production static validation.
- Ordered room search up to 16 caller-authorized rooms, and container search up
  to 8 caller-authorized containers. No room/task-answer table in runtime policy.
- LOOK_IN evidence: exact public listing alignment; separate complete, empty,
  partial, inaccessible and ambiguous states. Partial listings never prove absence.
- Effect-free identity bundles certify only input identity and preserve resolution;
  they do not fabricate environment evidence or concrete entity grounding.
- Search history distinguishes live revision bounds from immutable action indices,
  preserving failed search observations through rollback without authorizing outputs.
- ScienceWorld restore returns the actual verified restored digest and replay-action
  count expected by the shared runtime transaction layer (also fixed in baseline).
- Input/dataflow closure audit, now integrated into the independent Train Compiler.

## Verification

- Main full pytest: **1893 passed**, 105.71 seconds.
  Log: `/home/yangchengyu/sw_v21_full_roundC.log`.
- Subsequently added draft-export test plus current ScienceWorld tests: **25 passed**.
  No full-suite test was removed or weakened.
- Baseline full pytest: **895 passed, 4 skipped**, 29.59 seconds.
  Log: `/home/yangchengyu/sw_v21_baseline_full_roundB.log`.
- Real JVM, no model: `/home/yangchengyu/sw_v21_search_acceptance_r7/`.
  - A21/A22/A23: zero-action identity RETURN, Atomic R1, validated outputs,
    explicit DataFlow to a real bounded WAIT1 invocation.
  - A18: three authorized rooms visited inside one Tool; target found or no fabricated output.
  - A19: real OPEN/LOOK_IN, validated returned entity/container; empty-container
    scope exhaustion; partial listing fails closed.
  - Failed searches restored through the production transaction layer;
    immutable search history remains rolled back and non-authoritative.
  - Two further adapter prefix replays succeeded.
  - This is **not** SW-WF01/P0 selection, publication qualification, or proof of
    two independent source executions. Earlier failed diagnostic runs are retained.
- Earlier full round B detected missing optional revision capture on isolated
  interpreter contexts. Fixed by recording unavailable revision as null, never
  substituting an action index. Full round C passed.

## Authored draft output

`runs/scienceworld_authored_reference/v21_draft_20260925/`

The earlier checkpoint contains 90 draft JSON assets plus audit files. The current
export additionally includes 18 explicitly partial Composite drafts (108 total).
There is **no** Active publication, Composite or deployable Frozen snapshot in this
directory. It is never used as a learned Train asset source.

Reproduce into a new, nonexistent destination:

```bash
python -m experiments.scienceworld_reference_draft --output /path/to/new-draft
```

## Resolved design boundary

G01–G18 are partial capability workflows. Their effects do not establish the
official score-100 TaskContract. Complete P0 retrieval remains unchanged.
SW-WF01 checks production entry/DataFlow/zero-LLM downstream execution, not P0.
A20 count=0 remains fail-closed without fresh time evidence; it cannot create it.

Ordinary low/negative scores, high tokens, absence of naturally requested Runtime
Automation or a complete learned P0 graph are observations, not engineering gates.
Controlled bounded lane concurrency does not require a global provider queue.

## New entry points

- `experiments.scienceworld_campaign`: independent seed Train → Compiler →
  readonly Dev → same-digest Test; seed42 ×3, seed43/44 ×1; task-boundary stop/resume.
- `experiments.scienceworld_reference_release`: isolated authored JVM qualification
  and publication. Unqualified assets remain Draft, without invented training credit.
- `experiments.scienceworld_reference_test`: authored Frozen only, readonly Test,
  seed42 ×3 and seed43/44 ×1; never runs Train or the learned Compiler.
- Baseline `experiments.baselines.scienceworld.campaign`: unchanged algorithms,
  independent method/seed lanes with bounded concurrency and duplicate-lane locks.
