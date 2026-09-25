# ScienceWorld v2.1 implementation checkpoint — NOT released

Date: 2026-09-25. Base: main 3ad096d5; baseline 177e0a5.

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
- A standalone input/dataflow closure audit, not yet integrated into a Train compiler.

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

Contains 90 draft JSON assets plus inventory/static-validation/release-check files.
There is **no** Active publication, Composite or deployable Frozen snapshot in this
directory. It is never used as a learned Train asset source.

Reproduce into a new, nonexistent destination:

```bash
python -m experiments.scienceworld_reference_draft --output /path/to/new-draft
```

## Required decision / specification conflict

ScienceWorld task contracts are derived from official score=100
(`harness/scienceworld.py::task_contract`). P0 uses exact complete-contract
retrieval (`planner/composite_retriever.py`); terminal empirical evidence is not a
substitute for a complete contract.

Document section 6 defines G01–G18 as operation templates. Their listed children
provide search/measurement/observation effects, not the official task-goal effect.
SW-WF01 nevertheless asks for P0 complete-Composite selection and permits score<100.
Low execution score alone is not contradictory (execution can fail); the problem
is that the proposed graph has no static complete-goal effect coverage to qualify
for that P0 path in the first place.

Pending user choice: retain these as partial capability workflows, reserving P0
for routes with genuine full-contract coverage, or keep them as unpublished drafts
until the specification clarifies the missing goal authority. No `task_contract_covered`
flag, success counter, goal predicate or admission guard has been fabricated/relaxed.

A20 also permits count=0 while declaring time.progressed. Zero actions provide no
new time witness. Current draft must fail closed without an existing valid witness;
it cannot promise fresh evidence for this case. Publication needs this boundary settled.

## Remaining work — NO GO

- G01–G18 construction/qualification/publication, after resolving the contract boundary.
- Independent real source evidence and authored SW-WF01.
- Train-only Bank Compiler and its evidence/publication authority, closure output,
  preferences and read-only freeze; no authored asset leakage.
- Actual mini Train/evolution/final maintenance/compiler chain, SW-WF02 and SW-WF03.
- Combined Runtime Automation → ToolBuilder → persistent reuse acceptance on real JVM.
- Fresh complete 10-family diagnostic after final-maintenance accounting changes.
- Baseline final dependency/minimal-pipeline/resume qualification and shared provider queue gate.
- Final wrappers and formal launch commands after all required gates pass.

No formal Train120/Test90 or reference Test90 has been started. This checkpoint is
not permission to start those experiments and does not claim the user request complete.
