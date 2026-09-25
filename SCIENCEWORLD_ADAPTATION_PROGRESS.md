# ScienceWorld adaptation — implementation checkpoint, not release approval

Updated: 2026-09-25. This is an incomplete implementation checkpoint. No formal
Train120/Test90 experiment has been started and no authored bank has been
published as a training result.

## Scope and authority

- Main starting commit: `2034d4fa8fe24ac155c13614d12af2dc1a1e69e0`.
- Baseline starting commit: `45d1bdb5f9fbbf2d11279e1be4c1ca6be8bcd684`.
- Baseline working branch: `baselines-skillopt` (existing branch preserved).
- Scope: Ours, B0 Dynamic, B1 StaticSkill, B3 SkillOpt, B4 EmbodiSkill, B5 GEPA.
- ScienceWorld 1.2.3, OpenJDK 17, easy simplification, 100 environment steps,
  `generateGoldPath=False`. Existing ALFWorld environment remains untouched.
- Local environment: `/home/yangchengyu/asg_scienceworld_venv`.

## Implemented and tested so far

The two repositories contain independent copies of the ScienceWorld adapter and
identical manifests: Train120, Dev10 and Test90, selected from official splits.
Their SHA256 manifest digests are respectively:

```
c2f82323ceb0365a1e983a0d57da5c3ce222bba13679d2fe776c44f253d6a627
02e7a79aa5d26dd5e9e5e0b1719dc9bbfb40093e5f266433be4d14b7485824e8
0ee19e729f732fe23b861a911acf81115e118569f7a116a19a3814871a3e559c
```

Implemented: public-frame policy surface, exact action tuples, public clarification
actions, official-score reporting, environment prefix replay, provider accounting,
benchmark factory integration, candidate Support execution deduplication, public
discovery precedence, and partial reference-asset authoring with production
loaders/validators. Device activation requires the public affirmative effect;
offered USE actions can still be rejected by the environment.

Baseline adapters preserve the pinned ReflACT/GEPA/EmbodiSkill algorithms. Each
completed a minimal real-API pipeline, with the same three diagnostic Test games.
B3/B4/B5 also exercised two Train and two Dev games. This is a wiring check, not a
formal performance estimate or proof of full Train120 coverage. Ordinary failures
and raw negative official scores are retained.

The baseline text runner now isolates environment attempts. Failed prefixes remain
in `attempts/<id>/`; the episode result names one completed canonical attempt.
Paid provider requests from failed attempts remain in episode-level accounting.

## Evidence locations

Final verification: main **1870 passed** in 108.73 seconds in the existing ALFWorld
environment, including the new IR boundary tests and existing release regression
suites; baseline full suite **895 passed, 4 skipped** in 38.88 seconds. Running the
full main suite in the isolated ScienceWorld environment instead leaves two
ALFWorld-version preflight tests failing because that environment intentionally
does not contain ALFWorld; the full existing-environment run above passes them.

- Environment reset/replay gate: `runs/scienceworld_acceptance/environment.json`.
- Numeric clarification replay: `runs/scienceworld_acceptance/clarification.json`.
- Production ToolRunner tests and authority schema:
  `runs/scienceworld_acceptance/programs/`.
- Five baseline real-API diagnostic lanes:
  `/home/yangchengyu/asg_scienceworld_baseline_acceptance_20260925/`.
- Main ten-category diagnostic (all ten episodes returned; final maintenance failed):
  `/home/yangchengyu/asg_scienceworld_acceptance_20260925_r3/`.
- Previous diagnostic failures are retained under the same prefix without `_r3`
  and with `_r2`; they are not silently included in the new canonical run.
- Main full regression log:
  `/home/yangchengyu/scienceworld_ir_release_pytest.log`.
- Baseline full regression log:
  `/home/yangchengyu/scienceworld_baseline_final_pytest.log`.

For Windows Explorer, prepend `\\wsl.localhost\Ubuntu` to these Linux paths and
replace `/` with `\`. Diagnostics have not been treated as final release gates
while implementation is still changing.

## Remaining work — do not launch formal experiments yet

1. Complete high-value programs and their public effect witnesses. The current
   partial reference inventory has 19 A/I/T triples, not the complete A01–A30.
2. Apply the now-implemented public field projection / bounded count interface
   to the remaining authored reference tools and qualify their real effects.
3. Complete G01–G18, explicit data flow, truthful task-contract coverage and
   production-validated authored deployment publication. No fake success credit.
4. Implement and test the full Train-only Bank Compiler, independent source
   proof gates, closure/preparation audit, and read-only Dev/freeze workflow.
5. Qualify final snapshots with real program positive/negative/replay cases and
   read-only authored-bank Dev10; finish cross-method reporting/repeat summaries.
6. Provide audited three-lane launchers (seed42 repeats three times), including
   a separately labeled authored-reference Frozen Test launcher. The current
   adapter/IR checkpoint can be reviewed remotely; formal launch commands are
   intentionally not released yet.

## Public Tool IR extension — authorized and implemented

The five opcodes and core BindingExpression are unchanged. Tool-only references
accept 1–4 named object fields through field_path, with one shared resolver and
lexical schema validator. Arrays, JSONPath, expressions, cross-node and transform
authority are not permitted in these projections. Missing/non-object fields fail.

FOR_EACH source=bounded_count reads a required bounded integer input. Explicit
input_schema narrows (never changes) the Atomic input roles. Array maxItems and
integer maximum must fit max_iterations; loop limits and conservative worst-case
ACTION count must fit Tool max_actions. New-contract invocation also verifies the
Runtime global action ceiling, and actual actions consume the existing shared
budget. Inputs over limits are rejected before action, not truncated.

Explicit input_schema or new references opt into value_contract_version=2.
Existing v1 assets/proposals without these fields retain their legacy semantics
and serialized artifact layout. No existing Bank is rewritten; new array/count
guarantees are not retrospectively claimed for unversioned historical programs.
Native invocation schemas expose the persisted input bounds. Trace records carry
the resolved field references and loop count evidence. Public catalog-driven
discovery retains its bounded-search semantics; caller-input arrays cannot be
silently truncated. No benchmark-private opcode or answer routing is introduced.

Real JVM gate `runs/scienceworld_acceptance/ir_v2/` passed: exactly eight WAIT1
actions, structured loop-item navigation, oversized count/array rejected with
zero actions, and two identical prefix replays. The final-code rerun also passed:
`runs/scienceworld_acceptance/ir_v2_final/` (four checks, two prefix replays).

The ten-category real-API diagnostic returned seven perfect successes, one score
77 and two scores -100. Final maintenance then failed because the new runner had
not supplied the initial artifact audit snapshot. The runner now records that
snapshot/digest and task artifact growth, matching the existing maintenance
contract. The failed diagnostic and all paid usage remain untouched. It is NOT
an end-to-end passed run, nor a formal accuracy estimate.
