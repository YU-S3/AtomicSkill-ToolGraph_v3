# EmbodiSkill v2.2 integration validation — 2026-09-14

Status: v2.2 implementation and offline regressions completed; final real-API qualification is in progress. No formal training has been started.

## Implementation

- B4 SkillGen-S runtime, label/corpus builder, configurations and dedicated tests were removed. Historical runs and upstream source directories remain available; removed tracked code is recoverable from Git.
- B4 calls pinned EmbodiSkill `760126030eab1d33ec6a6f30988f0f1fb58df3a7` directly, including TeamSolver, its 1-shot ALFWorld prior, trajectory retrieval, skill-aware reflection and manual revision. The external core checkout has no patches.
- Each formal seed consumes one permutation of Train120 in four serial 30-task chunks, with read-only Val24 after each revision, strict best-snapshot selection, full persistent-state freeze, and isolated read-only Test134 copies.
- Success and the learning label use ALFWorld `infos["won"]`. Task context comes from the exact manifest gamefile's visible reset observation.
- Separate Python 3.12 worker and a pinned local MiniLM snapshot are installed. Provider attempts, unsuccessful billed responses, embedding calls, partial/unknown usage, manual/reflection/retrieval counts and failure categories are audited.
- B5 has three independent seed lanes and the 48/36/24 provider-cap protocol. No completed B3 result was modified or rerun.

## Checks completed

- Common/B3/B5 baseline suite: 279 passed, 3 skipped (B4 dependencies absent in that worker environment).
- EmbodiSkill worker-specific suite after additional coverage: 15 passed. This includes **scripted-provider** execution of the real upstream training → trajectory store → reflection → manual revision → snapshot reload → read-only evaluation path. It is not a real-API smoke pass.
- Follow-up GEPA/reporting checks: 44 passed.
- Follow-up B4 controller checks after atomic frozen-directory publication: 11 passed, 4 dependency skips in the B5 environment.
- The per-seed `run_manifest.json` and frozen provenance record the actual seed, source commit, data identities, model, ALFWorld package and selected concurrency. Frozen state is published by directory rename after the full copy verifies.
- Real GEPA dependency/ALFWorld single-worker load: passed, including exact gamefile reset. No model calls or optimizer execution.
- `git diff --check`: passed.

## Real API blocker

The latest complete attempt is:

`runs/baselines/b4_embodiskill_smoke_20260914_03/`

Its first task is `alfworld_train_2350_pick_and_place_simple`. All three upstream Solver attempts returned:

```text
model = deepseek-v4-flash
reasoning_effort = high
requested_max_tokens = 512
completion_tokens = 512
reasoning_tokens = 512
finish_reason = length
content = empty
```

TeamSolver consequently raised `Solver agent returned empty action after 3 attempts`. No environment action or learning transition completed. This is evidence of the configured completion budget being consumed before an action is emitted; it is not an API key or missing dependency error.

Evidence is under:

`seed_42/attempts/train_00_000/10d277972dd5443a85e89267ec4467aa/`

Files: `provider_calls.jsonl`, `model_responses.jsonl`, `rollout_failure.json`, `worker.log`. These original failure artifacts have been retained.

The v2.2 amendment supersedes that transport mapping. Solver/reflection hints remain 512 and revision hints remain 2048; the fixed provider completion cap is 65536. DeepSeek's documented HTTP field is `max_tokens`, so requests send `max_tokens=65536`, independently of those upstream hints. Only `content` is consumed; reasoning usage remains billed. A length-truncated, unusable response near the cap raises `COMPLETION_BUDGET_EXHAUSTED`, propagated as upstream `LLMRequestError`, with no same-cap retry. Non-length empty responses retain their original retry semantics.

B5 target/reflection transport also has this independent allowance; its 16384 upstream values are hints. Its provider load probe uses 65536. B3's upstream request path is unchanged, protected by a regression test. Ours source/config and external EmbodiSkill core are unchanged.

## Historical B3 audit (read-only)

Scanned the persisted provider sidecars under `runs/baselines/protocol_faithful_matched_train_v2/b3_skillopt/formal_3seed_authorityfix_20260912T052314Z`, including recovery evidence: 64,434 event rows (17,965 / 23,231 / 23,238 for seeds 42 / 43 / 44). These are persisted rows, not a new billed-call total; historical copies may repeat a call.

No scanned row reported failed status, completion >= 32,000, or completion equal to reasoning tokens. Historical events do not contain `finish_reason`, `content_present`, or `content_parse_success`, so this audit cannot retrospectively prove every response's finish/parse status. It found no evidence of the systematic reasoning-only exhaustion seen in B4. Original B3 results, configurations, costs and artifacts were retained; no B3 rerun was started.

## Concurrency verification

The current WSL instance exposes approximately 15.4 GiB, even though the Windows host has more memory. The first real load probe rejected 48 and 36 workers based on measured dependency/ALFWorld memory and a reserved-memory margin. Its 24-worker check was interrupted to replace slow sequential dependency initialization with bounded parallel initialization.

The replacement load-only probe writes its evidence to:

`runs/baselines/b4_embodiskill_load_20260914_02/`

The replacement probe **passed at 24 workers** (three seed lanes × eight evaluation workers). All 24 actual model requests succeeded with nonempty output and no truncation. At full load the measured available WSL memory was 6,688,051,200 bytes, above the 2,475,482,112-byte reserve. The 48/36 targets were rejected before full loading by the measured memory projection.

The authoritative result is `load_probe_summary.json` and the per-cap `load_probes/workers_*/report.json`. This qualifies the load test only; the end-to-end method smoke above is still blocked. Load-only mode never starts Train. All validation processes have exited.

## Entry points

The launch script loads the existing `AtomicSkill-ToolGraph_v3/.env` and uses a unique output directory:

```bash
cd /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3_baseline
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
bash experiments/baselines/launch_embodiskill.sh smoke "runs/baselines/b4_smoke_$STAMP" &&
bash experiments/baselines/launch_embodiskill.sh formal "runs/baselines/b4_formal_$STAMP" \
  "runs/baselines/b4_smoke_$STAMP/smoke_qualification.json"
```

The formal command also checks clean local source and requires a matching successful `smoke_qualification.json` before creating its campaign lock. Fixed B4 concurrency is three seed lanes × eight workers (global24), based on the successful load evidence above.
