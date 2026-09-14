# EmbodiSkill v2.2 integration validation — 2026-09-14

Status: **B4 and B5 real-API smoke passed; B4 v2.2 formal gate verified without starting Train.** Execution source commit: `6a85939`. No formal training has been started.

## Final v2.2 qualification

- Full common/B3/B5 regression suite: **290 passed, 4 skipped**. The four heavy B4 dependency cases were separately exercised in the EmbodiSkill worker environment (22 passing tests, including the real upstream lifecycle and typed budget-failure propagation).
- B4: `runs/baselines/b4_embodiskill_v22_smoke_20260914_03/`, **passed=true**, **source_config_unchanged=true**, 1127.13 seconds (18m47s). It executed 2 Train + 2 read-only Val + 2 frozen Test episodes, plus independent live recovery/diagnosis probes. All nine smoke checks passed. Frozen state remained unchanged. The two smoke Test episodes succeeded; this is not a formal accuracy estimate.
- All 86 B4 physical requests have known usage, unique attempt IDs and consumable/parseable content. No retry, failed request, length finish or budget exhaustion occurred. Every request records provider cap 65536 and HTTP field `max_tokens`.
- B5: `runs/baselines/b5_gepa_v22_smoke_20260914_01/smoke_report.json`, **passed=true**. Actual optimizer budget: 24 metric calls, 3 reflection calls, followed by 6 frozen read-only Test episodes. 63 physical requests, zero retries/failures/length finishes/budget exhaustion; frozen digest unchanged. This retains the existing two-action engineering smoke configuration and is not a formal performance estimate.
- EmbodiSkill pinned upstream core remains unpatched. Formal B4 config remains Train120/Val24/Test134 with seeds 42/43/44 and fixed 3×8/global24 parallelism. Completed B3 artifacts and Ours source/config were not changed.

Actual billed token evidence (completion includes reasoning):

| Smoke/role | Requests | Prompt | Completion | Reasoning | Visible completion |
| --- | ---: | ---: | ---: | ---: | ---: |
| B4 target | 74 | 100992 | 90398 | 89496 | 902 |
| B4 evolution | 12 | 14035 | 26216 | 25055 | 1161 |
| B5 target, including frozen Test | 60 | 92660 | 13462 | 8256 | 5206 |
| B5 evolution | 3 | 7734 | 26031 | 21207 | 4824 |

No pricing table is frozen, so monetary API cost remains null/unpriced. Costs are not estimated from the 65536 ceiling.

B4 required-role evidence, all with parse success and no length finish:

| Stage | Calls | Maximum actual completion | Reasoning subtotal | Visible subtotal |
| --- | ---: | ---: | ---: | ---: |
| Solver | 73 | 4541 | 86656 | 851 |
| Stuck recovery | 1 | 2891 | 2840 | 51 |
| Trajectory condensation | 2 | 6247 | 10914 | 459 |
| Failure diagnosis | 2 | 1378 | 2475 | 177 |
| Episode reflection | 2 | 3206 | 3996 | 325 |
| Manual revision | 1 | 275 | 99 | 176 |

The other five B4 calls are upstream trajectory reranking and are included in the billed totals. Per-request cap, usage, finish reason, content length and parse result are retained in `seed_42/stage_qualification.json` and each attempt's `provider_calls.jsonl` / `model_responses.jsonl`. B5 additionally writes `smoke/reasoning_budget_report.json` and `smoke_test/reasoning_budget_report.json`.

The successful B4 receipt is `runs/baselines/b4_embodiskill_v22_smoke_20260914_03/smoke_qualification.json`. Its formal-config hash, actual current code hash and pinned upstream tree were checked through the real formal gate without invoking the training runner.

Earlier v2.2 attempts are retained: `_01` completed the method chain but was rejected because source changed during integration; `_02` completed all method checks but revealed that the common episode reader omitted the newly persisted visible-token fields. The reader was corrected without weakening row consistency, regression fixtures now contain non-null visible usage, and the preserved `_02` artifacts successfully rebuild through the corrected report reader. `_03` is the fresh, fixed-source end-to-end release evidence.

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

## Historical v2.1 API blocker

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

Scanned the persisted provider sidecars under `runs/baselines/protocol_faithful_matched_train_v2/b3_skillopt/formal_3seed_authorityfix_20260912T052314Z`, including recovery evidence: 64,434 unique `(run_id, call_id)` records (17,965 / 23,231 / 23,238 for seeds 42 / 43 / 44). This is an audit record count, not a replacement billed-call total.

The historical target cap was 16384. There are 16 records with completion >=16000 and four exactly at 16384. No record has completion equal to reasoning tokens, but that alone does **not** establish usable output. Matching the three target records to the corresponding task/rollout conversation ordinal shows `missing action tag` followed by the upstream `look` fallback:

| Seed | Stage/task | Call ID | Conversation step |
| --- | --- | --- | --- |
| 42 | Test `alfworld_eval_out_of_distribution_97_pick_heat_then_place_in_recep` | `provider_3e08ba85291544a8a22e9f57aa7c9e0f` | 18 |
| 44 | Train `alfworld_train_3391_pick_heat_then_place_in_recep` | `provider_092e773f1f6f45958b4c93c1a1d02105` | 41 |
| 44 | Val `alfworld_eval_in_distribution_84_pick_two_obj_and_place` | `provider_58126e50b75f478a94486d680f04c87c` | 28 |
| 42 | Analyst | `provider_23c2be6f39394c008cfcf2486f0cbb4e` | Not attributable to an individual saved response |

The matching target conversations are respectively under `recovered_test_waiver_20260913T073609Z/seed_42/test/rollout/predictions/`, `seed_44/train/train/steps/step_0001/rollout/predictions/`, and `seed_44/train/train/steps/step_0008/selection_eval/predictions/`, each followed by the task ID and `conversation.json`.

These are concrete **cap-associated unusable-action observations**, not proof of systematic failure throughout B3. Historical events do not contain `finish_reason`, `content_present`, or `content_parse_success`, and the fallback replaces the raw malformed content, so definitive truncation attribution is unavailable. The Val call also records a recovered `empty_message` retry, without physical-attempt usage sufficient to reconstruct the failed attempt's tokens. Original B3 configurations, results and costs were retained under the frozen historical protocol; no B3 rerun or retrospective relabeling was started. This limitation must accompany use of the historical B3 comparison. New B4/B5 audit and fail-fast paths prevent silently accepting this budget-exhaustion pattern.

## Concurrency verification

The current WSL instance exposes approximately 15.4 GiB, even though the Windows host has more memory. The first real load probe rejected 48 and 36 workers based on measured dependency/ALFWorld memory and a reserved-memory margin. Its 24-worker check was interrupted to replace slow sequential dependency initialization with bounded parallel initialization.

The replacement load-only probe writes its evidence to:

`runs/baselines/b4_embodiskill_load_20260914_02/`

The replacement probe **passed at 24 workers** (three seed lanes × eight evaluation workers). All 24 actual model requests succeeded with nonempty output and no truncation. At full load the measured available WSL memory was 6,688,051,200 bytes, above the 2,475,482,112-byte reserve. The 48/36 targets were rejected before full loading by the measured memory projection.

The authoritative result is `load_probe_summary.json` and the per-cap `load_probes/workers_*/report.json`. That result qualified load only; the end-to-end blocker at that time is now resolved by the v2.2 qualification above. Load-only mode never starts Train.

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

For this already-qualified local checkout, start formal directly (the launcher loads the existing v3 `.env`):

```bash
cd /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3_baseline
bash experiments/baselines/launch_embodiskill.sh formal \
  "runs/baselines/b4_v22_3seed_$(date -u +%Y%m%dT%H%M%SZ)" \
  runs/baselines/b4_embodiskill_v22_smoke_20260914_03/smoke_qualification.json
```
