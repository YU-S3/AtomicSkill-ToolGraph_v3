# EmbodiSkill v2.1 integration validation — 2026-09-14

Status: implemented locally; real end-to-end smoke is **not qualified**. No formal training has been started.

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

The frozen design §21.3 explicitly fixes reflection to 512 tokens and manual revision to 2048; the upstream default Solver/condensation calls also use 512. DeepSeek's completion allowance includes reasoning tokens. Increasing the transport allowance requires an explicit B4 protocol decision, not an unrecorded workaround. An allowance of 16384 for B4 was proposed to the user and has **not** been applied.

## Concurrency verification

The current WSL instance exposes approximately 15.4 GiB, even though the Windows host has more memory. The first real load probe rejected 48 and 36 workers based on measured dependency/ALFWorld memory and a reserved-memory margin. Its 24-worker check was interrupted to replace slow sequential dependency initialization with bounded parallel initialization.

The replacement load-only probe writes its evidence to:

`runs/baselines/b4_embodiskill_load_20260914_02/`

The authoritative result is `load_probe_summary.json` and the per-cap `load_probes/workers_*/report.json`. Missing or failed results do not authorize formal execution. Load-only mode never starts Train.

## Entry points

The launch script loads the existing `AtomicSkill-ToolGraph_v3/.env` and uses a unique output directory:

```bash
cd /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3_baseline
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
bash experiments/baselines/launch_embodiskill.sh smoke "runs/baselines/b4_smoke_$STAMP" &&
bash experiments/baselines/launch_embodiskill.sh formal "runs/baselines/b4_formal_$STAMP"
```

This is the gated entry point, **not a claim that formal execution is currently cleared**. Do not remove the smoke condition while the budget blocker remains unresolved. The formal command also checks clean local source and actual allowed concurrency before creating its campaign lock.
