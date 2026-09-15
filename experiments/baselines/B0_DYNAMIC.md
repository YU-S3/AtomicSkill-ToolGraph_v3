# B0 Pure Dynamic

B0 reuses the exact pinned SkillOpt target rollout used by B3/B5, passing
`skill_text=None`. The entire Skill Knowledge section is absent. It never
loads initial skill text into an agent, trains, selects on Validation, retrieves
experience or opens an Ours bank. Source provenance hashes include upstream
files; hashing their bytes does not inject them into the method.

Formal evaluation is Test134 × seeds 42/43/44. There is no Train120 phase.
All calls use DeepSeek-v4-flash high with the v2.2 65,536 completion allowance,
content-only consumption, the upstream action/fallback protocol and 100 actual
actions per attempt. Persistent state is a frozen empty descriptor.

The launcher reuses the already verified B5 dependency venv (not its GEPA
optimizer). `B0_PY`, `SKILLOPT_ROOT`, `ENV_FILE`, `ALFWORLD_DATA` can override
local paths. No dependency installation changes a running experiment.

```bash
bash experiments/baselines/launch_dynamic.sh smoke runs/baselines/b0_smoke
bash experiments/baselines/launch_dynamic.sh formal runs/baselines/b0_formal \
  runs/baselines/b0_smoke/smoke_qualification.json
# Only if interrupted; completed tasks are verified and reused, not rerun:
bash experiments/baselines/launch_dynamic.sh resume runs/baselines/b0_formal
```

Formal refuses an active method campaign in any worktree of this repository.
Do not delete the shared lock. Its preflight holds real SkillOpt/ALFWorld
workers while testing provider load, then locks the highest safe concurrency
among 3×16, 3×12, 3×8. Only memory rejection permits a lower tier; provider
errors stop startup. Three independent seed lanes always overlap.

Smoke runs six complete engineering episodes (one per family, two per seed)
from Train identities, never heldout Test, with no learning. Qualification
binds code, config, upstream, interpreter and all raw smoke evidence.

Each task uses a fresh spawned process. Immutable episode checkpoints are
published before offline strict replay; a replay failure therefore does not
repeat model calls. Resume checks source/config/manifest/runtime, completed
provider and rollout bytes, frozen state and per-task records. Partial
attempts are retained; infra failures never become success=0 task rows.

Outputs: `REPORT.md`, `paper_report.json`, `campaign_summary.json`, plus each
`seed_*/test/{task_rows,results,evaluated_common_episodes,provider_calls}.jsonl`.
Full visible prompts/responses and action journals are under
`seed_*/test/episodes/task_*/attempts/attempt_*/`. Posthoc TaskContract facts
never enter agent decisions. Training cost is zero. Unknown physical-attempt
usage remains null with known subtotals; API prices are not assumed.
