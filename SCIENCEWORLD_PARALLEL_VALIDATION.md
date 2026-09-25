# ScienceWorld parallel launch verification — 2026-09-25

## Credential fix

The failed B0 child raised `KeyError: MODEL_API_KEY`. The baseline checkout had no `.env`.
The campaign now accepts `--env-file /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env`, loads only the key assignment without executing it, and fails before creating lanes if credentials are absent. No credential is stored in receipts.

## Execution-only concurrency

- 15 method/seed lanes; 2 independent episode workers per lane.
- Shared baseline cap: 12 in-flight HTTP requests and 18 live episode environments. File locks release on process death.
- Existing Ours and authored-reference campaigns are separate, three lanes each; their requests are not counted by the baseline gate.
- SkillOpt rollout/analyst and GEPA fixed-candidate evaluation are parallel; result order and optimizer update order are retained.
- EmbodiSkill state-changing train/revision remain sequential; only independent read-only evaluations are parallel.
- A failed child is recorded without cancelling other lanes. Reusing a lane while its lock is held is prohibited.

## Verified evidence

- Full pytest: **902 passed, 4 skipped**, 32.50 seconds. Log: `/home/yangchengyu/sw_parallel_regression_final.log`.
- Real environments: **18 simultaneously loaded JVMs**, barrier verified, then closed. Receipt: `/home/yangchengyu/sw_baselines_parallel_20260925_1710/environment_probe.json`.
- Provider load: **12/12 calls succeeded**, zero exhausted calls, high reasoning, 16,384 completion-token ceiling, audited SkillOpt transport. Receipt: `/home/yangchengyu/sw_parallel_probe_20260925_1710/`.
- Formal baseline root: `/home/yangchengyu/sw_baselines_parallel_20260925_1710`; all 15 child lanes started. This is launch verification, not completed-experiment acceptance.
- Original failed root `/home/yangchengyu/sw_baselines_20260925_165809` is preserved.

Baseline invocation (fresh root only; do not duplicate the running campaign):

```bash
cd /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3_baseline
export PYTHONPATH="$PWD/src:$PWD:$PWD/.external/skillopt:$PWD/.external/gepa/src" PYTHONUNBUFFERED=1
/home/yangchengyu/asg_scienceworld_venv/bin/python -m experiments.baselines.scienceworld.campaign \
  --root /home/yangchengyu/CHOOSE_A_NEW_ROOT \
  --env-file /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env \
  --seeds 42 43 44 --workers 15 --episode-workers 2 --provider-slots 12 --environment-slots 18
```

For recovery only after the corresponding processes stop, use the original root and identical concurrency settings plus `--resume`. Learning-stage interruption restrictions remain unchanged; this patch does not promise arbitrary mid-optimizer recovery.
