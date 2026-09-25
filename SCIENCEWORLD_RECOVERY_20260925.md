# ScienceWorld recovery — 2026-09-25

## Scope and evidence

- Main: restore ScienceWorld task reset identity (`task_name`, `variation_idx`, `source_split`) from the immutable replay source; reject conflicting caller identity.
- Baseline: retire only stale historical `valid_actions_compact` catalogs. Keep past observations, actions and the complete current catalog. No benchmark-specific policy or answer rules added.
- SkillOpt: reuse completed rollout episodes only when source identity and submitted skill match; preserve each provider-observer lifetime in its own evidence file.
- Main recovery: retain original manifests and known failed-attempt usage, record the new execution-code hash explicitly, restore the interrupted task's pre-task checkpoint. An unfinished maintenance tail remains incomplete; its existing Trace contributes recorded usage only, not a completed result.
- Five interrupted attempts have unproven usage tails. Their evidence is retained in `recovery_quarantine/`, with SHA256 receipts in `scienceworld_recovery.json`; unknown usage is not filled with zero.

## Verification

- Main final full pytest in the ALFWorld environment: **1921 passed**.
- Baseline final full pytest in the ScienceWorld environment: **905 passed, 4 skipped**.
- Historical replay probe: 17 old cases resolved, no identity errors; two actual ScienceWorld reset checks passed.
- All six Ours/reference state databases passed SQLite quick checks. Existing task evidence and training checkpoints were compared against the independent backup before recovery.
- The first main test run used the ScienceWorld-only environment: 1918 passed, two ALFWorld-version checks failed. The correct ALFWorld environment passed the complete suite; no guard was weakened.

## Storage

Ubuntu was relocated through `wsl --manage Ubuntu --move` to `D:\WSL\Ubuntu`; Linux paths are unchanged. C: regained about 110 GiB.

Complete pre-recovery backup:

`/home/yangchengyu/sw_recovery_20260925_verified/`

It includes the learned, authored-reference and baseline experiment roots. The earlier `sw_recovery_20260925_2022` copy is incomplete and must not be used as the authoritative backup.

TableGPT2 was moved to `D:\C_Drive_Relocated\ARR-model\models\TableGPT2-7B` with hashes checked, before the user requested deletion. Automatic deletion was blocked; the model still exists. The original C: location is a junction.

## Resumed experiments and monitoring (WSL)

- Ours supervisor PID 5556; original root `/home/yangchengyu/sw_learned_20260925_165515`; resume task ordinal 22 / 22 / 23 for seeds 42 / 43 / 44.
- Authored reference supervisor PID 5557; original root `/home/yangchengyu/sw_reference_parallel_20260925_1710`; resume test ordinal 27 / 27 / 26. Source frozen bank is unchanged.
- Baseline supervisor PID 418; original root `/home/yangchengyu/sw_baselines_parallel_20260925_1710`. Already completed B0/B1 seeds 43/44 are reused, not re-evaluated.
- SkillOpt 42/43 required an additional observer-file fix: restarted under locked supervisors 3440 / 3442 (workers 3441 / 3443); seed44 remains under PID 418. Old baseline supervisor log retains their earlier failure messages; those lines do not describe the restarted processes.

```bash
tail -n 10 -F /home/yangchengyu/sw_learned_20260925_165515/seed*/train.log
tail -n 10 -F /home/yangchengyu/sw_reference_parallel_20260925_1710/seed*/test_repeat*.log
tail -n 5 -F /home/yangchengyu/sw_baselines_parallel_20260925_1710/*_seed*.log
```

For a point-in-time state/process snapshot:

```bash
/home/yangchengyu/asg_scienceworld_venv/bin/python /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/runs/sw_recovery_status.py
```

Per-lane logs are appended to preserve earlier errors. A quiet log does not imply a stopped process: baseline API evidence files were checked and all eleven incomplete baseline lanes had fresh successful requests after recovery. No end-to-end completion or future success rate is claimed here.
