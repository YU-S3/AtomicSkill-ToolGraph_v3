# B1 — Static SkillOpt ALFWorld Seed

B1 runs the frozen Test134 manifest with seeds 42/43/44. There is no training,
Validation24 selection, optimizer, bank, or cross-task memory. The six-family
smoke uses engineering tasks from Train120, never counted as formal training.

Each seed copies the pinned SkillOpt
`skillopt/envs/alfworld/skills/initial.md` byte-for-byte to
`seed_<seed>/frozen/artifact/initial.md`. Its source path, upstream commit and
SHA-256 are recorded in the frozen metadata and campaign identity. Configuring
an alternative skill is prohibited. Workers use only the frozen copy; both its
digest and original pinned file hash must remain unchanged.

The common text executor delegates prompt/action handling to pinned SkillOpt,
identically to B0/B3/B5. The sole B0-to-B1 method difference is the fixed initial
skill supplied to the upstream Skill Knowledge section. Missing action tags in
successful responses retain upstream's look fallback. Provider failures and
completion-budget exhaustion never become task-semantic fallback actions.

## Launch (WSL)

The wrapper reuses the existing B5 virtual environment and v3 `.env`. It does
not install dependencies, read a trained skill, or modify previous campaigns.

```bash
cd /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3_baseline
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
bash experiments/baselines/launch_static_skill.sh smoke "runs/baselines/b1_smoke_$STAMP" &&
bash experiments/baselines/launch_static_skill.sh formal "runs/baselines/b1_formal_$STAMP" \
  "runs/baselines/b1_smoke_$STAMP/smoke_qualification.json"
```

With an existing passing receipt for the unchanged code/config/runtime, run only
the second command, substituting that receipt's actual path. Formal startup
loads real resident environments and probes the provider at global48, then
36/24 only for a memory rejection. Three seeds run concurrently; each lane has
one third of the qualified cap. Do not run a different method simultaneously.

Resume the same output directory after infrastructure recovery:

```bash
bash experiments/baselines/launch_static_skill.sh resume runs/baselines/ACTUAL_B1_FORMAL_DIR
```

Completed task evidence is hash-verified and reused. A committed episode needing
only post-hoc replay does not repeat model calls. Partial attempts remain in cost
accounting. Changed source/config/runtime/frozen artifacts fail closed.

## Reports and accounting

- Campaign: `REPORT.md`, `paper_report.json`, `campaign_summary.json`.
- Per seed: `test_report.json`, `test/task_rows.jsonl`,
  `test/evaluated_common_episodes.jsonl`, `test/provider_calls.jsonl`.
- Per episode: `test/episodes/task_*/record.json`, completion/checkpoint,
  `attempts/attempt_*/outcome.json`, provider calls and visible model responses.
- Frozen skill: `frozen/artifact/initial.md`, provenance in `frozen/digest.json`.

Official won and post-hoc strict success, per-family accuracy, calls, actions,
latency and token statistics use the existing common reporter. Training and
evolution costs are zero; target usage includes physical retries and failed
attempts. Prompt/completion/reasoning/visible completion are recorded separately;
unknown billed usage stays null with a known subtotal, not zero. Reasoning tokens
are part of completion tokens, not an extra charge to add twice. API prices are
unconfigured and marked unpriced. B1-only reports mark transfer relative to B0
unavailable until B0 artifacts are supplied to the cross-method reporter.
