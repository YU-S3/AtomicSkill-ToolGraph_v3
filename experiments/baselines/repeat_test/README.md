# Seed42 frozen Test134, two additional repetitions

The original seed42 test is repetition 0. Only repetitions 1 and 2 are launched.
These are inference repetitions of **one trained seed**, not three independent
training seeds. B0 retains its empty descriptor; B1 its original initial.md;
B3/B5 their original best_skill.md; B4 its entire selected state/Chroma/manual.
No training, new validation selection, or frozen knowledge updates are allowed.

## Equality boundary

The source map in configs/baselines/seed42_repeats.yaml pins each original
completed Test134 and its frozen artifact digest. Startup verifies all 134
original task identities, outcomes' frozen digests, original reports, original
resolved configuration, pinned upstream bytes, worker interpreter and current
runtime packages. Each round makes a byte-identical independent copy. B4 makes
an additional disposable state copy for every task using its existing controller.
Both the original and the repeat frozen assets are checked again after testing.

Each method keeps its own historical policy. B3 stays at completion cap 16384;
B0/B1/B4/B5 stay at 65536, all deepseek-v4-flash/high. Text methods retain the
16384 upstream output hint, 100-action outer limit, exact-gamefile environment
reset, unchanged upstream prompts/action parser and seed42. B4 retains TeamSolver,
ReasoningIO, few-shots, retrieval/reranking, embedding bytes, max_trials=30,
temperature=0.1, selected epoch, and readonly test hooks. Text temperature remains
unset (provider default); ambient temperature overrides are rejected.

New orchestration is not the old controller binary. It reuses the existing
per-task executor and transport with their already-audited correctness fixes;
it does not reproduce historical logging bugs. Original controller metadata is
retained through source evidence hashes. Server weights behind a mutable model
alias, unrecorded historical provider defaults, and load-dependent sampling
cannot be proven identical. Do not claim bitwise determinism or a pinned remote
model revision.

The intentional scheduling change is ten independent processes, **one task
worker per process**. Copying old lane concurrency would create over a hundred
workers and risk OOM. Each round has its own provider gate, lock, logs, assets,
checkpoints and output directory. There is no cross-method campaign lock or
shared failure/cancellation signal. Each round reports and exits as soon as it
finishes, without waiting for siblings. A process still executes the 134 tasks
of its own round in manifest order. Retries preserve the historical transport
policy and all failed-attempt evidence.

## Verification and launch

The launcher loads v3 .env and uses the original B3/B4/B5 virtual environments
for the relevant method workers. It does not install dependencies or update git.

```bash
cd /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3_baseline
bash experiments/baselines/launch_seed42_repeats.sh smoke runs/baselines/repeat_smoke_UNIQUE
# Only after qualification:
bash experiments/baselines/launch_seed42_repeats.sh formal runs/baselines/repeat_formal_UNIQUE \
  runs/baselines/repeat_smoke_UNIQUE/smoke_qualification.json
```

Engineering smoke executes ten independent processes on one Train120 engineering
task each, with the real original frozen assets. Its completion, distinct API
request IDs, immutable assets, no-learning policy, source/config/runtime identity
and memory reserve are checked before a source-bound receipt is issued.
It does not consume another formal Test134 repeat. Unit tests cover historical
wire budgets, prompt executor reuse, freeze tampering, accounting, process
isolation, durable resume and report publication.

Formal launch returns after starting ten detached sessions. The terminal may be
closed, but WSL/the computer must remain running. Do not launch another copy of
the same study unintentionally. All child PIDs/log paths are in launches.json.

```bash
# Resume only an interrupted round, not the other nine:
bash experiments/baselines/launch_seed42_repeats.sh resume \
  runs/baselines/repeat_formal_UNIQUE/b3_skillopt_repeat_1 b3_skillopt 1 \
  runs/baselines/repeat_smoke_UNIQUE/smoke_qualification.json

# Nonblocking status; optional aggregate uses original + two additional rounds.
PYTHONPATH=src:. .venv_b5_gepa/bin/python -m experiments.baselines.repeat_test.status \
  runs/baselines/repeat_formal_UNIQUE --write-report
```

Each round writes REPORT.md, test_report.json, progress.json and completion.json;
task rows/provider calls are in seed_42/test. A failed round's log and timestamped
failure JSON retain the cause. Partial API work is not reused as a response;
committed completed episodes are reused only when resuming that same round,
never between repetitions. Report-only recovery does not repeat model work.

Successful-response token metrics exclude infrastructure failures but include
valid responses in ultimately unsolved tasks; reasoning is already included in
completion. All physical attempts remain separately auditable, with unavailable
usage null rather than zero. B4 readonly retrieval/reranking is target runtime
cost, not learning. The optional repeat_comparison.json is deliberately labelled
as inference variance for one frozen asset, never as a new three-seed result.
