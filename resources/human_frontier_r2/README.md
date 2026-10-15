# LiveMath HumanFrontier R2

Independent external human reference, not naturally learned Ours assets.

`upstream_skillopt` preserves blob `2d1ac842841f771f4867aea5bfb2fc257fa806d3`.
`upstream_skillopt_lite` and its identical compilation `benchmark_prior` preserve
blob `6ddea929aa6068501f5f4e7b07dc881b6206bc78`. Original bytes and MIT licenses
are archived in `sources/` and `licenses/`. Every variant has six contiguous
Atomic sections and an explicit ordered Composite. The assembled text is
byte-identical to its source, including exceptions and original answer wording.
A separate neutral answer-format suffix uses the current displayed label.

The default campaign executes Train12 × NoSkill/SkillOpt/Lite, then frozen
Test100 × benchmark_prior/SkillOpt. The Train selection is a fixed public-input
hash, balanced 6 meta / 6 non-meta when available. Arm order rotates by task.
No policies are selected or edited using their scores. The Test is a developmental
retest of previously exposed questions, not a new blind evaluation. There is no
matched Test100 NoSkill arm and no claim of superiority over one.

The actual manual import is `integration/frontier.py`, loaded from the immutable
run bundle. Runtime text evaluation uses `human_frontier_runtime.solve`, the
same production Provider and official scorer, and the existing durable request
transport. Each task/arm has an independent parent and episode budget scope.
Completed responses are reused on resume; ambiguous sends are never repeated.
No P7 strategy, 600-token limit or learned-promotion qualification is used.

Parameters: DeepSeek v4 Flash, main enabled/high/65,536, 600 seconds, 128,000
tokens per task; one invalid/empty-answer finish at 2,048 with thinking disabled;
zero HTTP retries. Every main and finish request includes the selected full
policy exactly once. All paid arms use programs=off and graph=false. Composite
execution is prompt_compiled; it does not count as Program or graph node calls.
The two original optional programs remain in `optional_programs/`; digits is
bounded to 0–30, and finite examples require a task-related scope.

Campaign CLI (from a committed native checkout with PYTHONPATH=src):

```bash
python -m atomic_skillgraph.experiments.run_human_frontier_r2 prepare --root RUN \
  --base-config CONFIG --datasets CANONICAL_LIVEMATH \
  --ledger CUMULATIVE_LEDGER --history-run COMPLETED_HUMAN_RUN
python -m atomic_skillgraph.experiments.run_human_frontier_r2 check --root RUN
python -m atomic_skillgraph.experiments.run_human_frontier_r2 execute --root RUN --env-file ENV
python -m atomic_skillgraph.experiments.run_human_frontier_r2 monitor --root RUN
```

`execute` resumes only the exact saved source, plan, banks and requests. It never
overwrites an older experiment. The bundle also retains the native single-variant
entry `python integration/frontier.py --variant benchmark_prior ...`; the frozen
campaign command above is the supported paid entry for this revision.

Frozen stage caps: development 3M/72 HTTP; each Test arm 6.3M/200. These are
disjoint envelopes within the original cumulative 45M/9165. At preparation the
ledger must fit all three envelopes, with old unresolved reservations retained.
The phase allocations are based on all 100 historical task costs plus >2× mean
headroom; they are not a guarantee that every task consumes its 128K ceiling.
No phase borrowing, mid-test arm changes or answer-driven retries are allowed.

Observation requirement: all 36 development episodes legally scored/metered and
at least 20 normal Test episodes in each arm. All failures remain in denominators.
`progress.json`, append-only `events.jsonl`, phase budgets, full raw usage, policy
identity and frozen Bank digests are retained. Historical artifacts are read-only.
