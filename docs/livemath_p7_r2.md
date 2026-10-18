# LiveMath P7-R2

This is an explicit branch of the existing P7 method, `livemath.effect-evolution.p7.r2`.
Old P7 contracts and already frozen runs retain their original semantics.

R2 reuses the complete canonical Train60 history, including successful and failed episodes,
without re-solving them to produce learning evidence. The original 28/12/20 membership is
preserved as provenance. The old feedback12 is a development set, not an independent holdout.
Public questions and choices are never truncated. Actual process fields are typed; missing
process remains missing, and bounded text excerpts carry their full hash and byte offsets.

The single applicability object is `all_tasks` (empty condition lists) or `when` (positive public
conditions). Candidate categories are open strings. Procedures, benchmark priors and mixtures
have explicit identities. Fixed answer letters, physical-ID lookup and external human-bank
imports remain disallowed. The historical redundant global `comparable_conclusions` guard can
be migrated with an explicit receipt; real conditions cannot silently disappear.

The learner makes one induction per byte-bounded batch (the actual Train60 fits one), with an
optional single consolidation if necessary. Valid siblings survive a structural error. One
structural repair is separate from each substantive edit attempt, with every HTTP charged.
There are at most two substantive edit rounds and at most three evaluated policy versions.
Each policy retains an ordered roster of up to three candidates. Runtime injects at most one
global and one applicable local, within the unchanged 600-token combined card allowance.

Feedback is interleaved off/on by task. Later policy versions explicitly reference the same
new R2 NoSkill baseline responses; those responses are neither sent nor charged again. Actual
main Provider payloads establish intervention identity. Same-policy sampling wins and losses
are grouped symmetrically. Editor input contains full public tasks, actual policies and
selection, answers, correctness, finish states and intervention/meta slices. No Val/Test
feedback enters induction or editing. Ranking uses intervention net wins, intervention losses,
full-set net wins, token cost and candidate IDs, not small-subset percentages.

Candidate versions remain in the candidate store and active Bank. The chosen deployment is
frozen as `experimental_not_claimed_effective`, retaining `state=candidate` and `qualified=false`.
The dedicated R2 read-only path evaluates Val17 off/on even if observed training gain is zero.
There is no new Gate20 and no automatic Test100. Val's previous development exposure is disclosed.

Runtime stays DeepSeek high/enabled, main completion 65,536, one bounded enabled/low finish
16,384, timeout 600 s, parent 128,000 and reasoning tail 16,384 bytes. Episode scope includes
phase/arm/policy/task. Persisted received-response journals allow recovery after billing but
before the final trace commit; ambiguous sends are never blindly repeated.

The new allowance is the old P7 remainder, not a fresh 6.2M:

| Phase | Tokens | Physical HTTP |
|---|---:|---:|
| Learning, including repairs | 647,694 | 16 |
| Feedback, including unused old Gate allowance | 2,228,598 | 205 |
| Val off/on | 1,500,000 | 68 |
| Total | 4,376,292 | 289 |

Historical P7, HumanFrontier and the earlier LM1 ledger are deduplicated by physical request ID.
Unknown old reservations retain their original finite authorization and are not actual usage.
The cumulative limit remains 45M / 9,165. Stage ceilings do not borrow automatically. Actual
historical episode distributions support these allocations; worst-case completion is not
guaranteed. Admission refuses a request that cannot fit its original stage/parent limits.

CLI (use a clean committed checkout and a new output directory):

```bash
python -m atomic_skillgraph.experiments.run_math_effect_p7 prepare \
  --config configs/livemath_effect_p7_r2_seed42.yaml \
  --output "$OUT" --source-run "$SOURCE" --datasets "$DATA" --zero-http
python -m atomic_skillgraph.experiments.run_math_effect_p7 check --output "$OUT" --zero-http
# A plan-bound paid_authorization.json records the user's 2026-10-16 authorization and exact caps.
python -m atomic_skillgraph.experiments.run_math_effect_p7 execute --output "$OUT" --env-file "$ENV_FILE"
python -m atomic_skillgraph.experiments.run_math_effect_p7 monitor --output "$OUT"
```

The implementation delivery requires all 12 first feedback pairs to complete normally and all
17 pairs of any started Val stage. Invalid answers remain results but do not count as normal
delivery episodes. `progress.json` is atomic; monitoring is read-only. Implementation readiness,
paid startup observed, phase completion and observed accuracy improvement are separate fields.
