# OfficeQA V6

V6 prioritizes short Train-derived guidance and matched task utility. It does not modify the
official scorer, corpus, legacy V3/V4/V5 qualification rules, or historical runs. A policy may
be useful with zero new Programs. This implementation does not batch-build Programs before
policy utility is demonstrated.

`configs/officeqa_effect_v6_seed42.yaml` freezes the independent method, limits and gate.
The original canonical Train120 is partitioned by SHA256(physical key + method version):
80 induction, 20 edit feedback, 20 protected Train gate. The first four feedback keys form P1.
Only induction traces reach the initial Learner. Full immutable evidence is indexed separately;
model input uses declared first/last tool excerpts, followed by source-checked natural summaries.
No authored Bank, published checkpoint, Val/Test answer or held-out historical trace supplies a policy.

The initial policy has 2–4 steps, one registered public applicability condition and a fallback.
At most one local add/delete/replace edit is permitted after feedback, if its extra comparison
phase is authorized. Rejected edits stay in Learner-only memory. Both arms use the real
EmpiricalSystem Planner → Executor tool loop and the same 48 native-call / 192K-token budget,
including for held-out Train tasks. Both arms get a planning opportunity; there is no LLM router.
Planner and Executor receive the same active policy. Unrelated node wording cannot reintroduce
an inapplicable policy through lexical fallback. Native retrieval and temporary Python remain
available, under the existing sandbox and shared per-task tool budget.

Host hit references retain producer event/result IDs, exact protocol path and corpus/result hashes.
Path and offset are derived together. A unique historical hit/read match is an observed handoff,
not a Program utility qualification. A read in another file alone is not called a model error;
an explicitly claimed cross-file binding is invalid. Missing read controls return needs_input
without a Worker invocation. Engineering Docker probes never become learned assets.

The v1.1 learning contract uses forced `submit_policy` tool calls and distinct initial/edit
schemas. A stopped pre-pair first candidate may be followed by one explicitly identified child
run. Its existing summaries are replayed only when the exact outgoing payload hash matches;
all physical charges carry forward and the same 400K/30-HTTP learning ceiling covers both runs.
This consumes the second candidate opportunity, so no further feedback edit is available.

The utility gate requires wins > losses, losses <= 1, and no increase in empty answers.
Ties retain the parent. Protocol, frozen-bank, task and source identities are checked; official
hard/soft scores, physical HTTP usage, guidance exposure and raw traces remain separate.
P1 acceptance is only a development selection. A positive independent 20-task Train gate is
required before freeze and Val. Test additionally needs explicit authorization and positive Val.
Val has prior development exposure; authored-reference Test exposure must also be disclosed.

Run from a fixed clean WSL Git checkout with PYTHONPATH=src:

```bash
python -m atomic_skillgraph.experiments.run_office_effect_v6 prepare \
  --output "$RUN" --v5-run "$V5_RUN" --config configs/officeqa_effect_v6_seed42.yaml
python -m atomic_skillgraph.experiments.run_office_effect_v6 check --output "$RUN"
python -m atomic_skillgraph.experiments.run_office_effect_v6 authorize \
  --output "$RUN" --approval "$APPROVAL_JSON"
python -m atomic_skillgraph.experiments.run_office_effect_v6 pilot \
  --output "$RUN" --env-file /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env
python -m atomic_skillgraph.experiments.run_office_effect_v6 report --output "$RUN"
```

The approval JSON must bind the actual manifest hash, user authorization, approved phases and
any sibling HTTP reallocation. Full sibling token reservations remain in force. An explicitly
isolated stopped sibling retains all unknown-billing entries and their upper-bound reservations;
any sibling ledger change stops V6. Unknown billing in V6 itself stops admission. RequestStore and
the production task checkpoints preserve physical response identity on resume. No cross-version
resume or retry-until-success is allowed. The initial P1 allocation is learning 400K tokens/30 HTTP
plus four paired tasks 1.536M tokens/16,840 HTTP; the latter is the full loop/retry theoretical cap.
Revision, protected gate, Val and Test are separate authorizations and cannot borrow this allocation.
