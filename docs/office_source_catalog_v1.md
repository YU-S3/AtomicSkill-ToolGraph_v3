# OfficeQA source-catalog v1

This version derives a new Bank from completed canonical Train120 records. It does
not solve Train again or resume the old Val19. Other benchmark configurations keep
their existing policy. Old source requests, Bank, scores and cost ledger remain
read-only.

`experiment.officeqa_method_version: officeqa.source-catalog.v1` selects:

- Host-verified executed Python learning input: 24,576 full Provider wire bytes;
  Builder: 24,576 bytes; structural repair remains 16,384 bytes, one repair.
- Learner binds operation/input/output IDs from the current Host catalog. Host
  resolves values and selectors through the existing `canonical_binding()` check.
  Source validation still uses Broker and the restricted Docker Worker; this
  version does not lower the usable qualification threshold.
- Runtime rejects assignment to `ctx` and `context.call/read_result` misuse with
  model-visible feedback. It does not rewrite generated code. Historical source
  prefixes still replay through their original Broker contract.
- Val/Test: 192,000 tokens and 48 native calls per task. Runtime high thinking,
  32,768 output cap, 8,192 dynamic minimum and 18,432 finish reserve are retained.

The `office_source_recompile` entry creates a separate copied Train Bank and
archives inherited realization jobs. Newly versioned decisions start their own
jobs; no paid old response is reinterpreted under the new schema. Ordering is
fixed by successful Host Python source availability, then other local evidence,
then no evidence; within each group the original seed42 Train order is used.
No Val/Test scores participate. Every unit has a checkpoint, raw requests and a
source hash. A bounded model abstention or rejection remains a negative result.

## Frozen budget and phases

One ledger preserves all inherited physical request IDs and enforces the original
85,000,000 tokens / 480,000 HTTP limit. Additional independent token ceilings are:
recompile 3,000,000; Val24 4,608,000; Test102 19,584,000. Unused phase allowance is
not borrowed. Unknown billing or changed parent source/ledger blocks admission.
Recompile retains the 80,000-token per-record learning pool; exhausting the phase
records `censored`, not successful coverage of all 120 records.

Run from a clean, pinned WSL checkout with the existing Python environment. Replace
`SOURCE` and `OUTPUT` with the old seed42 root and a new independent root:

```bash
PY=/home/yangchengyu/asg_alfworld_venv/bin/python
export PYTHONPATH="$PWD/src"
$PY -m atomic_skillgraph.experiments.office_source_recompile prepare --source "$SOURCE" --output "$OUTPUT"
$PY -m atomic_skillgraph.experiments.office_source_recompile recompile --output "$OUTPUT" --env-file /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env
$PY -m atomic_skillgraph.experiments.office_source_recompile val --output "$OUTPUT" --env-file /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env
```

`recompile --max-new-records N` and `val --max-new-tasks N` stop at a boundary;
calling the same command continues through the recorded checkpoints. Creating
`OUTPUT/STOP_AFTER_TASK` stops recompile; `OUTPUT/val/STOP_AFTER_TASK` stops Val.
Source changes require an explicit independent derivation, never same-version
resume. Generic formal matrix launch is disabled for this method version.

Val completion stops for review. Test is a separate command, requiring an explicit
review JSON with `approved: true` and the exact `val_summary_sha256`. Do not create
that approval without the user's review. An existing Test uses the same checkpoints
and cannot be reset into another evaluation. `frozen_check.json` records Bank hashes
before and after each evaluation invocation.

## Acceptance evidence

The real archive regression verifies all 74 indexed files, then rebuilds the 25
Extractor and one Builder refusals with production modules and intercepted HTTP.
It preserves executed source and catalog selectors. A second zero-HTTP test uses
all 15 original Val requests at their recorded spending boundary to compare 96K
and 192K admission, retaining the conservative UTF-8 input bound and finish reserve.
Fixture transport results are not evidence of natural learning: report real
Builder, source-output-match, usable state, non-source consumption and Frozen
immutability separately from these engineering tests.
