# OfficeQA HumanStrong v1 — Programmatic Atomic SkillBank

**Purpose:** human-edited *reference upper-bound candidate* (not learned Ours; accuracy not yet measured). Intended for the existing offline OfficeQA Full-derived train/val/test protocol and native `tool_loop` on `YU-S3/AtomicSkill-ToolGraph_v3`.

**Source code compatibility verified offline:** exact submitted repository bundle `b4a518c1ad3eefe6f8c963aefdf0566d54977805` (Oct 18), and public main inspected at `4cfa14296317a0900b2a832679b9e3fe0bb0f52b`. Re-check on your new checkout; no assumption a moving main is identical.

**Assets:** **11 Atomic + 5 Program + 5 Implementation + 4 Composite**, all in `bank/bank.json`; `bank/bank.sqlite3` is a read-only portable snapshot for inspection. The native import will create a new SQLite Bank, remap environment-dependent Program identities and linked Workflow/Implementation references. Nothing overwrites V3/V3.1 runs or their Bills/Score receipts.

## Contents

```text
bank/bank.json, bank/bank.sqlite3
skills/*.md                     11 authored Atomic procedures
programs/*.py                  5 real run(ctx, inputs) implementations
implementation/*.json          exact Atomic↔Program bindings
workflows/*.json               4 empirical.workflow.v2 compositions
integration/frontier.py        isolated V3.1-compatible manual view
integration/selection.py       deterministic phase/domain router
sources/SOURCES_AND_EDITS.md   original strong-skill references, adaptation rationale
licenses/                     upstream MIT notices
SKILL.md                       human-readable concatenated policy, NOT the Runtime source
tools/import_native.py         actual project Bank.put import, no model calls
tools/inspect_bundle.py        static contracts, cross-ref checks, no model calls
tools/check_worker.py          optional *real Docker* pure-Program smoke
 tests/test_offline.py         ten zero-HTTP tests
```

## Safe import: required inputs and exact commands

The source experiment checkout must have a resolved JSON config with `harness.adapter=officeqa`, a matching `program_environment` and `experiment.seed=42`. Do not pass an unexpanded YAML.

```bash
cd /path/to/OfficeQA_HumanStrong_ProgrammaticAtomic_v1_20261019
python tools/inspect_bundle.py
python -m unittest discover -s tests -v

python tools/import_native.py \
  --repo-root /path/to/AtomicSkill-ToolGraph_v3 \
  --config /path/to/ACTUAL_resolved_officeqa_config.json \
  --output /new/office_humanstrong_v1_bank

# On the user's Linux server, with the locked image already present:
python tools/check_worker.py \
  --repo-root /path/to/AtomicSkill-ToolGraph_v3 \
  --config /path/to/ACTUAL_resolved_officeqa_config.json \
  --bank-dir /new/office_humanstrong_v1_bank
```

The offline 10-unit suite and real native Bank.put into a new readonly Bank passed in this authoring environment. **The real Docker Worker check has NOT run here**; call the third command on the Linux environment with Docker.

### Connection to the real OfficeQA eval Runner

The repository already has `EmpiricalSystem(..., bank_view_factory=...)`, native `Executor`, `BudgetGovernor`, and a V3.1 run script. To run this reference Bank, use a **new copy** of the V3.1 runner with only these changes:

1. Set package path to **this** directory, use `integration/frontier.py` rather than the V3.1 view, and import a fresh Bank into each distinct arm directory via `frontier.native_import(bundle, output/bank, resolved_config['program_environment'], office_method, seed=42)`.
2. With `frontier.bridge()` active, use `frontier.HumanBankView(bank, imported_bundle, arm='manual')` for the human arm; `arm='off'` for the exact matching NoSkill arm. Native `run_empirical.run` should use the same resolved config, tasks and corpus for both. The supplied `frontier.py` is a portable **view bridge**: **do not modify globally** the core Ours qualification, scoring, tool registry or BudgetGovernor.
3. Use the canonical `data/main_experiment_v1/officeqa` Train120 only for paid development. Keep NoSkill and manual on the same tasks/budget; preserve the actual individual responses and scoring. **Do not import test gold or oracle source_docs into the agent materials.**
4. Don't automatically enable any full graph. The default mode exposes phase-selected Atomic and permitted local Programs through the original `tool_loop`; Composite assets are present for manual/programmatic integration when the caller explicitly selects a valid workflow. The four workflows have real `from/field` edges but are not automatically executed by adding the JSON to the Bank.

**Never** set fake `local_validation` or report normal learned `usable` for these authored Programs. They enter the *separate manual view* via `author_trusted`; the ordinary Ours Bank uses its original qualification logic.

## How the new Bank differs from V3.1

- **Phase-aware guidance** remains, but each Runtime gets only up to 2 phase skills + 1 specific domain skill. Domain inference is entirely from the **public question**, not scores or task IDs.
- The broad OfficeQA-specific rules expand from generic “be careful” to explicit financial instrument/accounting roles, fiscal-year changes, table hierarchy, Box-Cox, units, growth and estimation boundaries.
- `scoped_search.py` and `table_extract.py` still retrieve/acquire real evidence. Success only means local operation succeeded; it never certifies the whole question.
- `period_ledger.py` (new) preserves an explicit source-referenced period→value list and reports missing/conflicting periods. It never fills missing observations and does not fetch pages on its own.
- `treasury_transform.py` (new) implements unit scales, 32nds quotation, Box-Cox and exchange orientation with exact Decimal. The caller supplies the real document interpretation.
- `series_metrics.py` retains named formula operations. No new ARIMA claim: the Bank explicitly avoids treating OLS as an ARIMA substitute.
- The normal `runtime_step` guidance may request a native finish. The actual `finish_only` material instead has **plain-text/no-tools** instructions; it does not ask a model to call a tool that is unavailable in that phase.

## Data/score/budget boundaries

- **Current fixed main scope:** Train120 / Val24 / Test102 (OfficeQA Full-based), offline Treasury text; no oracle pages, no unrestricted web, no new visual input. Normal evaluation uses the project's existing scorer, not a new grader.
- This is a reference **candidate**, not proof of accuracy. Don't claim SkillOpt, SkillOpt-Lite, EvoSkill, EnvHarness or Genesis reported accuracy as performance of this Bank. Their models/harnesses/splits differ.
- Historical V3.1 Train12: Human5/12, NoSkill4/12 (one formatting-only win), 7/12 human finish_only and 3 scoped-search calls. That is the reason for this redesign, **not** the new Bank's score.
- Historical budget snapshot (2026-10-19): cumulative **93,027,247 / 95,000,000** used or reserved, incl. unknown-billing reservation; remaining **1,972,753** at that snapshot. Reconcile with your current ledger. **This ZIP does not authorize spending above the cumulative ceiling or another Test102**.
- If code or the cumulative budget has changed, don't reuse old run IDs/Bank state in place. Native readonly import is zero model HTTP; paid comparisons require a fresh authorized run directory and real completion monitoring.

See `DESIGN.md` for the four-layer roles and `sources/SOURCES_AND_EDITS.md` for citations and editorial differences.
