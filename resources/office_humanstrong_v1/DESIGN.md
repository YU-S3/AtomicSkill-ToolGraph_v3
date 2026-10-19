# OfficeQA HumanStrong Programmatic Atomic — Implementation and Debug Design

## Objectives / non-objectives

Create an independently editable human Bank that supplies actual OfficeQA-relevant capabilities instead of adding more verification layers. It should be **possible** to learn its procedures from Train in a future Ours variant, but it is not an Ours train output. Main priority: valid source material reaching the final answer, not the Program count. Do not guarantee or pre-fill superior ANLS/EM.

## Asset responsibilities

| Layer | Assets | Input/output | Real boundary |
|---|---|---|---|
| Atomic guidance | 11 | question + actual observed operation state → bounded task-specific procedure | The guidance is not a theorem or value oracle; up to 3 cards are selected by phase & literal public terms |
| Program | 5 | explicit typed inputs → actual local results via Docker | `ok` is local success only; no whole-task certification |
| Implementation | 5 | binding one Atomic to its precise Program digest | Does not assert the program was called/consumed by a particular episode |
| Composite | 4 | ordered dynamic/program nodes with `after` and `from/field` dataflow | Stored valid structures; require an explicit graph executor, do not silently become auto-executing |
| Runtime bridge | `integration/frontier.py` | existing native `tool_loop`, actual observed evidence, result refs | Manual only; no hidden gold or global `Bank.freeze` waiver |

## Runtime selection

`integration/selection.py` selects two phase cards plus at most one matched domain card, using **only the visible question and actual Runtime progress**. In every normal step a short universal output rule accompanies them. 

- `locate`: target + bounded search (+ specific domain).
- `extract`: target + exact table (+ periods/units/finance if relevant).
- `compute`: table + statistical operation (+ actual domain-specific operation).
- `finalize`: commit only; the actual `finish_only` prompt is **plain text without any tool call**.

The project V3.1 evidence state and candidate/scalar ResultRef projection are reused; no separate LLM router. Program cards are advertised only for their plausible phase. A few tasks that already have sufficient information should finish without touching a Program.

## Program contracts and exact responsibility

### 1. Scoped search

`scoped_search.py` uses the existing native `glob`, `grep/grep_page`, `read/read_context` RPC surface, and binds each offset to its own real path. `ok` when windows acquired, but `target_periods_complete=null` and `table_header_bound=false`. A bounded hit does not prove the right table.

### 2. Table extract

`table_extract.py` requires the exact `rows`, `column`, `unit` and a bounded *already acquired* table window. Missing/ambiguous OCR returns `needs_input`. `target_periods_complete=true` only for **the explicitly passed rows**: the question's full range remains a separate responsibility.

### 3. Period coverage

`period_ledger.py` input `required_periods[]`, `observations[{period,value,unit,source_ref}]`, output `{complete, observed_periods, missing_periods, conflicting_periods, aligned_observations}`. This program cannot verify that `required_periods` were extracted correctly from the question or that the values match documents. Its positive status denotes consistent bookkeeping of explicit inputs only.

### 4. Series metrics

`series_metrics.py` reuses the previous typed arithmetic API: difference requires `from_value`/`to_value` and explicit `absolute`; ratio requires `numerator`/`denominator`; growth requires `base_value`/`new_value`, `periods` where relevant, and `output_percent`. Preserves sample/population stats, correlation, OLS, CAGR, log growth and Gini variants. The runtime chooses an actual proven statistic and binds sources.

### 5. Treasury transforms

`treasury_transform.py` adds `unit_scale`, `price_32nds`, `currency_convert` and `box_cox`. Examples: `99.27` is converted to `99+27/32` only under an actually observed `thirty_seconds` quote convention. `foreign_per_usd` divides a foreign amount by the rate; `usd_per_foreign` multiplies. Converting dimensions before Box-Cox is a human/Runtime precondition, not magically handled by the worker.

## Composite examples

- `search_to_anchored_read`: dynamic narrow query → real `scoped_search` → dynamic table/text interpretation.
- `table_to_numeric`: dynamic bounded complete table window → real `table_extract` → dynamic target-specific compute and final answer.
- `complete_time_series`: dynamic acquired period ledger → pure `period_ledger` → dynamic missing-period continuation or statistics.
- `array_statistic`: dynamic source-grounded array → pure `series_metrics` supported array statistic (sum/mean/variance/correlation/OLS) → dynamic answer confirmation. For difference/ratio/growth use the **named input schema directly**, not an incompatible positional `values` mapping.

**Integration caution**: the native `tool_loop` does not execute stored Workflow by virtue of `bank.json` containing it. End-to-end graph dispatch needs explicit activation and real dataflow, and can increase model calls. No fake “graph executed” metrics.

## Next test decision, without endless patching

1. Import the Bank on a fresh directory and run zero HTTP `inspect_bundle`/unittest. On Linux run the optional actual Worker smoke for three pure programs. No need to generate more candidates.
2. Compare **unmodified SkillOpt**, **unmodified SkillOpt-Lite**, **current V3.1**, and **this HumanStrong Bank** as four independently labeled reference text/Bank conditions on a **small locked Train development set** when authorized. The original upstream text arms require a separate faithful prompt adapter; do not secretly mix oracle inputs or file paths from upstream with the current corpus. When scores differ, inspect actual tool choices, period completeness, values, finish reason, and exact output.
3. Track paired wrong→correct, correct→wrong, valid finish, Program actually shown/run/used, mean Test-equivalent tokens/HTTP; never combine distinct model/harness scores. If only format improves and difficult content retrieval remains poor, stop generic Bank editing. Focus on data/table representation or modeling ability instead.
4. Existing Test102 has been inspected repeatedly. New runs on those tasks are development-exposed rechecks, not newly blind holdout results. No Test score appears in these Bank assets.
