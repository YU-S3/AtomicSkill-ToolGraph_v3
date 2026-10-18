# LiveMath HumanFrontier R3 — editable candidate Bank

**Status: new authored CANDIDATE, NOT model-tested, NOT measured improvement.** Derived from the original MIT-licensed SkillOpt / SkillOpt-Lite strategy and the already exposed R2 developmental Test. It cannot serve as a new blind-Test result or as naturally evolved Ours.

## What is included

- `variants/r3_score_prior/`: three reusable Atomic guidance assets (common / meta / non-meta), two prompt-compiled Composite workflows, matching `bank.json` and structurally compatible `bank.sqlite3`.
- `variants/r3_theorem_reasoning/`: alternative meta reasoning policy with the same non-meta discipline. It does **not** force the benchmark-construction prior.
- `integration/r3_view.py`: route on public choice text only; if exactly one meta-option appears, assemble common + meta, otherwise common + non-meta. No extra LLM call. This is a **text Bank**, not an LLM graph execution.
- `tools/import_native.py`: requires the user's real `atomic_skillgraph` repository and calls `Bank.put()/freeze()`; never overwrites prior Bank.
- `tools/meta_prior_diagnostic.py`: public-text classifier for the **separate optional deterministic benchmark-prior diagnostic**; it is NOT automatically executed in the text Bank.
- `tools/counterfactual_replay.py`: post-hoc R2 Test100 arithmetic, not new measured accuracy.
- `sources/` and `licenses/`: original published text and MIT notices.

## Native import (zero API HTTP)

```bash
python tools/import_native.py --repo /path/to/AtomicSkill-ToolGraph_v3 \
  --variant r3_score_prior --output /new/r3_score_prior_bank
python tools/import_native.py --repo /path/to/AtomicSkill-ToolGraph_v3 \
  --variant r3_theorem_reasoning --output /new/r3_reasoning_bank
```

## The exact runtime handoff

In an **isolated copy** of the user's existing R2 campaign, change only the content variant and selected view class. Keep `human_frontier_runtime.solve()` unchanged (enabled/high, 65,536 main completion, 128,000 parent tokens and original finish behavior):

```python
from integration.r3_view import R3View
from atomic_skillgraph.empirical.human_frontier_runtime import solve
from atomic_skillgraph.empirical.bank import Bank
import json
bundle=json.load(open('/new/r3_score_prior_bank_parent/variants/r3_score_prior/bank.json'))
bank=Bank('/new/r3_score_prior_bank', readonly=True, seed=42)
view=R3View(bank,bundle,arm='manual',programs='off')
# Original R2 runner still owns task, adapter, llm, output folder, Governor and metering.
# solve(task, adapter, view, llm, episode_output, governor, phase=..., arm=...)
```

The illustrative bundle path above must be replaced by the actual extracted package path. The frozen campaign runner has hardcoded R2 variant names and source hash locks: **do not drop R3 files into an ongoing R2 run**. Add R3 variant selection to a new runner version before any paid requests. Preserve all R2 files and billing.

## Readout and important boundary

- R2 actually measured 57/100, including meta 35/55 and non-meta 22/45.
- Published construction produces a distinctive meta-option. In the exposed R2 Test100 all 55 meta cases have that option as gold. A deterministic public-text meta selector substituted into the **existing stored R2 predictions** yields 77/100 by post-hoc scoring (55 + 22). This is **not an observed R3 result**, not proof of math reasoning and not a benchmark-generalizable test guarantee.
- Merely writing a stronger instruction in `SKILL_meta.md` may still be overridden by the target model. For a guaranteed meta branch, actual Host/Program execution must explicitly use the optional deterministic selector, labelled as benchmark-prior / exploitation. It must not be silently rebranded as mathematical reasoning or Ours learning.
- The 23 nonmeta misses are mathematically substantive. Twenty of them were missed by both SkillOpt and SkillOpt-Lite. Do not claim one more prompt can guarantee a large increase. Use a few TRAIN questions to compare the candidate Bank before further paid evaluation; do not choose based on repeatedly observed Test100.
