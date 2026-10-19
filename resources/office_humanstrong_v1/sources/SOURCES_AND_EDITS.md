# Source provenance and editorial decisions

This is a **human-edited reference**. It is NOT an asset naturally learned by SkillCompiler and it has no measured Test accuracy yet. Neither hidden gold answers, private answer files, Test task identities, nor Test-specific file/page pointers appear in any runtime skill or Program.

## Original public skill checkpoints

1. [Microsoft SkillOpt GPT-5.5 OfficeQA checkpoint](https://github.com/microsoft/SkillOpt/blob/main/ckpt/officeqa/gpt5.5_skill.md), Git blob `869c1d4865ce75713825ab7b53fa3a4410371bf3`. Transfer: strict source narrowing, table row/header/unit alignment, full period ledgers, Treasury tenders/refunding roles, quotation basis, formulas, final format. **Do not transfer** its oracle parsed-pages rule or external official search to this offline benchmark, because those sources are not provided in the current runtime.
2. [SkillOpt-Lite GPT-5.5 OfficeQA checkpoint](https://github.com/EvolvingLMMs-Lab/SkillOpt-Lite/blob/main/skillopt_lite_ckpt/officeqa/gpt5.5/skill.md), Git blob `b320a2dae089c7186e69b753a951a3beba116fb8`. Transfer: period-direction, growth intervals, percent *scale* versus percent *glyph*, numeric finish discipline, format-only review. **Edit**: do not require a redundant grep/read if correct evidence already exists; never request `<answer>` XML when the native tool_loop requires `runtime_step(action="finish")`.
3. [EvoSkill OfficeQA paper](https://arxiv.org/abs/2603.02766), authors report 60.6% to 67.9% in their Claude Code setting. Transfer design intuition: learned source/cell validation and quantitative procedures, not any hidden golds or undeclared tools. Their benchmark/model/harness differ.
4. [Genesis Treasury skill](https://github.com/kingmz816/Genesis/blob/main/skills/officeqa/SKILL.md). Domain ideas (re-expressed, not copied): FY1976 transition, issue versus maturity, stock versus flow, 32nds quotations, Box-Cox, department versus function, multilevel headers. Genesis's published MCP deployment contains missing tools / an unresolved placeholder; **no Genesis MCP operation is assumed available here**.
5. [Google EnvHarness OfficeQA](https://github.com/google-research/envharness/tree/main/experiments/officeqa), authors report NoSkill EM54.23, original environment skills 54.40, EnvHarness skills 56.20, on their Gemini-based 172-task evaluation. Transfer: stage-aware tool environment / information handoff, not direct scores.
6. [OfficeQA Pro paper](https://arxiv.org/abs/2603.08655) and [databricks/officeqa repository](https://github.com/databricks/officeqa): source representation and correct table/layout association matter; Pro133 differs from Full246. This Bank is for the project's Full-based fixed Train120 / Val24 / Test102 split only.

All claims of external performance are **author-reported in their own settings**; they are not measurements for this Bank. The SkillOpt/SkillOpt-Lite repository notices and MIT texts are carried in `licenses/`. Genesis is cited for ideas rather than redistributing its authored text or incomplete MCP code.

## Internal evidence and limitations

Input: `OfficeQA_HumanFrontier_V31_Bank_Traces24_20261019.zip`, 12 Train pairs, last source fix b4a518c, prior Human Bank commit c9022bd. Human 5/12 versus NoSkill 4/12; sole win is numeric formatting; 7/12 human finish_only, 3 scoped-search Programs (two with windows), zero table/compute Programs. The bundle contains previous human Bank, original traces, requests and accounted usage; no Test gold is used to create the new assets.

Structural changes from V3.1: 11 separate Atomic text capabilities; full Treasury semantics / fiscal / stat families selected by publicly visible question terms; add two pure Python operations (period ledger, finance transforms) on top of real scoped search, table extraction and numeric metrics; 5 real Skill→Program Implementations and 4 explicit `empirical.workflow.v2` graphs. All Programs return *local* results and do not claim whole-task correctness. The default tool_loop does **not** automatically take over these stored graphs.

## Runtime API adaptation

- Read **Bank guidance** from `bank/bank.json`; `SKILL.md` is a readable mirror, not the actual Runtime source.
- Keep at most two phase cards plus one specific domain card. Always prepend one brief final-answer invariant. The `finish_only` policy must be plain-text and tool-free; it does not request `runtime_step`.
- Use real `path/offset` hits, `result_id` and a schema-grounded `from/field` handoff. Do not expose test gold or oracle `source_docs` to the runtime.
- A retrieved window or `period_ledger.complete=true` checks an **explicitly supplied** local contract, not whether the question was fully answered. A locally computed value still requires aligned units and operands.
- The prebuilt `bank.sqlite3` is for browsing an exact portable asset snapshot; re-import via `tools/import_native.py` into a fresh local Bank bound to the actual Docker environment.
