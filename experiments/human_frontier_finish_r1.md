# HumanFrontier manual Bank: finish protocol patch

The supplied `OfficeQA_HumanFrontier_Bank_20261015` bridge appended authored policy and Program cards to every Runtime call, including tool-free finalization. Native finalization had already fitted the evidence to its complete 16,384-byte wire limit. The later injection inflated 32 real requests to roughly 25 KB, causing local refusal and empty scored answers.

`patch_human_frontier.py SOURCE DESTINATION` creates a separate hash-verified package copy. It changes only `integration/frontier.py`:

- Preserve `finish_only` / `finish_answer` prompt and material exactly, with no authored additions.
- Retain authored strategy during normal execution, followed by the native request's submission protocol. Authored answer-tag suggestions do not replace the actual ToolCall interface.
- Record the patched runtime version and original/new bridge hashes. Bank contents, program sources, promotion rules, token caps and repair limits remain unchanged.

The source package is external and is not vendored or reclassified as naturally learned Ours. Existing paid runs remain immutable; corrected evaluation uses a new directory and inherited physical billing entries.

Offline regression uses the production Provider with intercepted transport. Set `HUMAN_FRONTIER_ORIGINAL` to the original package, `HUMAN_FRONTIER_ARCHIVE` to the paused 39-task run, and optionally `HUMAN_FRONTIER_EVIDENCE` to a JSON output path, then run `pytest tests/test_human_frontier_finish.py`. It replays all 32 saved refusals from their original prepared evidence and checks unchanged evidence, actual serialized bytes, tool-free finalization, the unchanged 2,048 output cap, and Bank identity. No live model requests or official rescoring are involved.
