# SearchQA GLM refusal continuation

This directory preserves the tested continuation controller and provider patch. It does not change the main package's Bank logic or overwrite an existing run. The experiment uses the separately frozen release identified in `source_provenance.json`; apply `source_diff.patch` to that parent release, not blindly to the current main branch.

The production copies live in `D:/T3S_exp/deliveries/searchqa_glm53_refusal_resume_20261020`. Configs, canonical tasks, original requests, ledgers and Bank entities remain in their existing local run directories. No API credentials are included here.

- `provider_outcomes.py`: terminal refusals, exact-request conservative reservation receipts, and crash recovery without resending refused requests.
- `campaign.py`: paired scheduling with explicit inherited-prefix reporting and immutable artifact checks.
- `prepare_continuation.py`: validates the parent, preserves completed results and billing, and prepares a distinct child run.
- `check_continuation.py`: eleven intercepted production regressions, including real Docker execution, immutable payload comparison, real refusal replay, and interrupted refusal recovery.
- `source_diff.patch`: classifies explicit provider terminal outcomes before parsing optional usage.
- `offline_acceptance.json`, `startup_verification.json`, and the Chinese report: actual verification evidence and an initial running snapshot, not final Test1400 results.

Both Banks retain the original canonical Test1400 denominator. A completed provider content-filter refusal is scored as an empty answer and explicitly labeled. Missing usage stays unknown and the full reservation remains charged against admission limits. Ordinary missing usage, incomplete streams and inference errors still stop without being scored as model answers. Model payloads and retry settings are unchanged.

The live process uses the frozen delivery copies. Do not edit these copies or re-run preparation over an existing directory. To inspect the current run from WSL:

```bash
watch -n 10 bash /mnt/d/T3S_exp/deliveries/searchqa_glm53_refusal_resume_20261020/monitor_live.sh
```
