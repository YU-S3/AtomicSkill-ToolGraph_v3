# GEPA zero-output usage recovery

The provider observer already accepts a target response with empty visible
content, `finish_reason=stop`, and known positive prompt usage. Upstream SkillOpt
keeps its missing-action fallback. Episode accounting, cache replay and the
phase summarizer must also accept zero completion tokens. Missing, negative,
inconsistent and all-zero usage still fail; empty optimizer output and
completion-budget exhaustion retain their existing failure behavior.

`launch_gepa.sh recover SOURCE_CAMPAIGN RECOVERY_OUTPUT SMOKE_RECEIPT` is an
explicit, audited continuation for the proven usage-validator failure on
controller c4c9766. It requires two completed lanes and one failed lane, an
intact GEPA checkpoint, original source/config/data/upstream/load evidence,
and a real passing smoke receipt for the repaired code. It is not a generic
permission to bypass protocol errors. Source migration through 9ad0870 covers
the previously audited B0 addition and offline replay serialization; subsequent
changes are restricted to this repair's files. Configuration and method
authorities must still match the original campaign.

The recovery writes only to a separate output directory (plus the original
campaign's runtime provider semaphore). It copies the durable GEPA state and
runs only the missing seed, then freezes and tests its chosen skill normally.
Completed seeds are validated and referenced unchanged. The original failure
is not relabeled as infrastructure. Failed-prefix provider costs are recovered
from every immutable sidecar, validated, hashed, and charged exactly once.
Checkpoint replay costs follow the existing GEPA accounting.

The Python entry point accepts `--prepare-only` to exercise all recovery gates
without starting Train or Test. The same output may subsequently be passed to
the launcher. A failed new recovery lane is retained for inspection, not
silently restarted. The formal-method lease prevents B0/B4/B5 overlap.

Successful recovery produces `REPORT.md`, `campaign_report.json` (method,
Train/Val/Test and full attempt costs), `paper_report.json` (three-seed means,
sample standard deviations, per-family metrics), and `completion.json`.
`recovery_receipt.json` and `failed_prefix_usage.json` preserve migration and
cost provenance. No formal completion is published before all three seeds and
their frozen artifacts validate. Original campaign reports remain historical.
