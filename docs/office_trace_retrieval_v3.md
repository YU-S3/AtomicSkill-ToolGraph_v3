# OfficeQA Trace / Retrieval V3

`officeqa.trace-retrieval.v3` is an independent Train-record derivation. It never resolves Train120 again, mutates a parent run, or learns from Val/Test. Existing V2 qualification receipts and the eight usable Programs retain their original identity. An inherited candidate is still a candidate.

The new `grep_page` and `read_context` tools use authorized relative text paths and Unicode offsets. Pages are stable, bounded to 40 hits and authenticated against the query, scope, limit, corpus metadata and task. The Host cursor key persists outside Bank for checkpoint recovery. `complete=false` means unfinished search. Historical `glob/grep/read` results retain their old ABI.

`verified_retrieval_chain.v1` recovers ordered successful operations from actual Train events, including nested RPC. A unique recorded path/offset pair connects a grep hit to a later read. These are observed value handoffs, not inferred model intent or proof of numeric parsing. Original event/result IDs, action/result hashes and parent source code remain attached. The Learner chooses Host IDs; Host derives schemas, values and selectors. Historical chains compile against their original tools. Retrieval text has an intermediate role.

Qualification checks recorded actions and output, then runs the same candidate in the locked Docker with a changed query, another hit, absent hits and changed scope where applicable. Constant replay and source mismatches fail. Composite publication requires usable Program nodes, actual invocations and exact producer/consumer ResultRefs; partial retrieval graphs remain intermediate.

Source Extractor and ordinary Builder stay at 24 KiB. Only generation 1 with actual prior failure material receives the 32 KiB necessary-repair capacity. Structural repair stays at 16 KiB and the repair count stays one. Repeated source/scopes may use lossless references within the same request. Full reference expansion is tested. Runtime retains 192K tokens / 48 native calls for evaluation, high thinking / 32768 output, an 8192 dynamic floor and the original 16 KiB / 2048 finish reservation.

## Independent entry

Use a committed native WSL checkout, with `PYTHONPATH=src`, the existing Python environment and locked Docker. A Windows-managed Git worktree is not the production WSL checkout.

```bash
python -m atomic_skillgraph.experiments.office_source_recompile prepare-v3 \
  --source /home/yangchengyu/officeqa_source_catalog_v2_seed42_20261014 \
  --output /home/yangchengyu/officeqa_trace_v3_seed42_20261014
```

Without an approved `--budget-revision` JSON this creates a reviewable, non-executable preparation with the existing 85M ceiling. The document's 90M is not authorization. A new approved preparation must freeze the explicit authorization, cumulative limits, `phase_token_limits` (3,000,000 / 4,608,000 / 19,584,000) and `recompile_http_limit=400`. Pending or changed revisions cannot issue HTTP. Historical physical ledger entries are copied exactly once; V3 phase spending is separate from inherited phase spending.

After separate budget authorization and the zero-HTTP gates:

```bash
python -m atomic_skillgraph.experiments.office_source_recompile recompile --output "$RUN" --env-file "$ENV_FILE"
python -m atomic_skillgraph.experiments.office_source_recompile val --output "$RUN" --env-file "$ENV_FILE"
```

Val completion stops for review. Test requires a separate explicit review file approving the exact complete Val summary hash. There is no automatic Test launch. `STOP_AFTER_TASK` is the recompile task-boundary stop marker; checkpoints reuse paid responses without extra generations or double charging.

## Offline verification

`tests/test_office_trace_v3.py` uses the production Provider payload/admission/checkpoint paths with transport intercepted. Set `OFFICE_TRACE_PARENT` to the immutable V2 run, `PROGRAM_IMAGE_DIGEST` to the existing locked image and `OFFICE_TRACE_GATE_DIR` to an external evidence directory. It loads all canonical Train120, checks every recovered chain, preflights every new source, replays the real UID0066 refusal, runs actual Docker RPC/source qualification and exercises derive/recompile/freeze/Val24/Test102 with synthetic responses. Synthetic responses and fixture Programs are confined to temporary test directories; these are engineering checks, not natural learning or accuracy results.

The cumulative known cost before V3 is 60,174,533 tokens / 5,966 HTTP. The proposed full additional phase allocation reaches a theoretical 87,366,533 tokens. No natural candidate count, Program benefit or new accuracy is claimed until the separately authorized paid stages run.
