# SearchQA Train Trace R2

`searchqa.train-trace-capability.r2` derives a new empty Bank from one complete,
immutable canonical Train300 run. It never resolves Train300 again. Old public
answers may support advisory procedures; historical execution remains empty.

The single entry is `python -m atomic_skillgraph.experiments.run_searchqa_trace_r2`.
Pass `--config configs/searchqa_trace_r2_seed42.yaml`, `--source-run`, `--datasets`
and a new independent `--output`. `--stage verify` is zero HTTP. `--stage execute`
runs prepare → learn → worker_probe → train_probe → freeze → val → report and stops.
An optional `--env-file` supplies the existing model credentials; it is not copied
to artifacts. WSL requires the locked Docker image and `bwrap`.

Before paid calls, the runner fixes source hashes, committed clean code, model,
the 240/30/30 physical-ID partitions and independent phase/cumulative budgets.
At most 12 induction units use 2–4 correct public examples each. Actual unique
examples consumed are reported; the 240-member pool is not claimed as 240 model
inputs. Up to four natural code specs receive at most two Builder generations.
Programs remain candidates until real locked Worker source, independent Train
transfer and boundary receipts pass qualification. All transfer failures remain
in the audit. The supported output is an intermediate public text span, not a
claim of correct QA or complete semantic retrieval.

Per the user's latest instruction on 2026-10-14, Train paired utility is observed
without an automatic accuracy/cost release gate. Technically qualified assets
remain unchanged for Val observation; the user decides their utility afterward.
Guidance selects 0–2 cards, at most 1200 body characters. Val uses NoSkill,
Guidance-only, and Atomic-full; it skips redundant arms when no corresponding
capability exists. Frozen hashes must remain unchanged. No paid Test is authorized
by this configuration; Test additionally requires explicit review of the Val
report and its own approved phase budgets.

`--resume` checks the same source/config/code and verifies completed stage
receipts. Paid responses are reused from the new run's own checkpoint only.
Unresolved HTTP billing or a started Worker without its receipt stops recovery.
`status.json`, phase summaries, `budget.json` and `report.json` expose progress,
source costs, incremental costs, paired EM/F1, actual invocation/consumption and
Worker time. Create `STOP_AFTER_TASK` in the root to stop between stages/learning
units; during evaluation also create it in each active arm directory to stop at
the next task boundary.

Offline tests use intercepted HTTP and actual Docker; they demonstrate engineering
contracts, not natural model learning or positive method utility. Real paid stage
artifacts must separately establish those observations. Existing six-benchmark
behavior and the public SearchQA input/scorer remain unchanged.
