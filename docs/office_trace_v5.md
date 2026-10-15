# OfficeQA Trace V5

Method: `office-trace-v5-program-induction.v2`, material `office.operation-fields.v5.1`.
The first paid V5 run exposed omitted output/scope type requirements in the model-visible
contract. Required input references, optional scope references, output types and fixed
retrieval outputs now use the same tables in the exported prompt and Host validation.
Glob filename discovery and grep/read retrieval are explicitly distinct probe contracts.
Schema diagnostics include the actual declared type and required membership.
No oracle, formal promotion or repair limit is relaxed. V5 lives in explicitly versioned
modules so V4 checkpoints, prompts, receipts and qualification semantics remain
reproducible. The ordinary Ours Bank, `record_validation`, `program_eligible`,
Office tool defaults and running reference experiments are unchanged.

`office_program_contract_v5.py` owns the model JSON contract, schema-language
checks, tagged bindings, field-reference type checks and structured diagnostics.
The Extractor prompt serializes that same contract. No binding defaults, list
conversion, inferred table controls or silent extra fields are accepted.
`runtime_required` produces `spec_valid_pending_inputs`; Builder can emit code,
but no source probe is scheduled until real inputs exist. Missing Runtime inputs
produce `needs_input` before the Worker is called.

The new mining module calls the existing public Train projection, then inventories
operation requirements. It preserves original success/failure/handoff labels.
Default preregistration covers ten opportunities from eight physical Train tasks,
including the two reviewed V4 op2/op3 sources. Selection uses public operation
structure and hashes, without final predictions, scores or gold. Two other Train
tasks remain reserved for transfer. Only the first four opportunities are eligible
for the proposed paid phase. A no_change decision is a legitimate negative result.

V5 uses an independent JSON candidate directory, not a formal Bank. The states
`candidate_built`, `worker_checked`, and `transfer_checked` never mean
`formal_usable`. Optional Train pairs expose only transfer-checked candidates.
They separately log invocation, exposure, reading, declared use, exact downstream
binding, correctness and tokens. Graph edges require ordered, exact-value
producer-to-consumer references; the resulting graph is explicitly partial.

The table oracle supports uniquely identified pipe or clearly separated text
tables with an adjacent observed unit and exact row/column labels. Ambiguous
headers, duplicate periods, missing units/cells and unsupported layouts remain
`coverage_insufficient`. It does not infer missing fiscal years or resolve complex
multi-block layouts. Extra corpus-location outputs without an independently
checkable location contract also remain unqualified.

The original Train120 traces, V4 responses and ledger, V3 Frozen and Val summary
are hashed before use and checked on continuation. The source commit, Python
source hash, configuration, permission ABI, corpus and proposed budgets are
frozen. Each model response is saved before decoding. Unresolved HTTP or Worker
intents require inspection and never trigger an automatic duplicate execution.
One repair slot is shared by proposal and Builder validation per opportunity.
Unexpected Host exceptions interrupt the run rather than becoming model errors.

## Zero-model preparation

Run in a clean committed WSL checkout with the project's existing Python venv:

```bash
python -m atomic_skillgraph.experiments.run_office_trace_v5_pilot prepare \
  --v4-run /path/to/completed_v4_pilot \
  --reserved-run /path/to/ongoing_office_reference_run \
  --output /path/to/new_v5_run
python -m atomic_skillgraph.experiments.run_office_trace_v5_pilot check \
  --output /path/to/new_v5_run --zero-http
```

`--reserved-run` records an ongoing Office reference experiment's full remaining
token/request ceiling. Its Bank and content never enter V5 learning. Historical
V4 usage is copied with original attempt identities; sibling allowances are
subtracted from the cumulative limits, not represented as actual spent tokens.

The check replays all twelve old physical responses for diagnostics only, checks
new and repair request capacity, and executes engineering fixtures through the
production Docker Worker on real public Train corpus text. Fixtures are never
written to the candidate Bank. See `zero_http_check.json`,
`zero_http/v4_response_replay.json` and individual probe receipts.

## Paid phases (new authorization required)

The suggested limits remain unapproved: learn 500,000 tokens / 60 physical HTTP;
optional consume 250,000 tokens / 20 HTTP. They do not inherit V4 approval. A new
`authorization.json` must name this method, exact `manifest_sha256`,
`phase_limits`, `approved_phases`, and actual `user_authorization`, with
`approved: true`. There is no authorization creation command.

For a newly authorized implementation revision, pass `--prior-v5-run` for completed
prior V5 attempts. Settled physical attempt IDs are carried exactly once; conflicting
or unresolved billing is rejected. The original V5 phase ceilings are shared across
those attempts and the new run, not reset by a version change. Old responses are
replayed only for zero-HTTP semantic diagnostics, never imported as new candidates.

After that explicit decision, the stages are `learn`, `probe`, and optionally
`consume`, with `--output` set to the same prepared directory. `learn`/`consume`
can load the existing environment file using `--env-file`. `probe` uses Docker,
not model HTTP. Successful responses are reused only within this exact source
and request identity. No V5 Val or Test entry point exists.

```bash
OFFICE_V5_DOCKER=1 OFFICE_V4_DOCKER=1 PYTHONPATH=src python -m pytest -q \
  tests/test_office_trace_v5_contract.py tests/test_office_trace_v4_pilot.py
```

Optional legacy integration tests additionally use `PROGRAM_IMAGE_DIGEST` and
their documented immutable source paths. Do not count skipped tests as passed.
