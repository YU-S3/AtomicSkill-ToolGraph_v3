# Local integration, 2026-10-19

The user supplied this Bank and authorized a new fixed canonical Train12 paired
HumanStrong/NoSkill observation: cumulative 98,000,000 tokens, at most 4,608,000
new tokens and 2,400 physical HTTP requests. No Val/Test or extra baseline arms.
Each episode keeps 192,000 tokens and 48 native calls. Prior bills and all unknown
reservations remain in the cumulative ledger. Each new episode has its own Bank.

All original archive members passed FILE_SHA256S.json. That file remains the
**original archive** manifest. integration_receipt.json records the only local
bridge change: finish_only now uses the supplied FINISH_ONLY_RULE without
appending OA-Commit, whose normal-stage guidance asks for a runtime_step tool call.
The original bridge is preserved in sources/original_frontier.py.txt.

The 11 Skills, 5 Programs, 5 Implementations and 4 Workflows are unchanged.
The native importer remaps environment-dependent Program identities. Actual
Runtime uses bank.json guidance; the portable SQLite is only a source snapshot.
Workflow records do not imply graph execution. These assets retain authored
reference permissions; ordinary Ours qualification and scoring are unchanged.

experiments/run_office_humanstrong_v1.py is an independent copy of the fixed V3.1
pair runner, with this package/version, the new authorized limits and validated
resolved configs serialized identically to the native runner. Unknown billing
stops the campaign before another episode, preserving the original reservation.

## Test102 integration follow-up, 2026-10-19

The subsequent user approval covers the review's single HumanStrong Test102,
seed42, cumulative 116M ceiling. No repeat Train12 or added Val is required.
`experiments/run_office_humanstrong_test.py` freezes the plan and uses the same
audited execution/billing loop. All asset contents above remain byte-identical.
This is a development-exposed Test rerun, not a new blind evaluation; runtime
changes also prevent attributing differences from old NoSkill solely to Bank.

Both finish paths now use the same newest acquired-source ordering, reserving
bounded readable bodies before older previews/reference directories. Actual
compact computation results retain priority. The existing 16,384-byte wire cap
and provenance restrictions remain. Saved UID0153 public state is the offline
acceptance fixture; its gold is not supplied to the rebuilt request.

The local selector additionally advertises calculation Programs in extract when
real material is acquired and the public task asks for calculation. It does not
claim bound operands, force a call, alter source validation or enable Composites.
The original selector and bridge remain in sources/, with changes recorded in
integration_receipt.json. No Program-return/outputs_consumed change is made.
