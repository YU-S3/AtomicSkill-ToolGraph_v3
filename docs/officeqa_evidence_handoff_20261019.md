# Public evidence handoff

The opt-in Runtime contract `public-evidence-spans.v1` addresses evidence lost between
tool execution and final submission. Existing configurations keep their original behavior.

* Native and manual-Bank finalization use the same exact public-span packer. Computation
  results, explicitly bound outputs and operand quotes precede optional source pages.
  Whole acquired pages are preferred when they fit; table-row fallback retains the exact
  row, header and nearby title/unit text. Lexical ranking is not semantic certification.
* Source paths, character offsets and hashes remain auditable. Explicit `read_result`
  ranges retain their offsets. Required spans that cannot fit produce an explicit refusal;
  optional omissions are listed. No gold, UID rules or unseen source reads are involved.
* Temporary Python captures business stdout separately from the worker's 4 KiB diagnostic
  stream. It retains at most 64 KiB, records total/retained bytes and truncation, and only
  parses a complete JSON object as `data.computed`. Plain stdout remains a public result.
  The existing sandbox, path restrictions and shared internal-call budget still apply.
* Before a normal request exceeds the remaining budget, the existing Runtime turn can
  receive a compact acquired-evidence view and a consolidation instruction. Tools remain
  available. This adds neither a model stage nor retries and cannot guarantee another
  tool turn when the budget is exhausted.
* The Office observation uses a 65,536-byte final wire cap and an 8,192-token thinking-enabled
  final response. The normal 32,768 output cap, 192,000 task cap and 48 native calls stay fixed.
  Final reservation uses the prepared wire bound plus a 2,048 margin and output cap.
  Increasing these limits does not establish relevance, correct arithmetic or higher accuracy.

`experiments/run_office_evidence_handoff.py` prepares/checks/runs a separate bounded
observation: the first two canonical Train tasks, off/manual arms, seed42, unchanged
HumanStrong Bank, at most 768,000 new tokens and 400 HTTP within the existing 116M campaign
ceiling. Historical charges and unknown reservations are copied intact; the parent ledger
and frozen assets are checked before use. It does not launch Val/Test or learn new assets.
These Train tasks were exposed during Bank development, so their scores are engineering
observations rather than a held-out performance estimate.

Validation uses archived states through the production Provider boundary, real Docker
stdout/RPC tests, exact-span/Unicode/budget tests and existing Runtime regression tests.
Archived Test failures are used only for offline engineering replay, not new model calls.

Locally verified complete human-Bank Test results before this change: HumanFrontier
19/102 (18.63%), HumanStrong 27/102 (26.47%). These runs differ in implementation and are
not a controlled Bank-only comparison. The user's external full-Test NoSkill result above
33% is not available locally and its configuration has not been verified. No 70% accuracy
claim follows from these engineering fixes.
