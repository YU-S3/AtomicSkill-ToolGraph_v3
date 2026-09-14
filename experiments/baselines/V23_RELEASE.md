# v2.3 final formal release contract

The v2.3 implementation changes accounting/preflight/reporting only. EmbodiSkill
and GEPA upstream trees, frozen dataset manifests, method budgets, B3 completed
artifacts and Ours are unchanged. Formal training is not part of this validation.

Offline validation: 311 tests passed with pinned SkillOpt on the B5 environment's
import path; four heavy B4 cases are covered separately in the B4 environment
(23 tests passed). The real upstream readonly test compares pre-v2.3 and v2.3
transport with identical scripted responses: retrieval/solver prompts, actions,
manual state and master frozen digest agree. Reranking alone moves to target,
retains its original parser/retry semantics, and has independent physical costs.

Release still requires fresh real artifacts from the committed source:

1. B4 `campaign_summary.json` and matching `smoke_qualification.json`, including
   reranking target role, required learning roles and readonly frozen boundaries.
2. B5 `smoke_report.json` and matching `smoke_qualification.json`, including actual
   optimizer/reflection/candidate evaluation and frozen Test under the 65536 cap.
3. B5 real load preflight at 48, then 36 or 24 only after a memory rejection.
   Every passing worker imports pinned dependencies and resets the exact game;
   memory is sampled throughout provider load and all workers must exit normally.
   The preflight writes `load_probe_summary.json` and an immutable
   `campaign_lock.json`; no seed starts in `--preflight-only` mode.

The final user handoff points to those artifacts and the unused prepared B5
directory. `launch_gepa.sh formal PREPARED_OUTPUT` revalidates source/config,
smoke/load receipts, worker counts and evidence hashes before any Train process.
A changed/missing/failed receipt is rejected. Formal cannot silently downgrade
or upgrade the locked cap. B4 and B5 cannot run simultaneously.

B3 historical 16384 transport is disclosed in the read-only common paper report's
`methods.b3_skillopt.comparability_note`. Old success rows, task pairing and token
evidence are untouched; cap-associated missing-action fallback is not asserted to
prove truncation or systemic exhaustion. Dollar pricing is optional offline
accounting; unknown consumption remains null with a known subtotal.

See [README](README.md) for smoke, preflight and formal launcher syntax. The older
v2.2 real-API receipts are retained as history and cannot qualify this checkout.
