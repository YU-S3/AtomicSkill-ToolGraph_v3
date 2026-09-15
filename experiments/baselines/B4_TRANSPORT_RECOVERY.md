# B4 transient provider recovery

The September 15 formal continuation finished Train120 and all four Val24
rounds, then stopped during Test on `Provider solver failed: AttributeError;
attempts=1`. The old transport discarded the original exception details, so
those historical events cannot establish which response field or SDK call
raised the exception. Their usage remains unknown, not zero.

This patch validates the response envelope before consuming it and records
HTTP metadata, sanitized failed-body excerpts, exception phase and traceback.
Malformed envelopes, connection errors, HTTP 408/429/5xx use the existing five
physical attempts (SDK retries remain disabled). A local AttributeError is
not unconditionally made retryable. Prompts, parsers, reasoning, completion
cap, task selection and valid-response consumption are unchanged.

B4 may retry an operation at most three times, separated by 60/120 seconds,
only after an explicitly retryable infrastructure failure. Each attempt has
a new directory and starts at the same committed input state. No partial
state is published; no ordinary unsuccessful episode, protocol error, budget
exhaustion or authentication failure is automatically replayed. Independent
evaluation episodes finish even when another exhausts its recovery allowance.
Provider attempts and all-attempt cost summaries include failed attempts;
unknown historical usage remains null with known subtotals. The operation
recovery allowance is recorded separately from the unchanged provider policy.

## Continuing an existing campaign after this patch

Run the new real B4 smoke first. Then use:

```bash
bash experiments/baselines/launch_embodiskill.sh resume-repair \
  runs/baselines/EXISTING_CAMPAIGN \
  runs/baselines/NEW_PASSING_SMOKE/smoke_qualification.json
```

This mode verifies the original commit/code hash, unchanged config, model,
manifests, dependencies and upstream source, a clean committed checkout, an
explicit file allowlist for this repair, and a fresh passing smoke receipt.
It writes a separate `transport_recovery/<execution_code_hash>.json` receipt.
It never rewrites the original campaign lock, seed manifest, Frozen provenance
or successful checkpoint. New job records and final summaries disclose both
the original and recovery code identities. Arbitrary code/config changes are
still rejected. Normal same-code `resume` retains its strict identity check.

Old `failure.json` and `campaign_summary.json` may remain while continuation
is running. Check current checkpoints/recovery events; the campaign summary
is replaced only when the continuation finishes. A further outage can still
exhaust bounded recovery and require another continuation.
