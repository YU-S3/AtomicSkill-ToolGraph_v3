"""Baseline comparison experiment infrastructure.

External methods run through per-method venvs and subprocess workers; this
package only contains the controller, common protocol, and method adapters.
External checkouts live under ``.external/`` (gitignored) and are verified
against ``baseline_lock.yaml`` before any run.
"""
