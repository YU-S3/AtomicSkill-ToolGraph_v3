"""Audited continuation across a narrowly scoped transport repair.

Original campaign locks, seed manifests, Frozen provenance and checkpoints
remain immutable. A separate receipt records the code used for recovery.
"""
from __future__ import annotations

import copy
import hashlib
import subprocess
from pathlib import Path

from experiments.baselines.common.manifest import sha256_json
from experiments.baselines.common.source_identity import _CODE_SUFFIXES, _excluded
from experiments.baselines.b4_embodiskill.state import read_json, write_json


REPAIR_POLICY = "b4-transient-provider-recovery-v1"
REPAIR_FILES = frozenset({
    "experiments/baselines/common/model_client.py",
    "experiments/baselines/b4_embodiskill/model_adapter.py",
    "experiments/baselines/b4_embodiskill/worker.py",
    "experiments/baselines/b4_embodiskill/controller.py",
    "experiments/baselines/b4_embodiskill/campaign.py",
    "experiments/baselines/b4_embodiskill/transport_repair.py",
    "experiments/baselines/tests/test_b4_transport_repair.py",
    "experiments/baselines/launch_embodiskill.sh",
    "experiments/baselines/B4_TRANSPORT_RECOVERY.md",
})


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def historical_code_hash(repo, commit):
    prefix = "experiments/baselines/"
    names = git(repo, "ls-tree", "-r", "--name-only", commit, "--", prefix).decode().splitlines()
    rows = []
    for name in sorted(names):
        relative = Path(name.removeprefix(prefix))
        if relative.suffix.casefold() in _CODE_SUFFIXES and not _excluded(relative):
            content = git(repo, "show", f"{commit}:{name}")
            working = Path(repo) / name
            # Existing Windows/WSL checkouts can contain mixed newline styles.
            # Reuse original unchanged bytes only when their normalized content
            # is exactly the committed blob; the final full hash must still match.
            if working.is_file():
                local = working.read_bytes()
                if local.replace(b"\r\n", b"\n") == content:
                    content = local
            rows.append(dict(path=relative.as_posix(), sha256=hashlib.sha256(content).hexdigest()))
    return sha256_json(rows)


def prepare_transport_resume(repo, output, locked, current, smoke_path):
    from experiments.baselines.b4_embodiskill.campaign import verify_smoke_qualification

    if git(repo, "status", "--porcelain").strip():
        raise RuntimeError("Transport recovery requires a clean committed checkout")
    before, after = locked["identity"], current["identity"]
    if {k: v for k, v in before.items() if k != "code_hash"} != {
            k: v for k, v in after.items() if k != "code_hash"}:
        raise RuntimeError("Transport recovery cannot change config/model/data/upstream/runtime")
    if locked["config"] != current["config"] or locked.get("resolved_config_hash") != sha256_json(locked["config"]):
        raise RuntimeError("Transport recovery configuration differs from original lock")
    base = locked["git_state"]["commit"]
    if historical_code_hash(repo, base) != before["code_hash"]:
        raise RuntimeError("Original lock source hash does not match its recorded commit")
    changed = set(git(repo, "diff", "--name-only", base, "HEAD").decode().splitlines())
    if not changed or changed - REPAIR_FILES:
        raise RuntimeError(f"Changes outside audited transport repair: {sorted(changed - REPAIR_FILES)}")
    verify_smoke_qualification(smoke_path, after, current["config"])
    receipt = dict(policy=REPAIR_POLICY, original_lock_sha256=sha256_json(locked),
        original_commit=base, recovery_commit=current["git_state"]["commit"],
        original_identity=before, recovery_identity=after, changed_files=sorted(changed),
        smoke_receipt=str(Path(smoke_path).resolve()), smoke_sha256=sha256_json(read_json(smoke_path)),
        operation_attempt_limit=3, recovery_delays_seconds=[60, 120],
        original_results_preserved=True)
    path = Path(output) / "transport_recovery" / (after["code_hash"] + ".json")
    if path.exists():
        if read_json(path) != receipt:
            raise RuntimeError("Existing transport recovery receipt differs")
    else:
        write_json(path, receipt)
    spec = copy.deepcopy(locked)
    # Do not rewrite the original seed controller_commit or Frozen provenance.
    spec["execution_identity"] = after
    spec["transport_recovery"] = receipt
    return spec
