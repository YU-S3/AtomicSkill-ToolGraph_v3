"""Recover the three held-out B3 tests from an immutable Train campaign.

The legacy B3 campaign bound controller identity to raw checkout bytes.  That
digest cannot be reproduced after a checkout is rematerialized, even when the
Git commit and tracked source are identical.  This controller keeps every
formal authority check except that single digest comparison, records the
waiver, runs the locked legacy controller, and restores the caller checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, TextIO


REPO_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_SOURCE = Path(__file__).with_name(
    "b3_controller_code_waiver_bootstrap.py"
)
_METHOD = "b3_skillopt"
_SEEDS = (42, 43, 44)
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_FAILURE = "campaign lock does not match this formal run: controller_code"
_PRESERVED_CHECKS = (
    "controller_commit",
    "tracked_source_cleanliness",
    "campaign_lock_digest",
    "formal_config_digest",
    "model_identity",
    "python_runtime_authority",
    "skillopt_commit_and_runtime",
    "train_validation_test_manifests",
    "alfworld_data_and_gamefiles",
    "source_train_identity",
    "frozen_artifact_digest",
    "held_out_exactly_once",
    "episode_and_provider_evidence",
)


class RecoveryError(RuntimeError):
    """The controlled legacy Test recovery could not proceed."""


@dataclass(frozen=True)
class TrainAuthority:
    seed: int
    root: Path
    run_id: str
    frozen_digest: str


@dataclass(frozen=True)
class RecoveryAuthority:
    campaign_root: Path
    campaign_lock: Path
    campaign_lock_digest: str
    campaign_id: str
    locked_commit: str
    locked_tree: str
    locked_code_digest: str
    phase_python: Path
    config: Path
    manifests: Mapping[str, Path]
    trains: tuple[TrainAuthority, ...]
    failure_evidence: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class CheckoutState:
    branch: str
    commit: str
    tree: str


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoveryError(f"JSON authority is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise RecoveryError(f"JSON authority is not an object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest_digest(payload: Mapping[str, Any]) -> str:
    declared = str(payload.get("digest") or "")
    unsigned = dict(payload)
    unsigned.pop("digest", None)
    computed = _sha256_json(unsigned)
    if not _DIGEST.fullmatch(declared) or declared != computed:
        raise RecoveryError("manifest declared digest is absent or inconsistent")
    return computed


def _write_json_new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "unknown git error"
        raise RecoveryError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def _checkout_state(
    repo: Path,
    *,
    require_clean: bool,
    source_scope_only: bool = False,
) -> CheckoutState:
    commit = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    branch = _git(repo, "branch", "--show-current")
    if not _COMMIT.fullmatch(commit) or not _COMMIT.fullmatch(tree):
        raise RecoveryError("controller Git identity is not a full object id")
    if require_clean:
        status_args = ["status", "--porcelain", "--untracked-files=all"]
        if source_scope_only:
            status_args.extend(["--", "src", "experiments", "configs"])
        status = _git(repo, *status_args)
        if status:
            raise RecoveryError(
                "recovery requires a completely clean checkout before Git switching"
            )
    return CheckoutState(branch=branch, commit=commit, tree=tree)


def _locked_path(value: Any, *, field: str, kind: str = "file") -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RecoveryError(f"campaign lock field {field} is missing")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise RecoveryError(f"campaign lock field {field} is not absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RecoveryError(f"campaign lock path {field} is unavailable: {path}") from exc
    if kind == "file" and not resolved.is_file():
        raise RecoveryError(f"campaign lock path {field} is not a file: {resolved}")
    if kind == "dir" and not resolved.is_dir():
        raise RecoveryError(f"campaign lock path {field} is not a directory: {resolved}")
    return resolved


def _locked_python(lock: Mapping[str, Any]) -> Path:
    raw = lock.get("phase_python")
    if not isinstance(raw, str) or not raw.strip():
        raise RecoveryError("campaign lock field phase_python is missing")
    phase_python = Path(raw).expanduser()
    if not phase_python.is_absolute() or not phase_python.is_file():
        raise RecoveryError(f"campaign phase_python is unavailable: {phase_python}")
    for field in ("worker_python", "provider_probe_python"):
        if lock.get(field) != raw:
            raise RecoveryError(f"campaign {field} differs from phase_python")
    return phase_python


def _digest_directory(root: Path) -> str:
    if not root.is_dir():
        raise RecoveryError(f"frozen artifact directory is missing: {root}")
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        entries.append((path.relative_to(root).as_posix(), _sha256_file(path)))
    digest = hashlib.sha256()
    for relative, content_hash in entries:
        digest.update(f"{relative}\x1f{content_hash}\n".encode("utf-8"))
    return digest.hexdigest()


def _load_train_authority(
    campaign_root: Path,
    *,
    seed: int,
    lock_digest: str,
    locked_commit: str,
    locked_code_digest: str,
) -> TrainAuthority:
    root = (campaign_root / f"seed_{seed}" / "train").resolve(strict=True)
    completion = _read_json_object(root / "completion.json")
    report = _read_json_object(root / "report.json")
    manifest = _read_json_object(root / "run_manifest.json")
    if completion.get("passed") is not True or completion.get("phase") != "train":
        raise RecoveryError(f"seed {seed} Train completion is not successful")
    if report.get("passed") is not True or report.get("method") != _METHOD:
        raise RecoveryError(f"seed {seed} Train report is not successful B3")
    if (
        manifest.get("formal") is not True
        or manifest.get("phase") != "train"
        or manifest.get("method") != _METHOD
        or int(manifest.get("run_seed", -1)) != seed
    ):
        raise RecoveryError(f"seed {seed} Train manifest identity is invalid")
    controller_git = dict(manifest.get("controller_git") or {})
    identity = dict(manifest.get("identity") or {})
    campaign = dict(manifest.get("campaign") or {})
    if controller_git.get("commit") != locked_commit or controller_git.get("dirty"):
        raise RecoveryError(f"seed {seed} Train controller commit is inconsistent")
    if (
        manifest.get("controller_code_digest") != locked_code_digest
        or identity.get("controller_code_digest") != locked_code_digest
    ):
        raise RecoveryError(f"seed {seed} Train controller digest is inconsistent")
    if campaign.get("campaign_lock_digest") != lock_digest:
        raise RecoveryError(f"seed {seed} Train campaign lock digest is inconsistent")

    frozen_manifest = _read_json_object(root / "frozen" / "digest.json")
    frozen_digest = str(frozen_manifest.get("digest") or "")
    if not _DIGEST.fullmatch(frozen_digest):
        raise RecoveryError(f"seed {seed} Frozen digest is invalid")
    artifact_root = root / "frozen" / "artifact"
    if _digest_directory(artifact_root) != frozen_digest:
        raise RecoveryError(f"seed {seed} Frozen artifact has changed")
    report_frozen = dict(report.get("frozen") or {})
    if report_frozen.get("digest") != frozen_digest:
        raise RecoveryError(f"seed {seed} Train report names another Frozen artifact")
    run_id = str(manifest.get("run_id") or "")
    if not run_id or completion.get("run_id") != run_id:
        raise RecoveryError(f"seed {seed} Train run_id is inconsistent")
    return TrainAuthority(
        seed=seed,
        root=root,
        run_id=run_id,
        frozen_digest=frozen_digest,
    )


def _load_failure_evidence(
    failed_attempt_root: Path,
    *,
    campaign_root: Path,
    seeds: Sequence[int],
) -> tuple[Mapping[str, Any], ...]:
    root = failed_attempt_root.resolve(strict=True)
    if campaign_root not in root.parents:
        raise RecoveryError("failed Test evidence is outside the locked campaign")
    forbidden_names = {
        "run_manifest.json",
        "provider_calls.jsonl",
        "evaluated_common_episodes.jsonl",
        "task_rows.jsonl",
    }
    observed_forbidden = [
        path for path in root.rglob("*")
        if path.is_file() and path.name in forbidden_names
    ]
    if observed_forbidden:
        raise RecoveryError(
            "failed attempt contains held-out Test execution evidence: "
            + ", ".join(str(path) for path in observed_forbidden)
        )
    evidence = []
    for seed in seeds:
        path = root / f"seed_{seed}.log"
        payload = _read_json_object(path)
        if (
            payload.get("passed") is not False
            or payload.get("method") != _METHOD
            or payload.get("phase") != "test"
            or payload.get("failure_kind") != "protocol_failure"
            or payload.get("output_dir") is not None
            or payload.get("error") != _EXPECTED_FAILURE
        ):
            raise RecoveryError(
                f"seed {seed} does not prove a controller-code-only preflight failure"
            )
        evidence.append({
            "seed": seed,
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
            "run_id": str(payload.get("run_id") or ""),
            "error": _EXPECTED_FAILURE,
        })
    run_ids = [str(item["run_id"]) for item in evidence]
    if any(not value for value in run_ids) or len(set(run_ids)) != len(run_ids):
        raise RecoveryError("failed Test preflight run_ids are absent or duplicated")
    return tuple(evidence)


def load_authority(
    campaign_root: str | Path,
    failed_attempt_root: str | Path,
    *,
    repo: Path = REPO_ROOT,
    executable: str | Path = sys.executable,
) -> RecoveryAuthority:
    root = Path(campaign_root).expanduser().resolve(strict=True)
    lock_path = (root / "campaign_lock.json").resolve(strict=True)
    lock = _read_json_object(lock_path)
    lock_digest = _sha256_file(lock_path)
    if lock.get("method") != _METHOD:
        raise RecoveryError("campaign is not B3 SkillOpt")
    if Path(str(lock.get("campaign_root", ""))).resolve() != root:
        raise RecoveryError("campaign lock root does not match the requested campaign")
    seeds = tuple(int(value) for value in list(lock.get("seeds") or []))
    if seeds != _SEEDS:
        raise RecoveryError(f"campaign seeds must be exactly {list(_SEEDS)}")

    locked_commit = str(lock.get("controller_commit") or "")
    locked_code_digest = str(lock.get("controller_code_digest") or "")
    controller_git = dict(lock.get("controller_git") or {})
    if not _COMMIT.fullmatch(locked_commit):
        raise RecoveryError("campaign controller commit is not a full SHA-1")
    if not _DIGEST.fullmatch(locked_code_digest):
        raise RecoveryError("campaign controller code digest is invalid")
    if (
        controller_git.get("commit") != locked_commit
        or controller_git.get("code_digest") != locked_code_digest
        or controller_git.get("dirty") is not False
    ):
        raise RecoveryError("campaign controller Git authority is inconsistent")
    _git(repo, "cat-file", "-e", f"{locked_commit}^{{commit}}")
    locked_tree = _git(repo, "rev-parse", f"{locked_commit}^{{tree}}")
    if not _COMMIT.fullmatch(locked_tree):
        raise RecoveryError("campaign controller tree is invalid")

    phase_python = _locked_python(lock)
    try:
        if not Path(executable).resolve(strict=True).samefile(phase_python):
            raise RecoveryError(
                f"recovery must run with locked B3 Python: {phase_python}"
            )
    except OSError as exc:
        raise RecoveryError("B3 Python authority could not be compared") from exc

    config = _locked_path(lock.get("config_path"), field="config_path")
    manifest_payload = dict(lock.get("manifests") or {})
    manifests: dict[str, Path] = {}
    for role in ("train", "validation", "test"):
        entry = dict(manifest_payload.get(role) or {})
        path = _locked_path(entry.get("path"), field=f"manifests.{role}.path")
        if _manifest_digest(_read_json_object(path)) != entry.get("digest"):
            raise RecoveryError(f"campaign {role} manifest digest has changed")
        manifests[role] = path

    trains = tuple(
        _load_train_authority(
            root,
            seed=seed,
            lock_digest=lock_digest,
            locked_commit=locked_commit,
            locked_code_digest=locked_code_digest,
        )
        for seed in seeds
    )
    failures = _load_failure_evidence(
        Path(failed_attempt_root),
        campaign_root=root,
        seeds=seeds,
    )
    return RecoveryAuthority(
        campaign_root=root,
        campaign_lock=lock_path,
        campaign_lock_digest=lock_digest,
        campaign_id=str(lock.get("campaign_id") or ""),
        locked_commit=locked_commit,
        locked_tree=locked_tree,
        locked_code_digest=locked_code_digest,
        phase_python=phase_python,
        config=config,
        manifests=manifests,
        trains=trains,
        failure_evidence=failures,
    )


def _matching_test_manifests(
    repo: Path,
    *,
    campaign_lock_digest: str,
    seeds: Sequence[int],
) -> list[Path]:
    root = repo / "runs" / "baselines"
    if not root.is_dir():
        return []
    matched = []
    expected_seeds = set(seeds)
    for directory, dirnames, filenames in os.walk(root):
        dirnames.sort()
        if "run_manifest.json" not in filenames:
            continue
        manifest_path = Path(directory) / "run_manifest.json"
        manifest = _read_json_object(manifest_path)
        dirnames[:] = []
        campaign = dict(manifest.get("campaign") or {})
        if (
            manifest.get("method") == _METHOD
            and manifest.get("phase") == "test"
            and int(manifest.get("run_seed", -1)) in expected_seeds
            and campaign.get("campaign_lock_digest") == campaign_lock_digest
        ):
            matched.append(manifest_path.resolve())
    return sorted(matched)


@contextmanager
def _campaign_lease(campaign_root: Path) -> Iterator[None]:
    lock_path = campaign_root / ".controller_code_waiver.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            if not handle.read(1):
                handle.write(" ")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RecoveryError("another B3 recovery controller is active") from exc
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RecoveryError("another B3 recovery controller is active") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "started_at_unix": time.time()}))
        handle.flush()
        os.fsync(handle.fileno())
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _default_output(campaign_root: Path, *, preflight_only: bool) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = "recovery_preflight" if preflight_only else "recovered_test_waiver"
    return campaign_root / f"{label}_{stamp}"


def _copy_bootstrap(output_root: Path) -> tuple[Path, str]:
    source = BOOTSTRAP_SOURCE.resolve(strict=True)
    destination = output_root / "controller_code_waiver_bootstrap.py"
    with source.open("rb") as source_handle, destination.open("xb") as target:
        shutil.copyfileobj(source_handle, target)
        target.flush()
        os.fsync(target.fileno())
    return destination, _sha256_file(destination)


def _runner_arguments(
    authority: RecoveryAuthority,
    *,
    train: TrainAuthority,
    output_root: Path,
) -> list[str]:
    return [
        "--method", _METHOD,
        "--phase", "test",
        "--seed", str(train.seed),
        "--train-manifest", str(authority.manifests["train"]),
        "--validation-manifest", str(authority.manifests["validation"]),
        "--test-manifest", str(authority.manifests["test"]),
        "--source-run", str(train.root),
        "--campaign-lock", str(authority.campaign_lock),
        "--config", str(authority.config),
        "--output-dir", str((output_root / f"seed_{train.seed}").resolve()),
    ]


def _bootstrap_prefix(
    authority: RecoveryAuthority,
    *,
    bootstrap: Path,
    bootstrap_digest: str,
) -> list[str]:
    return [
        str(authority.phase_python),
        str(bootstrap),
        "--repo-root", str(REPO_ROOT),
        "--campaign-root", str(authority.campaign_root),
        "--campaign-lock", str(authority.campaign_lock),
        "--campaign-lock-digest", authority.campaign_lock_digest,
        "--locked-commit", authority.locked_commit,
        "--locked-tree", authority.locked_tree,
        "--locked-code-digest", authority.locked_code_digest,
        "--bootstrap-sha256", bootstrap_digest,
    ]


def _switch_to_locked(
    authority: RecoveryAuthority,
    *,
    repo: Path = REPO_ROOT,
) -> None:
    _git(repo, "switch", "--detach", authority.locked_commit)
    state = _checkout_state(
        repo,
        require_clean=True,
        source_scope_only=True,
    )
    if state.commit != authority.locked_commit or state.tree != authority.locked_tree:
        raise RecoveryError("Git switch did not produce the locked controller tree")


def _restore_checkout(
    original: CheckoutState,
    *,
    repo: Path = REPO_ROOT,
) -> CheckoutState:
    if original.branch:
        _git(repo, "switch", original.branch)
    else:
        _git(repo, "switch", "--detach", original.commit)
    restored = _checkout_state(repo, require_clean=True)
    if restored.commit != original.commit or restored.tree != original.tree:
        raise RecoveryError("original controller checkout was not restored exactly")
    if original.branch and restored.branch != original.branch:
        raise RecoveryError("original controller branch was not restored")
    return restored


def _terminate_processes(processes: Sequence[subprocess.Popen[Any]]) -> None:
    active = [process for process in processes if process.poll() is None]
    for process in active:
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 20.0
    for process in active:
        timeout = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
    for process in active:
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            pass


@contextmanager
def _restore_on_termination_signal() -> Iterator[None]:
    previous: dict[signal.Signals, Any] = {}

    def interrupt(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    candidates = [signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        candidates.append(signal.SIGHUP)
    for candidate in candidates:
        old = signal.getsignal(candidate)
        if old == signal.SIG_IGN:
            continue
        previous[candidate] = old
        signal.signal(candidate, interrupt)
    try:
        yield
    finally:
        for candidate, old in previous.items():
            signal.signal(candidate, old)


def _seed_result(output_root: Path, *, seed: int, returncode: int) -> dict[str, Any]:
    root = output_root / f"seed_{seed}"
    completion_path = root / "completion.json"
    report_path = root / "test_report.json"
    application_path = (
        output_root / f"seed_{seed}.controller_code_waiver_application.json"
    )
    completion = _read_json_object(completion_path) if completion_path.is_file() else {}
    report = _read_json_object(report_path) if report_path.is_file() else {}
    application = (
        _read_json_object(application_path) if application_path.is_file() else {}
    )
    passed = (
        returncode == 0
        and completion.get("passed") is True
        and completion.get("phase") == "test"
        and report.get("passed") is True
        and report.get("method") == _METHOD
        and application.get("passed") is True
        and application.get("waived_checks") == ["controller_code_digest"]
        and application.get("waived_hash_calls") == 2
    )
    return {
        "seed": seed,
        "passed": passed,
        "phase": "test",
        "state": "COMPLETED" if passed else "FAILED",
        "returncode": returncode,
        "test_root": str(root.resolve()),
        "run_id": str(completion.get("run_id") or report.get("run_id") or ""),
        "completion_sha256": (
            _sha256_file(completion_path) if completion_path.is_file() else None
        ),
        "test_report_sha256": (
            _sha256_file(report_path) if report_path.is_file() else None
        ),
        "waiver_application": (
            str(application_path.resolve()) if application_path.is_file() else None
        ),
        "waiver_application_sha256": (
            _sha256_file(application_path) if application_path.is_file() else None
        ),
    }


def _waiver_receipt(
    authority: RecoveryAuthority,
    *,
    original: CheckoutState,
    output_root: Path,
    bootstrap: Path,
    bootstrap_digest: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "waiver_id": f"b3_controller_code_{uuid.uuid4().hex}",
        "status": "authorized_for_legacy_test_recovery",
        "method": _METHOD,
        "phase": "test",
        "waived_checks": ["controller_code_digest"],
        "reason_code": "legacy_raw_worktree_digest_not_reproducible",
        "formal_identity_status": "accepted_with_controller_code_waiver",
        "claim_scope": "held_out_test_with_disclosed_identity_waiver",
        "not_authorized_for": ["train", "other_methods", "other_identity_mismatches"],
        "preserved_checks": list(_PRESERVED_CHECKS),
        "campaign": {
            "campaign_id": authority.campaign_id,
            "root": str(authority.campaign_root),
            "lock": str(authority.campaign_lock),
            "lock_digest": authority.campaign_lock_digest,
        },
        "locked_controller": {
            "commit": authority.locked_commit,
            "tree": authority.locked_tree,
            "code_digest": authority.locked_code_digest,
        },
        "launcher": {
            "branch": original.branch,
            "commit": original.commit,
            "tree": original.tree,
            "source": str(Path(__file__).resolve()),
            "source_sha256": _sha256_file(Path(__file__).resolve()),
            "bootstrap": str(bootstrap.resolve()),
            "bootstrap_sha256": bootstrap_digest,
        },
        "source_trains": [
            {
                "seed": train.seed,
                "root": str(train.root),
                "run_id": train.run_id,
                "frozen_digest": train.frozen_digest,
            }
            for train in authority.trains
        ],
        "prior_preflight_failures": list(authority.failure_evidence),
        "output_root": str(output_root.resolve()),
        "created_at_unix": time.time(),
    }


def _write_campaign_report(
    output_root: Path,
    *,
    authority: RecoveryAuthority,
    waiver_path: Path,
    results: Sequence[Mapping[str, Any]],
    original: CheckoutState,
    restored: CheckoutState | None,
    started_at: float,
) -> dict[str, Any]:
    passed = bool(results) and all(item.get("passed") is True for item in results)
    passed = passed and restored == original
    payload = {
        "schema_version": 1,
        "passed": passed,
        "method": _METHOD,
        "phase": "test",
        "campaign_id": authority.campaign_id,
        "source_campaign_root": str(authority.campaign_root),
        "campaign_lock": str(authority.campaign_lock),
        "campaign_lock_digest": authority.campaign_lock_digest,
        "formal_identity_status": "accepted_with_controller_code_waiver",
        "waiver_receipt": str(waiver_path.resolve()),
        "waiver_receipt_sha256": _sha256_file(waiver_path),
        "completed_seeds": [
            int(item["seed"]) for item in results if item.get("passed") is True
        ],
        "failed_seeds": [
            int(item["seed"]) for item in results if item.get("passed") is not True
        ],
        "lanes": list(results),
        "checkout": {
            "original": original.__dict__,
            "restored": restored.__dict__ if restored is not None else None,
            "restored_exactly": restored == original,
        },
        "started_at_unix": started_at,
        "completed_at_unix": time.time(),
    }
    _write_json_new(output_root / "campaign_report.json", payload)
    return payload


def _run_aggregate_report(output_root: Path) -> dict[str, Any]:
    output = output_root / "recovered_test_report.json"
    command = [
        sys.executable,
        "-m",
        "experiments.baselines.report_campaign",
        "--method-run",
        f"{_METHOD}={output_root}",
        "--expected-seeds",
        *(str(seed) for seed in _SEEDS),
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {
        "passed": completed.returncode == 0 and output.is_file(),
        "returncode": completed.returncode,
        "output": str(output.resolve()),
        "log": completed.stdout[-4000:],
        "sha256": _sha256_file(output) if output.is_file() else None,
    }


def run_recovery(
    *,
    campaign_root: str | Path,
    failed_attempt_root: str | Path,
    output_dir: str | Path | None,
    acknowledge: bool,
    preflight_only: bool,
) -> dict[str, Any]:
    if not acknowledge:
        raise RecoveryError(
            "--acknowledge-controller-code-waiver is required for this recovery"
        )
    original = _checkout_state(REPO_ROOT, require_clean=True)
    authority = load_authority(campaign_root, failed_attempt_root)
    output_root = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else _default_output(authority.campaign_root, preflight_only=preflight_only)
    )
    if output_root.parent != authority.campaign_root:
        raise RecoveryError(
            "recovery output must be a direct child of the locked campaign root"
        )
    output_root.mkdir(parents=True, exist_ok=False)
    started_at = time.time()
    bootstrap, bootstrap_digest = _copy_bootstrap(output_root)
    waiver_path = output_root / "controller_code_identity_waiver.json"
    _write_json_new(
        waiver_path,
        _waiver_receipt(
            authority,
            original=original,
            output_root=output_root,
            bootstrap=bootstrap,
            bootstrap_digest=bootstrap_digest,
        ),
    )

    restored: CheckoutState | None = None
    processes: list[subprocess.Popen[Any]] = []
    logs: list[TextIO] = []
    results: list[Mapping[str, Any]] = []
    switched = False
    failure: BaseException | None = None
    with _restore_on_termination_signal(), _campaign_lease(authority.campaign_root):
        try:
            prior_tests = _matching_test_manifests(
                REPO_ROOT,
                campaign_lock_digest=authority.campaign_lock_digest,
                seeds=_SEEDS,
            )
            if prior_tests:
                joined = ", ".join(str(path) for path in prior_tests)
                raise RecoveryError(
                    "held-out Test already has run manifests for this campaign: "
                    + joined
                )
            switched = True
            _switch_to_locked(authority)
            environment = dict(os.environ)
            environment["PYTHONPATH"] = os.pathsep.join(
                [str(REPO_ROOT), environment.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep)
            prefix = _bootstrap_prefix(
                authority,
                bootstrap=bootstrap,
                bootstrap_digest=bootstrap_digest,
            )
            preflight_log = output_root / "controller_waiver_preflight.log"
            preflight = subprocess.run(
                [*prefix, "--preflight-only"],
                cwd=REPO_ROOT,
                env=environment,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            preflight_log.write_text(preflight.stdout, encoding="utf-8")
            if preflight.returncode != 0:
                raise RecoveryError(
                    "locked-controller waiver preflight failed; see " + str(preflight_log)
                )
            if preflight_only:
                results = []
            else:
                for train in authority.trains:
                    runner_args = _runner_arguments(
                        authority,
                        train=train,
                        output_root=output_root,
                    )
                    args_path = output_root / f"seed_{train.seed}.runner_args.json"
                    _write_json_new(args_path, runner_args)
                    log_handle = (output_root / f"seed_{train.seed}.log").open(
                        "x", encoding="utf-8"
                    )
                    logs.append(log_handle)
                    command = [
                        *prefix,
                        "--runner-args-file", str(args_path),
                        "--runner-args-sha256", _sha256_file(args_path),
                        "--application-receipt",
                        str(
                            output_root
                            / (
                                f"seed_{train.seed}."
                                "controller_code_waiver_application.json"
                            )
                        ),
                        "--seed", str(train.seed),
                    ]
                    processes.append(subprocess.Popen(
                        command,
                        cwd=REPO_ROOT,
                        env=environment,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                        start_new_session=(os.name != "nt"),
                    ))
                returncodes = [process.wait() for process in processes]
                results = [
                    _seed_result(output_root, seed=train.seed, returncode=returncode)
                    for train, returncode in zip(
                        authority.trains, returncodes, strict=True
                    )
                ]
        except BaseException as exc:
            failure = exc
            _terminate_processes(processes)
        finally:
            for handle in logs:
                handle.close()
            if switched:
                try:
                    restored = _restore_checkout(original)
                except BaseException as exc:
                    failure = failure or exc

    if preflight_only and failure is None:
        payload = {
            "schema_version": 1,
            "passed": True,
            "preflight_only": True,
            "preflight_scope": "locked_checkout_and_waiver_bootstrap_only",
            "test_readiness": "not_evaluated_until_legacy_runner_starts",
            "verified_checks": [
                "campaign_and_train_authority",
                "frozen_artifact_digest",
                "prior_failure_scope",
                "held_out_exactly_once",
                "locked_controller_commit_and_tree",
                "locked_source_scope_cleanliness",
                "bootstrap_digest_and_hash_binding",
            ],
            "deferred_to_test_runner": [
                "model_and_api_key",
                "python_runtime_imports",
                "skillopt_runtime",
                "alfworld_data_and_gamefiles",
                "formal_manifest_preflight",
                "episode_and_provider_evidence",
            ],
            "output_root": str(output_root),
            "waiver_receipt": str(waiver_path),
            "locked_commit": authority.locked_commit,
            "locked_tree": authority.locked_tree,
            "locked_code_digest": authority.locked_code_digest,
            "checkout_restored": restored == original,
        }
        _write_json_new(output_root / "recovery_preflight.json", payload)
        return payload

    campaign_report = _write_campaign_report(
        output_root,
        authority=authority,
        waiver_path=waiver_path,
        results=results,
        original=original,
        restored=restored,
        started_at=started_at,
    )
    aggregate = (
        _run_aggregate_report(output_root)
        if campaign_report["passed"] is True
        else {"passed": False, "status": "not_run_for_failed_test_campaign"}
    )
    final = {
        "schema_version": 1,
        "passed": campaign_report["passed"] is True and aggregate["passed"] is True,
        "formal_identity_status": "accepted_with_controller_code_waiver",
        "output_root": str(output_root),
        "campaign_report": str((output_root / "campaign_report.json").resolve()),
        "aggregate_report": aggregate,
        "waiver_receipt": str(waiver_path.resolve()),
        "failure": (
            {
                "error_type": type(failure).__name__,
                "error": str(failure)[:4000],
            }
            if failure is not None
            else None
        ),
    }
    _write_json_new(output_root / "recovery_completion.json", final)
    if failure is not None:
        raise RecoveryError(
            f"B3 Test recovery failed; see {output_root / 'recovery_completion.json'}"
        ) from failure
    if final["passed"] is not True:
        raise RecoveryError(
            f"B3 Test recovery was incomplete; see {output_root / 'recovery_completion.json'}"
        )
    return final


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--failed-attempt-root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--acknowledge-controller-code-waiver",
        action="store_true",
        help="accept the disclosed legacy controller-code identity waiver",
    )
    args = parser.parse_args(argv)
    try:
        result = run_recovery(
            campaign_root=args.campaign_root,
            failed_attempt_root=args.failed_attempt_root,
            output_dir=args.output_dir,
            acknowledge=args.acknowledge_controller_code_waiver,
            preflight_only=args.preflight_only,
        )
    except (RecoveryError, FileExistsError, FileNotFoundError) as exc:
        print(json.dumps({
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }, ensure_ascii=False, sort_keys=True))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
