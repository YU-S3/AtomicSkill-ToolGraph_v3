"""Run one legacy B3 Test with a narrowly scoped controller-code waiver.

This file is copied below ``runs/`` before the controller checkout changes.
It must remain standard-library-only until the locked legacy controller has
been imported from the verified repository checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Sequence


class WaiverBootstrapError(RuntimeError):
    """The recovery bootstrap authority could not be proven."""


class ControllerCodeWaiver:
    """Replace only the two legacy ``hash_code(REPO_ROOT)`` evaluations."""

    def __init__(
        self,
        *,
        repo: Path,
        commit: str,
        tree: str,
        digest: str,
        original: Any,
    ) -> None:
        self.repo = repo
        self.commit = commit
        self.tree = tree
        self.digest = digest
        self.original = original
        self.calls = 0

    def __call__(self, root: str | Path) -> str:
        if Path(root).resolve() != self.repo:
            return str(self.original(root))
        if self.calls >= 2:
            raise WaiverBootstrapError("legacy controller requested excess hash waivers")
        _verify_checkout(self.repo, commit=self.commit, tree=self.tree)
        self.calls += 1
        return self.digest


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WaiverBootstrapError(f"JSON authority is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise WaiverBootstrapError(f"JSON authority is not an object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        raise WaiverBootstrapError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def _verify_checkout(repo: Path, *, commit: str, tree: str) -> None:
    if _git(repo, "rev-parse", "HEAD") != commit:
        raise WaiverBootstrapError("controller checkout is not the locked commit")
    if _git(repo, "rev-parse", "HEAD^{tree}") != tree:
        raise WaiverBootstrapError("controller Git tree is not the locked tree")
    status = _git(
        repo,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        "src",
        "experiments",
        "configs",
    )
    if status:
        raise WaiverBootstrapError(
            "controller source scope is dirty at the locked commit"
        )


def _one_value(arguments: Sequence[str], flag: str) -> str:
    positions = [index for index, value in enumerate(arguments) if value == flag]
    if len(positions) != 1:
        raise WaiverBootstrapError(f"runner arguments require exactly one {flag}")
    index = positions[0]
    if index + 1 >= len(arguments):
        raise WaiverBootstrapError(f"runner argument {flag} has no value")
    return str(arguments[index + 1])


def _verify_runner_arguments(
    arguments: Sequence[str],
    *,
    repo: Path,
    campaign_lock: Path,
    campaign_root: Path,
    seed: int,
) -> None:
    if _one_value(arguments, "--method") != "b3_skillopt":
        raise WaiverBootstrapError("waiver is restricted to method b3_skillopt")
    if _one_value(arguments, "--phase") != "test":
        raise WaiverBootstrapError("waiver is restricted to the Test phase")
    if int(_one_value(arguments, "--seed")) != seed:
        raise WaiverBootstrapError("runner seed does not match waiver authority")
    if Path(_one_value(arguments, "--campaign-lock")).resolve() != campaign_lock:
        raise WaiverBootstrapError("runner campaign lock does not match authority")
    expected_source = (campaign_root / f"seed_{seed}" / "train").resolve()
    if Path(_one_value(arguments, "--source-run")).resolve() != expected_source:
        raise WaiverBootstrapError("runner source run is not the locked seed Train")
    output = Path(_one_value(arguments, "--output-dir")).resolve()
    if repo not in output.parents:
        raise WaiverBootstrapError("runner output must remain below the repository")
    relative_output = output.relative_to(repo)
    if not relative_output.parts or relative_output.parts[0] != "runs":
        raise WaiverBootstrapError("runner output must remain below runs/")
    if output.parent.parent != campaign_root:
        raise WaiverBootstrapError(
            "runner output is not inside one campaign recovery directory"
        )
    forbidden = {"--resume-source-run"}
    if any(value in forbidden for value in arguments):
        raise WaiverBootstrapError("Train recovery flags are forbidden for Test waiver")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--campaign-lock", required=True)
    parser.add_argument("--campaign-lock-digest", required=True)
    parser.add_argument("--locked-commit", required=True)
    parser.add_argument("--locked-tree", required=True)
    parser.add_argument("--locked-code-digest", required=True)
    parser.add_argument("--bootstrap-sha256", required=True)
    parser.add_argument("--runner-args-file")
    parser.add_argument("--runner-args-sha256")
    parser.add_argument("--application-receipt")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)

    repo = Path(args.repo_root).resolve(strict=True)
    campaign_root = Path(args.campaign_root).resolve(strict=True)
    campaign_lock = Path(args.campaign_lock).resolve(strict=True)
    if campaign_lock != campaign_root / "campaign_lock.json":
        raise WaiverBootstrapError("campaign lock is not rooted in the campaign")
    if _sha256_file(Path(__file__).resolve()) != args.bootstrap_sha256:
        raise WaiverBootstrapError("waiver bootstrap digest changed after authorization")

    lock = _read_json_object(campaign_lock)
    if _sha256_file(campaign_lock) != args.campaign_lock_digest:
        raise WaiverBootstrapError("campaign lock digest changed after authorization")
    if lock.get("method") != "b3_skillopt":
        raise WaiverBootstrapError("campaign lock is not B3 SkillOpt")
    if lock.get("controller_commit") != args.locked_commit:
        raise WaiverBootstrapError("locked controller commit argument is inconsistent")
    if lock.get("controller_code_digest") != args.locked_code_digest:
        raise WaiverBootstrapError("locked controller digest argument is inconsistent")
    _verify_checkout(repo, commit=args.locked_commit, tree=args.locked_tree)

    sys.path.insert(0, str(repo))
    from experiments.baselines import run_method  # pylint: disable=import-outside-toplevel

    expected_module = (repo / "experiments" / "baselines" / "run_method.py").resolve()
    if Path(run_method.__file__).resolve() != expected_module:
        raise WaiverBootstrapError("legacy B3 controller was imported from another tree")
    if getattr(run_method.hash_code, "__module__", "") != "experiments.protocol":
        raise WaiverBootstrapError("legacy controller hash binding is not recognized")

    waiver = ControllerCodeWaiver(
        repo=repo,
        commit=args.locked_commit,
        tree=args.locked_tree,
        digest=args.locked_code_digest,
        original=run_method.hash_code,
    )
    run_method.hash_code = waiver
    if args.preflight_only:
        if any(
            value is not None
            for value in (
                args.runner_args_file,
                args.runner_args_sha256,
                args.application_receipt,
                args.seed,
            )
        ):
            raise WaiverBootstrapError(
                "preflight-only must not receive seed runner arguments"
            )
        if run_method.hash_code(repo) != args.locked_code_digest or waiver.calls != 1:
            raise WaiverBootstrapError("controller-code waiver preflight failed")
        print(json.dumps({
            "passed": True,
            "scope": "controller_code_digest",
            "locked_commit": args.locked_commit,
            "locked_tree": args.locked_tree,
            "locked_code_digest": args.locked_code_digest,
        }, sort_keys=True))
        return 0

    if (
        args.runner_args_file is None
        or args.runner_args_sha256 is None
        or args.application_receipt is None
    ):
        raise WaiverBootstrapError("runner argument authority is missing")
    if args.seed is None:
        raise WaiverBootstrapError("runner seed authority is missing")
    runner_args_path = Path(args.runner_args_file).resolve(strict=True)
    if _sha256_file(runner_args_path) != args.runner_args_sha256:
        raise WaiverBootstrapError("runner argument digest changed after authorization")
    runner_payload = json.loads(runner_args_path.read_text(encoding="utf-8"))
    if not isinstance(runner_payload, list) or not all(
        isinstance(value, str) for value in runner_payload
    ):
        raise WaiverBootstrapError("runner argument authority must be a string list")
    _verify_runner_arguments(
        runner_payload,
        repo=repo,
        campaign_lock=campaign_lock,
        campaign_root=campaign_root,
        seed=args.seed,
    )
    runner_output = Path(_one_value(runner_payload, "--output-dir")).resolve()
    application_receipt = Path(args.application_receipt).resolve()
    if application_receipt.parent != runner_output.parent:
        raise WaiverBootstrapError("application receipt is outside recovery root")
    if application_receipt.name != (
        f"seed_{args.seed}.controller_code_waiver_application.json"
    ):
        raise WaiverBootstrapError("application receipt name is not seed-bound")
    result = int(run_method.main(runner_payload))
    passed = result == 0 and waiver.calls == 2
    _write_json_new(application_receipt, {
        "schema_version": 1,
        "passed": passed,
        "method": "b3_skillopt",
        "phase": "test",
        "seed": args.seed,
        "waived_checks": ["controller_code_digest"],
        "waived_hash_calls": waiver.calls,
        "locked_commit": args.locked_commit,
        "locked_tree": args.locked_tree,
        "locked_code_digest": args.locked_code_digest,
        "campaign_lock_digest": args.campaign_lock_digest,
        "runner_args_sha256": args.runner_args_sha256,
        "returncode": result,
        "completed_at_unix": time.time(),
    })
    if result == 0 and not passed:
        raise WaiverBootstrapError(
            "successful legacy Test consumed "
            f"{waiver.calls} hash waivers instead of two"
        )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
