from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.baselines import b3_controller_code_waiver_bootstrap as bootstrap
from experiments.baselines import recover_b3_test as recovery


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _build_authority(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Recovery Test")
    (repo / "tracked.txt").write_text("authority\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-q", "-m", "authority")
    commit = _git(repo, "rev-parse", "HEAD")

    campaign = repo / "runs" / "baselines" / "campaign"
    failure_root = campaign / "failed_preflight"
    config = repo / "configs" / "baselines" / "b3_skillopt.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("method: b3_skillopt\n", encoding="utf-8")
    manifest_entries: dict[str, dict[str, str]] = {}
    for role in ("train", "validation", "test"):
        path = repo / "data" / f"{role}.json"
        unsigned = {"schema_version": 1, "role": role}
        payload = {**unsigned, "digest": recovery._sha256_json(unsigned)}
        _write_json(path, payload)
        manifest_entries[role] = {
            "path": str(path.resolve()),
            "digest": str(payload["digest"]),
        }

    code_digest = "a" * 64
    executable = str(Path(sys.executable).absolute())
    lock: dict[str, object] = {
        "schema_version": 1,
        "method": "b3_skillopt",
        "campaign_id": "test_campaign",
        "campaign_root": str(campaign.resolve()),
        "controller_commit": commit,
        "controller_code_digest": code_digest,
        "controller_git": {
            "commit": commit,
            "code_digest": code_digest,
            "dirty": False,
        },
        "phase_python": executable,
        "worker_python": executable,
        "provider_probe_python": executable,
        "config_path": str(config.resolve()),
        "manifests": manifest_entries,
        "seeds": [42, 43, 44],
    }
    campaign.mkdir(parents=True)
    lock_path = campaign / "campaign_lock.json"
    _write_json(lock_path, lock)
    lock_digest = recovery._sha256_file(lock_path)

    for seed in (42, 43, 44):
        train = campaign / f"seed_{seed}" / "train"
        frozen_artifact = train / "frozen" / "artifact"
        frozen_artifact.mkdir(parents=True)
        (frozen_artifact / "best_skill.md").write_text(
            f"seed {seed}\n", encoding="utf-8"
        )
        frozen_digest = recovery._digest_directory(frozen_artifact)
        _write_json(train / "frozen" / "digest.json", {
            "schema_version": 1,
            "method_id": "b3_skillopt",
            "digest": frozen_digest,
        })
        run_id = f"train_{seed}"
        _write_json(train / "completion.json", {
            "passed": True,
            "phase": "train",
            "run_id": run_id,
        })
        _write_json(train / "report.json", {
            "passed": True,
            "method": "b3_skillopt",
            "frozen": {"digest": frozen_digest},
        })
        _write_json(train / "run_manifest.json", {
            "formal": True,
            "phase": "train",
            "method": "b3_skillopt",
            "run_seed": seed,
            "run_id": run_id,
            "controller_code_digest": code_digest,
            "controller_git": {"commit": commit, "dirty": False},
            "identity": {"controller_code_digest": code_digest},
            "campaign": {"campaign_lock_digest": lock_digest},
        })
        _write_json(failure_root / f"seed_{seed}.log", {
            "passed": False,
            "method": "b3_skillopt",
            "phase": "test",
            "failure_kind": "protocol_failure",
            "output_dir": None,
            "error": recovery._EXPECTED_FAILURE,
            "run_id": f"failed_test_{seed}",
        })
    return repo, campaign, failure_root, lock


def test_load_authority_binds_three_trains_and_single_failed_gate(
    tmp_path: Path,
) -> None:
    repo, campaign, failure_root, lock = _build_authority(tmp_path)
    authority = recovery.load_authority(
        campaign,
        failure_root,
        repo=repo,
        executable=sys.executable,
    )
    assert authority.locked_commit == lock["controller_commit"]
    assert authority.locked_code_digest == "a" * 64
    assert [item.seed for item in authority.trains] == [42, 43, 44]
    assert len({item.frozen_digest for item in authority.trains}) == 3
    assert [item["seed"] for item in authority.failure_evidence] == [42, 43, 44]


def test_load_authority_rejects_nonexclusive_failure_evidence(tmp_path: Path) -> None:
    repo, campaign, failure_root, _ = _build_authority(tmp_path)
    path = failure_root / "seed_43.log"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["error"] = "campaign lock does not match this formal run: model"
    _write_json(path, payload)
    with pytest.raises(recovery.RecoveryError, match="controller-code-only"):
        recovery.load_authority(
            campaign,
            failure_root,
            repo=repo,
            executable=sys.executable,
        )


def test_matching_test_manifest_enforces_held_out_exactly_once(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    matching = repo / "runs" / "baselines" / "x" / "seed_42" / "test"
    _write_json(matching / "run_manifest.json", {
        "method": "b3_skillopt",
        "phase": "test",
        "run_seed": 42,
        "campaign": {"campaign_lock_digest": "d" * 64},
    })
    _write_json(
        repo / "runs" / "baselines" / "other" / "run_manifest.json",
        {
            "method": "b3_skillopt",
            "phase": "test",
            "run_seed": 43,
            "campaign": {"campaign_lock_digest": "e" * 64},
        },
    )
    assert recovery._matching_test_manifests(
        repo,
        campaign_lock_digest="d" * 64,
        seeds=(42, 43, 44),
    ) == [(matching / "run_manifest.json").resolve()]


def test_controller_code_waiver_is_repo_scoped_and_limited(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path.resolve()
    verified = []
    monkeypatch.setattr(
        bootstrap,
        "_verify_checkout",
        lambda root, *, commit, tree: verified.append((root, commit, tree)),
    )
    waiver = bootstrap.ControllerCodeWaiver(
        repo=repo,
        commit="c" * 40,
        tree="d" * 40,
        digest="e" * 64,
        original=lambda root: f"original:{Path(root).name}",
    )
    assert waiver(repo / "elsewhere") == "original:elsewhere"
    assert waiver(repo) == "e" * 64
    assert waiver(repo) == "e" * 64
    with pytest.raises(bootstrap.WaiverBootstrapError, match="excess"):
        waiver(repo)
    assert len(verified) == 2


def test_bootstrap_runner_arguments_allow_only_locked_b3_test(tmp_path: Path) -> None:
    repo = tmp_path.resolve()
    campaign = repo / "runs" / "campaign"
    lock = campaign / "campaign_lock.json"
    output = campaign / "recovered_test" / "seed_42"
    arguments = [
        "--method", "b3_skillopt",
        "--phase", "test",
        "--seed", "42",
        "--campaign-lock", str(lock),
        "--source-run", str(campaign / "seed_42" / "train"),
        "--output-dir", str(output),
    ]
    bootstrap._verify_runner_arguments(
        arguments,
        repo=repo,
        campaign_lock=lock,
        campaign_root=campaign,
        seed=42,
    )
    arguments[arguments.index("test")] = "train"
    with pytest.raises(bootstrap.WaiverBootstrapError, match="Test phase"):
        bootstrap._verify_runner_arguments(
            arguments,
            repo=repo,
            campaign_lock=lock,
            campaign_root=campaign,
            seed=42,
        )

    arguments[arguments.index("train")] = "test"
    arguments[arguments.index(str(output))] = str(
        repo / "runs" / "another_campaign" / "recovered_test" / "seed_42"
    )
    with pytest.raises(bootstrap.WaiverBootstrapError, match="campaign recovery"):
        bootstrap._verify_runner_arguments(
            arguments,
            repo=repo,
            campaign_lock=lock,
            campaign_root=campaign,
            seed=42,
        )


def test_restore_checkout_returns_to_original_branch_and_commit(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Recovery Test")
    path = repo / "tracked.txt"
    path.write_text("one\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-q", "-m", "one")
    first = _git(repo, "rev-parse", "HEAD")
    path.write_text("two\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-q", "-m", "two")
    original = recovery._checkout_state(repo, require_clean=True)
    _git(repo, "switch", "--detach", first)
    restored = recovery._restore_checkout(original, repo=repo)
    assert restored == original
    assert _git(repo, "rev-parse", "HEAD") == original.commit


def test_runner_arguments_never_include_train_or_resume(tmp_path: Path) -> None:
    repo, campaign, failure_root, _ = _build_authority(tmp_path)
    authority = recovery.load_authority(
        campaign,
        failure_root,
        repo=repo,
        executable=sys.executable,
    )
    output = repo / "runs" / "recovery"
    for train in authority.trains:
        arguments = recovery._runner_arguments(
            authority,
            train=train,
            output_root=output,
        )
        assert arguments[arguments.index("--phase") + 1] == "test"
        assert "--resume-source-run" not in arguments
        assert arguments[arguments.index("--seed") + 1] == str(train.seed)
        assert arguments[arguments.index("--source-run") + 1] == str(train.root)
