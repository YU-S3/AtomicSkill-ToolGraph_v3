"""Fail-closed external-source verification tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from experiments.baselines.bootstrap_external import (
    compute_runtime_tree,
    ensure_pinned_source,
    ensure_skillopt_source,
    load_lock,
    setup_method_worker_venv,
    source_file_sha256,
    verify_key_files,
    verify_runtime_tree,
    verify_worker_environment,
    verify_worker_isolation,
    verify_worker_python,
    worker_venv_python,
)
from experiments.baselines import bootstrap_external


def _materialize_runtime(root: Path) -> None:
    (root / "skillopt" / "engine").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "skillopt" / "engine" / "trainer.py").write_text(
        "VALUE = 1\n", encoding="utf-8",
    )
    (root / "skillopt" / "model.py").write_text(
        "MODEL = 'test'\n", encoding="utf-8",
    )
    (root / "configs" / "default.yaml").write_text(
        "seed: 42\n", encoding="utf-8",
    )
    for name in ("LICENSE", "README.md", "pyproject.toml", "requirements.txt"):
        (root / name).write_text(f"{name}\n", encoding="utf-8")


def test_create_venv_finds_user_local_uv_without_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = "uv.exe" if bootstrap_external.os.name == "nt" else "uv"
    uv = tmp_path / ".local" / "bin" / executable
    uv.parent.mkdir(parents=True)
    uv.write_text("#!/bin/sh\n", encoding="utf-8")
    uv.chmod(0o755)
    commands: list[list[str]] = []
    monkeypatch.delenv("UV_BIN", raising=False)
    monkeypatch.setattr(bootstrap_external.shutil, "which", lambda name: None)
    monkeypatch.setattr(bootstrap_external.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        bootstrap_external.os,
        "access",
        lambda path, mode: Path(path) == uv,
    )
    monkeypatch.setattr(
        bootstrap_external,
        "_run",
        lambda command, **kwargs: commands.append(list(command)),
    )

    destination = tmp_path / "worker"
    bootstrap_external._create_venv(destination, python_version="3.12")

    assert commands == [[
        str(uv), "venv", "--seed", "--python", "3.12", str(destination)
    ]]


def test_find_uv_rejects_invalid_explicit_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UV_BIN", str(tmp_path / "missing-uv"))
    with pytest.raises(RuntimeError, match="UV_BIN is not an executable"):
        bootstrap_external._find_uv()


def _runtime_spec() -> dict:
    return {
        "algorithm": "sha256-path-content-v1",
        "roots": ["skillopt", "configs"],
        "root_files": [
            "LICENSE", "README.md", "pyproject.toml", "requirements.txt",
        ],
    }


def _lock_for(root: Path) -> dict:
    spec = _runtime_spec()
    spec.update(compute_runtime_tree(root, spec))
    trainer = root / "skillopt" / "engine" / "trainer.py"
    return {
        "schema_version": 1,
        "skillopt": {
            "repo": "https://github.com/microsoft/SkillOpt",
            "commit": "a" * 40,
            "version": "0.2.0",
            "runtime_tree": spec,
            "key_files": {
                "skillopt/engine/trainer.py": hashlib.sha256(
                    trainer.read_bytes()
                ).hexdigest(),
            },
        },
    }


def test_complete_runtime_tree_accepts_exact_source_and_ignores_caches(tmp_path) -> None:
    root = tmp_path / "source"
    _materialize_runtime(root)
    lock = _lock_for(root)

    cache = root / "skillopt" / "__pycache__"
    cache.mkdir()
    (cache / "model.cpython-312.pyc").write_bytes(b"generated")

    verified = verify_runtime_tree(root, "skillopt", lock)
    assert verified["sha256"] == lock["skillopt"]["runtime_tree"]["sha256"]
    assert verified["file_count"] == 7
    assert verify_key_files(root, "skillopt", lock) == lock["skillopt"]["key_files"]


@pytest.mark.parametrize("mutation", ["modify", "add", "remove"])
def test_complete_runtime_tree_rejects_any_unlisted_source_drift(
    tmp_path,
    mutation: str,
) -> None:
    root = tmp_path / "source"
    _materialize_runtime(root)
    lock = _lock_for(root)

    model = root / "skillopt" / "model.py"
    if mutation == "modify":
        model.write_text("MODEL = 'changed'\n", encoding="utf-8")
    elif mutation == "add":
        (root / "skillopt" / "extra.py").write_text("EXTRA = 1\n", encoding="utf-8")
    else:
        model.unlink()

    with pytest.raises(RuntimeError, match="runtime tree does not match"):
        verify_key_files(root, "skillopt", lock)


def test_runtime_tree_rejects_traversing_lock_path(tmp_path) -> None:
    root = tmp_path / "source"
    _materialize_runtime(root)
    spec = _runtime_spec()
    spec["roots"] = ["../outside"]

    with pytest.raises(ValueError, match="invalid runtime_tree.roots path"):
        compute_runtime_tree(root, spec)


def test_ensure_skillopt_source_reports_commit_version_and_tree(tmp_path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "external" / "skillopt"
    _materialize_runtime(source)
    lock = _lock_for(source)
    lock_path = tmp_path / "baseline_lock.yaml"
    lock_path.write_text(yaml.safe_dump(lock, sort_keys=False), encoding="utf-8")

    result = ensure_skillopt_source(
        local_source=source,
        destination=destination,
        lock_path=lock_path,
    )

    assert result["declared_commit"] == "a" * 40
    assert result["declared_version"] == "0.2.0"
    assert result["verification"] == "runtime_tree_sha256+key_file_sha256"
    assert result["runtime_tree"] == {
        "algorithm": "sha256-path-content-v1",
        "sha256": lock["skillopt"]["runtime_tree"]["sha256"],
        "file_count": 7,
    }
    assert load_lock(lock_path)["skillopt"]["runtime_tree"]["file_count"] == 7


def test_committed_lock_pins_latest_verified_runtime_tree() -> None:
    lock = load_lock()
    skillopt = lock["skillopt"]

    assert skillopt["commit"] == "79124b37e9a6371e13b753f8bcd7adb1e493ade1"
    assert skillopt["version"] == "0.2.0"
    assert skillopt["local_snapshot_verification"] == (
        "runtime_tree_sha256+key_file_sha256"
    )
    assert skillopt["runtime_tree"]["file_count"] == 155
    assert len(skillopt["runtime_tree"]["sha256"]) == 64


def test_canonical_runtime_tree_is_independent_of_text_line_endings(tmp_path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    for root, newline in ((left, b"\n"), (right, b"\r\n")):
        (root / "package").mkdir(parents=True)
        (root / "package" / "module.py").write_bytes(
            newline.join((b"VALUE = 1", b"VALUE = 2", b""))
        )
        (root / "asset.png").write_bytes(b"\x89PNG\r\n\x1a\nraw\rbytes")
        (root / "uv.lock").write_bytes(
            newline.join((b"version = 1", b"revision = 2", b""))
        )
    spec = {
        "algorithm": "sha256-path-canonical-content-v2",
        "roots": ["package"],
        "root_files": ["asset.png", "uv.lock"],
    }

    assert compute_runtime_tree(left, spec) == compute_runtime_tree(right, spec)
    assert source_file_sha256(
        left / "package" / "module.py", algorithm=spec["algorithm"]
    ) == source_file_sha256(
        right / "package" / "module.py", algorithm=spec["algorithm"]
    )


def test_runtime_tree_algorithm_is_bound_into_digest_domain(tmp_path) -> None:
    root = tmp_path / "source"
    (root / "package").mkdir(parents=True)
    (root / "package" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    common = {"roots": ["package"], "root_files": ["package/module.py"]}

    raw = compute_runtime_tree(
        root, {"algorithm": "sha256-path-content-v1", **common}
    )
    canonical = compute_runtime_tree(
        root, {"algorithm": "sha256-path-canonical-content-v2", **common}
    )

    assert raw["sha256"] != canonical["sha256"]


def test_canonical_runtime_tree_preserves_binary_bytes(tmp_path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    for root in (left, right):
        (root / "package").mkdir(parents=True)
    (left / "package" / "asset.png").write_bytes(b"raw\r\nbytes")
    (right / "package" / "asset.png").write_bytes(b"raw\nbytes")
    spec = {
        "algorithm": "sha256-path-canonical-content-v2",
        "roots": ["package"],
        "root_files": ["package/asset.png"],
    }

    assert compute_runtime_tree(left, spec) != compute_runtime_tree(right, spec)


def test_pinned_local_source_requires_git_commit_proof(tmp_path) -> None:
    source = tmp_path / "source"
    (source / "package").mkdir(parents=True)
    (source / "package" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    spec = {
        "algorithm": "sha256-path-canonical-content-v2",
        "roots": ["package"],
        "root_files": ["package/module.py"],
    }
    spec.update(compute_runtime_tree(source, spec))
    lock = {
        "schema_version": 1,
        "skillgen": {
            "repo": "https://example.invalid/skillgen",
            "commit": "a" * 40,
            "runtime_tree": spec,
            "key_files": {
                "package/module.py": source_file_sha256(
                    source / "package" / "module.py", algorithm=spec["algorithm"]
                ),
            },
        },
    }
    lock_path = tmp_path / "lock.yaml"
    lock_path.write_text(yaml.safe_dump(lock), encoding="utf-8")

    with pytest.raises(RuntimeError, match="must retain Git metadata"):
        ensure_pinned_source(
            "skillgen",
            local_source=source,
            destination=tmp_path / "external" / "skillgen",
            lock_path=lock_path,
        )


def test_git_checkout_verifies_annotated_tag_target(tmp_path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "core.autocrlf", "true"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Baseline Test"], cwd=root, check=True)
    (root / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "source.py"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=root, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    subprocess.run(["git", "tag", "-a", "v1", "-m", "fixture tag"], cwd=root, check=True)

    receipt = bootstrap_external._verify_git_checkout(
        root, expected_commit=commit, expected_tag="v1"
    )
    assert receipt["tag_commit"] == commit
    with pytest.raises(RuntimeError, match="no verifiable tag"):
        bootstrap_external._verify_git_checkout(
            root, expected_commit=commit, expected_tag="v2"
        )


def test_worker_python_rejects_wrong_actual_version(tmp_path, monkeypatch) -> None:
    executable = tmp_path / "python"
    executable.touch()

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0], returncode=0,
            stdout=json.dumps({
                "version": "3.12",
                "version_info": [3, 12, 0],
                "executable": str(executable),
                "prefix": "/venv",
                "base_prefix": "/base",
                "venv_active": True,
            }),
            stderr="",
        )

    monkeypatch.setattr(bootstrap_external.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="expected 3.9"):
        verify_worker_python(executable, expected_version="3.9")


def test_worker_interpreter_path_is_platform_specific(tmp_path) -> None:
    assert worker_venv_python(
        tmp_path, host_os="nt"
    ) == tmp_path / "Scripts" / "python.exe"
    assert worker_venv_python(
        tmp_path, host_os="posix"
    ) == tmp_path / "bin" / "python"


def test_worker_environment_accepts_exact_versions_and_cpu_torch(
    tmp_path, monkeypatch,
) -> None:
    python = tmp_path / "python"
    installed = {
        "alfworld": "0.4.2",
        "numpy": "1.22.4",
        "torch": "2.6.0+cpu",
    }

    monkeypatch.setattr(
        bootstrap_external.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout=json.dumps(installed), stderr="",
        ),
    )

    receipt = verify_worker_environment(
        python,
        expected_versions={
            "alfworld": "0.4.2",
            "numpy": "1.22.4",
            "torch": "2.6.0",
        },
    )
    assert receipt["distributions"] == installed


def test_worker_environment_rejects_distribution_drift(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        bootstrap_external.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0], returncode=0,
            stdout=json.dumps({"numpy": "2.0.0"}), stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match="numpy: expected 1.22.4, got 2.0.0"):
        verify_worker_environment(
            tmp_path / "python", expected_versions={"numpy": "1.22.4"}
        )


def test_worker_isolation_rejects_forbidden_distribution_and_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bootstrap_external.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=json.dumps({
                "distributions": {"skillopt": "0.2.0"},
                "modules": {"skillopt": "/ambient/skillopt/__init__.py"},
            }),
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match="forbidden controller/method"):
        verify_worker_isolation(
            tmp_path / "python",
            forbidden_distributions=("skillopt",),
            forbidden_modules=("skillopt",),
        )


def test_worker_isolation_probe_discards_ambient_pythonpath(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "ambient"))

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=json.dumps({"distributions": {}, "modules": {}}),
            stderr="",
        )

    monkeypatch.setattr(bootstrap_external.subprocess, "run", fake_run)
    receipt = verify_worker_isolation(
        tmp_path / "python",
        forbidden_distributions=("skillopt",),
        forbidden_modules=("skillopt",),
    )

    environment = dict(captured["env"])
    assert "PYTHONPATH" not in environment
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert receipt["forbidden_distributions_absent"] == ["skillopt"]


def test_new_method_dependency_sets_are_exactly_pinned() -> None:
    package_sets = (
        bootstrap_external._COMMON_ALFWORLD_PACKAGES,
        bootstrap_external._SKILLGEN_INSTALL_PACKAGES,
        bootstrap_external._SKILLGEN_TORCH_PACKAGES,
        bootstrap_external._GEPA_INSTALL_PACKAGES,
    )
    assert all(
        "==" in requirement and ">=" not in requirement
        for packages in package_sets
        for requirement in packages
    )


def test_skillgen_python39_resolution_and_receipt_pin_upstream_textworld_chain() -> None:
    expected = {
        "textworld": "1.6.2",
        "fast_downward_textworld": "20.6.3",
        "jericho": "3.3.0",
        "spacy": "3.4.4",
        "thinc": "8.1.12",
        "pydantic": "1.10.8",
        "blis": "0.7.11",
        "catalogue": "2.0.10",
        "confection": "0.1.5",
        "cymem": "2.0.11",
        "preshed": "3.0.9",
        "srsly": "2.5.1",
    }
    requirements = set(bootstrap_external._SKILLGEN_INSTALL_PACKAGES)

    for distribution, version in expected.items():
        assert f"{distribution}=={version}" in requirements
        assert (
            bootstrap_external._WORKER_EXPECTED_DISTRIBUTIONS["skillgen"][
                distribution
            ]
            == version
        )


@pytest.mark.parametrize("method,version", [("skillgen", "3.9"), ("gepa", "3.12")])
def test_new_method_setup_does_not_install_repo_or_skillopt_distribution(
    tmp_path, monkeypatch, method: str, version: str,
) -> None:
    source = tmp_path / method
    source.mkdir()
    skillopt = tmp_path / "skillopt"
    skillopt.mkdir()
    python = tmp_path / "venv" / "bin" / "python"
    commands: list[list[str]] = []
    isolation_calls: list[dict[str, object]] = []
    monkeypatch.setattr(bootstrap_external, "verify_key_files", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        bootstrap_external,
        "_prepare_worker_python",
        lambda *args, **kwargs: (
            python,
            {"version": version, "venv_active": True},
        ),
    )
    monkeypatch.setattr(bootstrap_external, "_ensure_pip", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        bootstrap_external,
        "verify_worker_isolation",
        lambda *args, **kwargs: isolation_calls.append(dict(kwargs)) or {},
    )
    monkeypatch.setattr(
        bootstrap_external,
        "verify_worker_environment",
        lambda *args, **kwargs: {"distributions": {}},
    )
    monkeypatch.setattr(
        bootstrap_external,
        "_run",
        lambda command, **kwargs: commands.append([str(item) for item in command]),
    )

    setup_method_worker_venv(
        method,
        source_root=source,
        venv_path=tmp_path / "venv",
        skillopt_root=skillopt if method == "gepa" else None,
    )

    assert len(isolation_calls) == 1
    assert isolation_calls[0]["forbidden_distributions"] == (
        "atomic-skillgraph", "skillopt"
    )

    editable_targets = [
        command[-1] for command in commands if "-e" in command
    ]
    assert str(bootstrap_external.REPO_ROOT) not in editable_targets
    assert str(skillopt) not in editable_targets
    if method == "skillgen":
        assert editable_targets == []
        assert any(
            "--index-url" in command
            and bootstrap_external._PYTORCH_CPU_INDEX in command
            and "torch==2.6.0" in command
            for command in commands
        )
    else:
        assert editable_targets == [str(source)]
        assert any("openai==1.75.0" in command for command in commands)
