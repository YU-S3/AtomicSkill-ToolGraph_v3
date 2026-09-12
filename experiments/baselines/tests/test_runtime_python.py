from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from experiments.baselines.common import runtime_python


class _Distribution:
    def __init__(self, name: str, version: str, root: str) -> None:
        self.metadata = {"Name": name}
        self.version = version
        self._root = Path(root)

    def locate_file(self, _path: str) -> Path:
        return self._root


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_resolve_formal_python_preserves_venv_symlink(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    target = _executable(repo / "system" / "python3.12")
    launcher = repo / ".venv_b3_skillopt" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    try:
        launcher.symlink_to(target)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symbolic links unavailable: {exc}")

    selected = runtime_python.resolve_formal_python(
        repo, ".venv_b3_skillopt/bin/python",
    )
    assert selected == Path(os.path.abspath(launcher))
    assert selected != target


def test_verify_runtime_python_rejects_non_venv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_python.sys, "prefix", "/runtime")
    monkeypatch.setattr(runtime_python.sys, "base_prefix", "/runtime")
    monkeypatch.setattr(
        runtime_python.importlib.util, "find_spec", lambda name: object(),
    )
    with pytest.raises(RuntimeError, match="not running inside a virtual environment"):
        runtime_python.verify_runtime_python(
            expected_python=runtime_python.sys.executable,
            require_venv=True,
        )


@pytest.mark.parametrize("missing", ["alfworld", "skillopt"])
def test_verify_runtime_python_rejects_missing_required_package(
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    monkeypatch.setattr(runtime_python.sys, "prefix", "/runtime/venv")
    monkeypatch.setattr(runtime_python.sys, "base_prefix", "/runtime/base")
    monkeypatch.setattr(
        runtime_python.importlib.util,
        "find_spec",
        lambda name: None if name == missing else object(),
    )
    with pytest.raises(RuntimeError, match=f"cannot import {missing}"):
        runtime_python.verify_runtime_python(
            expected_python=runtime_python.sys.executable,
            require_venv=True,
        )


def test_verify_runtime_python_returns_complete_formal_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_python.sys, "prefix", "/runtime/venv")
    monkeypatch.setattr(runtime_python.sys, "base_prefix", "/runtime/base")
    monkeypatch.setattr(
        runtime_python.importlib.util, "find_spec", lambda name: object(),
    )

    receipt = runtime_python.verify_runtime_python(
        expected_python=runtime_python.sys.executable,
        require_venv=True,
    )
    assert receipt == {
        "expected_python": os.path.abspath(runtime_python.sys.executable),
        "sys_executable": os.path.abspath(runtime_python.sys.executable),
        "sys_prefix": os.path.abspath("/runtime/venv"),
        "sys_base_prefix": os.path.abspath("/runtime/base"),
        "venv_active": True,
        "python_version": (
            f"{runtime_python.sys.version_info.major}."
            f"{runtime_python.sys.version_info.minor}."
            f"{runtime_python.sys.version_info.micro}"
        ),
        "python_major_minor": (
            f"{runtime_python.sys.version_info.major}."
            f"{runtime_python.sys.version_info.minor}"
        ),
        "alfworld_importable": True,
        "skillopt_importable": True,
    }


def test_b5_runtime_requires_gepa_without_requiring_skillopt_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_python.sys, "prefix", "/runtime/venv")
    monkeypatch.setattr(runtime_python.sys, "base_prefix", "/runtime/base")
    monkeypatch.setattr(
        runtime_python.importlib.util,
        "find_spec",
        lambda name: object() if name in {"alfworld", "gepa"} else None,
    )
    monkeypatch.setattr(
        runtime_python.importlib.metadata,
        "version",
        lambda name: (_ for _ in ()).throw(
            runtime_python.importlib.metadata.PackageNotFoundError(name)
        ),
    )

    receipt = runtime_python.verify_runtime_python(
        expected_python=runtime_python.sys.executable,
        require_venv=True,
        method="b5_gepa",
    )

    assert receipt["alfworld_importable"] is True
    assert receipt["gepa_importable"] is True
    assert receipt["alfworld_distribution_version"] is None
    assert receipt["gepa_distribution_version"] is None
    assert receipt["skillopt_distribution_version"] is None
    assert "skillopt_importable" not in receipt


def test_b5_runtime_rejects_installed_skillopt_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_python.sys, "prefix", "/runtime/venv")
    monkeypatch.setattr(runtime_python.sys, "base_prefix", "/runtime/base")
    monkeypatch.setattr(
        runtime_python.importlib.util, "find_spec", lambda name: object()
    )
    monkeypatch.setattr(
        runtime_python.importlib.metadata,
        "distributions",
        lambda: [_Distribution("SkillOpt", "0.2.0", "/runtime/venv/site-packages")],
    )

    with pytest.raises(RuntimeError, match="must not install the SkillOpt distribution"):
        runtime_python.verify_runtime_python(
            expected_python=runtime_python.sys.executable,
            require_venv=True,
            method="b5_gepa",
        )


def test_b5_runtime_binds_expected_distribution_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_python.sys, "prefix", "/runtime/venv")
    monkeypatch.setattr(runtime_python.sys, "base_prefix", "/runtime/base")
    monkeypatch.setattr(
        runtime_python.importlib.util, "find_spec", lambda name: object()
    )
    versions = {"alfworld": "0.4.2", "gepa": "0.1.3"}

    def version(name: str) -> str:
        if name in versions:
            return versions[name]
        raise runtime_python.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(runtime_python.importlib.metadata, "version", version)
    receipt = runtime_python.verify_runtime_python(
        expected_python=runtime_python.sys.executable,
        require_venv=True,
        method="b5_gepa",
        expected_distributions=versions,
    )
    assert receipt["alfworld_distribution_version"] == "0.4.2"
    assert receipt["gepa_distribution_version"] == "0.1.3"

    with pytest.raises(RuntimeError, match="distribution mismatch for gepa"):
        runtime_python.verify_runtime_python(
            expected_python=runtime_python.sys.executable,
            require_venv=True,
            method="b5_gepa",
            expected_distributions={"alfworld": "0.4.2", "gepa": "0.1.4"},
        )


def test_b5_runtime_rejects_missing_gepa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_python.sys, "prefix", "/runtime/venv")
    monkeypatch.setattr(runtime_python.sys, "base_prefix", "/runtime/base")
    monkeypatch.setattr(
        runtime_python.importlib.util,
        "find_spec",
        lambda name: None if name == "gepa" else object(),
    )
    monkeypatch.setattr(
        runtime_python.importlib.metadata,
        "version",
        lambda name: (_ for _ in ()).throw(
            runtime_python.importlib.metadata.PackageNotFoundError(name)
        ),
    )

    with pytest.raises(RuntimeError, match="cannot import gepa"):
        runtime_python.verify_runtime_python(
            expected_python=runtime_python.sys.executable,
            require_venv=True,
            method="b5_gepa",
        )


def test_b4_runtime_binds_python_packages_and_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_python.sys, "prefix", "/runtime/venv")
    monkeypatch.setattr(runtime_python.sys, "base_prefix", "/runtime/base")
    monkeypatch.setattr(
        runtime_python.sys,
        "version_info",
        type("Version", (), {"major": 3, "minor": 9, "micro": 25})(),
    )
    monkeypatch.setattr(
        runtime_python.importlib.util,
        "find_spec",
        lambda name: object() if name == "alfworld" else None,
    )
    versions = {
        "alfworld": "0.4.2",
        "torch": "2.6.0+cpu",
        "sentence-transformers": "4.1.0",
    }

    def version(name: str) -> str:
        if name in versions:
            return versions[name]
        raise runtime_python.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(runtime_python.importlib.metadata, "version", version)
    receipt = runtime_python.verify_runtime_python(
        expected_python=runtime_python.sys.executable,
        require_venv=True,
        method="b4_skillgen_s",
        expected_python_major_minor="3.9",
        expected_distributions={
            "alfworld": "0.4.2",
            "torch": "2.6.0",
            "sentence-transformers": "4.1.0",
        },
    )
    assert receipt["python_major_minor"] == "3.9"
    assert receipt["alfworld_distribution_version"] == "0.4.2"
    assert receipt["torch_distribution_version"] == "2.6.0+cpu"
    assert receipt["skillopt_distribution_version"] is None
    assert receipt["atomic_skillgraph_distribution_version"] is None


def test_b4_runtime_rejects_wrong_python_or_forbidden_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_python.sys, "prefix", "/runtime/venv")
    monkeypatch.setattr(runtime_python.sys, "base_prefix", "/runtime/base")
    monkeypatch.setattr(
        runtime_python.importlib.util, "find_spec", lambda name: object()
    )
    monkeypatch.setattr(
        runtime_python.importlib.metadata,
        "distributions",
        lambda: [
            _Distribution(
                "atomic_skillgraph", "3.0.0", "/runtime/venv/site-packages"
            )
        ],
    )
    with pytest.raises(RuntimeError, match="requires Python 3.9"):
        runtime_python.verify_runtime_python(
            expected_python=runtime_python.sys.executable,
            require_venv=True,
            method="b4_skillgen_s",
            expected_python_major_minor="3.9",
        )

    monkeypatch.setattr(
        runtime_python.sys,
        "version_info",
        type("Version", (), {"major": 3, "minor": 9, "micro": 25})(),
    )
    with pytest.raises(RuntimeError, match="forbidden distributions"):
        runtime_python.verify_runtime_python(
            expected_python=runtime_python.sys.executable,
            require_venv=True,
            method="b4_skillgen_s",
            expected_python_major_minor="3.9",
        )


def test_b4_runtime_ignores_distribution_metadata_from_source_pythonpath(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_python.sys, "prefix", "/runtime/venv")
    monkeypatch.setattr(runtime_python.sys, "base_prefix", "/runtime/base")
    monkeypatch.setattr(
        runtime_python.sys,
        "version_info",
        type("Version", (), {"major": 3, "minor": 9, "micro": 25})(),
    )
    monkeypatch.setattr(
        runtime_python.importlib.util,
        "find_spec",
        lambda name: object() if name == "alfworld" else None,
    )
    monkeypatch.setattr(
        runtime_python.importlib.metadata,
        "distributions",
        lambda: [
            _Distribution(
                "atomic-skillgraph", "3.0.0", "/workspace/repo/src"
            )
        ],
    )

    receipt = runtime_python.verify_runtime_python(
        expected_python=runtime_python.sys.executable,
        require_venv=True,
        method="b4_skillgen_s",
        expected_python_major_minor="3.9",
    )

    assert receipt["atomic_skillgraph_distribution_version"] is None


def test_verify_runtime_python_executable_uses_controlled_source_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    python = _executable(repo / ".venv_b4_skillgen" / "bin" / "python")
    observed: dict[str, object] = {}
    receipt = {
        "sys_executable": str(python),
        "python_major_minor": "3.9",
        "venv_active": True,
    }

    def run(command, **kwargs):
        observed.update({"command": command, **kwargs})
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(receipt) + "\n",
            stderr="",
        )

    monkeypatch.setattr(runtime_python.subprocess, "run", run)
    result = runtime_python.verify_runtime_python_executable(
        repo_root=repo,
        expected_python=python,
        require_venv=True,
        method="b4_skillgen_s",
        expected_distributions={"alfworld": "0.4.2"},
        expected_python_major_minor="3.9",
    )

    assert result == receipt
    assert observed["cwd"] == repo.resolve()
    environment = observed["env"]
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONPATH"].split(os.pathsep) == [
        str(repo.resolve()),
        str(repo.resolve() / "src"),
    ]
