from __future__ import annotations

import os
from pathlib import Path

import pytest

from experiments.baselines.common import runtime_python


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
        "sys_prefix": "/runtime/venv",
        "sys_base_prefix": "/runtime/base",
        "venv_active": True,
        "alfworld_importable": True,
        "skillopt_importable": True,
    }
