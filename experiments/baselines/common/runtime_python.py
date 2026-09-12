"""Formal Python-runtime authority helpers.

Python virtual-environment launchers are commonly symbolic links.  Their
lexical path is part of the runtime authority: resolving the link to the base
interpreter can silently disable the virtual environment and its packages.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any


def _lexical_absolute(value: str | Path, *, base: Path | None = None) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        if base is None:
            raise ValueError(f"formal Python path must be absolute: {value}")
        raw = Path(base).expanduser() / raw
    # Intentionally do not use Path.resolve(), realpath(), or samefile().
    return Path(os.path.abspath(os.fspath(raw)))


def resolve_formal_python(repo_root: Path, value: str | Path) -> Path:
    """Return an executable's absolute lexical path without following links."""

    if not str(value).strip():
        raise ValueError("formal Python executable is empty")
    path = _lexical_absolute(value, base=repo_root)
    if not path.is_file():
        raise FileNotFoundError(f"formal Python executable is missing: {path}")
    if not os.access(path, os.X_OK):
        raise PermissionError(f"formal Python executable is not executable: {path}")
    return path


def runtime_python_receipt() -> dict[str, Any]:
    """Describe the interpreter which is executing the current process."""

    executable = os.path.abspath(sys.executable)
    prefix = os.path.abspath(sys.prefix)
    base_prefix = os.path.abspath(sys.base_prefix)
    return {
        "sys_executable": executable,
        "sys_prefix": prefix,
        "sys_base_prefix": base_prefix,
        "venv_active": prefix != base_prefix,
    }


def _package_importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def verify_runtime_python(
    *,
    expected_python: str | Path,
    require_venv: bool,
) -> dict[str, Any]:
    """Fail closed unless the current formal B3 runtime has its dependencies."""

    expected = _lexical_absolute(expected_python)
    receipt = runtime_python_receipt()
    receipt.update({
        "expected_python": str(expected),
        "alfworld_importable": _package_importable("alfworld"),
        "skillopt_importable": _package_importable("skillopt"),
    })
    if receipt["sys_executable"] != str(expected):
        raise RuntimeError(
            "formal B3 runtime interpreter mismatch: "
            f"expected {expected}, running {receipt['sys_executable']}"
        )
    if require_venv and receipt["venv_active"] is not True:
        raise RuntimeError("formal B3 runtime is not running inside a virtual environment")
    if receipt["alfworld_importable"] is not True:
        raise RuntimeError("formal B3 runtime cannot import alfworld")
    if receipt["skillopt_importable"] is not True:
        raise RuntimeError("formal B3 runtime cannot import skillopt")
    return receipt
