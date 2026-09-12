"""Formal Python-runtime authority helpers.

Python virtual-environment launchers are commonly symbolic links.  Their
lexical path is part of the runtime authority: resolving the link to the base
interpreter can silently disable the virtual environment and its packages.
"""

from __future__ import annotations

import importlib.util
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


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


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _normalized_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", str(name)).lower()


def _path_is_within(path: str | Path, parent: str | Path) -> bool:
    candidate = os.path.normcase(os.path.realpath(os.fspath(path)))
    boundary = os.path.normcase(os.path.realpath(os.fspath(parent)))
    try:
        return os.path.commonpath((candidate, boundary)) == boundary
    except ValueError:
        return False


def _installed_distribution_version(name: str) -> str | None:
    """Return a version only for metadata installed in this runtime prefix.

    Formal B4/B5 workers intentionally receive the controller source tree on
    ``PYTHONPATH``.  ``importlib.metadata.version`` also discovers the
    repository's adjacent ``*.egg-info`` and therefore cannot distinguish that
    source authority from a distribution installed into the worker venv.
    Enumerating all matching metadata keeps the check fail-closed for a real
    venv installation while ignoring metadata that merely arrived via
    ``PYTHONPATH``.
    """

    normalized = _normalized_distribution_name(name)
    for distribution in importlib.metadata.distributions():
        discovered = distribution.metadata.get("Name")
        if not discovered or _normalized_distribution_name(discovered) != normalized:
            continue
        if _path_is_within(distribution.locate_file(""), sys.prefix):
            return str(distribution.version)
    return None


def verify_runtime_python(
    *,
    expected_python: str | Path,
    require_venv: bool,
    method: str = "b3_skillopt",
    expected_distributions: Mapping[str, str] | None = None,
    expected_python_major_minor: str | None = None,
) -> dict[str, Any]:
    """Fail closed unless the current formal method runtime has its dependencies.

    B3 executes installed SkillOpt directly.  B4 and B5 keep SkillOpt and the
    controller distribution out of their isolated environments; method source
    is provided through identity-checked ``PYTHONPATH`` roots instead.
    """

    normalized = str(method).strip().lower()
    aliases = {
        "skillopt": "b3_skillopt",
        "skillgen": "b4_skillgen_s",
        "gepa": "b5_gepa",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"b3_skillopt", "b4_skillgen_s", "b5_gepa"}:
        raise ValueError(f"unsupported formal runtime method: {method!r}")

    expected = _lexical_absolute(expected_python)
    receipt = runtime_python_receipt()
    receipt["expected_python"] = str(expected)
    receipt["python_version"] = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    receipt["python_major_minor"] = (
        f"{sys.version_info.major}.{sys.version_info.minor}"
    )
    receipt["alfworld_importable"] = _package_importable("alfworld")
    if normalized == "b3_skillopt":
        receipt["skillopt_importable"] = _package_importable("skillopt")
    elif normalized == "b5_gepa":
        receipt["gepa_importable"] = _package_importable("gepa")
        receipt["alfworld_distribution_version"] = _distribution_version("alfworld")
        receipt["gepa_distribution_version"] = _distribution_version("gepa")
    if normalized in {"b4_skillgen_s", "b5_gepa"}:
        receipt["skillopt_distribution_version"] = _installed_distribution_version(
            "skillopt"
        )
        receipt["atomic_skillgraph_distribution_version"] = _installed_distribution_version(
            "atomic-skillgraph"
        )
    labels = {
        "b3_skillopt": "B3",
        "b4_skillgen_s": "B4",
        "b5_gepa": "B5",
    }
    label = labels[normalized]
    if receipt["sys_executable"] != str(expected):
        raise RuntimeError(
            f"formal {label} runtime interpreter mismatch: "
            f"expected {expected}, running {receipt['sys_executable']}"
        )
    if require_venv and receipt["venv_active"] is not True:
        raise RuntimeError(
            f"formal {label} runtime is not running inside a virtual environment"
        )
    if receipt["alfworld_importable"] is not True:
        raise RuntimeError(f"formal {label} runtime cannot import alfworld")
    if (
        expected_python_major_minor is not None
        and receipt["python_major_minor"] != str(expected_python_major_minor)
    ):
        raise RuntimeError(
            f"formal {label} runtime requires Python {expected_python_major_minor}, "
            f"got {receipt['python_major_minor']}"
        )
    if normalized == "b3_skillopt" and receipt["skillopt_importable"] is not True:
        raise RuntimeError("formal B3 runtime cannot import skillopt")
    if normalized == "b5_gepa":
        if receipt["gepa_importable"] is not True:
            raise RuntimeError("formal B5 runtime cannot import gepa")
    if normalized in {"b4_skillgen_s", "b5_gepa"}:
        contaminated = {
            name: receipt[f"{name.replace('-', '_')}_distribution_version"]
            for name in ("skillopt", "atomic-skillgraph")
            if receipt[f"{name.replace('-', '_')}_distribution_version"] is not None
        }
        if contaminated:
            if set(contaminated) == {"skillopt"}:
                detail = "must not install the SkillOpt distribution"
            else:
                detail = f"contains forbidden distributions {contaminated}"
            raise RuntimeError(
                f"formal {label} runtime {detail}; recreate the isolated worker venv"
            )
    frozen_distributions = dict(expected_distributions or {})
    for distribution in sorted(frozen_distributions):
        field = f"{distribution.replace('-', '_')}_distribution_version"
        receipt[field] = _distribution_version(distribution)
    for distribution, expected_version in sorted(frozen_distributions.items()):
        field = f"{distribution.replace('-', '_')}_distribution_version"
        actual_version = receipt.get(field)
        compatible = actual_version == str(expected_version) or (
            distribution == "torch"
            and str(actual_version).split("+", 1)[0] == str(expected_version)
        )
        if not compatible:
            raise RuntimeError(
                f"formal {label} runtime distribution mismatch for {distribution}: "
                f"expected {expected_version}, got {actual_version}"
            )
    return receipt


def verify_runtime_python_executable(
    *,
    repo_root: str | Path,
    expected_python: str | Path,
    require_venv: bool,
    method: str,
    expected_distributions: Mapping[str, str] | None = None,
    expected_python_major_minor: str | None = None,
) -> dict[str, Any]:
    """Verify an isolated worker interpreter without running its controller.

    Baseline controllers run in the main experiment environment.  B4 in
    particular requires Python 3.9 for pinned SkillGen dependencies, while
    post-hoc evaluation uses the newer controller runtime.  This probe checks
    the exact worker executable with a controlled source-only ``PYTHONPATH``.
    """

    root = Path(repo_root).resolve(strict=True)
    python = resolve_formal_python(root, expected_python)
    arguments = {
        "expected_python": str(python),
        "require_venv": bool(require_venv),
        "method": str(method),
        "expected_distributions": dict(expected_distributions or {}),
        "expected_python_major_minor": expected_python_major_minor,
    }
    script = (
        "import json,sys;"
        "from experiments.baselines.common.runtime_python import verify_runtime_python;"
        "payload=json.loads(sys.argv[1]);"
        "print(json.dumps(verify_runtime_python(**payload),sort_keys=True))"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(root), str(root / "src")))
    environment["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [str(python), "-c", script, json.dumps(arguments, sort_keys=True)],
        cwd=root,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"formal {method} worker runtime verification failed: {detail[:2000]}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(
            f"formal {method} worker runtime emitted an invalid receipt"
        )
    try:
        receipt = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"formal {method} worker runtime receipt is not JSON"
        ) from exc
    if not isinstance(receipt, dict):
        raise RuntimeError(f"formal {method} worker runtime receipt is not an object")
    return receipt
