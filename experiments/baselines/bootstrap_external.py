"""Bootstrap and verify external baseline checkouts from ``baseline_lock.yaml``.

Responsibilities (per the baseline design document, section 8.1):

1. materialize the pinned external source (git clone at the pinned commit, or
   a verified copy of a local snapshot);
2. verify the pinned runtime-tree digest and key-file hashes fail-closed;
3. optionally create the per-method worker venv and install the upstream
   package plus the pinned ALFWorld dependency.

Only ``baseline_lock.yaml`` and this bootstrap script are committed;
``.external/`` and the worker venv are gitignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = REPO_ROOT / ".external"

_SKILLOPT_INSTALL_PACKAGES = [
    "alfworld==0.4.2",
    "gymnasium>=0.29.0",
    "omegaconf>=2.3.0",
]

_RUNTIME_TREE_ALGORITHM = "sha256-path-content-v1"
_RUNTIME_TREE_IGNORED_DIRS = frozenset({
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
})
_RUNTIME_TREE_IGNORED_SUFFIXES = frozenset({".pyc", ".pyo"})


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_lock(lock_path: str | Path = REPO_ROOT / "experiments" / "baselines" / "baseline_lock.yaml") -> dict[str, Any]:
    payload = yaml.safe_load(_path(lock_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != 1:
        raise ValueError("baseline_lock.yaml must be a mapping with schema_version: 1")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: object, *, field: str) -> str:
    """Normalize one lock-file path and reject absolute/traversing entries."""

    raw = str(value or "").strip().replace("\\", "/")
    relative = PurePosixPath(raw)
    if (
        not raw
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"invalid {field} path in baseline_lock.yaml: {value!r}")
    return relative.as_posix()


def _runtime_tree_files(root: Path, spec: dict[str, Any]) -> list[tuple[str, Path]]:
    """Enumerate every locked runtime file using stable POSIX relative paths."""

    entries: dict[str, Path] = {}
    roots = list(spec.get("roots") or [])
    root_files = list(spec.get("root_files") or [])
    if not roots or not root_files:
        raise ValueError(
            "runtime_tree must declare non-empty roots and root_files"
        )

    for raw_relative in roots:
        relative = _safe_relative_path(raw_relative, field="runtime_tree.roots")
        directory = root.joinpath(*PurePosixPath(relative).parts)
        if not directory.is_dir():
            raise FileNotFoundError(f"locked runtime tree root is missing: {relative}")
        if directory.is_symlink():
            raise RuntimeError(f"locked runtime tree root must not be a symlink: {relative}")
        for path in directory.rglob("*"):
            rel_path = path.relative_to(root)
            if any(part in _RUNTIME_TREE_IGNORED_DIRS for part in rel_path.parts):
                continue
            if path.is_symlink():
                raise RuntimeError(
                    "locked runtime tree must not contain symlinks: "
                    + rel_path.as_posix()
                )
            if not path.is_file() or path.suffix.lower() in _RUNTIME_TREE_IGNORED_SUFFIXES:
                continue
            entries[rel_path.as_posix()] = path

    for raw_relative in root_files:
        relative = _safe_relative_path(raw_relative, field="runtime_tree.root_files")
        path = root.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file():
            raise FileNotFoundError(f"locked runtime root file is missing: {relative}")
        if path.is_symlink():
            raise RuntimeError(f"locked runtime root file must not be a symlink: {relative}")
        entries[relative] = path

    return sorted(entries.items())


def compute_runtime_tree(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic digest of the complete declared runtime tree.

    The digest binds both relative paths and raw file contents.  Consequently,
    adding, removing, renaming, or modifying any runtime file changes it.
    """

    algorithm = str(spec.get("algorithm") or "")
    if algorithm != _RUNTIME_TREE_ALGORITHM:
        raise ValueError(
            "unsupported runtime_tree algorithm: "
            f"{algorithm!r}; expected {_RUNTIME_TREE_ALGORITHM!r}"
        )
    entries = _runtime_tree_files(root, spec)
    digest = hashlib.sha256()
    digest.update((_RUNTIME_TREE_ALGORITHM + "\0").encode("ascii"))
    for relative, path in entries:
        relative_bytes = relative.encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return {
        "algorithm": algorithm,
        "sha256": digest.hexdigest(),
        "file_count": len(entries),
    }


def verify_runtime_tree(
    root: Path,
    method: str,
    lock: dict[str, Any],
) -> dict[str, Any]:
    """Verify the complete pinned runtime tree, failing closed on any drift."""

    spec = dict((lock.get(method) or {}).get("runtime_tree") or {})
    if not spec:
        raise ValueError(f"baseline_lock.yaml has no runtime_tree for {method}")
    expected_digest = str(spec.get("sha256") or "").lower()
    try:
        expected_count = int(spec.get("file_count"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"baseline_lock.yaml has invalid runtime_tree.file_count for {method}"
        ) from exc
    if len(expected_digest) != 64 or any(
        char not in "0123456789abcdef" for char in expected_digest
    ):
        raise ValueError(
            f"baseline_lock.yaml has invalid runtime_tree.sha256 for {method}"
        )
    actual = compute_runtime_tree(root, spec)
    if actual["file_count"] != expected_count or actual["sha256"] != expected_digest:
        raise RuntimeError(
            f"external {method} runtime tree does not match baseline_lock.yaml: "
            f"expected count={expected_count} sha256={expected_digest}, "
            f"got count={actual['file_count']} sha256={actual['sha256']}"
        )
    return actual


def verify_key_files(root: Path, method: str, lock: dict[str, Any]) -> dict[str, str]:
    """Verify key files and the complete runtime tree, fail-closed.

    The function name is retained for existing driver imports.  Runtime-tree
    verification is mandatory so a modified core file cannot evade preflight
    merely because it is absent from the shorter human-auditable key list.
    """

    expected = dict((lock.get(method) or {}).get("key_files") or {})
    if not expected:
        raise ValueError(f"baseline_lock.yaml has no key_files for {method}")
    mismatches: list[str] = []
    for relative, wanted in sorted(expected.items()):
        path = root / relative
        if not path.is_file():
            mismatches.append(f"{relative}: missing")
            continue
        actual = _sha256_file(path)
        if actual != str(wanted):
            mismatches.append(f"{relative}: expected {wanted}, got {actual}")
    if mismatches:
        raise RuntimeError(
            f"external {method} snapshot does not match baseline_lock.yaml: "
            + "; ".join(mismatches)
        )
    verify_runtime_tree(root, method, lock)
    return {relative: str(wanted) for relative, wanted in expected.items()}


def ensure_skillopt_source(
    *,
    local_source: str | Path | None,
    destination: Path | None = None,
    lock_path: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize and verify the pinned SkillOpt source under ``.external/``."""

    lock = load_lock(lock_path) if lock_path else load_lock()
    root = destination or (EXTERNAL_ROOT / "skillopt")
    if not root.exists():
        if local_source is None:
            raise FileNotFoundError(
                f"missing external checkout {root}; provide --local-source or clone "
                f"{lock['skillopt']['repo']} at {lock['skillopt']['commit']}"
            )
        source = Path(local_source)
        if not source.is_dir():
            raise FileNotFoundError(f"local SkillOpt source does not exist: {source}")
        root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            source, root,
            ignore=shutil.ignore_patterns(
                ".git", "__pycache__", "*.pyc", ".pytest_cache", ".mypy_cache",
                ".ruff_cache", "ckpt", "outputs", "logs", "*.egg-info",
                "index.html", "skillopt.html", "blog", "docs", "mkdocs.yml",
                "plugins", "scripts", "skillopt_sleep", "skillopt_webui",
                "tests", "data", ".cursor-plugin", "CONTRIBUTING.md",
                "SECURITY.md", "CHANGELOG.md", ".env.example", ".gitignore",
                ".github", "skillopt-assets",
            ),
        )
    verified = verify_key_files(root, "skillopt", lock)
    runtime_tree = verify_runtime_tree(root, "skillopt", lock)
    return {
        "method": "skillopt",
        "root": str(root),
        "declared_commit": str(lock["skillopt"]["commit"]),
        "declared_version": str(lock["skillopt"].get("version") or ""),
        "verification": "runtime_tree_sha256+key_file_sha256",
        "runtime_tree": runtime_tree,
        "verified_files": verified,
    }


def _run(command: list[str], *, cwd: Path, path_prepend: str | None = None) -> None:
    environment = dict(os.environ)
    if path_prepend:
        environment["PATH"] = f"{path_prepend}{os.pathsep}{environment.get('PATH', '')}"
    completed = subprocess.run(
        [str(item) for item in command], cwd=cwd, check=False, env=environment,
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: "
            + " ".join(map(str, command))
        )


def _create_venv(venv_path: Path) -> None:
    """Create the worker venv, preferring ``uv`` when ensurepip is absent."""

    uv = shutil.which("uv")
    if uv:
        # --seed installs pip/setuptools into the uv-created venv.
        _run([uv, "venv", "--seed", "--python", "3.12", str(venv_path)], cwd=REPO_ROOT)
        return
    _run([sys.executable, "-m", "venv", str(venv_path)], cwd=REPO_ROOT)


def setup_worker_venv(
    *,
    skillopt_root: Path,
    venv_path: Path,
    lock_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create the per-method worker venv and install upstream + pinned ALFWorld."""

    load_lock(lock_path) if lock_path else load_lock()
    verify_key_files(skillopt_root, "skillopt", load_lock(lock_path) if lock_path else load_lock())
    python = venv_path / "bin" / "python"
    venv_bin = str(venv_path / "bin")
    if not python.exists():
        _create_venv(venv_path)
        _run([str(python), "-m", "pip", "install", "--upgrade", "pip"], cwd=REPO_ROOT, path_prepend=venv_bin)
    # Some ALFWorld build backends spawn a bare ``python`` executable, so the
    # venv bin directory must be on PATH during installation.
    _run(
        [str(python), "-m", "pip", "install", "-e", str(skillopt_root)],
        cwd=REPO_ROOT, path_prepend=venv_bin,
    )
    _run(
        [str(python), "-m", "pip", "install", *(_SKILLOPT_INSTALL_PACKAGES)],
        cwd=REPO_ROOT, path_prepend=venv_bin,
    )
    # The backup repo itself provides experiments.baselines.common (pure) and
    # the b3 worker modules.  atomic_skillgraph is installed as a dependency of
    # this package but is never imported by the worker.
    _run([str(python), "-m", "pip", "install", "-e", str(REPO_ROOT)], cwd=REPO_ROOT, path_prepend=venv_bin)
    _run([str(python), "-m", "pip", "install", "pytest>=7.0"], cwd=REPO_ROOT, path_prepend=venv_bin)
    return {
        "venv": str(venv_path),
        "python": str(python),
        "skillopt_root": str(skillopt_root),
        "alfworld_pinned": "0.4.2",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock", default=str(REPO_ROOT / "experiments" / "baselines" / "baseline_lock.yaml"),
        help="baseline_lock.yaml path",
    )
    parser.add_argument(
        "--local-source", default=None,
        help="path to a local SkillOpt source snapshot (used when .external/skillopt is absent)",
    )
    parser.add_argument(
        "--setup-worker-venv", action="store_true",
        help="create/refresh the SkillOpt worker venv (.venv_b3_skillopt) and install dependencies",
    )
    parser.add_argument(
        "--venv", default=str(REPO_ROOT / ".venv_b3_skillopt"),
        help="worker venv destination",
    )
    args = parser.parse_args(argv)
    result: dict[str, Any] = {}
    try:
        source = ensure_skillopt_source(
            local_source=args.local_source, lock_path=args.lock,
        )
        result["skillopt_source"] = source
        if args.setup_worker_venv:
            result["worker_venv"] = setup_worker_venv(
                skillopt_root=Path(source["root"]),
                venv_path=_path(args.venv),
                lock_path=args.lock,
            )
        result["passed"] = True
    except Exception as exc:
        result["passed"] = False
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
