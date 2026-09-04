"""Bootstrap and verify external baseline checkouts from ``baseline_lock.yaml``.

Responsibilities (per the baseline design document, section 8.1):

1. materialize the pinned external source (git clone at the pinned commit, or
   a verified copy of a local snapshot);
2. verify the pinned commit / key-file hashes fail-closed;
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
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = REPO_ROOT / ".external"

_SKILLOPT_INSTALL_PACKAGES = [
    "alfworld==0.4.2",
    "gymnasium>=0.29.0",
    "omegaconf>=2.3.0",
]


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


def verify_key_files(root: Path, method: str, lock: dict[str, Any]) -> dict[str, str]:
    """Verify the pinned key-file hashes of an external snapshot, fail-closed."""

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
            ),
        )
    verified = verify_key_files(root, "skillopt", lock)
    return {
        "method": "skillopt",
        "root": str(root),
        "declared_commit": str(lock["skillopt"]["commit"]),
        "verification": "key_file_sha256",
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
