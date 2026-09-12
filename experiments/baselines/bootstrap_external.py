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

_COMMON_ALFWORLD_PACKAGES = [
    "alfworld==0.4.2",
    "gymnasium==1.1.1",
    "omegaconf==2.3.0",
]

_GEPA_INSTALL_PACKAGES = [
    *_COMMON_ALFWORLD_PACKAGES,
    # GEPA imports the pinned SkillOpt source from PYTHONPATH.  Its selected
    # OpenAI-compatible backend imports the SDK at module load time even
    # though SkillOpt itself is deliberately not installed as a distribution.
    "openai==1.75.0",
]

# SkillGen's published requirements file contains machine-local conda URLs and
# optional visualization packages.  These are the import/runtime dependencies
# of the frozen ALFWorld sampling, graph/TD, embedding, and retrieval path.
_SKILLGEN_INSTALL_PACKAGES = [
    *_COMMON_ALFWORLD_PACKAGES,
    # ALFWorld declares lower bounds for its TextWorld stack.  On Python 3.9,
    # an unconstrained 2026 resolution selects spaCy/thinc releases that no
    # longer publish a compatible build.  Keep the exact upstream SkillGen
    # environment identities for this method-defining dependency chain.
    "textworld==1.6.2",
    "fast_downward_textworld==20.6.3",
    "jericho==3.3.0",
    "spacy==3.4.4",
    "thinc==8.1.12",
    "pydantic==1.10.8",
    "blis==0.7.11",
    "catalogue==2.0.10",
    "confection==0.1.5",
    "cymem==2.0.11",
    "hashids==1.3.1",
    "langcodes==3.5.0",
    "language_data==1.3.0",
    "mementos==1.3.1",
    "more-itertools==10.6.0",
    "murmurhash==1.0.12",
    "pathy==0.11.0",
    "preshed==3.0.9",
    "prompt_toolkit==3.0.51",
    "smart-open==6.4.0",
    "srsly==2.5.1",
    "TatSu==5.8.3",
    "typer==0.7.0",
    "wasabi==0.10.1",
    "gym==0.26.0",
    "jsonlines==4.0.0",
    "matplotlib==3.8.4",
    "networkx==3.2.1",
    "numpy==1.22.4",
    "openai==1.75.0",
    "pandas==1.4.2",
    "python-dotenv==1.1.0",
    "PyYAML==6.0",
    "requests==2.27.1",
    "scikit-learn==1.1.1",
    "sentence-transformers==4.1.0",
    "tqdm==4.64.0",
    "transformers==4.51.3",
]
_SKILLGEN_TORCH_PACKAGES = ["torch==2.6.0"]
_PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

_WORKER_EXPECTED_DISTRIBUTIONS = {
    "skillgen": {
        "alfworld": "0.4.2",
        "blis": "0.7.11",
        "catalogue": "2.0.10",
        "confection": "0.1.5",
        "cymem": "2.0.11",
        "fast_downward_textworld": "20.6.3",
        "gym": "0.26.0",
        "gymnasium": "1.1.1",
        "hashids": "1.3.1",
        "jericho": "3.3.0",
        "jsonlines": "4.0.0",
        "langcodes": "3.5.0",
        "language_data": "1.3.0",
        "matplotlib": "3.8.4",
        "mementos": "1.3.1",
        "more-itertools": "10.6.0",
        "murmurhash": "1.0.12",
        "networkx": "3.2.1",
        "numpy": "1.22.4",
        "omegaconf": "2.3.0",
        "openai": "1.75.0",
        "pandas": "1.4.2",
        "pathy": "0.11.0",
        "preshed": "3.0.9",
        "prompt_toolkit": "3.0.51",
        "pydantic": "1.10.8",
        "python-dotenv": "1.1.0",
        "PyYAML": "6.0",
        "requests": "2.27.1",
        "scikit-learn": "1.1.1",
        "sentence-transformers": "4.1.0",
        "smart-open": "6.4.0",
        "spacy": "3.4.4",
        "srsly": "2.5.1",
        "TatSu": "5.8.3",
        "textworld": "1.6.2",
        "thinc": "8.1.12",
        "torch": "2.6.0",
        "tqdm": "4.64.0",
        "transformers": "4.51.3",
        "typer": "0.7.0",
        "wasabi": "0.10.1",
    },
    "gepa": {
        "alfworld": "0.4.2",
        "gepa": "0.1.3",
        "gymnasium": "1.1.1",
        "omegaconf": "2.3.0",
        "openai": "1.75.0",
    },
}
_WORKER_FORBIDDEN_DISTRIBUTIONS = {
    "skillgen": ("atomic-skillgraph", "skillopt"),
    "gepa": ("atomic-skillgraph", "skillopt"),
}
_WORKER_FORBIDDEN_MODULES = {
    "skillgen": ("atomic_skillgraph", "skillopt"),
    "gepa": ("atomic_skillgraph", "skillopt"),
}


def worker_expected_distributions(method: str) -> dict[str, str]:
    """Return a copy of the frozen distribution authority for one worker."""

    normalized = str(method).strip().lower()
    aliases = {"b4_skillgen_s": "skillgen", "b5_gepa": "gepa"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in _WORKER_EXPECTED_DISTRIBUTIONS:
        raise ValueError(f"unsupported isolated worker method: {method!r}")
    return dict(_WORKER_EXPECTED_DISTRIBUTIONS[normalized])

_RUNTIME_TREE_ALGORITHM = "sha256-path-content-v1"
_CANONICAL_RUNTIME_TREE_ALGORITHM = "sha256-path-canonical-content-v2"
_SUPPORTED_RUNTIME_TREE_ALGORITHMS = frozenset({
    _RUNTIME_TREE_ALGORITHM,
    _CANONICAL_RUNTIME_TREE_ALGORITHM,
})
_RUNTIME_TREE_IGNORED_DIRS = frozenset({
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
})
_RUNTIME_TREE_IGNORED_SUFFIXES = frozenset({".pyc", ".pyo"})
_CANONICAL_TEXT_SUFFIXES = frozenset({
    ".cfg", ".ini", ".ipynb", ".json", ".jsonl", ".md", ".py", ".rst",
    ".lock", ".sh", ".toml", ".txt", ".typed", ".yaml", ".yml",
})
_CANONICAL_TEXT_NAMES = frozenset({"LICENSE", "NOTICE"})


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


def _source_bytes(path: Path, *, algorithm: str) -> bytes:
    """Read source bytes using the line-ending policy declared by the lock."""

    content = path.read_bytes()
    if algorithm == _CANONICAL_RUNTIME_TREE_ALGORITHM and (
        path.suffix.casefold() in _CANONICAL_TEXT_SUFFIXES
        or path.name in _CANONICAL_TEXT_NAMES
    ):
        return content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return content


def source_file_sha256(path: str | Path, *, algorithm: str) -> str:
    """Hash one locked source file with the runtime-tree canonicalization."""

    if algorithm not in _SUPPORTED_RUNTIME_TREE_ALGORITHMS:
        raise ValueError(f"unsupported runtime_tree algorithm: {algorithm!r}")
    return hashlib.sha256(
        _source_bytes(Path(path), algorithm=algorithm)
    ).hexdigest()


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
    if algorithm not in _SUPPORTED_RUNTIME_TREE_ALGORITHMS:
        raise ValueError(
            "unsupported runtime_tree algorithm: "
            f"{algorithm!r}; expected one of "
            f"{sorted(_SUPPORTED_RUNTIME_TREE_ALGORITHMS)!r}"
        )
    entries = _runtime_tree_files(root, spec)
    digest = hashlib.sha256()
    digest.update((algorithm + "\0").encode("ascii"))
    for relative, path in entries:
        relative_bytes = relative.encode("utf-8")
        content = _source_bytes(path, algorithm=algorithm)
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
    runtime_spec = dict((lock.get(method) or {}).get("runtime_tree") or {})
    algorithm = str(runtime_spec.get("algorithm") or "")
    if algorithm not in _SUPPORTED_RUNTIME_TREE_ALGORITHMS:
        raise ValueError(f"unsupported runtime_tree algorithm: {algorithm!r}")
    mismatches: list[str] = []
    for relative, wanted in sorted(expected.items()):
        path = root / relative
        if not path.is_file():
            mismatches.append(f"{relative}: missing")
            continue
        actual = source_file_sha256(path, algorithm=algorithm)
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


def _verify_git_checkout(
    root: Path,
    *,
    expected_commit: str,
    expected_tag: str = "",
) -> dict[str, Any]:
    """Verify commit and cleanliness when the materialized source retains Git."""

    if not (root / ".git").exists():
        raise RuntimeError(
            f"external source must retain Git metadata to prove commit {expected_commit}: "
            f"{root}"
        )
    head = subprocess.run(
        ["git", "-c", "core.autocrlf=false", "rev-parse", "HEAD"],
        cwd=root, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stdout.strip()
    status = subprocess.run(
        [
            # Git's index is the semantic source authority.  Accept either LF
            # or CRLF worktree materialization; canonical source hashes below
            # still reject every non-line-ending content change.
            "git", "-c", "core.autocrlf=true", "status", "--porcelain",
            "--untracked-files=all",
        ], cwd=root,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stdout.strip()
    if head != expected_commit:
        raise RuntimeError(
            f"external checkout HEAD mismatch: expected {expected_commit}, got {head}"
        )
    if status:
        raise RuntimeError("external checkout is not clean")
    tag_commit: str | None = None
    if expected_tag:
        tag_ref = f"refs/tags/{expected_tag}^{{commit}}"
        completed = subprocess.run(
            ["git", "-c", "core.autocrlf=false", "rev-parse", "--verify", tag_ref],
            cwd=root, check=False,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        if completed.returncode:
            raise RuntimeError(
                f"external checkout has no verifiable tag {expected_tag!r}"
            )
        tag_commit = completed.stdout.strip()
        if tag_commit != expected_commit:
            raise RuntimeError(
                f"external tag {expected_tag!r} points at {tag_commit}, "
                f"expected {expected_commit}"
            )
    return {
        "git_metadata": True,
        "head": head,
        "clean": True,
        "tag": expected_tag or None,
        "tag_commit": tag_commit,
    }


def ensure_pinned_source(
    method: str,
    *,
    local_source: str | Path | None = None,
    destination: Path | None = None,
    lock_path: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize and fail-closed verify one Git-pinned external method."""

    normalized = str(method).strip().lower()
    if normalized == "skillopt":
        return ensure_skillopt_source(
            local_source=local_source,
            destination=destination,
            lock_path=lock_path,
        )
    lock = load_lock(lock_path) if lock_path else load_lock()
    entry = dict(lock.get(normalized) or {})
    if not entry:
        raise ValueError(f"baseline_lock.yaml has no source entry for {normalized}")
    expected_commit = str(entry.get("commit") or "").strip()
    expected_tag = str(entry.get("tag") or "").strip()
    repo = str(entry.get("repo") or "").strip()
    if len(expected_commit) != 40 or not repo:
        raise ValueError(f"source lock for {normalized} has no exact repo/commit")
    root = destination or (EXTERNAL_ROOT / normalized)
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        if local_source is not None:
            source = Path(local_source).expanduser().resolve()
            if not source.is_dir():
                raise FileNotFoundError(
                    f"local {normalized} source does not exist: {source}"
                )
            _verify_git_checkout(
                source,
                expected_commit=expected_commit,
                expected_tag=expected_tag,
            )
            _run(
                [
                    "git", "-c", "core.autocrlf=false", "clone", "--no-local",
                    str(source), str(root),
                ],
                cwd=REPO_ROOT,
            )
        else:
            _run(
                ["git", "-c", "core.autocrlf=false", "clone", repo, str(root)],
                cwd=REPO_ROOT,
            )
        _run(
            ["git", "-c", "core.autocrlf=false", "checkout", "--detach", expected_commit],
            cwd=root,
        )
    git_identity = _verify_git_checkout(
        root,
        expected_commit=expected_commit,
        expected_tag=expected_tag,
    )
    verified = verify_key_files(root, normalized, lock)
    runtime_tree = verify_runtime_tree(root, normalized, lock)
    return {
        "method": normalized,
        "root": str(root),
        "repo": repo,
        "declared_commit": expected_commit,
        "declared_tag": str(entry.get("tag") or ""),
        "declared_version": str(entry.get("version") or ""),
        "verification": str(entry.get("local_snapshot_verification") or ""),
        "git_identity": git_identity,
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


def _find_uv() -> str | None:
    """Find an executable uv without relying on login-shell PATH setup."""

    configured = os.environ.get("UV_BIN", "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            raise ValueError("UV_BIN must be an absolute executable path")
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise RuntimeError(f"UV_BIN is not an executable file: {candidate}")
        return str(candidate)

    discovered = shutil.which("uv")
    if discovered:
        return discovered
    executable_name = "uv.exe" if os.name == "nt" else "uv"
    for candidate in (
        Path.home() / ".local" / "bin" / executable_name,
        Path.home() / ".cargo" / "bin" / executable_name,
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _create_venv(venv_path: Path, *, python_version: str = "3.12") -> None:
    """Create the worker venv, preferring ``uv`` when ensurepip is absent."""

    uv = _find_uv()
    if uv:
        # --seed installs pip/setuptools into the uv-created venv.
        _run(
            [uv, "venv", "--seed", "--python", python_version, str(venv_path)],
            cwd=REPO_ROOT,
        )
        return
    current = f"{sys.version_info.major}.{sys.version_info.minor}"
    if current != python_version:
        raise RuntimeError(
            f"creating this worker requires Python {python_version}; install uv "
            f"or invoke bootstrap with that interpreter (current={current})"
        )
    _run([sys.executable, "-m", "venv", str(venv_path)], cwd=REPO_ROOT)


def worker_venv_python(
    venv_path: str | Path,
    *,
    host_os: str | None = None,
) -> Path:
    """Return the lexical interpreter path for the current host platform."""

    root = Path(venv_path)
    if (host_os or os.name) == "nt":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def verify_worker_python(
    python: str | Path,
    *,
    expected_version: str,
) -> dict[str, Any]:
    """Verify an existing worker interpreter before installing into its venv."""

    executable = Path(python)
    if not executable.is_file():
        raise FileNotFoundError(f"worker venv Python is missing: {executable}")
    probe = (
        "import json,sys; print(json.dumps({"
        "'version': f'{sys.version_info.major}.{sys.version_info.minor}',"
        "'version_info': list(sys.version_info[:3]),"
        "'executable': sys.executable,"
        "'prefix': sys.prefix,"
        "'base_prefix': sys.base_prefix,"
        "'venv_active': sys.prefix != sys.base_prefix"
        "}, sort_keys=True))"
    )
    completed = subprocess.run(
        [str(executable), "-c", probe],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"worker Python probe failed for {executable}: "
            f"{completed.stderr.strip() or 'unknown error'}"
        )
    try:
        receipt = json.loads(completed.stdout.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"worker Python probe returned invalid JSON for {executable}"
        ) from exc
    if str(receipt.get("version")) != expected_version:
        raise RuntimeError(
            f"worker Python version mismatch: expected {expected_version}, "
            f"got {receipt.get('version')} ({executable})"
        )
    if receipt.get("venv_active") is not True:
        raise RuntimeError(f"worker Python is not a virtual environment: {executable}")
    receipt["requested_executable"] = str(executable)
    receipt["expected_version"] = expected_version
    return receipt


def _prepare_worker_python(
    venv_path: Path,
    *,
    python_version: str,
) -> tuple[Path, dict[str, Any]]:
    python = worker_venv_python(venv_path)
    if not python.exists():
        if venv_path.exists() and any(venv_path.iterdir()):
            raise RuntimeError(
                f"worker venv exists but has no interpreter for this host at {python}; "
                "use a separate Windows/WSL venv path or remove the incompatible venv"
            )
        _create_venv(venv_path, python_version=python_version)
    if not (Path(venv_path) / "pyvenv.cfg").is_file():
        raise RuntimeError(f"worker venv metadata is missing: {venv_path}")
    return python, verify_worker_python(
        python,
        expected_version=python_version,
    )


def _ensure_pip(python: Path, *, path_prepend: str) -> None:
    completed = subprocess.run(
        [str(python), "-m", "pip", "--version"],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode:
        _run(
            [str(python), "-m", "ensurepip", "--upgrade"],
            cwd=REPO_ROOT,
            path_prepend=path_prepend,
        )


def verify_worker_environment(
    python: str | Path,
    *,
    expected_versions: dict[str, str],
    forbidden_distributions: tuple[str, ...] = (),
    forbidden_modules: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return and validate the installed distribution receipt for a worker."""

    names = sorted(str(name) for name in expected_versions)
    probe = (
        "import json; from importlib.metadata import PackageNotFoundError, version; "
        f"names={json.dumps(names)}; values={{}}; "
        "\nfor name in names:\n"
        "  try: values[name]=version(name)\n"
        "  except PackageNotFoundError: values[name]=None\n"
        "print(json.dumps(values, sort_keys=True))"
    )
    completed = subprocess.run(
        [str(python), "-c", probe],
        cwd=Path(python).parent,
        check=False,
        env=_isolated_probe_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"worker distribution probe failed: "
            f"{completed.stderr.strip() or 'unknown error'}"
        )
    try:
        installed = json.loads(completed.stdout.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError("worker distribution probe returned invalid JSON") from exc
    mismatches = []
    for name, expected in sorted(expected_versions.items()):
        actual = installed.get(name)
        # The official PyTorch CPU index appends a local ``+cpu`` marker.
        compatible = (
            actual == expected
            or (name == "torch" and str(actual).split("+", 1)[0] == expected)
        )
        if not compatible:
            mismatches.append(f"{name}: expected {expected}, got {actual}")
    if mismatches:
        raise RuntimeError(
            "worker distribution identity mismatch: " + "; ".join(mismatches)
        )
    isolation = verify_worker_isolation(
        python,
        forbidden_distributions=forbidden_distributions,
        forbidden_modules=forbidden_modules,
    )
    return {
        "python": str(python),
        "distributions": {name: installed[name] for name in names},
        "isolation": isolation,
    }


def _isolated_probe_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def verify_worker_isolation(
    python: str | Path,
    *,
    forbidden_distributions: tuple[str, ...],
    forbidden_modules: tuple[str, ...],
) -> dict[str, Any]:
    """Reject worker venvs contaminated by controller/method installations."""

    distributions = sorted(set(str(name) for name in forbidden_distributions))
    modules = sorted(set(str(name) for name in forbidden_modules))
    probe = (
        "import json; from importlib.metadata import PackageNotFoundError, version; "
        "from importlib.util import find_spec; "
        f"distributions={json.dumps(distributions)}; modules={json.dumps(modules)}; "
        "installed={}; origins={}; "
        "\nfor name in distributions:\n"
        "  try: installed[name]=version(name)\n"
        "  except PackageNotFoundError: installed[name]=None\n"
        "\nfor name in modules:\n"
        "  try: spec=find_spec(name)\n"
        "  except (ImportError, ModuleNotFoundError, ValueError): spec=None\n"
        "  origins[name]=(getattr(spec, 'origin', None) if spec is not None else None)\n"
        "print(json.dumps({'distributions': installed, 'modules': origins}, sort_keys=True))"
    )
    completed = subprocess.run(
        [str(python), "-c", probe],
        cwd=Path(python).parent,
        check=False,
        env=_isolated_probe_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            "worker isolation probe failed: "
            f"{completed.stderr.strip() or 'unknown error'}"
        )
    try:
        receipt = json.loads(completed.stdout.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError("worker isolation probe returned invalid JSON") from exc
    installed = {
        name: value
        for name, value in dict(receipt.get("distributions") or {}).items()
        if value is not None
    }
    origins = {
        name: value
        for name, value in dict(receipt.get("modules") or {}).items()
        if value is not None
    }
    if installed or origins:
        details = []
        if installed:
            details.append(f"distributions={installed}")
        if origins:
            details.append(f"module_origins={origins}")
        raise RuntimeError(
            "worker venv contains forbidden controller/method installations ("
            + ", ".join(details)
            + "); recreate this worker venv instead of reusing it"
        )
    return {
        "forbidden_distributions_absent": distributions,
        "forbidden_modules_absent": modules,
    }


def setup_worker_venv(
    *,
    skillopt_root: Path,
    venv_path: Path,
    lock_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create the per-method worker venv and install upstream + pinned ALFWorld."""

    load_lock(lock_path) if lock_path else load_lock()
    verify_key_files(skillopt_root, "skillopt", load_lock(lock_path) if lock_path else load_lock())
    python, python_receipt = _prepare_worker_python(
        venv_path,
        python_version="3.12",
    )
    venv_bin = str(python.parent)
    _ensure_pip(python, path_prepend=venv_bin)
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
        "python_receipt": python_receipt,
        "skillopt_root": str(skillopt_root),
        "alfworld_pinned": "0.4.2",
    }


def setup_method_worker_venv(
    method: str,
    *,
    source_root: Path,
    venv_path: Path,
    lock_path: str | Path | None = None,
    skillopt_root: Path | None = None,
) -> dict[str, Any]:
    """Create an isolated B4/B5 worker environment without patching upstream."""

    normalized = str(method).strip().lower()
    if normalized == "skillopt":
        return setup_worker_venv(
            skillopt_root=source_root,
            venv_path=venv_path,
            lock_path=lock_path,
        )
    if normalized not in {"skillgen", "gepa"}:
        raise ValueError(f"unsupported worker method: {normalized}")
    lock = load_lock(lock_path) if lock_path else load_lock()
    verify_key_files(source_root, normalized, lock)
    python_version = "3.9" if normalized == "skillgen" else "3.12"
    python, python_receipt = _prepare_worker_python(
        venv_path,
        python_version=python_version,
    )
    pre_install_isolation_receipt = verify_worker_isolation(
        python,
        forbidden_distributions=_WORKER_FORBIDDEN_DISTRIBUTIONS[normalized],
        forbidden_modules=_WORKER_FORBIDDEN_MODULES[normalized],
    )
    venv_bin = str(python.parent)
    _ensure_pip(python, path_prepend=venv_bin)
    if normalized == "gepa":
        if skillopt_root is None:
            raise ValueError("GEPA worker setup requires the shared SkillOpt source")
        verify_key_files(skillopt_root, "skillopt", lock)
        _run(
            [str(python), "-m", "pip", "install", "-e", str(source_root)],
            cwd=REPO_ROOT,
            path_prepend=venv_bin,
        )
        packages = _GEPA_INSTALL_PACKAGES
    else:
        packages = _SKILLGEN_INSTALL_PACKAGES
        _run(
            [
                str(python), "-m", "pip", "install", "--index-url",
                _PYTORCH_CPU_INDEX, *_SKILLGEN_TORCH_PACKAGES,
            ],
            cwd=REPO_ROOT,
            path_prepend=venv_bin,
        )
    _run(
        [str(python), "-m", "pip", "install", *packages],
        cwd=REPO_ROOT,
        path_prepend=venv_bin,
    )
    environment_receipt = verify_worker_environment(
        python,
        expected_versions=_WORKER_EXPECTED_DISTRIBUTIONS[normalized],
        forbidden_distributions=_WORKER_FORBIDDEN_DISTRIBUTIONS[normalized],
        forbidden_modules=_WORKER_FORBIDDEN_MODULES[normalized],
    )
    return {
        "method": normalized,
        "venv": str(venv_path),
        "python": str(python),
        "python_receipt": python_receipt,
        "pre_install_isolation_receipt": pre_install_isolation_receipt,
        "environment_receipt": environment_receipt,
        "source_root": str(source_root),
        "skillopt_root": str(skillopt_root) if skillopt_root is not None else None,
        "python_requested": python_version,
        "alfworld_pinned": "0.4.2",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock", default=str(REPO_ROOT / "experiments" / "baselines" / "baseline_lock.yaml"),
        help="baseline_lock.yaml path",
    )
    parser.add_argument(
        "--method", default="skillopt", choices=["skillopt", "skillgen", "gepa"],
        help="external method to materialize and verify",
    )
    parser.add_argument(
        "--local-source", default=None,
        help="optional local source snapshot (used only when the destination is absent)",
    )
    parser.add_argument(
        "--skillopt-local-source", default=None,
        help="optional shared SkillOpt snapshot required by a new GEPA worker venv",
    )
    parser.add_argument(
        "--setup-worker-venv", action="store_true",
        help="create/refresh the selected method's isolated worker venv",
    )
    parser.add_argument(
        "--venv", default=None,
        help="worker venv destination",
    )
    args = parser.parse_args(argv)
    result: dict[str, Any] = {}
    try:
        source = ensure_pinned_source(
            args.method,
            local_source=args.local_source,
            lock_path=args.lock,
        )
        result[f"{args.method}_source"] = source
        if args.setup_worker_venv:
            venv_defaults = {
                "skillopt": REPO_ROOT / ".venv_b3_skillopt",
                "skillgen": REPO_ROOT / ".venv_b4_skillgen",
                "gepa": REPO_ROOT / ".venv_b5_gepa",
            }
            shared_skillopt: Path | None = None
            if args.method == "gepa":
                skillopt = ensure_skillopt_source(
                    local_source=args.skillopt_local_source,
                    lock_path=args.lock,
                )
                result["skillopt_source"] = skillopt
                shared_skillopt = Path(skillopt["root"])
            result["worker_venv"] = setup_method_worker_venv(
                args.method,
                source_root=Path(source["root"]),
                venv_path=_path(args.venv or venv_defaults[args.method]),
                lock_path=args.lock,
                skillopt_root=shared_skillopt,
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
