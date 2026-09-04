"""Canonical directory digest used by the freeze protocol.

``digest_directory`` hashes every file byte plus its path relative to the
root, so any content change, addition, or deletion of the frozen artifact is
detected.  The digest deliberately excludes machine-specific metadata
(mtimes, owners, absolute paths).
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def digest_directory(root: str | Path) -> str:
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"digest root is not a directory: {root}")
    hasher = hashlib.sha256()
    entries: list[tuple[str, str]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append((relative, content_hash))
    for relative, content_hash in entries:
        hasher.update(f"{relative}\x1f{content_hash}\n".encode("utf-8"))
    return hasher.hexdigest()


def verify_digest(root: str | Path, expected: str) -> bool:
    return digest_directory(root) == expected
