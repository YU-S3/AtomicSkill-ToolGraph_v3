"""Authorized text selection and bounded grep, with decoded Unicode offsets."""
from pathlib import Path, PurePosixPath, PureWindowsPath
import re

ALL_PATHS = object()
GREP_SCHEMA = {'type': 'object', 'properties': {'pattern': {'type': 'string'},
    'paths': {'type': 'array', 'items': {'type': 'string'}}}, 'required': ['pattern'], 'additionalProperties': False}


def canonical_scope(paths=ALL_PATHS):
    if paths is ALL_PATHS: return None
    if not isinstance(paths, list): raise ValueError('paths must be an array; omit it for the entire corpus')
    normalized = set()
    for name in paths:
        if not isinstance(name, str) or not name.strip(): raise ValueError('paths contains an empty/non-string element')
        path = PurePosixPath(name.replace('\\', '/'))
        if path.is_absolute() or PureWindowsPath(name).drive or '..' in path.parts:
            raise ValueError('Corpus path is not authorized')
        if path.suffix != '.txt': raise ValueError('Corpus path must name a text file')
        normalized.add(path.as_posix())
    return sorted(normalized)


def select_paths(corpus, paths=ALL_PATHS):
    root = Path(corpus).resolve(strict=True)
    if not root.is_dir(): raise RuntimeError('Authorized corpus root is unavailable')
    scope = canonical_scope(paths)
    names = scope if scope is not None else sorted(p.relative_to(root).as_posix() for p in root.rglob('*.txt'))
    selected, parents = {}, {}
    for name in names:
        candidate = root / name
        # Resolve each parent once per selection; individual symlinks still
        # require resolution before the authorization check.
        if candidate.parent not in parents:
            parents[candidate.parent] = candidate.parent.resolve()
        path = parents[candidate.parent] / candidate.name
        if path.is_symlink(): path = path.resolve()
        if not path.is_relative_to(root): raise ValueError('Corpus path is not authorized')
        if not path.is_file(): raise ValueError('Corpus file is missing or not a file')
        if path.suffix != '.txt': raise ValueError('Corpus path must name a text file')
        selected[path.relative_to(root).as_posix()] = path
    return [(name, selected[name]) for name in sorted(selected)]


def grep(corpus, pattern, paths=ALL_PATHS):
    expression = re.compile(pattern, re.I)
    selected = select_paths(corpus, paths)
    hits = []
    for name, path in selected:
        offset = 0
        for index, line in enumerate(path.read_text(encoding='utf-8').splitlines(keepends=True)):
            if expression.search(line):
                hits.append({'path': name, 'line': index + 1, 'offset': offset, 'text': line.rstrip('\n')[:1000]})
                if len(hits) == 40:
                    return hits, {'execution_paths': [n for n, _ in selected], 'truncated': True}
            offset += len(line)
    return hits, {'execution_paths': [n for n, _ in selected], 'truncated': False}
