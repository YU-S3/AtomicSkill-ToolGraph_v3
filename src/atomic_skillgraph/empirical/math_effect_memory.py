"""Append-only, content-addressed Train experience. Never a Runtime input."""
import json
from pathlib import Path
from .contracts import digest
from .checkpoint import save_json


def append(root, event):
    path=Path(root)/(digest(event)+'.json')
    if path.exists():
        if json.loads(path.read_text())!=event:raise ValueError('Immutable experience changed')
    else:save_json(path,event)
    return path


def events(root):
    return [json.loads(p.read_text()) for p in sorted(Path(root).glob('*.json'))]
