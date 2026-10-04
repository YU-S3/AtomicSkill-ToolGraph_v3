"""Durable task stages and response recovery, separate from native side effects."""
import json
import os
from pathlib import Path


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class TaskCheckpoint:
    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / 'state.json'
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {'stage': 'task_started'}

    def advance(self, stage, **values):
        self.state.update(stage=stage, **values)
        save_json(self.path, self.state)

    def native_events(self, events):
        save_json(self.root / 'native_events.json', events)

    def response(self, key):
        path = self.root / 'responses' / (key + '.json')
        return json.loads(path.read_text()) if path.exists() else None

    def save_response(self, key, value):
        path = self.root / 'responses' / (key + '.json')
        save_json(path, value)
        path.chmod(0o600)
