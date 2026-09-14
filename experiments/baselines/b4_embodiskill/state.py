"""Durable state publication at upstream task/revision boundaries."""
import json
import os
import shutil
from pathlib import Path

from experiments.baselines.common.artifact_digest import digest_directory


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line] if Path(path).exists() else []


def copy_state(source, destination):
    if source is None:
        destination.mkdir(parents=True)
        return None
    before = digest_directory(source)
    shutil.copytree(source, destination)
    if digest_directory(destination) != before or digest_directory(source) != before:
        raise RuntimeError("Persistent snapshot copy did not preserve frozen bytes")
    return before


def publish_checkpoint(output, operation, attempt, state_dir):
    # Called only after the worker exited successfully and closed its SQLite/Chroma.
    result = read_json(attempt / "result.json")
    for path in state_dir.rglob("*"):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    receipt = dict(operation=operation, attempt=str(attempt), state=str(state_dir),
                   state_digest=digest_directory(state_dir), result=result)
    write_json(output / "checkpoints" / f"{operation}.json", receipt)
    return receipt


def load_checkpoint(output, operation):
    path = output / "checkpoints" / f"{operation}.json"
    if not path.exists():
        return None
    receipt = read_json(path)
    if digest_directory(Path(receipt["state"])) != receipt["state_digest"]:
        raise RuntimeError("Committed state changed since checkpoint")
    return receipt
