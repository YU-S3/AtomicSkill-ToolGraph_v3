"""Canonical dataclass serialization and crash-safe file persistence."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

T = TypeVar("T")


def to_primitive(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: to_primitive(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_primitive(item) for item in value]
    return value


def dataclass_from_dict(cls: type[T], payload: dict[str, Any]) -> T:
    """Best-effort typed dataclass loader used only at repository boundaries."""
    hints = get_type_hints(cls)
    values: dict[str, Any] = {}
    for item in fields(cls):
        if item.name not in payload:
            continue
        values[item.name] = _convert(hints.get(item.name, Any), payload[item.name])
    return cls(**values)


def _convert(annotation: Any, value: Any) -> Any:
    if value is None or annotation is Any:
        return value
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is list:
        return [_convert(args[0], item) for item in value]
    if origin is tuple:
        subtype = args[0] if args else Any
        return tuple(_convert(subtype, item) for item in value)
    if origin is dict:
        subtype = args[1] if len(args) > 1 else Any
        return {key: _convert(subtype, item) for key, item in value.items()}
    if origin is not None and str(origin).endswith("Union"):
        for subtype in args:
            try:
                return _convert(subtype, value)
            except Exception:
                continue
        return value
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation(value)
    if isinstance(annotation, type) and is_dataclass(annotation) and isinstance(value, dict):
        return dataclass_from_dict(annotation, value)
    return value


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    encoded = json.dumps(to_primitive(payload), ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    for attempt in range(8):
        try:
            os.replace(temporary, target)
            return target
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.02 * (attempt + 1))
    return target


def atomic_create_json(path: str | Path, payload: Any) -> Path:
    """Atomically create a JSON file and never replace an existing identity."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    encoded = json.dumps(
        to_primitive(payload), ensure_ascii=False, indent=2, sort_keys=True,
    ).encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # A same-directory hard link publishes the fully-written inode and is
        # create-only: an existing target raises FileExistsError atomically.
        os.link(temporary, target)
        return target
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
