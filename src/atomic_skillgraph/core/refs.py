"""Immutable, versioned references and canonical content hashing."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any


class RefError(ValueError):
    pass


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")


def _validate_ref_part(value: str, *, name: str, pattern: re.Pattern[str]) -> None:
    if not value or value != value.strip() or not pattern.fullmatch(value):
        raise RefError(
            f"{name} must be a path-safe identifier containing only letters, "
            "digits, '.', '_', '-' (and '+' for versions)"
        )


@dataclass(frozen=True, order=True)
class SkillRef:
    logical_id: str
    version: str

    def __post_init__(self) -> None:
        _validate_ref_part(self.logical_id, name="logical_id", pattern=_SAFE_ID)
        _validate_ref_part(self.version, name="version", pattern=_SAFE_VERSION)

    def __str__(self) -> str:
        return f"skill://{self.logical_id}@{self.version}"

    def to_dict(self) -> dict[str, str]:
        return {"logical_id": self.logical_id, "version": self.version}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SkillRef":
        return cls(str(value["logical_id"]), str(value["version"]))

    @classmethod
    def parse(cls, value: str | "SkillRef") -> "SkillRef":
        if isinstance(value, cls):
            return value
        text = str(value).removeprefix("skill://")
        if "@" not in text:
            raise RefError(f"invalid SkillRef: {value!r}")
        logical_id, version = text.rsplit("@", 1)
        return cls(logical_id, version)


@dataclass(frozen=True, order=True)
class ToolRef:
    tool_id: str
    version: str

    def __post_init__(self) -> None:
        _validate_ref_part(self.tool_id, name="tool_id", pattern=_SAFE_ID)
        _validate_ref_part(self.version, name="version", pattern=_SAFE_VERSION)

    def __str__(self) -> str:
        return f"tool://{self.tool_id}@{self.version}"

    def to_dict(self) -> dict[str, str]:
        return {"tool_id": self.tool_id, "version": self.version}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ToolRef":
        return cls(str(value["tool_id"]), str(value["version"]))

    @classmethod
    def parse(cls, value: str | "ToolRef") -> "ToolRef":
        if isinstance(value, cls):
            return value
        text = str(value).removeprefix("tool://")
        if "@" not in text:
            raise RefError(f"invalid ToolRef: {value!r}")
        tool_id, version = text.rsplit("@", 1)
        return cls(tool_id, version)


def canonical_json(value: Any) -> str:
    from .serialization import to_primitive

    return json.dumps(
        to_primitive(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    )


def content_hash(value: Any, *, exclude: tuple[str, ...] = ()) -> str:
    primitive = value
    from .serialization import to_primitive

    primitive = to_primitive(primitive)
    if isinstance(primitive, dict) and exclude:
        primitive = {key: item for key, item in primitive.items() if key not in exclude}
    return hashlib.sha256(canonical_json(primitive).encode("utf-8")).hexdigest()


def artifact_hash(body: str | bytes) -> str:
    raw = body if isinstance(body, bytes) else body.replace("\r\n", "\n").strip().encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def bump_version(version: str, part: str = "patch") -> str:
    try:
        major, minor, patch = (int(piece) for piece in version.split("."))
    except Exception as exc:
        raise RefError(f"semantic version required: {version!r}") from exc
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise RefError(f"unknown version part: {part!r}")
