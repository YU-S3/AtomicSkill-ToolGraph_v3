"""Provider-independent native tool-call protocol.

The runtime consumes only :class:`NativeToolCall` arguments or JSON that was
requested with a structured-output schema.  Visible assistant prose and
provider reasoning fields are deliberately outside the action protocol.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_CALL_ID = re.compile(r"^[!-~]{1,256}$")


class SchemaValidationError(ValueError):
    """Raised when a native tool argument or structured output is invalid."""


def parse_json_strict(text: str) -> Any:
    """Parse exactly one RFC JSON value, rejecting NaN and duplicate keys."""
    if not isinstance(text, str):
        raise TypeError("JSON input must be a string")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number is forbidden: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    return json.loads(
        text,
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )


@dataclass(frozen=True)
class NativeToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]

    def __post_init__(self) -> None:
        if not _TOOL_NAME.fullmatch(self.name):
            raise ValueError(f"invalid native tool name: {self.name!r}")
        if not isinstance(self.description, str):
            raise TypeError("native tool description must be a string")
        if not isinstance(self.input_schema, dict):
            raise TypeError("native tool input_schema must be a mapping")
        schema_type = self.input_schema.get("type", "object")
        if schema_type != "object":
            raise ValueError("native tool input_schema must describe an object")

    def to_openai(self) -> dict[str, Any]:
        """Return the OpenAI-compatible native function declaration."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass(frozen=True)
class NativeToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.call_id, str) or not _CALL_ID.fullmatch(self.call_id):
            raise ValueError(f"invalid native tool call_id: {self.call_id!r}")
        if not _TOOL_NAME.fullmatch(self.name):
            raise ValueError(f"invalid native tool name: {self.name!r}")
        if not isinstance(self.arguments, dict):
            raise TypeError("native tool arguments must be a JSON object")


@dataclass
class AgentTurn:
    content: str
    tool_calls: list[NativeToolCall]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: int | None
    latency_ms: float
    provider_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("AgentTurn.content must be a string")
        self.tool_calls = [
            value if isinstance(value, NativeToolCall) else NativeToolCall(**value)
            for value in self.tool_calls
        ]
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.reasoning_tokens is not None:
            if (
                isinstance(self.reasoning_tokens, bool)
                or not isinstance(self.reasoning_tokens, int)
                or self.reasoning_tokens < 0
            ):
                raise ValueError("reasoning_tokens must be a non-negative integer or None")
        if not isinstance(self.latency_ms, (int, float)) or self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        if not isinstance(self.provider_metadata, dict):
            raise TypeError("provider_metadata must be a mapping")


AgentMessage = dict[str, Any]


@runtime_checkable
class AgentProvider(Protocol):
    """Provider adapter consumed by a client-managed Agent session."""

    def complete(
        self,
        messages: list[AgentMessage],
        *,
        tools: list[NativeToolSpec] | None = None,
        structured_output_schema: dict[str, Any] | None = None,
    ) -> AgentTurn: ...

    def snapshot(self) -> dict[str, Any]: ...


@runtime_checkable
class AgentSession(Protocol):
    @property
    def session_id(self) -> str: ...

    def next_turn(
        self,
        user_input: str | None,
        *,
        tools: list[NativeToolSpec] | None = None,
        structured_output_schema: dict[str, Any] | None = None,
    ) -> AgentTurn: ...

    def submit_tool_result(
        self,
        call_id: str,
        result: dict[str, Any],
        *,
        tools: list[NativeToolSpec] | None = None,
    ) -> AgentTurn: ...

    def snapshot(self) -> dict[str, Any]: ...


def validate_schema_instance(instance: Any, schema: dict[str, Any], *, path: str = "$") -> None:
    """Validate the JSON-Schema subset used by Agent tool contracts.

    Providers are asked to enforce the same schema, but the client validates
    again before any action can cross the runtime boundary.  Common composition,
    object, array, scalar, enum, and range keywords are supported.
    """
    if not isinstance(schema, dict):
        raise SchemaValidationError(f"{path}: schema must be an object")

    if "allOf" in schema:
        branches = _schema_list(schema["allOf"], path, "allOf")
        for branch in branches:
            validate_schema_instance(instance, branch, path=path)
    if "anyOf" in schema:
        branches = _schema_list(schema["anyOf"], path, "anyOf")
        if not any(_schema_matches(instance, branch, path) for branch in branches):
            raise SchemaValidationError(f"{path}: value does not match anyOf")
    if "oneOf" in schema:
        branches = _schema_list(schema["oneOf"], path, "oneOf")
        if sum(_schema_matches(instance, branch, path) for branch in branches) != 1:
            raise SchemaValidationError(f"{path}: value must match exactly one oneOf branch")
    if "not" in schema and _schema_matches(instance, _schema(schema["not"], path, "not"), path):
        raise SchemaValidationError(f"{path}: value matches forbidden schema")

    if "const" in schema and instance != schema["const"]:
        raise SchemaValidationError(f"{path}: value does not match const")
    if "enum" in schema:
        choices = schema["enum"]
        if not isinstance(choices, list) or instance not in choices:
            raise SchemaValidationError(f"{path}: value is not in enum")

    declared_type = schema.get("type")
    if declared_type is not None:
        allowed_types = [declared_type] if isinstance(declared_type, str) else declared_type
        if not isinstance(allowed_types, list) or not allowed_types:
            raise SchemaValidationError(f"{path}: schema type must be a string or non-empty list")
        if not any(_matches_type(instance, item) for item in allowed_types):
            raise SchemaValidationError(
                f"{path}: expected {' | '.join(str(item) for item in allowed_types)}, "
                f"got {_json_type(instance)}"
            )

    if isinstance(instance, dict):
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise SchemaValidationError(f"{path}: schema properties must be an object")
        required = schema.get("required", [])
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            raise SchemaValidationError(f"{path}: schema required must be a string list")
        missing = [name for name in required if name not in instance]
        if missing:
            raise SchemaValidationError(f"{path}: missing required properties {missing!r}")
        additional = schema.get("additionalProperties", True)
        for name, value in instance.items():
            child_path = f"{path}.{name}"
            if name in properties:
                validate_schema_instance(value, _schema(properties[name], path, name), path=child_path)
            elif additional is False:
                raise SchemaValidationError(f"{child_path}: additional property is not allowed")
            elif isinstance(additional, dict):
                validate_schema_instance(value, additional, path=child_path)
        _check_size(instance, schema, path, "Properties")

    if isinstance(instance, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for index, value in enumerate(instance):
                validate_schema_instance(value, items, path=f"{path}[{index}]")
        elif items is not None:
            raise SchemaValidationError(f"{path}: tuple-style or invalid items schema is unsupported")
        _check_size(instance, schema, path, "Items")
        if schema.get("uniqueItems") is True:
            for index, value in enumerate(instance):
                if value in instance[:index]:
                    raise SchemaValidationError(f"{path}: array items must be unique")

    if isinstance(instance, str):
        _check_size(instance, schema, path, "Length")
        pattern = schema.get("pattern")
        if pattern is not None:
            if not isinstance(pattern, str):
                raise SchemaValidationError(f"{path}: schema pattern must be a string")
            try:
                matches = re.search(pattern, instance) is not None
            except re.error as exc:
                raise SchemaValidationError(f"{path}: invalid schema pattern") from exc
            if not matches:
                raise SchemaValidationError(f"{path}: string does not match pattern")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        _check_numeric(instance, schema, path)


def _schema(value: Any, path: str, keyword: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SchemaValidationError(f"{path}: {keyword} must contain a schema object")
    return value


def _schema_list(value: Any, path: str, keyword: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise SchemaValidationError(f"{path}: {keyword} must be a non-empty schema list")
    return [_schema(item, path, keyword) for item in value]


def _schema_matches(instance: Any, schema: dict[str, Any], path: str) -> bool:
    try:
        validate_schema_instance(instance, schema, path=path)
    except SchemaValidationError:
        return False
    return True


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return type(value).__name__


def _matches_type(value: Any, expected: Any) -> bool:
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    raise SchemaValidationError(f"unsupported JSON schema type: {expected!r}")


def _check_size(value: Any, schema: dict[str, Any], path: str, suffix: str) -> None:
    minimum = schema.get(f"min{suffix}")
    maximum = schema.get(f"max{suffix}")
    if minimum is not None and len(value) < int(minimum):
        raise SchemaValidationError(f"{path}: size is below min{suffix}")
    if maximum is not None and len(value) > int(maximum):
        raise SchemaValidationError(f"{path}: size exceeds max{suffix}")


def _check_numeric(value: int | float, schema: dict[str, Any], path: str) -> None:
    if "minimum" in schema and value < schema["minimum"]:
        raise SchemaValidationError(f"{path}: value is below minimum")
    if "maximum" in schema and value > schema["maximum"]:
        raise SchemaValidationError(f"{path}: value exceeds maximum")
    if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
        raise SchemaValidationError(f"{path}: value is below exclusiveMinimum")
    if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
        raise SchemaValidationError(f"{path}: value exceeds exclusiveMaximum")


__all__ = [
    "AgentMessage",
    "AgentProvider",
    "AgentSession",
    "AgentTurn",
    "NativeToolCall",
    "NativeToolSpec",
    "SchemaValidationError",
    "parse_json_strict",
    "validate_schema_instance",
]
