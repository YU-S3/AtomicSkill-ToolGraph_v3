"""Single deterministic authority for semantic boundary-type compatibility."""

from __future__ import annotations


_PRIMITIVE_ALIASES = {
    "str": "string",
    "string": "string",
    "bool": "boolean",
    "boolean": "boolean",
    "int": "number",
    "integer": "number",
    "float": "number",
    "number": "number",
    "list": "array",
    "array": "array",
    "dict": "object_map",
    "mapping": "object_map",
    "object_map": "object_map",
    "entity": "entity",
}


def normalize_semantic_type(value: str) -> str:
    """Return the stable primitive alias or declared symbolic type name."""

    normalized = str(value).strip().casefold()
    return _PRIMITIVE_ALIASES.get(normalized, normalized)


def is_symbolic_entity_subtype(value: str) -> bool:
    """Whether ``value`` denotes a non-primitive symbolic entity boundary."""

    normalized = normalize_semantic_type(value)
    if normalized in {
        "string",
        "boolean",
        "number",
        "array",
        "object_map",
    }:
        return False
    return bool(normalized)


def semantic_types_compatible(required: str, offered: str) -> bool:
    """Match exact types, primitive aliases, or one generic ``entity`` side.

    Unregistered non-empty names are legitimate symbolic entity subtypes, but
    two different concrete subtypes never match each other merely because both
    are symbolic.
    """

    required_type = normalize_semantic_type(required)
    offered_type = normalize_semantic_type(offered)
    if not required_type or not offered_type:
        return False
    if required_type == offered_type:
        return True
    if required_type == "entity" and is_symbolic_entity_subtype(offered_type):
        return True
    if offered_type == "entity" and is_symbolic_entity_subtype(required_type):
        return True
    return False


__all__ = [
    "is_symbolic_entity_subtype",
    "normalize_semantic_type",
    "semantic_types_compatible",
]
