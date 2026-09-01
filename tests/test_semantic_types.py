from __future__ import annotations

import pytest

from atomic_skillgraph.core.contracts import AbstractAtomicSkill, ParameterSpec
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.semantic_types import (
    normalize_semantic_type,
    semantic_types_compatible,
)
from atomic_skillgraph.core.status import SkillStatus
from atomic_skillgraph.evolution.contract_canonicalizer import (
    AtomicContractCanonicalizer,
    atomic_contract_signature,
)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("str", "string"),
        ("int", "number"),
        ("float", "number"),
        ("list", "array"),
        ("dict", "object_map"),
    ],
)
def test_primitive_aliases_share_one_normalized_type(
    left: str,
    right: str,
) -> None:
    assert normalize_semantic_type(left) == normalize_semantic_type(right)
    assert semantic_types_compatible(left, right)
    assert semantic_types_compatible(right, left)


@pytest.mark.parametrize(
    ("required", "offered"),
    [
        ("object", "entity"),
        ("location", "entity"),
        ("entity", "object"),
    ],
)
def test_generic_entity_matches_declared_symbolic_subtype(
    required: str,
    offered: str,
) -> None:
    assert semantic_types_compatible(required, offered)


@pytest.mark.parametrize(
    ("required", "offered"),
    [
        ("object", "location"),
        ("receptacle", "device"),
        ("entity", "string"),
        ("entity", "number"),
        ("", "entity"),
    ],
)
def test_unrelated_or_primitive_types_do_not_match_entity_contract(
    required: str,
    offered: str,
) -> None:
    assert not semantic_types_compatible(required, offered)


def _typed_atomic(logical_id: str, semantic_type: str) -> AbstractAtomicSkill:
    return AbstractAtomicSkill(
        ref=SkillRef(logical_id, "0.1.0"),
        summary="typed boundary",
        inputs=[ParameterSpec("value", semantic_type)],
        outputs=[],
        preconditions=[],
        effects=[],
        validator_spec={},
        failure_modes=[],
        guideline={},
        metadata={},
        status=SkillStatus.CANDIDATE,
    )


def test_atomic_canonical_identity_and_persisted_boundary_use_normalized_type() -> None:
    alias = _typed_atomic("alias", "str")
    canonical = _typed_atomic("canonical", "string")
    assert atomic_contract_signature(alias) == atomic_contract_signature(canonical)
    normalized = AtomicContractCanonicalizer().canonicalize(alias).atomic
    assert normalized.inputs[0].semantic_type == "string"
