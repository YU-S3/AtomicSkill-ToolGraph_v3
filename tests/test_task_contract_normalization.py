from __future__ import annotations

import pytest

from atomic_skillgraph.core.contracts import (
    IdentityConstraint,
    IdentityRelation,
    SemanticPredicate,
    TaskContract,
)
from atomic_skillgraph.harness.alfworld import AlfWorldAdapter
from atomic_skillgraph.harness.protocol import HarnessTask
from atomic_skillgraph.planner.multiplicity import normalize_task_contract


def test_pick_two_uses_cardinality_without_synthetic_identity_roles() -> None:
    contract = AlfWorldAdapter(split="train").task_contract(HarnessTask(
        "two",
        "find two remotecontrol and put them in armchair.",
        "alfworld",
        "pick_two_obj_and_place",
    ))

    assert contract.identity_constraints == []
    assert contract.cardinality_constraints == [{
        "constraint_id": "cc_object_at_location_distinct_object",
        "predicate": "object.at_location",
        "count": 2,
        "distinct_by": "object",
        "shared_roles": ["location"],
        "composition_mode": "repeat_unit",
    }]
    assert normalize_task_contract(contract).cardinality_constraints == (
        contract.cardinality_constraints
    )


def test_identity_constraints_must_reference_declared_target_roles() -> None:
    contract = TaskContract(
        target_effects=[SemanticPredicate(
            "P", {"object": "target", "location": "destination"},
        )],
        identity_constraints=[IdentityConstraint(
            "object_1",
            IdentityRelation.DISTINCT_FROM,
            "object_2",
            "task",
        )],
    )

    with pytest.raises(
        ValueError,
        match="identity constraints must reference declared target roles",
    ):
        normalize_task_contract(contract)


def test_same_role_distinctness_must_use_cardinality_authority() -> None:
    contract = TaskContract(
        target_effects=[SemanticPredicate("P", {"object": "target"})],
        identity_constraints=[IdentityConstraint(
            "object",
            IdentityRelation.DISTINCT_FROM,
            "object",
            "task",
        )],
    )

    with pytest.raises(ValueError, match="same-role distinctness"):
        normalize_task_contract(contract)


def test_transformation_same_object_identity_remains_valid() -> None:
    contract = TaskContract(
        target_effects=[
            SemanticPredicate("object.heated", {"object": "egg"}),
            SemanticPredicate(
                "object.at_location",
                {"object": "egg", "location": "bowl"},
            ),
        ],
        identity_constraints=[IdentityConstraint(
            "object", IdentityRelation.SAME_AS, "object", "task",
        )],
    )

    assert normalize_task_contract(contract).identity_constraints == (
        contract.identity_constraints
    )
