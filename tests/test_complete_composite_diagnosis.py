from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    CompositeOccurrence,
    CompositeSkill,
    SemanticPredicate,
    TaskContract,
)
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.status import RuntimeMode, SkillStatus
from atomic_skillgraph.knowledge.query import (
    complete_composite_contract_compatible,
    complete_composite_contract_diagnosis,
)
from atomic_skillgraph.planner.composite_retriever import CompositeRetriever


def _contract(predicate: str = "object.at_location") -> TaskContract:
    return TaskContract(target_effects=[SemanticPredicate(
        predicate,
        {"object": "$object", "location": "$location"},
    )])


def _atomic() -> AbstractAtomicSkill:
    return AbstractAtomicSkill(
        ref=SkillRef("atomic_move", "1.0.0"),
        summary="move",
        inputs=[],
        outputs=[],
        preconditions=[],
        effects=[],
        validator_spec={},
        failure_modes=[],
        guideline={},
        metadata={},
        status=SkillStatus.ACTIVE,
    )


def _composite(
    logical_id: str,
    atomic_ref: SkillRef,
    *,
    contract: TaskContract,
    valid_structure: bool = True,
    status: SkillStatus = SkillStatus.ACTIVE,
) -> CompositeSkill:
    occurrence = CompositeOccurrence(
        step_id="step_1",
        occurrence_id=f"occ_{logical_id}",
        node_ref=atomic_ref,
        binding_specs={},
    )
    return CompositeSkill(
        ref=SkillRef(logical_id, "1.0.0"),
        summary="move",
        occurrences=[occurrence],
        control_sequence=["step_1"] if valid_structure else [],
        data_edges=[],
        dependency_edges=[],
        goal_contract=contract,
        guideline={},
        insight={},
        validator_spec={},
        metadata={},
        status=status,
    )


def test_complete_contract_bool_matcher_delegates_to_structured_diagnosis() -> None:
    required = _contract()
    exact = _contract()
    mismatch = _contract("object.cleaned")

    exact_report = complete_composite_contract_diagnosis(required, exact)
    mismatch_report = complete_composite_contract_diagnosis(required, mismatch)
    assert exact_report.passed
    assert complete_composite_contract_compatible(required, exact)
    assert not mismatch_report.passed
    assert not complete_composite_contract_compatible(required, mismatch)
    assert mismatch_report.target_effect_missing
    assert mismatch_report.target_effect_extra
    assert mismatch_report.failure_codes == (
        "composite_target_effect_missing",
        "composite_target_effect_extra",
    )


class _Skills:
    def __init__(
        self,
        atomic: AbstractAtomicSkill,
        composites: list[CompositeSkill],
    ) -> None:
        self.atomic = atomic
        self._composites = list(composites)

    def composites(self, *, mode: RuntimeMode | str) -> list[CompositeSkill]:
        return list(self._composites)

    def get_atomic(self, ref: SkillRef) -> AbstractAtomicSkill:
        assert ref == self.atomic.ref
        return self.atomic


class _DenyCandidates:
    def allows(self, **_kwargs: Any) -> bool:
        return False


def test_composite_retrieval_rejections_identify_the_first_gate_stage() -> None:
    required = _contract()
    atomic = _atomic()
    composites = [
        _composite(
            "contract_rejected",
            atomic.ref,
            contract=_contract("object.cleaned"),
        ),
        _composite(
            "structure_rejected",
            atomic.ref,
            contract=required,
            valid_structure=False,
        ),
        _composite(
            "lifecycle_rejected",
            atomic.ref,
            contract=required,
            status=SkillStatus.CANDIDATE,
        ),
    ]
    result = CompositeRetriever(
        _Skills(atomic, composites),
        candidate_policy=_DenyCandidates(),
    ).retrieve_complete(
        SimpleNamespace(task_id="task", goal="move"),
        required,
        mode=RuntimeMode.ONLINE,
        harness_profile="alfworld",
    )

    by_ref = {item["composite_ref"]: item for item in result.rejections}
    contract_rejection = by_ref["skill://contract_rejected@1.0.0"]
    assert contract_rejection["stage"] == "retrieval_contract"
    assert contract_rejection["contract_diagnosis"]["passed"] is False
    assert by_ref["skill://structure_rejected@1.0.0"]["stage"] == (
        "retrieval_structure"
    )
    assert by_ref["skill://lifecycle_rejected@1.0.0"]["stage"] == (
        "lifecycle_policy"
    )
