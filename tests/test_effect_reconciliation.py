from __future__ import annotations

from collections.abc import Callable

import pytest

from experiments.fakes import FakeValidatorChannel
from atomic_skillgraph.core.bindings import (
    BindingExpression,
    BindingExprKind,
    BindingResolution,
    BindingSource,
    BindingStatus,
    RuntimeBinding,
)
from atomic_skillgraph.core.contracts import (
    AbstractAtomicSkill,
    ParameterSpec,
    SemanticPredicate,
)
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.results import (
    AtomicEffectResolution,
    RuntimeOccurrence,
)
from atomic_skillgraph.core.status import SkillStatus
from atomic_skillgraph.harness.alfworld import AlfWorldValidatorChannel
from atomic_skillgraph.harness.protocol import HarnessActionSpec
from atomic_skillgraph.validation.atomic_validator import AtomicValidator


def _action(
    action_id: str,
    revision: int,
    action_type: str,
    arguments: dict[str, str],
) -> HarnessActionSpec:
    return HarnessActionSpec(
        action_id,
        revision,
        action_type,
        arguments,
        action_type,
        {"action_type": action_type, "arguments": arguments},
        {},
    )


def _alf_at(location: str = "cabinet_3") -> AlfWorldValidatorChannel:
    channel = AlfWorldValidatorChannel()
    channel.record(
        _action("go", 0, "GO_TO", {"destination": location}),
        accepted=True,
        revision=1,
        done=False,
        won=False,
    )
    return channel


def _fake_at(location: str = "cabinet_3") -> FakeValidatorChannel:
    channel = FakeValidatorChannel()
    channel.record_fact("agent.at_location", {"location": location}, 1)
    return channel


@pytest.mark.parametrize("factory", [_alf_at, _fake_at])
def test_effect_resolution_fails_closed_for_empty_effects_and_revision(
    factory: Callable[[], object],
) -> None:
    channel = factory()
    effect = SemanticPredicate(
        "agent.at_location",
        {"location": BindingExpression(
            BindingExprKind.SKILL_INPUT,
            source_role="destination",
        )},
    )
    base = {
        "effects": [effect],
        "known_bindings": {},
        "semantic_anchors": {},
        "preferred_values": ["cabinet_3"],
        "input_specs": [ParameterSpec("destination", "entity")],
        "output_specs": [],
        "output_identity": [],
    }

    assert channel.resolve_atomic_effect({**base, "effects": [], "current_revision": 1}).passed is False
    assert channel.resolve_atomic_effect(base).failure_code == "atomic_effect_revision_invalid"
    assert channel.resolve_atomic_effect({**base, "current_revision": "bad"}).passed is False
    assert channel.resolve_atomic_effect({**base, "current_revision": 0}).failure_code == "stale_atomic_effect_witness"
    assert channel.resolve_atomic_effect({**base, "current_revision": 2}).failure_code == "stale_atomic_effect_witness"

    atomic = AbstractAtomicSkill(
        SkillRef("atomic_revision_gate", "1.0.0"),
        "navigate",
        [ParameterSpec("destination", "entity")],
        [],
        [],
        [effect],
        {"validator_id": "harness_atomic_effect"},
        [],
        {},
        {},
        SkillStatus.CANDIDATE,
    )
    occurrence = RuntimeOccurrence(
        "revision", "revision", atomic.ref, [], {}, [], [effect],
    )
    facade_result = AtomicValidator().resolve_current_effect(
        atomic,
        occurrence,
        {},
        channel,
        semantic_anchors={},
        preferred_values=["cabinet_3"],
        current_revision=2,
    )
    assert facade_result.passed is False
    assert facade_result.failure_code == "stale_atomic_effect_witness"

    unsupported = SemanticPredicate(
        "agent.at_location",
        {"location": BindingExpression(
            BindingExprKind.DATA_FLOW,
            source_role="destination",
            source_step="prior",
        )},
    )
    result = channel.resolve_atomic_effect({
        **base,
        "effects": [unsupported],
        "current_revision": 1,
    })
    assert result.passed is False
    assert result.failure_code == "atomic_effect_expression_unsupported"


def _cardinality_channels():
    alf = AlfWorldValidatorChannel()
    alf.record(
        _action(
            "put_one", 0, "PUT",
            {"object": "apple_1", "destination": "basket_1"},
        ),
        accepted=True,
        revision=1,
        done=False,
        won=False,
    )
    alf.record(
        _action(
            "put_two", 1, "PUT",
            {"object": "apple_2", "destination": "basket_1"},
        ),
        accepted=True,
        revision=2,
        done=False,
        won=False,
    )
    fake = FakeValidatorChannel()
    fake.record_fact(
        "object.at_location",
        {"object": "apple_1", "location": "basket_1"},
        1,
    )
    fake.record_fact(
        "object.at_location",
        {"object": "apple_2", "location": "basket_1"},
        2,
    )
    return alf, fake


@pytest.mark.parametrize("channel_index", [0, 1])
def test_effect_resolution_preserves_cardinality_distinct_by_and_constants(
    channel_index: int,
) -> None:
    channel = _cardinality_channels()[channel_index]
    effect = SemanticPredicate(
        "object.at_location",
        {"location": BindingExpression(
            BindingExprKind.CONSTANT,
            constant="basket_1",
        )},
        cardinality=2,
        distinct_by="object",
    )
    request = {
        "effects": [effect],
        "known_bindings": {},
        "semantic_anchors": {},
        "preferred_values": [],
        "input_specs": [],
        "output_specs": [],
        "output_identity": [],
        "current_revision": 2,
    }

    resolution = channel.resolve_atomic_effect(request)

    assert resolution.passed is True
    assert len(resolution.witness_refs) == 2
    atomic = AbstractAtomicSkill(
        SkillRef("atomic_cardinality", "1.0.0"),
        "two objects are placed",
        [],
        [],
        [],
        [effect],
        {"validator_id": "harness_atomic_effect"},
        [],
        {},
        {},
        SkillStatus.CANDIDATE,
    )
    occurrence = RuntimeOccurrence(
        "cardinality", "cardinality", atomic.ref, [], {}, [], [effect],
    )
    final = AtomicValidator().resolve_current_effect(
        atomic,
        occurrence,
        {},
        channel,
        semantic_anchors={},
        preferred_values=[],
        current_revision=2,
    )
    assert final.passed is True

    too_many = SemanticPredicate(
        effect.predicate,
        effect.args,
        cardinality=3,
        distinct_by="object",
    )
    assert channel.resolve_atomic_effect({
        **request,
        "effects": [too_many],
    }).passed is False


@pytest.mark.parametrize("use_fake", [False, True])
def test_effect_resolution_rejects_joint_inconsistency_and_ambiguity(
    use_fake: bool,
) -> None:
    if use_fake:
        inconsistent = FakeValidatorChannel()
        inconsistent.record_fact("agent.holds", {"object": "apple_1"}, 1)
        inconsistent.record_fact("object.heated", {"object": "apple_2"}, 2)
        ambiguous = FakeValidatorChannel()
        ambiguous.record_fact("agent.holds", {"object": "apple_1"}, 1)
        ambiguous.record_fact("agent.holds", {"object": "apple_2"}, 2)
    else:
        inconsistent = AlfWorldValidatorChannel()
        inconsistent.record(
            _action("take", 0, "TAKE", {"object": "apple_1"}),
            accepted=True, revision=1, done=False, won=False,
        )
        inconsistent.record(
            _action(
                "heat", 1, "HEAT",
                {"object": "apple_2", "station": "microwave_1"},
            ),
            accepted=True, revision=2, done=False, won=False,
        )
        ambiguous = AlfWorldValidatorChannel()
        ambiguous.record(
            _action("take_one", 0, "TAKE", {"object": "apple_1"}),
            accepted=True, revision=1, done=False, won=False,
        )
        ambiguous.record(
            _action("take_two", 1, "TAKE", {"object": "apple_2"}),
            accepted=True, revision=2, done=False, won=False,
        )
    role = BindingExpression(
        BindingExprKind.SKILL_INPUT,
        source_role="object",
    )
    request = {
        "known_bindings": {},
        "semantic_anchors": {},
        "preferred_values": ["apple_1", "apple_2"],
        "input_specs": [ParameterSpec("object", "entity")],
        "output_specs": [],
        "output_identity": [],
        "current_revision": 2,
    }

    joint = inconsistent.resolve_atomic_effect({
        **request,
        "effects": [
            SemanticPredicate("agent.holds", {"object": role}),
            SemanticPredicate("object.heated", {"object": role}),
        ],
    })
    multiple = ambiguous.resolve_atomic_effect({
        **request,
        "effects": [SemanticPredicate("agent.holds", {"object": role})],
    })

    assert joint.passed is False
    assert joint.failure_code == "atomic_effect_violation"
    assert multiple.passed is False
    assert multiple.failure_code == "atomic_effect_witness_ambiguous"


@pytest.mark.parametrize("factory", [_alf_at, _fake_at])
def test_contextual_output_uses_exact_witness_argument_without_name_heuristic(
    factory: Callable[[], object],
) -> None:
    channel = factory()
    effect = SemanticPredicate(
        "agent.at_location",
        {"location": BindingExpression(
            BindingExprKind.SKILL_INPUT,
            source_role="destination",
        )},
    )
    atomic = AbstractAtomicSkill(
        SkillRef("atomic_contextual_output", "1.0.0"),
        "navigate",
        [ParameterSpec("destination", "entity", True, True, "concrete")],
        [ParameterSpec("location", "entity")],
        [],
        [effect],
        {"validator_id": "harness_atomic_effect", "output_identity": []},
        [],
        {},
        {},
        SkillStatus.CANDIDATE,
    )
    occurrence = RuntimeOccurrence(
        "contextual", "contextual", atomic.ref, [], {}, [], [effect],
    )
    binding = RuntimeBinding(
        "destination",
        "cabinet_3",
        "entity",
        BindingSource.HARNESS_EVIDENCE,
        BindingStatus.GROUNDED,
        BindingResolution.CONCRETE,
        ["test:destination"],
        1,
    )

    resolution = AtomicValidator().resolve_current_effect(
        atomic,
        occurrence,
        {"destination": binding},
        channel,
        semantic_anchors={},
        preferred_values=[],
        current_revision=1,
    )

    assert resolution.passed is True
    assert resolution.output_candidates == {"location": "cabinet_3"}


def test_passed_resolution_without_witnesses_fails_closed() -> None:
    class MissingWitnessChannel:
        revision = 1

        @staticmethod
        def resolve_atomic_effect(_request):
            return AtomicEffectResolution(
                True,
                resolved_bindings={"destination": "cabinet_3"},
                output_candidates={"result_000": "cabinet_3"},
                witness_refs=[],
            )

        @staticmethod
        def validate_atomic_effect(_request):
            raise AssertionError("final validation must not run without witnesses")

    effect = SemanticPredicate(
        "agent.at_location",
        {"location": BindingExpression(
            BindingExprKind.SKILL_INPUT,
            source_role="destination",
        )},
    )
    atomic = AbstractAtomicSkill(
        SkillRef("atomic_missing_witness", "1.0.0"),
        "navigate",
        [ParameterSpec("destination", "entity")],
        [ParameterSpec("result_000", "entity")],
        [],
        [effect],
        {"output_identity": [{
            "output_role": "result_000",
            "input_role": "destination",
        }]},
        [],
        {},
        {},
        SkillStatus.CANDIDATE,
    )
    occurrence = RuntimeOccurrence(
        "missing", "missing", atomic.ref, [], {}, [], [effect],
    )

    result = AtomicValidator().resolve_current_effect(
        atomic,
        occurrence,
        {},
        MissingWitnessChannel(),
        semantic_anchors={},
        preferred_values=["cabinet_3"],
        current_revision=1,
    )

    assert result.passed is False
    assert result.failure_code == "atomic_effect_witness_missing"
