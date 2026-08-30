from __future__ import annotations

from atomic_skillgraph.core.bindings import BindingExpression, BindingExprKind
from atomic_skillgraph.core.contracts import SemanticPredicate
from atomic_skillgraph.harness.alfworld import (
    AlfWorldAdapter,
    AlfWorldValidatorChannel,
    _goal_roles,
    parse_alfworld_action,
)
from atomic_skillgraph.harness.protocol import HarnessActionSpec, HarnessTask


def _spec(action_type: str, **arguments: str) -> HarnessActionSpec:
    return HarnessActionSpec(
        action_id=f"a_{action_type}", revision=0, action_type=action_type,
        arguments=arguments, display_text=action_type, raw_action=action_type,
        metadata={},
    )


def _record(channel: AlfWorldValidatorChannel, revision: int, action_type: str, **arguments: str) -> None:
    channel.record(
        _spec(action_type, **arguments), accepted=True, revision=revision,
        done=False, won=False,
    )


def test_goal_roles_and_look_contract_keep_object_and_light_distinct() -> None:
    assert _goal_roles("look at book under the desklamp.") == {
        "object": "book", "light_source": "desklamp",
    }
    adapter = AlfWorldAdapter(split="train")
    contract = adapter.task_contract(HarnessTask(
        "look", "look at book under the desklamp.", "alfworld", "look_at_obj_in_light",
    ))
    assert len(contract.target_effects) == 1
    effect = contract.target_effects[0]
    assert effect.predicate == "object.observed_with"
    assert effect.args == {"object": "book", "light": "desklamp"}
    assert _goal_roles("examine the book with the desklamp.") == {
        "object": "book", "light_source": "desklamp",
    }
    assert _goal_roles("put a hot egg in garbagecan.") == {
        "object": "egg", "destination": "garbagecan",
    }
    assert _goal_roles("put a clean mug in coffeemachine.") == {
        "object": "mug", "destination": "coffeemachine",
    }


def test_pick_two_contract_requires_two_distinct_placement_witnesses() -> None:
    adapter = AlfWorldAdapter(split="train")
    contract = adapter.task_contract(HarnessTask(
        "two", "find two remotecontrol and put them in armchair.",
        "alfworld", "pick_two_obj_and_place",
    ))
    channel = AlfWorldValidatorChannel()
    _record(channel, 1, "TAKE", object="remotecontrol_1", source="desk_1")
    _record(channel, 2, "PUT", object="remotecontrol_1", destination="armchair_1")
    one = channel.validate_task_contract(contract)
    assert not one.passed
    assert not one.checks["target_0"]
    assert not one.checks["cardinality_constraints"]
    assert not one.checks["identity_constraints"]

    _record(channel, 3, "TAKE", object="remotecontrol_2", source="shelf_1")
    _record(channel, 4, "PUT", object="remotecontrol_2", destination="armchair_1")
    two = channel.validate_task_contract(contract)
    assert two.passed


def test_state_facts_replace_stale_holds_and_locations() -> None:
    channel = AlfWorldValidatorChannel()
    _record(channel, 1, "TAKE", object="apple_1", source="cabinet_1")
    facts = channel.snapshot()["facts"]
    assert {"predicate": "agent.holds", "args": {"object": "apple_1"}} in facts

    _record(channel, 2, "PUT", object="apple_1", destination="fridge_1")
    facts = channel.snapshot()["facts"]
    assert {"predicate": "agent.holds", "args": {"object": "apple_1"}} not in facts
    assert {"predicate": "object.at_location", "args": {"location": "fridge_1", "object": "apple_1"}} in facts

    _record(channel, 3, "TAKE", object="apple_1", source="fridge_1")
    facts = channel.snapshot()["facts"]
    assert not any(item["predicate"] == "object.at_location" for item in facts)


def test_navigation_container_and_light_actions_publish_current_atomic_facts() -> None:
    channel = AlfWorldValidatorChannel()
    _record(channel, 1, "GO_TO", destination="cabinet_1")
    _record(channel, 2, "OPEN", object="cabinet_1")
    _record(channel, 3, "TOGGLE_ON", object="desklamp_1")
    facts = channel.snapshot()["facts"]
    assert {"predicate": "agent.at_location", "args": {"location": "cabinet_1"}} in facts
    assert {"predicate": "container.open", "args": {"container": "cabinet_1"}} in facts
    assert {"predicate": "light.on", "args": {"light": "desklamp_1"}} in facts

    _record(channel, 4, "GO_TO", destination="desk_1")
    _record(channel, 5, "CLOSE", object="cabinet_1")
    _record(channel, 6, "TOGGLE_OFF", object="desklamp_1")
    facts = channel.snapshot()["facts"]
    assert {"predicate": "agent.at_location", "args": {"location": "desk_1"}} in facts
    assert not any(
        item == {"predicate": "agent.at_location", "args": {"location": "cabinet_1"}}
        for item in facts
    )
    assert {"predicate": "container.closed", "args": {"container": "cabinet_1"}} in facts
    assert {"predicate": "container.open", "args": {"container": "cabinet_1"}} not in facts
    assert {"predicate": "light.off", "args": {"light": "desklamp_1"}} in facts
    assert {"predicate": "light.on", "args": {"light": "desklamp_1"}} not in facts


def test_atomic_effect_requires_concrete_binding_and_current_fact() -> None:
    channel = AlfWorldValidatorChannel()
    _record(channel, 1, "HEAT", object="egg_1", station="microwave_1")
    effect = SemanticPredicate(
        "object.heated",
        {"object": BindingExpression(BindingExprKind.SKILL_INPUT, source_role="object")},
    )
    assert not channel.validate_atomic_effect({"effects": [effect], "bindings": {"object": "egg"}}).passed
    assert channel.validate_atomic_effect({"effects": [effect], "bindings": {"object": "egg_1"}}).passed
    assert not channel.validate_atomic_effect({"effects": [effect], "bindings": {"object": "egg_2"}}).passed


def test_look_witness_comes_from_use_while_holding_target() -> None:
    channel = AlfWorldValidatorChannel()
    _record(channel, 1, "USE", object="desklamp_1")
    assert not any(item["predicate"] == "object.observed_with" for item in channel.snapshot()["facts"])
    _record(channel, 2, "USE", object="desklamp_1")  # turn it back off
    _record(channel, 3, "TAKE", object="book_1", source="desk_1")
    _record(channel, 4, "USE", object="desklamp_1")
    assert {
        "predicate": "object.observed_with",
        "args": {"light": "desklamp_1", "object": "book_1"},
    } in channel.snapshot()["facts"]
    _record(channel, 5, "USE", object="desklamp_1")
    assert not any(item["predicate"] == "object.observed_with" for item in channel.snapshot()["facts"])

    _record(channel, 6, "EXAMINE", object="book_1")
    assert not any(item["predicate"] == "object.observed_with" for item in channel.snapshot()["facts"])


def test_slice_is_structured_at_adapter_boundary() -> None:
    action_type, arguments, _, _ = parse_alfworld_action("slice apple 1 with knife 1")
    assert action_type == "SLICE"
    assert arguments == {"object": "apple_1", "tool": "knife_1"}
