"""Build the state-transition-authoritative extractor view of a TraceRecord."""

from __future__ import annotations

import copy
import re
from typing import AbstractSet, Any, Mapping

from ..core.errors import AtomicSkillGraphError, FailureLayer
from ..core.refs import canonical_json
from ..core.serialization import to_primitive
from .atomicizer import reduce_action_state


def _fact_identity(fact: dict[str, Any]) -> tuple[str, str]:
    return (
        str(fact.get("predicate", "")),
        canonical_json(dict(fact.get("args") or {})),
    )


def _state_delta(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    before_ids = {_fact_identity(item) for item in before}
    after_ids = {_fact_identity(item) for item in after}
    return (
        [item for item in after if _fact_identity(item) not in before_ids],
        [item for item in before if _fact_identity(item) not in after_ids],
    )


def _same_concrete_identity(left: Any, right: Any) -> bool:
    """Return the narrow identity relation used by E1 boundary authorities."""

    return type(left) is type(right) and left == right


def _semantic_alias_input_authorities(
    actions: list[dict[str, Any]],
    primitive_authorities: list[dict[str, Any]],
    *,
    excluded_event_indexes: AbstractSet[int] = frozenset(),
) -> list[dict[str, Any]]:
    """Project deterministic semantic-role aliases from one accepted action.

    The current Validator snapshot delta is the semantic authority.  An alias
    exists only when one (and only one) primitive argument of that same action
    has the exact concrete identity named by a positive semantic fact.
    """

    primitive_by_ref = {
        str(item.get("authority_ref", "")): item
        for item in primitive_authorities
        if str(item.get("kind", "")) == "action_argument"
    }
    candidates: list[dict[str, Any]] = []
    for fallback_index, action in enumerate(actions):
        if action.get("accepted") is not True:
            continue
        event_id = str(action.get("event_id", action.get("action_id", "")))
        event_index = action.get("event_index", fallback_index)
        if (
            not event_id
            or isinstance(event_index, bool)
            or not isinstance(event_index, int)
            or event_index in excluded_event_indexes
        ):
            continue
        arguments = dict(action.get("arguments") or {})
        effects = [
            dict(item)
            for item in list(action.get("authoritative_positive_effects") or [])
            if isinstance(item, Mapping)
        ]
        effects.sort(key=lambda item: (
            str(item.get("predicate", "")),
            canonical_json(dict(item.get("args") or {})),
            str(item.get("witness_ref", "")),
        ))
        for effect in effects:
            if (
                str(effect.get("source_kind", ""))
                != "semantic_snapshot_delta"
                or effect.get("event_index") != event_index
                or str(effect.get("action_id", ""))
                != str(action.get("action_id", ""))
            ):
                continue
            predicate = str(effect.get("predicate", ""))
            witness_ref = str(effect.get("witness_ref", ""))
            effect_domain = str(effect.get("effect_domain", ""))
            effect_args = effect.get("args")
            if (
                not predicate
                or not witness_ref
                or effect_domain not in {"world", "evidence"}
                or not isinstance(effect_args, Mapping)
            ):
                continue
            for raw_semantic_role, value in sorted(
                effect_args.items(), key=lambda item: str(item[0])
            ):
                semantic_role = str(raw_semantic_role)
                matching_roles = [
                    str(raw_role)
                    for raw_role, argument_value in arguments.items()
                    if _same_concrete_identity(argument_value, value)
                ]
                if len(matching_roles) != 1:
                    continue
                source_role = matching_roles[0]
                if source_role == semantic_role:
                    continue
                source_ref = f"action_arg:{event_id}:{source_role}"
                source = primitive_by_ref.get(source_ref)
                if (
                    source is None
                    or str(source.get("role", "")) != source_role
                    or not _same_concrete_identity(source.get("value"), value)
                ):
                    continue
                candidates.append({
                    "authority_ref": (
                        f"semantic_alias:{event_id}:{source_role}:"
                        f"{semantic_role}"
                    ),
                    "event_id": event_id,
                    "event_index": event_index,
                    "kind": "semantic_alias",
                    "source_kind": "semantic_snapshot_alias",
                    "role": semantic_role,
                    "value": copy.deepcopy(value),
                    "source_authority_ref": source_ref,
                    "source_argument_role": source_role,
                    "predicate": predicate,
                    "predicate_argument_role": semantic_role,
                    "witness_ref": witness_ref,
                    "effect_domain": effect_domain,
                })

    # A semantic role can be certified by more than one fact.  Keep the first
    # proof under the frozen deterministic proof order.
    candidates.sort(key=lambda item: (
        str(item.get("predicate", "")),
        str(item.get("predicate_argument_role", "")),
        str(item.get("witness_ref", "")),
        str(item.get("event_id", "")),
        str(item.get("source_argument_role", "")),
        str(item.get("role", "")),
        repr(item.get("value")),
    ))
    aliases: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for candidate in candidates:
        identity = (
            str(candidate["event_id"]),
            str(candidate["source_argument_role"]),
            str(candidate["role"]),
            repr(candidate.get("value")),
        )
        if identity in seen:
            continue
        seen.add(identity)
        aliases.append(candidate)
    return aliases


def _semantic_snapshot_error(message: str) -> AtomicSkillGraphError:
    return AtomicSkillGraphError(
        "semantic_snapshot_integrity_error",
        message,
        layer=FailureLayer.INFRASTRUCTURE,
    )


def _semantic_snapshot_states(
    trace: Any,
) -> tuple[dict[int, dict[str, Any]], set[int]]:
    """Validate and fold the current v3.2 Validator state timeline.

    Repeated records at one revision are permitted only when their complete
    semantic state is identical.  Fact identity is predicate plus canonical
    arguments; revision-local witness certificates are retained as payload.
    """

    metadata = getattr(trace, "metadata", {})
    raw_timeline = (
        metadata.get("semantic_state_snapshots")
        if isinstance(metadata, Mapping)
        else None
    )
    if not isinstance(raw_timeline, list) or not raw_timeline:
        raise _semantic_snapshot_error(
            "formal v3.2 trace is missing semantic_state_snapshots"
        )

    states: dict[int, dict[str, Any]] = {}
    signatures: dict[int, str] = {}
    reset_revisions: set[int] = set()
    predicate_domains: dict[str, str] = {}
    for sequence_index, raw_snapshot in enumerate(raw_timeline):
        if not isinstance(raw_snapshot, Mapping):
            raise _semantic_snapshot_error(
                f"semantic snapshot {sequence_index} is not an object"
            )
        revision = raw_snapshot.get("revision")
        if isinstance(revision, bool) or not isinstance(revision, int):
            raise _semantic_snapshot_error(
                f"semantic snapshot {sequence_index} has invalid revision"
            )
        done = raw_snapshot.get("done")
        won = raw_snapshot.get("won")
        if not isinstance(done, bool) or not isinstance(won, bool):
            raise _semantic_snapshot_error(
                f"semantic snapshot r{revision} has invalid terminal state"
            )
        raw_facts = raw_snapshot.get("facts")
        if not isinstance(raw_facts, list):
            raise _semantic_snapshot_error(
                f"semantic snapshot r{revision} facts are not a list"
            )
        facts: list[dict[str, Any]] = []
        seen_identities: set[tuple[str, str]] = set()
        for fact_index, raw_fact in enumerate(raw_facts):
            if not isinstance(raw_fact, Mapping):
                raise _semantic_snapshot_error(
                    f"semantic snapshot r{revision} fact {fact_index} is not an object"
                )
            predicate = raw_fact.get("predicate")
            if not isinstance(predicate, str) or not predicate.strip():
                raise _semantic_snapshot_error(
                    f"semantic snapshot r{revision} fact {fact_index} lacks predicate"
                )
            args = raw_fact.get("args")
            if not isinstance(args, Mapping):
                raise _semantic_snapshot_error(
                    f"semantic snapshot r{revision} fact {fact_index} args are not an object"
                )
            effect_domain = raw_fact.get("effect_domain")
            if effect_domain not in {"world", "evidence"}:
                raise _semantic_snapshot_error(
                    f"semantic snapshot r{revision} fact {fact_index} has invalid effect_domain"
                )
            witness_ref = raw_fact.get("witness_ref")
            if not isinstance(witness_ref, str) or not witness_ref.strip():
                raise _semantic_snapshot_error(
                    f"semantic snapshot r{revision} fact {fact_index} lacks witness_ref"
                )
            fact = {
                "predicate": predicate,
                "args": copy.deepcopy(dict(args)),
                "effect_domain": effect_domain,
                "witness_ref": witness_ref,
            }
            try:
                identity = _fact_identity(fact)
            except (TypeError, ValueError) as exc:
                raise _semantic_snapshot_error(
                    f"semantic snapshot r{revision} fact {fact_index} arguments are not canonical"
                ) from exc
            if identity in seen_identities:
                raise _semantic_snapshot_error(
                    f"semantic snapshot r{revision} contains duplicate fact {predicate}"
                )
            seen_identities.add(identity)
            known_domain = predicate_domains.setdefault(
                predicate, effect_domain,
            )
            if known_domain != effect_domain:
                raise _semantic_snapshot_error(
                    f"semantic predicate {predicate} changes effect_domain within one trace"
                )
            facts.append(fact)
        facts.sort(key=_fact_identity)
        state = {
            "revision": revision,
            "done": done,
            "won": won,
            "facts": facts,
        }
        signature = canonical_json(state)
        if revision in signatures and signatures[revision] != signature:
            raise _semantic_snapshot_error(
                f"semantic snapshot state conflicts at revision {revision}"
            )
        signatures[revision] = signature
        states.setdefault(revision, state)
        if str(raw_snapshot.get("origin", "")) == "reset":
            reset_revisions.add(revision)

    if not reset_revisions:
        raise _semantic_snapshot_error(
            "formal v3.2 trace is missing its reset semantic snapshot"
        )
    return states, reset_revisions


def _entity_family(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(
        r"[^a-z0-9]", "",
        re.sub(r"(?:_|\s)\d+$", "", value.casefold()),
    )


def _terminal_certificate_projection(
    certificates: list[dict[str, Any]],
    *,
    action: dict[str, Any],
    before_facts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach exact, action/state-derived candidates to narrow certificates."""

    available = [
        value
        for value in dict(action.get("arguments") or {}).values()
        if isinstance(value, str) and value
    ]
    available.extend(
        value
        for fact in before_facts
        for value in dict(fact.get("args") or {}).values()
        if isinstance(value, str) and value
    )
    projected: list[dict[str, Any]] = []
    for certificate in certificates:
        candidates: dict[str, list[str]] = {}
        for role, family_value in dict(certificate.get("args") or {}).items():
            family = _entity_family(family_value)
            matching = sorted({
                value for value in available
                if family and _entity_family(value) == family
            })
            if matching:
                candidates[str(role)] = matching
        projected.append({
            **certificate,
            "concrete_binding_candidates": candidates,
        })
    return projected


class TraceNormalizer:
    def build(self, trace: Any) -> dict[str, Any]:
        task_contract = to_primitive(trace.task_contract)
        current_v32 = str(
            dict(getattr(trace, "metadata", {}) or {}).get(
                "method_patch", ""
            )
        ) == "3.2"
        actions = []
        for index, record in enumerate(trace.environment_actions):
            value = to_primitive(record)
            actions.append({
                "event_index": index, "action_id": value["action_id"],
                "action_type": value["action_type"], "arguments": value["arguments"],
                "accepted": value["accepted"],
                "before_revision": value["revision"], "after_revision": value["new_revision"],
                "done": value["done"], "won": value["won"], "span_id": value["span_id"],
            })
        semantic_states: dict[int, dict[str, Any]] = {}
        reset_revisions: set[int] = set()
        if current_v32:
            semantic_states, reset_revisions = _semantic_snapshot_states(trace)
            if actions:
                first_before = actions[0].get("before_revision")
                if first_before not in reset_revisions:
                    raise _semantic_snapshot_error(
                        "first EnvironmentAction before revision has no reset semantic snapshot"
                    )
        for index, action in enumerate(actions):
            if current_v32:
                before_revision = action.get("before_revision")
                after_revision = action.get("after_revision")
                if (
                    isinstance(before_revision, bool)
                    or not isinstance(before_revision, int)
                    or isinstance(after_revision, bool)
                    or not isinstance(after_revision, int)
                ):
                    raise _semantic_snapshot_error(
                        f"EnvironmentAction {index} has invalid semantic revisions"
                    )
                if before_revision not in semantic_states:
                    raise _semantic_snapshot_error(
                        f"EnvironmentAction {index} before revision {before_revision} has no semantic snapshot"
                    )
                if after_revision not in semantic_states:
                    raise _semantic_snapshot_error(
                        f"EnvironmentAction {index} after revision {after_revision} has no semantic snapshot"
                    )
                if index and before_revision != actions[index - 1].get(
                    "after_revision"
                ):
                    raise _semantic_snapshot_error(
                        f"EnvironmentAction {index} breaks semantic revision continuity"
                    )
                before = [
                    {**copy.deepcopy(fact), "revision": before_revision}
                    for fact in semantic_states[before_revision]["facts"]
                ]
                after = [
                    {**copy.deepcopy(fact), "revision": after_revision}
                    for fact in semantic_states[after_revision]["facts"]
                ]
            else:
                before = reduce_action_state(actions[:index])
                after = reduce_action_state(actions[:index + 1])
            positive, negative = _state_delta(before, after)
            if current_v32:
                action_id = str(action.get("action_id", ""))
                positive = [{
                    **fact,
                    "revision": int(action["after_revision"]),
                    "event_index": index,
                    "source_kind": "semantic_snapshot_delta",
                    "action_id": action_id,
                } for fact in positive]
                negative = [{
                    **fact,
                    "revision": int(action["before_revision"]),
                    "event_index": index,
                    "source_kind": "semantic_snapshot_delta",
                    "action_id": action_id,
                } for fact in negative]
            action.update({
                # The E1 transport uses the normal Python half-open interval.
                # AtomicOccurrenceProposal stores the converted inclusive end.
                "extractor_event_start": index,
                "extractor_event_end_exclusive": index + 1,
                "input_role_candidates": dict(action.get("arguments") or {}),
                "authoritative_before_state_facts": before,
                "authoritative_positive_effects": positive,
                "authoritative_negative_effects": negative,
                # Reserved for adapter-provided, state-derived certificates.
                # Official ``won`` is never used to synthesize a semantic fact.
                "authoritative_terminal_effect_certificates": [],
            })
        spans = [to_primitive(item) for item in trace.runtime_spans if item.learnable]
        validations = [to_primitive(item) for item in trace.validations]
        input_authorities: list[dict[str, Any]] = []
        seen_input_authorities: set[tuple[str, str, str]] = set()
        for action in actions:
            if action.get("accepted") is not True:
                continue
            event_id = str(
                action.get("event_id", action.get("action_id", ""))
            )
            if not event_id:
                continue
            for raw_role, value in dict(
                action.get("arguments") or {}
            ).items():
                role = str(raw_role)
                authority_ref = f"action_arg:{event_id}:{role}"
                identity = (authority_ref, role, repr(value))
                if identity in seen_input_authorities:
                    continue
                seen_input_authorities.add(identity)
                input_authorities.append({
                    "authority_ref": authority_ref,
                    "event_id": event_id,
                    "argument_role": role,
                    "kind": "action_argument",
                    "source_kind": "action_argument",
                    "role": role,
                    "value": value,
                })
        if current_v32:
            runtime_trial_event_indexes: set[int] = set()
            raw_trials = dict(
                dict(getattr(trace, "metadata", {}) or {}).get(
                    "runtime_tool_trials", {}
                )
                or {}
            )
            for raw_trial in raw_trials.values():
                if not isinstance(raw_trial, Mapping):
                    continue
                start = raw_trial.get("trial_event_start")
                end = raw_trial.get("trial_event_end")
                if (
                    isinstance(start, bool)
                    or not isinstance(start, int)
                    or isinstance(end, bool)
                    or not isinstance(end, int)
                    or start < 0
                    or end < start
                    or end >= len(actions)
                ):
                    continue
                runtime_trial_event_indexes.update(range(start, end + 1))
            input_authorities.extend(
                _semantic_alias_input_authorities(
                    actions,
                    input_authorities,
                    excluded_event_indexes=runtime_trial_event_indexes,
                )
            )
        return {
            "trace_id": trace.trace_id, "task_goal": trace.task.goal,
            "source_task": {
                "task_id": trace.task.task_id,
                "task_signature": trace.task.task_signature,
                "goal": trace.task.goal,
                "benchmark": trace.task.benchmark,
                "task_type": trace.task.task_type,
                "context": {
                    "env_index": trace.task.metadata.get("env_index"),
                    "game_file": trace.task.metadata.get("game_file", ""),
                },
                "metadata": dict(trace.task.metadata),
            },
            "task_contract": task_contract, "benchmark_success": trace.benchmark_success,
            "actions": actions, "runtime_spans": spans, "validations": validations,
            "semantic_authority_source": (
                "validator_snapshot_v3_2"
                if current_v32
                else "legacy_action_reducer"
            ),
            "node_records": [to_primitive(item) for item in trace.node_records],
            "implementation_invocations": [to_primitive(item) for item in trace.implementation_invocations],
            "tool_executions": [to_primitive(item) for item in trace.tool_executions],
            "boundary_authorities": {
                "inputs": input_authorities,
                "effects": [],
            },
        }
