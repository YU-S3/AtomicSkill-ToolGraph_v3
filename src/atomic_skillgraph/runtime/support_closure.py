"""Recursive ordinary-Atomic closure of typed Runtime obligations."""
from __future__ import annotations

import copy
import uuid
from dataclasses import replace
from typing import Any

from ..core.bindings import BindingStatus, resolution_satisfies
from ..core.semantic_types import semantic_types_compatible
from ..core.refs import canonical_json
from ..core.results import RuntimeOccurrence
from ..core.serialization import to_primitive
from ..core.support_authority import input_identity_source_role, support_role_authority
from .checkpoint import increment
from .support_retriever import SupportObligation, predicate_input_mapping
from .support_request import SupportRequest, prove_request, consumer_guard, known_consumer_values, transfer_inputs, consumer_constraints, validate_transfer


def _active(value: Any) -> bool:
    return getattr(value.status, "value", value.status) in {"active", "preferred"}


def mapped_support_bindings(producer, input_mapping, anchor_inputs, occurrence, store):
    """Prove automatic helper inputs before its independent affordance resolver.

    An identity output cannot discover an unknown consumer entity by choosing
    an unrelated helper affordance. A declared semantic-output constraint may
    instead pass the consumer's stable semantic intent to a discovery input.
    Unmapped/insufficient required inputs remain an Agent breakpoint.
    """
    parent = store.snapshot_for_node(occurrence)
    result = {}
    for spec in producer.inputs:
        source = input_mapping.get(spec.name)
        anchor = store.semantic_anchor_for(occurrence, source) if source else None
        binding = (anchor if spec.name in anchor_inputs else parent.get(source)) if source else None
        if (binding is None or binding.status is not BindingStatus.GROUNDED) and anchor is not None:
            binding = anchor
        if binding is None or binding.status is not BindingStatus.GROUNDED:
            if source:
                return None
            continue
        if not semantic_types_compatible(binding.semantic_type, spec.semantic_type):
            return None
        if not resolution_satisfies(binding.resolution, spec.required_resolution):
            if anchor is None or spec.name in anchor_inputs:
                return None
            # A stable task/graph semantic intent is safe to carry to the
            # ordinary resolver, but is NOT promoted to concrete authority.
            # A missing intent (e.g. unknown station) cannot take this path.
            binding = anchor
        result[spec.name] = replace(copy.deepcopy(binding), role=spec.name)
    return result


class SupportClosure:
    def __init__(self, executor: Any) -> None:
        self.executor = executor

    def obligations(self, occurrence: Any, atomic: Any, ctx: Any, invocations: list) -> list[SupportObligation]:
        state = self.executor._activate_occurrence_state(occurrence, atomic, invocations, ctx)
        missing = set(ctx.binding_store.runtime_prompt_projection(
            occurrence, atomic.inputs,
        )["missing_or_insufficient_bindings"])
        result = [SupportObligation(
            "binding", str(atomic.ref), occurrence.occurrence_id, role=item.name,
            semantic_type=item.semantic_type, required_resolution=item.required_resolution,
        ) for item in atomic.inputs if item.required and item.name in missing]
        statuses = state.get("precondition_status", state.get("preconditions", []))
        for index, predicate in enumerate(atomic.preconditions):
            if index < len(statuses) and statuses[index].get("status") == "satisfied":
                continue
            result.append(SupportObligation(
                "predicate", str(atomic.ref), occurrence.occurrence_id,
                predicate=predicate.predicate, predicate_args=tuple(sorted(predicate.args.items())),
                effect_domain=predicate.effect_domain.value,
                cardinality=predicate.cardinality, distinct_by=predicate.distinct_by,
            ))
        return result

    def close(self, occurrence: Any, ctx: Any, invocations: list, stack: tuple = (),
              *, effect_guard=None, input_guard=None) -> bool:
        ex = self.executor
        atomic = ex.invocation_compiler.skills.get_atomic(occurrence.node_ref)
        if ex._complete_from_current_effect(occurrence, ctx, mode="entry", preferred_values=[], effect_guard=effect_guard) is not None:
            return True
        obligations = self.obligations(occurrence, atomic, ctx, invocations)
        increment(ctx, "support_obligation_count", len(obligations))
        changed = False
        for obligation in obligations:
            if obligation not in self.obligations(occurrence, atomic, ctx, invocations):
                continue  # A previous helper may have discharged multiple gaps.
            identity = to_primitive(obligation)
            identity.pop("consumer_occurrence_id")
            key = (str(atomic.ref), canonical_json(identity))
            if key in stack:
                increment(ctx, "support_closure_cycle_count")
                return changed
            pool = [item for item in ex.invocation_compiler.skills.atomics(
                mode=ex.invocation_compiler.mode) if _active(item)]
            availability = ex._support_execution_availability(pool)
            pool = [item for item in pool if availability.get(str(item.ref), False)]
            options = []
            if obligation.kind == "binding":
                candidates = ex.support_retriever.retrieve(
                    blocked_atomic=atomic, missing_roles=[obligation.role], atomics=pool,
                    execution_availability=ex._support_execution_availability(pool), top_k=None,
                )
                for candidate in candidates:
                    producer = ex.invocation_compiler.skills.get_atomic(candidate.atomic_ref)
                    # Multiple legal output mappings are a real ambiguity.
                    for mapping in candidate.role_mappings:
                        inputs = {}
                        anchors = set()
                        source_input = input_identity_source_role(producer, mapping.producer_role)
                        if source_input:
                            inputs[source_input] = mapping.consumer_role
                        else:
                            constraint = producer.validator_spec.get("output_semantic_constraints", {}).get(mapping.producer_role, {})
                            source_input = constraint.get("compatible_with_input")
                            if source_input:
                                inputs[source_input] = mapping.consumer_role
                                anchors.add(source_input)
                        # Other declared discovery outputs may identify the
                        # helper's semantic input even when THIS obligation
                        # asks for its correlated location output.
                        for output_role, constraint in producer.validator_spec.get("output_semantic_constraints", {}).items():
                            source_input = constraint.get("compatible_with_input")
                            matches = [item.name for item in atomic.inputs
                                       if ctx.binding_store.semantic_anchor_for(occurrence, item.name) is not None
                                       and support_role_authority(producer, output_role, atomic, item.name).authorized]
                            if source_input and len(matches) == 1 and source_input not in inputs:
                                inputs[source_input] = matches[0]
                                anchors.add(source_input)
                        options.append((producer, inputs, {mapping.producer_role: mapping.consumer_role}, anchors))
            else:
                for producer in pool:
                    for mapping in predicate_input_mapping(producer, atomic, obligation):
                        inputs = {p: c for p, c in mapping.items()
                                  if p in {item.name for item in producer.inputs}}
                        # A fresh output is not the already-bound consumer
                        # entity unless the formal identity contract proves it.
                        safe = True
                        for p, c in mapping.items():
                            if p in inputs:
                                continue
                            identity_input = input_identity_source_role(producer, p)
                            if not identity_input or (identity_input in inputs and inputs[identity_input] != c):
                                safe = False
                                break
                            inputs[identity_input] = c
                        if safe:
                            options.append((producer, inputs, {}, set()))
            # Inapplicable identity mappings must not compete with a genuinely
            # executable discovery route (nor choose their own unrelated value).
            options = [(p, i, o, a) for p, i, o, a in options
                       if mapped_support_bindings(p, i, a, occurrence, ctx.binding_store) is not None]
            requests = {}
            for p, i, o, a in options:
                request = SupportRequest(ctx.budget.current_occurrence_id, occurrence, p, dict(i), dict(o), set(a))
                request.consumer_value_guard = input_guard
                request.grounding_constraints = consumer_constraints(ex.invocation_compiler, atomic)
                if not prove_request(request, atomic, ctx).passed:
                    continue
                support_bindings = mapped_support_bindings(p, request.input_mapping, request.anchor_inputs, occurrence, ctx.binding_store)
                if support_bindings is None or not consumer_guard(request, atomic,
                        known_consumer_values(request, {k: v.value for k, v in support_bindings.items()}), ctx).passed:
                    continue
                identity = canonical_json({'producer': str(p.ref), 'inputs': request.input_mapping, 'outputs': request.output_mapping})
                requests[identity] = request
            options = list(requests.values())
            if len(options) != 1:
                if len(options) > 1:
                    increment(ctx, "support_closure_ambiguity_count")
                continue
            request = options[0]
            producer, input_mapping, output_mapping, anchor_inputs = (
                request.producer, request.input_mapping, request.output_mapping, request.anchor_inputs)
            support = RuntimeOccurrence(
                step_id=f"support::{uuid.uuid4().hex}", occurrence_id=f"support::{uuid.uuid4().hex}",
                node_ref=producer.ref, requirement_ids=[], binding_specs={},
                implementation_candidates=[str(item.ref) for item in ex.invocation_compiler.skills.implementations_for(
                    producer.ref, mode=ex.invocation_compiler.mode) if _active(item)],
                expected_effects=list(producer.effects),
            )
            if not support.implementation_candidates:
                continue
            request.producer_occurrence_id = support.occurrence_id
            support_bindings = mapped_support_bindings(
                producer, input_mapping, anchor_inputs, occurrence, ctx.binding_store)
            if support_bindings is None:
                continue
            ctx.binding_store.commit_grounded(support.occurrence_id, support_bindings)
            implementations = ex.invocation_compiler.compile_candidates(
                support, ctx.binding_store, max_candidates=3, task_id=ctx.task_id,
            )
            implementations = [item for item in implementations if _active(item.implementation)
                               and all(_active(tool) for tool in item.tools)
                               and str(item.implementation.ref) not in ctx.rejected_runtime_implementations.get(occurrence.occurrence_id, set())]
            preferred = [item for item in implementations if item.implementation.quality.get("preferred")]
            if not implementations or (len(implementations) > 1 and len(preferred) != 1):
                increment(ctx, "support_closure_ambiguity_count", int(len(implementations) > 1))
                continue
            parent_refresh = ctx._after_action_refresh
            failed = ctx.last_failed_invocation
            result = None
            increment(ctx, "support_closure_attempt_count")
            try:
                ctx.begin_occurrence(support)
                guard = lambda resolution: validate_transfer(request, atomic, resolution.output_candidates, ctx)
                result = ex._complete_from_current_effect(support, ctx, mode="entry", preferred_values=[], effect_guard=guard)
                if result is None:
                    self.close(support, ctx, implementations, (*stack, key), effect_guard=guard,
                        input_guard=lambda values: consumer_guard(request, atomic, known_consumer_values(request, values), ctx))
                if ctx.benchmark_terminal():
                    return changed
                if result is None:
                    result = ex._complete_from_current_effect(support, ctx, mode="entry", preferred_values=[], effect_guard=guard)
                current = ctx.binding_store.snapshot_for_node(support)
                # Unmapped inputs may be supplied by recursive validated
                # Support, but cannot be chosen incidentally by this helper's
                # unrelated affordances. Mapped semantic anchors remain
                # subject to ordinary resolution/preflight.
                supplied = all(not item.required or (
                    item.name in current and current[item.name].status is BindingStatus.GROUNDED)
                    for item in producer.inputs)
                if result is None and supplied:
                    result = ex.try_autonomous(support, implementations, ctx,
                        accept_result=lambda value: transfer_inputs(request, atomic, value, ctx), consumer=occurrence)
                elif result is not None:
                    if not transfer_inputs(request, atomic, result, ctx).passed:
                        result = None
            finally:
                ctx.begin_occurrence(occurrence)
                ctx._after_action_refresh = parent_refresh
                ctx.last_failed_invocation = failed
                if parent_refresh:
                    parent_refresh()
            ctx.trace_builder.trace.metadata.setdefault("support_closure_attempts", []).append({
                "obligation": to_primitive(obligation), "support_occurrence_id": support.occurrence_id,
                "support_atomic_ref": str(producer.ref), "result": to_primitive(result),
                "input_mapping": input_mapping, "output_mapping": output_mapping,
            })
            if result is None or not result.atomic_effect_passed:
                if result is not None and result.started and result.implementation_ref:
                    ctx.record_failed_invocation(
                        occurrence_id=occurrence.occurrence_id, implementation_ref=result.implementation_ref,
                        failure_code=result.failure_code, message="Automatic Support failed; see the invocation and rollback audit",
                    )
                continue
            outputs = {consumer: result.validated_outputs[producer_role]
                       for producer_role, consumer in output_mapping.items()
                       if producer_role in result.validated_outputs}
            if len(outputs) != len(output_mapping):
                continue
            refs = list(result.atomic_witness_refs)
            increment(ctx, "support_closure_success_count")
            increment(ctx, "support_auto_execution_count")
            ctx.trace_builder.trace.metadata.setdefault("runtime_support_node_records", []).append({
                "occurrence_id": support.occurrence_id, "step_id": support.step_id,
                "atomic_ref": str(producer.ref), "status": result.node_status.value,
                "direct_result": to_primitive(result), "validated_outputs": outputs,
            })
            if producer.metadata.get("runtime_support_promotion"):
                increment(ctx, "runtime_support_reuse_count")
            changed = True
            if ex._complete_from_current_effect(occurrence, ctx, mode="entry", preferred_values=[], effect_guard=effect_guard) is not None:
                values = ctx.trace_builder.trace.metadata.setdefault('r101_metrics', {})
                values['atomic_satisfied_after_support'] = values.get('atomic_satisfied_after_support', 0) + 1
                return True
        return changed
