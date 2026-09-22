"""Task-local proof and consumer-input handoff for Agent-selected Support."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any

from ..core.bindings import BindingResolution, BindingSource, BindingStatus, RuntimeBinding, resolution_satisfies
from ..core.results import ValidationResult
from ..core.semantic_types import semantic_types_compatible
from ..core.support_authority import input_identity_source_role, _referenced_role
from ..core.serialization import to_primitive
from .support_retriever import SupportObligation, predicate_input_mapping


@dataclass
class SupportRequest:
    root_occurrence_id: str
    consumer: Any
    producer: Any
    input_mapping: dict[str, str]
    output_mapping: dict[str, str]
    anchor_inputs: set[str] = field(default_factory=set)
    mapping_evidence: list[dict] = field(default_factory=list)
    consumer_value_guard: Any = None
    grounding_constraints: list[Any] = field(default_factory=list)
    grounding_proofs: list[Any] = field(default_factory=list)
    producer_occurrence_id: str = ''


def consumer_constraints(compiler, atomic):
    """Read formal entry constraints, translating the declared Tool input map.

    Later serial Tools may consume future outputs: they cannot authorize a
    parent-input handoff now. No ACTION body is inspected or inferred here.
    """
    from ..core.bindings import BindingExprKind, BindingExpression, GroundingConstraint
    from ..tooling.entry_contract import normalize_entry_contract
    result = []
    for implementation in compiler.skills.implementations_for(atomic.ref, mode=compiler.mode):
        result.extend(implementation.grounding_constraints)
        bindings = implementation.tool_bindings
        if not bindings:
            continue
        first = min(bindings, key=lambda binding: binding.order)
        tool = compiler.tools.get(first.tool_ref)
        try:
            entry = normalize_entry_contract(tool.interface.get('entry_contract'), tool.signature.get('properties', {}))
        except (ValueError, TypeError):
            continue  # Historical tools without an entry declaration confer no authority.
        for raw in entry['grounding_constraints']:
            constraint = GroundingConstraint(**raw)
            mapping = {}
            for role, expression in constraint.argument_mapping.items():
                if expression.kind is BindingExprKind.CONSTANT:
                    mapping[role] = expression
                else:
                    source = first.parameter_mapping.get(expression.source_role)
                    if source is None or source.kind not in {BindingExprKind.SKILL_INPUT, BindingExprKind.CONSTANT}:
                        break
                    mapping[role] = copy.deepcopy(source)
            else:
                result.append(replace(constraint, argument_mapping=mapping))
    return result


def consumer_relations(request, consumer_atomic, ctx):
    """Declared preconditions, or formal public adapter projections of entry.

    A multi-argument action alone is NOT a semantic relation. The existing
    adapter schema must explicitly map its arguments to a named predicate.
    """
    from ..core.bindings import GroundingConstraintKind
    from ..core.contracts import SemanticPredicate
    result = [(predicate, None) for predicate in consumer_atomic.preconditions]
    if not request.grounding_constraints:
        return result
    schema = getattr(ctx.harness, 'public_catalog_relation_schema', None)
    predicates = getattr(ctx.harness, 'semantic_predicate_schema', None)
    if not callable(schema) or not callable(predicates):
        return result
    domains = {spec.predicate: spec.effect_domain for spec in predicates()}
    for constraint in request.grounding_constraints:
        if constraint.kind is not GroundingConstraintKind.HARNESS_AFFORDANCE:
            continue
        for relation in schema():
            if relation['action_type'] != constraint.action_type:
                continue
            for projection in relation['predicates']:
                mapping = projection['argument_mapping']
                if projection['predicate'] not in domains or not set(mapping.values()) <= constraint.argument_mapping.keys():
                    continue
                predicate = SemanticPredicate(projection['predicate'],
                    {role: constraint.argument_mapping[argument] for role, argument in mapping.items()},
                    effect_domain=domains[projection['predicate']])
                result.append((predicate, constraint))
    return result


def _fail(ctx, metric, code, message):
    values = ctx.trace_builder.trace.metadata.setdefault('r101_metrics', {})
    values[metric] = values.get(metric, 0) + 1
    return ValidationResult.fail('support', code, message)


def _prove_request_core(request, consumer_atomic, ctx, *, agent_selected=False):
    """Aliases recall candidates; anchors or a complete relation authorize them."""
    producer = request.producer
    parent = ctx.binding_store.snapshot_for_node(request.consumer)
    outputs = {p.name: p for p in producer.outputs}
    inputs = {p.name: p for p in consumer_atomic.inputs}
    constraints = producer.validator_spec.get('output_semantic_constraints', {})
    authorized = set()
    for out, dest in request.output_mapping.items():
        if out not in outputs or dest not in inputs or not semantic_types_compatible(
                outputs[out].semantic_type, inputs[dest].semantic_type):
            return ValidationResult.fail('support', 'support_mapping_boundary_invalid', 'Unknown or incompatible boundary role')
        anchor = ctx.binding_store.semantic_anchor_for(request.consumer, dest) or parent.get(dest)
        identity = input_identity_source_role(producer, out)
        semantic_input = constraints.get(out, {}).get('compatible_with_input')
        if anchor is not None and anchor.status is BindingStatus.GROUNDED:
            if identity and request.input_mapping.get(identity) == dest:
                authorized.add(out)
                request.mapping_evidence.append({'kind': 'input_identity', 'output': out, 'consumer_input': dest})
            elif semantic_input and request.input_mapping.get(semantic_input) == dest:
                authorized.add(out)
                request.anchor_inputs.add(semantic_input)
                request.mapping_evidence.append({'kind': 'semantic_anchor', 'output': out, 'consumer_input': dest})
    # A multi-argument formal relation can carry a correlated fresh output.
    # A unary current-location predicate alone cannot identify an unknown station.
    relation_options = {}
    for predicate, grounding_proof in consumer_relations(request, consumer_atomic, ctx):
        if len(predicate.args) < 2:
            continue
        obligation = SupportObligation('predicate', str(consumer_atomic.ref), request.consumer.occurrence_id,
            predicate=predicate.predicate, predicate_args=tuple(sorted(predicate.args.items())),
            effect_domain=predicate.effect_domain.value, cardinality=predicate.cardinality, distinct_by=predicate.distinct_by)
        for mapping in predicate_input_mapping(producer, consumer_atomic, obligation):
            output_part = {p: c for p, c in mapping.items() if p in outputs}
            if not output_part or any(request.output_mapping.get(p, c) != c for p, c in output_part.items()):
                continue
            input_part = {p: c for p, c in mapping.items() if p not in outputs}
            proposed_inputs = dict(request.input_mapping)
            proposed_anchors = set(request.anchor_inputs)
            anchored, conflict = False, False
            for p, c in mapping.items():
                anchor = ctx.binding_store.semantic_anchor_for(request.consumer, c) or parent.get(c)
                if anchor is None or anchor.status is not BindingStatus.GROUNDED:
                    continue
                # The full predicate correlates this output to an anchored
                # consumer role. Delivery must still match that anchor. An
                # input identity is needed only to infer a producer INPUT;
                # it is not required for an independently witnessed output.
                anchored = True
                source = p if p in input_part else input_identity_source_role(producer, p)
                semantic_source = constraints.get(p, {}).get('compatible_with_input') if p in outputs else None
                source = source or semantic_source
                if not source:
                    continue
                if source in proposed_inputs and proposed_inputs[source] != c:
                    conflict = True
                    break
                proposed_inputs[source] = c
                if semantic_source:
                    proposed_anchors.add(source)
            if anchored and not conflict and all(proposed_inputs.get(p) == c for p, c in input_part.items()):
                from ..core.refs import canonical_json
                combined = {**request.output_mapping, **output_part}
                identity = canonical_json({'inputs': proposed_inputs, 'outputs': combined})
                relation_options.setdefault(identity, (proposed_inputs, combined, proposed_anchors,
                    {'kind': 'joint_relation', 'predicate': predicate.predicate, 'mapping': mapping}, grounding_proof))
    if len(relation_options) > 1:
        return ValidationResult.fail('support', 'support_mapping_ambiguous', 'Multiple complete relation mappings')
    if relation_options:
        new_inputs, new_outputs, new_anchors, proof, grounding_proof = next(iter(relation_options.values()))
        request.input_mapping, request.output_mapping, request.anchor_inputs = new_inputs, new_outputs, new_anchors
        authorized.update(proof['mapping'].keys() & outputs.keys())
        request.mapping_evidence.append(proof)
        if grounding_proof is not None:
            request.grounding_proofs.append(grounding_proof)
            proof['public_relation_constraint'] = to_primitive(grounding_proof)
    if set(request.output_mapping) - authorized:
        return ValidationResult.fail('support', 'support_mapping_authority_missing',
                     'Role aliases do not prove this consumer input; supply an anchored identity or full relation')
    # Predicate-only helpers must operate on their consumer's values, not pick
    # an arbitrary entity through their own action affordances.
    if not request.output_mapping and not request.input_mapping:
        return ValidationResult.fail('support', 'support_mapping_authority_missing', 'Missing formal input mapping')
    owner = ctx.binding_store._repeat_step_owner.get(request.consumer.step_id)
    if owner is not None and not agent_selected:
        constraint, _ = owner
        restricted = {constraint.step_role_bindings[request.consumer.step_id][r]
                      for r in constraint.distinct_roles if r in constraint.step_role_bindings[request.consumer.step_id]}
        if any(dest in restricted and not input_identity_source_role(producer, out)
               for out, dest in request.output_mapping.items()):
            return ValidationResult.fail('support', 'support_fresh_repeat_requires_agent',
                         'Fresh output is unknown; no implicit exclusion input may be invented')
    return ValidationResult.ok('support', mapping_proved=True)


def prove_request(request, consumer_atomic, ctx, *, agent_selected=False):
    """Execution boundary: recheck current evidence and count a rejection once."""
    result = _prove_request_core(request, consumer_atomic, ctx, agent_selected=agent_selected)
    if not result.passed:
        return _fail(ctx, 'support_mapping_authority_rejects', result.failure_codes[0], result.messages[0])
    return result


def preview_support_mapping_authority(request, consumer_atomic, ctx, *, agent_selected=True):
    """Pure preview; no binding, metric or caller-request mutation, no authority token."""
    proposed = copy.copy(request)
    for name in ('input_mapping', 'output_mapping', 'anchor_inputs', 'mapping_evidence', 'grounding_proofs'):
        setattr(proposed, name, copy.deepcopy(getattr(request, name)))
    result = _prove_request_core(proposed, consumer_atomic, ctx, agent_selected=agent_selected)
    code = result.failure_codes[0] if not result.passed else ''
    return {
        'status': 'proven' if result.passed else 'needs_anchor' if code == 'support_mapping_authority_missing' else 'incompatible',
        'input_mapping': proposed.input_mapping, 'output_mapping': proposed.output_mapping,
        'anchor_inputs': sorted(proposed.anchor_inputs), 'mapping_evidence': proposed.mapping_evidence,
        'error_code': code, 'required_anchor_or_relation': list(result.messages),
        'relevant_revision': ctx.world_revision,
    }


def binding_accepts_proposal(binding, role, value, ctx, *, semantic_type=None):
    """Compatibility only, not evidence that an Agent proposal is grounded."""
    if binding.resolution in {BindingResolution.CONCRETE, BindingResolution.RELATION_VERIFIED}:
        return value == binding.value
    check = getattr(getattr(ctx, 'harness', None), 'semantic_value_compatible', None)
    return check(role=role, concrete_value=value, semantic_anchor=binding.value,
                 semantic_type=semantic_type or binding.semantic_type) if callable(check) else value == binding.value


def consumer_guard(request, consumer_atomic, values, ctx):
    """Read-only: validates the entire correlated value group before any commit."""
    current = ctx.binding_store.snapshot_for_node(request.consumer)
    specs = {p.name: p for p in consumer_atomic.inputs}
    projected = {k: v.value for k, v in current.items() if v.status is BindingStatus.GROUNDED}
    for role, value in values.items():
        if role not in specs:
            return _fail(ctx, 'support_input_transfer_rejects', 'support_input_unknown', role)
        anchor = ctx.binding_store.semantic_anchor_for(request.consumer, role)
        for binding in (anchor, current.get(role)):
            if binding is None or binding.status is not BindingStatus.GROUNDED:
                continue
            if not binding_accepts_proposal(binding, role, value, ctx, semantic_type=specs[role].semantic_type):
                return _fail(ctx, 'support_parent_identity_rejects', 'runtime_semantic_anchor_mismatch',
                             f'Support cannot replace consumer identity for {role}')
        projected[role] = value
    repeat = ctx.binding_store.preflight_repeat_bindings(request.consumer.step_id, projected)
    if not repeat.passed:
        return _fail(ctx, 'support_parent_identity_rejects', repeat.failure_codes[0], repeat.messages[0])
    for constraint in ctx.task_contract.identity_constraints:
        if constraint.scope != 'occurrence' or constraint.left_role not in projected or constraint.right_role not in projected:
            continue
        equal = projected[constraint.left_role] == projected[constraint.right_role]
        if (constraint.relation.value == 'same_as' and not equal) or (constraint.relation.value == 'distinct_from' and equal):
            return _fail(ctx, 'support_parent_identity_rejects', 'runtime_identity_constraint_mismatch', 'Consumer identity constraint failed')
    if request.consumer_value_guard is not None:
        inherited = request.consumer_value_guard(projected)
        if not inherited.passed:
            return inherited
    return ValidationResult.ok('support', consumer_input_valid=True)


def known_consumer_values(request, actual_arguments):
    # Input mappings are formal identities even for predicate-only helpers.
    return {consumer: actual_arguments[producer] for producer, consumer in request.input_mapping.items()
            if producer in actual_arguments and producer not in request.anchor_inputs}


def validate_transfer(request, consumer_atomic, outputs, ctx):
    """Read-only acceptance used before both effect-shortcut and Tool commit."""
    values = {dest: outputs[out] for out, dest in request.output_mapping.items() if out in outputs}
    if len(values) != len(request.output_mapping):
        return _fail(ctx, 'support_input_transfer_rejects', 'support_atomic_output_unresolved', 'Missing correlated output')
    report = consumer_guard(request, consumer_atomic, values, ctx)
    if not report.passed:
        return report
    for constraint in request.grounding_proofs:
        assignments = []
        for action in ctx.action_catalog:
            if action.revision != ctx.world_revision or action.action_type != constraint.action_type:
                continue
            assignment = {}
            for argument, expr in constraint.argument_mapping.items():
                role = _referenced_role(expr)
                if role:
                    assignment[role] = action.arguments.get(argument)
            if any(assignment.get(role) != value for role, value in values.items() if role in assignment):
                continue
            if not assignment or any(value is None for value in assignment.values()):
                continue
            if not consumer_guard(request, consumer_atomic, assignment, ctx).passed:
                continue
            if ctx.evidence_store.match_constraint(constraint, assignment, ctx.world_revision):
                assignments.append(assignment)
        if not assignments:
            return _fail(ctx, 'support_input_transfer_rejects', 'support_consumer_relation_not_grounded',
                         'Support output has no current complete consumer relation witness')
    specs = {p.name: p for p in consumer_atomic.inputs}
    from ..core.support_authority import output_resolution_authority
    for out, dest in request.output_mapping.items():
        resolution, _ = output_resolution_authority(request.producer, out)
        if not resolution_satisfies(resolution, specs[dest].required_resolution):
            return _fail(ctx, 'support_input_transfer_rejects', 'support_output_resolution_insufficient', dest)
    return report


def transfer_inputs(request, consumer_atomic, result, ctx):
    if not result.atomic_effect_passed or not result.atomic_witness_refs:
        return _fail(ctx, 'support_input_transfer_rejects', 'atomic_effect_witness_missing',
                     'validated support inputs require actual Atomic witnesses')
    report = validate_transfer(request, consumer_atomic, result.validated_outputs, ctx)
    if not report.passed:
        return report
    specs = {p.name: p for p in consumer_atomic.inputs}
    values = {dest: result.validated_outputs[out] for out, dest in request.output_mapping.items()}
    bindings = {}
    ctx.binding_store._check_certified_values(result.validated_outputs, result.validated_output_bindings)
    for out, dest in request.output_mapping.items():
        actual = result.validated_output_bindings[out]
        if not semantic_types_compatible(actual.semantic_type, specs[dest].semantic_type):
            return _fail(ctx, 'support_input_transfer_rejects', 'support_output_type_mismatch', dest)
        if actual.resolution is BindingResolution.RELATION_VERIFIED and actual.world_revision != ctx.world_revision:
            actual = replace(actual, resolution=BindingResolution.CONCRETE)
        if not resolution_satisfies(actual.resolution, specs[dest].required_resolution):
            return _fail(ctx, 'support_input_transfer_rejects', 'support_output_resolution_insufficient', dest)
        bindings[dest] = replace(actual, role=dest, source=BindingSource.DATA_FLOW)
    # No parent output publication or parent Repeat commit here.
    ctx.binding_store.commit_validated_support_inputs(request.consumer.occurrence_id, bindings)
    ctx.trace_builder.trace.metadata.setdefault('support_input_transfers', []).append({
        'consumer_occurrence_id': request.consumer.occurrence_id, 'producer_atomic_ref': str(request.producer.ref),
        'producer_occurrence_id': request.producer_occurrence_id, 'root_occurrence_id': request.root_occurrence_id,
        'output_mapping': dict(request.output_mapping), 'values': values,
        'mapping_evidence': to_primitive(request.mapping_evidence), 'witness_refs': list(result.atomic_witness_refs),
        'revision': ctx.world_revision})
    return report



def mapped_support_bindings(producer, input_mapping, anchor_inputs, occurrence, store):
    """Read the explicit helper input mapping without selecting new values.

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
