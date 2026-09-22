"""Read-only public projection of the actual compiled Support boundary."""
from __future__ import annotations

import copy
import json
from dataclasses import replace

from ..core.support_authority import input_identity_source_role
from .support_request import (SupportRequest, consumer_constraints,
    preview_support_mapping_authority)


def project_candidate(candidate, producer, consumer, occurrence, ctx, compiler, routes):
    # The selected candidate is compiled using the very same empty occurrence
    # as its execution call. Do not reconstruct its schema from type names.
    schema = copy.deepcopy(routes[0].spec.input_schema) if routes else {}
    if any(route.spec.input_schema != schema for route in routes):
        raise ValueError('Support routes disagree on their Atomic input schema')
    if occurrence is None or getattr(occurrence, 'consumer_scope', 'node') == 'task':
        return replace(candidate, input_schema=schema)
    constraints = consumer_constraints(compiler, consumer)
    proposals = [{m.producer_role: m.consumer_role} for m in candidate.role_mappings]
    outputs = {p.name for p in producer.outputs}
    for option in candidate.predicate_obligations:
        proposals.append({p: c for p, c in option['input_mapping'].items() if p in outputs})
    # Explicit relations also recall joint fresh-output mappings missed by aliases.
    from .support_request import consumer_relations
    from .support_retriever import SupportObligation, predicate_input_mapping
    probe = SupportRequest('', occurrence, producer, {}, {}, grounding_constraints=constraints)
    for predicate, _ in consumer_relations(probe, consumer, ctx):
        if len(predicate.args) < 2:
            continue
        obligation = SupportObligation('predicate', str(consumer.ref), occurrence.occurrence_id,
            predicate=predicate.predicate, predicate_args=tuple(sorted(predicate.args.items())),
            effect_domain=predicate.effect_domain.value, cardinality=predicate.cardinality,
            distinct_by=predicate.distinct_by)
        proposals.extend({p: c for p, c in m.items() if p in outputs}
                         for m in predicate_input_mapping(producer, consumer, obligation))
    previews, allowed = [], []
    seen = set()
    for mapping in proposals:
        key = json.dumps(mapping, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        inputs = {}
        for out, dest in mapping.items():
            source = input_identity_source_role(producer, out) or producer.validator_spec.get(
                'output_semantic_constraints', {}).get(out, {}).get('compatible_with_input')
            if source:
                inputs[source] = dest
        if not mapping:
            for option in candidate.predicate_obligations:
                inputs.update({p: c for p, c in option['input_mapping'].items() if p not in outputs})
        request = SupportRequest('', occurrence, producer, inputs, mapping, grounding_constraints=constraints)
        preview = preview_support_mapping_authority(request, consumer, ctx)
        previews.append(preview)
        if preview['status'] == 'proven' and preview['output_mapping'] not in allowed:
            allowed.append(preview['output_mapping'])
    proven_pairs = {(p, c) for mapping in allowed for p, c in mapping.items()}
    return replace(candidate, input_schema=schema,
        allowed_output_mappings=tuple(allowed), mapping_previews=tuple(previews),
        mapping_proven=bool(allowed), input_ready=bool(schema) and not schema.get('required'),
        role_mappings=tuple(m for m in candidate.role_mappings if (m.producer_role, m.consumer_role) in proven_pairs),
        supplied_roles=tuple(sorted({p for p, _ in proven_pairs})),
        score=float(len({c for _, c in proven_pairs}) + any(not m for m in allowed)))


def public_arguments_schema(candidates):
    """Expose common constraints, never intersect incompatible candidate schemas."""
    schemas = [c.input_schema for c in candidates]
    if schemas and all(schemas) and all(s == schemas[0] for s in schemas):
        return copy.deepcopy(schemas[0])
    properties = {}
    for schema in schemas:
        for name, value in schema.get('properties', {}).items():
            alternatives = [s['properties'][name] for s in schemas if name in s.get('properties', {})]
            properties[name] = copy.deepcopy(value) if all(v == value for v in alternatives) else {
                'description': 'Use the complete input_schema of the selected candidate.'}
    return {'type': 'object', 'properties': properties, 'additionalProperties': False}
