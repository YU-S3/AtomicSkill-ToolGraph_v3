"""Offline authorship of the enumerated old-first jobs, never Runtime policy.

The edit plan supplies all asset identities, occurrences and data-flow edges.
Programs use public selectors only; no episode values or search routes exist here.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from ..core.bindings import BindingExpression, ToolBinding
from ..core.contracts import (AbstractAtomicSkill, ImplementationAtom, ToolAsset,
    CompositeOccurrence, ParameterSpec, SemanticPredicate)
from ..core.edges import GraphEdge
from ..core.refs import SkillRef, ToolRef
from ..core.serialization import to_primitive
from ..core.status import SkillStatus
from ..tooling.entry_contract import parameter_schema
from ..tooling.ir import walk_program_nodes
from .asset_revision import _fix_navigation
from .release_protocol import OLDFIRST_PROTOCOL_VERSION, ReleaseError


def binding(role):
    return BindingExpression('skill_input', source_role=role)


def predicate(name, **roles):
    return SemanticPredicate(name, {k: binding(v) for k, v in roles.items()},
        effect_domain='evidence' if name == 'entity.discovered_at' else 'world')


def parameter(name, semantic_type='entity', resolution='concrete'):
    return ParameterSpec(name, semantic_type, runtime_resolvable=True, required_resolution=resolution)


def metadata(job):
    return {'authoring_source': 'oldfirst_edit_plan', 'release_revision': OLDFIRST_PROTOCOL_VERSION,
        'job_id': job['job_id'], 'source_ref': job.get('source_ref', ''),
        'historical_execution_claimed': False}


def revise_atomic(job, assets, program_job):
    source = assets.get(job.get('source_ref'))
    if job.get('operation') == 'version_existing_discovery_contract':
        if source is None or not job.get('preserve_inputs_outputs_and_effects'):
            raise ReleaseError('discovery delta requires an immutable source contract')
        atomic = copy.deepcopy(source)
        atomic.ref = SkillRef.parse(job['target_ref'])
        atomic.summary = 'Find a matching entity with a current public location witness in the caller supplied scope'
        atomic.guideline = {'steps': [
            'Inspect only the caller supplied ordered locations; open only with authorization.',
            'Return an entity and location from the same current explicit public relation; takeability is not required.'],
            'notes': ['A closed or unparsed scope is not a complete inspection. Scope exhaustion is not global absence.']}
        atomic.metadata = metadata(job)
        atomic.status = SkillStatus.DRAFT
        return atomic
    if program_job.get('query_input_role'):
        query = program_job['query_input_role']
        inputs = [parameter(query, resolution='semantic'), parameter('locations', 'list', 'semantic'),
                  parameter('allow_open', 'bool', 'semantic')]
        outputs = [parameter(r) for r in program_job['output_roles']]
        effect = predicate('entity.discovered_at', entity='entity', location='location')
        spec = {'validator_id': 'harness_atomic_effect', 'identity_strict': True,
            'output_identity': [], 'output_derivations': {r: {'kind': 'effect_witness',
                'predicate': effect.predicate, 'argument_role': r} for r in ('entity', 'location')},
            'output_semantic_constraints': {'entity': {'compatible_with_input': query}},
            'input_authorization': {'allow_open': {'kind': 'caller_boolean'}, 'locations': {
                'kind': 'ordered_entity_scope', 'element_semantic_type': 'entity',
                'min_items': 1, 'max_items': 8, 'unique_items': True}}}
        return AbstractAtomicSkill(SkillRef.parse(job['target_ref']), 'Find a matching object in the caller supplied scope',
            inputs, outputs, [], [effect], spec, [], {'steps': [
                'Inspect only the caller supplied ordered locations; open only with authorization.',
                'Return a publicly takeable matching entity and its jointly witnessed location without taking it.'],
                'notes': ['An exhausted scope is not evidence of global absence.']}, metadata(job))
    if source is None:
        raise ReleaseError('observation revision requires its original Atomic')
    atomic = copy.deepcopy(source)
    atomic.ref = SkillRef.parse(job['target_ref'])
    atomic.inputs = [parameter('object'), parameter('light_source'), parameter('location')]
    atomic.preconditions = [predicate('agent.holds', object='object')]
    # All original effects and output roles are retained. Entry changes are versioned.
    identities = {p.name: ('light_source' if p.name == 'light' else 'object') for p in atomic.outputs}
    atomic.validator_spec = {'validator_id': 'harness_atomic_effect', 'identity_strict': True,
        'output_identity': [{'input_role': src, 'output_role': out} for out, src in identities.items()],
        'output_derivations': {out: {'kind': 'input_identity', 'input_role': src} for out, src in identities.items()},
        'output_semantic_constraints': {out: {'compatible_with_input': src} for out, src in identities.items()}}
    atomic.guideline = {'steps': ['Preserve the held object identity.',
        'Reach the explicitly supplied light location and use the supplied light when available.',
        'Verify observation of that same object with that light.'], 'notes': []}
    atomic.metadata = metadata(job)
    atomic.status = SkillStatus.DRAFT
    return atomic


def arg(role, local=False):
    return {'kind': 'local_variable' if local else 'skill_input', 'source_role': role}


def value(role, local=False):
    return {'source': 'local_variable' if local else 'tool_input', 'field': role}


def selector(action, role, anchor, local=False):
    return {'source': 'action_catalog', 'where': {'action_type': action, 'argument_role': role,
        'semantic_compatible_with': {**value(anchor, local), 'semantic_type': 'entity'}},
        'project': {'kind': 'argument', 'role': role}, 'distinct': True}


def conditional(node, match, body):
    return {'op': 'IF', 'node_id': node, 'condition': {'op': 'exists', 'match': match},
            'then_branch': body, 'else_branch': []}


def action(node, kind, arguments, effect):
    return {'op': 'ACTION', 'node_id': node, 'action_type': kind,
            'argument_mapping': arguments, 'expected_effects': [effect]}


def navigation(role, local=False):
    return action('navigate', 'GO_TO', {'destination': arg(role, local)},
        {'predicate': 'agent.at_location', 'args': {'location': arg(role, local)}, 'effect_domain': 'world'})


def author_program(job, atomic):
    query = job.get('query_input_role')
    if query:
        found = selector('TAKE', 'object', query)
        found_source = {'source': 'semantic_evidence', 'where': {
            'predicate': 'entity.discovered_at', 'argument_role': 'entity',
            'semantic_compatible_with': {**value('found', True), 'semantic_type': 'entity'}},
            'project': {'kind': 'argument', 'role': 'location'}, 'distinct': True}
        if job.get('selector_policy') == 'current_joint_public_discovery':
            found = {'source': 'semantic_evidence', 'where': {
                'predicate': 'entity.discovered_at', 'argument_role': 'entity',
                'semantic_compatible_with': {**value(query), 'semantic_type': 'entity'},
                'location': value('searched_location', True)},
                'project': {'kind': 'argument', 'role': 'entity'}, 'distinct': True}
            found_source['where']['location'] = value('searched_location', True)
        open_node = conditional('open_when_offered', selector('OPEN', 'object', 'searched_location', True),
            [action('open', 'OPEN', {'object': arg('searched_location', True)},
                {'predicate': 'container.open', 'args': {'container': arg('searched_location', True)}, 'effect_domain': 'world'})])
        body = [navigation('searched_location', True),
            {'op': 'IF', 'node_id': 'authorized_open', 'condition': {'op': 'equals',
                **value('allow_open'), 'value': True}, 'then_branch': [open_node], 'else_branch': []},
            conditional('matching_object_available', found, [{'op': 'FOR_EACH', 'node_id': 'match_object', 'collection_source': found,
                'iteration_variable': 'found', 'max_iterations': 1, 'body': [
                    {'op': 'RETURN', 'node_id': 'found_result', 'output_sources': {
                        'entity': value('found', True), 'location': found_source}}]}])]
        _fix_navigation(body)
        program = [{'op': 'FOR_EACH', 'node_id': 'caller_scope', 'collection_source': value('locations'),
            'iteration_variable': 'searched_location', 'max_iterations': 8, 'body': body}]
        sources = {'entity': value('found', True), 'location': found_source}
    else:
        destination = 'destination' if any(p.name == 'destination' for p in atomic.inputs) else 'location'
        sources = {out: value(d['input_role']) for out, d in atomic.validator_spec['output_derivations'].items()
                   if d['kind'] == 'input_identity'}
        program = [navigation(destination)]
        if not any(p.predicate == 'light.on' for p in atomic.preconditions):
            program.append(conditional('use_when_offered', selector('USE', 'object', 'light_source'), [
                action('use_light', 'USE', {'object': arg('light_source')}, to_primitive(
                    predicate('object.observed_with', object='object', light='light_source')))]))
        program.append({'op': 'RETURN', 'node_id': 'observed_result', 'output_sources': sources})
        _fix_navigation(program)
    signature = parameter_schema(atomic.inputs)
    if query:
        signature['properties']['locations'].update(items={'type': 'string'}, minItems=1, maxItems=8, uniqueItems=True)
    return ToolAsset(ToolRef.parse(job['tool_ref']), atomic.summary, signature,
        {'output_schema': parameter_schema(atomic.outputs), 'entry_contract': {
            'conditions': to_primitive(atomic.preconditions), 'grounding_constraints': []}},
        'tool_ir_v1', {'schema_version': 1, 'max_actions': 16 if query else 2, 'program': program,
            'final_effects': to_primitive(atomic.effects), 'path_expectations': [],
            'evidence_outputs': [{'role': r, **s} for r, s in sources.items()]}, [],
        {'allowed_action_types': sorted({n['action_type'] for n in walk_program_nodes(program) if n['op'] == 'ACTION'}),
         'zero_llm': True, 'terminal_interruptible': True},
        {'source': 'oldfirst_edit_plan', 'source_ref': job.get('source_tool_ref', ''), 'historical_execution_claimed': False},
        metadata(job))


def author_implementation(job, atomic, tool=None):
    if tool is None:
        bindings = [ToolBinding(ToolRef.parse(b['tool_ref']), b['role'],
            {r: BindingExpression.from_dict(e) for r, e in b['parameter_mapping'].items()}, b['order'])
            for b in job['tool_bindings']]
        outputs = copy.deepcopy(job['output_mapping'])
    else:
        bindings = [ToolBinding(tool.ref, 'primary', {p.name: binding(p.name) for p in atomic.inputs}, 0)]
        outputs = {p.name: BindingExpression('tool_output', source_role=p.name, source_step='primary') for p in atomic.outputs}
    return ImplementationAtom(SkillRef.parse(job['implementation_ref']), atomic.ref, bindings, [],
        {'mode': 'serial', 'output_mapping': outputs}, {'harness_profiles': ['alfworld_v3']}, {}, metadata=metadata(job))


def revise_graph(job, source):
    graph = copy.deepcopy(source)
    if job['action'] != 'revise':
        return graph
    graph.ref = SkillRef.parse(job['target_ref'])
    graph.occurrences, graph.data_edges, graph.dependency_edges = [], [], []
    for node in job['target_nodes']:
        bindings = {r: BindingExpression.from_dict(b) for r, b in node['binding_specs'].items()}
        step = node['step_id']
        graph.occurrences.append(CompositeOccurrence(step, step, SkillRef.parse(node['atomic_ref']), bindings))
        for role, b in bindings.items():
            if b.kind.value == 'data_flow':
                edge_id = f'{graph.ref.logical_id}_{step}_{role}'
                graph.data_edges.append(GraphEdge(edge_id, 'data_flow', b.source_step, step, b.source_role,
                    role, 'existing_active', edge_id))
    graph.control_sequence = [o.step_id for o in graph.occurrences]
    graph.metadata = {'authoring_source': 'oldfirst_edit_plan', 'source_ref': str(source.ref),
        'release_revision': OLDFIRST_PROTOCOL_VERSION, 'historical_execution_claimed': False,
        'external_inputs': {o.step_id: {r: to_primitive(b) for r, b in o.binding_specs.items()
            if b.kind.value == 'skill_input'} for o in graph.occurrences}}
    graph.insight = {}
    graph.validator_spec = {'canonical_sequence': True, 'self_sufficiency_required': True, 'task_contract_covered': False}
    graph.status = SkillStatus.DRAFT
    return graph
