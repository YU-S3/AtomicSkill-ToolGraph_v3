"""Static preparation dependencies, not automatic Runtime helper selection."""
from ..core.serialization import to_primitive

VERSION = 'r103.preparation-coverage.v1'


def audit_coverage(skills, *, required_composites=()):
    from ..harness.alfworld import AlfWorldAdapter
    from .preferences import static_closure
    reports, blocking = [], []
    for graph in skills.composites(mode='frozen'):
        closure = static_closure(skills, graph, AlfWorldAdapter(), 'frozen')
        routes = {r['step_id']: r for r in closure['nodes']}
        order = {step: i for i, step in enumerate(graph.control_sequence)}
        occurrences = {o.step_id: o for o in graph.occurrences}
        inputs = []
        for node in graph.occurrences:
            atomic = skills.get_atomic(node.node_ref)
            controls = atomic.validator_spec.get('input_authorization', {})
            for parameter in atomic.inputs:
                spec = node.binding_specs.get(parameter.name)
                kind, dependencies, reason = 'uncovered', [], ''
                if spec is not None and spec.kind.value == 'skill_input':
                    kind, reason = 'task', 'formal task input; concrete grounding is still required at runtime'
                elif spec is not None and spec.kind.value == 'data_flow':
                    producer = occurrences.get(spec.source_step)
                    edges = [e for e in graph.data_edges if e.source_step == spec.source_step and e.target_step == node.step_id
                        and e.source_role == spec.source_role and e.target_role == parameter.name]
                    valid = producer is not None and len(edges) == 1 and order.get(spec.source_step, -1) < order.get(node.step_id, -1)
                    source = skills.get_atomic(producer.node_ref) if producer else None
                    valid = valid and spec.source_role in {p.name for p in source.outputs}
                    valid = valid and bool(routes.get(spec.source_step, {}).get('available_count'))
                    if valid:
                        kind = ('declared_preparation_node' if any(e.predicate == 'entity.discovered_at' for e in source.effects)
                                else 'data_flow')
                        dependencies = [spec.source_step]
                        reason = 'single registered forward producer; outputs require current validator witness'
                    else:
                        reason = 'missing/ambiguous/nonforward edge, output, or executable producer'
                elif spec is None and (parameter.runtime_resolvable or parameter.name in controls or not parameter.required):
                    kind = 'agent_choice'
                    reason = 'explicit caller choice required; not static program coverage or injected default'
                else:
                    reason = 'no certified provider for the declared boundary'
                inputs.append({'step_id': node.step_id, 'atomic_ref': str(atomic.ref), 'role': parameter.name,
                    'occurrence_id': node.occurrence_id, 'semantic_type': parameter.semantic_type,
                    'required': parameter.required, 'required_resolution': parameter.required_resolution,
                    'provider_kind': kind, 'binding': to_primitive(spec), 'entry_dependencies': dependencies,
                    'executable_implementations': routes.get(node.step_id, {}).get('available_implementations', []),
                    'provider_refs': [str(occurrences[s].node_ref) for s in dependencies],
                    'exact_role_mapping': {spec.source_role: parameter.name} if spec is not None else {},
                    'current_evidence_required': parameter.required_resolution != 'semantic',
                    'program_scope': ('explicit_public_relation' if kind == 'declared_preparation_node'
                        else 'native_observation' if kind == 'agent_choice' else 'other_declared'),
                    'caller_authorization': to_primitive(controls.get(parameter.name)),
                    'evidence_requirement': 'current runtime grounding, never a static witness',
                    'cycle': bool(spec is not None and spec.kind.value == 'data_flow'
                        and order.get(spec.source_step, -1) >= order.get(node.step_id, -1)),
                    'validation': reason})
        # Dependency closure follows actual roles, never equality of their names.
        by_step = {n.step_id: [r for r in inputs if r['step_id'] == n.step_id and r['required']]
                   for n in graph.occurrences}
        def dependencies(step, path=()):
            if step in path:
                return [{'cycle': list(path) + [step]}]
            result = []
            for row in by_step.get(step, []):
                result.append({'step_id': step, 'role': row['role'], 'resolution': row['required_resolution'],
                    'provider_kind': row['provider_kind']})
                for producer in row['entry_dependencies']:
                    result.extend(dependencies(producer, (*path, step)))
            return result
        for row in inputs:
            row['preparation_dependency_closure'] = [d for step in row['entry_dependencies'] for d in dependencies(step)]
            row['cycle_detected'] = row['cycle'] or any('cycle' in d for d in row['preparation_dependency_closure'])
            row['status'] = 'uncovered' if row['provider_kind'] == 'uncovered' or row['cycle_detected'] else (
                'explicit_agent_boundary' if row['provider_kind'] == 'agent_choice' else 'declared_provider')
            row['validation_ref'] = f"{graph.ref}#{row['step_id']}.{row['role']}"
        gaps = [r for r in inputs if r['required'] and r['provider_kind'] == 'uncovered']
        if str(graph.ref) in required_composites:
            blocking.extend({'composite_ref': str(graph.ref), **row} for row in gaps)
            if not closure['program_static_closure']:
                blocking.append({'composite_ref': str(graph.ref), 'reason': 'missing program closure'})
        reports.append({'composite_ref': str(graph.ref), 'status': str(graph.status.value),
            'task_contract': to_primitive(graph.goal_contract),
            'nodes': [{'step_id': n.step_id, 'atomic_ref': str(n.node_ref),
                       'outputs': to_primitive(skills.get_atomic(n.node_ref).outputs),
                       'preconditions': to_primitive(skills.get_atomic(n.node_ref).preconditions)} for n in graph.occurrences],
            'inputs': inputs, 'uncovered': gaps, 'program_closure': closure})
    absent = set(required_composites) - {r['composite_ref'] for r in reports}
    blocking.extend({'composite_ref': ref, 'reason': 'required graph not deployed'} for ref in sorted(absent))
    return {'version': VERSION, 'passed': not blocking, 'blocking_gaps': blocking,
        'all_deployed_composites': reports, 'agent_choice_is_program_coverage': False}
