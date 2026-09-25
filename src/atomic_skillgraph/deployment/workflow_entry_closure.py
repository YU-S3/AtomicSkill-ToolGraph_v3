"""Static input provenance audit, separate from task-goal completion authority."""
from ..core.bindings import resolution_satisfies
from ..core.semantic_types import semantic_types_compatible
from .preferences import static_closure


def audit_workflow(skills, graph, harness, *, mode='frozen'):
    closure = static_closure(skills,graph,harness,mode)
    routes = {r['step_id']:r for r in closure['nodes']}
    order = {step:index for index,step in enumerate(graph.control_sequence)}
    occurrences = {o.step_id:o for o in graph.occurrences}
    sequence_valid = len(order) == len(graph.control_sequence) == len(occurrences) and set(order) == set(occurrences)
    rows = []
    unresolved = agent_required = 0
    for step in graph.control_sequence:
        if step not in occurrences:
            continue
        node = occurrences[step]
        atomic = skills.get_atomic(node.node_ref)
        sources, diagnostics = {}, {}
        for parameter in atomic.inputs:
            if not parameter.required:
                continue
            role = parameter.name
            binding = node.binding_specs.get(role)
            kind, reason = 'unresolved','No validated input provider'
            if order[step] == 0 and binding is None and parameter.runtime_resolvable:
                kind,reason = 'bootstrap_declared','Caller supplies this entry input once; not an environment fact'
            elif binding is not None and binding.kind.value == 'data_flow':
                producer = occurrences.get(binding.source_step)
                edges = [e for e in graph.data_edges if e.target_step==step and e.target_role==role]
                output = next((p for p in skills.get_atomic(producer.node_ref).outputs if p.name==binding.source_role),None) if producer else None
                valid = (sequence_valid and producer is not None and order[binding.source_step] < order[step]
                    and len(edges)==1 and edges[0].source_step==binding.source_step and edges[0].source_role==binding.source_role
                    and edges[0].origin != 'planner_proposed' and output is not None and output.required
                    and semantic_types_compatible(output.semantic_type,parameter.semantic_type)
                    and resolution_satisfies(output.required_resolution,parameter.required_resolution))
                if valid:
                    kind,reason = 'data_flow','Single forward, required, type/resolution-compatible output'
                else:
                    reason = 'Missing, optional, incompatible, cyclic or unvalidated producer'
            elif binding is not None and binding.kind.value == 'constant':
                value = binding.constant
                # Concrete episode identities are never portable constants.
                if parameter.required_resolution == 'semantic' and (value is None or isinstance(value,(str,int,float,bool))):
                    kind,reason = 'portable_constant','Semantic literal; no concrete resolution claim'
            elif parameter.runtime_resolvable or binding is not None and binding.kind.value=='skill_input':
                kind,reason = 'runtime_agent_required','Task-specific value needs current Runtime grounding'
            sources[role],diagnostics[role] = kind,reason
            if order[step] > 0:
                unresolved += kind == 'unresolved'
                agent_required += kind == 'runtime_agent_required'
        route = routes.get(step,{})
        available = route.get('available_implementations',[])
        rows.append({'step_id':step,'atomic_ref':str(atomic.ref),'input_sources':sources,
            'input_diagnostics':diagnostics,'available_implementations':available,
            'zero_llm_executable':bool(route.get('available_count'))})
    return {'composite_ref':str(graph.ref),'program_static_closure':bool(closure['program_static_closure']),
        'post_bootstrap_program_closed':bool(sequence_valid and rows and closure['program_static_closure']
            and not unresolved and not agent_required
            and all(v!='unresolved' for row in rows for v in row['input_sources'].values())),
        'downstream_runtime_agent_required_count':agent_required,'downstream_unresolved_count':unresolved,
        'nodes':rows,'goal_completion_authority_not_inferred':True}
