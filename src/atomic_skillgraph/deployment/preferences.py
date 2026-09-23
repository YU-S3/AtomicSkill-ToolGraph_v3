"""Frozen deployment-view preferences; never mutate historical quality or status."""
import json
from .release_protocol import OLDFIRST_PROTOCOL_VERSION, ReleaseError, sha


def load_preferences(store):
    path = store.data_dir / 'deployment_preferences.json'
    if not path.is_file():
        return {}
    value = json.loads(path.read_text())
    plan = store.data_dir / 'edit_plan.lock.json'
    if value.get('protocol_version') != OLDFIRST_PROTOCOL_VERSION or value.get('edit_plan_hash') != sha(plan):
        raise ReleaseError('deployment preferences do not match the locked edit plan')
    return value


def resolve_implementation(skills, ref, atomic_ref):
    prefs = load_preferences(skills.store)
    aliases = prefs.get('canonical_implementation_aliases', {})
    target = aliases.get(str(ref), str(ref))
    result = skills.get_implementation(target)
    # Cross-Atomic identity requires a separate role-rebinding compiler. It is
    # not safe to simply execute a different Atomic's I under this occurrence.
    if result.abstract_ref != atomic_ref:
        return skills.get_implementation(ref)
    return result


def implementation_priority(skills, ref, ready):
    rows = load_preferences(skills.store).get('preferred_implementations', [])
    return max((r['priority'] for r in rows if r['implementation_ref'] == str(ref)
        and r['applicable_condition'] in ('any', 'ready' if ready else 'not_ready')), default=0)


def verify_preferences(skills):
    """Validate every deployment override against the locked plan and real I proof."""
    from ..evolution.identity_matching import match_implementation, verify_implementation_proof, match_tool, verify_tool_proof
    from ..knowledge.tool_registry import ToolRegistry
    prefs = load_preferences(skills.store)
    if not prefs:
        return
    plan = json.loads((skills.store.data_dir / 'edit_plan.lock.json').read_text())
    tools = ToolRegistry(skills.store, skills.database)
    expected_aliases = {j['source_implementation_ref']: j['target_implementation_ref'] for j in plan['identity_rebindings']}
    expected_tools = {j['alias_tool_ref']: j['canonical_tool_ref'] for j in plan['merge_groups']}
    if (prefs['canonical_implementation_aliases'] != expected_aliases
            or prefs['canonical_tool_aliases'] != expected_tools):
        raise ReleaseError('deployment aliases differ from the edit plan')
    for source, target in expected_tools.items():
        left, right = tools.get(source), tools.get(target)
        proof = match_tool(left, right)
        if proof.status != 'exact' or not verify_tool_proof(left, right, proof.proof):
            raise ReleaseError('deployment Tool alias has no valid proof')
    for source, target in expected_aliases.items():
        left, right = skills.get_implementation(source), skills.get_implementation(target)
        if left.abstract_ref != right.abstract_ref:
            raise ReleaseError('deployment I alias crosses Atomic authority')
        args = dict(source_atomic=skills.get_atomic(left.abstract_ref), target_atomic=skills.get_atomic(right.abstract_ref),
            source_tools={str(b.tool_ref): tools.get(b.tool_ref) for b in left.tool_bindings},
            target_tools={str(b.tool_ref): tools.get(b.tool_ref) for b in right.tool_bindings})
        proof = match_implementation(left, right, **args)
        if proof.status != 'exact' or not verify_implementation_proof(left, right, proof.proof, **args):
            raise ReleaseError('deployment I alias has no valid proof')
    from .oldfirst_plan import program_jobs
    expected = [{'atomic_ref': j['atomic_ref'], 'implementation_ref': j['implementation_ref'],
        'applicable_condition': 'ready' if j.get('route_preference') else 'any',
        'priority': 200 if j.get('route_preference') else 100} for j in program_jobs(plan)]
    if prefs['preferred_implementations'] != expected:
        raise ReleaseError('deployment I preference differs from the edit plan')
    for item in expected:
        if str(skills.get_implementation(item['implementation_ref']).abstract_ref) != item['atomic_ref']:
            raise ReleaseError('deployment I preference crosses Atomic authority')
    expected_graphs = [{'composite_ref': j['target_ref'], 'priority': 100} for j in plan['workflow_targets']]
    if plan.get('derived_revision_jobs'):
        expected_graphs.append({'composite_ref': plan['deployment_preferences_patch']['prefer_workflow_ref'], 'priority': 200})
    if prefs['preferred_composites'] != expected_graphs:
        raise ReleaseError('deployment graph preference differs from the edit plan')
    if prefs['blocked_equivalent_groups'] != [[j['alias_tool_ref'], j['canonical_tool_ref']]
            for j in plan['merge_groups'] if j['action'] == 'blocked_equivalence_group']:
        raise ReleaseError('deployment failure group was changed')


def static_closure(skills, graph, harness, mode):
    from ..knowledge.tool_registry import ToolRegistry
    from ..runtime.invocation_compiler import InvocationCompiler
    compiler = InvocationCompiler(skills, ToolRegistry(skills.store, skills.database), harness, mode=mode)
    nodes = []
    for occurrence in graph.occurrences:
        atomic = skills.get_atomic(occurrence.node_ref)
        available, rejected = [], []
        for impl in skills.implementations_for(atomic.ref, mode=mode):
            try:
                tools = [compiler.tools.get(b.tool_ref) for b in impl.tool_bindings]
                compiled = compiler.compile(atomic, impl, tools, {})
                failure = compiler._compatibility_failure(type('StaticInvocation', (), {
                    'implementation': impl, 'tools': tools})())
                if failure:
                    raise ValueError(failure.message)
                available.append(str(compiled.implementation_ref))
            except (KeyError, TypeError, ValueError) as exc:
                rejected.append({'implementation_ref': str(impl.ref), 'reason': str(exc)})
        nodes.append({'step_id': occurrence.step_id, 'atomic_ref': str(atomic.ref),
                      'available_implementations': available, 'available_count': len(available), 'rejected': rejected})
    return {'program_static_closure': bool(nodes) and all(n['available_count'] for n in nodes), 'nodes': nodes}
