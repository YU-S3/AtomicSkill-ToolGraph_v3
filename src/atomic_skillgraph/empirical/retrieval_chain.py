"""Office-only, Host-observed retrieval handoffs; never inferred numeric answers."""
from copy import deepcopy

from .contracts import digest, object_schema
from .local_validation import at, reference_value

KIND = 'verified_retrieval_chain.v1'
TOOLS = {'glob', 'grep', 'read'}


def recover(task, trace, environment):
    events = trace.get('tools', [])
    source = {'task': {'goal': task.goal, 'inputs': task.inputs}, 'events': events,
              'execution': trace.get('execution', {}), 'results': trace.get('result_store', {})}
    accepted = []
    chains = []
    for event in events:
        if (event.get('state') != 'finished' or event.get('backend_invoked') is not True
                or event.get('result', {}).get('accepted') is not True or not event.get('result_id')):
            continue
        args = event.get('arguments', {})
        if event.get('name') == 'read' and isinstance(event['result'].get('data', {}).get('text'), str):
            # A pair of matching path AND offset fields is an observed handoff, not proof
            # of the model's intent. Store that distinction and verify it executably.
            producer = None
            for previous in reversed(accepted):
                if previous['name'] != 'grep': continue
                hits = previous['result'].get('data')
                if not isinstance(hits, list): continue
                matches = [i for i, hit in enumerate(hits) if isinstance(hit, dict)
                           and hit.get('path') == args.get('path') and hit.get('offset') == args.get('offset')]
                if len(matches) == 1:
                    producer, hit_index = previous, matches[0]
                    break
            if producer:
                nodes = [producer, event]
                controls = {'query': producer['arguments']['pattern'], 'hit_index': hit_index}
                roots = {'query': [0, 'pattern']}
                if 'paths' in producer['arguments']:
                    controls['paths'] = deepcopy(producer['arguments']['paths']); roots['paths'] = [0, 'paths']
                for key in args.keys() - {'path', 'offset'}:
                    controls['read_' + key] = deepcopy(args[key]); roots['read_' + key] = [1, key]
                edges = [{'from': 0, 'field': ['data', {'input': 'hit_index'}, key], 'to': 1, 'arg': key}
                         for key in ('path', 'offset')]
                glob = next((p for p in reversed(accepted[:accepted.index(producer)]) if p['name'] == 'glob'
                    and p['result'].get('data') and p['result']['data'] == producer['arguments'].get('paths')), None)
                if glob:
                    nodes.insert(0, glob)
                    controls.pop('paths'); roots.pop('paths')
                    roots = {k: [v[0]+1, v[1]] for k, v in roots.items()}
                    controls['file_pattern'] = glob['arguments']['pattern']; roots['file_pattern'] = [0, 'pattern']
                    edges = [{'from': 0, 'field': ['data'], 'to': 1, 'arg': 'paths'},
                             *[{**e, 'from': e['from']+1, 'to': e['to']+1} for e in edges]]
                contexts = []
                for node in nodes:
                    parent = next((p for p in events if p.get('event_id') == node.get('parent_event_id')), None)
                    if parent and parent.get('arguments', {}).get('source') and parent['event_id'] not in [p['event_id'] for p in contexts]:
                        contexts.append({'event_id': parent['event_id'], 'source': parent['arguments']['source']})
                actions = [{'name': n['name'], 'arguments': deepcopy(n['arguments']), 'event_id': n['event_id'],
                            'result_id': n['result_id'], 'result_hash': digest(n['result'])} for n in nodes]
                operation = {'name': 'retrieval_chain', 'arguments': controls, 'roots': roots, 'actions': actions,
                    'edges': edges, 'source_context': contexts, 'handoff_proof': 'recorded_unique_path_offset_match',
                    'search_hit_count':len(producer['result']['data']),
                    'local_goal': 'read the actual text at a selected search hit; no verified numeric extraction',
                    'output_excerpt': event['result']['data']['text'][:512]}
                chains.append({'id': 'chain:' + digest(actions), 'kind': KIND, 'source_physical_key': task.physical_key,
                    'source_trace_sha256': digest(source), 'event_ids': [n['event_id'] for n in nodes],
                    'operation': operation, 'reference': deepcopy(event['result']), 'prefix': [],
                    'result_protocol': event.get('result_protocol'), 'public_task': source['task'],
                    'environment_identity': deepcopy(environment)})
        accepted.append(event)
    return chains


def authentic(evidence, case):
    """Check every node and every original handoff against raw Host events."""
    op = evidence['operation']; results = []
    indices = []
    for action in op['actions']:
        found = [(i, e) for i, e in enumerate(case['events']) if e.get('event_id') == action['event_id']]
        if len(found) != 1: return False
        index, event = found[0]; indices.append(index)
        if (event.get('state') != 'finished' or event.get('backend_invoked') is not True
                or event.get('result', {}).get('accepted') is not True or event['name'] not in TOOLS
                or action != {'name': event['name'], 'arguments': event['arguments'], 'event_id': event['event_id'],
                              'result_id': event.get('result_id'), 'result_hash': digest(event['result'])}): return False
        results.append(event['result'])
    if indices != sorted(set(indices)) or results[-1] != evidence['reference']: return False
    if op['search_hit_count']!=len(next(r['data'] for a,r in zip(op['actions'],results) if a['name']=='grep')):return False
    for control, (node, key) in op['roots'].items():
        if op['arguments'][control] != op['actions'][node]['arguments'][key]: return False
    for edge in op['edges']:
        path = [op['arguments'][p['input']] if isinstance(p, dict) else p for p in edge['field']]
        if at(results[edge['from']], path) != op['actions'][edge['to']]['arguments'][edge['arg']]: return False
    for context in op['source_context']:
        if not any(e.get('event_id') == context['event_id'] and e.get('arguments', {}).get('source') == context['source']
                   for e in case['events']): return False
    return True


def schema(value):
    if isinstance(value, dict): return object_schema({k: schema(v) for k, v in value.items()}, list(value))
    if isinstance(value, list):
        items = list({digest(schema(v)): schema(v) for v in value}.values())
        return {'type': 'array', 'items': items[0] if len(items) == 1 else {'anyOf': items} if items else {}}
    return {'type': {str:'string', int:'integer', float:'number', bool:'boolean', type(None):'null'}[type(value)]}


def replay(operation, controls, broker):
    """Reference interpreter executes recorded actions, not an invented candidate Program."""
    results = []
    actions = []
    for index, action in enumerate(operation['actions']):
        args = deepcopy(action['arguments'])
        for key, (node, field) in operation['roots'].items():
            if node == index: args[field] = controls[key]
        for edge in operation['edges']:
            if edge['to'] != index: continue
            path = [controls[p['input']] if isinstance(p, dict) else p for p in edge['field']]
            try: args[edge['arg']] = at(results[edge['from']], path)
            except ValueError: return None, actions
        result = broker.call(action['name'], args)
        actions.append({'name': action['name'], 'arguments': args})
        if not result.get('accepted'): return None, actions
        results.append(result)
    return results[-1], actions


def calls_match(calls, actions):
    actual = [e for e in calls if e.get('backend_invoked')]
    if any(e['name'] not in TOOLS for e in actual): return False
    index = 0
    for event in actual:
        if index < len(actions) and {k:event[k] for k in ('name','arguments')} == actions[index] and event['result'].get('accepted'):
            index += 1
    return index == len(actions)


def validate_variations(system, program, binding, evidence, adapter, trial_id):
    from ..harness.simple_protocol import Broker
    from .task_context import TaskContext
    op = evidence['operation']
    aliases = {ref['path'][0]: k for k, ref in binding['input_refs'].items() if ref['kind'] == 'operation_json' and len(ref['path']) == 1}
    if set(aliases) != set(op['arguments']): return False, {'reason': 'bind_all_chain_controls'}
    reports = []
    # Equivalent regex with changed public input verifies parameter forwarding;
    # absent-hit probe checks the same code's non-success branch without any gold.
    variants=[('changed_query',{'query':op['arguments']['query']+'|(?!)'}), ('missing_hit',{'query':'(?!)'}),
              ('changed_hit',{'hit_index':(op['arguments']['hit_index']+1)%op['search_hit_count'] if op['search_hit_count']>1 else 1})]
    if 'paths' in aliases:variants.append(('empty_scope',{'paths':[]}))
    if 'file_pattern' in aliases:variants.append(('empty_file_scope',{'file_pattern':'__absent_public_file_scope__/*.txt'}))
    for label, change in variants:
        inputs = {**binding['inputs'], **{aliases[k]:v for k,v in change.items()}}
        controls = {**op['arguments'], **change}
        reference_broker = Broker(adapter, system.config['runtime']['global_action_budget'], context=TaskContext(system.config['runtime']))
        expected, actions = replay(op, controls, reference_broker)
        broker = Broker(adapter, system.config['runtime']['global_action_budget'], context=TaskContext(system.config['runtime']))
        if system.budget_governor: system.budget_governor.admit_validation(trial_id + ':' + label)
        actual = system.worker.execute(program, inputs, broker)
        if expected is None:
            passed = actual.get('status') in {'not_found','needs_input','blocked'} and calls_match(broker.events, actions)
        else:
            values = {k: reference_value({**evidence, 'reference':expected}, ref) for k, ref in binding['reference_fields'].items()}
            passed = (actual.get('status') == 'ok' and all(actual.get('outputs', {}).get(k) == v for k,v in values.items())
                      and calls_match(broker.events, actions))
        reports.append({'case':label, 'passed':passed, 'input_hash':digest(inputs), 'reference_hash':digest(expected),
            'result':actual, 'reference_tools':reference_broker.events, 'tools':broker.events})
    return all(r['passed'] for r in reports), {'cases':reports}


def verify_workflow(workflow, bank, experience):
    """Persist a composition only when its Program invocations and refs actually occurred."""
    from .contracts import normalize_workflow, validate_workflow
    value=normalize_workflow(workflow,bank);validate_workflow(value,bank=bank)
    calls=experience.get('program_calls',[])
    selected={}
    for node in value['nodes']:
        if node.get('execution_mode')!='program':raise ValueError('Composite requires verified executable nodes')
        program=bank.get(node['program_id'])
        if not program or not bank.program_eligible(program):raise ValueError('Composite node is not usable')
        matches=[c for c in calls if c.get('program_id')==program['id'] and c.get('status')=='ok' and c.get('result_id')]
        if len(matches)!=1:raise ValueError('Composite requires an unambiguous actual Program invocation')
        selected[node['id']]=matches[0]
        if program.get('result_role')!='intermediate':
            # Whole-task output needs an actual executed end-to-end graph receipt.
            raise ValueError('This retrieval composition supports intermediate results only')
    for node in value['nodes']:
        invocation=selected[node['id']]
        for key,ref in node['args'].items():
            if 'from' not in ref:
                if 'literal' in ref:expected=ref['literal']
                elif 'task' in ref:expected={**experience['task'].get('inputs',{}),'goal':experience['task']['goal']}.get(ref['task'])
                else:raise ValueError('Composite has unresolved inputs')
                if key not in invocation.get('arguments',{}) or invocation['arguments'][key]!=expected:
                    raise ValueError('Composite input differs from actual invocation')
                continue
            producer=selected[ref['from']]
            results=experience.get('result_store',{}).get('program_results',{})
            actual=results.get(producer['result_id'],{}).get('outputs',{})
            producer_node=next(n for n in value['nodes'] if n['id']==ref['from'])
            field=next((k for k,v in producer_node.get('output_aliases',{}).items() if v==ref['field']),ref['field'])
            if (ref['from'] not in node.get('after',[]) or field not in actual
                    or actual[field]!=invocation.get('arguments',{}).get(key)
                    or calls.index(producer)>=calls.index(invocation)
                    or invocation.get('input_refs',{}).get(key)!={'result_id':producer['result_id'],'path':['outputs',field]}):
                raise ValueError('Composite lacks an actual producer/consumer ResultRef')
    value['result_role']='intermediate'
    value['source_invocations']={k:v['id'] for k,v in selected.items()}
    return value
