"""Optional V5 Train pairs with task-local references and separate exposure/use counters."""
from copy import deepcopy

from ..agents.protocol import SchemaValidationError, validate_schema_instance
from ..core.errors import BudgetExhausted
from ..empirical.checkpoint import save_json
from ..empirical.contracts import PublicTask, digest, validate_workflow
from ..empirical.office_operation_mining import read, at
from ..empirical.office_program_probe import runtime_view
from ..empirical.office_trace_program_induction import RequestStore, ModelOutputError
from ..empirical.program_worker import ProgramWorker
from ..empirical.task_context import TaskContext
from ..harness.benchmarks import OfficeAdapter, score_answer
from ..harness.simple_protocol import Broker
from .office_v4_consumption import PROMPT as V4_PROMPT, snapshot, restore, execute_action as execute_v4_action
from .run_office_trace_v5_pilot import checked, PhaseGovernor, provider_for, verify_candidates, report

PROMPT = V4_PROMPT + '''\nV5: optionally include used_results:[{result_id,path}] to explicitly declare use
of an actually exposed Program output. This is a declaration, separate from Host-bound downstream use.
Use only exact task-local public references. Missing required Program inputs return needs_input without
executing the Worker; continue native retrieval. Candidate status is not full-task correctness.'''


def execute_action(action, broker, visible, worker, state):
    if not isinstance(action, dict): raise ValueError('Runtime action must be an object')
    kind = action.get('action')
    allowed = {'native': {'action', 'name', 'arguments', 'input_refs', 'used_results'},
               'program': {'action', 'program_id', 'input_refs', 'used_results'},
               'finish': {'action', 'answer', 'answer_ref', 'used_results'}}
    if not isinstance(kind, str) or kind not in allowed or set(action) - allowed[kind]: raise ValueError('Invalid Runtime action fields')
    if kind == 'native' and (not isinstance(action.get('name'), str) or not isinstance(action.get('arguments', {}), dict)):
        raise ValueError('Native action needs name:string and arguments:object')
    if kind == 'program' and not isinstance(action.get('program_id'), str): raise ValueError('program_id must be a string')
    refs = action.get('input_refs', {})
    if not isinstance(refs, dict): raise ValueError('input_refs must map parameter names to ResultRefs')
    declarations = action.get('used_results', [])
    if not isinstance(declarations, list): raise ValueError('used_results must be an array')
    for ref in [*refs.values(), *declarations, *([action['answer_ref']] if 'answer_ref' in action else [])]:
        if not isinstance(ref, dict) or set(ref) != {'result_id', 'path'}: raise ValueError('Exact task-local result_id/path required')
        broker.context.resolve_public(ref)
    for ref in declarations:
        if ref['result_id'] not in state.get('exposed_program_results', []) or ref['path'][:1] != ['outputs'] or len(ref['path']) < 2:
            raise ValueError('Usage declaration must name an actually exposed Program output field')
    if kind == 'program':
        selected = next((b for b in visible if b['id'] == action.get('program_id')), None)
        if selected is None: raise ValueError('Unknown Program')
        values = {k: broker.context.resolve_public(ref) for k, ref in refs.items()}
        try: validate_schema_instance(values, selected['program']['input_schema'])
        except SchemaValidationError as exc:
            return {'kind': 'program', 'program_id': selected['id'], 'accepted': False, 'status': 'needs_input',
                    'diagnostic': {'code': 'missing_runtime_precondition', 'field_path': exc.path, 'detail': str(exc)},
                    'program_invoked': False, 'resolved_refs': refs}
    record = execute_v4_action({k:v for k,v in action.items() if k != 'used_results'}, broker, visible, worker, state)
    record['used_results_declared'] = declarations
    record['program_invoked'] = kind == 'program'
    if kind == 'native': record['name'] = action['name']
    return record


def consumption_metrics(actions, exposures, programs):
    """Count only ordered exact-value producer→consumer receipts from this episode."""
    producers, uses, declared, read_ids = {}, [], [], set()
    nodes, seen, edges = [], {}, []
    for action in actions:
        accepted = action.get('accepted') is True
        valid_refs = {}
        for name, ref in action.get('resolved_refs', {}).items():
            producer = producers.get(ref.get('result_id'))
            if not accepted or not producer or ref.get('path', [])[:1] != ['outputs'] or len(ref['path']) < 2: continue
            try: actual = at(producer['result'], ref['path'])
            except ValueError: continue
            bound = action.get('bound_inputs', action.get('arguments', {})).get(name)
            if action.get('kind') == 'finish': bound = action.get('prediction')
            if (str(actual) == bound if action.get('kind') == 'finish' else actual == bound):
                valid_refs[name] = ref; uses.append({'producer_result_id': ref['result_id'], 'consumer_result_id': action.get('result_id'), 'input': name, 'ref': ref})
        for ref in action.get('used_results_declared', []):
            if accepted and ref['result_id'] in producers: declared.append(ref)
        if accepted and action.get('name') == 'read_result' and action.get('arguments', {}).get('result_id') in producers:
            read_ids.add(action['arguments']['result_id'])
        if accepted and action.get('kind') == 'program' and action.get('status') == 'ok' and action.get('result_id'):
            program = programs[action['program_id']]
            validate_schema_instance(action['bound_inputs'], program['input_schema'])
            validate_schema_instance(action['result']['outputs'], program['output_schema'])
            node = {'id': 'observed_' + str(len(nodes)), 'execution_mode': 'program', 'program_id': action['program_id'],
                    'purpose': 'Observed intermediate operation', 'args': {}, 'after': []}
            for name in action['bound_inputs']:
                ref = action.get('resolved_refs', {}).get(name)
                if name in valid_refs and len(ref['path']) == 2 and ref['result_id'] in seen:
                    parent = seen[ref['result_id']]; node['args'][name] = {'from': parent, 'field': ref['path'][1]}
                    node['after'].append(parent); edges.append({'producer': parent, 'consumer': node['id'], 'ref': ref})
                else: node['args'][name] = {'unresolved': {'public_result_ref': ref}}
            node['after'] = sorted(set(node['after'])); nodes.append(node); seen[action['result_id']] = node['id']
            producers[action['result_id']] = action
    workflow = None
    if edges:
        workflow = {'interface_version': 'empirical.workflow.v2', 'goal': 'Observed bounded Program dataflow',
                    'nodes': nodes, 'outputs': {}, 'partial': True, 'observed_edges': edges, 'formal_publication': False}
        validate_workflow(workflow, known_programs=programs, known_skills={})
    return {'program_invoked': sum(a.get('program_invoked', False) for a in actions),
        'result_exposed': len(set(exposures) & set(producers)), 'result_read': len(read_ids),
        'result_used_declared': len(declared), 'downstream_bound': len(uses), 'outputs_consumed': bool(uses),
        'uses': uses, 'partial_workflow': workflow}


def consume(root):
    plan, config = checked(root, corpus=True); governor = PhaseGovernor(root, 'consume')
    candidates = [b for b in verify_candidates(root) if b['status'] == 'transfer_checked']
    if not candidates:
        save_json(root / 'consumption_status.json', {'status': 'completed_no_qualified_candidate', 'http': 0})
        return report(root)
    sources = read(root / 'source_records.json'); cap = plan['config']['paired_arm_limits']
    for task_id in plan['config']['transfer_tasks']:
        task = next(s['task'] for s in sources if s['task']['task_id'] == task_id)
        for arm in ('without_program', 'with_program'):
            folder = root / 'consumption' / task_id.replace(':', '_') / arm
            if (folder / 'trace.json').exists(): continue
            visible = candidates if arm == 'with_program' else []
            cfg = deepcopy(config); cfg['experiment']['output_dir'] = str(folder)
            adapter = OfficeAdapter({task_id: {}}, cfg); adapter.reset(PublicTask(**task))
            state_path = folder / 'state.json'
            if state_path.exists(): state = read(state_path); context = restore(state['context'])
            else:
                context = TaskContext(cfg['runtime'])
                rid = context.register('public_task', {'data': {'goal': task['goal'], **task['inputs']}})
                state = {'actions': [], 'events': [], 'producers': [], 'step': 0, 'finished': False,
                         'exposed_program_results': [], 'public_task_result_id': rid, 'context': snapshot(context)}
                save_json(state_path, state)
            broker = Broker(adapter, cap['native_calls'], context=context); broker.events = state['events']
            scope = f"{plan['run_id']}:{task_id}:{arm}"
            client = RequestStore(folder / 'requests', provider_for(config, 'runtime', governor), phase='consume',
                wire_max_bytes=plan['config']['wire_max_bytes'], attribution={'parent_scope': scope, 'parent_token_limit': cap['tokens'],
                    'episode_scope': scope, 'episode_request_limit': cap['http']})
            try:
                while not state['finished'] and state['step'] < cap['http']:
                    step = state['step']; intent = folder / f'action_{step}.intent.json'
                    if intent.exists(): raise RuntimeError('Runtime action started without receipt; no silent replay')
                    view = runtime_view(context, visible)
                    exposed = {row['result_id'] for row in view['actual_results']}
                    newly_exposed = {p['result_id'] for p in state['producers'] if p['result_id'] in exposed}
                    data = {'task': task, 'public_task_result_id': state['public_task_result_id'], 'tools': broker.available_tools(),
                            'state': view, 'remaining_calls': broker.remaining_calls(),
                            'recent_actions': [{k: a.get(k) for k in ('kind', 'accepted', 'status', 'diagnostic', 'error', 'result_id', 'controls_result_id')}
                                               for a in state['actions'][-4:]]}
                    try: action, _ = client.ask(str(step), PROMPT, data)
                    except BudgetExhausted: state.update(finished=True, terminal_reason='budget_exhausted'); break
                    except ModelOutputError as exc:
                        state['actions'].append({'kind': 'model_error', 'accepted': False, 'error': str(exc)})
                        state['step'] += 1; state['context'] = snapshot(context); save_json(state_path, state); continue
                    state['exposed_program_results'] = sorted(set(state['exposed_program_results']) | newly_exposed)
                    save_json(intent, {'action': action, 'context_hash': digest(snapshot(context))})
                    try: action_record = execute_action(action, broker, visible, ProgramWorker({'record_execution_receipt': True}), state)
                    except ValueError as exc: action_record = {'kind': 'invalid_action', 'accepted': False, 'error': str(exc)}
                    state['actions'].append(action_record); state['step'] += 1
                    state.update(events=broker.events, context=snapshot(context)); save_json(state_path, state)
                labels = read(read(Path(plan['source']) / 'config.json')['harness']['evaluator_records'])
                if isinstance(labels, list): labels = {r['task_id']: r for r in labels}
                label = labels.get(task_id, labels.get(task_id.split(':')[-1]))
                if label is None: raise RuntimeError('Private Train scoring record missing')
                score = score_answer('officeqa', state.get('prediction', ''), label)
                metrics = consumption_metrics(state['actions'], state['exposed_program_results'], {b['id']: b['program'] for b in visible})
                own = [a for a in governor.state['attempts'].values() if a['context'].get('parent_scope') == scope]
                trace = {'task': task, 'arm': arm, 'visible_program_ids': [b['id'] for b in visible], 'actions': state['actions'],
                    'events': broker.events, 'prediction': state.get('prediction'), 'score': score, 'task_correct': score['hard'],
                    'metrics': metrics, 'terminal_reason': state.get('terminal_reason', 'request_limit'),
                    'tokens': sum(a.get('accounted_tokens', a['reserved_tokens']) for a in own), 'http': len(own),
                    'limits': cap, 'prompt_sha256': digest(PROMPT), 'provider': client.provider.snapshot()}
                save_json(folder / 'trace.json', trace)
            finally: adapter.close()
    return report(root)
