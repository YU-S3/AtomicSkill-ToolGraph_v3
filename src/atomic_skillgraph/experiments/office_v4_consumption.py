"""Optional pre-registered Train pairs; differences are limited to candidate visibility."""
from copy import deepcopy

from ..agents.protocol import validate_schema_instance
from ..core.errors import BudgetExhausted
from ..empirical.checkpoint import save_json
from ..empirical.contracts import PublicTask, digest
from ..empirical.office_operation_mining import read
from ..empirical.office_program_probe import consumed, partial_workflow, runtime_view
from ..empirical.office_trace_program_induction import RequestStore, ModelOutputError
from ..empirical.program_worker import ProgramWorker
from ..empirical.task_context import TaskContext
from ..harness.benchmarks import OfficeAdapter
from ..harness.simple_protocol import Broker
from .run_office_trace_v4_pilot import checked, PhaseGovernor, provider_for, rows, report, verify_candidates

PROMPT = '''Solve this public task using actual public corpus evidence. Return ONE JSON action:
{"action":"native","name":<advertised tool>,"arguments":{...},"input_refs":{arg:{result_id,path}}},
{"action":"program","program_id":<visible ID>,"input_refs":{input:{result_id,path}}}, or
{"action":"finish","answer":<final string>} or {"action":"finish","answer_ref":{result_id,path}}.
Explicit native arguments and input_refs must be disjoint. Every program input must resolve from current
public ResultRefs, including the public task or controls of already performed calls. Never invent future
values, IDs, paths or numbers. Choose a program only if its applicability and prerequisites fit the local
operation. Missing inputs mean continue native retrieval. not_found/needs_input does not finish the task.
Use read_result to access an actual public field/window. Directory/search previews are incomplete.
Keep acquired paths, periods, columns and units consistent; do not repeatedly acquire the same evidence.
Report only a supported answer; submitting a literal answer does not establish Program output lineage.'''


def snapshot(context):
    return {k: sorted(v) if isinstance(v, set) else v for k, v in vars(context).items()}


def restore(state):
    context = TaskContext()
    for key, value in state.items():
        if isinstance(getattr(context, key, None), set): value = set(value)
        setattr(context, key, value)
    return context


def execute_action(action, broker, visible, worker, state):
    refs = action.get('input_refs', {})
    if not isinstance(refs, dict): raise ValueError('input_refs must map fields to current ResultRefs')
    resolved = {name: broker.context.resolve_public(ref) for name, ref in refs.items()}
    kind = action.get('action')
    record = {'kind': kind, 'resolved_refs': deepcopy(refs), 'accepted': False}
    if kind == 'native':
        explicit = action.get('arguments', {})
        arguments = broker.context.bind(explicit, refs)
        result = broker.call(action['name'], arguments)
        # Controls really sent in this episode become public bindable facts, never hypothetical values.
        control_id = broker.context.register('controls:' + broker.events[-1]['event_id'], {'data': arguments})
        record.update(accepted=result.get('accepted') is True, arguments=arguments, result_id=broker.events[-1]['result_id'],
                      controls_result_id=control_id, result=result)
    elif kind == 'program':
        if set(action) - {'action', 'program_id', 'input_refs'}: raise ValueError('Program accepts references only')
        selected = next((b for b in visible if b['id'] == action.get('program_id')), None)
        if selected is None: raise ValueError('Unknown or invisible program')
        validate_schema_instance(resolved, selected['program']['input_schema'])
        result = worker.execute(selected['program'], resolved, broker)
        rid = broker.context.register('program:' + str(len(state['actions'])), result, name='call_program',
            arguments={'program_id': selected['id'], 'input_refs': refs}, origin='program.execution-result.v1')
        record.update(kind='program', program_id=selected['id'], accepted=result.get('status') == 'ok',
                      status=result.get('status'), result_id=rid, result=result, bound_inputs=resolved)
        state['producers'].append({'program_id': selected['id'], 'result_id': rid, 'status': result.get('status'),
                                   'pilot': selected.get('formal_publication') is False})
    elif kind == 'finish':
        if 'answer_ref' in action:
            if 'answer' in action: raise ValueError('Finish chooses one exact answer source')
            value = broker.context.resolve_public(action['answer_ref'])
            if not isinstance(value, (str, int, float)) or isinstance(value, bool): raise ValueError('Answer reference must be scalar')
            answer = str(value); record['resolved_refs'] = {'answer': action['answer_ref']}
        else: answer = action.get('answer', '')
        if not isinstance(answer, str) or not answer.strip(): raise ValueError('Empty final answer')
        state.update(finished=True, prediction=answer, terminal_reason='submitted')
        record.update(accepted=True, prediction=answer)
    else: raise ValueError('Unknown Runtime action')
    return record


def consume(root):
    plan, config = checked(root, corpus=True); governor = PhaseGovernor(root, 'consume')
    candidates = [b for b in verify_candidates(root) if b['status'] in {'worker_probe_passed', 'cross_train_passed'}
                  and not b.get('superseded_by')]
    if not candidates: return {'status': 'not_applicable', 'reason': 'No Worker-passed candidates; no paired HTTP'}
    parent = [{'id': p['id'], 'program': p, 'atomic': {'goal': p.get('entry_constraints', ''),
               'applicability': 'Read-only V3 usable capability; inspect schema prerequisites', 'preconditions': [],
               'result_role': 'intermediate', 'stop_condition': 'Missing inputs'}, 'formal_publication': True}
              for p in read(root / 'parent_programs.json')]
    sources = read(root / 'source_records.json'); cap = plan['config']['paired_arm_limits']
    for task_id in plan['config']['transfer_tasks']:
        task = next(s['task'] for s in sources if s['task']['task_id'] == task_id)
        for arm in ('without_program', 'with_program'):
            folder = root / 'consumption' / task_id.replace(':', '_') / arm
            trace_path = folder / 'trace.json'
            if trace_path.exists(): continue
            state_path = folder / 'state.json'; cfg = deepcopy(config)
            cfg['experiment']['output_dir'] = str(folder)
            adapter = OfficeAdapter({task_id: {}}, cfg); adapter.reset(PublicTask(**task))
            if state_path.exists(): state = read(state_path); context = restore(state['context'])
            else:
                context = TaskContext(cfg['runtime'])
                rid = context.register('public_task', {'data': {'goal': task['goal'], **task['inputs']}})
                state = {'actions': [], 'events': [], 'producers': [], 'step': 0, 'finished': False,
                         'public_task_result_id': rid, 'context': snapshot(context)}
                save_json(state_path, state)
            broker = Broker(adapter, cap['native_calls'], context=context); broker.events = state['events']
            visible = parent + (candidates if arm == 'with_program' else [])
            scope = f"{plan['run_id']}:{task_id}:{arm}"
            client = RequestStore(folder / 'requests', provider_for(config, 'runtime', governor), phase='consume',
                attribution={'parent_scope': scope, 'parent_token_limit': cap['tokens'],
                    'episode_scope': scope, 'episode_request_limit': cap['http']})
            worker = ProgramWorker({'record_execution_receipt': True})
            try:
                while not state['finished'] and state['step'] < cap['http']:
                    step = state['step']; intent = folder / f'action_{step}.intent.json'
                    if intent.exists(): raise RuntimeError('Unresolved Runtime action; do not repeat paid/model/native execution blindly')
                    data = {'task': task, 'public_task_result_id': state['public_task_result_id'],
                            'tools': broker.available_tools(), 'state': runtime_view(context, visible),
                            'recent_actions': [{k: a.get(k) for k in ('kind', 'accepted', 'error', 'result_id', 'controls_result_id', 'status', 'arguments')}
                                               for a in state['actions'][-6:]], 'remaining_calls': broker.remaining_calls()}
                    try:
                        action, _ = client.ask(str(step), PROMPT, data)
                    except BudgetExhausted:
                        state.update(finished=True, terminal_reason='budget_exhausted'); break
                    except ModelOutputError as exc:
                        state['actions'].append({'kind': 'model_error', 'accepted': False, 'error': str(exc)})
                        state['step'] += 1; state['context'] = snapshot(context); save_json(state_path, state); continue
                    save_json(intent, {'action': action, 'context_hash': digest(snapshot(context))})
                    try: record = execute_action(action, broker, visible, worker, state)
                    except (ValueError, KeyError, TypeError) as exc:
                        record = {'kind': 'invalid_action', 'accepted': False, 'error': str(exc)}
                    state['actions'].append(record); state['step'] += 1
                    state['events'] = broker.events; state['context'] = snapshot(context)
                    save_json(state_path, state)
                    # Completed intent is retained separately; recovery relies on the committed step counter.
                state.setdefault('terminal_reason', 'request_limit')
                # Private Train evaluator is loaded only after all model/Worker interaction ends.
                old = read(plan['source'] + '/config.json')
                from ..harness.benchmarks import score_answer
                labels = read(old['harness']['evaluator_records'])
                if isinstance(labels, list): labels = {r['task_id']: r for r in labels}
                label = labels.get(task_id, labels.get(task_id.split(':')[-1]))
                if label is None: raise RuntimeError('Private Train scoring record missing')
                score = score_answer('officeqa', state.get('prediction', ''), label)
                uses = [{'producer': p, 'uses': consumed(p, state['actions'])} for p in state['producers'] if p['pilot']]
                trace = {'task': task, 'arm': arm, 'visible_program_ids': [b['id'] for b in visible],
                    'actions': state['actions'], 'events': broker.events, 'prediction': state.get('prediction'), 'score': score,
                    'terminal_reason': state['terminal_reason'], 'consumed': [u for u in uses if u['uses']],
                    'partial_workflow': partial_workflow(state['producers'], state['actions'], {b['id']:b['program'] for b in visible}),
                    'limits': cap, 'runtime_prompt_sha256': digest(PROMPT), 'provider': client.provider.snapshot()}
                save_json(trace_path, trace)
            finally: adapter.close()
    return report(root)
