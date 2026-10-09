"""Program-first execution with task-local patches and bounded recovery."""
from copy import deepcopy
from uuid import uuid4
import json

from .contracts import (ResultRef, RuntimeDecision, ValueStore, digest, validate_schema_instance,
                        resolve_node_interface, handoff_requirements, output_view, HandoffError)
from .planner import dynamic
from .prompts import finish_text_prompt, PATCH, RUNTIME_PROMPT, STEP
from .task_context import progress_key, public_view
from ..harness.simple_protocol import UnknownSideEffect
from .model_view import project
from .finish_evidence import finalize_text
from .trial_snapshot import seal_trial_workspace, restore_trial_workspace
from ..core.errors import BudgetExhausted
from .program_submission import (effective_output_schema, submission_contract,
                                 normalize_program_result, prepare_program_submission, positive_eligible)


class Executor:
    def __init__(self, bank, agent, worker, planner, *, frozen=False):
        self.bank, self.agent, self.worker, self.planner = bank, agent, worker, planner
        self.frozen = frozen

    def run(self, task, adapter, broker, plan):
        self.partial_execution = {'attempts': []}
        context = broker.context
        values, attempts = ValueStore(task, context), []
        blocked, attempted, failures, history = set(), set(), {}, []
        completed, abandoned, replans, escapes, node_index = set(), set(), 0, 0, 0
        producers, supplied_by, consumed = {}, {}, set()
        input_reads, replaced_inputs = [], {}
        reason, final, last_output, last_role = 'completed', None, {}, 'intermediate'
        last_producer, submission_producer_attempt_id = None, None
        finish_evidence = None
        pending_step, pending_decision_id, owner_version, recovery_needed = None, None, 0, False
        checkpoint = getattr(self, 'checkpoint', None)
        submission_kind = adapter.capabilities.final_submission_kind
        if checkpoint and checkpoint.state.get('executor_finished'):
            receipt = checkpoint.state.get('executor_finished_workspace')
            if receipt: restore_trial_workspace(adapter, receipt)
            elif checkpoint.state['executor_finished'].get('submission_producer_attempt_id') and getattr(adapter,'workspace',None):
                raise UnknownSideEffect('Program submission lacks a reconstructible public workspace')
            return checkpoint.state['executor_finished']
        saved = checkpoint.state.get('executor_state') if checkpoint else None
        if saved:
            plan, node_index, replans, escapes = saved['plan'], saved['node_index'], saved['replans'], saved['escapes']
            values.results, values.history = saved['values'], saved['value_history']
            for node_id, key in saved['reference_fields']:
                values.results[node_id][key] = ResultRef(values.results[node_id][key])
            attempts, history, failures = saved['attempts'], saved['history'], saved['failures']
            completed, abandoned, blocked, attempted, consumed = (set(saved[k]) for k in
                ['completed', 'abandoned', 'blocked', 'attempted', 'consumed'])
            producers, input_reads, replaced_inputs = saved['producers'], saved['input_reads'], saved['replaced_inputs']
            supplied_by = {(n, k): v for n, k, v in saved['supplied_by']}
            final, last_output, last_role = saved['final'], saved['last_output'], saved['last_role']
            last_producer = saved.get('last_producer')
            submission_producer_attempt_id = saved.get('submission_producer_attempt_id')
            reason = saved.get('reason', reason)
            for key, value in saved['context'].items():
                setattr(context, key, set(value) if key in {'progress_contents', 'progress_states'} else value)
            pending_step, pending_decision_id = saved['pending_step'], saved['pending_decision_id']
            owner_version, recovery_needed = saved['owner_version'], saved['recovery_needed']
            for event in broker.events:
                if event['state'] == 'finished' and event['event_id'] not in context.sources:
                    result_id = context.register(event['event_id'], event['result'], name=event['name'], arguments=event['arguments'])
                    if result_id != event['result_id']:
                        raise ValueError('Recovered result identity differs')

        def save(step=None, status=None):
            nonlocal owner_version, pending_decision_id
            self.partial_execution = {'attempts': deepcopy(attempts)}
            if status:
                owner_version += 1
            if not checkpoint:
                if status: pending_decision_id = None
                return
            encoded = {'actions': list(step.actions), 'call_ids': list(step.call_ids), 'response_id': step.response_id} if isinstance(step, RuntimeDecision) else step
            state = {'plan': plan, 'node_index': node_index, 'replans': replans, 'escapes': escapes,
                'values': values.results, 'value_history': values.history,
                'reference_fields': [[n,k] for n, fields in values.results.items() for k, value in fields.items() if isinstance(value, ResultRef)],
                'attempts': attempts, 'history': history, 'failures': failures, 'completed': sorted(completed),
                'abandoned': sorted(abandoned), 'blocked': sorted(blocked), 'attempted': sorted(attempted),
                'consumed': sorted(consumed), 'producers': producers, 'input_reads': input_reads,
                'replaced_inputs': replaced_inputs, 'supplied_by': [[n,k,v] for (n,k),v in supplied_by.items()],
                'final': final, 'reason': reason, 'last_output': last_output, 'last_role': last_role, 'pending_step': encoded,
                'last_producer': last_producer, 'submission_producer_attempt_id': submission_producer_attempt_id,
                'pending_decision_id': None if status else pending_decision_id, 'owner_version': owner_version,
                'recovery_needed': recovery_needed,
                'context': {k: sorted(getattr(context,k)) if k in {'progress_contents','progress_states'} else getattr(context,k)
                    for k in ['scope','results','sources','memory','local_reads','pending_outputs','active_node',
                              'progress_observations','progress_contents','progress_states','loop_hits','loop_feedback']}}
            if status:
                checkpoint.commit_decision(pending_decision_id, status, executor_state=state, program_started=False)
                pending_decision_id = None
            else:
                checkpoint.advance(checkpoint.state['stage'], executor_state=state)

        def submission(output, role='intermediate', producer=None):
            if producer:
                attempt = next(a for a in attempts if a['id'] == producer)
                result = context.results[attempt['result_id']]
                program = self.bank.get(attempt['program_id'])
                prepared = prepare_program_submission(adapter, program, result,
                    workspace_before=attempt.get('workspace_before'))
                return prepared['status'] == 'ready' or (prepared['output_contract_status'] == 'valid'
                    and program.get('result_role', 'intermediate') == 'intermediate' and role == 'final_answer'
                    and getattr(adapter, 'submission_ready', lambda *a, **k: False)(output, result_role=role))
            return getattr(adapter, 'submission_ready', lambda *a, **k: False)(output, result_role=role)

        def recent():
            return [context.view(row['result_id']) if 'result_id' in row else
                    {k: context.preview(v) for k,v in row.items()} for row in history[-3:]]

        def finalize():
            nonlocal final, reason, pending_decision_id, finish_evidence, submission_producer_attempt_id
            if submission(last_output, last_role, last_producer):
                final = last_output['answer']
                submission_producer_attempt_id = last_producer
                return
            try:
                final, finish_evidence = finalize_text(self.agent, context,
                    {'original_goal': task.goal, 'public_state': public_view(adapter),
                     'working_memory': context.model_memory(), 'recent': recent(),
                     'completed_results': values.model_view(), 'pending_outputs': context.pending_outputs},
                    finish_text_prompt(getattr(adapter, 'answer_contract', lambda: '')()),
                    owner_state_version=owner_version)
                candidate_id = getattr(getattr(self.agent, '__self__', None), 'last_decision_id', None)
                pending_decision_id = candidate_id if checkpoint and candidate_id in checkpoint.state['decisions'] else None
                reason = 'finish_only'
                save(status='applied')
            except ValueError as exc:
                candidate_id = getattr(exc, 'logical_decision_id', None)
                pending_decision_id = candidate_id if checkpoint and candidate_id in checkpoint.state['decisions'] else None
                history.append({'error': str(exc)})
                final, reason = '', str(exc) if str(exc).startswith('finish_only_') else 'finish_only_protocol_error'
                save(status='rejected')

        def fail(kind, interface, *, name='', arguments=None, detail=None):
            nonlocal recovery_needed
            arguments = getattr(adapter, 'failure_arguments', lambda n, a: a)(name, arguments) if name else arguments
            signature = digest([kind, interface['execution_mode'], interface['bound_skill_id'],
                                interface['bound_program_id'], name, arguments, progress_key(adapter)])
            failures[signature] = failures.get(signature, 0) + 1
            history.append({'error': kind, 'failure_signature': signature, 'count': failures[signature],
                            'feedback': detail or {}})
            recovery_needed = failures[signature] >= 2
            return signature

        def escape():
            nonlocal plan, node_index, escapes
            if escapes:
                return False
            abandoned.update(n['id'] for n in plan['nodes'] if n['id'] not in completed)
            plan = dynamic(task)
            plan['nodes'][0]['id'] = 'remaining_task'
            if 'remaining_task' in completed:
                plan['nodes'][0]['id'] = 'remaining_task_escape'
            escapes, node_index = 1, 0
            history.append({'dynamic_escape': True, 'abandoned': sorted(abandoned)})
            return True

        def recover():
            nonlocal replans, plan, node_index, recovery_needed, pending_decision_id
            recovery_needed = False
            if not replans:
                replans = 1
                revised = self.planner.plan(task, adapter, feedback=history[-3:], completed_results=values.model_view())
                if not set(n['id'] for n in revised['nodes']) & completed:
                    abandoned.update(n['id'] for n in plan['nodes'] if n['id'] not in completed)
                    plan, node_index = revised, 0
                    history.append({'replan_reason': 'repeated_unchanged_failure'})
                    pending_decision_id = getattr(self.planner, 'last_decision_id', None)
                    return True
            return escape()

        def complete(node, outputs, origin, *, refs=None, producer=None, result_role=None):
            nonlocal node_index, last_output, last_role, last_producer
            interface = resolve_node_interface(node, self.bank)
            actual = context.bind(outputs, refs or {})
            requirements = handoff_requirements(plan, node['id'], completed)
            context.pending_outputs[node['id']] = {'outputs': outputs, 'output_refs': refs or {},
                                                   'origin': origin, 'producer': producer, 'result_role': result_role}
            role = result_role or interface['result_role']
            view = actual
            try:
                if interface['output_schema']:
                    schema = effective_output_schema(interface['output_schema'], result_role=role,
                        publication_contract=submission_contract(adapter)['publication_contract']) if producer else interface['output_schema']
                    validate_schema_instance(actual, schema)
                view = output_view(actual, node.get('output_aliases'))
                if set(requirements['required_fields']) - set(view):
                    raise ValueError('Required handoff fields are absent')
                if escapes and node['id'] == plan['nodes'][-1]['id'] and not broker.done and not submission(actual, role, producer):
                    raise ValueError('Remaining task is not terminal or ready for submission')
            except ValueError as exc:
                raise HandoffError({'node_id': node['id'], 'execution_mode': interface['execution_mode'],
                    **requirements, 'received_fields': sorted(view),
                    'missing_fields': sorted(set(requirements['required_fields']) - set(view)),
                    'detail': str(exc)}) from exc
            values.publish(node['id'], outputs, origin, refs, node.get('output_aliases'))
            if producer:
                producers[node['id']] = producer
            context.pending_outputs.pop(node['id'], None)
            context.mark_progress('handoff', actual)
            completed.add(node['id'])
            last_output, last_role = actual, role
            last_producer = producer
            node_index += 1

        def patch(value):
            nonlocal plan
            validate_schema_instance(value, PATCH)
            kind, target = value['kind'], value['target_node']
            allowed = {'args': {'args'}, 'handoff': {'output_aliases','consumer_refs'}, 'detach': {'dynamic_goal'}}[kind]
            if set(value) - {'target_node','kind','reason'} - allowed:
                raise ValueError('Patch fields do not match its kind')
            draft = deepcopy(plan)
            nodes = {n['id']: n for n in draft['nodes']}
            if target not in nodes:
                raise ValueError('Unknown patch target')
            node = nodes[target]
            if kind != 'handoff' and target in completed:
                raise ValueError('Cannot patch an executed node')
            alias_view = None
            if kind == 'args':
                if not value.get('args'):
                    raise ValueError('args patch requires replacement arguments')
                node.setdefault('args', {}).update(value['args'])
            elif kind == 'detach':
                interface = resolve_node_interface(node, self.bank)
                if interface['execution_mode'] == 'dynamic' or not value.get('dynamic_goal'):
                    raise ValueError('detach needs an actual binding and dynamic_goal')
                node.pop('skill_id', None); node.pop('program_id', None)
                node.update(execution_mode='dynamic', goal=value['dynamic_goal'])
                node['reference_skill_ids'] = list(dict.fromkeys([
                    *([interface['bound_skill_id']] if interface['bound_skill_id'] else []),
                    *node.get('reference_skill_ids', [])]))[:3]
            else:
                pending = context.pending_outputs.get(target)
                if pending:
                    actual = context.bind(pending['outputs'], pending['output_refs'])
                elif target in completed:
                    original = next(row['outputs'] for row in values.history if row['node'] == target and row['origin'] != 'handoff')
                    actual = {key: values.results[target][key] for key in original}
                else:
                    raise ValueError('No actual or pending outputs to hand off')
                node['output_aliases'] = {**node.get('output_aliases', {}), **value.get('output_aliases', {})}
                alias_view = output_view(actual, node['output_aliases'])
                for consumer_id, refs in value.get('consumer_refs', {}).items():
                    if consumer_id not in nodes or consumer_id in completed or consumer_id == target:
                        raise ValueError('Only unfinished related consumers may be patched')
                    for name, ref in refs.items():
                        old = nodes[consumer_id].get('args', {}).get(name, {})
                        if set(ref) != {'from','field'} or ref['from'] != target or ref['field'] not in alias_view or old.get('from') != target:
                            raise ValueError('consumer_refs must use an actual field of the same producer')
                    nodes[consumer_id]['args'].update(refs)
            projected = {**values.results, **({target: alias_view} if target in completed and alias_view is not None else {})}
            future = {**draft, 'nodes': [n for n in draft['nodes'] if n['id'] not in completed]}
            if future['nodes']:
                self.planner.validate(future, projected)
            history.append({'patch_node': deepcopy(value), 'before': deepcopy(plan), 'after': deepcopy(draft)})
            plan = draft
            if target in completed and alias_view is not None:
                values.results[target] = alias_view
                view_id = context.register(uuid4().hex, {'outputs': {k: context.resolve(v) if isinstance(v, ResultRef) else v
                    for k,v in alias_view.items()}, 'source_node': target})
                values.history.append({'node': target, 'origin': 'handoff', 'outputs': alias_view, 'view_result_id': view_id})
            if kind == 'args':
                for name in value['args']:
                    replaced_inputs.setdefault(target, []).append(name)
                    supplied_by.pop((target, name), None)
            if target not in completed and target == plan['nodes'][node_index]['id'] and target in context.pending_outputs:
                pending = context.pending_outputs[target]
                if pending['origin'] != 'preparation':
                    try:
                        complete(nodes[target], pending['outputs'], pending['origin'], refs=pending['output_refs'],
                                 producer=pending['producer'], result_role=pending.get('result_role'))
                    except ValueError as exc:
                        fail('handoff_error', resolve_node_interface(nodes[target], self.bank),
                             arguments=context.bind(pending['outputs'], pending['output_refs']), detail=getattr(exc, 'feedback', str(exc)))

        def invoke(program, arguments, node_id):
            start, was_terminal = len(broker.events), broker.done
            workspace_before = adapter.observe().get('workspace', {})
            if checkpoint:
                checkpoint.advance(checkpoint.state['stage'], program_started=True)
            result = self.worker.execute(program, arguments, broker)
            result = normalize_program_result(adapter, program, result, workspace_before=workspace_before)
            local = broker.check_local(arguments, result.get('outputs', {}), start) if result['status'] == 'ok' else 'unavailable'
            result['local_check'] = local
            if local == 'failed' or result['status'] == 'execution_error':
                blocked.add(program['id'])
                outcome = 'execution_failure'
            else:
                outcome = 'positive' if positive_eligible(result) and local == 'passed' else 'normal'
            attempt = {'id': uuid4().hex, 'program_id': program['id'], 'task_key': task.physical_key,
                'origin': 'online', 'split': task.split, 'outcome': outcome, 'calls': len(broker.events)-start,
                'status': result['status'], 'local_check': local, 'basis': 'local_check' if outcome == 'positive' else None,
                'node': node_id, 'outputs_consumed': False, 'terminal_by_program': not was_terminal and broker.done}
            attempt.update(worker_status=result.get('worker_status'), output_contract_status=result['output_contract_status'],
                publication_receipt=result.get('publication_receipt'), workspace_before=workspace_before,
                error_code=result.get('error_code'), submission_by_program=False)
            attempts.append(attempt)
            result_id = context.register(attempt['id'], result, name=program['id'], arguments=arguments)
            history.append({'program': program['id'], 'arguments': deepcopy(arguments), 'result_id': result_id})
            attempt.update(result_id=result_id, native_event_start=start, native_event_end=len(broker.events))
            return result

        for _ in range(max(16, broker.call_limit * 4 + 16)):
            save(pending_step)
            if reason.startswith('finish_only'):
                break
            if broker.done:
                reason = 'environment_terminal'; break
            if recovery_needed:
                if not recover():
                    reason = 'repeated_unchanged_failure'; break
                save(status='applied')
            if node_index >= len(plan['nodes']):
                final_values, _ = values.resolve(plan.get('outputs', {}))
                answer_ref = plan.get('outputs', {}).get('answer', {})
                answer_producer = producers.get(answer_ref.get('from')) if answer_ref.get('field') else None
                if submission_kind == 'text' and (('answer' in final_values and (not answer_producer or
                        submission(final_values, 'final_answer', answer_producer))) or submission(last_output, last_role, last_producer)):
                    final, reason = final_values.get('answer', last_output.get('answer')), 'plan_submitted'
                    submission_producer_attempt_id = answer_producer if 'answer' in final_values else last_producer
                    break
                if submission_kind == 'files' and submission(last_output, last_role, last_producer):
                    final, reason = final_values, 'plan_submitted'
                    submission_producer_attempt_id = last_producer
                    break
                if final is not None or adapter.capabilities.interaction == 'single_answer':
                    break
                if not escape():
                    reason = 'remaining_task_incomplete'; break
            node = plan['nodes'][node_index]
            interface = resolve_node_interface(node, self.bank)
            context.activate_node(digest([len(completed), interface['execution_mode'], interface['node_goal'], interface['bound_program_id']]))
            arguments, missing = values.resolve(node.get('args', {}))
            input_producers = {field: producers.get(ref.get('from')) or supplied_by.get((node['id'], field))
                               for field, ref in node.get('args', {}).items() if field in arguments}
            routes = [p for p in self.bank.routes(node) if p['id'] not in blocked]
            automatic = False
            if pending_step is None:
                for program in routes if broker.remaining_calls() > 0 and not missing else []:
                    key = digest([len(values.history), program['id'], arguments, progress_key(adapter)])
                    if key in attempted:
                        continue
                    try:
                        if interface['input_schema']:
                            validate_schema_instance(arguments, interface['input_schema'])
                        validate_schema_instance(arguments, program['input_schema'])
                        if len(json.dumps(arguments, ensure_ascii=False).encode()) > self.worker.settings['max_rpc_message_bytes']:
                            raise ValueError('Use bounded read_result; arguments exceed RPC limit')
                    except ValueError:
                        continue
                    result = invoke(program, arguments, node['id'])
                    attempted.update({key, digest([len(values.history), program['id'], arguments, progress_key(adapter)])})
                    if arguments:
                        input_reads.append(node['id'])
                    consumed.update(p for p in input_producers.values() if p)
                    if result['status'] == 'ok' and result['local_check'] != 'failed':
                        try:
                            complete(node, result['outputs'], 'program', producer=attempts[-1]['id'], result_role=program.get('result_role'))
                        except ValueError as exc:
                            fail('handoff_error', interface, arguments=result['outputs'], detail=getattr(exc, 'feedback', str(exc)))
                    else:
                        fail('program_result', interface, name=program['id'], arguments=arguments, detail=result['status'])
                    save(status='applied')
                    automatic = True
                    break
            if automatic:
                continue
            if broker.remaining_calls() <= 0:
                reason = 'tool_budget_exhausted'
                if submission_kind == 'text':
                    finalize()
                elif submission_kind == 'files': final = {}
                break
            references = interface['reference_skill_ids']
            guidance = [self.bank.get(ref) for ref in dict.fromkeys(references)]
            if interface['bound_skill_id']:
                guidance = [self.bank.get(interface['bound_skill_id']), *[s for s in guidance if s and s['id'] != interface['bound_skill_id']]]
            guidance = guidance[:3]
            if not guidance:
                guidance = [s for s in self.bank.retrieve(interface['node_goal']) if 'guidance' in s][:3]
            requirements = handoff_requirements(plan, node['id'], completed)
            materials = {'original_task': {'goal': task.goal, 'inputs': task.inputs}, 'node_goal': interface['node_goal'],
                'node_id': node['id'], 'node_interface': interface, 'inputs': arguments, 'missing': missing,
                'required_handoff_fields': requirements['required_fields'], 'handoff_consumers': requirements['consumers'],
                'return_example': {k: '<actual ordinary value>' for k in requirements['required_fields']},
                'pending_outputs': context.preview(context.pending_outputs.get(node['id'], {})),
                'output_aliases': node.get('output_aliases', {}),
                'public_state': public_view(adapter), 'tools': broker.available_tools(),
                'programs': self.bank.program_options(task.goal, node=node, allow_candidate=not self.frozen, excluded=blocked),
                'guidance': [{'goal': s.get('goal'), 'guidance': s.get('guidance','')} for s in guidance if s],
                'working_memory': context.model_memory(), 'recent': recent(), 'remaining_calls': broker.remaining_calls(),
                'allowed_calls': '1-3 independent read_only/batchable calls; otherwise one operation' if any(t.get('batchable') for t in broker.available_tools()) else 'one operation',
                'may_replan': replans == 0, 'may_escape': escapes == 0, 'loop_feedback': context.loop_feedback,
                'completed_results': values.model_view()}
            materials = project('runtime', materials, task=task, adapter=adapter, context=context)
            save(pending_step)
            def validate_step(step):
                context.bind(step.get('arguments', {}), step.get('argument_refs', {}))
                context.bind(step.get('outputs', {}), step.get('output_refs', {}))
            try:
                if pending_step is not None:
                    step = RuntimeDecision(tuple(pending_step['actions']), tuple(pending_step['call_ids']), pending_step['response_id']) if 'actions' in pending_step else pending_step
                else:
                    step = self.agent('runtime', RUNTIME_PROMPT, materials, 'runtime_step', STEP,
                                      repair_limit=1, validator=validate_step, owner_state_version=owner_version)
                    candidate_id = getattr(getattr(self.agent, '__self__', None), 'last_decision_id', None)
                    pending_decision_id = candidate_id if checkpoint and candidate_id in checkpoint.state['decisions'] else None
                save(step)
                pending_step = None
            except BudgetExhausted as exc:
                if exc.code != 'runtime_finish_reserved' or submission_kind != 'text': raise
                finalize()
                break
            except ValueError as exc:
                candidate_id = getattr(exc, 'logical_decision_id', None)
                pending_decision_id = candidate_id if checkpoint and candidate_id in checkpoint.state['decisions'] else None
                fail('runtime_protocol_error', interface, detail=str(exc))
                save(status='rejected')
                continue
            decisions = list(zip(step.actions, step.call_ids)) if isinstance(step, RuntimeDecision) else [(step, None)]
            response_id = step.response_id if isinstance(step, RuntimeDecision) else None
            rejected, finish = False, False
            for index, (action, call_id) in enumerate(decisions):
                kind = action['action']
                used = set(action.get('used_inputs', [])) & set(arguments)
                replaced = set(action.get('replaced_inputs', [])) & set(arguments)
                if used - replaced:
                    input_reads.append(node['id'])
                consumed.update(input_producers[k] for k in used-replaced if input_producers.get(k))
                for field in replaced:
                    if input_producers.get(field): consumed.discard(input_producers[field])
                    replaced_inputs.setdefault(node['id'], []).append(field)
                try:
                    if kind == 'finish':
                        final, reason, finish = action.get('answer'), 'agent_submitted', True
                        submission_producer_attempt_id = None
                    elif kind == 'revise_plan':
                        if replans:
                            if not escape():
                                raise ValueError('Full replan and Dynamic escape exhausted')
                        else:
                            revised = self.planner.validate(action['workflow'], values.results)
                            if set(n['id'] for n in revised['nodes']) & completed:
                                raise ValueError('Revision contains a completed node')
                            abandoned.update(n['id'] for n in plan['nodes'] if n['id'] not in completed)
                            replans, plan, node_index = 1, revised, 0
                            history.append({'plan_revision': deepcopy(revised), 'completed_results': sorted(completed)})
                    elif kind == 'patch_node':
                        patch(action['patch'])
                    elif kind == 'complete_node':
                        complete(node, action.get('outputs', {}), 'agent', refs=action.get('output_refs', {}))
                    elif kind in {'call_program','call_tool'}:
                        name = action.get('name', '')
                        call_args = context.bind(action.get('arguments', {}), action.get('argument_refs', {}))
                        signature = digest(['dispatch', name, getattr(adapter, 'failure_arguments', lambda n, a: a)(name, call_args), progress_key(adapter)])
                        if failures.get(signature, 0) >= 2:
                            recovery_needed = True
                            raise ValueError('Same known invalid operation already failed twice')
                        if kind == 'call_program':
                            program = self.bank.get(name)
                            if not program or 'source' not in program or name in blocked or program['state'] == 'disabled' or self.frozen and program['state'] != 'usable':
                                raise ValueError('Program is unavailable in this task')
                            result = invoke(program, call_args, node['id'])
                            accepted = result['status'] == 'ok' and result['local_check'] != 'failed'
                            if accepted and (name in {p['id'] for p in routes} or interface['execution_mode'] == 'dynamic' and
                                'output_mapping' not in action and set(requirements['required_fields']).issubset(result['outputs'])):
                                complete(node, result['outputs'], 'program', producer=attempts[-1]['id'], result_role=program.get('result_role'))
                            elif accepted:
                                supplied = result['outputs']
                                context.pending_outputs[node['id']] = {'outputs': supplied, 'output_refs': {},
                                                                       'origin': 'preparation', 'producer': attempts[-1]['id']}
                                expected = set(node.get('args', {}))
                                for route in routes: expected.update(route['input_schema'].get('properties', {}))
                                if interface['input_schema']: expected.update(interface['input_schema'].get('properties', {}))
                                mapping = action.get('output_mapping', {k:k for k in supplied if k in expected})
                                if any(source not in supplied or target not in expected for source,target in mapping.items()) or len(set(mapping.values())) != len(mapping):
                                    raise ValueError('Invalid preparation output_mapping')
                                if any(target in arguments and target not in replaced for target in mapping.values()):
                                    raise ValueError('Replacing a known input requires replaced_inputs')
                                node.setdefault('args', {}).update({target:{'literal':supplied[source]} for source,target in mapping.items()})
                                supplied_by.update({(node['id'],target):attempts[-1]['id'] for target in mapping.values()})
                        else:
                            event_id = response_id + ':' + call_id if response_id else None
                            prior = next((e for e in broker.events if e.get('event_id') == event_id), None) if event_id else None
                            if not prior and broker.remaining_calls() <= 0:
                                history.extend({'batch_id': response_id, 'call_id':c, 'status':'not_executed_budget'} for _,c in decisions[index:])
                                break
                            result = broker.call(name, call_args, event_id=event_id, batch_id=response_id, call_id=call_id)
                            event = next(e for e in reversed(broker.events) if not event_id or e['event_id'] == event_id)
                            context.results[event['result_id']], context.sources[event['event_id']] = result, event['result_id']
                            if not any(h.get('result_id') == event['result_id'] for h in history):
                                history.append({'tool': name, 'result_id': event['result_id']})
                            accepted = bool(result.get('accepted'))
                        failures[signature] = 0 if accepted else failures.get(signature, 0) + 1
                        recovery_needed = failures[signature] >= 2
                        if context.loop_feedback and context.loop_feedback['hits'] >= 2:
                            recovery_needed = True
                        if not accepted:
                            rejected = True
                    else:
                        raise ValueError('Invalid Runtime action')
                except UnknownSideEffect:
                    history.extend({'batch_id':response_id, 'call_id':c, 'status':'not_executed_infrastructure'} for _,c in decisions[index+1:])
                    save(step)
                    raise
                except (ValueError, PermissionError, KeyError) as exc:
                    error_kind = 'handoff_error' if isinstance(exc, HandoffError) or kind == 'complete_node' else 'plan_binding_error'
                    failed_values = context.bind(action.get('outputs', {}), action.get('output_refs', {})) if kind == 'complete_node' else action.get('arguments', {})
                    if kind == 'patch_node':
                        failed_values = {k:v for k,v in action['patch'].items() if k not in {'target_node','reason'}}
                    fail(error_kind, interface, name=action.get('name',''), arguments=failed_values, detail=getattr(exc, 'feedback', str(exc)))
                    rejected = True
                if len(decisions) > 1:
                    save(step)
                if recovery_needed or finish:
                    break
            save(status='rejected' if rejected else 'applied')
            if finish:
                break
        else:
            reason = 'decision_budget_exhausted'
        final_values, _ = values.resolve(plan.get('outputs', {}))
        if final is None or reason == 'plan_submitted':
            for ref in plan.get('outputs', {}).values():
                if ref.get('from') in producers and ref.get('field') in values.results.get(ref['from'], {}):
                    consumed.add(producers[ref['from']])
        for attempt in attempts:
            attempt['outputs_consumed'] = attempt['id'] in consumed
            attempt['submission_by_program'] = attempt['id'] == submission_producer_attempt_id
        result = {'prediction': final if final is not None else final_values, 'reason': reason,
                  'values': values.history, 'history': history, 'attempts': attempts, 'plan': plan,
                  'plan_revisions': replans, 'dynamic_escapes': escapes, 'abandoned': sorted(abandoned),
                  'pending_outputs': context.pending_outputs, 'input_reads': input_reads, 'replaced_inputs': replaced_inputs}
        result['submission_producer_attempt_id'] = submission_producer_attempt_id
        result['finish_evidence'] = finish_evidence
        if checkpoint:
            receipt = seal_trial_workspace(adapter, checkpoint.root/'executor_finished_workspace') if getattr(adapter,'workspace',None) else None
            checkpoint.advance(checkpoint.state['stage'], executor_finished=result, program_started=False,
                               executor_finished_workspace=receipt)
        return result
