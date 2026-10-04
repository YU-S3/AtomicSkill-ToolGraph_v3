"""Program-first execution, ordinary values and one task-local plan revision."""
from uuid import uuid4
import json

from .contracts import ResultRef, RuntimeDecision, ValueStore, digest, validate_schema_instance
from .planner import dynamic
from .prompts import FINISH, RUNTIME_PROMPT, STEP
from .task_context import progress_key, public_view
from ..harness.simple_protocol import UnknownSideEffect


class Executor:
    def __init__(self, bank, agent, worker, planner, *, frozen=False):
        self.bank, self.agent, self.worker, self.planner = bank, agent, worker, planner
        self.frozen = frozen

    def run(self, task, adapter, broker, plan):
        context = broker.context
        values, attempts = ValueStore(task, context), []
        blocked, attempted, failures, history = set(), set(), {}, []
        completed, replans, node_index, final = set(), 0, 0, None
        producers, supplied_by, consumed = {}, {}, set()
        input_reads, replaced_inputs = [], {}
        dynamic_sequence = 0
        reason = "completed"
        submission_kind = adapter.capabilities.final_submission_kind
        last_output, last_role = {}, 'intermediate'
        checkpoint = getattr(self, 'checkpoint', None)
        pending_step = None
        if checkpoint and checkpoint.state.get('executor_finished'):
            return checkpoint.state['executor_finished']
        saved = checkpoint.state.get('executor_state') if checkpoint else None
        if saved:
            plan, node_index, replans = saved['plan'], saved['node_index'], saved['replans']
            values.results, values.history = saved['values'], saved['value_history']
            for node_id, key in saved['reference_fields']:
                values.results[node_id][key] = ResultRef(values.results[node_id][key])
            attempts, history, failures = saved['attempts'], saved['history'], saved['failures']
            completed, blocked, attempted, consumed = (set(saved[k]) for k in ['completed','blocked','attempted','consumed'])
            producers, input_reads, replaced_inputs = saved['producers'], saved['input_reads'], saved['replaced_inputs']
            supplied_by = {(node_id, field): producer for node_id, field, producer in saved['supplied_by']}
            dynamic_sequence, final = saved['dynamic_sequence'], saved['final']
            last_output, last_role = saved['last_output'], saved['last_role']
            context.scope = saved['context']['scope']
            for key in ['results','sources','memory','local_reads']: setattr(context, key, saved['context'][key])
            pending_step = saved['pending_step']
            for event in broker.events:
                if event['state'] == 'finished' and event['event_id'] not in context.sources:
                    result_id = context.register(event['event_id'], event['result'], name=event['name'], arguments=event['arguments'])
                    if result_id != event['result_id']: raise ValueError('Recovered result identity differs')

        def save(step=None):
            if not checkpoint: return
            encoded = {'actions': list(step.actions), 'call_ids': list(step.call_ids), 'response_id': step.response_id} if isinstance(step, RuntimeDecision) else step
            checkpoint.advance(checkpoint.state['stage'], executor_state={
                'plan': plan, 'node_index': node_index, 'replans': replans, 'values': values.results,
                'value_history': values.history, 'reference_fields': [[n,k] for n, fields in values.results.items()
                    for k, value in fields.items() if isinstance(value, ResultRef)],
                'attempts': attempts, 'history': history, 'failures': failures, 'completed': sorted(completed),
                'blocked': sorted(blocked), 'attempted': sorted(attempted), 'consumed': sorted(consumed),
                'producers': producers, 'input_reads': input_reads, 'replaced_inputs': replaced_inputs,
                'supplied_by': [[n,k,v] for (n,k),v in supplied_by.items()], 'dynamic_sequence': dynamic_sequence,
                'final': final, 'last_output': last_output, 'last_role': last_role, 'pending_step': encoded,
                'context': {k: getattr(context,k) for k in ['scope','results','sources','memory','local_reads']}})

        def submission(output, role='intermediate'):
            ready = getattr(adapter, 'submission_ready', lambda *a, **k: False)
            return ready(output, result_role=role)

        def recent():
            return [context.view(row['result_id']) if 'result_id' in row else {
                k: context.preview(v) for k, v in row.items()} for row in history[-3:]]

        def validate_step(step):
            context.bind(step.get('arguments', {}), step.get('argument_refs', {}))
            context.bind(step.get('outputs', {}), step.get('output_refs', {}))

        def fallback():
            nonlocal dynamic_sequence
            result = dynamic(task)
            while True:
                dynamic_sequence += 1
                node_id = 'dynamic_' + str(dynamic_sequence)
                if node_id not in completed:
                    result['nodes'][0]['id'] = node_id
                    return result

        def replan_failure():
            nonlocal replans, plan, node_index
            if replans: return False
            plan = self.planner.plan(task, adapter, feedback=history[-3:], completed_results=values.model_view())
            if set(n['id'] for n in plan['nodes']) & completed: plan = fallback()
            replans, node_index = 1, 0
            history.append({'replan_reason': 'repeated_unchanged_failure'})
            return True

        def invoke(program, arguments, node_id):
            start = len(broker.events)
            was_terminal = broker.done
            if checkpoint: checkpoint.advance(checkpoint.state['stage'], program_started=True)
            result = self.worker.execute(program, arguments, broker)
            local = broker.check_local(arguments, result.get("outputs", {}), start) if result["status"] == "ok" else "unavailable"
            result["local_check"] = local
            if local == "failed" or result["status"] == "execution_error":
                blocked.add(program["id"])
                outcome = "execution_failure"
            elif result["status"] == "ok" and local == "passed":
                outcome = "positive"
            else:
                outcome = "normal"
            attempt = {"id": uuid4().hex, "program_id": program["id"], "task_key": task.physical_key,
                "origin": "online", "split": task.split, "outcome": outcome, "calls": len(broker.events) - start,
                "status": result["status"], "local_check": local, "basis": "local_check" if outcome == "positive" else None,
                "node": node_id, "outputs_consumed": False,
                "terminal_by_program": not was_terminal and broker.done}
            attempts.append(attempt)
            history.append({"program": program["id"], "arguments": arguments, "result": result})
            result_id = context.register(attempt['id'], result, name=program['id'], arguments=arguments)
            history[-1] = {'program': program['id'], 'result_id': result_id}
            attempt.update(result_id=result_id, native_event_start=start, native_event_end=len(broker.events))
            return result

        for _ in range(max(16, broker.call_limit * 4 + 16)):
            save(pending_step)
            if broker.done:
                reason = "environment_terminal"
                break
            if node_index >= len(plan["nodes"]):
                # A workflow is a strategy, not a guarantee. Continue from the
                # real state if it ends before the interactive task terminates.
                final_values, _ = values.resolve(plan.get('outputs', {}))
                if submission_kind == 'text' and ('answer' in final_values or submission(last_output, last_role)):
                    final = final_values.get('answer', last_output.get('answer'))
                    reason = 'plan_submitted'
                    break
                if submission_kind == 'files' and submission(last_output, last_role):
                    final, reason = final_values, 'plan_submitted'
                    break
                if final is not None or adapter.capabilities.interaction == "single_answer":
                    break
                plan = fallback()
                node_index = 0
            node = plan["nodes"][node_index]
            arguments, missing = values.resolve(node.get("args", {}))
            input_producers = {field: producers.get(ref.get('from')) or supplied_by.get((node['id'], field))
                               for field, ref in node.get('args', {}).items() if field in arguments}
            routes = [p for p in self.bank.routes(node) if p["id"] not in blocked]
            for program in routes if broker.remaining_calls() > 0 else []:
                route_key = digest({'node': node['id'], 'program': program['id'], 'args': arguments, 'state': progress_key(adapter)})
                if route_key in attempted: continue
                try:
                    validate_schema_instance(arguments, program["input_schema"])
                    if len(json.dumps(arguments, ensure_ascii=False).encode()) > self.worker.settings['max_rpc_message_bytes']:
                        missing['program_inputs'] = 'Use bounded read_result; resolved arguments exceed RPC limit'
                        continue
                except ValueError:
                    continue
                else:
                    if arguments:
                        input_reads.append(node['id'])
                    result = invoke(program, arguments, node["id"])
                    consumed.update(producer for producer in input_producers.values() if producer)
                    attempted.add(route_key)
                    attempted.add(digest({'node': node['id'], 'program': program['id'], 'args': arguments, 'state': progress_key(adapter)}))
                    if result["status"] == "ok" and result["local_check"] != "failed":
                        values.publish(node["id"], result["outputs"], "program")
                        producers[node['id']] = attempts[-1]['id']
                        completed.add(node["id"])
                        last_output, last_role = result['outputs'], program.get('result_role', 'intermediate')
                        node_index += 1
                    break
                    # Normal non-result is useful feedback, not a rollback.
                    # Non-result only blocks this unchanged call, not the
                    # Program version or a later call with different inputs.
            if node['id'] in completed: continue
            if broker.remaining_calls() <= 0:
                reason = "tool_budget_exhausted"
                if broker.done: reason = 'environment_terminal'
                elif submission_kind == 'text':
                    if submission(last_output, last_role):
                        final = last_output['answer']
                    else:
                        try:
                            step = self.agent('runtime', 'Submit your final answer using only the information already obtained. No tools, programs or replanning.',
                                {'original_goal': task.goal, 'public_state': public_view(adapter),
                                 'working_memory': context.model_memory(), 'recent': recent(),
                                 'completed_results': values.model_view()}, 'finish_answer', FINISH, repair_limit=0)
                            final, reason = step['answer'], 'finish_only'
                        except ValueError as exc:
                            history.append({'error': str(exc)})
                            reason = 'finish_only_protocol_error'
                elif submission_kind == 'files': final = {}
                break
            state = public_view(adapter)
            skill = self.bank.get(node.get('skill_id', ''))
            guidance = [skill] if skill else [s for s in self.bank.retrieve(node['goal']) if 'guidance' in s][:3]
            materials = {"original_goal": task.goal, "node_goal": node["goal"], "inputs": arguments,
                "missing": missing, "public_state": state, "tools": broker.available_tools(),
                "programs": self.bank.program_options(task.goal, node=node, allow_candidate=not self.frozen, excluded=blocked),
                'guidance': [{'goal': s.get('goal'), 'guidance': s.get('guidance','')} for s in guidance],
                'working_memory': context.model_memory(),
                "recent": recent(), "remaining_calls": broker.remaining_calls(),
                'allowed_calls': '1-3 independent read_only/batchable calls; otherwise one operation' if any(t.get('batchable') for t in broker.available_tools()) else 'one operation',
                "may_replan": replans == 0, "completed_results": values.model_view()}
            try:
                if pending_step is not None:
                    step = RuntimeDecision(tuple(pending_step['actions']), tuple(pending_step['call_ids']), pending_step['response_id']) if 'actions' in pending_step else pending_step
                else:
                    step = self.agent("runtime", RUNTIME_PROMPT, materials, "runtime_step", STEP, repair_limit=1, validator=validate_step)
                save(step)
                pending_step = None
            except ValueError as exc:
                reason = "runtime_protocol_error"
                history.append({"error": str(exc)})
                break
            if isinstance(step, RuntimeDecision):
                if len(step.actions) > 1:
                    # The response has been validated in full before any dispatch.
                    for action in step.actions: validate_step(action)
                    repeated = False
                    for index, (action, call_id) in enumerate(zip(step.actions, step.call_ids)):
                        event_id = step.response_id+':'+call_id
                        already = next((e for e in broker.events if e.get('event_id') == event_id and e['state']=='finished'), None)
                        if not already and broker.remaining_calls() <= 0:
                            history.extend({'batch_id': step.response_id, 'call_id': c, 'status': 'not_executed_budget'} for c in step.call_ids[index:])
                            break
                        args = context.bind(action.get('arguments', {}), action.get('argument_refs', {}))
                        key = digest({'action': 'call_tool', 'name': action['name'], 'arguments': args, 'state': progress_key(adapter)})
                        if not already and failures.get(key, 0) >= 2:
                            history.extend({'batch_id': step.response_id, 'call_id': c, 'status': 'not_executed_failure_limit'} for c in step.call_ids[index:])
                            repeated = True
                            break
                        try:
                            result = broker.call(action['name'], args, event_id=event_id, batch_id=step.response_id, call_id=call_id)
                        except UnknownSideEffect:
                            history.extend({'batch_id': step.response_id, 'call_id': c, 'status': 'not_executed_infrastructure'} for c in step.call_ids[index+1:])
                            save(step)
                            raise
                        event = already or broker.events[-1]
                        if not any(h.get('result_id') == event['result_id'] for h in history):
                            history.append({'tool': action['name'], 'result_id': event['result_id']})
                        context.results[event['result_id']] = result
                        context.sources[event['event_id']] = event['result_id']
                        if not already: failures[key] = 0 if result.get('accepted') else failures.get(key, 0) + 1
                        save(step)
                    if repeated and not replan_failure():
                        reason = 'repeated_unchanged_failure'
                        break
                    continue
                step = step.actions[0]
            dispatch_id = None
            saved_decision = checkpoint.state.get('executor_state', {}).get('pending_step') if checkpoint else None
            if saved_decision and 'response_id' in saved_decision:
                dispatch_id = saved_decision['response_id']+':'+saved_decision['call_ids'][0]
            action = step["action"]
            used = set(step.get('used_inputs', [])) & set(arguments)
            replaced = set(step.get('replaced_inputs', [])) & set(arguments)
            if used - replaced:
                input_reads.append(node['id'])
            consumed.update(input_producers[field] for field in used - replaced if input_producers.get(field))
            for field in replaced:
                producer = input_producers.get(field)
                if producer:
                    consumed.discard(producer)
                replaced_inputs.setdefault(node['id'], []).append(field)
            if action == "finish":
                final, reason = step.get("answer"), "agent_submitted"
                break
            if action == "revise_plan":
                if replans:
                    reason = "plan_revision_limit"
                    break
                try:
                    revised = self.planner.validate(step["workflow"], completed)
                    if set(n["id"] for n in revised["nodes"]) & completed:
                        raise ValueError("Revision must contain only unfinished nodes")
                except (ValueError, KeyError) as exc:
                    history.append({"error": str(exc)})
                    replans += 1
                    plan, node_index = fallback(), 0
                    continue
                history.append({"plan_revision": revised, "completed_results": sorted(completed)})
                replans, plan, node_index = replans + 1, revised, 0
                continue
            if action == "complete_node":
                outputs = context.bind(step.get('outputs', {}), step.get('output_refs', {}))
                interface = self.bank.get(node.get('skill_id', '')) or self.bank.get(node.get('program_id', ''))
                if interface:
                    try:
                        validate_schema_instance(outputs, interface['output_schema'])
                    except ValueError as exc:
                        history.append({'error': str(exc)})
                        continue
                values.publish(node["id"], step.get('outputs', {}), "agent", step.get('output_refs'))
                last_output, last_role = outputs, (interface or {}).get('result_role', 'intermediate')
                completed.add(node["id"])
                node_index += 1
                continue
            name = step.get('name', '')
            call_args = context.bind(step.get('arguments', {}), step.get('argument_refs', {}))
            key = digest({"action": action, "name": name, "arguments": call_args, "state": progress_key(adapter)})
            if failures.get(key, 0) >= 2:
                if replan_failure(): continue
                reason = "repeated_unchanged_failure"
                break
            try:
                if action == "call_program":
                    program = self.bank.get(name)
                    if not program or name in blocked or program.get("state") == "disabled" or (
                            self.frozen and program.get("state") != "usable"):
                        raise ValueError("Program is unavailable in this task")
                    result = invoke(program, call_args, node["id"])
                    if result["status"] == "ok" and result["local_check"] != "failed":
                        if name in {route['id'] for route in routes}:
                            values.publish(node['id'], result['outputs'], 'program')
                            producers[node['id']] = attempts[-1]['id']
                            completed.add(node['id'])
                            node_index += 1
                            last_output, last_role = result['outputs'], program.get('result_role', 'intermediate')
                            continue
                        # Preparation can supply the current node's missing
                        # inputs, then its authorized program runs automatically.
                        supplied = result["outputs"]
                        expected = set(node.get("args", {}))
                        for route in routes: expected.update(route['input_schema'].get('properties', {}))
                        if skill: expected.update(skill['input_schema'].get('properties', {}))
                        mapping = step.get('output_mapping', {k: k for k in supplied if k in expected})
                        if any(source not in supplied or target not in expected for source, target in mapping.items()) or len(set(mapping.values())) != len(mapping):
                            raise ValueError('Invalid preparation output_mapping')
                        if any(target in arguments and target not in replaced for target in mapping.values()):
                            raise ValueError('Replacing a known input requires replaced_inputs')
                        node.setdefault('args', {}).update({target: {'literal': supplied[source]} for source, target in mapping.items()})
                        supplied_by.update({(node['id'], target): attempts[-1]['id'] for target in mapping.values()})
                        failures[key] = 0
                    else:
                        failures[key] = failures.get(key, 0) + 1
                elif action == "call_tool":
                    result = broker.call(name, call_args, event_id=dispatch_id)
                    event = next(e for e in reversed(broker.events) if not dispatch_id or e.get('event_id')==dispatch_id)
                    context.results[event['result_id']] = result
                    context.sources[event['event_id']] = event['result_id']
                    history.append({"tool": name, 'result_id': event['result_id']})
                    failures[key] = 0 if result.get("accepted") else failures.get(key, 0) + 1
                else:
                    raise ValueError("Invalid runtime action")
            except (ValueError, PermissionError) as exc:
                failures[key] = failures.get(key, 0) + 1
                history.append({"error": str(exc), "name": name, "arguments": call_args})
        else:
            reason = "decision_budget_exhausted"
        final_values, _ = values.resolve(plan.get("outputs", {}))
        if final is None:
            for ref in plan.get('outputs', {}).values():
                if ref.get('from') in producers and ref.get('field') in values.results.get(ref['from'], {}):
                    consumed.add(producers[ref['from']])
        for attempt in attempts:
            attempt["outputs_consumed"] = attempt['id'] in consumed
        result = {"prediction": final if final is not None else final_values, "reason": reason,
                "values": values.history, "history": history, "attempts": attempts,
                "plan": plan, "plan_revisions": replans, "input_reads": input_reads,
                "replaced_inputs": replaced_inputs}
        if checkpoint: checkpoint.advance(checkpoint.state['stage'], executor_finished=result)
        return result
