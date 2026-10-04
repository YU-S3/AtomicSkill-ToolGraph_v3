"""Program-first execution, ordinary values and one task-local plan revision."""
from uuid import uuid4

from .contracts import ValueStore, digest, validate_schema_instance
from .planner import dynamic
from .prompts import RUNTIME_PROMPT, STEP


class Executor:
    def __init__(self, bank, agent, worker, planner, *, frozen=False):
        self.bank, self.agent, self.worker, self.planner = bank, agent, worker, planner
        self.frozen = frozen

    def run(self, task, adapter, broker, plan):
        values, attempts = ValueStore(task), []
        blocked, attempted, failures, history = set(), set(), {}, []
        completed, replans, node_index, final = set(), 0, 0, None
        producers, supplied_by, consumed = {}, {}, set()
        input_reads = []
        dynamic_sequence = 0
        reason = "completed"

        def fallback():
            nonlocal dynamic_sequence
            result = dynamic(task)
            while True:
                dynamic_sequence += 1
                node_id = 'dynamic_' + str(dynamic_sequence)
                if node_id not in completed:
                    result['nodes'][0]['id'] = node_id
                    return result

        def invoke(program, arguments, node_id):
            start = len(broker.events)
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
                "node": node_id, "outputs_consumed": False}
            attempts.append(attempt)
            history.append({"program": program["id"], "arguments": arguments, "result": result})
            return result

        for _ in range(max(16, broker.call_limit * 4 + 16)):
            if broker.done:
                reason = "environment_terminal"
                break
            if node_index >= len(plan["nodes"]):
                # A workflow is a strategy, not a guarantee. Continue from the
                # real state if it ends before the interactive task terminates.
                if final is not None or adapter.capabilities.interaction == "single_answer":
                    break
                plan = fallback()
                node_index = 0
            node = plan["nodes"][node_index]
            arguments, missing = values.resolve(node.get("args", {}))
            for field, ref in node.get("args", {}).items():
                if field in arguments:
                    producer = producers.get(ref.get('from')) or supplied_by.get((node['id'], field))
                    if producer:
                        consumed.add(producer)
            routes = [p for p in self.bank.routes(node) if p["id"] not in blocked]
            if not missing and routes and digest({"node": node["id"], "program": routes[0]["id"], "args": arguments,
                                                 "state": broker.observe()}) not in attempted:
                program = routes[0]
                try:
                    validate_schema_instance(arguments, program["input_schema"])
                except ValueError:
                    missing["program_inputs"] = program["input_schema"]
                else:
                    if arguments:
                        input_reads.append(node['id'])
                    result = invoke(program, arguments, node["id"])
                    attempted.add(digest({"node": node["id"], "program": program["id"], "args": arguments, "state": broker.observe()}))
                    if result["status"] == "ok" and result["local_check"] != "failed":
                        values.publish(node["id"], result["outputs"], "program")
                        producers[node['id']] = attempts[-1]['id']
                        completed.add(node["id"])
                        node_index += 1
                        continue
                    # Normal non-result is useful feedback, not a rollback.
                    # Non-result only blocks this unchanged call, not the
                    # Program version or a later call with different inputs.
            if broker.remaining_calls() <= 0:
                reason = "tool_budget_exhausted"
                break
            state = broker.observe()
            materials = {"original_goal": task.goal, "node_goal": node["goal"], "inputs": arguments,
                "missing": missing, "public_state": state, "tools": broker.available_tools(),
                "programs": [{k: p[k] for k in ("id", "state", "input_schema", "output_schema")}
                    for p in self.bank.all("program") if p["state"] == "usable" or (
                        not self.frozen and p["state"] == "candidate")][:8],
                "recent": history[-3:], "remaining_calls": broker.remaining_calls(),
                "may_replan": replans == 0, "completed_results": values.results}
            try:
                if arguments:
                    input_reads.append(node['id'])
                step = self.agent("runtime", RUNTIME_PROMPT, materials, "runtime_step", STEP, repair_limit=1)
            except ValueError as exc:
                reason = "runtime_protocol_error"
                history.append({"error": str(exc)})
                break
            action = step["action"]
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
                values.publish(node["id"], step.get("outputs", {}), "agent")
                completed.add(node["id"])
                node_index += 1
                continue
            name, call_args = step.get("name", ""), step.get("arguments", {})
            key = digest({"action": action, "name": name, "arguments": call_args, "state": state})
            if failures.get(key, 0) >= 2:
                if replans == 0:
                    plan = self.planner.plan(task, adapter, feedback=history[-3:], completed_results=values.results)
                    if set(n['id'] for n in plan['nodes']) & completed:
                        plan = fallback()
                    replans, node_index = 1, 0
                    history.append({"replan_reason": "repeated_unchanged_failure"})
                    continue
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
                        # Preparation can supply the current node's missing
                        # inputs, then its authorized program runs automatically.
                        supplied = result["outputs"]
                        expected = set(node.get("args", {}))
                        if routes:
                            expected.update(routes[0]["input_schema"].get("properties", {}))
                        node.setdefault("args", {}).update({k: {"literal": v} for k, v in supplied.items()
                                                            if k in expected})
                        supplied_by.update({(node['id'], k): attempts[-1]['id'] for k in supplied if k in expected})
                        failures[key] = 0
                    else:
                        failures[key] = failures.get(key, 0) + 1
                elif action == "call_tool":
                    result = broker.call(name, call_args)
                    history.append({"tool": name, "arguments": call_args, "result": result})
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
        return {"prediction": final if final is not None else final_values, "reason": reason,
                "values": values.history, "history": history, "attempts": attempts,
                "plan": plan, "plan_revisions": replans, "input_reads": input_reads}
