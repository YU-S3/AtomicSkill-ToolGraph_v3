"""The empirical production chain; no legacy System is constructed."""
from dataclasses import asdict
from pathlib import Path
import json
import os
from uuid import uuid4

from ..agents.protocol import AgentTurn, NativeToolSpec, validate_schema_instance
from ..agents.provider import OpenAICompatibleConfig, OpenAICompatibleProvider, ProviderAgentProtocolError
from ..agents.usage import UsageLedger
from ..agents.session import repair_messages
from ..core.errors import BudgetExhausted, FailureLayer
from ..harness.simple_protocol import Broker, EpisodeResult
from . import PROFILE
from .bank import Bank
from .contracts import PublicTask, digest
from .executor import Executor
from .learner import Learner
from .planner import Planner, dynamic
from .program_worker import ProgramWorker


STAGE_BUCKETS = {"planner": "planner_p1", "runtime": "runtime_dynamic", "extractor": "extractor_e1",
                "tool_builder": "tool_builder_evolution"}


def validate_config(config):
    if config.get("mechanism_profile") != PROFILE:
        raise ValueError("Empirical System requires its explicit profile")
    config = dict(config)
    if config.get("schema_version") != "empirical.v1":
        raise ValueError("Empirical System requires a fresh empirical.v1 schema")
    allowed = {'mechanism_profile', 'schema_version', 'data_dir', 'experiment', 'harness', 'runtime',
               'learning', 'planning', 'program_worker', 'program_environment', 'llm', 'benchmark_profile', 'manifest'}
    if set(config) - allowed:
        raise ValueError('Legacy or unsupported configuration keys: ' + ', '.join(sorted(set(config) - allowed)))
    if not config.get('llm', {}).get('model') or not config['llm'].get('api_key_env'):
        raise ValueError('Actual model ID and key environment variable name are required')
    experiment = config.get('experiment', {})
    if experiment.get('seed') not in {42, 43, 44} or experiment.get('runtime_mode') not in {'online', 'frozen'} or not experiment.get('output_dir'):
        raise ValueError('Explicit seed, output directory and online/frozen mode are required')
    if not config.get('harness', {}).get('adapter') or not config.get('runtime', {}).get('global_action_budget'):
        raise ValueError('Benchmark adapter and native call budget are required')
    if not config.get('benchmark_profile') or not config.get('manifest'):
        raise ValueError('Explicit benchmark profile and fixed manifest are required')
    environment = config.get('program_environment', {})
    if not environment.get('adapter_abi') or not environment.get('image_digest'):
        raise ValueError('Lock the Adapter ABI and container image before running')
    for section, limits in {"learning": {"max_new_programs_per_task": 1, "builder_repair_limit": 1,
        "max_train_test_cases_per_program_version": 2}, "planning": {"retrieval_top_k": 8,
        "structural_repair_limit": 1, "task_replan_limit": 1}, "runtime": {"repeated_unchanged_failure_limit": 2}}.items():
        for key, expected in limits.items():
            if config.get(section, {}).get(key, expected) != expected:
                raise ValueError("Unsupported empirical policy " + section + "." + key)
    return config


class EmpiricalSystem:
    def __init__(self, config, *, harness=None, provider=None, readonly=None, adapter_factory=None):
        self.config = validate_config(config)
        self.readonly = bool(self.config.get("experiment", {}).get("runtime_mode") == "frozen") if readonly is None else readonly
        self.bank = Bank(self.config["data_dir"], readonly=self.readonly,
                         seed=self.config.get("experiment", {}).get("seed", 42))
        self.adapter_factory = adapter_factory
        if harness is None:
            from ..harness.registry import create_simple_harness
            self.adapter_factory = self.adapter_factory or (lambda: create_simple_harness(self.config))
            harness = self.adapter_factory()
        self.adapter = harness
        self.provider_override = provider
        self.providers = {}
        self.usage = UsageLedger()
        self.requests = []
        self.planner = Planner(self.bank, self.agent)
        self.worker = ProgramWorker(self.config.get("program_worker"))
        self.executor = Executor(self.bank, self.agent, self.worker, self.planner, frozen=self.readonly)
        self.learner = Learner(self)
        self._task_start = 0
        self._runtime_start = 0
        self._learning_start = None
        self.audit_path = None
        self._request_start = 0
        self.checkpoint = None
        self.phase = 'train'
        self.budget_scope = None
        self.prior_usage = []

    def _budget(self, stage):
        if stage == 'runtime':
            prefix, cap = {'runtime_dynamic'}, self.config['llm'].get('runtime', {}).get('max_total_tokens_per_task', 600000)
        elif stage == 'planner':
            prefix, cap = {'planner_p1'}, self.config['llm'].get('planner', {}).get('max_total_tokens_per_phase', 120000)
        else:
            prefix, cap = {'extractor_e1','tool_builder_evolution'}, self.config['llm'].get('extractor', {}).get('max_total_tokens_per_task', 262144)
        usage = {e['event_id']: e for e in self.prior_usage}
        usage.update({e.event_id: e.to_dict() for e in self.usage.events[self._task_start:]})
        used = sum(e['total_tokens'] for e in usage.values() if e['bucket'] in prefix and
            e.get('provider_metadata', {}).get('budget_scope') == self.budget_scope)
        return used, cap

    def _save_requests(self):
        if self.audit_path is not None:
            path = Path(self.audit_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + '.tmp')
            prior = json.loads(path.read_text()) if path.exists() else {'requests': [], 'usage': []}
            requests = {r['id']: r for r in prior['requests']}
            for record in self.requests[self._request_start:]:
                requests[record['id']] = {**requests.get(record['id'], {}), **record}
            usage = {e['event_id']: e for e in prior['usage']}
            usage.update({e.event_id: {**e.to_dict(), 'phase': e.provider_metadata.get('phase')} for e in self.usage.events[self._task_start:]})
            temporary.write_text(json.dumps({'requests': list(requests.values()),
                'usage': list(usage.values())}, ensure_ascii=False), encoding='utf-8')
            os.replace(temporary, path)

    def provider(self, stage):
        if self.provider_override is not None:
            return self.provider_override.get(stage) if isinstance(self.provider_override, dict) else self.provider_override
        if stage not in self.providers:
            llm = self.config["llm"]
            settings = {**llm, **llm.get(stage, {})}
            self.providers[stage] = OpenAICompatibleProvider(OpenAICompatibleConfig(
                base_url=llm["base_url"], model=llm["model"], api_key_env=llm["api_key_env"],
                dialect=llm.get('dialect', 'deepseek_v4_chat'),
                input_modalities=tuple(llm.get('input_modalities', ['text'])),
                token_limit_field=llm.get('token_limit_field', 'max_tokens'),
                max_completion_tokens=settings["max_completion_tokens"],
                thinking_type=llm.get("protocol", {}).get("thinking_type", "enabled"),
                reasoning_effort=settings.get("reasoning_effort", "high"),
                request_timeout_seconds=settings.get("request_timeout_seconds", 180),
                max_retries=llm.get("max_retries", 4)))
        return self.providers[stage]

    def agent(self, stage, prompt, materials, name, schema, *, validator=None, repair_limit=0):
        provider = self.provider(stage)
        tools = [NativeToolSpec(name, "Submit the requested result", schema)] if name else []
        messages = [{"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(materials, ensure_ascii=False, allow_nan=False)}]
        content_parts = materials.get('content_parts') if isinstance(materials, dict) else None
        if content_parts:
            if 'image' not in self.config.get('llm', {}).get('input_modalities', ['text']):
                raise ValueError('Model capability lock does not support image input')
            messages[1]['content'] = [{'type': 'text', 'text': json.dumps(
                {k: v for k, v in materials.items() if k != 'content_parts'}, ensure_ascii=False)}, *content_parts]
        for repair in range(repair_limit + 1):
            used, cap = self._budget(stage)
            if used >= cap:
                raise BudgetExhausted('empirical_token_budget_exhausted', 'Role token budget exhausted', layer=FailureLayer.RUNTIME_AGENT)
            turn = None
            response_key = digest({'stage': stage, 'phase': self.phase, 'budget_scope': self.budget_scope,
                                   'messages': messages, 'tools': [t.to_openai() for t in tools]})
            recovered = self.checkpoint.response(response_key) if self.checkpoint else None
            request_id = recovered['request_id'] if recovered else uuid4().hex
            provider_offset = getattr(provider, "request_record_count", 0)
            record = {"id": request_id, "stage": stage, "repair": repair,
                      "phase": self.phase, 'budget_scope': self.budget_scope,
                      "messages": messages, "tools": [t.to_openai() for t in tools]}
            # Trace messages omit replay-private reasoning. The immediate repair
            # retains the actual assistant envelope in process only.
            record["messages"] = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in messages]
            self.requests.append(record)
            self._save_requests()
            try:
                if recovered:
                    request_id = recovered['request_id']
                    record['id'] = request_id
                    turn = AgentTurn(**recovered['turn'])
                    record['recovered_response'] = True
                else:
                    turn = provider.complete(messages, tools=tools)
                    if self.checkpoint:
                        self.checkpoint.save_response(response_key, {'turn': asdict(turn), 'request_id': request_id})
            except Exception as exc:
                turn = getattr(exc, "usage_turn", None)
                if turn is not None:
                    turn.provider_metadata['phase'] = self.phase
                    turn.provider_metadata['budget_scope'] = self.budget_scope
                    self.usage.record_turn(session_id=request_id, turn_index=0, bucket=STAGE_BUCKETS[stage], turn=turn)
                record["error"] = str(exc)
                self._save_requests()
                if isinstance(exc, ProviderAgentProtocolError):
                    if repair < repair_limit and turn.finish_reason != 'length':
                        messages.append({'role': 'user', 'content': 'Repair only the invalid ToolCall JSON: ' + str(exc)[:2048]})
                        continue
                    raise ValueError(str(exc)) from exc
                raise
            finally:
                records = getattr(provider, "request_records_since", None)
                if records and not recovered:
                    record["http_attempts"] = list(records(provider_offset))
                self._save_requests()
            prior_usage = json.loads(Path(self.audit_path).read_text())['usage'] if self.audit_path and Path(self.audit_path).exists() else []
            if not any(e['session_id'] == request_id for e in prior_usage):
                turn.provider_metadata['phase'] = self.phase
                turn.provider_metadata['budget_scope'] = self.budget_scope
                self.usage.record_turn(session_id=request_id, turn_index=0, bucket=STAGE_BUCKETS[stage], turn=turn)
            record["response"] = {"content": turn.content, "finish_reason": turn.finish_reason,
                "tool_calls": [{"id": c.call_id, "name": c.name, "arguments": c.arguments} for c in turn.tool_calls],
                "usage": {k: getattr(turn, k) for k in ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens")}}
            self._save_requests()
            if self._budget(stage)[0] > cap:
                raise BudgetExhausted('empirical_token_budget_exhausted', 'Metered turn exceeded role budget', layer=FailureLayer.RUNTIME_AGENT)
            try:
                if not name:
                    return turn.content
                if len(turn.tool_calls) != 1 or turn.tool_calls[0].name != name:
                    raise ValueError("Expected one " + name + " ToolCall")
                value = turn.tool_calls[0].arguments
                validate_schema_instance(value, schema)
                if validator:
                    validator(value)
                return value
            except ValueError as exc:
                if repair == repair_limit or (turn.finish_reason == 'length' and not turn.tool_calls):
                    raise
                messages.extend(repair_messages(turn, exc))
                messages.append({"role": "user", "content": "Repair only the invalid structure. Preserve the requested goal and valid content. "
                    "If you returned JSON as text, submit those same arguments through the requested ToolCall: " + str(exc)[:2048]})

    def run_task(self, task, *, learn=None, attempt_id=None):
        if not isinstance(task, PublicTask):
            raise TypeError("Empirical System accepts only PublicTask")
        learn = not self.readonly if learn is None else learn
        if learn and (self.readonly or task.split != "train"):
            raise RuntimeError("Only Train may update the empirical Bank")
        before = self.bank.digest()
        self._task_start, request_start = len(self.usage.events), len(self.requests)
        self._runtime_start = self._task_start
        self._request_start = request_start
        trace = {"schema": "empirical.trace.v1", "task": asdict(task), "attempt_id": attempt_id or uuid4().hex,
                 "knowledge_before": before, "tools": [], "execution": {}, "score": None}
        self.budget_scope = trace['attempt_id']
        self.prior_usage = json.loads(Path(self.audit_path).read_text())['usage'] if self.audit_path and Path(self.audit_path).exists() else []
        self.phase = task.split
        if self.checkpoint and self.checkpoint.state['stage'] in {'task_execution_finished', 'learning_finished', 'task_committed'}:
            trace = self.checkpoint.state['trace']
            if learn and self.checkpoint.state['stage'] == 'task_execution_finished':
                self.learn_trace(task, trace)
                self.checkpoint.advance('learning_finished', trace=trace)
            trace['knowledge_after'] = self.bank.digest()
            self._save_requests()
            if self.audit_path and Path(self.audit_path).exists():
                trace.update(json.loads(Path(self.audit_path).read_text()))
            return trace
        self.adapter.reset(task)
        broker = Broker(self.adapter, self.config.get('runtime', {}).get('global_action_budget', 100),
            step_limit=self.config.get('runtime', {}).get('environment_step_budget'),
            journal=self.checkpoint.native_events if self.checkpoint else None)
        try:
            if self.adapter.capabilities.interaction == "single_answer":
                # No extra planning solve, and no protocol repair/re-solving.
                answer = self.agent("runtime", "Answer the question once using only the public input and any supplied guidance.",
                    {"goal": task.goal, "inputs": task.inputs, "guidance": self.bank.retrieve(task.goal),
                     'content_parts': getattr(self.adapter, 'content_parts', lambda: [])()}, None, None)
                execution = {"prediction": answer, "reason": "single_answer", "attempts": []}
            else:
                plan = self.planner.plan(task, self.adapter)
                trace["initial_plan"] = plan
                execution = self.executor.run(task, self.adapter, broker, plan)
            sealed = self.adapter.submit(execution["prediction"])
            score = self.adapter.evaluate(sealed)
            trace.update(execution=execution, score=score, tools=broker.events)
            trace['episode_result'] = asdict(EpisodeResult(task.task_id, task.split, sealed, score,
                execution['reason'], cost_path=str(self.audit_path) if self.audit_path else None))
            trace['native_call_attempts'] = len(broker.events)
            trace['environment_steps'] = broker.environment_steps
            if self.checkpoint:
                self.checkpoint.advance('task_execution_finished', trace=trace)
            if learn:
                self.learn_trace(task, trace)
            if self.checkpoint:
                self.checkpoint.advance('learning_finished', trace=trace)
        except BudgetExhausted as exc:
            trace["error"] = {"code": exc.code, "message": str(exc)}
            trace["tools"] = broker.events
            sealed = self.adapter.submit(None)
            trace["score"] = self.adapter.evaluate(sealed)
            trace['execution'].setdefault('attempts', [])
            trace['execution'].update(prediction=None, reason='token_budget_exhausted')
            trace['native_call_attempts'], trace['environment_steps'] = len(broker.events), broker.environment_steps
            trace['episode_result'] = asdict(EpisodeResult(task.task_id, task.split, sealed, trace['score'],
                'token_budget_exhausted', cost_path=str(self.audit_path) if self.audit_path else None))
            if self.checkpoint:
                self.checkpoint.advance('learning_finished', trace=trace)
        finally:
            trace["usage"] = [e.to_dict() for e in self.usage.events[self._task_start:]]
            trace["requests"] = self.requests[request_start:]
            self._save_requests()
            trace["knowledge_after"] = self.bank.digest()
            if self.audit_path and Path(self.audit_path).exists():
                trace['usage'] = json.loads(Path(self.audit_path).read_text())['usage']
            if self.readonly and trace["knowledge_after"] != before:
                raise RuntimeError("Frozen Bank changed during evaluation")
        return trace

    def learn_trace(self, task, trace):
        if self.readonly or task.split != 'train':
            raise RuntimeError('Only Train may learn')
        for attempt in trace["execution"]["attempts"]:
            if attempt["status"] == "ok" and attempt["outputs_consumed"] and trace["score"]["hard"] and attempt["local_check"] == "unavailable":
                attempt.update(outcome="positive", basis="task_outcome")
            self.bank.record(attempt)
        self._learning_start = len(self.usage.events)
        try:
            trace["learning"] = self.learner.learn(task, trace)
        except (ValueError, SyntaxError, BudgetExhausted) as exc:
            trace["learning"] = {"error": str(exc), "rejected": True}
        finally:
            self._learning_start = None

    def test_program(self, program, inputs, task, *, trial_id, trial_case=None):
        if self.adapter_factory is None:
            return {"id": trial_id, "program_id": program["id"], "task_key": task.physical_key,
                    "origin": "train_test", "outcome": "inapplicable", "error": "No isolated Adapter factory"}
        existing = next((a for a in self.bank.attempts(program['id']) if a['id'] == trial_id), None)
        if existing:
            return existing
        from .checkpoint import TaskCheckpoint
        trial_checkpoint = TaskCheckpoint(self.checkpoint.root / 'trials' / trial_id) if self.checkpoint else None
        if trial_checkpoint and trial_checkpoint.state.get('record'):
            self.bank.record(trial_checkpoint.state['record'])
            return trial_checkpoint.state['record']
        if trial_checkpoint and (trial_checkpoint.root / 'native_events.json').exists():
            record = {'id': trial_id, 'program_id': program['id'], 'task_key': task.physical_key,
                      'origin': 'train_test', 'outcome': 'inapplicable', 'basis': None,
                      'error': 'Interrupted trial side effects; no replay or positive credit'}
            self.bank.record(record)
            return record
        previous_phase = self.phase
        previous_scope = self.budget_scope
        self.phase = 'trial'
        self.budget_scope = trial_id
        adapter = self.adapter_factory()
        previous_runtime_start = self._runtime_start
        self._runtime_start = len(self.usage.events)
        broker = Broker(adapter, self.config.get('runtime', {}).get('global_action_budget', 100),
            step_limit=self.config.get('runtime', {}).get('environment_step_budget'),
            journal=trial_checkpoint.native_events if trial_checkpoint else None)
        try:
            adapter.reset(task)
            if trial_case:
                if trial_case.physical_task_key != task.physical_key or task.split != 'train':
                    raise ValueError('TrialCase physical task/split mismatch')
                for action in trial_case.prefix:
                    if broker.done or not broker.call(action['name'], action['arguments']).get('accepted'):
                        record = {'id': trial_id, 'program_id': program['id'], 'task_key': task.physical_key,
                                  'origin': 'train_test', 'outcome': 'inapplicable', 'basis': None,
                                  'error': 'Prefix could not reconstruct an active start state'}
                        self.bank.record(record)
                        return record
                if broker.done:
                    record = {'id': trial_id, 'program_id': program['id'], 'task_key': task.physical_key,
                              'origin': 'train_test', 'outcome': 'inapplicable', 'basis': None,
                              'error': 'Trial prefix reached terminal'}
                    self.bank.record(record)
                    return record
            start = len(broker.events)
            try:
                validate_schema_instance(inputs, program["input_schema"])
            except ValueError as exc:
                outcome, result, basis = "inapplicable", {"error": str(exc)}, None
            else:
                result = self.worker.execute(program, inputs, broker)
                local = broker.check_local(inputs, result.get("outputs", {}), start) if result["status"] == "ok" else "unavailable"
                result['local_check'] = local
                basis, outcome = None, "normal"
                if result["status"] == "execution_error" or local == "failed":
                    outcome = "execution_failure"
                elif result["status"] == "ok" and local == "passed":
                    basis, outcome = "local_check", "positive"
                elif result["status"] == "ok":
                    # Without a local oracle, run the remainder from the actual
                    # Train state; this is billed training, not replay credit.
                    executor = Executor(self.bank, self.agent, self.worker, self.planner)
                    continuation = dynamic(task)
                    continuation["nodes"][0]["args"] = {k: {"literal": v} for k, v in result.get("outputs", {}).items()}
                    rest = executor.run(task, adapter, broker, continuation)
                    result["continuation"] = rest
                    score = adapter.evaluate(adapter.submit(rest["prediction"]))
                    result["score"] = score
                    node_id = continuation['nodes'][0]['id']
                    if score["hard"] and result.get('outputs') and node_id in rest['input_reads'] and not rest['replaced_inputs'].get(node_id):
                        basis, outcome = "task_outcome", "positive"
            record = {"id": trial_id, "program_id": program["id"], "task_key": task.physical_key,
                      "origin": "train_test", "split": "train", "outcome": outcome,
                      "basis": basis, "calls": len(broker.events), "result": result, "tools": broker.events}
            if trial_checkpoint:
                trial_checkpoint.advance('trial_finished', record=record)
            self.bank.record(record)
            return record
        finally:
            self._runtime_start = previous_runtime_start
            self.phase = previous_phase
            self.budget_scope = previous_scope
            adapter.close()

    def close(self):
        self.adapter.close()
        self.bank.close()
