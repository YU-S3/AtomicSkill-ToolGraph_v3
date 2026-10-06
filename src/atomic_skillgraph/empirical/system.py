"""The empirical production chain; no legacy System is constructed."""
from dataclasses import asdict
from copy import deepcopy
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
from . import PROFILE, POLICY_DEFAULTS
from .bank import Bank
from .contracts import PublicTask, RuntimeDecision, digest
from .task_context import TaskContext
from .executor import Executor
from .learner import Learner
from .planner import Planner, dynamic
from .program_worker import ProgramWorker


STAGE_BUCKETS = {"planner": "planner_p1", "runtime": "runtime_dynamic", "extractor": "extractor_e1",
                "tool_builder": "tool_builder_evolution"}


def validate_config(config):
    if config.get("mechanism_profile") != PROFILE:
        raise ValueError("Empirical System requires its explicit profile")
    config = deepcopy(config)
    for section, defaults in POLICY_DEFAULTS.items():
        config[section] = {**defaults, **config.get(section, {})}
        for key, value in defaults.items():
            if config[section][key] != value:
                raise ValueError('Unsupported execution policy ' + section + '.' + key)
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
    if environment['adapter_abi'] not in {'simple.v1','simple.v2'}:
        raise ValueError('Unsupported Adapter ABI')
    environment['adapter_abi'] = 'simple.v2'
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
        self.observer = None
        self.task_context = None
        self.last_decision_id = None

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
            if self.observer:
                self.observer.requests(list(requests.values()))

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
                generation_seed=self.config['experiment']['seed'] if llm.get('generation_seed_supported') is True else None,
                max_completion_tokens=settings["max_completion_tokens"],
                thinking_type=llm.get("protocol", {}).get("thinking_type", "enabled"),
                reasoning_effort=settings.get("reasoning_effort", "high"),
                request_timeout_seconds=settings.get("request_timeout_seconds", 180),
                max_retries=llm.get("max_retries", 4)))
        return self.providers[stage]

    def agent(self, stage, prompt, materials, name, schema, *, validator=None, repair_limit=0,
              completion_override=None, repair_reason=None, job_key=None, owner_state_version=None):
        from .model_view import callable_tools
        if self.checkpoint and self.task_context:
            self.checkpoint.advance(self.checkpoint.state['stage'], model_context={k: getattr(self.task_context, k)
                for k in ('scope', 'results', 'sources', 'memory')})
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
        purpose = 'finish_only' if name == 'finish_answer' else {'tool_builder': 'builder'}.get(stage, stage)
        scope = json.dumps([str(self.checkpoint.root) if self.checkpoint else '', self.budget_scope, job_key])
        decision_id = self.checkpoint.prepare_decision(scope, purpose, owner_state_version) if self.checkpoint else uuid4().hex
        self.last_decision_id = decision_id
        runtime_owned = name == 'runtime_step' and owner_state_version is not None
        def commit(status, repair):
            if self.checkpoint:
                self.checkpoint.commit_decision(decision_id, status, repair_index=repair)
        def accepted(value, repair):
            if owner_state_version is None:
                commit('applied', repair)
            return value
        for repair in range(repair_limit + 1):
            used, cap = self._budget(stage)
            if used >= cap:
                raise BudgetExhausted('empirical_token_budget_exhausted', 'Role token budget exhausted', layer=FailureLayer.RUNTIME_AGENT)
            turn = None
            completion_cap = completion_override or self.config['llm'].get(stage, {}).get('max_completion_tokens', 32768)
            if stage == 'tool_builder': completion_cap = min(completion_cap, cap-used)
            response_key = digest({'scope': scope, 'logical_decision_id': decision_id, 'repair_index': repair,
                                   'stage': stage, 'phase': self.phase, 'budget_scope': self.budget_scope,
                                   'messages': messages, 'tools': [t.to_openai() for t in tools],
                                   'completion_cap': completion_cap, 'repair_reason': repair_reason, 'job_key': job_key,
                                   'model_identity': self.config['llm'],
                                   'policy': self.config['runtime']['plan_execution_policy'],
                                   'implementation_revision': self.config['experiment']['implementation_revision']})
            recovered = self.checkpoint.response(response_key) if self.checkpoint else None
            request_id = recovered['request_id'] if recovered else uuid4().hex
            provider_offset = getattr(provider, "request_record_count", 0)
            record = {"id": request_id, "stage": stage, "repair": repair,
                      'logical_decision_id': decision_id, 'decision_scope': scope,
                      'purpose': purpose, 'owner_state_version': owner_state_version,
                      "phase": self.phase, 'budget_scope': self.budget_scope,
                      "messages": messages, "tools": [t.to_openai() for t in tools]}
            record.update(completion_cap=completion_cap, repair_reason=repair_reason, job_key=job_key)
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
                    if 'http_attempts' in recovered:
                        record['http_attempts'] = recovered['http_attempts']
                    if recovered.get('protocol_error'):
                        raise ProviderAgentProtocolError('runtime_agent_schema_error', recovered['protocol_error'], usage_turn=turn)
                else:
                    kwargs = {'max_completion_tokens': completion_cap} if isinstance(provider, OpenAICompatibleProvider) or completion_override is not None else {}
                    turn = provider.complete(messages, tools=tools, **kwargs)
                    if self.checkpoint:
                        records = getattr(provider, 'request_records_since', None)
                        self.checkpoint.save_response(response_key, {'turn': asdict(turn), 'request_id': request_id,
                            'http_attempts': list(records(provider_offset)) if records else []})
                commit('response_received', repair)
            except Exception as exc:
                turn = getattr(exc, "usage_turn", None)
                if turn is not None:
                    turn.provider_metadata['phase'] = self.phase
                    turn.provider_metadata['budget_scope'] = self.budget_scope
                    prior = json.loads(Path(self.audit_path).read_text())['usage'] if self.audit_path and Path(self.audit_path).exists() else []
                    if not any(e['session_id'] == request_id for e in prior) and not any(e.session_id == request_id for e in self.usage.events):
                        self.usage.record_turn(session_id=request_id, turn_index=0, bucket=STAGE_BUCKETS[stage], turn=turn)
                    record['response'] = {'content': turn.content, 'finish_reason': turn.finish_reason,
                        'tool_calls': [{'id': c.call_id, 'name': c.name, 'arguments': c.arguments} for c in turn.tool_calls],
                        'usage': {k: getattr(turn, k) for k in ('prompt_tokens','completion_tokens','total_tokens','reasoning_tokens')}}
                    if self.checkpoint:
                        records = getattr(provider, 'request_records_since', None)
                        self.checkpoint.save_response(response_key, {'turn': asdict(turn), 'request_id': request_id,
                            'protocol_error': str(exc) if isinstance(exc, ProviderAgentProtocolError) else None,
                            'http_attempts': recovered.get('http_attempts', []) if recovered else list(records(provider_offset)) if records else []})
                record["error"] = str(exc)
                self._save_requests()
                if isinstance(exc, ProviderAgentProtocolError):
                    if repair < repair_limit and turn.finish_reason != 'length':
                        commit('prepared', repair + 1)
                        messages.append({'role': 'user', 'content': 'Repair only the invalid ToolCall JSON: ' + str(exc)[:2048]})
                        continue
                    failure = ValueError(str(exc))
                    failure.model_authored = True
                    failure.finish_reason = turn.finish_reason
                    failure.logical_decision_id = decision_id
                    if not runtime_owned: commit('rejected', repair)
                    raise failure from exc
                raise
            finally:
                records = getattr(provider, "request_records_since", None)
                if records and not recovered:
                    record["http_attempts"] = list(records(provider_offset))
                self._save_requests()
            prior_usage = json.loads(Path(self.audit_path).read_text())['usage'] if self.audit_path and Path(self.audit_path).exists() else []
            if not any(e['session_id'] == request_id for e in prior_usage) and not any(
                    e.session_id == request_id for e in self.usage.events):
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
                    return accepted(turn.content, repair)
                if name == 'runtime_step' and turn.tool_calls:
                    actions = []
                    for call in turn.tool_calls:
                        if call.name != name: raise ValueError('Unexpected Runtime ToolCall name')
                        validate_schema_instance(call.arguments, schema)
                        self.task_context.bind(call.arguments.get('arguments', {}), call.arguments.get('argument_refs', {})) if self.task_context else None
                        actions.append(call.arguments)
                    if len(actions) > 1:
                        specs = {t['name']: t for t in callable_tools(materials)}
                        if len(actions) > self.config['runtime']['read_batch_max_calls'] or any(
                            a['action'] != 'call_tool' or not specs.get(a.get('name'), {}).get('batchable') or
                            specs.get(a.get('name'), {}).get('effect') != 'read_only' for a in actions):
                            raise ValueError('Batch must contain at most 3 independent approved read-only calls')
                        for action in actions:
                            args = self.task_context.bind(action.get('arguments', {}), action.get('argument_refs', {})) if self.task_context else action.get('arguments', {})
                            validate_schema_instance(args, specs[action['name']]['input_schema'])
                    for action in actions:
                        if validator: validator(action)
                    return accepted(RuntimeDecision(tuple(actions), tuple(c.call_id for c in turn.tool_calls), request_id), repair)
                if len(turn.tool_calls) != 1 or turn.tool_calls[0].name != name:
                    raise ValueError("Expected one " + name + " ToolCall")
                value = turn.tool_calls[0].arguments
                validate_schema_instance(value, schema)
                if validator:
                    validator(value)
                return accepted(value, repair)
            except ValueError as exc:
                if repair == repair_limit or (turn.finish_reason == 'length' and not turn.tool_calls):
                    exc.model_authored = True
                    exc.finish_reason = turn.finish_reason
                    exc.logical_decision_id = decision_id
                    if not runtime_owned: commit('rejected', repair)
                    raise
                commit('prepared', repair + 1)
                if hasattr(exc, 'repair_material'):
                    messages[1] = {'role': 'user', 'content': json.dumps(exc.repair_material, ensure_ascii=False)}
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
        self.last_decision_id = None
        self.prior_usage = json.loads(Path(self.audit_path).read_text())['usage'] if self.audit_path and Path(self.audit_path).exists() else []
        self.phase = task.split
        self.planner.checkpoint = self.checkpoint
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
        self.task_context = TaskContext(self.config['runtime'])
        if self.checkpoint:
            for key, value in self.checkpoint.state.get('model_context', {}).items():
                setattr(self.task_context, key, value)
        broker = Broker(self.adapter, self.config.get('runtime', {}).get('global_action_budget', 100),
            step_limit=self.config.get('runtime', {}).get('environment_step_budget'),
            journal=self.checkpoint.native_events if self.checkpoint else None,
            observer=self.observer.native_observer(trace['attempt_id']) if self.observer else None,
            context=self.task_context)
        self.executor.checkpoint = self.checkpoint
        if self.checkpoint and self.checkpoint.state.get('program_started'):
            from ..harness.simple_protocol import UnknownSideEffect
            raise UnknownSideEffect('Interrupted Program/workspace execution requires explicit reconstruction')
        native_path = self.checkpoint.root/'native_events.json' if self.checkpoint else None
        if native_path and native_path.exists():
            prior_events = json.loads(native_path.read_text())
            specs = {t['name']: t for t in broker.available_tools()}
            if self.adapter.capabilities.checkpoint_mode != 'workspace_copy' or any(
                e['state'] != 'finished' or specs.get(e['name'], {}).get('effect') != 'read_only' for e in prior_events):
                from ..harness.simple_protocol import UnknownSideEffect
                raise UnknownSideEffect('Interrupted stateful or unknown operation requires explicit reconstruction')
            broker.events = prior_events
            broker.environment_steps = sum(e.get('environment_step', 0) for e in prior_events)
        try:
            if self.adapter.capabilities.interaction == "single_answer":
                # No extra planning solve, and no protocol repair/re-solving.
                answer = self.agent("runtime", "Answer the question once using only the public input and any supplied guidance. " +
                    getattr(self.adapter, 'answer_contract', lambda: '')(),
                    {"goal": task.goal, "inputs": task.inputs, "guidance": [{k: a.get(k,'') for k in ('goal','guidance')}
                        for a in self.bank.retrieve(task.goal) if 'guidance' in a][:3],
                     'content_parts': getattr(self.adapter, 'content_parts', lambda: [])()}, None, None,
                    owner_state_version='single_solver')
                execution = {"prediction": answer, "reason": "single_answer", "attempts": []}
            else:
                plan = self.checkpoint.state['executor_state']['plan'] if self.checkpoint and self.checkpoint.state.get('executor_state') else (
                    self.checkpoint.state.get('initial_plan') if self.checkpoint else None) or self.planner.plan(task, self.adapter)
                if self.checkpoint and not self.checkpoint.state.get('executor_state'):
                    self.checkpoint.commit_decision(getattr(self.planner, 'last_decision_id', None), 'applied', initial_plan=plan)
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
                self.checkpoint.commit_decision(self.last_decision_id if self.adapter.capabilities.interaction == 'single_answer'
                    else None, 'applied', stage='task_execution_finished', trace=trace)
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
                decision = self.checkpoint.state['decisions'].get(self.last_decision_id, {})
                self.checkpoint.commit_decision(self.last_decision_id if decision.get('status') in {'prepared','response_received'}
                    else None, 'rejected', stage='learning_finished', trace=trace)
        finally:
            native_ids = {e.get('result_id') for e in broker.events}
            trace['result_store'] = {'native_index': [{'result_id': e.get('result_id'), 'event_id': e.get('event_id'), 'index': e['index']}
                for e in broker.events if e.get('result_id')],
                'program_results': {rid: value for rid, value in broker.context.results.items() if rid not in native_ids},
                'local_reads': broker.context.local_reads}
            trace["usage"] = [e.to_dict() for e in self.usage.events[self._task_start:]]
            trace["requests"] = self.requests[request_start:]
            self._save_requests()
            trace["knowledge_after"] = self.bank.digest()
            if self.observer:
                trace['scoring_audit'] = getattr(self.adapter, 'score_audit', {})
            if self.audit_path and Path(self.audit_path).exists():
                trace['usage'] = json.loads(Path(self.audit_path).read_text())['usage']
            if self.readonly and trace["knowledge_after"] != before:
                raise RuntimeError("Frozen Bank changed during evaluation")
        return trace

    def learn_trace(self, task, trace):
        if self.readonly or task.split != 'train':
            raise RuntimeError('Only Train may learn')
        learning_observation = self.observer.learning_start(task) if self.observer else None
        for attempt in trace["execution"]["attempts"]:
            if attempt["status"] == "ok" and (attempt["outputs_consumed"] or attempt.get("terminal_by_program", False)) and trace["score"]["hard"] and attempt["local_check"] == "unavailable":
                attempt.update(outcome="positive", basis="task_outcome")
            self.bank.record(attempt)
        self._learning_start = len(self.usage.events)
        try:
            trace["learning"] = self.learner.learn(task, trace)
        except (ValueError, SyntaxError, BudgetExhausted) as exc:
            trace["learning"] = {"error": str(exc), "rejected": True}
        finally:
            if self.observer:
                self.observer.learning_end(learning_observation, trace.get('learning'))
            self._learning_start = None

    def test_program(self, program, inputs, task, *, trial_id, trial_case=None):
        if self.readonly or task.split != 'train': raise ValueError('Program trials require Train')
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
        previous_checkpoint = self.checkpoint
        previous_planner_checkpoint = getattr(self.planner, 'checkpoint', None)
        previous_scope = self.budget_scope
        previous_context = self.task_context
        self.task_context = TaskContext(self.config['runtime'])
        self.phase = 'trial'
        self.checkpoint = trial_checkpoint
        self.planner.checkpoint = trial_checkpoint
        self.budget_scope = trial_id
        adapter = self.adapter_factory()
        trial_observation = self.observer.trial_start(trial_id, task) if self.observer else None
        previous_runtime_start = self._runtime_start
        self._runtime_start = len(self.usage.events)
        broker = Broker(adapter, self.config.get('runtime', {}).get('global_action_budget', 100),
            step_limit=self.config.get('runtime', {}).get('environment_step_budget'),
            journal=trial_checkpoint.native_events if trial_checkpoint else None,
            observer=self.observer.native_observer(trial_id) if self.observer else None,
            context=self.task_context)
        try:
            inherit = getattr(adapter, 'inherit_discovery', None)
            if inherit:
                inherit(self.adapter)
            adapter.reset(task)
            if trial_observation:
                trial_observation['consumed'] = True
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
            was_terminal = broker.done
            workspace_before = adapter.observe().get('workspace', {})
            try:
                validate_schema_instance(inputs, program["input_schema"])
                if len(json.dumps(inputs, ensure_ascii=False).encode()) > self.worker.settings['max_rpc_message_bytes']:
                    raise ValueError('Program inputs exceed RPC limit')
                tools = getattr(adapter, 'tool_definitions', adapter.available_tools)()
                if not set(program['allowed_tools']).issubset({t['name'] for t in tools} | {'read_result'}):
                    raise ValueError('Required public tools are unavailable for this case')
            except ValueError as exc:
                outcome, result, basis = "inapplicable", {"error": str(exc)}, None
            else:
                result = self.worker.execute(program, inputs, broker)
                local = broker.check_local(inputs, result.get("outputs", {}), start) if result["status"] == "ok" else "unavailable"
                result['local_check'] = local
                result['terminal_by_program'] = not was_terminal and broker.done
                basis, outcome = None, "normal"
                if result['terminal_by_program']:
                    score = adapter.evaluate(adapter.submit(result.get('outputs', {})))
                    result['score'] = score
                    if score['hard'] and result['status'] == 'ok': basis, outcome = 'task_outcome', 'positive'
                    elif result['status'] == 'execution_error': outcome = 'execution_failure'
                elif result["status"] == "execution_error" or local == "failed":
                    outcome = "execution_failure"
                elif result["status"] == "ok" and local == "passed":
                    basis, outcome = "local_check", "positive"
                elif result["status"] == "ok":
                    role = program.get('result_role', 'intermediate')
                    readiness = {'result_role': role}
                    if role == 'final_files': readiness['previous_workspace'] = workspace_before
                    ready = getattr(adapter, 'submission_ready', lambda *a, **k: False)(result.get('outputs', {}), **readiness)
                    if ready:
                        outputs = result.get('outputs', {})
                        sealed = adapter.submit(outputs['answer'] if role == 'final_answer' else outputs)
                        score = adapter.evaluate(sealed)
                        result.update(score=score, submission='direct_program_output')
                        if score['hard']: basis, outcome = 'task_outcome', 'positive'
                    if role in {'final_answer', 'final_files'}:
                        record = {"id": trial_id, "program_id": program["id"], "task_key": task.physical_key,
                            "origin": "train_test", "split": "train", "outcome": outcome, "basis": basis,
                            "calls": len(broker.events), "result": result, "tools": broker.events}
                        if trial_checkpoint: trial_checkpoint.advance('trial_finished', record=record)
                        self.bank.record(record)
                        return record
                    # Without a local oracle, run the remainder from the actual
                    # Train state; this is billed training, not replay credit.
                    executor = Executor(self.bank, self.agent, self.worker, self.planner)
                    executor.checkpoint = trial_checkpoint
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
            if self.observer:
                self.observer.trial_end(trial_observation, broker.events, locals().get('record'), locals().get('result'),
                                        getattr(adapter, 'score_audit', {}))
            self._runtime_start = previous_runtime_start
            self.phase = previous_phase
            self.checkpoint = previous_checkpoint
            self.planner.checkpoint = previous_planner_checkpoint
            self.budget_scope = previous_scope
            self.task_context = previous_context
            adapter.close()

    def close(self):
        self.adapter.close()
        self.bank.close()
