"""The empirical production chain; no legacy System is constructed."""
from dataclasses import asdict
from pathlib import Path
import json
import os
import platform
from uuid import uuid4

from ..agents.protocol import NativeToolSpec, validate_schema_instance
from ..agents.provider import OpenAICompatibleConfig, OpenAICompatibleProvider, ProviderAgentProtocolError
from ..agents.usage import UsageLedger
from ..core.errors import BudgetExhausted, FailureLayer
from ..harness.simple_protocol import Broker
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
    config.setdefault("program_environment", {"backend": "sandbox_python_v1", "python": "3.12", "dependencies": []})
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

    def _save_requests(self):
        if self.audit_path is not None:
            path = Path(self.audit_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + '.tmp')
            temporary.write_text(json.dumps({'requests': self.requests[self._request_start:],
                'usage': [event.to_dict() for event in self.usage.events[self._task_start:]]}, ensure_ascii=False), encoding='utf-8')
            os.replace(temporary, path)

    def provider(self, stage):
        if self.provider_override is not None:
            return self.provider_override.get(stage) if isinstance(self.provider_override, dict) else self.provider_override
        if stage not in self.providers:
            llm = self.config["llm"]
            settings = {**llm, **llm.get(stage, {})}
            self.providers[stage] = OpenAICompatibleProvider(OpenAICompatibleConfig(
                base_url=llm["base_url"], model=llm["model"], api_key_env=llm["api_key_env"],
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
        for repair in range(repair_limit + 1):
            runtime_cap = self.config["llm"].get("runtime", {}).get("max_total_tokens_per_task", 600000)
            if stage == "runtime" and sum(e.usage.total_tokens for e in self.usage.events[self._runtime_start:]
                                          if e.bucket.value.startswith("runtime")) >= runtime_cap:
                raise BudgetExhausted("empirical_runtime_budget_exhausted", "task token budget exhausted", layer=FailureLayer.RUNTIME_AGENT)
            if stage in {'extractor', 'tool_builder'} and self._learning_start is not None and sum(
                    e.usage.total_tokens for e in self.usage.events[self._learning_start:]
                    if e.bucket.value in {'extractor_e1', 'tool_builder_evolution'}) >= self.config["llm"].get(
                    "extractor", {}).get("max_total_tokens_per_task", 262144):
                raise BudgetExhausted("extractor_token_budget_exhausted", "shared learning budget exhausted", layer=FailureLayer.RUNTIME_AGENT)
            if stage == 'planner' and sum(e.usage.total_tokens for e in self.usage.events[self._task_start:]
                    if e.bucket.value.startswith('planner')) >= self.config['llm'].get('planner', {}).get('max_total_tokens_per_phase', 120000):
                raise BudgetExhausted('planner_token_budget_exhausted', 'planning budget exhausted', layer=FailureLayer.RUNTIME_AGENT)
            turn = None
            request_id = uuid4().hex
            provider_offset = getattr(provider, "request_record_count", 0)
            record = {"id": request_id, "stage": stage, "repair": repair,
                      "messages": messages, "tools": [t.to_openai() for t in tools]}
            # Trace messages omit replay-private reasoning. The immediate repair
            # retains the actual assistant envelope in process only.
            record["messages"] = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in messages]
            self.requests.append(record)
            self._save_requests()
            try:
                turn = provider.complete(messages, tools=tools)
            except Exception as exc:
                turn = getattr(exc, "usage_turn", None)
                if turn is not None:
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
                if records:
                    record["http_attempts"] = list(records(provider_offset))
                self._save_requests()
            self.usage.record_turn(session_id=request_id, turn_index=0, bucket=STAGE_BUCKETS[stage], turn=turn)
            record["response"] = {"content": turn.content, "finish_reason": turn.finish_reason,
                "tool_calls": [{"id": c.call_id, "name": c.name, "arguments": c.arguments} for c in turn.tool_calls],
                "usage": {k: getattr(turn, k) for k in ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens")}}
            self._save_requests()
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
                assistant = dict(turn.replay_assistant_message)
                if assistant and turn.tool_calls:
                    messages.extend([assistant, {"role": "tool", "tool_call_id": turn.tool_calls[0].call_id,
                        "content": json.dumps({"error": str(exc)})}])
                messages.append({"role": "user", "content": "Repair only the invalid structure: " + str(exc)[:2048]})

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
        self.adapter.reset(task)
        broker = Broker(self.adapter, self.config.get("runtime", {}).get("global_action_budget", 100))
        try:
            if self.adapter.capabilities.interaction == "single_answer":
                # No extra planning solve, and no protocol repair/re-solving.
                answer = self.agent("runtime", "Answer the question once using only the public input and any supplied guidance.",
                    {"goal": task.goal, "inputs": task.inputs, "guidance": self.bank.retrieve(task.goal)}, None, None)
                execution = {"prediction": answer, "reason": "single_answer", "attempts": []}
            else:
                plan = self.planner.plan(task, self.adapter)
                trace["initial_plan"] = plan
                execution = self.executor.run(task, self.adapter, broker, plan)
            sealed = self.adapter.submit(execution["prediction"])
            score = self.adapter.evaluate(sealed)
            trace.update(execution=execution, score=score, tools=broker.events)
            if learn:
                for attempt in execution["attempts"]:
                    if attempt["status"] == "ok" and attempt["outputs_consumed"] and score["hard"] and attempt["local_check"] == "unavailable":
                        attempt.update(outcome="positive", basis="task_outcome")
                    self.bank.record(attempt)
                self._learning_start = len(self.usage.events)
                try:
                    trace["learning"] = self.learner.learn(task, trace)
                except (ValueError, SyntaxError, BudgetExhausted) as exc:
                    trace["learning"] = {"error": str(exc), "rejected": True}
                finally:
                    self._learning_start = None
        except BudgetExhausted as exc:
            trace["error"] = {"code": exc.code, "message": str(exc)}
            trace["tools"] = broker.events
            trace["score"] = self.adapter.evaluate(self.adapter.submit(None))
        finally:
            trace["usage"] = [e.to_dict() for e in self.usage.events[self._task_start:]]
            trace["requests"] = self.requests[request_start:]
            self._save_requests()
            trace["knowledge_after"] = self.bank.digest()
            if self.readonly and trace["knowledge_after"] != before:
                raise RuntimeError("Frozen Bank changed during evaluation")
        return trace

    def test_program(self, program, inputs, task, *, trial_id):
        if self.adapter_factory is None:
            return {"id": trial_id, "program_id": program["id"], "task_key": task.physical_key,
                    "origin": "train_test", "outcome": "inapplicable", "error": "No isolated Adapter factory"}
        adapter = self.adapter_factory()
        previous_runtime_start = self._runtime_start
        self._runtime_start = len(self.usage.events)
        broker = Broker(adapter, self.config.get("runtime", {}).get("global_action_budget", 100))
        try:
            adapter.reset(task)
            try:
                validate_schema_instance(inputs, program["input_schema"])
            except ValueError as exc:
                outcome, result, basis = "inapplicable", {"error": str(exc)}, None
            else:
                result = self.worker.execute(program, inputs, broker)
                local = broker.check_local(inputs, result.get("outputs", {}), 0) if result["status"] == "ok" else "unavailable"
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
                    if score["hard"] and result.get('outputs') and continuation['nodes'][0]['id'] in rest['input_reads']:
                        basis, outcome = "task_outcome", "positive"
            record = {"id": trial_id, "program_id": program["id"], "task_key": task.physical_key,
                      "origin": "train_test", "split": "train", "outcome": outcome,
                      "basis": basis, "calls": len(broker.events), "result": result, "tools": broker.events}
            self.bank.record(record)
            return record
        finally:
            self._runtime_start = previous_runtime_start
            adapter.close()

    def close(self):
        self.adapter.close()
        self.bank.close()
