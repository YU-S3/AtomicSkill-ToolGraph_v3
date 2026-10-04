"""Public tools and sealed submission; scoring is not a callable tool."""
from dataclasses import dataclass, field
import json
import queue
import threading
import time
from typing import Protocol

from ..agents.protocol import validate_schema_instance


@dataclass(frozen=True)
class Capabilities:
    interaction: str = "tool_loop"
    input_modalities: tuple = ("text",)
    tool_surface: str = "named_tools"
    checkpoint_mode: str = "none"
    local_check: str = "unavailable"
    final_submission_kind: str = 'environment'


@dataclass
class HarnessTask:
    task_id: str
    goal: str
    benchmark: str
    task_type: str = ""
    context: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class HarnessActionSpec:
    action_id: str
    revision: int
    action_type: str
    arguments: dict
    display_text: str
    raw_action: object
    metadata: dict


@dataclass
class HarnessActionResult:
    accepted: bool
    observation: str
    done: bool
    won: bool
    new_revision: int
    catalog: list
    metadata: dict = field(default_factory=dict)
    benchmark_score: float | None = None
    benchmark_reward: float | None = None


@dataclass(frozen=True)
class EpisodeResult:
    task_id: str
    split: str
    sealed_prediction: object
    score: dict
    termination_reason: str
    log_path: str | None = None
    cost_path: str | None = None


class UnknownSideEffect(RuntimeError):
    """The environment owner has not confirmed the last operation's result."""


class SimpleAdapter(Protocol):
    capabilities: Capabilities

    def reset(self, public_task): ...
    def observe(self): ...
    def available_tools(self): ...
    def call(self, tool_name, arguments): ...
    def submit(self, final_output): ...
    def evaluate(self, sealed_output): ...
    def close(self): ...


class Broker:
    """Shared task counter and the only host bridge exposed to a Program."""
    def __init__(self, adapter, call_limit, *, step_limit=None, journal=None, native_timeout=180, observer=None, context=None):
        from ..empirical.task_context import TaskContext
        self.adapter, self.call_limit = adapter, call_limit
        self.context = context or TaskContext()
        self.events = []
        self.done = False
        self.environment_steps = 0
        self.step_limit = step_limit
        self.journal = journal
        self.unknown = False
        self.native_timeout = native_timeout
        self.observer = observer
        self._lock = threading.Lock()
        self._leases = set()
        self._responses = {}

    def open_lease(self, invocation_id):
        self._leases.add(invocation_id)

    def close_lease(self, invocation_id):
        self._leases.discard(invocation_id)

    def rpc(self, invocation_id, sequence, method, values, *, deadline, allowed_tools):
        key = (invocation_id, sequence)
        fingerprint = json.dumps([method, values], sort_keys=True, allow_nan=False)
        if key in self._responses:
            prior, result = self._responses[key]
            if prior != fingerprint:
                raise ValueError("Conflicting duplicate RPC")
            return result
        if invocation_id not in self._leases or time.monotonic() >= deadline:
            raise RuntimeError("Invocation lease closed")
        if method == 'observe':
            result = self.observe()
        elif method == 'available_tools':
            result = [t for t in self.available_tools() if t['name'] in allowed_tools]
        elif method == 'remaining_calls':
            result = self.remaining_calls()
        elif method == 'call':
            result = self.call(values.get('name'), values.get('arguments'), deadline=deadline,
                               allowed_tools=allowed_tools)
        elif method == 'read_result':
            result = self.call('read_result', values, deadline=deadline, allowed_tools=allowed_tools)
        else:
            raise ValueError("Unknown broker RPC")
        self._responses[key] = (fingerprint, result)
        return result

    def observe(self):
        return self.adapter.observe()

    def available_tools(self):
        if self.done: return []
        tools = self.adapter.available_tools()
        if self.adapter.capabilities.interaction != 'single_answer':
            tools = [*tools, self.context.tool()]
        return tools

    def remaining_calls(self):
        return 0 if self.done or self.unknown else max(0, self.call_limit - len(self.events))

    def call(self, name, arguments, *, deadline=None, allowed_tools=None, event_id=None, batch_id=None, call_id=None):
        from uuid import uuid4
        from ..empirical.task_context import progress_key
        if event_id:
            prior = next((e for e in self.events if e.get('event_id') == event_id), None)
            if prior:
                if prior['name'] != name or prior['arguments'] != arguments:
                    raise ValueError('Conflicting completed call identity')
                if prior['state'] != 'finished':
                    raise UnknownSideEffect('Previous call result is unknown')
                return prior['result']
        if self.unknown:
            raise UnknownSideEffect("Unknown native side effect; attempt cannot continue")
        if self.remaining_calls() <= 0:
            raise RuntimeError("Task tool budget exhausted or environment terminated")
        with self._lock:
            deadline = deadline or time.monotonic() + self.native_timeout
            started_at = time.time()
            event = {"name": name, "arguments": arguments, "index": len(self.events), 'state': 'intent',
                     'event_id': event_id or uuid4().hex, 'batch_id': batch_id, 'call_id': call_id,
                     'backend_invoked': False, 'tool_call_consumed': True,
                     'progress_before': progress_key(self.adapter)}
            self.events.append(event)
            if self.observer:
                self.observer(event, started_at, None)
            if self.journal:
                self.journal(self.events)
            specs = {tool["name"]: tool for tool in self.available_tools()}
            try:
                if allowed_tools is not None and (name not in allowed_tools or name == 'execute_python'):
                    raise ValueError('Program tool is not authorized')
                if name not in specs:
                    raise ValueError("tool is not currently available")
                validate_schema_instance(arguments, specs[name]["input_schema"])
                if name != 'read_result' and self.step_limit is not None and self.environment_steps >= self.step_limit:
                    raise ValueError("Environment step budget exhausted")
            except ValueError as exc:
                result = {'accepted': False, 'observation': self.observe(), 'data': {},
                          'error': str(exc), 'done': self.done}
            else:
                if name == 'read_result':
                    try:
                        result = self.context.read(**arguments)
                    except ValueError as exc:
                        result = {'accepted': False, 'observation': '', 'data': {}, 'error': str(exc), 'done': False}
                    event['local_result_read'] = True
                    result['environment_step'] = 0
                    outcome = None
                else:
                    event['backend_invoked'] = True
                    outcome = queue.Queue(maxsize=1)
                def invoke():
                    try:
                        outcome.put((True, self.adapter.call(name, arguments)))
                    except BaseException as exc:
                        outcome.put((False, exc))
                owner = threading.Thread(target=invoke, daemon=True) if outcome is not None else None
                if owner: owner.start()
                try:
                    ok, value = outcome.get(timeout=max(0, deadline - time.monotonic()) if deadline else None) if owner else (True, result)
                except queue.Empty:
                    self.unknown = True
                    event['state'] = 'unknown'
                    if self.observer:
                        self.observer(event, started_at, time.time())
                    if self.journal:
                        self.journal(self.events)
                    # No Agent recovery is permitted on this owner, even if a
                    # late reply arrives. Close its episode before worker kill.
                    abort = getattr(self.adapter, 'abort', None)
                    if abort:
                        threading.Thread(target=abort, daemon=True).start()
                    raise UnknownSideEffect("Native call deadline expired; attempt stopped")
                if owner: owner.join()
                if not ok:
                    self.unknown = True
                    event['state'] = 'unknown'
                    if self.observer:
                        self.observer(event, started_at, time.time())
                    if self.journal:
                        self.journal(self.events)
                    raise UnknownSideEffect(str(value)) from value
                result = value
                self.environment_steps += int(result.get('environment_step', 0))
            self.done = bool(result.get('done', False))
            result_id = self.context.register(event['event_id'], result, name=name, arguments=arguments)
            event.update(state='finished', result=result, result_id=result_id,
                         environment_steps=self.environment_steps, environment_step=int(result.get('environment_step', 0)),
                         progress_after=progress_key(self.adapter))
            if event['backend_invoked'] and hasattr(self.adapter, 'public_update'):
                event['state_update'] = self.adapter.public_update
            if self.observer:
                self.observer(event, started_at, time.time())
            if self.journal:
                self.journal(self.events)
            return result

    def check_local(self, inputs, outputs, start):
        check = getattr(self.adapter, "check_local", None)
        return check(inputs, outputs, self.events[start:]) if check else "unavailable"
