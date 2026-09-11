"""Fail-closed observation for SkillOpt's OpenAI-compatible provider.

The upstream SkillOpt implementation owns request construction and token
tracking. This module wraps its OpenAI-compatible call boundary, makes retries
explicit and auditable, and persists one row per logical LLM call so provider
failures remain distinct from ALFWorld task failures.

No prompts, responses, or credentials are written to disk.
"""

from __future__ import annotations

import concurrent.futures
import contextvars
import json
import os
import threading
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class ObservedProviderFailure(RuntimeError):
    """A target-provider call failed during the active ALFWorld episode."""


class ProviderCallExhausted(BaseException):
    """Internal abort after provider retries are exhausted.

    This deliberately does not inherit from ``Exception``: pinned SkillOpt
    catches ordinary model exceptions in several optimizer helpers and would
    otherwise turn an infrastructure outage into a fallback candidate.  The
    worker catches this sentinel explicitly at its outer boundary.
    """


class _ProviderResponseError(RuntimeError):
    """Retryable malformed provider response with a non-secret audit code."""

    def __init__(self, failure_code: str) -> None:
        super().__init__(failure_code)
        self.failure_code = failure_code


_TRANSIENT_FAILURE_CODES = frozenset({
    "connection",
    "empty_choices",
    "empty_message",
    "invalid_usage",
    "rate_limit",
    "server_error",
    "timeout",
})


def _provider_failure_code(exc: BaseException) -> str:
    """Classify a provider exception without persisting its message or body."""

    explicit = str(getattr(exc, "failure_code", "") or "").strip()
    if explicit:
        return explicit

    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    try:
        status_code = int(status) if status is not None else None
    except (TypeError, ValueError):
        status_code = None
    if status_code in {408}:
        return "timeout"
    if status_code == 409:
        return "request_conflict"
    if status_code == 429:
        return "rate_limit"
    if status_code is not None and status_code >= 500:
        return "server_error"
    if status_code == 401:
        return "authentication"
    if status_code == 403:
        return "permission"
    if status_code == 404:
        return "not_found"
    if status_code is not None and 400 <= status_code < 500:
        return "bad_request"

    name = type(exc).__name__.casefold()
    message = str(exc).casefold()
    combined = f"{name} {message}"
    if "empty non-tool completion" in combined:
        return "empty_completion"
    if "no choices" in combined:
        return "empty_choices"
    if "no usable token" in combined or "token-usage evidence" in combined:
        return "invalid_usage"
    if "authentication" in combined or "unauthorized" in combined:
        return "authentication"
    if "permission" in combined or "forbidden" in combined:
        return "permission"
    if "rate" in combined and "limit" in combined:
        return "rate_limit"
    if "timeout" in combined or "timed out" in combined:
        return "timeout"
    if "connection" in combined or "connecterror" in combined:
        return "connection"
    if "internalserver" in combined or "server error" in combined:
        return "server_error"
    if isinstance(exc, (TypeError, ValueError)):
        return "invalid_request"
    return "unknown_provider_error"


def _is_transient_provider_failure(code: str) -> bool:
    return str(code) in _TRANSIENT_FAILURE_CODES


class ProviderCallObserver:
    """Observe the upstream backend without replacing its implementation.

    SkillOpt's ALFWorld rollout intentionally catches target-call exceptions
    and substitutes a ``look`` action.  The observer records such an exception;
    the environment proxy in :mod:`episode_runner` then raises before that
    fallback action can be executed or returned to the optimizer as a normal
    failed trajectory.

    Episode rollouts may execute concurrently.  A context variable binds each
    outer episode to its task and rollout identity; the temporary executor
    wrapper installed below propagates that context into SkillOpt's inner API
    worker without changing request construction or model behavior.
    """

    def __init__(
        self,
        *,
        output_path: str | Path,
        method: str,
        phase: str,
        model: str,
        reasoning_effort: str,
        run_id: str,
        application_retry_limit: int | None = None,
        retry_delays_seconds: tuple[float, ...] | list[float] | None = None,
        expected_sdk_max_retries: int | None = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.method = str(method)
        self.phase = str(phase)
        self.model = str(model)
        self.reasoning_effort = str(reasoning_effort)
        self.run_id = str(run_id)
        self.application_retry_limit = (
            int(application_retry_limit)
            if application_retry_limit is not None else None
        )
        if self.application_retry_limit is not None and not (
            1 <= self.application_retry_limit <= 32
        ):
            raise ValueError("application_retry_limit must be within 1..32")
        self.retry_delays_seconds = tuple(
            float(delay) for delay in (retry_delays_seconds or ())
        )
        if any(delay < 0 for delay in self.retry_delays_seconds):
            raise ValueError("retry delays must be non-negative")
        if (
            self.application_retry_limit is not None
            and self.retry_delays_seconds
            and len(self.retry_delays_seconds) != self.application_retry_limit - 1
        ):
            raise ValueError(
                "retry_delays_seconds must contain one delay per retry"
            )
        self.expected_sdk_max_retries = (
            int(expected_sdk_max_retries)
            if expected_sdk_max_retries is not None else None
        )
        if self.expected_sdk_max_retries is not None and (
            self.expected_sdk_max_retries < 0
        ):
            raise ValueError("expected_sdk_max_retries must be non-negative")
        self._lock = threading.RLock()
        self._episode_context: contextvars.ContextVar[tuple[str, str] | None] = (
            contextvars.ContextVar(
                f"asg_skillopt_episode_{id(self)}", default=None,
            )
        )
        self._active_episode_keys: set[tuple[str, str]] = set()
        self._episode_failures: dict[tuple[str, str], dict[str, Any]] = {}
        self._attempt_diagnostics: contextvars.ContextVar[dict[str, Any] | None] = (
            contextvars.ContextVar(
                f"asg_skillopt_provider_attempt_{id(self)}", default=None,
            )
        )
        self._events: list[dict[str, Any]] = []
        self._backend: Any | None = None
        self._original_call: Any | None = None
        self._original_usage_parser: Any | None = None
        self._original_executor_submit: Any | None = None
        self._original_get_client: Any | None = None
        self._transport_clients: dict[str, Any] = {}

    def _note_provider_boundary_failure(self, exc: BaseException) -> str:
        code = _provider_failure_code(exc)
        diagnostics = self._attempt_diagnostics.get()
        if diagnostics is not None:
            diagnostics.setdefault("failure_codes", []).append(code)
            request_id = str(getattr(exc, "request_id", "") or "").strip()
            if request_id:
                diagnostics.setdefault("provider_request_ids", []).append(request_id)
        return code

    def _retry_delay(self, call_id: str, failed_attempt: int) -> float:
        del call_id  # The frozen formal schedule is deterministic; no jitter.
        index = int(failed_attempt) - 1
        if index < 0 or index >= len(self.retry_delays_seconds):
            return 0.0
        return self.retry_delays_seconds[index]

    def install(self) -> None:
        """Install one process-local wrapper around SkillOpt's backend."""

        import skillopt.model.openai_compatible_backend as backend

        if self._backend is not None:
            raise RuntimeError("provider observer is already installed")
        if getattr(backend, "_asg_provider_observer", None) is not None:
            raise RuntimeError("another provider observer is already installed")
        if self.output_path.exists():
            raise FileExistsError(
                f"refusing to append provider evidence to an existing file: "
                f"{self.output_path}"
            )

        original_call = backend._chat_messages_impl
        original_usage_parser = backend.usage_from_openai_usage
        original_get_client = backend._get_client
        original_executor_submit = concurrent.futures.ThreadPoolExecutor.submit
        observer = self

        class _CompletionsProxy:
            def __init__(self, wrapped: Any) -> None:
                self._wrapped = wrapped

            def create(self, **kwargs: Any) -> Any:
                diagnostics = observer._attempt_diagnostics.get()
                if diagnostics is not None:
                    diagnostics["sdk_boundary_attempts"] = int(
                        diagnostics.get("sdk_boundary_attempts", 0)
                    ) + 1
                try:
                    declared = kwargs.get("reasoning_effort")
                    if declared not in {None, observer.reasoning_effort}:
                        error = _ProviderResponseError("configuration")
                        observer._note_provider_boundary_failure(error)
                        raise error
                    kwargs["reasoning_effort"] = observer.reasoning_effort
                    response = self._wrapped.create(**kwargs)
                except Exception as exc:
                    failure_codes = (
                        diagnostics.get("failure_codes", [])
                        if diagnostics is not None else []
                    )
                    if not failure_codes or failure_codes[-1] != "configuration":
                        observer._note_provider_boundary_failure(exc)
                    raise
                request_id = str(getattr(response, "_request_id", "") or "").strip()
                if diagnostics is not None and request_id:
                    diagnostics.setdefault("provider_request_ids", []).append(request_id)
                choices = getattr(response, "choices", None) or []
                if not choices:
                    error = _ProviderResponseError("empty_choices")
                    observer._note_provider_boundary_failure(error)
                    raise error
                message = getattr(choices[0], "message", None)
                if message is None:
                    error = _ProviderResponseError("empty_message")
                    observer._note_provider_boundary_failure(error)
                    raise error
                tool_calls = getattr(message, "tool_calls", None) or []
                content = getattr(message, "content", None)
                usage = getattr(response, "usage", None)
                completion_tokens = int(
                    getattr(usage, "completion_tokens", 0) or 0
                )
                if completion_tokens <= 0:
                    error = _ProviderResponseError("invalid_usage")
                    observer._note_provider_boundary_failure(error)
                    raise error
                return response

            def __getattr__(self, name: str) -> Any:
                return getattr(self._wrapped, name)

        class _ChatProxy:
            def __init__(self, wrapped: Any) -> None:
                self._wrapped = wrapped
                self.completions = _CompletionsProxy(wrapped.completions)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._wrapped, name)

        class _ClientProxy:
            def __init__(self, wrapped: Any) -> None:
                self._wrapped = wrapped
                self.chat = _ChatProxy(wrapped.chat)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._wrapped, name)

        def get_client_with_reasoning(role: str) -> Any:
            with observer._lock:
                client = observer._transport_clients.get(role)
                if client is None:
                    client = original_get_client(role)
                    expected = observer.expected_sdk_max_retries
                    if expected is not None:
                        # openai 2.x does not read an OPENAI_MAX_RETRIES
                        # environment variable.  Clone the configured upstream
                        # client with the formal transport policy instead of
                        # modifying the pinned SkillOpt source tree.
                        actual = getattr(client, "max_retries", None)
                        if actual is None or int(actual) != expected:
                            with_options = getattr(client, "with_options", None)
                            if not callable(with_options):
                                raise _ProviderResponseError("configuration")
                            client = with_options(max_retries=expected)
                        actual = getattr(client, "max_retries", None)
                        if actual is None or int(actual) != expected:
                            raise _ProviderResponseError("configuration")
                    observer._transport_clients[role] = client
            return _ClientProxy(client)

        def submit_with_context(
            executor: concurrent.futures.ThreadPoolExecutor,
            function: Any,
            /,
            *args: Any,
            **kwargs: Any,
        ):
            inherited = contextvars.copy_context()
            return original_executor_submit(
                executor, inherited.run, function, *args, **kwargs,
            )

        def usage_with_reasoning(raw_usage: Any) -> dict[str, Any]:
            parsed = dict(original_usage_parser(raw_usage))
            details = getattr(raw_usage, "completion_tokens_details", None)
            reasoning = getattr(details, "reasoning_tokens", None) if details else None
            parsed["reasoning_tokens"] = (
                int(reasoning) if reasoning is not None else None
            )
            parsed["reasoning_tokens_status"] = (
                "reported" if reasoning is not None else "unavailable"
            )
            return parsed

        def observed_call(
            messages: list[dict[str, Any]],
            max_completion_tokens: int,
            retries: int,
            stage: str,
            *,
            role: str,
            tools: list[dict[str, Any]] | None = None,
            tool_choice: str | dict[str, Any] | None = None,
            return_message: bool = False,
            deployment: str | None = None,
            timeout: float | None = None,
        ) -> tuple[Any, dict[str, Any]]:
            started = time.perf_counter()
            call_id = f"provider_{uuid.uuid4().hex}"
            requested_retry_limit = int(retries)
            retry_limit = int(requested_retry_limit)
            if observer.application_retry_limit is not None:
                # The formal policy is a ceiling.  It must not silently expand
                # an upstream call that intentionally requested fewer tries.
                retry_limit = min(retry_limit, observer.application_retry_limit)
            if requested_retry_limit <= 0 or retry_limit <= 0:
                raise ValueError("provider retry limits must be positive")
            diagnostics: dict[str, Any] = {
                "sdk_boundary_attempts": 0,
                "failure_codes": [],
                "provider_request_ids": [],
            }
            diagnostic_token = observer._attempt_diagnostics.set(diagnostics)
            result: Any = None
            normalized: dict[str, Any] = {}
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            attempts_used = 0
            backoff_ms = 0
            last_exception: Exception | None = None
            try:
                for attempt in range(1, retry_limit + 1):
                    attempts_used = attempt
                    failures_before = len(diagnostics["failure_codes"])
                    try:
                        # One upstream attempt plus zero SDK-internal retries
                        # gives this adapter an exact, auditable retry boundary.
                        result, usage = original_call(
                            messages,
                            max_completion_tokens,
                            1,
                            stage,
                            role=role,
                            tools=tools,
                            tool_choice=tool_choice,
                            return_message=return_message,
                            deployment=deployment,
                            timeout=timeout,
                        )
                        normalized = dict(usage or {})
                        prompt_tokens = int(
                            normalized.get("prompt_tokens", 0) or 0
                        )
                        completion_tokens = int(
                            normalized.get("completion_tokens", 0) or 0
                        )
                        total_tokens = int(
                            normalized.get(
                                "total_tokens", prompt_tokens + completion_tokens,
                            ) or 0
                        )
                        if completion_tokens <= 0 or total_tokens <= 0:
                            raise _ProviderResponseError("invalid_usage")
                        break
                    except Exception as exc:
                        last_exception = exc
                        if len(diagnostics["failure_codes"]) == failures_before:
                            observer._note_provider_boundary_failure(exc)
                        failure_code = str(diagnostics["failure_codes"][-1])
                        if (
                            attempt >= retry_limit
                            or not _is_transient_provider_failure(failure_code)
                        ):
                            break
                        delay = observer._retry_delay(call_id, attempt)
                        if delay > 0:
                            # The pinned upstream function sleeps one second
                            # even when invoked with retries=1. Subtract that
                            # known delay so the actual inter-attempt schedule
                            # remains the frozen 2/5/10/20 seconds.
                            time.sleep(max(0.0, delay - 1.0))
                            backoff_ms += int(delay * 1000)

                failure_counts = dict(sorted(Counter(
                    str(code) for code in diagnostics["failure_codes"]
                ).items()))
                recovered = bool(failure_counts) and last_exception is not None and (
                    completion_tokens > 0 and total_tokens > 0
                )
                if completion_tokens > 0 and total_tokens > 0:
                    try:
                        observer._record(
                            call_id=call_id,
                            role=role,
                            stage=stage,
                            status="succeeded",
                            latency_ms=int((time.perf_counter() - started) * 1000),
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=total_tokens,
                            reasoning_tokens=normalized.get("reasoning_tokens"),
                            reasoning_tokens_status=str(
                                normalized.get(
                                    "reasoning_tokens_status", "unavailable",
                                )
                            ),
                            requested_retry_limit=requested_retry_limit,
                            retry_limit=retry_limit,
                            application_attempts=attempts_used,
                            sdk_boundary_attempts=int(
                                diagnostics["sdk_boundary_attempts"]
                            ),
                            retry_backoff_ms=backoff_ms,
                            failure_code_counts=failure_counts,
                            provider_request_ids=list(dict.fromkeys(
                                str(value)
                                for value in diagnostics["provider_request_ids"]
                            )),
                            last_failure_code=(
                                str(diagnostics["failure_codes"][-1])
                                if diagnostics["failure_codes"] else None
                            ),
                            recovered=recovered,
                        )
                    except Exception as audit_exc:
                        abort = ProviderCallExhausted(
                            "provider audit persistence failed "
                            f"(failure_code=audit_io, attempts={attempts_used})"
                        )
                        if str(role) == "target":
                            observer._mark_active_episode_failure(
                                abort,
                                failure_code="audit_io",
                                application_attempts=attempts_used,
                            )
                        raise abort from audit_exc
                    return result, normalized

                failure_code = (
                    str(diagnostics["failure_codes"][-1])
                    if diagnostics["failure_codes"] else "unknown_provider_error"
                )
                exhausted = ProviderCallExhausted(
                    "provider call failed under the formal retry policy "
                    f"(failure_code={failure_code}, attempts={attempts_used})"
                )
                if str(role) == "target":
                    observer._mark_active_episode_failure(
                        exhausted,
                        failure_code=failure_code,
                        application_attempts=attempts_used,
                    )
                # Mark the episode before persisting the terminal failure. If
                # audit I/O fails, the environment proxy still prevents the
                # upstream swallowed ``look`` fallback from becoming an action.
                try:
                    observer._record(
                        call_id=call_id,
                        role=role,
                        stage=stage,
                        status="failed",
                        latency_ms=int((time.perf_counter() - started) * 1000),
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        reasoning_tokens=None,
                        reasoning_tokens_status="unavailable",
                        requested_retry_limit=requested_retry_limit,
                        retry_limit=retry_limit,
                        application_attempts=attempts_used,
                        sdk_boundary_attempts=int(
                            diagnostics["sdk_boundary_attempts"]
                        ),
                        retry_backoff_ms=backoff_ms,
                        failure_code_counts=failure_counts,
                        provider_request_ids=list(dict.fromkeys(
                            str(value)
                            for value in diagnostics["provider_request_ids"]
                        )),
                        last_failure_code=failure_code,
                        recovered=False,
                        error_type=type(exhausted).__name__,
                    )
                except Exception as audit_exc:
                    raise ProviderCallExhausted(
                        "provider audit persistence failed "
                        f"(failure_code=audit_io, attempts={attempts_used})"
                    ) from audit_exc
                raise exhausted from last_exception
            finally:
                observer._attempt_diagnostics.reset(diagnostic_token)

        backend.usage_from_openai_usage = usage_with_reasoning
        backend._get_client = get_client_with_reasoning
        backend._chat_messages_impl = observed_call
        backend._asg_provider_observer = self
        concurrent.futures.ThreadPoolExecutor.submit = submit_with_context
        self._backend = backend
        self._original_call = original_call
        self._original_usage_parser = original_usage_parser
        self._original_get_client = original_get_client
        self._original_executor_submit = original_executor_submit

    def uninstall(self) -> None:
        """Restore the exact upstream functions."""

        with self._lock:
            backend = self._backend
            if backend is None:
                return
            backend._chat_messages_impl = self._original_call
            backend.usage_from_openai_usage = self._original_usage_parser
            backend._get_client = self._original_get_client
            backend._asg_provider_observer = None
            if self._original_executor_submit is not None:
                concurrent.futures.ThreadPoolExecutor.submit = (
                    self._original_executor_submit
                )
            self._backend = None
            self._original_call = None
            self._original_usage_parser = None
            self._original_get_client = None
            self._original_executor_submit = None
            self._transport_clients.clear()

    @contextmanager
    def episode(self, task_id: str, *, rollout_id: str = "") -> Iterator[None]:
        """Associate target calls with exactly one sequential episode."""

        normalized_task_id = str(task_id).strip()
        normalized_rollout_id = str(rollout_id).strip()
        if not normalized_task_id:
            raise ValueError("provider observer episode task_id must be non-empty")
        if not normalized_rollout_id:
            raise ValueError("provider observer rollout_id must be non-empty")
        key = (normalized_rollout_id, normalized_task_id)
        with self._lock:
            if key in self._active_episode_keys:
                raise RuntimeError(
                    "provider observer does not allow duplicate overlapping episodes"
                )
            self._active_episode_keys.add(key)
            self._episode_failures.pop(key, None)
        token = self._episode_context.set(key)
        try:
            yield
        finally:
            self._episode_context.reset(token)
            with self._lock:
                self._active_episode_keys.discard(key)
                self._episode_failures.pop(key, None)

    def raise_if_active_episode_failed(self) -> None:
        """Raise before SkillOpt can execute a swallowed provider fallback."""

        key = self._episode_context.get()
        with self._lock:
            failure = dict(self._episode_failures.get(key, {}) if key else {})
        if failure:
            raise ObservedProviderFailure(
                "target provider failed during episode "
                f"{failure.get('task_id', '<unknown>')} "
                f"({failure.get('error_type', 'ProviderError')}; "
                f"failure_code={failure.get('failure_code', 'unknown')}; "
                f"attempts={failure.get('application_attempts', 0)})"
            )

    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return self._validated_persisted_events_locked()

    def event_cursor(self) -> int:
        """Return a stable in-process cursor for one rollout receipt."""

        with self._lock:
            return len(self._events)

    def events_since(
        self,
        cursor: int,
        *,
        rollout_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return events recorded after ``cursor``, optionally for one rollout."""

        with self._lock:
            if cursor < 0 or cursor > len(self._events):
                raise ValueError("provider event cursor is outside the recorded event range")
            events = self._validated_persisted_events_locked()[cursor:]
        if rollout_id is None:
            return events
        return [
            event for event in events
            if event.get("rollout_id") == str(rollout_id)
        ]

    def _mark_active_episode_failure(
        self,
        exc: BaseException,
        *,
        failure_code: str | None = None,
        application_attempts: int = 0,
    ) -> None:
        key = self._episode_context.get()
        with self._lock:
            if key is None or key not in self._active_episode_keys:
                return
            self._episode_failures[key] = {
                "task_id": key[1],
                "error_type": type(exc).__name__,
                "failure_code": str(
                    failure_code or _provider_failure_code(exc)
                ),
                "application_attempts": int(application_attempts),
            }

    def _record(self, **payload: Any) -> None:
        key = self._episode_context.get()
        with self._lock:
            rollout_id, task_id = key if key is not None else ("", "")
            event = {
                "schema_version": 1,
                "event": "provider_call",
                "method": self.method,
                "phase": self.phase,
                "run_id": self.run_id,
                "rollout_id": rollout_id,
                "episode_task_id": task_id,
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
                **payload,
            }
            self._events.append(event)
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def _validated_persisted_events_locked(self) -> list[dict[str, Any]]:
        """Make the durable provider sidecar authoritative for later readers."""

        if not self._events:
            if self.output_path.exists() and self.output_path.read_bytes():
                raise RuntimeError(
                    "provider evidence file contains rows absent from this observer"
                )
            return []
        if not self.output_path.is_file():
            raise RuntimeError("provider evidence file is missing after recorded calls")
        persisted: list[dict[str, Any]] = []
        try:
            lines = self.output_path.read_text(encoding="utf-8").splitlines()
            for line_number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"provider evidence row {line_number} is not a mapping"
                    )
                persisted.append(dict(payload))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(
                f"provider evidence file is unreadable: {self.output_path}"
            ) from exc
        if persisted != self._events:
            raise RuntimeError(
                "provider evidence file differs from in-memory call evidence"
            )
        return [dict(item) for item in persisted]


_ACTIVE_OBSERVER: ProviderCallObserver | None = None
_ACTIVE_LOCK = threading.RLock()


def install_provider_observer(
    *,
    output_path: str | Path,
    method: str,
    phase: str,
    model: str,
    reasoning_effort: str,
    run_id: str,
    application_retry_limit: int | None = None,
    retry_delays_seconds: tuple[float, ...] | list[float] | None = None,
    expected_sdk_max_retries: int | None = None,
) -> ProviderCallObserver:
    global _ACTIVE_OBSERVER
    with _ACTIVE_LOCK:
        if _ACTIVE_OBSERVER is not None:
            raise RuntimeError("a provider observer is already active")
        observer = ProviderCallObserver(
            output_path=output_path,
            method=method,
            phase=phase,
            model=model,
            reasoning_effort=reasoning_effort,
            run_id=run_id,
            application_retry_limit=application_retry_limit,
            retry_delays_seconds=retry_delays_seconds,
            expected_sdk_max_retries=expected_sdk_max_retries,
        )
        observer.install()
        _ACTIVE_OBSERVER = observer
        return observer


def uninstall_provider_observer(observer: ProviderCallObserver) -> None:
    global _ACTIVE_OBSERVER
    with _ACTIVE_LOCK:
        if _ACTIVE_OBSERVER is not observer:
            raise RuntimeError("attempted to uninstall a non-active provider observer")
        observer.uninstall()
        _ACTIVE_OBSERVER = None


def active_provider_observer() -> ProviderCallObserver | None:
    with _ACTIVE_LOCK:
        return _ACTIVE_OBSERVER
