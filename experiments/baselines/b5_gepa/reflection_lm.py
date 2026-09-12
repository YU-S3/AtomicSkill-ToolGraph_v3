"""Audited GEPA ``LanguageModel`` transport over the shared provider client."""

from __future__ import annotations

import threading
from typing import Any


class DeepSeekReflectionLM:
    """Forward GEPA's unmodified default prompt through SkillOpt's audited LM.

    The B3 provider observer wraps this same backend, so reflection calls have
    the same retry, reasoning-effort, token-accounting, and campaign-gate
    authority as target calls.
    """

    def __init__(
        self,
        *,
        reasoning_effort: str,
        max_completion_tokens: int = 16384,
        retries: int = 5,
    ) -> None:
        if reasoning_effort != "high":
            raise ValueError("formal GEPA reflection requires reasoning_effort='high'")
        if max_completion_tokens <= 0 or retries <= 0:
            raise ValueError("reflection token and retry limits must be positive")
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = int(max_completion_tokens)
        self.retries = int(retries)
        self._lock = threading.Lock()
        self._calls = 0

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        from skillopt.model import chat_optimizer_messages

        if isinstance(prompt, str):
            if not prompt.strip():
                raise ValueError("GEPA reflection prompt is empty")
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and prompt:
            messages = []
            for message in prompt:
                if not isinstance(message, dict):
                    raise TypeError("GEPA reflection messages must be mappings")
                role = str(message.get("role", ""))
                content = message.get("content")
                if role not in {"system", "user", "assistant"}:
                    raise ValueError(f"invalid GEPA reflection message role: {role!r}")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("GEPA reflection message content is empty")
                messages.append({"role": role, "content": content})
        else:
            raise TypeError("GEPA reflection prompt must be text or chat messages")
        response, _ = chat_optimizer_messages(
            messages=messages,
            max_completion_tokens=self.max_completion_tokens,
            retries=self.retries,
            stage="gepa_reflection",
            reasoning_effort=self.reasoning_effort,
            return_message=False,
        )
        if not isinstance(response, str) or not response.strip():
            raise RuntimeError("GEPA reflection provider returned no text")
        with self._lock:
            self._calls += 1
        return response

    @property
    def calls(self) -> int:
        with self._lock:
            return self._calls
