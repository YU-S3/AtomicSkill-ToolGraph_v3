"""Model identity for every generative LLM role of every baseline.

Per the design document, section 6, the base model identity is frozen for all
generative roles; method-owned decoding parameters are recorded but not
removed.  The wire protocol (§28) transfers only the model identity and the
name of the environment variable that holds the API key — never a key value.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

FROZEN_BASELINE_MODEL = {
    "provider": "openai_compatible",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-v4-flash",
    "api_key_env": "MODEL_API_KEY",
}


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    base_url: str
    model: str
    api_key_env: str

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "ModelConfig":
        if not isinstance(payload, dict):
            raise ValueError("model config must be a mapping")
        try:
            config = cls(
                provider=str(payload["provider"]),
                base_url=str(payload["base_url"]).rstrip("/"),
                model=str(payload["model"]),
                api_key_env=str(payload["api_key_env"]),
            )
        except KeyError as exc:
            raise ValueError(f"model config is missing {exc.args[0]}") from exc
        return config

    def validate_formal_identity(self) -> None:
        """Fail closed unless the base model identity matches the frozen spec."""

        expected = dict(FROZEN_BASELINE_MODEL)
        mismatches = [
            f"{name}: expected {expected[name]!r}, got {getattr(self, name)!r}"
            for name in expected
            if getattr(self, name) != expected[name]
        ]
        if mismatches:
            raise ValueError("baseline model identity mismatch: " + "; ".join(mismatches))

    def require_api_key(self) -> str:
        """Return the API key from the environment; fail closed when absent."""

        value = os.environ.get(self.api_key_env, "").strip()
        if not value:
            raise RuntimeError(
                f"{self.api_key_env} is not set in the process environment; "
                "export it before starting the run (never write it to config files)"
            )
        return value

    def to_wire(self) -> dict[str, Any]:
        """Wire-safe model identity (no secrets)."""

        return asdict(self)
