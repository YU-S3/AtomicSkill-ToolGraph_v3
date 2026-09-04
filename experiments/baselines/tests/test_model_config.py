"""Frozen baseline model identity checks (design doc section 6)."""

from __future__ import annotations

import pytest

from experiments.baselines.common.model_config import ModelConfig


def test_frozen_identity_accepts_formal_config() -> None:
    ModelConfig(
        provider="openai_compatible",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
        api_key_env="MODEL_API_KEY",
    ).validate_formal_identity()


def test_frozen_identity_rejects_drift() -> None:
    with pytest.raises(ValueError, match="model identity mismatch"):
        ModelConfig(
            provider="openai_compatible",
            base_url="https://other.example.com",
            model="deepseek-v4-flash",
            api_key_env="MODEL_API_KEY",
        ).validate_formal_identity()


def test_missing_api_key_fail_closed(monkeypatch) -> None:
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    config = ModelConfig(
        provider="openai_compatible",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
        api_key_env="MODEL_API_KEY",
    )
    with pytest.raises(RuntimeError, match="not set"):
        config.require_api_key()


def test_wire_never_carries_keys(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_API_KEY", "sk-secret-value-0123456789abcdef")
    config = ModelConfig(
        provider="openai_compatible",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
        api_key_env="MODEL_API_KEY",
    )
    wire = config.to_wire()
    assert set(wire) == {"provider", "base_url", "model", "api_key_env"}
    assert "sk-secret-value-0123456789abcdef" not in str(wire)
