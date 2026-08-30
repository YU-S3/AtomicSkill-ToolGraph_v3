"""Harness-injectable value-sensitive TaskContract matching."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from ..core.bindings import BindingExpression, BindingExprKind
from ..core.contracts import SemanticPredicate


@runtime_checkable
class ContractMatcher(Protocol):
    def effect_covers_target(
        self,
        *,
        offered_predicate: SemanticPredicate,
        offered_arguments: Mapping[str, Any],
        target_predicate: SemanticPredicate,
    ) -> bool: ...


class ExactContractMatcher:
    """Exact matcher with optional values for TaskContract binding expressions."""

    def __init__(self, bindings: Mapping[str, Any] | None = None) -> None:
        self.bindings = dict(bindings or {})

    def _resolve(self, raw: Any) -> Any:
        if isinstance(raw, dict) and "kind" in raw:
            raw = BindingExpression.from_dict(raw)
        if isinstance(raw, BindingExpression):
            if raw.kind is BindingExprKind.CONSTANT:
                return raw.constant
            return self.bindings.get(raw.source_role)
        if isinstance(raw, str) and raw.startswith("$"):
            return self.bindings.get(raw[1:])
        return raw

    def effect_covers_target(
        self,
        *,
        offered_predicate: SemanticPredicate,
        offered_arguments: Mapping[str, Any],
        target_predicate: SemanticPredicate,
    ) -> bool:
        expected = {
            name: self._resolve(value)
            for name, value in target_predicate.args.items()
        }
        return (
            target_predicate.predicate.casefold()
            == offered_predicate.predicate.casefold()
            and set(expected) == set(offered_arguments)
            and all(
                expected[name] == offered_arguments.get(name)
                for name in expected
            )
        )

    def matches(
        self,
        target: SemanticPredicate,
        offered: SemanticPredicate,
        offered_args: Mapping[str, Any],
    ) -> bool:
        """Compatibility alias for callers outside the v3 extraction path."""

        return self.effect_covers_target(
            offered_predicate=offered,
            offered_arguments=offered_args,
            target_predicate=target,
        )


__all__ = ["ContractMatcher", "ExactContractMatcher"]
