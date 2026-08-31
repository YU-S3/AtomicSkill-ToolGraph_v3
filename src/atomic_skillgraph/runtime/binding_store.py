"""The task-local unique source of all values and their provenance."""

from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from ..core.bindings import (
    BindingExpression, BindingExprKind, BindingResolution, BindingSource,
    BindingStatus, GroundingConstraint, RuntimeBinding, RuntimeBindingChange,
    resolution_satisfies,
)
from ..core.contracts import ParameterSpec, TaskContract
from ..core.results import (
    RuntimeLinearPlan,
    RuntimeOccurrence,
    RuntimeRepeatConstraint,
    ValidationResult,
)


def _looks_concrete(value: Any) -> bool:
    return bool(isinstance(value, str) and re.search(r"(?:_|\s)\d+$", value.strip()))


@dataclass
class RepeatBindingState:
    """Effect-committed repetition values for one task only.

    Keys are ``<block_id>::<block_role>`` so independent RepeatBlocks that
    use a common semantic role cannot contaminate one another.
    """

    committed_distinct_values: dict[str, dict[int, Any]] = field(
        default_factory=dict,
    )
    committed_shared_values: dict[str, Any] = field(default_factory=dict)


class RuntimeBindingStore:
    def __init__(self, *, on_change: Callable[[RuntimeBindingChange], None] | None = None) -> None:
        self._bindings: dict[tuple[str, str], RuntimeBinding] = {}
        self._outputs: dict[tuple[str, str], RuntimeBinding] = {}
        self._proposals: dict[tuple[str, str], RuntimeBinding] = {}
        self._transforms: dict[str, Callable[[Any], Any]] = {}
        # RuntimeLinearPlan edges and BindingExpression values address their
        # producers by step_id, while validated outputs are owned by the
        # occurrence_id that actually executed.  Keep the task-local mapping
        # explicit instead of relying on the two identifiers being equal.
        self._step_to_occurrence: dict[str, str] = {}
        self._on_change = on_change
        self.repeat_state = RepeatBindingState()
        self._repeat_constraints: dict[str, RuntimeRepeatConstraint] = {}
        self._repeat_step_owner: dict[
            str, tuple[RuntimeRepeatConstraint, int]
        ] = {}

    @staticmethod
    def _repeat_key(block_id: str, block_role: str) -> str:
        return f"{block_id}::{block_role}"

    def configure_repeat_constraints(
        self,
        constraints: list[RuntimeRepeatConstraint]
        | tuple[RuntimeRepeatConstraint, ...],
    ) -> None:
        """Install one validated plan's repeat constraints and reset state."""

        by_block: dict[str, RuntimeRepeatConstraint] = {}
        by_step: dict[str, tuple[RuntimeRepeatConstraint, int]] = {}
        for constraint in constraints:
            if (
                not constraint.block_id
                or constraint.block_id in by_block
                or constraint.count < 2
                or len(constraint.iteration_steps) != constraint.count
                or set(constraint.distinct_roles)
                & set(constraint.shared_roles)
            ):
                raise ValueError("invalid RuntimeRepeatConstraint")
            declared_steps = {
                step_id
                for iteration in constraint.iteration_steps
                for step_id in iteration
            }
            if set(constraint.step_role_bindings) != declared_steps:
                raise ValueError(
                    "repeat step_role_bindings must cover exactly declared steps"
                )
            allowed_roles = {
                *constraint.distinct_roles,
                *constraint.shared_roles,
            }
            if any(
                not set(bindings).issubset(allowed_roles)
                or any(not str(value) for value in bindings.values())
                for bindings in constraint.step_role_bindings.values()
            ):
                raise ValueError("invalid repeat step role mapping")
            for repeat_index, iteration in enumerate(
                constraint.iteration_steps
            ):
                if not iteration:
                    raise ValueError("repeat iteration cannot be empty")
                for step_id in iteration:
                    if not step_id or step_id in by_step:
                        raise ValueError(
                            "a runtime step may belong to only one repeat iteration"
                        )
                    by_step[step_id] = (constraint, repeat_index)
            by_block[constraint.block_id] = constraint
        self._repeat_constraints = by_block
        self._repeat_step_owner = by_step
        self.repeat_state = RepeatBindingState()

    def _repeat_values(
        self,
        step_id: str,
        values: dict[str, Any],
        *,
        require_complete: bool,
    ) -> tuple[
        RuntimeRepeatConstraint | None,
        int,
        dict[str, Any],
        list[str],
    ]:
        owner = self._repeat_step_owner.get(step_id)
        if owner is None:
            return None, -1, {}, []
        constraint, repeat_index = owner
        mappings = constraint.step_role_bindings.get(step_id, {})
        missing = [
            atomic_role
            for atomic_role in mappings.values()
            if atomic_role not in values
        ] if require_complete else []
        projected = {
            block_role: values[atomic_role]
            for block_role, atomic_role in mappings.items()
            if atomic_role in values
        }
        return constraint, repeat_index, projected, missing

    def preflight_repeat_bindings(
        self,
        step_id: str,
        values: dict[str, Any],
    ) -> ValidationResult:
        """Check proposed concrete values against earlier effect commits."""

        constraint, repeat_index, projected, _ = self._repeat_values(
            step_id,
            values,
            require_complete=False,
        )
        if constraint is None:
            return ValidationResult.ok(
                "runtime_repeat", repeat_step_or_unconstrained=True,
            )
        for block_role in constraint.distinct_roles:
            if block_role not in projected:
                continue
            key = self._repeat_key(constraint.block_id, block_role)
            committed = self.repeat_state.committed_distinct_values.get(
                key, {},
            )
            value = projected[block_role]
            if repeat_index in committed and committed[repeat_index] != value:
                return ValidationResult.fail(
                    "runtime_repeat",
                    "runtime_repetition_distinctness_violation",
                    "one repeat iteration proposed conflicting values for "
                    f"{block_role}",
                    repeat_distinct_values_valid=False,
                )
            if any(
                index != repeat_index and previous == value
                for index, previous in committed.items()
            ):
                return ValidationResult.fail(
                    "runtime_repeat",
                    "runtime_repetition_distinctness_violation",
                    "a distinct repeat role reused an earlier iteration value",
                    repeat_distinct_values_valid=False,
                )
        for block_role in constraint.shared_roles:
            if block_role not in projected:
                continue
            key = self._repeat_key(constraint.block_id, block_role)
            if (
                key in self.repeat_state.committed_shared_values
                and self.repeat_state.committed_shared_values[key]
                != projected[block_role]
            ):
                return ValidationResult.fail(
                    "runtime_repeat",
                    "runtime_repetition_shared_value_violation",
                    "a shared repeat role changed across iterations",
                    repeat_shared_values_valid=False,
                )
        return ValidationResult.ok(
            "runtime_repeat",
            repeat_distinct_values_valid=True,
            repeat_shared_values_valid=True,
        )

    def commit_repeat_bindings(
        self,
        step_id: str,
        values: dict[str, Any],
        *,
        effect_passed: bool,
    ) -> ValidationResult:
        """Commit values atomically, and only after an Atomic Effect passes."""

        if not effect_passed:
            return ValidationResult.ok(
                "runtime_repeat",
                failed_effect_left_repeat_state_unchanged=True,
            )
        constraint, repeat_index, projected, missing = self._repeat_values(
            step_id,
            values,
            require_complete=True,
        )
        if constraint is None:
            return ValidationResult.ok(
                "runtime_repeat", repeat_step_or_unconstrained=True,
            )
        if missing:
            return ValidationResult.fail(
                "runtime_repeat",
                "runtime_binding_unresolved",
                "effect-success repeat commit lacks mapped Atomic roles: "
                + ", ".join(sorted(missing)),
                repeat_commit_values_complete=False,
            )
        preflight = self.preflight_repeat_bindings(step_id, values)
        if not preflight.passed:
            return preflight

        distinct_updates: list[tuple[str, int, Any]] = []
        shared_updates: list[tuple[str, Any]] = []
        for block_role in constraint.distinct_roles:
            if block_role in projected:
                distinct_updates.append((
                    self._repeat_key(constraint.block_id, block_role),
                    repeat_index,
                    copy.deepcopy(projected[block_role]),
                ))
        for block_role in constraint.shared_roles:
            if block_role in projected:
                shared_updates.append((
                    self._repeat_key(constraint.block_id, block_role),
                    copy.deepcopy(projected[block_role]),
                ))
        # Apply only after every check passes, preserving transactionality.
        for key, index, value in distinct_updates:
            self.repeat_state.committed_distinct_values.setdefault(
                key, {},
            )[index] = value
        for key, value in shared_updates:
            self.repeat_state.committed_shared_values.setdefault(key, value)
        return ValidationResult.ok(
            "runtime_repeat",
            repeat_effect_values_committed=True,
        )

    def register_transform(self, transform_id: str, function: Callable[[Any], Any]) -> None:
        if not transform_id or transform_id in self._transforms:
            raise ValueError(f"invalid or duplicate transform: {transform_id!r}")
        self._transforms[transform_id] = function

    def _set(self, occurrence_id: str, binding: RuntimeBinding, reason: str) -> None:
        key = (occurrence_id, binding.role)
        previous = self._bindings.get(key)
        self._bindings[key] = binding
        if self._on_change:
            self._on_change(RuntimeBindingChange(
                occurrence_id, binding.role, asdict(previous) if previous else None,
                asdict(binding), reason, binding.world_revision,
            ))

    def seed_task_bindings(self, task: Any, contract: TaskContract, revision: int) -> None:
        context = getattr(task, "context", {}) or {}
        values = dict(context.get("semantic_bindings") or context.get("semantic_params") or context.get("bindings") or {})
        for role, value in values.items():
            self._set("__task__", RuntimeBinding(
                role=str(role), value=value, semantic_type=str(context.get("binding_types", {}).get(role, "entity")),
                source=BindingSource.TASK, status=BindingStatus.GROUNDED,
                resolution=BindingResolution.SEMANTIC, evidence_refs=[f"task:{getattr(task, 'task_id', '')}"],
                world_revision=revision,
            ), "seed_task")

    def bind_task_value(self, role: str, value: Any, semantic_type: str, revision: int) -> None:
        self._set("__task__", RuntimeBinding(
            role, value, semantic_type, BindingSource.TASK, BindingStatus.GROUNDED,
            BindingResolution.SEMANTIC, [f"task_role:{role}"], revision,
        ), "seed_task")

    def apply_data_flow(
        self, plan: RuntimeLinearPlan, current_step: str,
        validated_outputs: dict[str, dict[str, Any]] | None = None,
        *, revision: int = 0,
    ) -> None:
        self._step_to_occurrence = {
            occurrence.step_id: occurrence.occurrence_id
            for occurrence in plan.occurrences
        }
        current = plan.occurrence(current_step)
        for edge in plan.data_edges:
            if edge.target_step != current_step:
                continue
            source_occurrence = plan.occurrence(edge.source_step)
            source = self._outputs.get((source_occurrence.occurrence_id, edge.source_role))
            if source is None and validated_outputs:
                raw = validated_outputs.get(source_occurrence.occurrence_id, {}).get(edge.source_role)
                if raw is not None:
                    source = RuntimeBinding(
                        edge.source_role, raw, "entity", BindingSource.TOOL_OUTPUT,
                        BindingStatus.GROUNDED,
                        BindingResolution.CONCRETE if _looks_concrete(raw) else BindingResolution.SEMANTIC,
                        [edge.edge_id], revision,
                    )
            if source is None:
                continue
            self._set(current.occurrence_id, RuntimeBinding(
                edge.target_role, source.value, source.semantic_type, BindingSource.DATA_FLOW,
                BindingStatus.GROUNDED, source.resolution, list(source.evidence_refs) + [edge.edge_id], revision,
            ), "data_flow")

    def resolve_expression(
        self, occurrence_id: str, expression: BindingExpression,
        *, tool_outputs: dict[tuple[str, str], Any] | None = None,
    ) -> RuntimeBinding | None:
        expression = BindingExpression.from_dict(expression)
        if expression.kind is BindingExprKind.CONSTANT:
            return RuntimeBinding(
                role="", value=expression.constant, semantic_type=type(expression.constant).__name__,
                source=BindingSource.TASK, status=BindingStatus.GROUNDED,
                resolution=BindingResolution.CONCRETE, evidence_refs=["constant"], world_revision=0,
            )
        if expression.kind is BindingExprKind.SKILL_INPUT:
            binding = self._bindings.get((occurrence_id, expression.source_role)) or self._bindings.get(("__task__", expression.source_role))
        elif expression.kind is BindingExprKind.DATA_FLOW:
            source_owner = self._step_to_occurrence.get(
                expression.source_step, expression.source_step
            )
            binding = self._outputs.get((source_owner, expression.source_role))
            if binding is None:
                binding = self._bindings.get((source_owner, expression.source_role))
        elif expression.kind is BindingExprKind.TOOL_OUTPUT:
            value = (tool_outputs or {}).get((expression.source_step, expression.source_role))
            binding = None if value is None else RuntimeBinding(
                expression.source_role, value, "entity", BindingSource.TOOL_OUTPUT,
                BindingStatus.GROUNDED, BindingResolution.CONCRETE if _looks_concrete(value) else BindingResolution.SEMANTIC,
                [f"tool_output:{expression.source_step}:{expression.source_role}"], 0,
            )
        elif expression.kind is BindingExprKind.ADAPTER_TRANSFORM:
            source = self._bindings.get((occurrence_id, expression.source_role)) or self._bindings.get(("__task__", expression.source_role))
            if source is None or expression.transform_id not in self._transforms:
                return None
            binding = RuntimeBinding(
                expression.source_role, self._transforms[expression.transform_id](source.value), source.semantic_type,
                source.source, source.status, source.resolution,
                list(source.evidence_refs) + [f"transform:{expression.transform_id}"], source.world_revision,
            )
        else:
            return None
        return binding

    def resolve_occurrence_specs(self, occurrence: RuntimeOccurrence, revision: int) -> None:
        for role, raw_expression in occurrence.binding_specs.items():
            expression = BindingExpression.from_dict(raw_expression)
            binding = self.resolve_expression(occurrence.occurrence_id, expression)
            if binding is None:
                continue
            source = (
                BindingSource.DATA_FLOW
                if expression.kind is BindingExprKind.DATA_FLOW
                else binding.source
            )
            self._set(occurrence.occurrence_id, RuntimeBinding(
                role, binding.value, binding.semantic_type, source,
                binding.status, binding.resolution, list(binding.evidence_refs), revision,
            ), "binding_expression")

    def propose_agent_arguments(
        self, occurrence: RuntimeOccurrence | str, arguments: dict[str, Any], revision: int,
        semantic_types: dict[str, str] | None = None,
    ) -> dict[str, RuntimeBinding]:
        occurrence_id = occurrence if isinstance(occurrence, str) else occurrence.occurrence_id
        proposals: dict[str, RuntimeBinding] = {}
        for role, value in arguments.items():
            proposal = RuntimeBinding(
                role, value, (semantic_types or {}).get(role, "entity"), BindingSource.AGENT_PROPOSED,
                BindingStatus.PROPOSED, BindingResolution.SEMANTIC, [], revision,
            )
            self._proposals[(occurrence_id, role)] = proposal
            proposals[role] = proposal
        return proposals

    def ground_from_evidence(
        self, occurrence_id: str, proposal: dict[str, RuntimeBinding],
        constraints: list[GroundingConstraint], evidence_store: Any,
    ) -> tuple[dict[str, RuntimeBinding], list[str]]:
        values = {role: binding.value for role, binding in proposal.items()}
        matches: list[str] = []
        relation_roles: set[str] = set()
        for constraint in constraints:
            result = evidence_store.match_constraint(constraint, values, proposal[next(iter(proposal))].world_revision if proposal else evidence_store.revision)
            if not result:
                return {}, []
            matches.extend(item.evidence_id for item in result)
            if constraint.required_resolution == "relation_verified":
                relation_roles.update(expr.source_role for expr in constraint.argument_mapping.values() if expr.source_role)
        grounded: dict[str, RuntimeBinding] = {}
        for role, binding in proposal.items():
            resolution = BindingResolution.RELATION_VERIFIED if role in relation_roles else BindingResolution.CONCRETE
            grounded[role] = RuntimeBinding(
                role, binding.value, binding.semantic_type, BindingSource.HARNESS_EVIDENCE,
                BindingStatus.GROUNDED, resolution, list(dict.fromkeys(matches)), binding.world_revision,
            )
        return grounded, list(dict.fromkeys(matches))

    def commit_grounded(self, occurrence_id: str, bindings: dict[str, RuntimeBinding]) -> None:
        for binding in bindings.values():
            self._set(occurrence_id, binding, "grounding_preflight_passed")

    def commit_atomic_effect_witnesses(
        self,
        occurrence_id: str,
        values: dict[str, Any],
        input_specs: list[ParameterSpec],
        witness_refs: list[str],
        revision: int,
    ) -> dict[str, RuntimeBinding]:
        """Commit validator-resolved concrete inputs after full effect proof."""

        if not witness_refs:
            raise ValueError("Atomic effect witness bindings require validator refs")
        by_role = {item.name: item for item in input_specs}
        unknown = set(values) - set(by_role)
        if unknown:
            raise ValueError(
                f"Atomic effect witnesses reference unknown input roles: {sorted(unknown)}"
            )
        if any(value in (None, "") for value in values.values()):
            raise ValueError("Atomic effect witness bindings must be concrete")
        pending = {
            role: RuntimeBinding(
                role=role,
                value=value,
                semantic_type=by_role[role].semantic_type,
                source=BindingSource.HARNESS_EVIDENCE,
                status=BindingStatus.GROUNDED,
                resolution=BindingResolution.CONCRETE,
                evidence_refs=list(dict.fromkeys(witness_refs)),
                world_revision=revision,
            )
            for role, value in values.items()
        }
        committed: dict[str, RuntimeBinding] = {}
        for role, binding in pending.items():
            self._set(
                occurrence_id,
                binding,
                "atomic_effect_witness_grounded",
            )
            committed[role] = binding
        return committed

    def invalidate_revision(self, revision: int) -> None:
        for key, binding in list(self._bindings.items()):
            if binding.source is BindingSource.HARNESS_EVIDENCE and binding.world_revision < revision:
                invalid = RuntimeBinding(
                    binding.role, binding.value, binding.semantic_type, binding.source,
                    BindingStatus.INVALIDATED, binding.resolution, list(binding.evidence_refs), revision,
                )
                self._set(key[0], invalid, "world_revision_invalidated")

    def publish_validated_outputs(
        self, occurrence: RuntimeOccurrence | str, outputs: dict[str, Any],
        validation_refs: list[str], revision: int = 0,
    ) -> None:
        if not validation_refs:
            raise ValueError("validated outputs require validator witness refs")
        occurrence_id = occurrence if isinstance(occurrence, str) else occurrence.occurrence_id
        for role, value in outputs.items():
            binding = RuntimeBinding(
                role, value, "entity", BindingSource.TOOL_OUTPUT, BindingStatus.GROUNDED,
                BindingResolution.CONCRETE if _looks_concrete(value) else BindingResolution.SEMANTIC,
                list(validation_refs), revision,
            )
            self._outputs[(occurrence_id, role)] = binding
            self._set(occurrence_id, binding, "validated_output_published")

    def snapshot_for_node(self, occurrence: RuntimeOccurrence | str) -> dict[str, RuntimeBinding]:
        occurrence_id = occurrence if isinstance(occurrence, str) else occurrence.occurrence_id
        return {
            role: binding
            for (owner, role), binding in self._bindings.items()
            if owner == occurrence_id
        }

    def runtime_prompt_projection(
        self,
        occurrence: RuntimeOccurrence | str,
        parameters: list[ParameterSpec],
    ) -> dict[str, Any]:
        """Separate semantic intent from bindings that are executable now."""

        occurrence_id = occurrence if isinstance(occurrence, str) else occurrence.occurrence_id
        task_bindings = {
            role: binding for (owner, role), binding in self._bindings.items()
            if owner == "__task__"
        }
        current = self.snapshot_for_node(occurrence_id)
        task_semantic_context = {
            role: binding.value for role, binding in task_bindings.items()
            if binding.status is BindingStatus.GROUNDED
        }
        occurrence_anchors = {
            role: binding.value for role, binding in current.items()
            if binding.source in {BindingSource.TASK, BindingSource.DATA_FLOW}
            and binding.status is BindingStatus.GROUNDED
        }
        execution_ready: dict[str, Any] = {}
        missing: list[str] = []
        for parameter in parameters:
            binding = current.get(parameter.name)
            sufficient = bool(
                binding is not None
                and binding.status is BindingStatus.GROUNDED
                and resolution_satisfies(
                    binding.resolution, parameter.required_resolution,
                )
            )
            if sufficient:
                execution_ready[parameter.name] = binding.value
            elif parameter.required:
                missing.append(parameter.name)
        return {
            "task_semantic_context": task_semantic_context,
            "occurrence_semantic_anchors": occurrence_anchors,
            "execution_ready_bindings": execution_ready,
            "missing_or_insufficient_bindings": missing,
        }

    def semantic_anchor_for(
        self,
        occurrence: RuntimeOccurrence | str,
        role: str,
    ) -> RuntimeBinding | None:
        """Return an explicitly projected Task/DataFlow anchor for one input."""

        occurrence_id = (
            occurrence if isinstance(occurrence, str) else occurrence.occurrence_id
        )
        binding = self._bindings.get((occurrence_id, str(role)))
        if binding is None or binding.status is not BindingStatus.GROUNDED:
            return None
        if binding.source not in {BindingSource.TASK, BindingSource.DATA_FLOW}:
            return None
        return binding

    def validated_outputs(self, occurrence_id: str) -> dict[str, RuntimeBinding]:
        return {role: binding for (owner, role), binding in self._outputs.items() if owner == occurrence_id}
