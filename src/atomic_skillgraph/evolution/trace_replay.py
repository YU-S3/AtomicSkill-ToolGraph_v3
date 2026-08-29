"""Fresh-episode replay authority for typed Atomic/Implementation repairs."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from ..core.bindings import BindingExprKind, BindingExpression
from ..core.contracts import AbstractAtomicSkill, CompositeSkill, ImplementationAtom
from ..core.errors import AtomicSkillGraphError
from ..core.refs import content_hash
from ..core.results import PrimitiveToolStep, RuntimeOccurrence
from ..core.serialization import to_primitive
from ..core.status import SkillStatus
from .typed_repairs import RepairEvidence


_INTRINSIC_CODES = {
    "atomic": {"atomic_effect_violation"},
    "implementation": {
        "implementation_mapping_error",
        "implementation_constraint_error",
        "implementation_compatibility_error",
        "implementation_invocation_failed",
    },
}


def build_trace_repair_evidence(
    payload: Mapping[str, Any],
    failure: Mapping[str, Any],
    *,
    target_layer: str,
    target_ref: str,
    skills: Any,
    harness_profile: str,
) -> RepairEvidence | None:
    """Build one replay case only from a matching started intrinsic attempt."""
    code = str(failure.get("code", ""))
    started = failure.get("started") is True
    preflight_implementation = (
        target_layer == "implementation"
        and code in {
            "implementation_mapping_error",
            "implementation_constraint_error",
            "implementation_compatibility_error",
        }
    )
    if (
        target_layer not in _INTRINSIC_CODES
        or str(failure.get("layer", "")) != target_layer
        or code not in _INTRINSIC_CODES[target_layer]
        or (not started and not preflight_implementation)
    ):
        return None
    task = dict(payload.get("task") or {})
    task_id = str(failure.get("task_id") or task.get("task_id") or "")
    trace_id = str(failure.get("trace_id") or payload.get("trace_id") or "")
    attempt_id = str(failure.get("attempt_id", ""))
    occurrence_id = str(failure.get("occurrence_id", ""))
    failure_id = str(failure.get("failure_id", ""))
    if not all((task_id, trace_id, attempt_id, occurrence_id, failure_id)):
        return None

    invocations = [dict(item) for item in payload.get("implementation_invocations", [])]
    invocation = next(
        (
            item for item in invocations
            if str(item.get("attempt_id", "")) == attempt_id
            and str(item.get("occurrence_id", "")) == occurrence_id
        ),
        None,
    )
    if invocation is None:
        invocation = next(
            (
                item for item in invocations
                if str(item.get("occurrence_id", "")) == occurrence_id
                and (
                    target_layer != "implementation"
                    or str(item.get("implementation_ref", "")) == target_ref
                )
            ),
            None,
        )
    if invocation is None or not isinstance(invocation.get("arguments"), dict):
        return None
    bindings = dict(invocation["arguments"])
    if not bindings:
        return None
    if (
        target_layer == "implementation"
        and str(invocation.get("implementation_ref", "")) != target_ref
    ):
        return None

    node = next(
        (
            dict(item) for item in payload.get("node_records", [])
            if str(item.get("occurrence_id", "")) == occurrence_id
        ),
        None,
    )
    if node is None:
        return None
    try:
        if target_layer == "atomic":
            source = skills.get_atomic(target_ref)
            if str(node.get("atomic_ref", "")) != target_ref:
                return None
            atomic = source
            constraints: list[Any] = []
        else:
            source = skills.get_implementation(target_ref)
            atomic = skills.get_atomic(source.abstract_ref)
            constraints = to_primitive(source.grounding_constraints)
    except KeyError:
        return None

    span_id = str(invocation.get("span_id", ""))
    span = next(
        (
            dict(item) for item in payload.get("runtime_spans", [])
            if str(item.get("span_id", "")) == span_id
        ),
        None,
    )
    if span is None:
        return None
    start, end = int(span.get("action_start", -1)), int(span.get("action_end", -1))
    actions = list(payload.get("environment_actions", []))
    if start < 0 or end < start or end > len(actions):
        return None

    def accepted(raw: Any) -> dict[str, Any] | None:
        item = dict(raw)
        if item.get("accepted") is not True:
            return None
        return {
            "action_type": str(item.get("action_type", "")),
            "arguments": dict(item.get("arguments") or {}),
        }

    prefix = [item for raw in actions[:start] if (item := accepted(raw)) is not None]
    occurrence_actions = [
        item for raw in actions[start:end] if (item := accepted(raw)) is not None
    ]
    if target_layer == "atomic" and not occurrence_actions:
        return None
    source_task = {
        "task_id": str(task.get("task_id", "")),
        "task_signature": str(task.get("task_signature", "")),
        "goal": str(task.get("goal", "")),
        "benchmark": str(task.get("benchmark", "")),
        "task_type": str(task.get("task_type", "")),
        "context": {
            "env_index": dict(task.get("metadata") or {}).get("env_index"),
            "game_file": dict(task.get("metadata") or {}).get("game_file", ""),
        },
        "metadata": dict(task.get("metadata") or {}),
    }
    if not all(source_task.get(key) for key in ("task_id", "goal", "benchmark", "task_type")):
        return None
    semantic_types = {
        item.name: item.semantic_type for item in atomic.inputs
    }
    harness_context = {
        "profile": str(harness_profile),
        "task_type": source_task["task_type"],
        "split": str(source_task["metadata"].get("split", "")),
    }
    cluster_key = content_hash({
        "failure_code": code,
        "input_semantic_types": semantic_types,
        "harness_context": harness_context,
        "parameter_constraints": constraints,
    })
    return RepairEvidence(
        evidence_id=failure_id,
        task_id=task_id,
        trace_id=trace_id,
        cluster_key=cluster_key,
        replay_case={
            "kind": "typed_trace_replay",
            "target_layer": target_layer,
            "target_ref": target_ref,
            "source_task": source_task,
            "occurrence_id": occurrence_id,
            "bindings": bindings,
            "prefix": prefix,
            "occurrence_actions": occurrence_actions,
            "source_attempt_started": started,
        },
        failure_layer=target_layer,
        failure_code=code,
    )


class TraceRepairExecutor:
    """Replay typed candidates in a fresh Harness episode and admit statically."""

    def __init__(
        self,
        *,
        harness: Any,
        skills: Any,
        tools: Any,
        validation: Any,
        admission: Any,
    ) -> None:
        self.harness = harness
        self.skills = skills
        self.tools = tools
        self.validation = validation
        self.admission = admission

    def replay(
        self, candidate: AbstractAtomicSkill | ImplementationAtom, case: dict[str, Any],
    ) -> bool:
        try:
            if isinstance(candidate, AbstractAtomicSkill):
                return self._replay_atomic(candidate, case)
            if isinstance(candidate, ImplementationAtom):
                return self._replay_implementation(candidate, case)
            return False
        except AtomicSkillGraphError:
            raise
        except (KeyError, TypeError, ValueError):
            return False

    def replay_composite(
        self, candidate: CompositeSkill, case: dict[str, Any],
    ) -> bool:
        """Execute every reordered occurrence against a fresh benchmark task."""
        try:
            if case.get("target_layer") != "composite":
                return False
            occurrence_cases = dict(case.get("occurrence_cases") or {})
            by_step = {item.step_id: item for item in candidate.occurrences}
            if set(candidate.control_sequence) != set(by_step):
                return False
            self._reset_and_execute_prefix(case)
            for step_id in candidate.control_sequence:
                occurrence = by_step[step_id]
                source = dict(
                    occurrence_cases.get(occurrence.occurrence_id)
                    or occurrence_cases.get(step_id)
                    or {}
                )
                implementation_ref = str(source.get("implementation_ref") or "")
                values = dict(source.get("bindings") or {})
                if not implementation_ref or not values:
                    return False
                implementation = self.skills.get_implementation(implementation_ref)
                if implementation.abstract_ref != occurrence.node_ref:
                    return False
                passed, terminal_won = self._execute_implementation_body(
                    implementation, values, occurrence.occurrence_id,
                )
                if not passed:
                    return False
                if terminal_won:
                    return self.validation.task.validate(
                        candidate.goal_contract, self.harness.validator_channel(),
                    ).passed
            return self.validation.task.validate(
                candidate.goal_contract, self.harness.validator_channel(),
            ).passed
        except AtomicSkillGraphError:
            raise
        except (KeyError, TypeError, ValueError):
            return False

    def validate(self, candidate: AbstractAtomicSkill | ImplementationAtom) -> bool:
        if isinstance(candidate, AbstractAtomicSkill):
            return self._atomic_static(candidate)
        return self._admit_implementation(candidate).status is SkillStatus.CANDIDATE

    def admit(
        self, candidate: AbstractAtomicSkill | ImplementationAtom,
    ) -> AbstractAtomicSkill | ImplementationAtom:
        if isinstance(candidate, AbstractAtomicSkill):
            return replace(
                candidate,
                status=(
                    SkillStatus.CANDIDATE
                    if self._atomic_static(candidate)
                    else SkillStatus.SHADOW
                ),
            )
        return self._admit_implementation(candidate)

    def _replay_atomic(
        self, candidate: AbstractAtomicSkill, case: dict[str, Any],
    ) -> bool:
        if case.get("target_layer") != "atomic":
            return False
        bindings = dict(case.get("bindings") or {})
        actions = list(case.get("occurrence_actions") or [])
        tool_ref = str(case.get("tool_ref") or "")
        if not bindings or (not actions and not tool_ref):
            return False
        self._reset_and_execute_prefix(case)
        if actions:
            if not self._execute_events(actions):
                return False
        else:
            tool = self.tools.get(tool_ref)
            required = set(tool.signature.get("required") or [])
            if required - set(bindings):
                return False
            executed = 0
            for raw in tool.artifact.get("steps") or []:
                arguments = {
                    role: self._resolve(expression, bindings, {})
                    for role, expression in dict(
                        raw.get("argument_mapping") or {}
                    ).items()
                }
                primitive = PrimitiveToolStep(
                    str(raw["action_type"]),
                    {
                        role: BindingExpression(
                            BindingExprKind.CONSTANT, constant=value,
                        )
                        for role, value in arguments.items()
                    },
                )
                result = self.harness.execute_primitive(primitive, {})
                executed += 1
                if not result.accepted or (result.done and not result.won):
                    return False
            if executed <= 0:
                return False
        occurrence = RuntimeOccurrence(
            "repair", str(case.get("occurrence_id", "repair")), candidate.ref,
            [], {}, [], list(candidate.effects),
        )
        outputs = {
            item.name: bindings[item.name]
            for item in candidate.outputs if item.name in bindings
        }
        return self.validation.atomic.validate(
            candidate,
            occurrence,
            bindings,
            self.harness.validator_channel(),
            outputs,
        ).passed

    def _replay_implementation(
        self, candidate: ImplementationAtom, case: dict[str, Any],
    ) -> bool:
        if case.get("target_layer") != "implementation":
            return False
        atomic_values = dict(case.get("bindings") or {})
        if not atomic_values:
            return False
        self._reset_and_execute_prefix(case)
        if (
            case.get("source_attempt_started") is not True
            and not self._execute_events(list(case.get("occurrence_actions") or []))
        ):
            return False
        passed, _terminal_won = self._execute_implementation_body(
            candidate, atomic_values, str(case.get("occurrence_id", "repair")),
        )
        return passed

    def _execute_implementation_body(
        self,
        candidate: ImplementationAtom,
        atomic_values: Mapping[str, Any],
        occurrence_id: str,
    ) -> tuple[bool, bool]:
        atomic = self.skills.get_atomic(candidate.abstract_ref)
        if not self._constraints_pass(candidate, atomic_values):
            return False, False
        tool_outputs: dict[tuple[str, str], Any] = {}
        executed = 0
        terminal_won = False
        for binding in sorted(candidate.tool_bindings, key=lambda item: item.order):
            tool = self.tools.get(binding.tool_ref)
            arguments = {
                role: self._resolve(expression, atomic_values, tool_outputs)
                for role, expression in binding.parameter_mapping.items()
            }
            required = set(tool.signature.get("required") or [])
            if required - set(arguments):
                return False
            for raw in tool.artifact.get("steps") or []:
                primitive = PrimitiveToolStep(
                    str(raw["action_type"]), dict(raw.get("argument_mapping") or {}),
                )
                result = self.harness.execute_primitive(primitive, arguments)
                executed += 1
                if not result.accepted or (result.done and not result.won):
                    return False, False
                terminal_won = terminal_won or bool(result.done and result.won)
            for role, expression in dict(tool.artifact.get("output_mapping") or {}).items():
                value = self._resolve(expression, arguments, {})
                tool_outputs[(binding.role, str(role))] = value
        if executed <= 0:
            return False, False
        output_candidates = {
            role: self._resolve(expression, atomic_values, tool_outputs)
            for role, expression in dict(
                candidate.execution_policy.get("output_mapping") or {}
            ).items()
        }
        occurrence = RuntimeOccurrence(
            "repair", occurrence_id, atomic.ref,
            [], {}, [], list(atomic.effects),
        )
        passed = self.validation.atomic.validate(
            atomic,
            occurrence,
            atomic_values,
            self.harness.validator_channel(),
            output_candidates,
        ).passed
        return passed, terminal_won

    def _reset_and_execute_prefix(self, case: dict[str, Any]) -> None:
        source = dict(case.get("source_task") or {})
        from ..harness.protocol import HarnessTask

        task = HarnessTask(
            str(source["task_id"]), str(source["goal"]), str(source["benchmark"]),
            str(source["task_type"]), dict(source.get("context") or {}),
            dict(source.get("metadata") or {}),
        )
        self.harness.reset(task)
        if not self._execute_events(list(case.get("prefix") or [])):
            raise ValueError("typed replay prefix rejected")

    def _execute_events(self, events: list[dict[str, Any]]) -> bool:
        for event in events:
            primitive = PrimitiveToolStep(
                str(event["action_type"]),
                {
                    str(role): BindingExpression(
                        BindingExprKind.CONSTANT, constant=value,
                    )
                    for role, value in dict(event.get("arguments") or {}).items()
                },
            )
            result = self.harness.execute_primitive(primitive, {})
            if not result.accepted or (result.done and not result.won):
                return False
        return True

    @staticmethod
    def _resolve(
        raw: Any,
        skill_inputs: Mapping[str, Any],
        tool_outputs: Mapping[tuple[str, str], Any],
    ) -> Any:
        expression = BindingExpression.from_dict(raw) if isinstance(raw, dict) else raw
        if not isinstance(expression, BindingExpression):
            raise TypeError("typed repair mapping must use BindingExpression")
        if expression.kind is BindingExprKind.CONSTANT:
            return expression.constant
        if expression.kind is BindingExprKind.SKILL_INPUT:
            return skill_inputs[expression.source_role]
        if expression.kind is BindingExprKind.TOOL_OUTPUT:
            return tool_outputs[(expression.source_step, expression.source_role)]
        raise ValueError("unsupported typed repair binding expression")

    @staticmethod
    def _atomic_static(candidate: AbstractAtomicSkill) -> bool:
        input_names = [item.name for item in candidate.inputs]
        output_names = [item.name for item in candidate.outputs]
        if (
            not candidate.effects
            or len(input_names) != len(set(input_names))
            or len(output_names) != len(set(output_names))
            or not str(candidate.validator_spec.get("validator_id", ""))
        ):
            return False
        available = set(input_names)
        for predicate in [*candidate.preconditions, *candidate.effects]:
            for raw in predicate.args.values():
                if isinstance(raw, BindingExpression) and (
                    raw.kind is not BindingExprKind.CONSTANT
                    and raw.source_role not in available
                ):
                    return False
        return True

    def _admit_implementation(self, candidate: ImplementationAtom) -> ImplementationAtom:
        try:
            atomic = self.skills.get_atomic(candidate.abstract_ref)
        except KeyError:
            return replace(
                candidate, status=SkillStatus.SHADOW,
                quality={**candidate.quality, "admission_failure": ["abstract_ref_missing"]},
            )
        try:
            tools = [
                self.tools.get(item.tool_ref) for item in candidate.tool_bindings
            ]
        except KeyError:
            return replace(
                candidate, status=SkillStatus.SHADOW,
                quality={**candidate.quality, "admission_failure": ["tool_ref_missing"]},
            )
        return self.admission.admit_implementation(
            candidate,
            tools,
            atomic=atomic,
            harness=self.harness,
        )

    def _constraints_pass(
        self, candidate: ImplementationAtom, bindings: Mapping[str, Any],
    ) -> bool:
        profiles = set(candidate.compatibility.get("harness_profiles") or [])
        if str(self.harness.profile_name) not in profiles:
            return False
        seen: set[str] = set()
        for constraint in candidate.grounding_constraints:
            if (
                not constraint.constraint_id
                or constraint.constraint_id in seen
                or constraint.required_resolution not in {
                    "semantic", "concrete", "relation_verified",
                }
                or not self.harness.supports_constraint(
                    constraint.kind.value, constraint.verifier_id,
                )
            ):
                return False
            seen.add(constraint.constraint_id)
            try:
                values = {
                    role: self._resolve(raw, bindings, {})
                    for role, raw in constraint.argument_mapping.items()
                }
            except (KeyError, TypeError, ValueError):
                return False
            if any(value is None or value == "" for value in values.values()):
                return False
            if constraint.kind.value in {"argument_exists", "argument_concrete"}:
                if not values:
                    return False
                continue
            if constraint.kind.value in {"harness_affordance", "current_context"}:
                if not constraint.action_type:
                    return False
                primitive = PrimitiveToolStep(
                    constraint.action_type,
                    {
                        role: BindingExpression(
                            BindingExprKind.CONSTANT, constant=value,
                        )
                        for role, value in values.items()
                    },
                )
                try:
                    self.harness.compile_primitive(primitive, {})
                except (KeyError, TypeError, ValueError):
                    return False
                continue
            # A custom verifier is not executable through the v3 Harness
            # protocol and therefore cannot be replay-certified here.
            return False
        return True


__all__ = ["TraceRepairExecutor", "build_trace_repair_evidence"]
