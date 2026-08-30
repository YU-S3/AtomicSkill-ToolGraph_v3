"""ALFWorld v3 adapter: raw commands are parsed exactly once at this boundary."""

from __future__ import annotations

import os
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..core.bindings import BindingExpression, BindingExprKind
from ..core.contracts import (
    ContractSource, IdentityConstraint, IdentityRelation, SemanticPredicate, TaskContract,
)
from ..core.errors import AtomicSkillGraphError, FailureLayer
from ..core.results import PrimitiveToolStep, ValidationResult
from ..validation.contract_matcher import ContractMatcher
from .action_catalog import HarnessActionCatalog
from .protocol import (
    HarnessActionResult,
    HarnessActionSpec,
    HarnessTask,
    build_transition_certificate,
)


TASK_TYPE_IDS = {
    "pick_and_place_simple": 1,
    "look_at_obj_in_light": 2,
    "pick_clean_then_place_in_recep": 3,
    "pick_heat_then_place_in_recep": 4,
    "pick_cool_then_place_in_recep": 5,
    "pick_two_obj_and_place": 6,
}

_GAME_TYPE_RE = re.compile("(" + "|".join(map(re.escape, TASK_TYPE_IDS)) + ")")


def normalize_entity(value: Any) -> str:
    return re.sub(r"\s+", "_", str(value).strip().casefold())


def same_entity_family(left: Any, right: Any) -> bool:
    def family(value: Any) -> str:
        normalized = re.sub(r"_\d+$", "", normalize_entity(value))
        # Goal text may use ``alarm clock`` while admissible commands use
        # ``alarmclock 1``.  Separators are not semantic in ALFWorld names.
        return re.sub(r"[^a-z0-9]", "", normalized)

    left_family, right_family = family(left), family(right)
    return bool(left_family and right_family and left_family == right_family)


def entity_matches(left: Any, right: Any) -> bool:
    """Match a semantic family, but preserve identity for concrete instances."""
    expected = normalize_entity(right)
    if re.search(r"_\d+$", expected):
        return normalize_entity(left) == expected
    return same_entity_family(left, expected)


def semantic_value_compatible(
    *,
    role: str,
    concrete_value: Any,
    semantic_anchor: Any,
    semantic_type: str,
) -> bool:
    """Check a concrete ALFWorld proposal against its semantic task anchor.

    Semantic entity anchors such as ``apple`` accept concrete members such as
    ``apple_2``.  Concrete anchors preserve identity.  Non-entity scalar
    values use equality after harmless string normalization.
    """

    if semantic_anchor is None or semantic_anchor == "":
        return True
    kind = str(semantic_type).strip().casefold()
    entity_roles = {
        "object", "item", "source", "destination", "location", "station",
        "tool", "light", "light_source", "held_object", "target_location",
        "object_location", "container", "receptacle", "appliance",
    }
    entity_typed = kind in {
        "entity", "object", "location", "container", "receptacle",
        "appliance", "tool", "light", "station",
    } or str(role).strip().casefold() in entity_roles
    if entity_typed and isinstance(concrete_value, str) and isinstance(semantic_anchor, str):
        return entity_matches(concrete_value, semantic_anchor)
    return concrete_value == semantic_anchor


_ACTION_PATTERNS: list[tuple[str, re.Pattern[str], tuple[str, ...]]] = [
    ("TAKE", re.compile(r"^take (.+?) from (.+)$", re.I), ("object", "source")),
    ("PUT", re.compile(r"^put (.+?) in/on (.+)$", re.I), ("object", "destination")),
    ("MOVE", re.compile(r"^move (.+?) to (.+)$", re.I), ("object", "destination")),
    ("HEAT", re.compile(r"^heat (.+?) with (.+)$", re.I), ("object", "station")),
    ("COOL", re.compile(r"^cool (.+?) with (.+)$", re.I), ("object", "station")),
    ("CLEAN", re.compile(r"^clean (.+?) with (.+)$", re.I), ("object", "station")),
    ("SLICE", re.compile(r"^slice (.+?) with (.+)$", re.I), ("object", "tool")),
    ("GO_TO", re.compile(r"^go to (.+)$", re.I), ("destination",)),
    ("OPEN", re.compile(r"^open (.+)$", re.I), ("object",)),
    ("CLOSE", re.compile(r"^close (.+)$", re.I), ("object",)),
    ("TOGGLE_ON", re.compile(r"^(?:turn on|toggle) (.+)$", re.I), ("object",)),
    ("TOGGLE_OFF", re.compile(r"^turn off (.+)$", re.I), ("object",)),
    ("EXAMINE", re.compile(r"^(?:examine|look at) (.+)$", re.I), ("object",)),
    ("USE", re.compile(r"^use (.+)$", re.I), ("object",)),
]


def parse_alfworld_action(raw_action: Any) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    """The only ALFWorld raw-command parser used by v3 runtime."""
    text = re.sub(r"\s+", " ", str(raw_action).strip())
    for action_type, pattern, roles in _ACTION_PATTERNS:
        match = pattern.fullmatch(text)
        if match:
            arguments = {role: normalize_entity(value) for role, value in zip(roles, match.groups())}
            return action_type, arguments, text, {"parser": "alfworld_v3"}
    lowered = text.casefold()
    if lowered == "inventory":
        return "INVENTORY", {}, text, {"parser": "alfworld_v3"}
    if lowered == "look":
        return "LOOK", {}, text, {"parser": "alfworld_v3"}
    return "UNKNOWN", {}, text, {"parser": "alfworld_v3", "unparsed": True}


class AlfWorldContractMatcher:
    """Value-sensitive goal-family matcher owned by the ALFWorld adapter."""

    def effect_covers_target(
        self,
        *,
        offered_predicate: SemanticPredicate,
        offered_arguments: dict[str, Any],
        target_predicate: SemanticPredicate,
    ) -> bool:
        if target_predicate.predicate.casefold() != offered_predicate.predicate.casefold():
            return False
        if set(target_predicate.args) != set(offered_arguments):
            return False
        for role, expected in target_predicate.args.items():
            observed = offered_arguments.get(role)
            if isinstance(expected, str) and isinstance(observed, str):
                if not entity_matches(observed, expected):
                    return False
            elif observed != expected:
                return False
        return True

    def matches(
        self,
        target: SemanticPredicate,
        offered: SemanticPredicate,
        offered_args: dict[str, Any],
    ) -> bool:
        """Compatibility alias for callers on the previous matcher protocol."""

        return self.effect_covers_target(
            offered_predicate=offered,
            offered_arguments=offered_args,
            target_predicate=target,
        )


class AlfWorldValidatorChannel:
    """Private action-derived facts plus official win signal; never policy-facing."""

    validation_strength = "official_goal_plus_action_derived_effects"

    def __init__(self) -> None:
        self.revision = 0
        self.won = False
        self.done = False
        self._facts: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
        self._held: set[str] = set()
        self._locations: dict[str, str] = {}
        self._properties: dict[str, set[str]] = {
            "object.heated": set(), "object.cooled": set(),
            "object.cleaned": set(), "object.sliced": set(),
        }
        self._containers_open: set[str] = set()
        self._containers_closed: set[str] = set()
        self._lights_on: set[str] = set()
        self._lights_off: set[str] = set()
        self._observed: set[str] = set()
        self._observed_with: set[tuple[str, str]] = set()
        self._agent_location = ""

    def reset(self) -> None:
        self.revision = 0
        self.won = self.done = False
        self._facts.clear()
        self._held.clear()
        self._locations.clear()
        for values in self._properties.values():
            values.clear()
        self._containers_open.clear()
        self._containers_closed.clear()
        self._lights_on.clear()
        self._lights_off.clear()
        self._observed.clear()
        self._observed_with.clear()
        self._agent_location = ""

    def _rebuild_facts(self) -> None:
        facts: set[tuple[str, tuple[tuple[str, str], ...]]] = set()

        def add(predicate: str, **arguments: str) -> None:
            facts.add((predicate, tuple(sorted((key, str(value)) for key, value in arguments.items()))))

        for obj in self._held:
            add("agent.holds", object=obj)
        for obj, location in self._locations.items():
            add("object.at_location", object=obj, location=location)
        for predicate, objects in self._properties.items():
            for obj in objects:
                add(predicate, object=obj)
        if self._agent_location:
            add("agent.at_location", location=self._agent_location)
        for container in self._containers_open:
            add("container.open", container=container)
        for container in self._containers_closed:
            add("container.closed", container=container)
        for light in self._lights_on:
            add("light.on", light=light)
        for light in self._lights_off:
            add("light.off", light=light)
        for obj in self._observed:
            add("object.observed", object=obj)
        for obj, light in self._observed_with:
            add("object.observed_with", object=obj, light=light)
        self._facts = facts

    def record(self, spec: HarnessActionSpec, *, accepted: bool, revision: int, done: bool, won: bool) -> None:
        self.revision, self.done, self.won = revision, done, won
        if not accepted:
            return
        args = spec.arguments
        obj = str(args.get("object", ""))
        if spec.action_type == "TAKE":
            if obj:
                self._held.add(obj)
                self._locations.pop(obj, None)
        elif spec.action_type in {"PUT", "MOVE"}:
            destination = str(args.get("destination", ""))
            if obj and destination:
                self._held.discard(obj)
                self._locations[obj] = destination
        elif spec.action_type == "HEAT":
            if obj:
                self._properties["object.heated"].add(obj)
                self._properties["object.cooled"].discard(obj)
        elif spec.action_type == "COOL":
            if obj:
                self._properties["object.cooled"].add(obj)
                self._properties["object.heated"].discard(obj)
        elif spec.action_type == "CLEAN":
            if obj:
                self._properties["object.cleaned"].add(obj)
        elif spec.action_type == "SLICE":
            if obj:
                self._properties["object.sliced"].add(obj)
        elif spec.action_type == "GO_TO":
            self._agent_location = str(args.get("destination", ""))
        elif spec.action_type == "OPEN":
            if obj:
                self._containers_closed.discard(obj)
                self._containers_open.add(obj)
        elif spec.action_type == "CLOSE":
            if obj:
                self._containers_open.discard(obj)
                self._containers_closed.add(obj)
        elif spec.action_type == "TOGGLE_ON":
            if obj:
                self._lights_off.discard(obj)
                self._lights_on.add(obj)
        elif spec.action_type == "TOGGLE_OFF":
            if obj:
                self._lights_on.discard(obj)
                self._lights_off.add(obj)
        elif spec.action_type == "USE":
            if obj:
                # ALFWorld exposes lamp interaction as ``use <lamp>``.  The
                # generated look-at tasks start with the target lamp off, and
                # an accepted USE toggles its state.  The observation is a
                # contextual effect: the object is the concrete item already
                # held, while the action argument identifies the lamp.
                if obj in self._lights_on:
                    self._lights_on.discard(obj)
                    self._lights_off.add(obj)
                    self._observed_with = {
                        pair for pair in self._observed_with if pair[1] != obj
                    }
                else:
                    self._lights_off.discard(obj)
                    self._lights_on.add(obj)
                    for held_object in self._held:
                        self._observed_with.add((held_object, obj))
        elif spec.action_type == "EXAMINE":
            if obj:
                self._observed.add(obj)
        self._rebuild_facts()

    def snapshot(self) -> dict[str, Any]:
        return {
            "revision": self.revision, "done": self.done, "won": self.won,
            "facts": [{"predicate": predicate, "args": dict(arguments)} for predicate, arguments in sorted(self._facts)],
            "validation_strength": self.validation_strength,
        }

    @staticmethod
    def _expected_args(
        predicate: SemanticPredicate | dict[str, Any], bindings: dict[str, Any],
    ) -> tuple[str, dict[str, Any], int, str]:
        name = predicate.predicate if isinstance(predicate, SemanticPredicate) else str(predicate.get("predicate", ""))
        raw_args = predicate.args if isinstance(predicate, SemanticPredicate) else dict(predicate.get("args", {}))
        cardinality = predicate.cardinality if isinstance(predicate, SemanticPredicate) else int(predicate.get("cardinality", 1))
        distinct_by = predicate.distinct_by if isinstance(predicate, SemanticPredicate) else str(predicate.get("distinct_by", ""))
        expected: dict[str, Any] = {}
        for role, value in raw_args.items():
            if isinstance(value, BindingExpression):
                expected[role] = bindings.get(value.source_role)
            elif isinstance(value, str) and value.startswith("$"):
                expected[role] = bindings.get(value[1:])
            else:
                expected[role] = bindings.get(role, value)
        return name, expected, max(1, int(cardinality)), distinct_by

    def _matching_facts(
        self, predicate: SemanticPredicate | dict[str, Any], bindings: dict[str, Any],
    ) -> list[dict[str, str]]:
        name, expected, _, _ = self._expected_args(predicate, bindings)
        matches: list[dict[str, str]] = []
        for fact_name, fact_items in self._facts:
            if fact_name != name:
                continue
            actual = dict(fact_items)
            if all(value in (None, "") or entity_matches(actual.get(role, ""), value) for role, value in expected.items()):
                matches.append(actual)
        return matches

    def _matches(self, predicate: SemanticPredicate | dict[str, Any], bindings: dict[str, Any]) -> bool:
        _, _, cardinality, distinct_by = self._expected_args(predicate, bindings)
        matches = self._matching_facts(predicate, bindings)
        if distinct_by:
            return len({item.get(distinct_by, "") for item in matches if item.get(distinct_by)}) >= cardinality
        return len(matches) >= cardinality

    def validate_atomic_effect(self, request: dict[str, Any]) -> ValidationResult:
        effects = request.get("effects", [])
        bindings = request.get("bindings", {})
        checks: dict[str, bool] = {}
        for index, effect in enumerate(effects):
            _, raw_expected, _, _ = self._expected_args(effect, bindings)
            # Atomic completion/AlreadySatisfied must be tied to concrete
            # realized entities, never only to a goal class such as ``apple``.
            concrete = all(
                value not in (None, "") and bool(re.search(r"(?:_|\s)\d+$", normalize_entity(value)))
                for value in raw_expected.values()
            )
            checks[f"effect_{index}_bindings_concrete"] = concrete
            checks[f"effect_{index}"] = concrete and self._matches(effect, bindings)
        passed = bool(effects) and all(checks.values())
        if passed:
            witnesses = [f"alfworld_action_fact:r{self.revision}:{key}" for key in checks]
            return ValidationResult("atomic", True, checks=checks, witness_refs=witnesses)
        return ValidationResult(
            "atomic", False, checks=checks, failure_codes=["atomic_effect_violation"],
            messages=["declared Atomic effect has no current validator witness"],
        )

    def validate_task_contract(self, contract: TaskContract) -> ValidationResult:
        # The benchmark win signal is deliberately not consulted here.
        # TaskValidator combines it with this independent action-derived
        # contract result at the terminal boundary.
        matches = [self._matching_facts(effect, {}) for effect in contract.target_effects]
        checks = {f"target_{index}": self._matches(effect, {}) for index, effect in enumerate(contract.target_effects)}
        checks["contract_mapped_from_goal"] = bool(contract.target_effects)
        cardinality_ok = True
        for constraint in contract.cardinality_constraints:
            predicate = str(constraint.get("predicate", ""))
            role = str(constraint.get("distinct_by") or constraint.get("role") or "object")
            count = max(1, int(constraint.get("count", 1)))
            candidates = [
                item for index, effect_matches in enumerate(matches)
                if contract.target_effects[index].predicate == predicate
                for item in effect_matches
            ]
            cardinality_ok &= len({item.get(role, "") for item in candidates if item.get(role)}) >= count
        checks["cardinality_constraints"] = cardinality_ok

        identity_ok = True
        for constraint in contract.identity_constraints:
            if constraint.relation is IdentityRelation.DISTINCT_FROM:
                # The adapter uses object_1/object_2 to make the pick-two
                # obligation explicit; the actual witnesses are certified by
                # the corresponding distinct_by cardinality constraint.
                object_values = {
                    item.get("object", "") for effect_matches in matches for item in effect_matches
                    if item.get("object")
                }
                identity_ok &= len(object_values) >= 2
            elif constraint.left_role == constraint.right_role:
                # Reusing one logical role across transformation and placement
                # means at least one concrete object must witness every effect
                # that carries that role.
                role = constraint.left_role
                role_sets: list[set[str]] = []
                for effect, effect_matches in zip(contract.target_effects, matches):
                    if role not in effect.args:
                        continue
                    role_sets.append({item.get(role, "") for item in effect_matches if item.get(role)})
                if len(role_sets) > 1:
                    identity_ok &= bool(set.intersection(*role_sets))
        checks["identity_constraints"] = identity_ok
        passed = bool(contract.target_effects) and all(checks.values())
        witnesses = []
        if passed:
            witnesses = [
                f"alfworld_action_fact:r{self.revision}:{effect.predicate}:{index}"
                for index, effect in enumerate(contract.target_effects)
            ]
        return ValidationResult(
            "task_contract", passed, checks=checks,
            failure_codes=[] if passed else ["task_contract_mismatch"],
            messages=[] if passed else ["task contract is not yet satisfied"],
            witness_refs=witnesses,
        )


class AlfWorldAdapter:
    profile_name = "alfworld_v3"

    def __init__(
        self, *, split: str = "eval_out_of_distribution", max_steps: int = 100,
        task_type: str | None = None, alfworld_data: str | None = None,
    ) -> None:
        self.split = split
        self.max_steps = max_steps
        self.task_type = task_type
        self.alfworld_data = alfworld_data or os.environ.get("ALFWORLD_DATA", str(Path.home() / ".cache" / "alfworld"))
        self._env: Any = None
        self._tw_env: Any = None
        self._task_index = 0
        self._revision = 0
        self._catalog = HarnessActionCatalog(parse_alfworld_action)
        self._validator = AlfWorldValidatorChannel()
        self._current_task: HarnessTask | None = None
        self._observation = ""
        self._done = self._won = False

    def _build_config(self) -> dict[str, Any]:
        split_map = {"eval_out_of_distribution": "valid_unseen", "eval_in_distribution": "valid_seen", "train": "train"}
        type_ids = [TASK_TYPE_IDS[self.task_type]] if self.task_type else list(TASK_TYPE_IDS.values())
        data = self.alfworld_data
        return {
            "env": {
                "type": "AlfredTWEnv", "regen_game_files": False, "domain_randomization": False,
                "task_types": type_ids, "expert_type": "handcoded", "goal_desc_human_anns_prob": 0.0,
                "data_path": data,
                "logic": {"domain": os.path.join(data, "logic", "alfred.pddl"), "grammar": os.path.join(data, "logic", "alfred.twl2")},
                "json_game": {"data_path": os.path.join(data, "json_2.1.1")},
            },
            "dataset": {
                "data_path": os.path.join(data, "json_2.1.1"),
                "eval_id_data_path": os.path.join(data, "json_2.1.1", "valid_seen"),
                "eval_ood_data_path": os.path.join(data, "json_2.1.1", "valid_unseen"),
                "num_train_games": -1, "num_eval_games": -1,
            },
            "general": {"training_method": "dagger", "random_seed": 42, "use_cuda": False},
            "dagger": {"training": {
                "batch_size": 10,
                "max_nb_steps_per_episode": self.max_steps,
                "nb_epochs": 50,
            }},
            "controller": {"type": "oracle", "debug": False},
        }

    def initialize(self) -> int:
        try:
            import alfworld.agents.environment as alf_env
        except ImportError as exc:
            raise AtomicSkillGraphError(
                "infrastructure_failure",
                "ALFWorld is not installed; install the 'alfworld' optional dependency",
                layer=FailureLayer.INFRASTRUCTURE,
            ) from exc
        try:
            env_class = alf_env.get_environment("AlfredTWEnv")
            self._tw_env = env_class(self._build_config(), train_eval=self.split)
            self._env = self._tw_env.init_env(batch_size=1)
        except AtomicSkillGraphError:
            raise
        except Exception as exc:
            raise AtomicSkillGraphError(
                "infrastructure_failure", f"failed to initialize ALFWorld: {exc}",
                layer=FailureLayer.INFRASTRUCTURE,
            ) from exc
        files = getattr(self._tw_env, "gamefiles", None) or getattr(self._tw_env, "game_files", None)
        self._task_index = 0
        return len(files) if files is not None else 0

    def _raw_reset(self) -> tuple[HarnessTask, str, list[str]]:
        if self._env is None:
            self.initialize()
        try:
            observations, info = self._env.reset()
        except Exception as exc:
            raise AtomicSkillGraphError(
                "infrastructure_failure", f"ALFWorld reset failed: {exc}",
                layer=FailureLayer.INFRASTRUCTURE,
            ) from exc
        observation = str(observations[0])
        admissible = list(info.get("admissible_commands", [[]])[0])
        game_file = str((info.get("extra.gamefile") or [""])[0])
        match = _GAME_TYPE_RE.search(game_file)
        task_type = match.group(1) if match else "unknown"
        marker = "your task is to:"
        offset = observation.casefold().find(marker)
        goal = observation[offset + len(marker):].strip() if offset >= 0 else observation[:300].strip()
        roles = _goal_roles(goal)
        signature = hashlib.sha256(
            f"{self.split}\x1f{game_file}\x1f{goal}".encode("utf-8")
        ).hexdigest()
        task = HarnessTask(
            task_id=f"alfworld_{self.split}_{self._task_index}_{task_type}", goal=goal, benchmark="alfworld",
            task_type=task_type,
            context={
                "env_index": self._task_index,
                "game_file": game_file,
                "goal_roles": roles,
                "semantic_bindings": roles,
                "binding_types": {role: "entity" for role in roles},
            },
            metadata={"task_signature": signature},
        )
        self._task_index += 1
        return task, observation, admissible

    def load_tasks(self, *, limit: int = 0, task_type: str | None = None) -> list[HarnessTask]:
        if self._env is None:
            total = self.initialize()
        else:
            files = getattr(self._tw_env, "gamefiles", None) or getattr(self._tw_env, "game_files", None)
            total = len(files) if files is not None else 0
        wanted = task_type or self.task_type
        result: list[HarnessTask] = []
        scan_limit = total or (limit * 20 if limit else 10000)
        for _ in range(scan_limit):
            task, observation, admissible = self._raw_reset()
            task.context.update({"initial_observation": observation, "initial_admissible": admissible})
            if not wanted or task.task_type == wanted:
                result.append(task)
            if limit and len(result) >= limit:
                break
        return result

    def load_balanced_tasks(
        self, task_types: list[str], per_type_limit: int,
    ) -> list[HarnessTask]:
        """Select a deterministic 6-way prefix while preserving global env indices."""
        labels = [str(item) for item in task_types]
        if not labels or per_type_limit <= 0:
            raise ValueError("task_types must be non-empty and per_type_limit must be positive")
        unknown = sorted(set(labels) - set(TASK_TYPE_IDS))
        if unknown:
            raise ValueError(f"unknown ALFWorld task types: {unknown}")
        if len(set(labels)) != len(labels):
            raise ValueError("task_types must not contain duplicates")

        # A filtered AlfredTWEnv has a different episode index space.  Formal
        # manifests therefore always scan the unfiltered deterministic order.
        original_type = self.task_type
        self.task_type = None
        try:
            self.initialize()
            files = getattr(self._tw_env, "gamefiles", None) or getattr(self._tw_env, "game_files", None)
            total = len(files) if files is not None else 0
            buckets: dict[str, list[HarnessTask]] = {label: [] for label in labels}
            for _ in range(total):
                task, observation, admissible = self._raw_reset()
                normalized_file = str(task.context.get("game_file", "")).replace("\\", "/")
                if self.split == "train" and "/json_2.1.1/train/" not in normalized_file:
                    continue
                if task.task_type in buckets and len(buckets[task.task_type]) < per_type_limit:
                    task.context.update({
                        "initial_observation": observation,
                        "initial_admissible": admissible,
                    })
                    buckets[task.task_type].append(task)
                if all(len(bucket) >= per_type_limit for bucket in buckets.values()):
                    break
        finally:
            self.task_type = original_type
        missing = {key: len(value) for key, value in buckets.items() if len(value) < per_type_limit}
        if missing:
            raise ValueError(
                f"insufficient balanced ALFWorld tasks: requested {per_type_limit} per type, got {missing}"
            )
        selected = [task for label in labels for task in buckets[label]]
        return sorted(selected, key=lambda task: int(task.context["env_index"]))

    def reset(self, task: HarnessTask) -> HarnessActionResult:
        index = int(task.context.get("env_index", 0))
        self.initialize()
        try:
            for _ in range(index):
                self._env.reset()
                self._task_index += 1
        except Exception as exc:
            raise AtomicSkillGraphError(
                "infrastructure_failure", f"ALFWorld deterministic seek failed: {exc}",
                layer=FailureLayer.INFRASTRUCTURE,
            ) from exc
        actual, observation, admissible = self._raw_reset()
        expected = str(task.context.get("game_file", "")).replace("\\", "/")
        observed = str(actual.context.get("game_file", "")).replace("\\", "/")
        if expected and observed and expected != observed:
            raise AtomicSkillGraphError(
                "infrastructure_failure",
                f"ALFWorld deterministic task mapping changed: expected={expected}, actual={observed}",
                layer=FailureLayer.INFRASTRUCTURE,
            )
        self._current_task, self._observation = task, observation
        self._revision = 0
        self._done = self._won = False
        self._validator.reset()
        catalog = self._catalog.replace(admissible, self._revision)
        return HarnessActionResult(True, observation, False, False, self._revision, catalog, {"reset": True})

    def action_catalog(self) -> list[HarnessActionSpec]:
        return self._catalog.items()

    def semantic_value_compatible(
        self, *, role: str, concrete_value: Any,
        semantic_anchor: Any, semantic_type: str,
    ) -> bool:
        return semantic_value_compatible(
            role=role,
            concrete_value=concrete_value,
            semantic_anchor=semantic_anchor,
            semantic_type=semantic_type,
        )

    def execute_action(self, action_id: str, revision: int) -> HarnessActionResult:
        spec = self._catalog.get(action_id, revision)
        before_snapshot = self._validator.snapshot()
        try:
            observations, scores, dones, infos = self._env.step([spec.raw_action])
        except Exception as exc:
            raise AtomicSkillGraphError(
                "infrastructure_failure", f"ALFWorld step failed: {exc}",
                layer=FailureLayer.INFRASTRUCTURE,
            ) from exc
        observation = str(observations[0])
        done = bool(dones[0])
        won_values = infos.get("won", [False])
        won = bool(won_values[0]) if won_values else False
        # A revision-scoped catalog entry is the admission authority.  A
        # normal env.step return means the admitted action was executed;
        # observation prose is not a semantic acceptance oracle.
        accepted = True
        admissible = list(infos.get("admissible_commands", [[]])[0])
        old_revision = self._revision
        self._revision += 1
        catalog = self._catalog.replace(admissible, self._revision)
        self._observation, self._done, self._won = observation, done, won
        self._validator.record(spec, accepted=accepted, revision=self._revision, done=done, won=won)
        after_snapshot = self._validator.snapshot()

        def fact_identity(raw: dict[str, Any]) -> tuple[str, tuple[tuple[str, Any], ...]]:
            return (
                str(raw.get("predicate", "")),
                tuple(sorted(dict(raw.get("args") or {}).items())),
            )

        before_facts = list(before_snapshot.get("facts", []))
        after_facts = list(after_snapshot.get("facts", []))
        action_values = {
            normalize_entity(value)
            for value in spec.arguments.values()
            if value not in (None, "")
        }
        required: set[tuple[str, tuple[tuple[str, Any], ...]]] = set()
        for fact in before_facts:
            predicate = str(fact.get("predicate", ""))
            fact_values = {
                normalize_entity(value)
                for value in dict(fact.get("args") or {}).values()
            }
            related = bool(action_values.intersection(fact_values))
            if related and predicate in {
                "agent.holds",
                "agent.at_location",
                "container.open",
            }:
                required.add(fact_identity(fact))
        # This adapter knows that contextual device interaction consumes the
        # currently held entity even though that entity is not an action arg.
        if spec.action_type == "USE":
            required.update(
                fact_identity(fact)
                for fact in before_facts
                if str(fact.get("predicate", "")) == "agent.holds"
            )

        before_ids = {fact_identity(item) for item in before_facts}
        terminal: set[tuple[str, tuple[tuple[str, Any], ...]]] = set()
        if won and self._current_task is not None:
            for fact in after_facts:
                fact_id = fact_identity(fact)
                if fact_id in before_ids:
                    continue
                actual_args = dict(fact.get("args") or {})
                for target in self.task_contract(self._current_task).target_effects:
                    if str(fact.get("predicate", "")) != target.predicate:
                        continue
                    if set(actual_args) != set(target.args):
                        continue
                    if all(
                        entity_matches(actual_args[role], expected)
                        for role, expected in target.args.items()
                    ):
                        terminal.add(fact_id)
                        break
        certificate = build_transition_certificate(
            action_id=spec.action_id,
            revision_before=old_revision,
            revision_after=self._revision,
            action_type=spec.action_type,
            arguments=dict(spec.arguments),
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
            accepted=accepted,
            required_fact_identities=required,
            terminal_fact_identities=terminal,
            evidence_refs=(f"alfworld_transition:{spec.action_id}:r{self._revision}",),
        )
        return HarnessActionResult(
            accepted, observation, done, won, self._revision, catalog,
            {"score": float(scores[0]), "previous_revision": old_revision, "action_type": spec.action_type},
            certificate,
        )

    def task_contract(self, task: HarnessTask) -> TaskContract:
        goal = re.sub(r"\s+", " ", task.goal.casefold())
        roles = _goal_roles(goal)
        target_object = roles.get("object", "")
        destination = roles.get("destination", "")
        light_source = roles.get("light_source", "")
        effects: list[SemanticPredicate] = []
        if task.task_type == "pick_heat_then_place_in_recep" or re.search(r"\b(?:heat|heated|hot)\b", goal):
            effects.append(SemanticPredicate("object.heated", {"object": target_object}))
        if task.task_type == "pick_clean_then_place_in_recep" or re.search(r"\b(?:clean|cleaned)\b", goal):
            effects.append(SemanticPredicate("object.cleaned", {"object": target_object}))
        if task.task_type == "pick_cool_then_place_in_recep" or re.search(r"\b(?:cool|cooled|cold)\b", goal):
            effects.append(SemanticPredicate("object.cooled", {"object": target_object}))
        count = _goal_cardinality(goal)
        if task.task_type == "pick_two_obj_and_place":
            count = 2
        if destination or task.task_type != "look_at_obj_in_light" and re.search(r"\b(?:put|place)\b", goal):
            effects.append(SemanticPredicate(
                "object.at_location", {"object": target_object, "location": destination},
                cardinality=count, distinct_by="object" if count > 1 else "",
            ))
        if task.task_type == "look_at_obj_in_light" or re.search(r"\b(?:examine|look at)\b", goal):
            effects.append(SemanticPredicate(
                "object.observed_with", {"object": target_object, "light": light_source},
            ))
        cardinality = ([{
            "predicate": "object.at_location", "role": "object",
            "count": count, "distinct_by": "object",
        }] if count > 1 else [])
        identity: list[IdentityConstraint] = []
        if task.task_type in {
            "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
        }:
            identity.append(IdentityConstraint("object", IdentityRelation.SAME_AS, "object", "task"))
        if count > 1:
            identity.append(IdentityConstraint("object_1", IdentityRelation.DISTINCT_FROM, "object_2", "task"))
        return TaskContract(effects, cardinality, identity, ContractSource.ADAPTER_DERIVED, 1.0, "alfworld_v3_goal")

    def contract_matcher(self) -> ContractMatcher:
        return AlfWorldContractMatcher()

    def validator_channel(self) -> AlfWorldValidatorChannel:
        return self._validator

    def compile_primitive(self, primitive: PrimitiveToolStep, bindings: dict[str, Any]) -> HarnessActionSpec:
        expected: dict[str, Any] = {}
        for role, expression in primitive.argument_mapping.items():
            if isinstance(expression, BindingExpression):
                value = expression.constant if expression.kind is BindingExprKind.CONSTANT else bindings.get(expression.source_role)
            elif isinstance(expression, dict) and "kind" in expression:
                expr = BindingExpression.from_dict(expression)
                value = expr.constant if expr.kind is BindingExprKind.CONSTANT else bindings.get(expr.source_role)
            else:
                value = expression
            expected[role] = normalize_entity(value)
        for spec in self.action_catalog():
            if spec.action_type == primitive.action_type and all(spec.arguments.get(key) == value for key, value in expected.items()):
                return spec
        raise KeyError(f"no current {primitive.action_type} affordance matches {expected}")

    def execute_primitive(self, primitive: PrimitiveToolStep, bindings: dict[str, Any]) -> HarnessActionResult:
        spec = self.compile_primitive(primitive, bindings)
        return self.execute_action(spec.action_id, spec.revision)

    def replay_tool(self, task: HarnessTask, tool: Any, case: dict[str, Any]) -> bool:
        """Replay the recorded source prefix and a parameterized Tool in a fresh episode."""
        expected_task = str((case.get("source_task") or {}).get("task_id", ""))
        if expected_task and expected_task != task.task_id:
            return False
        try:
            self.reset(task)
            for event in case.get("prefix", []):
                spec = self._match_action_event(event)
                result = self.execute_action(spec.action_id, spec.revision)
                if not result.accepted or (result.done and not result.won):
                    return False
            bindings = dict(case.get("bindings") or {})
            for raw in tool.artifact.get("steps", []):
                primitive = PrimitiveToolStep(
                    action_type=str(raw["action_type"]),
                    argument_mapping=dict(raw.get("argument_mapping", {})),
                )
                result = self.execute_primitive(primitive, bindings)
                if not result.accepted or (result.done and not result.won):
                    return False
            validation = self._validator.validate_atomic_effect({
                "effects": list(case.get("effects") or []),
                "bindings": bindings,
            })
            return validation.passed
        except AtomicSkillGraphError:
            # Harness/API/process failures are infrastructure failures and
            # must never be converted into negative replay evidence.
            raise
        except (KeyError, ValueError, RuntimeError):
            return False

    def _match_action_event(self, event: dict[str, Any]) -> HarnessActionSpec:
        action_type = str(event.get("action_type", ""))
        arguments = {key: normalize_entity(value) for key, value in dict(event.get("arguments", {})).items()}
        for spec in self.action_catalog():
            if spec.action_type == action_type and all(spec.arguments.get(key) == value for key, value in arguments.items()):
                return spec
        raise KeyError(f"source replay prefix action is not currently admissible: {action_type} {arguments}")

    def supports_constraint(self, kind: str, verifier_id: str = "") -> bool:
        return kind in {"argument_exists", "argument_concrete", "harness_affordance", "current_context"} or bool(verifier_id)


def _goal_roles(goal: str) -> dict[str, str]:
    goal = goal.strip().rstrip(".!?")
    roles: dict[str, str] = {}
    look = re.search(
        r"\b(?:look at|examine)\s+(?:a\s+|an\s+|some\s+|the\s+)?(.+?)\s+"
        r"(?:under|with)\s+(?:a\s+|an\s+|the\s+)?(.+?)\s*$",
        goal,
    )
    if look:
        roles["object"] = _normalize_goal_entity(look.group(1))
        roles["light_source"] = _normalize_goal_entity(look.group(2))
        return roles

    relation = re.search(r"\b(?:in|on|into|onto)\s+(?:a\s+|an\s+|the\s+)?([a-z][a-z0-9 ]*?)\s*$", goal)
    prefix = goal
    if relation:
        roles["destination"] = normalize_entity(relation.group(1))
        prefix = goal[:relation.start()]
    first_clause = re.split(r"\b(?:and then|then|and)\b", prefix, maxsplit=1)[0].strip()
    object_match = re.match(
        r"^(?:find|put|place|pick(?: up)?|heat|cool|clean)\s+"
        r"(?:(?:a|an|the|some|one|two|three|four|five|[2-9])\s+)?(.+?)$",
        first_clause,
    )
    if object_match:
        roles["object"] = _normalize_goal_entity(object_match.group(1))
    return roles


def _normalize_goal_entity(value: str) -> str:
    # Alternative human annotations sometimes encode the required state as an
    # adjective ("a hot egg", "a clean mug").  It is a target effect, not
    # part of the ALFWorld object family name.
    value = re.sub(
        r"^(?:(?:clean|cleaned|hot|heated|cool|cooled|cold|sliced)\s+)+",
        "",
        value.strip(),
    )
    return normalize_entity(value)


def _goal_cardinality(goal: str) -> int:
    words = {"two": 2, "three": 3, "four": 4, "five": 5}
    match = re.search(r"\b(?:put|place|pick|find)\s+(two|three|four|five|[2-9])\b", goal)
    if not match:
        return 1
    token = match.group(1)
    return words.get(token, int(token) if token.isdigit() else 1)
