"""ALFWorld native boundary: exact task reset, public catalog and feedback."""
from __future__ import annotations
import os
import copy
import re
import hashlib
import json
from pathlib import Path
from typing import Any
from ..core.errors import AtomicSkillGraphError, FailureLayer
from .action_catalog import HarnessActionCatalog
from .simple_protocol import HarnessTask, HarnessActionSpec, HarnessActionResult

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


class AlfWorldAdapter:
    profile_name = "alfworld_v3"

    def __init__(
        self, *, split: str = "eval_out_of_distribution", max_steps: int = 100,
        task_type: str | None = None, alfworld_data: str | None = None,
        public_discovery_version: str | None = None,
    ) -> None:
        self.split = split
        self.max_steps = max_steps
        self.task_type = task_type
        from .public_discovery import VERSION
        if public_discovery_version not in {None, VERSION, 'alfworld.public-discovery.v1'}:
            raise ValueError('unsupported harness.public_discovery_version')
        self.public_discovery_version = public_discovery_version
        self._public_discovery_frame = None
        self._public_discovery_signature = None
        self._public_discovery_accepted = True
        self.alfworld_data = alfworld_data or os.environ.get("ALFWORLD_DATA", str(Path.home() / ".cache" / "alfworld"))
        self._env: Any = None
        self._tw_env: Any = None
        self._task_index = 0
        self._revision = 0
        self._catalog = HarnessActionCatalog(parse_alfworld_action)
        self._current_task: HarnessTask | None = None
        self._observation = ""
        self._done = self._won = False
        self._runtime_accepted_prefix: list[dict[str, Any]] = []
        # Only real discovery resets populate this immutable index prefix.
        # It contains file locations, never world facts or model-supplied values.
        self._discovered_files: tuple[str, ...] = ()
        self._discovery_identity = ""
        self._backend_identity = ""
        self._exact_file: str | None = None


    def _configuration_identity(self) -> str:
        return json.dumps([self.split, self._build_config()], sort_keys=True)


    def _close_backend(self) -> None:
        env, self._env = self._env, None
        self._tw_env = None
        self._exact_file = None
        self._backend_identity = ""
        if env is not None:
            env.close()


    def _check_discovery_identity(self) -> str:
        identity = self._configuration_identity()
        if identity != self._discovery_identity:
            self._discovered_files = ()
            self._discovery_identity = identity
        return identity


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
        identity = self._check_discovery_identity()
        self._close_backend()
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
            self._backend_identity = identity
        except AtomicSkillGraphError:
            raise
        except Exception as exc:
            self._close_backend()
            raise AtomicSkillGraphError(
                "infrastructure_failure", f"failed to initialize ALFWorld: {exc}",
                layer=FailureLayer.INFRASTRUCTURE,
            ) from exc
        files = getattr(self._tw_env, "gamefiles", None) or getattr(self._tw_env, "game_files", None)
        self._task_index = 0
        return len(files) if files is not None else 0


    def _prepare_exact_backend(self, game_file: str, identity: str) -> bool:
        import alfworld.agents.environment as alf_env

        env_class = alf_env.get_environment("AlfredTWEnv")
        if not callable(getattr(env_class, "collect_game_files", None)):
            return False  # Older dependencies retain the legal discovery path.
        if self._env is not None and self._exact_file == game_file and self._backend_identity == identity:
            return True
        self._close_backend()

        class _ExactFileEnv(env_class):
            def collect_game_files(self, verbose=False):
                self.game_files = [game_file]
                self.num_games = 1

        try:
            self._tw_env = _ExactFileEnv(self._build_config(), train_eval=self.split)
            self._env = self._tw_env.init_env(batch_size=1)
            self._exact_file = game_file
            self._backend_identity = identity
        except Exception as exc:
            self._close_backend()
            raise AtomicSkillGraphError(
                "infrastructure_failure", f"failed to initialize exact ALFWorld game: {exc}",
                layer=FailureLayer.INFRASTRUCTURE,
            ) from exc
        return True


    def _raw_reset(self) -> tuple[HarnessTask, str, list[str]]:
        if self._env is None:
            self.initialize()
        try:
            observations, info = self._env.reset()
        except Exception as exc:
            self._close_backend()
            raise AtomicSkillGraphError(
                "infrastructure_failure", f"ALFWorld reset failed: {exc}",
                layer=FailureLayer.INFRASTRUCTURE,
            ) from exc
        observation = str(observations[0])
        admissible = list(info.get("admissible_commands", [[]])[0])
        game_file = str((info.get("extra.gamefile") or [""])[0])
        if self._exact_file is None:
            normalized = game_file.replace("\\", "/")
            if self._task_index == len(self._discovered_files):
                self._discovered_files += (normalized,)
            elif self._task_index < len(self._discovered_files) and self._discovered_files[self._task_index] != normalized:
                self._close_backend()
                raise AtomicSkillGraphError(
                    "infrastructure_failure", "ALFWorld discovery index changed",
                    layer=FailureLayer.INFRASTRUCTURE,
                )
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
            },
            metadata={"task_signature": signature},
        )
        self._task_index += 1
        return task, observation, admissible


    def load_tasks(self, *, limit: int = 0, task_type: str | None = None) -> list[HarnessTask]:
        if (self._env is None or self._exact_file is not None
                or self._backend_identity != self._configuration_identity()):
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
        expected = str(task.context.get("game_file", "")).replace("\\", "/")
        identity = self._check_discovery_identity()
        if index < 0:
            raise AtomicSkillGraphError(
                "infrastructure_failure", "ALFWorld task index must be nonnegative",
                layer=FailureLayer.INFRASTRUCTURE,
            )
        # A caller's file/index pair is not its own authority. Discover it once
        # through the original deterministic ordering if not already observed.
        if expected and index >= len(self._discovered_files):
            total = self.initialize()
            if index >= total:
                raise AtomicSkillGraphError(
                    "infrastructure_failure", "ALFWorld task index outside discovery manifest",
                    layer=FailureLayer.INFRASTRUCTURE,
                )
            for _ in range(index + 1):
                self._raw_reset()
        if expected and expected != self._discovered_files[index]:
            raise AtomicSkillGraphError(
                "infrastructure_failure", "ALFWorld deterministic task mapping changed: file/index mismatch",
                layer=FailureLayer.INFRASTRUCTURE,
            )
        deterministic = not self._build_config()["env"].get("domain_randomization", False)
        if expected and deterministic and self._prepare_exact_backend(expected, identity):
            self._task_index = index
        else:
            self.initialize()
            for _ in range(index):
                self._raw_reset()
        actual, observation, admissible = self._raw_reset()
        observed = str(actual.context.get("game_file", "")).replace("\\", "/")
        signature = str(task.metadata.get("task_signature") or "")
        if (expected and expected != observed) or (expected and (
            actual.task_id != task.task_id or actual.goal != task.goal
            or (signature and signature != actual.metadata["task_signature"])
        )):
            self._close_backend()
            raise AtomicSkillGraphError(
                "infrastructure_failure",
                f"ALFWorld deterministic task mapping changed: expected={expected}, actual={observed}",
                layer=FailureLayer.INFRASTRUCTURE,
            )
        self._current_task, self._observation = task, observation
        self._revision = 0
        self._runtime_accepted_prefix = []
        self._done = self._won = False
        catalog = self._replace_action_catalog(admissible, self._revision)
        self._refresh_public_discovery(None, True)
        return HarnessActionResult(True, observation, False, False, self._revision, catalog, {"reset": True})


    def action_catalog(self) -> list[HarnessActionSpec]:
        return self._catalog.items()


    def _refresh_public_discovery(self, signature, accepted):
        if self.public_discovery_version is None:
            return
        from .public_discovery import project_discovery
        self._public_discovery_signature = copy.deepcopy(signature)
        self._public_discovery_accepted = bool(accepted)
        self._public_discovery_frame = project_discovery(observation=self._observation,
            action_signature=signature, accepted=accepted, revision=self._revision,
            catalog=self.action_catalog(), episode_id=self._current_task.task_id,
            version=self.public_discovery_version)


    def public_discovery_frame(self):
        return self._public_discovery_frame


    def _replace_action_catalog(
        self, admissible: list[Any], revision: int,
    ) -> list[HarnessActionSpec]:
        # TextWorld advertises meta commands such as ``help`` alongside task
        # actions.  The formal runtime can execute only commands represented
        # by the typed v3 boundary, so do not publish unparsed UNKNOWN entries
        # as selectable native-tool arguments.
        supported = [
            raw for raw in admissible
            if parse_alfworld_action(raw)[0] != "UNKNOWN"
        ]
        return self._catalog.replace(supported, revision)


    def execute_action(self, action_id: str, revision: int) -> HarnessActionResult:
        if self._done or self._won:
            raise AtomicSkillGraphError(
                "harness_terminal_latched",
                "benchmark terminal is latched; no further env.step is allowed",
                layer=FailureLayer.RUNTIME_AGENT,
            )
        spec = self._catalog.get(action_id, revision)
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
        accepted = "nothing happens" not in observation.casefold()
        if accepted:
            self._runtime_accepted_prefix.append({
                "action_type": spec.action_type, "arguments": copy.deepcopy(spec.arguments),
            })
        admissible = list(infos.get("admissible_commands", [[]])[0])
        old_revision = self._revision
        self._revision += 1
        catalog = self._replace_action_catalog(admissible, self._revision)
        metadata = {
            "score": float(scores[0]),
            "previous_revision": old_revision,
            "action_type": spec.action_type,
        }
        self._observation, self._done, self._won = observation, done, won
        self._refresh_public_discovery({'action_type': spec.action_type, 'arguments': spec.arguments}, accepted)
        if self._public_discovery_frame is not None:
            metadata['public_discovery_frame'] = self._public_discovery_frame.to_dict()
        return HarnessActionResult(
            accepted, observation, done, won, self._revision, catalog,
            metadata,
        )


    def primitive_action_schema(self) -> list[dict[str, Any]]:
        """Single parser-derived source of truth for Builder/Runtime/Static."""

        schema: list[dict[str, Any]] = []
        seen: set[str] = set()
        for action_type, _pattern, roles in _ACTION_PATTERNS:
            if action_type in seen:
                continue
            seen.add(action_type)
            schema.append({
                "action_type": action_type,
                "argument_roles": list(roles),
                "public_semantics": {
                    "HEAT": "Environment abstract heating action on an object at the named heat source; not a simulation of appliance assembly.",
                    "COOL": "Environment abstract cooling action at the named cooling source.",
                    "CLEAN": "Environment abstract cleaning action at the named cleaning source.",
                    "TAKE": "Pick up the specified currently accessible object from its named source.",
                    "PUT": "Place the held object into or onto the named destination.",
                    "MOVE": "Place the held object into or onto the named destination.",
                    "GO_TO": "Navigate to the named currently available location.",
                    "OPEN": "Open the named accessible container.",
                    "CLOSE": "Close the named accessible container.",
                    "EXAMINE": "Inspect the named accessible entity.",
                    "USE": "Use the named available entity according to the public action catalog.",
                }.get(action_type, "Execute this primitive only when its exact arguments appear in the current public action catalog."),
                "applicability": "The current public action catalog is authoritative; description alone grants no action or effect.",
            })
        schema.extend([
            {"action_type": "LOOK", "argument_roles": []},
            {"action_type": "INVENTORY", "argument_roles": []},
        ])
        return schema


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
