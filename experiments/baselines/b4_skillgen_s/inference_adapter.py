"""SkillGen sampling and frozen retrieval-guided ALFWorld inference adapter."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Protocol

from experiments.baselines.common.manifest import ManifestTask
from experiments.baselines.common.provider_gate import CampaignProviderGate
from experiments.baselines.common.usage import RoleUsage

from .progress_adapter import SkillGenLabel, SubgoalProgressTracker, task_key_from_gamefile


_SPLIT_MAP = {
    "train": "train",
    "valid_seen": "eval_in_distribution",
    "valid_unseen": "eval_out_of_distribution",
}


class ProviderExecutionError(RuntimeError):
    """A provider call failed outside normal agent behavior."""

    def __init__(
        self, message: str, *, failure_kind: str, event: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.failure_kind = failure_kind
        self.event = dict(event)


class EpisodeInfrastructureError(RuntimeError):
    failure_kind = "infrastructure_failure"


@dataclass(frozen=True)
class ProviderResult:
    text: str
    usage: RoleUsage
    event: dict[str, Any]


class ChatProvider(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        task_id: str,
        stage: str,
        call_index: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        do_sample: bool,
    ) -> ProviderResult: ...


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _failure_code(exc: BaseException) -> tuple[str, bool]:
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    try:
        status_code = int(status) if status is not None else None
    except (TypeError, ValueError):
        status_code = None
    name = type(exc).__name__.casefold()
    text = str(exc).casefold()
    if status_code == 429:
        return "http_429", True
    if status_code is not None and status_code >= 500:
        return f"http_{status_code}", True
    if status_code is not None and status_code >= 400:
        return f"http_{status_code}", False
    if isinstance(exc, (TimeoutError, ConnectionError)) or any(
        token in name or token in text
        for token in ("timeout", "connection", "connecterror", "ratelimit")
    ):
        return "transport", True
    return "provider_protocol", False


def _usage_from_response(response: Any) -> RoleUsage:
    raw = getattr(response, "usage", None)
    if raw is None:
        raise RuntimeError("provider response has no token usage")
    prompt = int(getattr(raw, "prompt_tokens", -1))
    completion = int(getattr(raw, "completion_tokens", -1))
    total = int(getattr(raw, "total_tokens", prompt + completion))
    details = getattr(raw, "completion_tokens_details", None)
    reasoning_raw = getattr(details, "reasoning_tokens", None) if details is not None else None
    reasoning = int(reasoning_raw or 0)
    if prompt < 0 or completion <= 0 or total != prompt + completion:
        raise RuntimeError("provider response has invalid token usage")
    if reasoning < 0 or reasoning > completion:
        raise RuntimeError("provider response has invalid reasoning-token usage")
    return RoleUsage(
        calls=1,
        prompt_tokens=prompt,
        completion_tokens=completion,
        reasoning_tokens=reasoning,
    )


class AuditedDeepSeekClient:
    """OpenAI-compatible transport with frozen retry and evidence semantics."""

    def __init__(
        self,
        *,
        model: dict[str, str],
        run_id: str,
        run_seed: int,
        phase: str,
        retry_delays: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0),
        jitter_ratio: float = 0.10,
        timeout_seconds: float = 300.0,
        campaign: dict[str, Any] | None = None,
        sdk_client: Any | None = None,
    ) -> None:
        if str(model.get("provider")) != "openai_compatible":
            raise ValueError("SkillGen requires the OpenAI-compatible provider")
        if str(model.get("model")) != "deepseek-v4-flash":
            raise ValueError("SkillGen base model must be deepseek-v4-flash")
        if str(model.get("reasoning_effort")) != "high":
            raise ValueError("SkillGen reasoning_effort must be high")
        if len(retry_delays) != 4 or any(delay < 0 for delay in retry_delays):
            raise ValueError("provider retry schedule must contain four non-negative delays")
        if not 0.0 <= jitter_ratio <= 0.10:
            raise ValueError("provider retry jitter_ratio must be within 0..0.10")
        self.model = dict(model)
        self.run_id = str(run_id)
        self.run_seed = int(run_seed)
        self.phase = str(phase)
        self.retry_delays = tuple(float(value) for value in retry_delays)
        self.jitter_ratio = float(jitter_ratio)
        self.timeout_seconds = float(timeout_seconds)
        self._client = sdk_client or self._make_client()
        self._gate = self._make_gate(campaign)

    def _make_client(self) -> Any:
        from openai import OpenAI

        env_name = str(self.model.get("api_key_env", "MODEL_API_KEY"))
        api_key = os.environ.get(env_name, "").strip()
        if not api_key:
            raise RuntimeError(f"{env_name} is missing from the worker environment")
        return OpenAI(
            api_key=api_key,
            base_url=str(self.model["base_url"]).rstrip("/"),
            timeout=self.timeout_seconds,
            max_retries=0,
        )

    @staticmethod
    def _make_gate(campaign: dict[str, Any] | None) -> CampaignProviderGate | None:
        if campaign is None:
            return None
        return CampaignProviderGate(
            gate_dir=Path(str(campaign["provider_gate_dir"])),
            campaign_id=str(campaign["campaign_id"]),
            max_inflight=int(campaign["campaign_provider_max_inflight"]),
        )

    def _delay(self, call_id: str, failure_index: int) -> float:
        base = self.retry_delays[failure_index]
        if base == 0.0 or self.jitter_ratio == 0.0:
            return base
        digest = hashlib.sha256(
            f"{self.run_id}\0{call_id}\0{failure_index + 1}".encode("utf-8")
        ).digest()
        unit = int.from_bytes(digest[:8], "big") / float((1 << 64) - 1)
        return base * (1.0 + self.jitter_ratio * ((2.0 * unit) - 1.0))

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        task_id: str,
        stage: str,
        call_index: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        do_sample: bool,
    ) -> ProviderResult:
        request_digest = hashlib.sha256(_canonical_json(messages).encode("utf-8")).hexdigest()
        call_id = "skillgen_" + hashlib.sha256(
            f"{self.run_id}\0{task_id}\0{stage}\0{call_index}\0{request_digest}".encode("utf-8")
        ).hexdigest()[:24]
        started = time.perf_counter()
        failure_codes: list[str] = []
        retry_backoff_ms = 0
        queue_wait_ms = 0
        slot_ids: list[int] = []
        request_ids: list[str] = []
        base_event = {
            "schema_version": 1,
            "event": "provider_call",
            "method": "b4_skillgen_s",
            "phase": self.phase,
            "run_id": self.run_id,
            "run_seed": self.run_seed,
            "episode_task_id": str(task_id),
            "call_id": call_id,
            "role": "target",
            "stage": str(stage),
            "model": self.model["model"],
            "reasoning_effort": self.model["reasoning_effort"],
            "messages_sha256": request_digest,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "sample_seed": int(seed),
            "do_sample": bool(do_sample),
            "requested_retry_limit": 5,
        }
        for attempt in range(1, 6):
            try:
                gate = nullcontext(None)
                if self._gate is not None:
                    gate = self._gate.acquire(
                        run_id=self.run_id,
                        seed=self.run_seed,
                        role="target",
                        stage=stage,
                        logical_call_id=call_id,
                    )
                with gate as lease:
                    if lease is not None:
                        queue_wait_ms += int(lease.queue_wait_ms)
                        slot_ids.append(int(lease.slot_id))
                    response = self._client.chat.completions.create(
                        model=self.model["model"],
                        messages=messages,
                        max_tokens=int(max_tokens),
                        temperature=float(temperature),
                        top_p=float(top_p),
                        seed=int(seed),
                        reasoning_effort=self.model["reasoning_effort"],
                    )
                request_id = str(getattr(response, "_request_id", "") or "").strip()
                if request_id:
                    request_ids.append(request_id)
                choices = list(getattr(response, "choices", None) or [])
                if not choices:
                    raise RuntimeError("provider response contains no choices")
                text = str(getattr(choices[0].message, "content", "") or "").strip()
                if not text:
                    raise RuntimeError("provider response content is empty")
                usage = _usage_from_response(response)
                event = {
                    **base_event,
                    "status": "succeeded",
                    "application_attempts": attempt,
                    "sdk_boundary_attempts": attempt,
                    "retry_backoff_ms": retry_backoff_ms,
                    "failure_codes": failure_codes,
                    "recovered": bool(failure_codes),
                    "provider_queue_wait_ms": queue_wait_ms,
                    "provider_slot_ids": slot_ids,
                    "provider_request_ids": request_ids,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "reasoning_tokens": usage.reasoning_tokens,
                    "reasoning_tokens_status": (
                        "reported" if usage.reasoning_tokens else "unavailable"
                    ),
                    "total_tokens": usage.prompt_tokens + usage.completion_tokens,
                    "response_text": text,
                    "response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "latency_ms": max(0, int((time.perf_counter() - started) * 1000)),
                }
                return ProviderResult(text=text, usage=usage, event=event)
            except Exception as exc:
                code, transient = _failure_code(exc)
                failure_codes.append(code)
                request_id = str(getattr(exc, "request_id", "") or "").strip()
                if request_id:
                    request_ids.append(request_id)
                if transient and attempt < 5:
                    delay = self._delay(call_id, attempt - 1)
                    retry_backoff_ms += int(delay * 1000)
                    time.sleep(delay)
                    continue
                event = {
                    **base_event,
                    "status": "failed",
                    "application_attempts": attempt,
                    "sdk_boundary_attempts": attempt,
                    "retry_backoff_ms": retry_backoff_ms,
                    "failure_codes": failure_codes,
                    "recovered": False,
                    "provider_queue_wait_ms": queue_wait_ms,
                    "provider_slot_ids": slot_ids,
                    "provider_request_ids": request_ids,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "reasoning_tokens": None,
                    "reasoning_tokens_status": "unavailable",
                    "total_tokens": 0,
                    "error_type": type(exc).__name__,
                    "error_code": code,
                    "latency_ms": max(0, int((time.perf_counter() - started) * 1000)),
                }
                kind = "infrastructure_failure" if transient else "protocol_failure"
                raise ProviderExecutionError(
                    f"SkillGen provider call {call_id} failed as {code}",
                    failure_kind=kind,
                    event=event,
                ) from exc
        raise AssertionError("unreachable provider retry state")


class PinnedPromptCore:
    """Load pure prompt/parser functions directly from the pinned source AST."""

    def __init__(self, external_root: str | Path) -> None:
        self.root = Path(external_root).resolve(strict=True)
        self.sampling_make_prompt = self._functions(
            self.root / "sampling.py", {"make_prompt"},
        )["make_prompt"]
        action_functions = self._functions(
            self.root / "utils.py",
            {"remove_parentheses_content", "remove_number_prefix", "extract_action"},
        )
        self.extract_action = action_functions["extract_action"]
        self.skill_template = self._string_assignment(
            self.root / "prompt" / "prompts.py", "skillgen_demoac_skill",
        )
        self.prompt_dict = json.loads(
            (self.root / "prompt" / "task" / "alfworld_base.json").read_text(
                encoding="utf-8"
            )
        )

    @staticmethod
    def _functions(path: Path, names: set[str]) -> dict[str, Callable[..., Any]]:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        selected = [
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in names
        ]
        found = {node.name for node in selected}
        if found != names:
            raise RuntimeError(f"pinned SkillGen source {path} lacks {sorted(names - found)}")
        namespace: dict[str, Any] = {"re": __import__("re")}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
        return {name: namespace[name] for name in names}

    @staticmethod
    def _string_assignment(path: Path, name: str) -> str:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            ):
                value = ast.literal_eval(node.value)
                if isinstance(value, str):
                    return value
        raise RuntimeError(f"pinned SkillGen source {path} lacks string {name}")


class UpstreamRetrievalCore:
    def __init__(self, external_root: str | Path) -> None:
        path = Path(external_root).resolve(strict=True) / "prompt" / "func_prompting.py"
        spec = importlib.util.spec_from_file_location("_asg_skillgen_func_prompting", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load pinned SkillGen retrieval core: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.retrieve = module.retrieve_skills_by_action_embedding


class ExactGamefileSkillGenEnv:
    """One pinned-ALFWorld-semantics episode on exactly one manifest gamefile."""

    def __init__(
        self,
        *,
        task: ManifestTask,
        alfworld_data: str | Path,
        random_seed: int,
    ) -> None:
        self.task = task
        self.data_root = Path(alfworld_data).expanduser().resolve(strict=True)
        self.random_seed = int(random_seed)
        self.expected_gamefile = (self.data_root / task.gamefile_rel).resolve(strict=True)
        try:
            self.expected_gamefile.relative_to(self.data_root)
        except ValueError as exc:
            raise ValueError("manifest gamefile resolves outside ALFWORLD_DATA") from exc
        self._wrapper: Any = None
        self._env: Any = None
        self.valid_actions: list[str] = []
        self.env_ob = ""
        self.done = False
        self.won = False
        self.actual_gamefile = ""

    def _config(self) -> dict[str, Any]:
        data = str(self.data_root)
        return {
            "env": {
                "type": "AlfredTWEnv",
                "regen_game_files": False,
                "domain_randomization": False,
                "task_types": [1, 2, 3, 4, 5, 6],
                "expert_type": "handcoded",
                "goal_desc_human_anns_prob": 0.0,
                "data_path": data,
                "logic": {
                    "domain": str(self.data_root / "logic" / "alfred.pddl"),
                    "grammar": str(self.data_root / "logic" / "alfred.twl2"),
                },
                "json_game": {"data_path": str(self.data_root / "json_2.1.1")},
            },
            "dataset": {
                "data_path": str(self.data_root / "json_2.1.1"),
                "eval_id_data_path": str(self.data_root / "json_2.1.1" / "valid_seen"),
                "eval_ood_data_path": str(self.data_root / "json_2.1.1" / "valid_unseen"),
                "num_train_games": -1,
                "num_eval_games": -1,
            },
            "general": {
                "training_method": "dagger", "random_seed": self.random_seed,
                "use_cuda": False,
            },
            "dagger": {"training": {"batch_size": 1, "max_nb_steps_per_episode": 100}},
            "controller": {"type": "oracle", "debug": False},
        }

    def reset(self) -> tuple[str, str, str]:
        try:
            from alfworld.agents.environment import get_environment

            split = _SPLIT_MAP[self.task.source_split]
            env_class = get_environment("AlfredTWEnv")
            self._wrapper = env_class(self._config(), train_eval=split)
            if hasattr(self._wrapper, "game_files"):
                self._wrapper.game_files = [str(self.expected_gamefile)]
            elif hasattr(self._wrapper, "gamefiles"):
                self._wrapper.gamefiles = [str(self.expected_gamefile)]
            else:
                raise RuntimeError("ALFWorld wrapper exposes no gamefile collection")
            if hasattr(self._wrapper, "num_games"):
                self._wrapper.num_games = 1
            self._env = self._wrapper.init_env(batch_size=1)
            observations, info = self._env.reset()
        except Exception as exc:
            raise EpisodeInfrastructureError(f"ALFWorld initialization/reset failed: {exc}") from exc
        observed = Path(str((info.get("extra.gamefile") or [""])[0])).resolve(strict=True)
        if observed != self.expected_gamefile:
            raise EpisodeInfrastructureError(
                f"exact gamefile mismatch: expected={self.expected_gamefile}, observed={observed}"
            )
        self.actual_gamefile = str(observed)
        self.valid_actions = list(info.get("admissible_commands", [[]])[0])
        raw = str(observations[0])
        stripped = "\n".join(raw.split("\n\n")[1:])
        lines = stripped.split("\n")
        if len(lines) < 2 or "Your task is to:" not in lines[1]:
            raise EpisodeInfrastructureError("ALFWorld reset observation lacks SkillGen goal layout")
        self.env_ob = lines[0]
        goal = lines[1].split("Your task is to:", 1)[1].replace(".", "").strip()
        return stripped, goal, task_key_from_gamefile(observed)

    def get_action_space(self) -> list[str]:
        if "look" not in self.valid_actions:
            self.valid_actions.append("look")
        if "check valid actions" not in self.valid_actions:
            self.valid_actions.append("check valid actions")
        return self.valid_actions

    def step(self, raw_action: str) -> dict[str, Any]:
        action = str(raw_action)
        if action.endswith("."):
            action = action[:-1]
        env_action_executed = action != "check valid actions"
        info: dict[str, Any] | None = None
        score = 0.0
        try:
            if action == "look":
                observation, scores, dones, info = self._env.step([action])
                score = float(scores[0])
                done = bool(dones[0])
                observation = [self.env_ob]
            elif action == "check valid actions":
                observation = [
                    "Choose an action from these valid actions: "
                    + ", ".join(self.valid_actions)
                ]
                done = self.done
            else:
                observation, scores, dones, info = self._env.step([action])
                score = float(scores[0])
                done = bool(dones[0])
        except Exception as exc:
            raise EpisodeInfrastructureError(f"ALFWorld step failed: {exc}") from exc
        text = str(observation[0])
        if ("go to" in action or "open" in action) and "Nothing happens" not in text:
            self.env_ob = text
        if info is not None:
            self.valid_actions = list(info.get("admissible_commands", [[]])[0])
            won_values = info.get("won", [False])
            self.won = self.won or bool(won_values[0] if won_values else False)
        if text.startswith("You arrive at loc "):
            text = text[text.find(". ") + 2:]
        self.done = bool(done)
        return {
            "action": action,
            "observation": text,
            "done": self.done,
            "won": self.won,
            "score": score,
            "environment_action_executed": env_action_executed,
        }

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if callable(close):
            close()


def _sum_usage(events: list[dict[str, Any]]) -> RoleUsage:
    usage = RoleUsage()
    for event in events:
        if event.get("status") != "succeeded":
            continue
        usage.calls += 1
        usage.prompt_tokens += int(event["prompt_tokens"])
        usage.completion_tokens += int(event["completion_tokens"])
        usage.reasoning_tokens += int(event.get("reasoning_tokens") or 0)
    return usage


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid frozen JSONL {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"frozen JSONL row is not a mapping: {path}:{line_number}")
        rows.append(row)
    return rows


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("SkillGen embedding dimensions do not match")
    dot = sum(a * b for a, b in zip(left, right))
    nl = math.sqrt(sum(value * value for value in left))
    nr = math.sqrt(sum(value * value for value in right))
    return dot / (nl * nr) if nl and nr else 0.0


def _retrieved_domains(
    *,
    goal: str,
    category: str,
    metadata: list[dict[str, Any]],
    encoder: Any,
    top_k: int = 3,
) -> list[str]:
    query = f"Goal: {goal} Domain: {category}."
    embedding = [
        float(value)
        for value in encoder.encode([query], convert_to_numpy=True)[0]
    ]
    ranked = sorted(
        metadata,
        key=lambda row: (
            -_cosine(embedding, [float(value) for value in row["embedding"]]),
            str(row["task_id"]),
        ),
    )
    result: list[str] = []
    for row in ranked:
        domain = str(row["domain"])
        if domain == category or domain in result:
            continue
        result.append(domain)
        if len(result) == top_k:
            break
    return result


def _format_skills(result: dict[str, Any]) -> str:
    prompt_parts: list[str] = []
    for index, skill in enumerate(result.get("skills", []), 1):
        node = str(skill["node"])
        parts = [f"### Skill {index}: Centered on action '{node}'"]
        if skill.get("backward"):
            parts.append("  - Common precursors to this action:")
            for source, target, _score in skill["backward"]:
                parts.append(
                    f"    \u2022 Agents often perform '{source}' before '{target}'."
                )
        if skill.get("forward"):
            parts.append("  - Typical next steps after this action:")
            for source, target, _score in skill["forward"]:
                parts.append(
                    f"    \u2022 After '{source}', agents usually continue with '{target}'."
                )
        prompt_parts.append("\n".join(parts))
    return "\n\n".join(prompt_parts).replace(
        "INIT_STATE", "the beginning of the task"
    )


def make_frozen_inference_prompt(
    *,
    prompt_core: PinnedPromptCore,
    retrieval_core: UpstreamRetrievalCore | Any,
    frozen_root: str | Path,
    category: str,
    goal: str,
    history: list[list[str]],
    encoder: Any,
    top_s: int = 1,
    top_ac: int = 1,
    hist_size: int = 20,
    retrieved_domains: list[str] | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if top_s != 1 or top_ac != 1 or hist_size != 20:
        raise ValueError("SkillGen heldout retrieval settings are frozen at top_s=top_ac=1,hist=20")
    root = Path(frozen_root)
    domain_embedding_calls = 0
    if retrieved_domains is None:
        metadata = _load_jsonl(root / "train_metadata_embeddings.jsonl")
        related = _retrieved_domains(
            goal=goal, category=category, metadata=metadata, encoder=encoder,
        )
        domain_embedding_calls = 1
    else:
        related = [str(value) for value in retrieved_domains]
    selected_domain = ""
    graph_data: list[dict[str, Any]] = []
    for candidate in [category, *related]:
        path = root / "step_skill_embeddings" / f"{candidate}.jsonl"
        if path.is_file():
            candidate_rows = _load_jsonl(path)
            if candidate_rows:
                selected_domain = candidate
                graph_data = candidate_rows
                break
    query_action = "INIT_STATE" if len(history) == 1 else str(history[-2][1])
    if graph_data:
        retrieval = retrieval_core.retrieve(
            graph_data, query_action, encoder,
            top_k_actions=top_ac, top_k_skills=top_s,
        )
        skills = _format_skills(retrieval)
    else:
        retrieval = {"matched_nodes": [], "skills": []}
        skills = ""
    golden_path = root / "golden_segments" / f"{category}.txt"
    golden = golden_path.read_text(encoding="utf-8") if golden_path.is_file() else ""
    local_history = history[-hist_size * 2:]
    hist_info = "\n".join(f"{item[0]}: {item[1]}" for item in local_history)
    prompt = prompt_core.skill_template.format(
        instructions=prompt_core.prompt_dict["simple_instruction"],
        goal=goal,
        skills=skills,
        trajectories=golden,
        history=hist_info,
    )
    messages = [
        {"role": "system", "content": prompt_core.prompt_dict["system_msg"]},
        {"role": "user", "content": prompt},
    ]
    return messages, {
        "retrieved_domains": related,
        "selected_skill_domain": selected_domain,
        "retrieved_skill_count": sum(
            len(item.get("forward", [])) + len(item.get("backward", []))
            for item in retrieval.get("skills", [])
        ),
        "matched_nodes": list(retrieval.get("matched_nodes", [])),
        "golden_segment_used": bool(golden),
        # One query embeds the visible goal.  The pinned action-retrieval core
        # performs one additional encoder call only when a non-empty graph is
        # available.  Persist the real invocation count instead of inferring
        # it later from the number of retrieval events.
        "embedding_calls": domain_embedding_calls + int(bool(graph_data)),
    }


def run_skillgen_episode(
    *,
    task: ManifestTask,
    external_root: str | Path,
    alfworld_data: str | Path,
    model: dict[str, str],
    run_id: str,
    run_seed: int,
    phase: str,
    max_steps: int,
    max_length: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    sample_seed: int,
    sample_idx: int | None = None,
    label: SkillGenLabel | None = None,
    frozen_root: str | Path | None = None,
    encoder: Any | None = None,
    prompt_core: PinnedPromptCore | None = None,
    retrieval_core: UpstreamRetrievalCore | Any | None = None,
    provider: ChatProvider | None = None,
    env_factory: Callable[..., Any] = ExactGamefileSkillGenEnv,
    retry_delays: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0),
    jitter_ratio: float = 0.10,
    campaign: dict[str, Any] | None = None,
    max_environment_actions: int = 100,
) -> dict[str, Any]:
    """Run one method-faithful sampling or frozen inference episode."""

    sampling = label is not None
    if sampling:
        if (max_steps, max_length, temperature, top_p, do_sample) != (10, 64, 1.0, 0.95, True):
            raise ValueError("SkillGen sampling settings differ from the frozen protocol")
        if frozen_root is not None:
            raise ValueError("Train sampling must not receive a frozen Test artifact")
    else:
        if frozen_root is None:
            raise ValueError("SkillGen inference requires a frozen retrieval artifact")
        if (max_steps, max_length, temperature) != (20, 64, 0.0):
            raise ValueError("SkillGen inference settings differ from the frozen protocol")
    if max_environment_actions != 100:
        raise ValueError("outer ALFWorld action ceiling must remain 100")

    started = time.perf_counter()
    prompt_core = prompt_core or PinnedPromptCore(external_root)
    provider = provider or AuditedDeepSeekClient(
        model=model,
        run_id=run_id,
        run_seed=run_seed,
        phase=phase,
        retry_delays=retry_delays,
        jitter_ratio=jitter_ratio,
        campaign=campaign,
    )
    env = env_factory(task=task, alfworld_data=alfworld_data, random_seed=sample_seed)
    provider_calls: list[dict[str, Any]] = []
    action_events: list[dict[str, Any]] = []
    progress_events: list[list[float | int]] = []
    grounding: list[int] = []
    retrieval_events: list[dict[str, Any]] = []
    retrieved_domains: list[str] | None = None
    environment_actions = 0
    environment_done = False
    tracker = SubgoalProgressTracker(label) if label is not None else None
    logical_id = (
        f"{task.task_id}::sample_{sample_idx}" if sample_idx is not None else task.task_id
    )
    try:
        initial_observation, goal, task_key = env.reset()
        if label is not None and task_key != label.task_key:
            raise EpisodeInfrastructureError(
                f"Train label/gamefile mismatch: label={label.task_key}, game={task_key}"
            )
        trajectory: list[list[str]] = [["OBSERVATION", initial_observation]]
        progress_rate = 0.0
        termination_reason = "internal_step_limit"
        for step in range(max_steps):
            if sampling:
                args = SimpleNamespace(
                    dataset_name="alfworld", prompt_mode="1-shot", hist_size=20,
                )
                messages = prompt_core.sampling_make_prompt(
                    step=step,
                    args=args,
                    category=task_key,
                    goal=goal,
                    history=trajectory,
                    prompt_dict=prompt_core.prompt_dict,
                )
                retrieval = None
                stage = "sampling"
            else:
                if encoder is None:
                    from .extraction_adapter import load_sentence_encoder

                    encoder = load_sentence_encoder("sentence-transformers/all-MiniLM-L6-v2")
                retrieval_core = retrieval_core or UpstreamRetrievalCore(external_root)
                messages, retrieval = make_frozen_inference_prompt(
                    prompt_core=prompt_core,
                    retrieval_core=retrieval_core,
                    frozen_root=Path(frozen_root),
                    category=task.task_type,
                    goal=goal,
                    history=trajectory,
                    encoder=encoder,
                    top_s=1,
                    top_ac=1,
                    hist_size=20,
                    retrieved_domains=retrieved_domains,
                )
                retrieved_domains = [
                    str(value) for value in retrieval["retrieved_domains"]
                ]
                retrieval_events.append({"step": step, **retrieval})
                stage = "heldout_inference"
            result = provider.complete(
                messages,
                task_id=logical_id,
                stage=stage,
                call_index=step,
                max_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                seed=sample_seed,
                do_sample=do_sample,
            )
            provider_calls.append(dict(result.event))
            action = str(prompt_core.extract_action(result.text))
            valid_before = list(env.get_action_space())
            if action in valid_before:
                grounding.append(step)
            outcome = env.step(action)
            environment_done = bool(outcome["done"])
            if bool(outcome["environment_action_executed"]):
                if environment_actions >= max_environment_actions:
                    raise RuntimeError("outer environment-action ceiling would be exceeded")
                action_events.append({
                    "episode_task_id": logical_id,
                    "step_index": environment_actions,
                    "command_turn_index": step,
                    "action": str(outcome["action"]),
                    "env_feedback": str(outcome["observation"]),
                    "reward": float(outcome["score"]),
                    "done": bool(outcome["done"]),
                    "won": bool(outcome["won"]),
                    "admissible_before": action in valid_before,
                })
                environment_actions += 1
            trajectory.extend([
                ["ACTION", str(outcome["action"])],
                ["OBSERVATION", str(outcome["observation"])],
            ])
            if tracker is not None:
                new_progress = tracker.observe(
                    str(outcome["observation"]), done=bool(outcome["done"]),
                )
                if new_progress > progress_rate:
                    progress_events.append([step, float(new_progress)])
                progress_rate = float(new_progress)
            if bool(outcome["won"]):
                termination_reason = "official_won"
                break
            if bool(outcome["done"]):
                termination_reason = "environment_done_without_win"
                break
        usage = _sum_usage(provider_calls)
        command_turns = (len(trajectory) - 1) // 2
        return {
            "job_index": None,
            "task_id": task.task_id,
            "task_key": task_key,
            "task_type": task.task_type,
            "manifest_index": task.index,
            "sample_idx": sample_idx,
            "sample_seed": int(sample_seed),
            "goal": goal,
            "trajectory": trajectory,
            "grounding": grounding,
            "progress": progress_events,
            "command_turns": command_turns,
            "grounding_rate": len(grounding) / command_turns if command_turns else 0.0,
            "progress_rate": progress_rate,
            "official_success": bool(env.won),
            "termination_reason": termination_reason,
            "environment_done": environment_done,
            "actions": action_events,
            "environment_actions": environment_actions,
            "provider_calls": provider_calls,
            "target_usage": asdict(usage),
            "retrieval": retrieval_events,
            "embedding_calls": sum(
                int(item.get("embedding_calls", 0)) for item in retrieval_events
            ),
            "actual_gamefile": str(env.actual_gamefile),
            "gamefile": task.gamefile_rel,
            "gamefile_hash": task.gamefile_sha256,
            "wall_time_ms": max(0, int((time.perf_counter() - started) * 1000)),
            "failure_kind": "",
        }
    except ProviderExecutionError as exc:
        provider_calls.append(dict(exc.event))
        exc.event["provider_calls"] = [dict(event) for event in provider_calls]
        exc.event["action_events"] = [dict(event) for event in action_events]
        raise
    except BaseException as exc:
        # Preserve already-paid calls/actions when ALFWorld or adapter code
        # fails after one or more successful turns.
        try:
            setattr(exc, "provider_calls", [dict(event) for event in provider_calls])
            setattr(exc, "action_events", [dict(event) for event in action_events])
        except Exception:
            pass
        raise
    finally:
        env.close()
