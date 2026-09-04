"""Benchmark authority: official ``won`` plus post-hoc strict TaskContract.

The official success signal comes from the method's own environment
(``infos["won"]``).  The strict signal replays the baseline's recorded action
sequence through the Ours ALFWorld Harness boundary and asks the same
``AlfWorldValidatorChannel.validate_task_contract`` authority the main
experiment uses.  The replay never feeds facts back to a baseline agent and
never participates in baseline training.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atomic_skillgraph.harness.alfworld import (
    AlfWorldAdapter,
    parse_alfworld_action,
)
from atomic_skillgraph.harness.protocol import (
    HarnessActionSpec,
    HarnessTask,
)

from .manifest import ManifestTask

# AlfWorldAdapter split names for the three ALFWorld physical splits.
_SPLIT_MAP = {
    "train": "train",
    "valid_seen": "eval_in_distribution",
    "valid_unseen": "eval_out_of_distribution",
}


@dataclass(frozen=True)
class StrictOutcome:
    official_success: bool
    task_contract_success: bool
    strict_success: bool
    environment_actions: int
    invalid_actions: int
    replayed_terminal_won: bool


def _extract_goal(observation: str) -> str:
    """Same goal derivation the harness uses at its raw-reset boundary."""

    marker = "your task is to:"
    offset = observation.casefold().find(marker)
    if offset >= 0:
        return observation[offset + len(marker):].strip()
    return observation[:300].strip()


class StrictTaskEvaluator:
    """Post-hoc strict evaluator; consumes saved baseline action sequences."""

    def __init__(self, alfworld_data: str | Path | None = None) -> None:
        self.alfworld_data = Path(
            alfworld_data
            or os.environ.get("ALFWORLD_DATA", str(Path.home() / ".cache" / "alfworld"))
        )

    def _build_probe_task(self, entry: ManifestTask) -> tuple[AlfWorldAdapter, HarnessTask]:
        if entry.source_split not in _SPLIT_MAP:
            raise ValueError(f"unsupported source split: {entry.source_split}")
        adapter = AlfWorldAdapter(
            split=_SPLIT_MAP[entry.source_split], alfworld_data=str(self.alfworld_data),
        )
        adapter.initialize()
        game_file = str(self.alfworld_data / entry.gamefile_rel)
        probe = HarnessTask(
            task_id=entry.task_id,
            goal="",
            benchmark="alfworld",
            task_type=entry.task_type,
            context={"env_index": entry.env_index, "game_file": game_file},
        )
        adapter.reset(probe)
        return adapter, probe

    def evaluate(
        self,
        entry: ManifestTask,
        action_texts: list[str],
        *,
        official_success: bool,
    ) -> StrictOutcome:
        adapter, probe = self._build_probe_task(entry)
        goal = _extract_goal(str(adapter._observation))
        task = HarnessTask(
            task_id=entry.task_id,
            goal=goal,
            benchmark="alfworld",
            task_type=entry.task_type,
            context={
                "env_index": entry.env_index,
                "game_file": str(self.alfworld_data / entry.gamefile_rel),
            },
        )
        validator = adapter.validator_channel()
        invalid_actions = 0
        executed = 0
        for index, raw_text in enumerate(action_texts):
            if adapter._done or adapter._won:
                break
            action_type, arguments, text, parser_metadata = parse_alfworld_action(raw_text)
            # Mirror the harness execute_action boundary exactly, including its
            # "nothing happens" acceptance criterion and its terminal latch.
            spec = HarnessActionSpec(
                action_id=f"strict_post:{index}",
                revision=adapter._revision + 1,
                action_type=action_type,
                arguments=arguments,
                display_text=text,
                raw_action=text,
                metadata={"origin": "baseline_strict_replay", **parser_metadata},
            )
            observations, scores, dones, infos = adapter._env.step([text])
            observation = str(observations[0])
            done = bool(dones[0])
            won_values = infos.get("won", [False])
            won = bool(won_values[0]) if won_values else False
            accepted = "nothing happens" not in observation.casefold()
            adapter._revision += 1
            catalog = adapter._replace_action_catalog(
                list(infos.get("admissible_commands", [[]])[0]), adapter._revision,
            )
            validator.record(
                spec,
                accepted=accepted,
                revision=adapter._revision,
                done=done,
                won=won,
                observation=observation,
                metadata={"score": float(scores[0])},
                catalog=catalog,
            )
            adapter._observation, adapter._done, adapter._won = observation, done, won
            executed += 1
            if not accepted or action_type == "UNKNOWN":
                invalid_actions += 1
        contract = adapter.task_contract(task)
        task_contract_success = bool(
            validator.validate_task_contract(contract).passed
        )
        return StrictOutcome(
            official_success=bool(official_success),
            task_contract_success=task_contract_success,
            strict_success=bool(official_success) and task_contract_success,
            environment_actions=executed,
            invalid_actions=invalid_actions,
            replayed_terminal_won=bool(adapter._won),
        )


def normalize_baseline_action_text(text: str) -> str:
    """Normalize one baseline action string for audit/display purposes."""

    return re.sub(r"\s+", " ", str(text)).strip()
