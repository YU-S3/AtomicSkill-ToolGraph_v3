"""Convert immutable identities and visible reset text to upstream tasks."""
from __future__ import annotations

import random
import re

from experiments.baselines.common.manifest import ManifestTask

FAMILIES = {
    "pick_and_place_simple": ("pick_and_place", "put"),
    "look_at_obj_in_light": ("look_at_obj", "examine"),
    "pick_clean_then_place_in_recep": ("pick_clean_then_place", "clean"),
    "pick_heat_then_place_in_recep": ("pick_heat_then_place", "heat"),
    "pick_cool_then_place_in_recep": ("pick_cool_then_place", "cool"),
    "pick_two_obj_and_place": ("pick_two_obj", "puttwo"),
}


def train_chunks(tasks, *, seed: int, epochs: int, chunk_size: int):
    # Equivalent to upstream get_shuffled_task_order(total, pass_id=0, seed).
    if len(tasks) != epochs * chunk_size:
        raise ValueError("Every Train task must be consumed exactly once")
    order = list(tasks)
    random.Random(seed).shuffle(order)
    return [order[i:i + chunk_size] for i in range(0, len(order), chunk_size)]


def visible_task(task: ManifestTask, observation: str) -> dict:
    match = re.search(r"Your task is to:\s*(.+)", observation, flags=re.I | re.S)
    if not match:
        raise ValueError("Exact reset observation has no public task goal")
    env_name, prior_type = FAMILIES[task.task_type]
    return {
        "task_main": env_name + "-" + match.group(1).strip(),
        "task_description": observation.split("___")[0],
        "task_type": prior_type,
        "env_name": env_name,
    }
