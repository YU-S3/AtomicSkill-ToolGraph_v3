"""No-fold adapter around pinned SkillGen extraction and embedding cores."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import random
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class Encoder(Protocol):
    def encode(self, values: list[str], *, convert_to_numpy: bool = True) -> Any: ...


@dataclass(frozen=True)
class ExtractionResult:
    root: Path
    files: dict[str, Path]
    metrics: dict[str, Any]


class UpstreamExtractionCore:
    """Verified callable surface loaded from the exact external checkout.

    SkillGen uses flat imports inside ``skill_extraction``.  The loader binds
    those names only while the pinned modules are executed, then restores the
    caller's module table.  No upstream file is copied or patched.
    """

    def __init__(self, external_root: str | Path) -> None:
        self.external_root = Path(external_root).resolve(strict=True)
        source_root = self.external_root / "skill_extraction"
        self.extraction_utils = self._load(
            "_asg_skillgen_extraction_utils",
            source_root / "extraction_utils.py",
        )
        previous = sys.modules.get("extraction_utils")
        sys.modules["extraction_utils"] = self.extraction_utils
        try:
            self.domain_graph = self._load(
                "_asg_skillgen_domain_graph",
                source_root / "domain_graph.py",
            )
        finally:
            if previous is None:
                sys.modules.pop("extraction_utils", None)
            else:
                sys.modules["extraction_utils"] = previous
        self.embed_skills = self._load(
            "_asg_skillgen_embed_skills",
            source_root / "embed_skills.py",
        )
        self._assert_contract()

    @staticmethod
    def _load(name: str, path: Path) -> Any:
        if not path.is_file():
            raise FileNotFoundError(f"pinned SkillGen source file is missing: {path}")
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot construct module spec for {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        origin = Path(str(module.__file__)).resolve()
        if origin != path.resolve():
            raise RuntimeError(f"SkillGen module loaded from wrong origin: {origin}")
        return module

    def _assert_contract(self) -> None:
        signature = inspect.signature(self.domain_graph.ActionContributionEstimator)
        expected = {"gamma": 0.95, "lambda_": 0.9, "alpha": 0.05}
        for name, value in expected.items():
            actual = signature.parameters[name].default
            if actual != value:
                raise RuntimeError(
                    f"pinned SkillGen {name} default drifted: {actual!r} != {value!r}"
                )
        for name in (
            "keep_one_of_duplicates_dicts", "collect_valid_traj",
            "process_trajectories",
        ):
            if not callable(getattr(self.extraction_utils, name, None)):
                raise RuntimeError(f"pinned SkillGen extraction core lacks {name}")
        if not callable(getattr(self.embed_skills, "add_current_node_embeddings", None)):
            raise RuntimeError("pinned SkillGen embedding core is unavailable")


def load_sentence_encoder(model_name: str) -> Encoder:
    """Load the method-owned local encoder lazily in the SkillGen venv."""

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_name), device="cpu")
    model.eval()
    return model


def _upstream_sample(row: dict[str, Any]) -> dict[str, Any]:
    """Strip audit-only fields before upstream duplicate/trajectory handling."""

    return {
        "task_uid": str(row["task_key"]),
        "task_name": str(row["task_key"]),
        "goal": str(row["goal"]),
        "progress": [list(item) for item in row["progress"]],
        "trajectory": [list(item) for item in row["trajectory"]],
        "grounding": [int(item) for item in row["grounding"]],
        "num_step": int(row["command_turns"]),
        "grounding_rate": float(row["grounding_rate"]),
        "progress_rate": float(row["progress_rate"]),
    }


def _edge_rows(task_graph: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, target in sorted(task_graph.graph.edges):
        rows.append({
            "source": str(source),
            "target": str(target),
            "progress_deltas": [
                float(value)
                for value in task_graph.edge_progress.get((source, target), [])
            ],
        })
    return rows


def _skill_rows(ranking: dict[str, list[tuple[str, float]]]) -> list[dict[str, Any]]:
    return [
        {
            "current_node": str(parent),
            "children": [
                {"action": str(child), "score": float(score)}
                for child, score in children
            ],
        }
        for parent, children in sorted(ranking.items())
    ]


def _golden_segment(trajectory: dict[str, Any]) -> str:
    lines = [f"Goal: {trajectory['goal']}", str(trajectory["init_obs"])]
    lines.extend(f"ACTION: {action}" for action in trajectory["actions"])
    return "Example 1:\n" + "\n".join(lines)


def _write_bytes_atomic(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _write_json(path: Path, payload: Any) -> Path:
    return _write_bytes_atomic(
        path,
        (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        .encode("utf-8"),
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    body = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    return _write_bytes_atomic(path, body.encode("utf-8"))


def extract_skill_library(
    corpus_rows: list[dict[str, Any]],
    *,
    output_dir: str | Path,
    external_root: str | Path,
    encoder: Encoder | None = None,
    encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    upstream: UpstreamExtractionCore | Any | None = None,
    extraction_seed: int = 42,
    q_iterations: int = 500,
    gamma: float = 0.95,
    lambda_: float = 0.9,
    alpha: float = 0.05,
) -> ExtractionResult:
    """Run one deterministic extraction barrier over the complete Train corpus."""

    if q_iterations != 500 or (gamma, lambda_, alpha) != (0.95, 0.9, 0.05):
        raise ValueError("SkillGen graph/TD hyperparameters differ from the frozen method")
    if extraction_seed != 42:
        raise ValueError("pinned SkillGen extraction seed must remain 42")
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    core = upstream or UpstreamExtractionCore(external_root)
    encoder = encoder or load_sentence_encoder(encoder_model)

    upstream_samples = [_upstream_sample(row) for row in corpus_rows]
    positive = [
        row for row in upstream_samples
        if row["grounding"] and float(row["progress_rate"]) > 0.0
    ]
    deduplicated = core.extraction_utils.keep_one_of_duplicates_dicts(positive)
    valid = core.extraction_utils.collect_valid_traj(
        deduplicated, dataset_name="alfworld",
    )

    categories = sorted({str(row["task_name"]).split("-")[0] for row in upstream_samples})
    files: dict[str, Path] = {}
    graph_nodes = graph_edges = skill_count = 0
    embedding_calls = 0
    per_category: dict[str, Any] = {}
    random.seed(extraction_seed)
    core.domain_graph.np.random.seed(extraction_seed)

    for category in categories:
        category_valid = [row for row in valid if row["task_name"] == category]
        graph_path = root / "domain_graphs" / f"{category}.json"
        skills_path = root / "step_skills" / f"{category}.jsonl"
        embedded_path = root / "step_skill_embeddings" / f"{category}.jsonl"
        golden_path = root / "golden_segments" / f"{category}.txt"

        ranking: dict[str, list[tuple[str, float]]] = {}
        nodes: list[str] = []
        edges: list[dict[str, Any]] = []
        if category_valid:
            task_graph = core.domain_graph.TaskGraph(
                save_dir=str(root / "domain_graphs"), task_name=category,
            )
            trajectories = core.extraction_utils.process_trajectories(category_valid)
            task_graph.add_sample_trajectory([item["trajs"] for item in trajectories])
            nodes = sorted(str(node) for node in task_graph.graph.nodes)
            edges = _edge_rows(task_graph)
            if "INIT_STATE" in task_graph.graph and "FINAL_STATE" in task_graph.graph:
                estimator = core.domain_graph.ActionContributionEstimator(
                    task_graph, gamma=gamma, lambda_=lambda_, alpha=alpha,
                )
                estimator.compute_q_values(num_iterations=q_iterations)
                ranking = estimator.combined_ranking_step_wise()
        skill_rows = _skill_rows(ranking)
        embedded_rows = (
            core.embed_skills.add_current_node_embeddings(
                json.loads(json.dumps(skill_rows)), encoder,
            )
            if skill_rows else []
        )
        if skill_rows:
            embedding_calls += 1
        chosen = sorted(
            category_valid,
            key=lambda row: (-float(row["progress_rate"]), int(row["unique_id"])),
        )[:1]
        golden = _golden_segment(chosen[0]) if chosen else ""

        _write_json(graph_path, {"nodes": nodes, "edges": edges})
        _write_jsonl(skills_path, skill_rows)
        _write_jsonl(embedded_path, embedded_rows)
        _write_bytes_atomic(golden_path, golden.encode("utf-8"))
        files.update({
            f"domain_graphs/{category}.json": graph_path,
            f"step_skills/{category}.jsonl": skills_path,
            f"step_skill_embeddings/{category}.jsonl": embedded_path,
            f"golden_segments/{category}.txt": golden_path,
        })
        graph_nodes += len(nodes)
        graph_edges += len(edges)
        skill_count += sum(len(row["children"]) for row in skill_rows)
        per_category[category] = {
            "valid_trajectory_count": len(category_valid),
            "graph_nodes": len(nodes),
            "graph_edges": len(edges),
            "skill_edges": sum(len(row["children"]) for row in skill_rows),
            "golden_segment": bool(golden),
        }

    # Test-time domain retrieval is computed from the visible Test goal against
    # this frozen Train-only index.  No shipped valid_unseen label metadata is
    # copied into the artifact.
    task_metadata: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    for row in corpus_rows:
        task_id = str(row["task_id"])
        if task_id in seen_tasks:
            continue
        seen_tasks.add(task_id)
        domain = str(row["task_type"])
        goal = str(row["goal"])
        meta_text = f"Goal: {goal} Domain: {domain}."
        embedding = encoder.encode([meta_text], convert_to_numpy=True)[0]
        embedding_calls += 1
        task_metadata.append({
            "task_id": task_id,
            "task_key": str(row["task_key"]),
            "goal": goal,
            "domain": domain,
            "meta_data": meta_text,
            "embedding": [float(value) for value in embedding],
        })
    metadata_path = root / "train_metadata_embeddings.jsonl"
    _write_jsonl(metadata_path, task_metadata)
    files["train_metadata_embeddings.jsonl"] = metadata_path

    metrics = {
        "sampling_episode_count": len(corpus_rows),
        "valid_trajectory_count": len(valid),
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
        "skill_count": skill_count,
        "golden_segment_count": sum(
            int(item["golden_segment"]) for item in per_category.values()
        ),
        "embedding_calls": embedding_calls,
        "embedding_model": encoder_model,
        "td_gamma": gamma,
        "td_lambda": lambda_,
        "td_alpha": alpha,
        "q_iterations": q_iterations,
        "extraction_seed": extraction_seed,
        "per_category": per_category,
        "upstream_source_sha256": {
            name: hashlib.sha256(
                (Path(external_root) / relative).read_bytes()
            ).hexdigest()
            for name, relative in {
                "domain_graph": "skill_extraction/domain_graph.py",
                "extraction_utils": "skill_extraction/extraction_utils.py",
                "embed_skills": "skill_extraction/embed_skills.py",
            }.items()
        },
    }
    manifest_path = root / "artifact_manifest.json"
    _write_json(manifest_path, {"schema_version": 1, "metrics": metrics})
    files["artifact_manifest.json"] = manifest_path
    return ExtractionResult(root=root, files=files, metrics=metrics)
