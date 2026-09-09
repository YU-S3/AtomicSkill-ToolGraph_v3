from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

import experiments.run_v3_smoke as smoke
from atomic_skillgraph.system import load_config


def _alias(trace_index: int) -> dict[str, object]:
    event_id = f"e{trace_index}"
    return {
        "authority_ref": f"semantic_alias:{event_id}:object:light",
        "event_id": event_id,
        "event_index": trace_index,
        "kind": "semantic_alias",
        "source_kind": "semantic_snapshot_alias",
        "role": "light",
        "value": f"lamp_{trace_index + 1}",
        "source_authority_ref": f"action_arg:{event_id}:object",
        "source_argument_role": "object",
        "predicate": "light.on",
        "predicate_argument_role": "light",
        "witness_ref": f"witness:{event_id}:light.on",
        "effect_domain": "world",
    }


def _audit_inputs() -> dict[str, object]:
    train_records = [
        {
            "task_id": f"train-{index}",
            "task_signature": f"train-signature-{index}",
            "task_type": "look_at_obj_in_light",
            "trace_id": f"train-trace-{index}",
            "benchmark_success": True,
            "infrastructure_failure": False,
            "resource_usage_complete": True,
            "semantic_aliases": [_alias(index)] if index == 0 else [],
            "e1_validated_object_observed": index == 0,
        }
        for index in range(5)
    ]
    eval_records = [
        {
            "task_id": f"eval-{index}",
            "task_signature": f"eval-signature-{index}",
            "task_type": "look_at_obj_in_light",
            "trace_id": f"eval-trace-{index}",
            "benchmark_success": True,
            "graph_self_sufficient_success": index == 0,
            "task_rescue_required": False,
            "task_contract_success": True,
            "infrastructure_failure": False,
            "resource_usage_complete": True,
            "runtime_source": "stored_composite" if index == 0 else "full_dynamic",
            "source_composite_ref": (
                "skill://observe-composite@1.0.0" if index == 0 else ""
            ),
        }
        for index in range(5)
    ]
    return {
        "configuration_checks": {
            "provider_capability_passed": True,
            "train_preflight_passed": True,
            "fresh_empty_bank": True,
            "method_patch_3_2": True,
            "train_split": True,
            "frozen_system_readonly": True,
            "eval_preflight_passed": True,
            "eval_split_valid_unseen": True,
            "freeze_manifest_digest_match": True,
        },
        "train_records": train_records,
        "composite_records": [{
            "composite_ref": "skill://observe-composite@1.0.0",
            "status": "active",
            "task_contract_covered": True,
            "goal_covers_object_observed": True,
            "source_trace_ids": ["train-trace-0"],
        }],
        "eval_records": eval_records,
        "final_maintenance_pending_count": 0,
        "digests": {
            "source_before_freeze": "digest",
            "source_after_freeze": "digest",
            "frozen_before_eval": "digest",
            "frozen_after_eval": "digest",
        },
    }


def test_r7_targeted_audit_accepts_one_continuous_authority_chain() -> None:
    audit = smoke._r7_look_targeted_audit(**_audit_inputs())

    assert audit["passed"] is True
    assert all(audit["checks"].values())
    assert audit["authority_closed_trace_ids"] == ["train-trace-0"]
    assert audit["active_chain_composite_refs"] == [
        "skill://observe-composite@1.0.0"
    ]
    assert audit["heldout_stored_composite_task_ids"] == ["eval-0"]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_alias",
        "missing_e1",
        "uncovered_composite",
        "candidate_composite",
        "unrelated_composite_trace",
        "no_stored_composite",
        "stored_composite_rescued_dynamically",
        "overlapping_signature",
        "pending_maintenance",
        "changed_frozen_digest",
    ],
)
def test_r7_targeted_audit_fails_closed_when_chain_breaks(
    mutation: str,
) -> None:
    inputs = copy.deepcopy(_audit_inputs())
    if mutation == "missing_alias":
        inputs["train_records"][0]["semantic_aliases"] = []
    elif mutation == "missing_e1":
        inputs["train_records"][0]["e1_validated_object_observed"] = False
    elif mutation == "uncovered_composite":
        inputs["composite_records"][0]["task_contract_covered"] = False
    elif mutation == "candidate_composite":
        inputs["composite_records"][0]["status"] = "candidate"
    elif mutation == "unrelated_composite_trace":
        inputs["composite_records"][0]["source_trace_ids"] = ["other-trace"]
    elif mutation == "no_stored_composite":
        inputs["eval_records"][0].update({
            "runtime_source": "full_dynamic",
            "source_composite_ref": "",
        })
    elif mutation == "stored_composite_rescued_dynamically":
        inputs["eval_records"][0].update({
            "benchmark_success": True,
            "graph_self_sufficient_success": False,
            "task_rescue_required": True,
        })
    elif mutation == "overlapping_signature":
        inputs["eval_records"][0]["task_signature"] = (
            inputs["train_records"][0]["task_signature"]
        )
    elif mutation == "pending_maintenance":
        inputs["final_maintenance_pending_count"] = 1
    elif mutation == "changed_frozen_digest":
        inputs["digests"]["frozen_after_eval"] = "changed"

    audit = smoke._r7_look_targeted_audit(**inputs)

    assert audit["passed"] is False
    assert not all(audit["checks"].values())


def test_r7_targeted_config_freezes_exact_look_five_selection() -> None:
    config_path = (
        Path(smoke.REPO_ROOT) / "configs" / "alfworld_r7_look_targeted.yaml"
    )
    config = load_config(config_path)

    smoke._require_r7_look_selection(config)
    selection = config["harness"]["task_selection"]
    assert selection == {
        "policy": "balanced_fixed_manifest",
        "task_types": ["look_at_obj_in_light"],
        "tasks_per_type": 5,
        "total_tasks": 5,
        "require_exact_count": True,
    }


def test_cli_dispatches_r7_look_targeted_mode(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        smoke,
        "run_r7_look_targeted",
        lambda config: calls.append(str(config)) or 29,
    )

    assert smoke.main([
        "--r7-look-targeted",
        "--config",
        "configs/alfworld_r7_look_targeted.yaml",
    ]) == 29
    assert calls == ["configs/alfworld_r7_look_targeted.yaml"]


def test_r7_targeted_provider_failure_writes_one_fail_closed_audit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = load_config(
        Path(smoke.REPO_ROOT) / "configs" / "alfworld_r7_look_targeted.yaml"
    )
    source.pop("_config_path", None)
    source["experiment"]["output_dir"] = str(tmp_path / "targeted")
    config_path = tmp_path / "targeted.yaml"
    config_path.write_text(
        yaml.safe_dump(source, sort_keys=False),
        encoding="utf-8",
    )
    # This is a fail-closed control-flow test, not a repository hashing test.
    # Avoid making the full suite rescan the worktree before the injected
    # provider-capability failure is reached.
    monkeypatch.setattr(smoke, "hash_config", lambda _path: "config-hash")
    monkeypatch.setattr(smoke, "hash_code", lambda _path: "code-hash")
    monkeypatch.setattr(
        smoke,
        "ensure_provider_capability",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("provider unavailable")
        ),
    )

    assert smoke.run_r7_look_targeted(config_path) == 1

    audit_paths = list((tmp_path / "targeted").glob(
        "run_*/r7_look_targeted_result.json"
    ))
    assert len(audit_paths) == 1
    result = json.loads(audit_paths[0].read_text(encoding="utf-8"))
    assert result["passed"] is False
    assert result["gate"] == "r7_look_targeted_full_chain"
    assert result["error_type"] == "RuntimeError"
    assert result["error"] == "provider unavailable"
