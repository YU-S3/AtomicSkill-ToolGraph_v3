from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import experiments.run_v3_smoke as smoke


def _trace() -> SimpleNamespace:
    return SimpleNamespace(
        task=SimpleNamespace(task_type="look_at_obj_in_light"),
        benchmark_success=True,
        task_contract_success=True,
        infrastructure_failure=False,
        resource_usage_complete=True,
        failures=[],
        environment_actions=[SimpleNamespace(
            action_id="r000_a001",
            revision=0,
            action_type="GO_TO",
            arguments={"location": "desk_1"},
            accepted=True,
            observation="arrived",
            done=False,
            won=False,
            new_revision=1,
            span_id="span-1",
        )],
        metadata={
            "method_patch": "3.2",
            "extraction": {"attempted": True, "error_code": ""},
            "semantic_state_snapshots": [
                {
                    "sequence_index": 0,
                    "revision": 0,
                    "origin": "reset",
                    "action_id": "",
                    "occurrence_id": "",
                    "accepted": True,
                    "done": False,
                    "won": False,
                    "facts": [{
                        "predicate": "entity.discovered_at",
                        "args": {"entity": "lamp_1", "location": "desk_1"},
                        "effect_domain": "evidence",
                        "witness_ref": (
                            "alfworld_action_fact:r0:entity.discovered_at:"
                            "entity=lamp_1,location=desk_1"
                        ),
                    }],
                },
                {
                    "sequence_index": 1,
                    "revision": 1,
                    "origin": "environment_action",
                    "action_id": "r000_a001",
                    "occurrence_id": "occ-1",
                    "accepted": True,
                    "done": False,
                    "won": False,
                    "facts": [{
                        "predicate": "agent.at_location",
                        "args": {"location": "desk_1"},
                        "effect_domain": "world",
                        "witness_ref": (
                            "alfworld_action_fact:r1:agent.at_location:"
                            "location=desk_1"
                        ),
                    }],
                },
            ],
        },
    )


def test_deterministic_runner_collects_complete_ours_roots(
    monkeypatch, capsys,
) -> None:
    captured: dict[str, object] = {}

    def run(command, *, cwd, check):
        captured.update(command=command, cwd=cwd, check=check)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(smoke.subprocess, "run", run)

    assert smoke.run_deterministic() == 0
    assert captured["command"] == [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests",
        "experiments/tests",
        "src/atomic_skillgraph/governance/tests",
    ]
    assert captured["cwd"] == smoke.REPO_ROOT
    assert captured["check"] is False
    result = json.loads(capsys.readouterr().out)
    assert result == {
        "passed": True,
        "gate": "deterministic_no_api_fullchain",
        "collection": "full_ours_test_roots",
    }


def test_deterministic_runner_propagates_pytest_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        smoke.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 7),
    )

    assert smoke.run_deterministic() == 7


def test_look_at_authority_audit_accepts_complete_snapshot_timeline() -> None:
    trace = _trace()
    audit = smoke._look_at_authority_smoke_audit(trace, {
        "semantic_authority_source": "validator_snapshot_v3_2",
        "actions": [{}],
    })

    assert audit["passed"] is True
    assert all(audit["checks"].values())
    assert audit["semantic_snapshot_count"] == 2
    assert audit["environment_action_count"] == 1


def test_look_at_authority_audit_rejects_missing_or_conflicting_authority() -> None:
    trace = _trace()
    trace.metadata["semantic_state_snapshots"][1].update({
        "revision": 0,
        "facts": [{
            "predicate": "light.on",
            "args": {"light": "lamp_1"},
            "effect_domain": "world",
            "witness_ref": "alfworld_action_fact:r0:light.on:light=lamp_1",
        }],
    })
    trace.metadata["extraction"] = {
        "attempted": False,
        "error_code": "semantic_snapshot_integrity_error",
    }

    audit = smoke._look_at_authority_smoke_audit(trace, {
        "semantic_authority_source": "legacy_action_reducer",
        "actions": [{}],
    })

    assert audit["passed"] is False
    assert audit["checks"]["semantic_revisions_consistent"] is False
    assert audit["checks"]["environment_action_timeline_consistent"] is False
    assert audit["checks"]["validator_snapshot_authority_used"] is False
    assert audit["checks"]["success_evolution_attempted"] is False
    assert audit["checks"]["semantic_snapshot_integrity_error_absent"] is False


def test_cli_dispatches_independent_look_at_authority_mode(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        smoke,
        "run_look_at_authority_smoke",
        lambda config: calls.append(str(config)) or 23,
    )

    assert smoke.main([
        "--look-at-authority", "--config", "configs/formal.yaml",
    ]) == 23
    assert calls == ["configs/formal.yaml"]
