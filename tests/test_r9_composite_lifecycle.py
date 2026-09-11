from __future__ import annotations

import pytest

from atomic_skillgraph.core.status import SkillStatus
from atomic_skillgraph.governance.lifecycle import LifecyclePolicy, LifecycleThresholds
from atomic_skillgraph.governance.projections import ArtifactStats


REF = "skill://r9-composite@1.0.0"


def _stats(successes: int, unsuccessful: int, *, consecutive: int = 0) -> ArtifactStats:
    return ArtifactStats(
        REF,
        "composite",
        event_task_ids={
            "deployment_success": [f"success-{index}" for index in range(successes)],
            "deployment_unsuccessful": [
                f"unsuccessful-{index}" for index in range(unsuccessful)
            ],
        },
        consecutive_deployment_unsuccessful=consecutive,
    )


def test_r9_candidate_zero_of_three_is_suppressed() -> None:
    decision = LifecyclePolicy().review_composite(
        REF, SkillStatus.CANDIDATE, _stats(0, 3)
    )
    assert decision.next_status == SkillStatus.SUPPRESSED.value


def test_r9_candidate_one_of_five_hits_activation_deadline() -> None:
    decision = LifecyclePolicy().review_composite(
        REF, SkillStatus.CANDIDATE, _stats(1, 4)
    )
    assert decision.next_status == SkillStatus.SUPPRESSED.value
    assert decision.reason == "candidate_failed_to_activate_within_deployment_window"


def test_r9_candidate_second_success_on_trial_four_activates() -> None:
    decision = LifecyclePolicy().review_composite(
        REF, SkillStatus.CANDIDATE, _stats(2, 2)
    )
    assert decision.next_status == SkillStatus.ACTIVE.value
    assert decision.reason == "independent_deployment_successes"


def test_r9_active_three_consecutive_unsuccessful_uses_empirical_circuit_breaker() -> None:
    decision = LifecyclePolicy().review_composite(
        REF, SkillStatus.ACTIVE, _stats(2, 3, consecutive=3)
    )
    assert decision.next_status == SkillStatus.SUPPRESSED.value
    assert decision.reason == "repeated_empirical_deployment_unsuccessful"


def test_r9_deployment_success_reset_is_projection_owned() -> None:
    stats = _stats(2, 2, consecutive=0)
    decision = LifecyclePolicy().review_composite(REF, SkillStatus.ACTIVE, stats)
    assert decision.next_status == SkillStatus.ACTIVE.value


@pytest.mark.parametrize("field", [
    "composite_active_deployment_successes",
    "composite_candidate_zero_success_trial_limit",
    "composite_candidate_activation_trial_limit",
    "composite_active_consecutive_deployment_unsuccessful_limit",
])
@pytest.mark.parametrize("invalid", [True, 1.5, 0, -1])
def test_r9_lifecycle_thresholds_are_strict_positive_integers(
    field: str, invalid: object
) -> None:
    with pytest.raises(ValueError, match=field):
        LifecycleThresholds(**{field: invalid})
