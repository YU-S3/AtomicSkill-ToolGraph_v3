"""R01: provisional contracts use the real single-node Agent protocol."""
from types import SimpleNamespace

import pytest

from atomic_skillgraph.core.contracts import ColdStartPlanStep, ParameterSpec
from atomic_skillgraph.core.serialization import to_primitive
from atomic_skillgraph.runtime.cold_start_executor import ProvisionalNodeExecutor
from test_r10_runtime import setup, action


@pytest.mark.parametrize("complete", [True, False])
def test_provisional_preserves_actual_contract_guidance_and_explicit_completion(tmp_path, complete):
    def choose(request, count):
        assert not any(t.name.startswith("invoke_impl_") for t in request.tools)
        guidance = request.policy_context["current_state_snapshot"]["current_atomic"]["skill_guidance"]
        assert guidance["steps"] == ["Use current public evidence to choose your own method."]
        assert guidance["soft_reference"]
        if count == 1:
            return action(request, "GO_TO", destination="cabinet_1")
        if count == 2:
            return action(request, "OPEN")
        if not complete:
            return "report_runtime_status", {"status": "cannot_resolve"}
        name, args = action(request, "TAKE", object="egg_1")
        args.update(intent="attempt_current_atomic", candidate_bindings={"object": "egg_1", "source": "cabinet_1"},
                    candidate_outputs={"held_object": "egg_1"})
        return name, args

    system, ctx, occurrence, invocations, provider = setup(tmp_path, choose)
    try:
        atomic = invocations[0].atomic
        atomic.outputs = [ParameterSpec('held_object', 'entity')]
        atomic.validator_spec['output_derivations'] = {'held_object': {'kind': 'input_identity', 'input_role': 'object'}}
        record = SimpleNamespace(provisional_ref="provisional://take@1.0.0", canonical_intent=atomic.summary,
            atomic_contract={key: to_primitive(getattr(atomic, key)) for key in
                             ("inputs", "outputs", "preconditions", "effects", "validator_spec")},
            seeded_guideline={"steps": ["Use current public evidence to choose your own method."]},
            harness_profile=ctx.harness.profile_name)
        step = ColdStartPlanStep("provisional", [], "provisional", record.provisional_ref, "seeded_only",
                                 occurrence.binding_specs, {}, atomic.effects)
        result = ProvisionalNodeExecutor(system.orchestrator.node_executor).execute(record, ctx, step,
            progress_tracker=SimpleNamespace(record=lambda source: SimpleNamespace(progress_digest=source)))
        assert result.local_effect_passed is complete
        assert len(provider.requests) == 3
        assert all([m["role"] for m in r.messages] == ["system", "user"] for r in provider.requests)
        trace = ctx.trace_builder.trace
        assert not trace.tool_executions and not trace.implementation_invocations
        assert len(trace.environment_actions) == (3 if complete else 2)
        if complete:
            assert result.witness_refs
            assert ctx.validated_outputs["cold::provisional"] == {"held_object": "egg_1"}
        else:
            assert result.failure_code == "runtime_binding_unresolved"
            assert not ctx.validated_outputs
    finally:
        system.close()
