"""Production authorization/Repeat consumers, without mocked validators."""
from copy import deepcopy
from dataclasses import replace
import pytest
from test_r10_runtime import setup
from atomic_skillgraph.core.contracts import ParameterSpec
from atomic_skillgraph.core.bindings import BindingExpression,BindingSource,RuntimeBinding,BindingStatus,BindingResolution
from atomic_skillgraph.core.edges import GraphEdge
from atomic_skillgraph.runtime.input_authorization import authorize,validate_committed
from atomic_skillgraph.traces.compiler_observer import _dataflow_inputs


def test_compiled_control_preparation_is_pending_and_failure_does_not_commit(tmp_path):
    system,ctx,occ,inv,_=setup(tmp_path,lambda *a:pytest.fail('unexpected provider'))
    try:
        from experiments.r10_world_checks import install_fixture
        from atomic_skillgraph.core.contracts import SemanticPredicate
        from atomic_skillgraph.core.results import RuntimeOccurrence
        a,i=install_fixture(system,'move_control',[],
            [SemanticPredicate('agent.at_location',{'location':BindingExpression('skill_input',source_role='target')})],
            [('GO_TO','destination')],source_target='countertop_1')
        occ=RuntimeOccurrence('control','control',a.ref,[],{},[i.ref],a.effects)
        ctx.plan.occurrences.append(occ)
        compiled=system.invocation_compiler.compile_candidates(occ,ctx.binding_store,task_id=ctx.task_id)[0]
        compiled=deepcopy(compiled)
        compiled.atomic.inputs.append(ParameterSpec('permission','bool'))
        compiled.atomic.validator_spec['input_authorization']={'permission':{'kind':'caller_boolean'}}
        compiled.spec.input_schema['properties']['permission']={'type':'boolean'}
        compiled.spec.input_schema['required'].append('permission')
        args={'target':'countertop_1','permission':True}
        before=deepcopy(ctx.binding_store.snapshot_for_node(occ))
        result=system.invocation_compiler.prepare_arguments(compiled,call_name=compiled.spec.name,
            call_id='explicit',arguments=args,occurrence=occ,binding_store=ctx.binding_store,
            evidence_store=ctx.evidence_store,revision=ctx.world_revision)
        assert result.passed,result
        binding=next(b for b in result.binding_updates if b.role=='permission')
        assert binding.source is BindingSource.CALLER_AUTHORIZED
        assert ctx.binding_store.snapshot_for_node(occ)==before
        invalid=system.invocation_compiler.prepare_arguments(compiled,call_name=compiled.spec.name,
            call_id='bad',arguments={**args,'permission':1},occurrence=occ,binding_store=ctx.binding_store,
            evidence_store=ctx.evidence_store,revision=ctx.world_revision)
        assert not invalid.passed and ctx.binding_store.snapshot_for_node(occ)==before
        # Auto entry may consume that exact committed control, not a forged
        # Harness-certified boolean, and cannot invent it when absent.
        ctx.binding_store.commit_grounded(occ.occurrence_id,{'permission':binding})
        validate_committed(compiled.atomic,'permission',binding,evidence_store=ctx.evidence_store,revision=ctx.world_revision)
        with pytest.raises(ValueError):validate_committed(compiled.atomic,'permission',replace(binding,source=BindingSource.HARNESS_EVIDENCE),evidence_store=ctx.evidence_store,revision=ctx.world_revision)
    finally:system.close()


def test_repeat_consumption_requires_the_real_expression_and_output(tmp_path):
    system,ctx,occ,_,_=setup(tmp_path,lambda *a:None)
    try:
        producer=replace(occ,occurrence_id='p',step_id='p')
        ctx.plan.occurrences.insert(0,producer)
        occ.binding_specs['container']=BindingExpression('data_flow',source_step='p',source_role='value')
        ctx.plan.data_edges=[GraphEdge('edge','data_flow','p',occ.step_id,'value','container')]
        original=RuntimeBinding('value','cabinet_1','entity',BindingSource.TOOL_OUTPUT,
            BindingStatus.GROUNDED,BindingResolution.CONCRETE,['observed'],ctx.world_revision)
        ctx.binding_store.publish_validated_outputs('p',{'value':'cabinet_1'},['observed'],ctx.world_revision,certified_bindings={'value':original})
        carried=replace(original,role='container',source=BindingSource.REPEAT)
        ctx.binding_store._set(occ.occurrence_id,carried,'binding_expression')
        assert len(_dataflow_inputs(ctx,occ,{'container':'cabinet_1'}))==1
        ctx.binding_store.commit_grounded(occ.occurrence_id,{'container':carried})
        assert len(_dataflow_inputs(ctx,occ,{'container':'cabinet_1'}))==1
        ctx.binding_store._set(occ.occurrence_id,replace(carried,source=BindingSource.HARNESS_EVIDENCE),'grounding')
        assert not _dataflow_inputs(ctx,occ,{'container':'cabinet_1'})
    finally:system.close()
