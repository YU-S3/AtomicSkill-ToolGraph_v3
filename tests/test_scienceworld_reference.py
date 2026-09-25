from atomic_skillgraph.deployment.scienceworld_reference import reference_atomics
from atomic_skillgraph.harness.scienceworld import ScienceWorldAdapter
from atomic_skillgraph.tooling.validator import ToolStaticValidator


def test_reference_assets_use_production_static_authority():
    harness = ScienceWorldAdapter()
    validator = ToolStaticValidator()
    for atomic, implementation, tool in reference_atomics():
        report = validator.validate_tool_asset(tool, atomic, harness)
        assert report.passed, (atomic.ref, report.failure_codes, report.messages)
        assert not atomic.metadata['historical_execution_claimed']
        assert implementation.abstract_ref == atomic.ref


def test_reference_caller_scopes_are_valid_and_still_bounded():
    import copy
    import pytest
    from atomic_skillgraph.runtime.input_authorization import validate_declarations
    for atomic,_,_ in reference_atomics():
        validate_declarations(atomic)
    atomic=copy.deepcopy(reference_atomics()[17][0])
    atomic.validator_spec['input_authorization']['locations']['max_items']=17
    with pytest.raises(ValueError):validate_declarations(atomic)


def test_reference_fresh_outputs_are_explicit_effect_bindings():
    from atomic_skillgraph.core.bindings import BindingExpression
    for atomic,_,_ in reference_atomics():
        for role,derivation in atomic.validator_spec.get('output_derivations',{}).items():
            if derivation['kind']=='effect_witness' and derivation['argument_role']=='evidence':
                effect=next(e for e in atomic.effects if e.predicate==derivation['predicate'])
                assert isinstance(effect.args['evidence'],BindingExpression)
                assert effect.args['evidence'].source_role==role


def test_partial_workflows_use_metadata_protocol_without_official_goal_claims():
    from atomic_skillgraph.deployment.scienceworld_workflows import reference_workflows
    graphs=reference_workflows()
    assert len(graphs)==18
    for graph in graphs:
        assert graph.metadata['completion_authority']=={'kind':'partial_capability'}
        assert graph.metadata['task_contract_covered'] is False
        assert graph.validator_spec['task_contract_covered'] is False


def test_authored_draft_export_never_claims_frozen_or_execution(tmp_path):
    import json
    import pytest
    from experiments.scienceworld_reference_draft import export
    output = tmp_path/'draft'
    assert len(export(output)) == 30
    checks = json.loads((output/'audit'/'release_checks.json').read_text())
    assert not checks['go'] and not checks['deployable_frozen_bank']
    assert len(list((output/'bank').glob('*/*.json'))) == 108
    assert checks['composites'] == 18 and not checks['blocks_formal_learned_campaign']
    assert not (output/'frozen').exists()
    with pytest.raises(FileExistsError):
        export(output)
