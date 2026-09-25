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


def test_authored_draft_export_never_claims_frozen_or_execution(tmp_path):
    import json
    import pytest
    from experiments.scienceworld_reference_draft import export
    output = tmp_path/'draft'
    assert len(export(output)) == 30
    checks = json.loads((output/'audit'/'release_checks.json').read_text())
    assert not checks['go'] and not checks['deployable_frozen_bank']
    assert len(list((output/'bank').glob('*/*.json'))) == 90
    assert not (output/'frozen').exists()
    with pytest.raises(FileExistsError):
        export(output)
