from atomic_skillgraph.deployment.scienceworld_reference import basic_assets, preparation_assets
from atomic_skillgraph.harness.scienceworld import ScienceWorldAdapter
from atomic_skillgraph.tooling.validator import ToolStaticValidator


def test_reference_assets_use_production_static_authority():
    harness = ScienceWorldAdapter()
    validator = ToolStaticValidator()
    for atomic, implementation, tool in basic_assets() + preparation_assets():
        report = validator.validate_tool_asset(tool, atomic, harness)
        assert report.passed, (atomic.ref, report.failure_codes, report.messages)
        assert not atomic.metadata['historical_execution_claimed']
        assert implementation.abstract_ref == atomic.ref
