import copy
from dataclasses import replace
from pathlib import Path

import pytest

from atomic_skillgraph.core.contracts import ParameterSpec, SemanticPredicate
from atomic_skillgraph.core.bindings import BindingExpression, BindingExprKind
from atomic_skillgraph.system import load_config
from atomic_skillgraph.tooling.validator import ToolStaticValidator
from experiments.protocol import ProtocolError, capture_execution_manifest, code_file_manifest, hash_code, validate_deepseek_formal_llm


@pytest.mark.parametrize('expression', ['$output_001', BindingExpression(BindingExprKind.SKILL_INPUT, source_role='output_001')])
def test_E01_E04_formal_numbered_output_is_not_an_episode_constant(expression):
    from test_r92_runtime_automation_interface import _context, _future_output_draft, FakeAgentFactory
    _, ctx, occurrence, _ = _context(FakeAgentFactory())
    draft = replace(_future_output_draft(occurrence.occurrence_id),
        outputs=[ParameterSpec('output_001', 'entity')],
        effects=[SemanticPredicate('agent.holds', {'object': expression})])
    report = ToolStaticValidator().validate_automation_draft(draft, ctx.harness, ctx=ctx, occurrence=occurrence)
    assert report.passed, report
    bad = replace(draft, effects=[SemanticPredicate('agent.holds', {'object': '$unknown'})])
    rejected = ToolStaticValidator().validate_automation_draft(bad, ctx.harness, ctx=ctx, occurrence=occurrence)
    assert 'runtime_automation_r0_role_closure' in rejected.failure_codes
    leaked = replace(draft, effects=[SemanticPredicate('agent.holds', {'object': 'cabinet_1'})])
    assert 'runtime_automation_r0_episode_concrete_id' in ToolStaticValidator().validate_automation_draft(
        leaked, ctx.harness, ctx=ctx, occurrence=occurrence).failure_codes


def test_E05_persistent_support_policy_view_does_not_mutate_provenance():
    from test_r101_boundaries import boundary
    from atomic_skillgraph.agents.portable_support_view import portable_support_view
    from atomic_skillgraph.core.serialization import to_primitive
    request, _, _ = boundary()
    atomic = request.producer
    atomic.summary = 'Find soapbar in cabinet_1'
    atomic.inputs[0].description = 'soapbar instance'
    atomic.guideline = {'steps': ['go cabinet_1']}
    atomic.metadata = {'runtime_support_promotion': True, 'source_task': 'soapbar provenance'}
    before = copy.deepcopy(to_primitive(atomic))
    view = portable_support_view(atomic)
    assert 'soapbar' not in view.summary and 'cabinet_1' not in view.summary
    assert all(not item.description for item in view.inputs + view.outputs)
    assert view.effects == atomic.effects and view.validator_spec == atomic.validator_spec
    assert view.metadata == atomic.metadata
    assert to_primitive(atomic) == before


@pytest.mark.parametrize('phase', ['train', 'frozen'])
def test_F04_r101_is_explicitly_registered_and_budget_guarded(phase):
    from experiments.run_v3_train import _train_protocol, _validate_formal_config as train_guard
    from experiments.run_v3_frozen_eval import _frozen_protocol, _validate_formal_config as frozen_guard
    root = Path(__file__).resolve().parents[1]
    name, protocol, guard, expected = (
        ('alfworld_train_full_120_r101_seed42', _train_protocol, train_guard, ('r101_full120', 42, 20, 120))
        if phase == 'train' else
        ('alfworld_frozen_eval_134_r101_seed42', _frozen_protocol, frozen_guard, ('r101_frozen134', 42, 0, 134)))
    config = load_config(root / 'configs' / (name + '.yaml'))
    assert protocol(config) == expected
    guard(config, root / config['experiment']['output_dir'])
    for field, value in [('max_total_tokens_per_node', 100001), ('max_total_tokens_per_task', 300001), ('reasoning_effort', 'low')]:
        changed = copy.deepcopy(config)
        changed['llm']['runtime'][field] = value
        with pytest.raises(ProtocolError):
            validate_deepseek_formal_llm(changed)
    changed = copy.deepcopy(config)
    changed['runtime']['short_runtime_steps'] = False
    with pytest.raises(ProtocolError):
        validate_deepseek_formal_llm(changed)


def test_F05_manifest_detects_changed_and_added_files(tmp_path):
    code = tmp_path / 'code'
    code.mkdir()
    (code / 'module.py').write_text('original\n')
    output = tmp_path / 'run'
    output.mkdir()
    capability = {'code_hash': hash_code(code), 'config_hash': 'configuration', 'passed': True}
    recorded = capture_execution_manifest(code, output, 'configuration', capability)
    assert recorded['files'] == code_file_manifest(code)
    assert capture_execution_manifest(code, output, 'configuration', capability) == recorded
    (code / 'additional.py').write_text('new\n')
    with pytest.raises(ProtocolError, match='provider capability'):
        capture_execution_manifest(code, output, 'configuration', capability)
    capability['code_hash'] = hash_code(code)
    with pytest.raises(ProtocolError, match='per-file manifest'):
        capture_execution_manifest(code, output, 'configuration', capability)


def test_F01_official_success_never_uses_contract_or_old_strict_veto():
    from experiments.report import trace_to_row
    row = trace_to_row({'trace_id': 'won_only', 'task': {'task_id': 'won_only'},
        'benchmark_success': True, 'task_contract_success': False, 'strict_task_success': False})
    assert row['benchmark_success'] and row['strict_task_success']
    assert not row['task_contract_success']
