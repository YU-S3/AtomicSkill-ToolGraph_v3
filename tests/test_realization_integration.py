"""Real source -> production Builder/validator/queue; only model choices controlled."""
import json

from atomic_skillgraph.evolution.realization_queue import PREFIX
from test_r1021_i import source_case, author


def test_known_atomic_without_program_realizes_once_per_qualified_source(tmp_path):
    system, ctx, provider, occurrence, normalized = source_case(tmp_path)
    try:
        system.r103 = True
        trace = ctx.trace_builder.trace
        trace.learning_eligible = True
        trace.metadata['execution_source'] = {'split': 'train', 'experiment_kind': 'formal'}
        atomic = system._canonical_atomic_for_occurrence(occurrence)
        system.skills.register_atomic(atomic)
        assert not [i for i in system.skills.implementations() if i.abstract_ref == atomic.ref]
        provider.choose = lambda request, _: author(request)
        count = len(provider.requests)
        for _ in range(2):
            compiled, _ = system._build_tool_for_occurrence(
                occurrence, atomic, normalized, trace, source_task=ctx.task)
            assert compiled is None
        assert len(provider.requests) == count + 1
        rows = system.database.rows('SELECT value FROM metadata WHERE key LIKE ?', (PREFIX + '%',))
        assert len(rows) == 1
        record = json.loads(rows[0]['value'])
        assert record['status'] == 'no_tool'
        assert record['online_execution_credit'] is False
        assert trace.metadata['evolution_tool_builds'][-1]['outcome'] == 'duplicate_evidence_skipped'
        assert not [i for i in system.skills.implementations() if i.abstract_ref == atomic.ref]
    finally:
        system.close()


def test_native_existing_contract_reuse_keeps_source_validation_and_adds_support(tmp_path):
    from dataclasses import replace
    from atomic_skillgraph.core.status import SkillStatus
    from atomic_skillgraph.evolution.extraction_view import CONTRACT_FIELDS, VERSION
    from test_r1021_final import raw_proposal
    system, ctx, provider, occurrence, _, proposal = source_case(tmp_path, include_proposal=True)
    try:
        atomic = replace(system._canonical_atomic_for_occurrence(occurrence), status=SkillStatus.CANDIDATE)
        system.skills.register_atomic(atomic)
        source = {k: v for k, v in raw_proposal(proposal).items() if k not in CONTRACT_FIELDS}
        def choose(request, _):
            if 'canonical_trace' in request.policy_context:
                assert any(c['atomic_ref'] == str(atomic.ref)
                           for c in request.policy_context['known_atomic_contracts'])
                choices = request.policy_context['canonical_trace']['source_reference_choices']['choices']
                for mapping in source['input_provenance_refs'].values():
                    matching = [r for r in choices if r['original_ref'] == mapping['authority_ref']]
                    assert matching
                    mapping['authority_ref'] = matching[0]['reference_id']
                return 'submit_extractor_atomics', {'occurrences': [], 'reuse_existing': [{
                    'version': VERSION, 'atomic_ref': str(atomic.ref),
                    'source': source, 'output_witness_refs': {}}]}
            return author(request, create=True)
        provider.choose = choose
        trace = ctx.trace_builder.trace
        prepared = system._prepare_evolution(trace, ctx.task)
        assert trace.metadata['extraction']['e1_validated'] == 1, trace.metadata['extraction']
        result = system._apply_evolution(prepared, trace, ctx.task)
        assert result['atomic_refs'] and result['tool_refs']
        assert trace.metadata['evolution_tool_builds'][-1]['static_passed']
        # This is source replay/admission, not a fabricated independent online run.
        assert not system.database.rows('SELECT * FROM runtime_support_observations') if system.r103 else True
    finally:
        system.close()
