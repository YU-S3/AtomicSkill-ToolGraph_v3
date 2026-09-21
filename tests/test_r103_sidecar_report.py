"""Ordinary E1 proposal conservation does not absorb sidecar Builder work."""
import copy
import pytest
from experiments.tests.test_r4_learning_report import _r4_trace
from experiments.report import trace_to_row
from atomic_skillgraph.system import AtomicSkillGraphSystem
from types import SimpleNamespace


def test_sidecar_record_count_is_separate_but_all_builder_work_stays_counted():
    trace=_r4_trace()
    meta=trace['metadata']
    extra=copy.deepcopy(meta['evolution_tool_builds'][0])
    extra.update(occurrence_id='sidecar',phase_id='sidecar',source_scope='generalization_sidecar')
    meta['evolution_tool_builds'].append(extra)
    meta['tool_build_rejections'].append(copy.deepcopy(meta['tool_build_rejections'][0]))
    meta['generalization']={'source_proof':{'kind':'controlled_verified_source_proof'},'builder_record_indices':[1]}
    AtomicSkillGraphSystem._finalize_r4_learning_metrics(SimpleNamespace(metadata=meta))
    row=trace_to_row(trace)
    assert row['extractor_e1_validated_occurrence_count']==1
    assert row['tool_builder_call_count']==2
    for mutation in ('missing_index','missing_source_proof','fake_tag'):
        bad=copy.deepcopy(trace)
        if mutation=='missing_index': bad['metadata']['generalization']['builder_record_indices']=[]
        elif mutation=='missing_source_proof': bad['metadata']['generalization']['source_proof']={}
        else: bad['metadata']['evolution_tool_builds'][1].pop('source_scope')
        with pytest.raises(ValueError,match='sidecar builder'):
            trace_to_row(bad)
