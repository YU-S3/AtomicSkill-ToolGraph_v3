"""Model fields stay literal; only Host-produced material references are decoded."""
from copy import deepcopy
import json
import pytest

from atomic_skillgraph.empirical.model_view import pack_material,expand_material
from atomic_skillgraph.empirical.prompts import GUIDANCE_LEARNING,GUIDANCE_LEARNER_PROMPT
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.harness.benchmarks import AnswerAdapter
from test_cf2_contracts import http,response
from test_empirical import config_for


def test_literal_refs_cannot_capture_a_host_contract_table():
    contract={'input_schema':{'type':'object'},'output_schema':{'type':'object'}}
    material={'proposal':{'guidance_skill':{'contract_ref':'c0'},'job':{'job_ref':'j0'}},
        'related':[deepcopy(contract),deepcopy(contract)]}
    assert expand_material(material)==material
    assert expand_material(pack_material(material))==material
    ordinary={'related':material['related']}
    packed=pack_material(ordinary)
    assert packed['material_tables']['contracts']
    assert expand_material(packed)==ordinary
    packed['related'][0]['contract_ref']='missing'
    with pytest.raises(KeyError):expand_material(packed)


@pytest.mark.parametrize('valid_repair',[True,False])
def test_invalid_guidance_reference_uses_only_existing_repair(tmp_path,monkeypatch,valid_repair):
    invalid={'decision':'upsert_guidance','guidance_skill':{'goal':'Read a labeled field',
        'guidance':'Read the value adjacent to the requested label.','contract_ref':'c0'}}
    fixed=deepcopy(invalid);fixed['guidance_skill'].pop('contract_ref')
    sent=http(monkeypatch,[response([invalid],name='submit_learning'),
        response([fixed if valid_repair else invalid],name='submit_learning')])
    s=EmpiricalSystem(config_for(tmp_path/'bank'),harness=AnswerAdapter('searchqa',{}))
    try:
        def call():return s.agent('extractor',GUIDANCE_LEARNER_PROMPT,{},'submit_learning',GUIDANCE_LEARNING,
            repair_limit=1,decision_purpose='guidance_learning')
        if valid_repair:assert call()==fixed
        else:
            with pytest.raises(ValueError) as failure:call()
            assert failure.value.model_authored
        assert len(sent)==2
        repair=json.loads(sent[1]['messages'][1]['content'])
        assert repair['proposal']['guidance_skill']==invalid['guidance_skill']
        assert [r['purpose'] for r in s.requests]==['guidance_learning','learning_structure_repair']
    finally:s.close()
