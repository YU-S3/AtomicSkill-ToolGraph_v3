"""Plan B bounded production execution; model transport is intercepted throughout."""
from copy import deepcopy
from pathlib import Path

import pytest

from atomic_skillgraph.core.errors import BudgetExhausted
from atomic_skillgraph.empirical.budget_governor import BudgetGovernor,input_token_bound
from atomic_skillgraph.empirical.contracts import digest
from atomic_skillgraph.experiments import capability_schedule as schedule
from atomic_skillgraph.experiments import run_choice_capability_batch as batch
from atomic_skillgraph.experiments import run_derived_choice as derived
from atomic_skillgraph.experiments.formal_choice_batch import BatchGovernor
from test_capability_batch import fixture_transport
from test_capability_schedule import prepare_fixture


def test_lossless_p2_tables_preserve_tasks_observations_offsets_and_utf8():
    from atomic_skillgraph.empirical.capability_batch import records
    from atomic_skillgraph.empirical.model_view import expand_material,pack_capability_material,reference_exact_span_quotes
    from test_capability_batch import sources
    rows=records(sources());material={'train_cases':rows,'guidance':'𝕔'*256,'required_conditions':'𝕒'*256,
        'references':[{'case_id':r['task']['physical_key'],'span_id':r['spans'][0]['span_id']} for r in rows]}
    before=deepcopy(material);packed=pack_capability_material(material)
    assert material==before
    assert expand_material(packed)==reference_exact_span_quotes(material)
    assert expand_material(pack_capability_material(packed))==reference_exact_span_quotes(material)
    assert expand_material({'ordinary':{'p2_text_ref':'not-an-internal-reference'}})=={'ordinary':{'p2_text_ref':'not-an-internal-reference'}}
    for row in expand_material(packed)['train_cases']:
        for span in row['spans']:
            text=row['task']
            for key in span['task_path']:text=text[key]
            assert text[span['start']:span['end']]


def test_full_wire_retry_and_budget_envelope(tmp_path,monkeypatch):
    source,root,receipt,_,before=prepare_fixture(tmp_path,monkeypatch,full=True)
    cfg=batch.read(root/'p2/config.json');plan=batch.read(root/'p2/plan.json')
    bound=schedule.request_bound(cfg,plan['settings'])
    assert bound['p2_token_upper']==9974784 and bound['p2_physical_http_upper']==576
    assert plan['unit_limits']=={'token_limit':277504,'request_limit':8}
    assert cfg['learning']['capability_payload_max_bytes']==32768 and cfg['learning']['extractor_payload_max_bytes']==16384
    assert cfg['learning']['source_program_payload_max_bytes']==24576 and cfg['llm']['extractor']['max_total_tokens_per_task']==277504
    assert all(cfg['llm']['purpose_overrides'][p]['max_retries']==1 for p in
        ('capability_induction','guidance_learning','guidance_grounding','capability_routing'))
    assert batch.read(root/'budget_revision.json')['physical_attempts_unchanged']
    assert plan['cumulative_limits']=={'token_limit':31000000,'request_limit':9165,'finish_reserve':0}
    assert plan['new_limits']=={'token_limit':10000000,'request_limit':600} and plan['prior_phase_http']==0
    old=batch.read(source/'train/config.json')['llm']
    for evaluation in batch.read(root/'derivation_manifest.json')['configs'].values():
        assert evaluation['llm']['max_retries']==old['max_retries']==4
        assert evaluation['llm']['runtime']==old['runtime']
    assert before=={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}


def test_all_applicable_two_candidates_complete_frozen_and_no_automatic_test(tmp_path,monkeypatch):
    calls=fixture_transport(monkeypatch)
    source,root,_,env,before=prepare_fixture(tmp_path,monkeypatch,full=True)
    result=derived.execute(root,env)
    assert result['status']=='awaiting_test_review' and result['val']['tasks']==17
    p2=batch.read(root/'p2/result.json')
    assert p2['status']=='completed_selection' and p2['attempted_units']==12 and len(p2['seen_sources'])==48
    assert p2['source_checked_candidate_count']==12 and p2['selected_for_routing_count']==2
    assert p2['paired_tasks']==24 and all(c['passed'] and c['actual_exposed']==12 for c in p2['candidates'])
    traces=list((root/'p2/candidates').glob('*/selection/*/*/trace.json'))
    assert len(traces)==48 and (root/'train/frozen_bank/freeze.json').exists() and not (root/'test').exists()
    assert all(batch.read(p.parent/'requests.json')['requests'][0]['effective_call_settings']['max_retries']==0 for p in traces)
    n=len(calls);derived.execute(root,env);assert len(calls)==n
    assert before=={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}


def test_authorized_old_reservation_keeps_charge_and_new_unknown_stops(tmp_path,monkeypatch):
    calls=fixture_transport(monkeypatch,no_candidate=True)
    source,root,receipt,env,before=prepare_fixture(tmp_path,monkeypatch,full=True,unresolved=True,recovery=True)
    prefix=batch.read(root/'budget.json');entry=deepcopy(prefix['attempts']['unfinished'])
    assert receipt['billing_ready'] and receipt['inherited_reserved_tokens']==entry['reserved_tokens']
    result=derived.execute(root,env)['p2']
    assert result['new_tokens']==60 and result['new_http']==12 and result['inherited_unknown_billing']
    assert result['cumulative_tokens']==60
    assert result['cumulative_reserved_or_accounted_tokens']==60+entry['reserved_tokens']
    ledger=batch.read(root/'budget.json');assert ledger['unknown_billing'] and ledger['attempts']['unfinished']==entry
    assert before=={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    governor=BudgetGovernor(root/'budget.json',**ledger['limits'])
    governor.admit('new-unknown',{'messages':[],'max_tokens':1},{'parent_scope':'new-run'})
    governor.complete({'request_id':'new-unknown','outcome':'connection_error','raw_usage':None})
    assert governor.billing_blocked and not derived.check(root)['billing_ready']
    with pytest.raises(BudgetExhausted):governor.admit('another',{'messages':[],'max_tokens':1},{})
    assert len(calls)==12
    tampered=deepcopy(ledger);tampered['attempts']['unfinished']['reserved_tokens']+=1
    batch.write_json(root/'budget.json',tampered)
    with pytest.raises(ValueError,match='identity changed'):BudgetGovernor(root/'budget.json',**ledger['limits'])


def test_required_capacity_refusal_is_not_no_candidate(tmp_path,monkeypatch):
    calls=fixture_transport(monkeypatch,no_candidate=True)
    _,root,_,env,_=prepare_fixture(tmp_path,monkeypatch,full=True)
    from atomic_skillgraph.empirical.learner import Learner
    original=Learner.learn_capability_batch;seen=[]
    def learn(self,*args,**kwargs):
        seen.append(args)
        if len(seen)==1:
            exc=ValueError('Necessary source material exceeds 32 KiB');exc.code='material_too_large';raise exc
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Learner,'learn_capability_batch',learn)
    result=derived.execute(root,env)
    assert result['status']=='stopped_material_capacity' and result['completed_units']==12
    assert result['learning_units'][0]['decision']=='capacity_rejected' and len(calls)==11
    assert not (root/'train/frozen_bank').exists() and not (root/'val').exists()


def test_full_576_http_nested_stage_and_unit_boundary_survive_restore(tmp_path):
    owner=BudgetGovernor(tmp_path/'campaign.json',token_limit=31000000,request_limit=9165,finish_reserve=0)
    batch.write_json(tmp_path/'campaign_budget_prefix.json',owner.state)
    plan={'run_id':'bounded','new_limits':{'token_limit':10000000,'request_limit':600}}
    phase=BatchGovernor(owner,plan,tmp_path)
    # Twelve units: four logical calls, each with its one permitted HTTP retry.
    for unit in range(12):
        for i in range(8):
            payload={'messages':[],'max_tokens':(34816 if i<6 else 34304)-input_token_bound({'messages':[],'max_tokens':34816})}
            cost=input_token_bound(payload)+payload['max_tokens']
            context={'parent_scope':'bounded:unit:'+str(unit),'parent_token_limit':277504,
                'episode_scope':str(unit),'episode_request_limit':8}
            key=f'g-{unit}-{i}';phase.admit(key,payload,context)
            phase.complete({'request_id':key,'outcome':'success','raw_usage':{'total_tokens':cost}})
        with pytest.raises(BudgetExhausted):phase.admit('ninth-'+str(unit),{'messages':[],'max_tokens':1},context)
    for i in range(48):
        payload={'messages':[],'max_tokens':18432-input_token_bound({'messages':[],'max_tokens':18432})}
        cost=input_token_bound(payload)+payload['max_tokens'];key='route-'+str(i)
        phase.admit(key,payload,{'parent_scope':'bounded:route:'+str(i)})
        phase.complete({'request_id':key,'outcome':'success','raw_usage':{'total_tokens':cost}})
    for arm in range(48):
        for i in range(9):
            payload={'messages':[],'max_tokens':13000};key=f'pair-{arm}-{i}'
            context={'parent_scope':'bounded:arm:'+str(arm),'budget_pool':'solve','budget_pool_limit':120000}
            phase.admit(key,payload,context)
            phase.complete({'request_id':key,'outcome':'success','raw_usage':{'total_tokens':13333 if i<8 else 13336}})
    assert len(phase.attempts())==576 and phase.headroom({})['tokens']>=25216
    restored=BatchGovernor(BudgetGovernor(owner.path,**owner.limits),plan,tmp_path)
    assert restored.headroom({})==phase.headroom({})
    for i in range(24):
        key='extra-'+str(i);restored.admit(key,{'messages':[],'max_tokens':1},{'parent_scope':'bounded:extra'})
        restored.complete({'request_id':key,'outcome':'success','raw_usage':{'total_tokens':1}})
    with pytest.raises(BudgetExhausted):restored.admit('601',{'messages':[],'max_tokens':1},{'parent_scope':'bounded:extra'})
