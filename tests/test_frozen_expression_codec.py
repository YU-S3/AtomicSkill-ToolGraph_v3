import copy
import pytest
from atomic_skillgraph.agents.runtime_expression_codec import (
    pack_catalog_rows,unpack_catalog_rows,pack_discovery_rows,unpack_discovery_rows)
from atomic_skillgraph.agents.session import _full_catalog_entry_count,_catalog_revision

def test_full_catalog_roundtrip_retains_order_ids_and_types():
    raw={'revision':4,'actions':[{'action_id':f'long_original_id_{i}','action_type':'EXAMINE','arguments':{'object':f'item_{i}'}} for i in range(40)]}
    before=copy.deepcopy(raw);packed,audit=pack_catalog_rows(raw)
    assert audit.applied and audit.roundtrip
    assert unpack_catalog_rows(packed)==raw==before
    assert _full_catalog_entry_count(packed)==40 and _catalog_revision(packed)==4

def test_discoveries_missing_null_and_revision_preserved():
    raw={f'item_{i}':{'last_seen_revision':i,'observed_at_revision':None,'last_known_location':'',
        'public_evidence_ref':f'evidence_{i}','evidence_status':'historical','location_evidence_status':'historical'} for i in range(40)}
    raw['other']={'evidence_status':'observed'}
    packed,audit=pack_discovery_rows(raw)
    assert audit.applied
    assert unpack_discovery_rows(packed)==raw
    assert list(unpack_discovery_rows(packed))==list(raw)

def test_unknown_shape_falls_back_without_dropping_fields():
    raw={'entity':{'unknown_future_field':True}}
    packed,audit=pack_discovery_rows(raw)
    assert packed==raw and not audit.applied and audit.reason=='unsupported_shape'

def test_small_payload_does_not_pay_codec_overhead():
    raw={'revision':0,'actions':[]}
    packed,audit=pack_catalog_rows(raw)
    assert packed==raw and not audit.applied

def test_dynamic_diagnostics_and_selected_summary_restore():
    from atomic_skillgraph.agents.runtime_expression_codec import project_lean,restore_lean
    raw={'task_runtime_frame':{'capability_candidates':[{'atomic_ref':'a','diagnostics':['ranking'],
        'inputs':[{'name':'x'}],'execution_available':True}]},'recent_failed_learned_invocation':{'failure':'keep me'}}
    packed,audit=project_lean(raw,{'a':'portable summary','not_selected':'must not appear'})
    assert 'diagnostics' not in packed['task_runtime_frame']['capability_candidates'][0]
    assert packed['recent_failed_learned_invocation']==raw['recent_failed_learned_invocation']
    assert restore_lean(packed,audit)==raw
    with pytest.raises(ValueError):restore_lean({**packed,'extra':1},audit)

def test_duplicate_candidate_refs_and_existing_summary_roundtrip():
    from atomic_skillgraph.agents.runtime_expression_codec import project_lean,restore_lean
    raw={'support_atomic_candidates':[
        {'atomic_ref':'a','summary':'old','diagnostics':['first'],'inputs':[]},
        {'atomic_ref':'a','diagnostics':['second'],'summary':None},
        {'atomic_ref':'a','summary':'third','inputs':[]}]}
    packed,audit=project_lean(raw,{'a':'selected summary'})
    assert restore_lean(packed,audit)==raw

def test_final_http_audit_observes_packed_payload_once_without_private_data(monkeypatch):
    import json
    from test_provider_recovery import provider,Response,GOOD
    from atomic_skillgraph.agents.runtime_expression_codec import project_lean
    raw={'current_action_catalog':{'revision':2,'actions':[
        {'action_id':f'action_{i}','action_type':'EXAMINE','arguments':{'object':f'item_{i}'}}
        for i in range(40)]}}
    lean,audit=project_lean(raw)
    p,calls=provider(monkeypatch,[Response(GOOD)])
    p.complete([{'role':'user','content':'PUBLIC\nPOLICY_CONTEXT_JSON\n'+json.dumps(lean)}])
    assert len(calls)==len(p.request_records)==1
    final=p.request_records[0]['final_payload_audit']
    assert final['captured_after_build_payload'] and final['policy_contexts']==[lean]
    assert audit['transforms']['catalog']['applied']
    assert 'fixture-key' not in json.dumps(final)
    assert 'reasoning_content' not in json.dumps(final)
