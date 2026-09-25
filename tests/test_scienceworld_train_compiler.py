import json
from pathlib import Path
import pytest
import yaml

from atomic_skillgraph.core.serialization import atomic_write_json
from atomic_skillgraph.deployment.train_bank_compiler import compile_train,bank_digest,clone_bank
from atomic_skillgraph.knowledge.database import StateDatabase
from atomic_skillgraph.knowledge.artifact_store import ArtifactStore
from atomic_skillgraph.knowledge.skill_registry import SkillRegistry
from atomic_skillgraph.deployment.preferences import verify_preferences
from atomic_skillgraph.traces.schema import TraceRecord,TaskRecord
from atomic_skillgraph.traces.store import TraceStore
from experiments.protocol import RunManifest,RunState,TaskManifest,ManifestStore,hash_config
from experiments.scienceworld_manifest import load


def source(tmp_path, with_routes=False):
    run=tmp_path/'seed42'/'train';bank=run/'data_v3'
    manifest_path=Path(__file__).resolve().parents[1]/'data/scienceworld_manifests/train_120.json'
    selection=load(manifest_path);row=selection['tasks'][0]
    config={'schema_version':3,'repair_revision':'R10.3','data_dir':str(bank),'trace_data_dir':str(run),'harness':{'adapter':'scienceworld_v1','manifest':str(manifest_path)},
        'experiment':{'name':'compiler_unit_train','phase':'train','benchmark':'scienceworld','seed':42,'output_dir':str(run)}}
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(config))
    with StateDatabase(bank/'state.sqlite3',r103=True) as db:
        artifacts=ArtifactStore(bank,db)
        if with_routes:
            from dataclasses import replace
            from atomic_skillgraph.deployment.scienceworld_reference import bounded_wait
            from atomic_skillgraph.core.refs import ToolRef,SkillRef
            from atomic_skillgraph.core.bindings import ToolBinding
            from atomic_skillgraph.core.status import SkillStatus,ToolStatus
            from atomic_skillgraph.knowledge.tool_registry import ToolRegistry
            a,i,t=bounded_wait()
            # Unit fixture only; not an authored asset accepted as Train input.
            a=replace(a,status=SkillStatus.ACTIVE,metadata={'unit_fixture':True})
            t=replace(t,status=ToolStatus.ACTIVE,metadata={'unit_fixture':True})
            i=replace(i,status=SkillStatus.ACTIVE,metadata={'unit_fixture':True})
            skills=SkillRegistry(artifacts,db);tools=ToolRegistry(artifacts,db)
            skills.register_atomic(a);tools.register(t);skills.register_implementation(i)
            t2=replace(t,ref=ToolRef('z_duplicate_program','1.0.0'))
            i2=replace(i,ref=SkillRef('z_duplicate_implementation','1.0.0'),
                tool_bindings=[replace(i.tool_bindings[0],tool_ref=t2.ref)])
            tools.register(t2);skills.register_implementation(i2)
        digest=bank_digest(bank)
        task=TaskManifest(0,row['task_id'],row['task_signature'],digest,'scienceworld','train',json.dumps(row))
        manifest=RunManifest.create(run_id='compiler_unit_train',phase='train',config_hash=hash_config(path),
            code_commit='unit_fixture',knowledge_digest=digest,tasks=[task],metadata={'seed':42,'reference_manifest_digest':selection['digest']})
        store=ManifestStore(run.parent,db);store.persist_before_run(manifest)
        traces=TraceStore(run)
        trace=TraceRecord.create(TaskRecord(task.task_id,'scienceworld','unit fixture','1-1',task.task_signature),{},{},{})
        trace.resource_usage_complete=True
        traces.save(trace)
        store.mark_task_running(manifest.run_id,task.task_id,max_attempts=1)
        store.mark_task_completed(manifest.run_id,task.task_id,trace_id=trace.trace_id,result={
            'final_batch_maintenance':{'pending_count':0,'knowledge_digest_after':digest,'maintenance_trace_id':trace.trace_id}})
        store.mark_run_state(manifest.run_id,RunState.COMPLETED)
        atomic_write_json(run/'summary.json',{'knowledge_digest':digest,'completed':1,'expected':1})
    return path,bank


def test_compiler_empty_mini_chain_preserves_raw_and_verifies_frozen(tmp_path):
    config,raw=source(tmp_path)
    before=bank_digest(raw)
    result=compile_train(config,tmp_path/'compiled')
    assert result['completed'] and result['source_train_bank_digest']==before
    assert bank_digest(raw)==before
    bank=tmp_path/'compiled/data_v3'
    assert bank_digest(bank)==result['compiled_bank_digest']
    with StateDatabase(bank/'state.sqlite3',readonly=True,r103=True) as db:
        verify_preferences(SkillRegistry(ArtifactStore(bank,db),db))
    assert (tmp_path/'compiled/workflow_closure.jsonl').is_file()
    with pytest.raises(FileExistsError):compile_train(config,tmp_path/'compiled')


def test_compiler_nonempty_routes_exact_aliases_without_credit_changes(tmp_path):
    config,raw=source(tmp_path,with_routes=True)
    digest=bank_digest(raw)
    report=compile_train(config,tmp_path/'compiled')
    assert report['routes']==2 and bank_digest(raw)==digest
    bank=tmp_path/'compiled/data_v3'
    prefs=json.loads((bank/'deployment_preferences.json').read_text())
    assert len(prefs['canonical_tool_aliases'])==len(prefs['canonical_implementation_aliases'])==1
    routes=[json.loads(line) for line in (tmp_path/'compiled/effective_routes.jsonl').read_text().splitlines()]
    assert all(r['frozen_executable'] for r in routes)
    with StateDatabase(bank/'state.sqlite3',readonly=True,r103=True) as db:
        verify_preferences(SkillRegistry(ArtifactStore(bank,db),db))
        assert db.execute('SELECT count(*) FROM release_deployments').fetchone()[0]==5


@pytest.mark.parametrize('damage',['phase','seed','state','digest','maintenance'])
def test_compiler_refuses_nontrain_incomplete_or_changed_source(tmp_path,damage):
    config,bank=source(tmp_path)
    if damage in ('phase','seed'):
        value=yaml.safe_load(config.read_text());value['experiment'][damage]='test' if damage=='phase' else 43
        config.write_text(yaml.safe_dump(value))
    elif damage=='digest':
        value=json.loads((bank.parent/'summary.json').read_text());value['knowledge_digest']='bad'
        atomic_write_json(bank.parent/'summary.json',value)
    else:
        with StateDatabase(bank/'state.sqlite3',r103=True) as db:
            if damage=='state':db.execute("UPDATE run_manifests SET state='running'")
            else:db.execute("UPDATE run_tasks SET result_json='{}'")
            db.connection.commit()
    with pytest.raises(ValueError):compile_train(config,tmp_path/'compiled')
    assert not (tmp_path/'compiled').exists()


def test_clone_refuses_nested_or_existing_target(tmp_path):
    _,bank=source(tmp_path)
    with pytest.raises(ValueError):clone_bank(bank,bank/'nested')
    with pytest.raises(ValueError):clone_bank(bank,bank)


def test_scienceworld_task_audit_closes_common_report_boundary(tmp_path):
    from experiments.run_scienceworld import artifact_report_fields
    from experiments.protocol import artifact_audit_snapshot, load_task_report_traces
    _,bank=source(tmp_path)
    with StateDatabase(bank/'state.sqlite3',r103=True) as db:
        snapshot=artifact_audit_snapshot(db)
        row=db.execute('SELECT result_json FROM run_tasks').fetchone()
        result=json.loads(row[0])
        result.update(artifact_report_fields(snapshot,snapshot))
        assert 'artifact_lifecycle_after' not in result
        db.execute('UPDATE run_tasks SET result_json=?',(json.dumps(result),))
        traces=load_task_report_traces(TraceStore(bank.parent,readonly=True),db,'compiler_unit_train')
        assert len(traces)==1
        assert traces[0]['metadata']['artifact_lifecycle']==snapshot
