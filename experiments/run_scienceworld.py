"""ScienceWorld production entry: existing System, attempt ledger and checkpoints.

No alternate planner/executor and no task-type policy. Formal evaluation never
writes into its source bank. An interrupted train uses the pre-task checkpoint;
failed paid attempts remain in the immutable accounting ledger.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from atomic_skillgraph.system import AtomicSkillGraphSystem, load_config
from atomic_skillgraph.knowledge.database import StateDatabase
from atomic_skillgraph.core.serialization import atomic_write_json
from atomic_skillgraph.agents.provider_probe import ensure_provider_capability
from .benchmark_protocol import BenchmarkExperimentProtocol
from .protocol import (AttemptTraceLedger, TaskCheckpointStore, ManifestStore, RunManifest, RunState,
    TaskManifest, hash_config, hash_code, ensure_task_manifest, audit_failed_attempt,
    load_task_report_traces, validate_deepseek_formal_llm)
from .protocol import artifact_audit_snapshot, artifact_growth_audit
from .report import validate_formal_usage, validate_usage_event_persistence, write_reports

ROOT = Path(__file__).resolve().parents[1]

def artifact_report_fields(before, after):
    """Use the common immutable-Trace report overlay protocol verbatim."""
    return {'artifact_growth': artifact_growth_audit(before, after),
            'artifact_lifecycle': after}

def run(config_path, *, resume=False, stop_file=None, stop_after_tasks=None):
    started = time.monotonic()
    returned = 0
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    validate_deepseek_formal_llm(config)
    experiment = config['experiment']
    output = Path(experiment['output_dir']).resolve()
    if Path(config['trace_data_dir']).resolve() != output:
        raise ValueError('Trace root must be the episode output root')
    run_id, phase = experiment['name'], experiment['phase']
    online = phase == 'train'
    if experiment['runtime_mode'] != ('online' if online else 'frozen'):
        raise ValueError('Phase/runtime mode mismatch')
    if not output.is_relative_to(Path.home()) or str(output).startswith('/mnt/'):
        raise ValueError('ScienceWorld output must be on the WSL Linux home filesystem')
    protocol = BenchmarkExperimentProtocol.scienceworld(config['harness']['manifest'])
    if protocol.manifest['split'] != phase:
        raise ValueError('Phase/source split mismatch')
    diagnostic = experiment.get('experiment_kind') == 'diagnostic'
    entries = protocol.selected(diagnostic_macros=diagnostic and online)
    if not resume and ((output / 'run_state.sqlite3').exists() or
                       (output / 'attempt_history').exists() or
                       (online and (Path(config['data_dir']) / 'state.sqlite3').exists())):
        raise FileExistsError(f'Run already exists; use --resume: {output}')
    output.mkdir(parents=True, exist_ok=True)
    config_hash, code_hash = hash_config(config_path), hash_code(ROOT)
    from .scienceworld_recovery import read as read_recovery
    recovery = read_recovery(config, config_hash, code_hash) if resume else None
    execution_code_hash = code_hash
    if recovery:
        config['execution_provenance'] = recovery
        code_hash = recovery['original_code_hash']
    atomic_write_json(output / 'actual_config.json', config)
    ensure_provider_capability(config, output_dir=output/'recovery_provider_probe' if recovery else output,
                              config_hash=config_hash, code_hash=execution_code_hash,
                              run_if_missing=not resume or bool(recovery))
    ledger = AttemptTraceLedger(output / 'attempt_history', output / 'traces')
    if resume:
        ledger.recover_pending(run_id=run_id)
        if ledger.unresolved(run_id=run_id):
            raise RuntimeError('Uncaptured interrupted usage; preserve this run for recovery, do not silently retry')
    checkpoint = TaskCheckpointStore(output / '.task_checkpoint', Path(config['data_dir']).resolve()) if online else None
    if checkpoint:
        checkpoint.recover_if_present(run_id=run_id, config_hash=config_hash, code_commit=code_hash, resume=resume)
    with AtomicSkillGraphSystem(config, readonly=not online) as system:
        preflight = system.preflight(require_api_key=True, initialize_harness=True, require_empty_bank=online and not resume)
        atomic_write_json(output / 'preflight.json', preflight)
        if not preflight['passed']:
            raise RuntimeError(f'Preflight failed: {preflight}')
        db = system.database if online else StateDatabase(output / 'run_state.sqlite3')
        store = ManifestStore(output.parent, db)
        try:
            before = system.knowledge_digest()
            if not online:
                frozen_manifest = Path(config['data_dir']) / 'freeze_manifest.json'
                if not frozen_manifest.is_file():
                    raise ValueError('Readonly evaluation requires an authenticated frozen snapshot')
                frozen_metadata=json.loads(frozen_manifest.read_text())
                if (frozen_metadata.get('experiment_kind')=='authored_reference') != (experiment.get('experiment_kind')=='authored_reference'):
                    raise ValueError('Authored reference and learned evaluation identities must not be mixed')
                if frozen_metadata['knowledge_digest'] != before:
                    raise ValueError('Frozen manifest digest does not match actual bank')
                from atomic_skillgraph.deployment.preferences import verify_preferences
                verify_preferences(system.skills)
            initial = store.load(run_id).knowledge_digest if resume else before
            items = [TaskManifest(i, row['task_id'], row['task_signature'], initial,
                'scienceworld', phase, json.dumps(row)) for i, row in enumerate(entries)]
            if resume:
                manifest = store.validate_resume(run_id, config_hash=config_hash, code_commit=code_hash,
                                                knowledge_digest=initial, tasks=items)
                if online:
                    from .run_v3_train import _verify_resume_knowledge
                    _verify_resume_knowledge(store, manifest, before)
            else:
                initial_artifacts = artifact_audit_snapshot(system.database)
                manifest = RunManifest.create(run_id=run_id, phase=phase, config_hash=config_hash,
                    code_commit=code_hash, knowledge_digest=initial, tasks=items,
                    metadata={'environment': protocol.manifest['resource_identity'], 'seed': experiment['seed'],
                        'benchmark':'scienceworld', 'adapter':'scienceworld_v1',
                        'reference_manifest_digest': protocol.manifest['digest'],
                        'run_started_at': datetime.now(timezone.utc).isoformat(),
                        'experiment_kind': experiment.get('experiment_kind','formal'),
                        'initial_artifact_snapshot': initial_artifacts,
                        'initial_artifact_snapshot_digest': initial_artifacts['snapshot_digest'],
                        'final_batch_maintenance_milestone': 'scienceworld_train_final_batch'})
                store.persist_before_run(manifest)
            ensure_task_manifest(experiment['task_manifest_path'], manifest)
            store.mark_run_state(run_id, RunState.RUNNING)
            by_id = {e['task_id']: e for e in entries}
            for item in store.tasks_to_run(manifest):
                if ((stop_file and Path(stop_file).exists()) or
                    (stop_after_tasks is not None and returned >= stop_after_tasks)):
                    store.mark_run_state(run_id, RunState.PENDING)
                    atomic_write_json(output / 'progress.json', {'state':'paused_at_task_boundary',
                        'next_task_id':item.task_id,'total':len(items)})
                    return 75
                atomic_write_json(output / 'progress.json', {'state': 'running', 'task_id': item.task_id,
                    'ordinal': item.ordinal, 'total': len(items), 'updated_at': datetime.now(timezone.utc).isoformat()})
                print(json.dumps({'starting_task': item.task_id, 'ordinal': item.ordinal + 1, 'total': len(items)}), flush=True)
                task = protocol.task(by_id[item.task_id], system.harness)
                before = system.knowledge_digest()
                artifacts_before = artifact_audit_snapshot(system.database)
                sequence = store.mark_task_running(run_id, item.task_id, max_attempts=experiment.get('max_task_attempts', 3))
                periodic = system.expected_periodic_maintenance_milestone_after_success() if online else ''
                attempt = ledger.begin(run_id=run_id, task_id=item.task_id, task_signature=item.task_signature,
                    attempt_kind='task', sequence=sequence, expected_periodic_milestone=periodic)
                try:
                    if checkpoint:
                        checkpoint.create(db, run_id=run_id, task_id=item.task_id, before_digest=before,
                                          config_hash=config_hash, code_commit=code_hash)
                    trace = system.run_task(task, attempt_id=attempt.attempt_id)
                    capture = ledger.capture(attempt, reason='run_task_returned')
                    result = {**protocol.result(trace), 'knowledge_digest_before': before,
                        'knowledge_digest_after': system.knowledge_digest(), 'attempt_capture': capture}
                    artifacts_after = artifact_audit_snapshot(system.database)
                    result.update(artifact_report_fields(artifacts_before, artifacts_after))
                    if not online and result['knowledge_digest_after'] != before:
                        raise RuntimeError('Frozen evaluation mutated its source bank')
                    if trace.infrastructure_failure:
                        raise RuntimeError(f'Infrastructure failure: {trace.trace_id}')
                    store.mark_task_completed(run_id, item.task_id, trace_id=trace.trace_id, result=result)
                    returned += 1
                    if checkpoint:
                        checkpoint.clear()
                    print(json.dumps({'completed_task': item.task_id, **protocol.result(trace)}), flush=True)
                except Exception as exc:
                    def failed():
                        store.mark_task_failed(run_id, item.task_id, infrastructure=True,
                            result={'error_type': type(exc).__name__, 'error': str(exc)})
                        store.mark_run_state(run_id, RunState.INFRASTRUCTURE_FAILED)
                    audit_failed_attempt(primary=exc, attempt=attempt, attempt_ledger=ledger,
                        receipt_root=output / 'failure_receipts', update_state=failed, capture_reason='task_exception')
                    raise
            if online:
                from .run_v3_train import _run_final_batch_maintenance
                _run_final_batch_maintenance(system, store, checkpoint, manifest, attempt_ledger=ledger,
                    config_digest=config_hash, code_digest=code_hash)
            traces = load_task_report_traces(system.traces, db, run_id)
            other = ledger.auxiliary_traces(manifest=manifest, excluded_trace_ids={t['trace_id'] for t in traces})
            from .scienceworld_recovery import recovery_usage_traces
            other += recovery_usage_traces(output, recovery)
            validate_formal_usage([*traces, *other])
            coverage = validate_usage_event_persistence(system.usage.events, [*traces, *other])
            from .scienceworld_report import write_scienceworld_reports
            score_report = write_scienceworld_reports(traces, output, auxiliary=other, recovery=recovery)
            results = [json.loads(r['result_json']) for r in db.rows(
                'SELECT result_json FROM run_tasks WHERE run_id=? ORDER BY rowid', (run_id,))]
            summary = {'benchmark': 'scienceworld', 'phase': phase, 'seed': experiment['seed'],
                'completed': len(results), 'expected': len(items), 'mean_official_score':
                    sum(r['official_score'] for r in results) / len(results),
                'perfect_success_rate': sum(r['perfect_success'] for r in results) / len(results),
                'knowledge_digest': system.knowledge_digest(), 'usage_coverage': coverage,
                'benchmark_scores': {k:v for k,v in score_report.items() if k != 'per_task'},
                'invocation_wall_seconds': time.monotonic() - started, 'manifest': manifest.to_dict()}
            if recovery:
                summary.update(recovery=recovery, interrupted_usage_complete=not recovery['unknown_interrupted_attempts'])
            atomic_write_json(output / 'summary.json', summary)
            store.mark_run_state(run_id, RunState.COMPLETED)
            atomic_write_json(output / 'progress.json', {'state': 'completed', 'completed': len(items), 'total': len(items)})
        finally:
            if not online:
                db.close()
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-file', help='Stop before the next task when this file exists; resume after removing it')
    parser.add_argument('--stop-after-tasks', type=int, help='Task-boundary acceptance control, does not change dataset identity')
    args = parser.parse_args()
    raise SystemExit(run(args.config, resume=args.resume, stop_file=args.stop_file, stop_after_tasks=args.stop_after_tasks))
