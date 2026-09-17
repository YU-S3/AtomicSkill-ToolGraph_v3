"""Predeclared R10.1 development diagnostics, never formal benchmark results.

The copied-bank set runs ordinary run_task unmodified. The empty-bank route
declares only a generic parent and first-call menu, as in R10's route fixtures;
the real model owns values/programs and normal governance owns activation.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import time

from atomic_skillgraph.core.contracts import SemanticPredicate
from atomic_skillgraph.core.refs import SkillRef
from atomic_skillgraph.core.serialization import atomic_create_json, atomic_write_json, to_primitive
from atomic_skillgraph.system import AtomicSkillGraphSystem, load_config
from experiments.protocol import artifact_audit_snapshot, capture_execution_manifest, code_file_manifest, hash_code, hash_config
from experiments.run_v3_self_tooling_targeted import isolated_config
from experiments.r10_promotion_checks import DraftEntryProvider
from experiments.r10_reuse_checks import ReuseEntryProvider
from experiments import self_tooling_targeted as route

REPO = Path(__file__).resolve().parents[1]
DEV_IDS = (2, 3, 15, 16, 28, 34, 36, 40, 45, 46, 53, 57, 1, 7, 12, 31)
CHAIN_IDS = (0, 1, 3, 4, 7, 8, 10, 12)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy_checkpoint(source_run, output):
    """Copy the immutable pre-task checkpoint; rebase copy-only path indexes."""
    checkpoint = source_run / '.task_checkpoint'
    declaration = json.loads((checkpoint / 'checkpoint_manifest.json').read_text())
    before = {str(p.relative_to(checkpoint)): file_hash(p) for p in checkpoint.rglob('*') if p.is_file()}
    bank = output / 'bank'
    shutil.copytree(checkpoint, bank)
    original_bank = Path(declaration['data_dir'])
    with sqlite3.connect(bank / 'state.sqlite3') as db:
        for table in ('artifact_index', 'provisional_artifacts', 'failure_experiences'):
            for row_id, old_path in db.execute(f'SELECT rowid,file_path FROM {table}').fetchall():
                relative = Path(old_path).relative_to(original_bank)
                new_path = bank / relative
                if not new_path.is_file():
                    raise RuntimeError(f'Checkpoint lacks registered payload: {relative}')
                db.execute(f'UPDATE {table} SET file_path=? WHERE rowid=?', (str(new_path), row_id))
    shutil.copytree(source_run / 'traces', output / 'trace_store' / 'traces')
    after = {str(p.relative_to(checkpoint)): file_hash(p) for p in checkpoint.rglob('*') if p.is_file()}
    if before != after:
        raise RuntimeError('Source checkpoint changed during diagnostic copy')
    return {'source_checkpoint': str(checkpoint), 'checkpoint_manifest_hash': file_hash(checkpoint / 'checkpoint_manifest.json'),
        'source_files': before, 'copy_only_path_rebase': True, 'not_fresh_training': True}


def row_for(trace, elapsed):
    return {'task_id': trace.task.task_id, 'trace_id': trace.trace_id,
        'official_success': trace.benchmark_success, 'contract_agreement': trace.task_contract_success,
        'infrastructure_failure': trace.infrastructure_failure,
        'runtime_source': trace.runtime_plan.get('source'),
        'failures': [to_primitive(value) for value in trace.failures],
        'r10_metrics': trace.metadata.get('r10_metrics', {}),
        'r101_metrics': trace.metadata.get('r101_metrics', {}),
        'support_transfers': trace.metadata.get('support_input_transfers', []),
        'promotions': trace.metadata.get('runtime_support_promotions', []),
        'promotion_rejections': trace.metadata.get('runtime_support_promotion_rejections', []),
        'duration_seconds': elapsed, 'llm_usage': trace.llm_usage}


def chain_parent(system, task):
    case = route.RouteCase(task.task_id, target=task.context['semantic_bindings']['object'])
    base, implementations = route.install_parent(system, case)
    parent = replace(base, ref=SkillRef('r101_diagnostic_discovered_take', '1.0.0'),
        preconditions=[SemanticPredicate('entity.discovered_at', {'entity': '$object', 'location': '$source'}, effect_domain='evidence')])
    system.skills.register_atomic(parent)
    implementation = replace(system.skills.get_implementation(implementations[0]),
        ref=SkillRef('r101_diagnostic_discovered_take_impl', '1.0.0'), abstract_ref=parent.ref)
    system.skills.register_implementation(implementation)
    return parent, [implementation.ref]


def run(config_path, output, *, source_run=None, chain=False):
    start = time.monotonic()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = isolated_config(load_config(config_path), output)
    config['experiment'].update(task_manifest_path=None, phase='r101_diagnostic')
    config['harness']['split'] = 'train'
    source = copy_checkpoint(Path(source_run).resolve(), output) if not chain else {'empty_bank': True}
    if not chain:
        config['experiment']['initialize_v3_bank'] = 'existing'
    ids = CHAIN_IDS if chain else DEV_IDS
    reference = json.loads((REPO / 'data/baseline_manifests/train_120.json').read_text())
    selected = {int(item['task_id'].split('_')[2]): item for item in reference['tasks']}
    entries = [selected[index] for index in ids]
    manifest = {'diagnostic_kind': 'real_same_bank_route_fixture' if chain else 'real_copied_bank_development',
        'source': source, 'fixed_task_ids': [entry['task_id'] for entry in entries], 'tasks': entries,
        'code_hash': hash_code(REPO), 'files': code_file_manifest(REPO), 'config_hash': hash_config(config),
        'formal_experiment': False, 'declared_before_execution': True}
    atomic_create_json(output / 'declared_manifest.json', manifest)
    atomic_create_json(output / 'config.json', config)
    from atomic_skillgraph.agents.provider_probe import ensure_provider_capability
    capability = ensure_provider_capability(config, output_dir=output,
        code_hash=manifest['code_hash'], config_hash=manifest['config_hash'], run_if_missing=True)
    capture_execution_manifest(REPO, output, manifest['config_hash'], capability)
    rows = []
    with AtomicSkillGraphSystem(config) as system:
        artifact_audit_snapshot(system.database)
        tasks = {task.task_id: task for task in system.harness.load_tasks(limit=max(e['env_index'] for e in entries) + 1)}
        for entry in entries:
            task = tasks[entry['task_id']]
            if task.context['env_index'] != entry['env_index'] or file_hash(task.context['game_file']) != entry['gamefile_sha256']:
                raise RuntimeError('Predeclared training task identity changed')
        parent_fixture, promoted_refs = None, None
        for entry in entries:
            task = tasks[entry['task_id']]
            status_before = []
            if chain:
                if parent_fixture is None:
                    parent_fixture = chain_parent(system, task)
                parent, implementations = parent_fixture
                plan = route.make_plan(task, parent, implementations, system.harness)
                plan.source, plan.source_composite_ref = 'atomic_composition', None
                system.planner.build_plan = lambda *args, **kwargs: plan
                provider_class = DraftEntryProvider if promoted_refs is None else ReuseEntryProvider
                if promoted_refs:
                    status_before = [system.skills.get_atomic(promoted_refs[0]).status.value,
                        system.skills.get_implementation(promoted_refs[1]).status.value,
                        system.tools.get(promoted_refs[2]).status.value]
                provider = provider_class(route.RouteCase(task.task_id), 'runtime', [], delegate=system._provider('runtime_preparation'))
                if promoted_refs:
                    provider.automatic = all(status in {'active', 'preferred'} for status in status_before)
                from atomic_skillgraph.system import _SYSTEM_PROMPTS
                delegates = {stage: system._provider(stage) for stage in _SYSTEM_PROMPTS}
                system._provider_override = {**delegates, 'runtime_preparation': provider, 'runtime_seeded': provider}
            started = time.monotonic()
            trace = system.run_task(task)
            system._provider_override = None
            artifact_audit_snapshot(system.database)
            row = row_for(trace, time.monotonic() - started)
            if chain:
                if promoted_refs is None and row['promotions']:
                    promoted_refs = row['promotions'][-1]['refs']
                row.update(status_before=status_before, promoted_refs=promoted_refs,
                    automatic_entry=bool(status_before and all(s in {'active', 'preferred'} for s in status_before)))
            rows.append(row)
            atomic_write_json(output / 'progress.json', rows)
            print(json.dumps({key: value for key, value in row.items() if key not in {'llm_usage', 'support_transfers'}}, ensure_ascii=False), flush=True)
        # Includes maintenance calls as well as task calls, no failed trial filtering.
        usage = [event.to_dict() for event in system.usage.events]
        atomic_create_json(output / 'all_usage.json', usage)
    result = {'cases': rows, 'complete': len(rows) == len(entries), 'official_successes': sum(row['official_success'] for row in rows),
        'tasks': len(rows), 'wall_seconds': time.monotonic() - start,
        'code_hash': manifest['code_hash'], 'code_unchanged': hash_code(REPO) == manifest['code_hash'],
        'formal_experiment': False, 'total_tokens': sum(event.get('total_tokens') or 0 for event in usage),
        'same_bank_automatic_reuse_observed': bool(chain and any(row.get('automatic_entry') and row['r10_metrics'].get('runtime_support_reuse_count', 0) for row in rows))}
    atomic_create_json(output / 'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/alfworld_train_full_120_r101_seed42.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-run')
    parser.add_argument('--chain', action='store_true')
    args = parser.parse_args()
    if not args.chain and not args.source_run:
        parser.error('--source-run is required for the copied-bank development set')
    run(args.config, args.output, source_run=args.source_run, chain=args.chain)
