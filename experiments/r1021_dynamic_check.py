"""Declared real-provider T5 routes, not autonomous benchmark performance.

The initial case declares no reusable graph. The rescue case declares a real
take contract from the existing route fixture, with a one-call Agent give-up
menu to enter normal rescue. Neither invents a parent for task automation.
All draft roles, values, programs, actions and validation are production paths.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

from atomic_skillgraph.core.serialization import atomic_create_json, atomic_write_json
from atomic_skillgraph.system import AtomicSkillGraphSystem, load_config, _SYSTEM_PROMPTS
from experiments import self_tooling_targeted as route
from experiments.protocol import artifact_audit_snapshot, capture_execution_manifest, hash_code, hash_config
from experiments.run_v3_r102_targeted import row_for, finalize_diagnostic
from experiments.run_v3_self_tooling_targeted import isolated_config

REPO = Path(__file__).resolve().parents[1]


class TaskAutomationProvider:
    def __init__(self, delegate):
        self.delegate, self.first = delegate, True

    def snapshot(self):
        return self.delegate.snapshot()

    def complete(self, messages, *, tools):
        messages = copy.deepcopy(messages)
        if self.first:
            tools = [tool for tool in tools if tool.name == 'request_runtime_automation']
            if not tools:
                raise RuntimeError('task automation request missing from real Dynamic menu')
            self.first = False
        if any(tool.name == 'propose_runtime_automation_atomic' for tool in tools):
            messages[0]['content'] += (
                '\nDECLARED T5 ACCEPTANCE: propose a bounded reusable discovery capability '
                'for an entity needed by this task. Use its semantic target as an input, '
                'and return the discovered entity and location. Internal search values '
                'must come from the public catalog. Choose all formal names and input '
                'sources from the provided task interface. There is no parent Atomic. '
                'The independent ToolBuilder authors the program; after a successful '
                'trial, continue from its resulting world state to finish the episode.')
        return self.delegate.complete(messages=messages, tools=tools)


class GiveUpEntryProvider:
    def __init__(self, delegate):
        self.delegate = delegate

    def snapshot(self):
        return self.delegate.snapshot()

    def complete(self, messages, *, tools):
        messages = copy.deepcopy(messages)
        messages[0]['content'] += (
            '\nDECLARED RESCUE ENTRY FIXTURE: report_runtime_status give_up now '
            'so the normal task-rescue route is exercised. Do not claim completion.')
        return self.delegate.complete(messages=messages,
            tools=[tool for tool in tools if tool.name == 'report_runtime_status'])


def run(config_path, output):
    start = time.monotonic()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = isolated_config(load_config(config_path), output)
    if config.get('repair_revision') != 'R10.2.1':
        raise ValueError('R10.2.1 required')
    config['experiment'].update(task_manifest_path=None, phase='r1021_t5')
    reference = json.loads((REPO / 'data/baseline_manifests/train_120.json').read_text())
    entries = [{**next(e for e in reference['tasks'] if int(e['task_id'].split('_')[2]) == i),
                'declared_route': mode} for i, mode in [(0, 'initial'), (1, 'rescue')]]
    manifest = {'formal_experiment': False, 'declared_menu_fixture': True,
        'tasks': entries, 'code_hash': hash_code(REPO), 'config_hash': hash_config(config)}
    atomic_create_json(output / 'declared_manifest.json', manifest)
    atomic_create_json(output / 'config.json', config)
    from atomic_skillgraph.agents.provider_probe import ensure_provider_capability
    capability = ensure_provider_capability(config, output_dir=output,
        code_hash=manifest['code_hash'], config_hash=manifest['config_hash'], run_if_missing=True)
    capture_execution_manifest(REPO, output, manifest['config_hash'], capability)
    rows = []
    with AtomicSkillGraphSystem(config) as system:
        append = system.usage.append
        def persist(event):
            result = append(event)
            atomic_write_json(output / 'all_usage.json', [e.to_dict() for e in system.usage.events])
            return result
        system.usage.append = persist
        tasks = {t.task_id: t for t in system.harness.load_tasks(limit=max(e['env_index'] for e in entries) + 1)}
        for entry in entries:
            if hash_code(REPO) != manifest['code_hash']:
                raise RuntimeError('code changed during acceptance')
            task = tasks[entry['task_id']]
            if hashlib.sha256(Path(task.context['game_file']).read_bytes()).hexdigest() != entry['gamefile_sha256']:
                raise RuntimeError('game identity mismatch')
            delegates = {stage: system._provider(stage) for stage in _SYSTEM_PROMPTS}
            dynamic = TaskAutomationProvider(delegates['runtime_dynamic'])
            system._provider_override = {**delegates, 'runtime_dynamic': dynamic}
            if entry['declared_route'] == 'initial':
                system.planner.build_plan = lambda *args, **kwargs: None
            else:
                case = route.RouteCase(task.task_id, target=task.context['semantic_bindings']['object'])
                parent, implementations = route.install_parent(system, case)
                plan = route.make_plan(task, parent, implementations, system.harness)
                plan.source, plan.source_composite_ref = 'atomic_composition', None
                system.planner.build_plan = lambda *args, **kwargs: plan
                system._provider_override['runtime_preparation'] = GiveUpEntryProvider(delegates['runtime_preparation'])
                system._provider_override['runtime_seeded'] = GiveUpEntryProvider(delegates['runtime_seeded'])
            started = time.monotonic()
            try:
                trace = system.run_task(task)
            finally:
                system._provider_override = None
            artifact_audit_snapshot(system.database)
            row = row_for(trace, time.monotonic() - started)
            trials = list(row['runtime_tool_trials'].values())
            row.update(declared_route=entry['declared_route'], task_rescue_required=trace.task_rescue_required,
                route_triggered=bool(trials) and all(t.get('consumer_scope') == 'task' and not t.get('parent_atomic_ref') for t in trials),
                r1_passed=any(t.get('r1', {}).get('admission_eligible') for t in trials))
            rows.append(row)
            atomic_write_json(output / 'progress.json', rows)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        finalization = finalize_diagnostic(system, output, rows)
    result = {**manifest, 'cases': rows, 'complete': True, 'wall_seconds': time.monotonic() - start,
        'passed': all(r['route_triggered'] and r['r1_passed'] for r in rows) and rows[1]['task_rescue_required'],
        'finalization': finalization, 'code_unchanged': hash_code(REPO) == manifest['code_hash']}
    atomic_write_json(output / 'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/alfworld_train_full_120_r1021_seed42.yaml')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.config, args.output)
