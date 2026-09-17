"""Predeclared R10.2 fresh-bank diagnostics; never a formal score or old-bank resume."""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

from atomic_skillgraph.core.serialization import atomic_create_json, atomic_write_json, to_primitive
from atomic_skillgraph.system import AtomicSkillGraphSystem, load_config, _SYSTEM_PROMPTS
from experiments.protocol import artifact_audit_snapshot, capture_execution_manifest, code_file_manifest, hash_code, hash_config, validate_deepseek_formal_llm
from experiments.run_v3_self_tooling_targeted import isolated_config
from atomic_skillgraph.core.contracts import SemanticPredicate
from atomic_skillgraph.core.refs import SkillRef
from experiments import self_tooling_targeted as route

REPO = Path(__file__).resolve().parents[1]
DEV_IDS = (2, 3, 15, 16, 28, 34, 36, 40, 45, 46, 53, 57, 1, 7, 12, 31)
CHAIN_IDS = (0, 1, 3, 4, 7, 10, 12, 18)


class DraftEntryProvider(route.RouteProvider):
    def complete(self, messages, *, tools):
        messages = copy.deepcopy(messages)
        selected = list(tools)
        if not self.forced:
            selected = [tool for tool in selected if tool.name == "request_runtime_automation"]
            if not selected:
                raise RuntimeError("R10 lightweight request entry is missing")
            self.forced = True
        if any(tool.name == "propose_runtime_automation_atomic" for tool in selected):
            messages[0]["content"] += (
                "\nTARGETED ACCEPTANCE: author a reusable bounded discovery helper for the "
                "current parent target. The input is the semantic target family. Expose both "
                "a discovered concrete entity and its discovered concrete location, with the "
                "public interface's output semantic constraints. Do not take/place the entity. "
                "For this declared acceptance boundary use one required semantic entity input "
                "(runtime_resolvable=true), two required concrete entity outputs "
                "(runtime_resolvable=false), no preconditions, and the single evidence Effect "
                "entity.discovered_at(entity, location) with cardinality=1 and distinct_by=''. "
                "Constrain the discovered entity to the semantic input. Choose your own role "
                "names and author the full draft under R0; the Tool program is built independently."
            )
        index = len(self.requests)
        self.requests.append({"messages": route.safe_messages(messages), "tools": [t.to_openai() for t in selected]})
        if self.audit:
            self.audit(self.stage, index, "request", self.requests[-1])
        turn = self.delegate.complete(messages=messages, tools=selected)
        if self.audit:
            self.audit(self.stage, index, "result", {"calls": [t.to_dict() if hasattr(t, "to_dict") else {"name": t.name, "arguments": t.arguments} for t in turn.tool_calls],
                "prompt_tokens": turn.prompt_tokens, "completion_tokens": turn.completion_tokens})
        return turn



def chain_parent(system, task):
    case = route.RouteCase(task.task_id, target=task.context['semantic_bindings']['object'])
    base, implementations = route.install_parent(system, case)
    parent = replace(base, ref=SkillRef('r102_diagnostic_discovered_take', '1.0.0'),
        preconditions=[SemanticPredicate('entity.discovered_at', {'entity': '$object', 'location': '$source'}, effect_domain='evidence')])
    system.skills.register_atomic(parent)
    implementation = replace(system.skills.get_implementation(implementations[0]),
        ref=SkillRef('r102_diagnostic_discovered_take_impl', '1.0.0'), abstract_ref=parent.ref)
    system.skills.register_implementation(implementation)
    return parent, [implementation.ref]


class AgentReuseProvider(route.RouteProvider):
    """Declare only the acceptance menu, never values, routes, or successful evidence."""
    def complete(self, messages, *, tools):
        import copy
        messages = copy.deepcopy(messages)
        if not self.forced:
            tools = [tool for tool in tools if tool.name == 'invoke_support_atomic']
            if not tools:
                raise RuntimeError('Explicit learned helper is not offered')
            messages[0]['content'] += (
                '\nTARGETED ACCEPTANCE: explicitly invoke the persistent discovery helper '
                'for the current parent target. Choose values from the public context and '
                'map its verified entity/location outputs to the parent inputs. '
                'This is an explicit Agent choice, not automatic Support execution.')
            self.forced = True
        request = {'messages': route.safe_messages(messages), 'tools': [t.to_openai() for t in tools]}
        self.requests.append(request)
        if self.audit:
            self.audit(self.stage, len(self.requests), 'request', request)
        return self.delegate.complete(messages=messages, tools=tools)


def row_for(trace, elapsed):
    usage = trace.llm_usage
    metrics = trace.metadata.get('r10_metrics', {})
    return {'task_id': trace.task.task_id, 'trace_id': trace.trace_id,
        'official_won': trace.benchmark_success, 'contract_agreement': trace.task_contract_success,
        'infrastructure_failure': trace.infrastructure_failure,
        'graph_source': trace.runtime_plan.get('source'),
        'runtime_agent_calls': sum('runtime' in str(item.get('bucket', '')) and 'tool_builder' not in str(item.get('bucket', '')) for item in usage),
        'automatic_graph_nodes': metrics.get('composite_auto_node_success_count', 0),
        'agent_selected_helpers': metrics.get('support_agent_selected_count', 0),
        'runtime_tool_trials': trace.metadata.get('runtime_tool_trials', {}),
        'entry_decisions': trace.metadata.get('node_entry_checks', []),
        'action_source_audit': action_source_audit(trace),
        'failures': to_primitive(trace.failures),
        'prompt_tokens': sum(int(item.get('prompt_tokens') or 0) for item in usage),
        'reasoning_tokens': sum(int(item.get('reasoning_tokens') or 0) for item in usage),
        'non_reasoning_completion_tokens': sum(int(item.get('completion_tokens') or 0) - int(item.get('reasoning_tokens') or 0) for item in usage),
        'total_tokens': sum(int(item.get('total_tokens') or 0) for item in usage),
        'environment_actions': len(trace.environment_actions),
        'restore_replay_actions': trace.metadata.get('r101_metrics', {}).get('physical_restore_replay_actions', 0),
        'duration_seconds': elapsed, 'r10_metrics': metrics,
        'support_transfers': trace.metadata.get('support_input_transfers', []),
        'promotions': trace.metadata.get('runtime_support_promotions', []),
        'promotion_rejections': trace.metadata.get('runtime_support_promotion_rejections', []),
        'llm_usage': usage}


def action_source_audit(trace):
    """Match physical actions to native selections or recorded invoked Tool spans."""
    native = {call.arguments.get('action_id'): call.call_id for call in trace.native_tool_calls
              if call.call_kind == 'environment_action'}
    tool_spans = {record.span_id: record for record in trace.tool_executions}
    rows = []
    for index, action in enumerate(trace.environment_actions):
        tool = tool_spans.get(action.span_id)
        source = ('tool_program' if tool is not None else
                  'agent_native' if action.action_id in native else 'unattributed')
        rows.append({'event_index': index, 'action_id': action.action_id,
            'span_id': action.span_id, 'source': source,
            'source_ref': tool.tool_ref if tool is not None else native.get(action.action_id),
            'attempt_id': tool.attempt_id if tool is not None else None})
    return {'unattributed_world_actions': sum(row['source'] == 'unattributed' for row in rows),
            'events': rows}


def finalize_diagnostic(system, output, rows):
    """Use the formal maintenance-before-freeze boundary, without running tasks again."""
    maintenance = system.run_maintenance(
        triggering_task_id=rows[-1]['task_id'],
        milestone='r102_diagnostic_final_batch', finalize_pending=True)
    if maintenance.pending_count != 0:
        raise RuntimeError('Diagnostic final maintenance left pending repairs')
    artifact_audit_snapshot(system.database)
    digest = system.knowledge_digest()
    system.freeze(output / 'frozen_bank')
    if system.knowledge_digest() != digest:
        raise RuntimeError('Freeze changed source knowledge')
    return {'maintenance': to_primitive(maintenance), 'knowledge_digest': digest,
            'source_digest_unchanged': True}


def run(config_path, output, *, mode):
    start = time.monotonic()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = isolated_config(load_config(config_path), output)
    if config.get('repair_revision') != 'R10.2':
        raise ValueError('R10.2 configuration required')
    validate_deepseek_formal_llm(config)
    config['experiment'].update(task_manifest_path=None, phase='r102_diagnostic')
    ids = DEV_IDS if mode == 'dev16' else CHAIN_IDS
    reference = json.loads((REPO / 'data/baseline_manifests/train_120.json').read_text())
    by_id = {int(item['task_id'].split('_')[2]): item for item in reference['tasks']}
    entries = [by_id[index] for index in ids]
    manifest = {'mode': mode, 'empty_bank': True, 'formal_experiment': False,
        'declared_before_execution': True, 'tasks': entries,
        'code_hash': hash_code(REPO), 'files': code_file_manifest(REPO), 'config_hash': hash_config(config)}
    atomic_create_json(output / 'declared_manifest.json', manifest)
    atomic_create_json(output / 'config.json', config)
    from atomic_skillgraph.agents.provider_probe import ensure_provider_capability
    capability = ensure_provider_capability(config, output_dir=output, code_hash=manifest['code_hash'],
        config_hash=manifest['config_hash'], run_if_missing=True)
    capture_execution_manifest(REPO, output, manifest['config_hash'], capability)
    rows = []
    with AtomicSkillGraphSystem(config) as system:
        append_usage = system.usage.append
        def persist_usage(event):
            result = append_usage(event)
            atomic_write_json(output / 'all_usage.json', [item.to_dict() for item in system.usage.events])
            return result
        system.usage.append = persist_usage
        artifact_audit_snapshot(system.database)
        tasks = {task.task_id: task for task in system.harness.load_tasks(limit=max(e['env_index'] for e in entries) + 1)}
        for entry in entries:
            task = tasks[entry['task_id']]
            if task.context['env_index'] != entry['env_index'] or hashlib.sha256(Path(task.context['game_file']).read_bytes()).hexdigest() != entry['gamefile_sha256']:
                raise RuntimeError('Predeclared source task identity changed')
        parent_fixture, promoted_refs = None, None
        for entry in entries:
            if hash_code(REPO) != manifest['code_hash']:
                raise RuntimeError('Code changed during fixed diagnostic')
            task = tasks[entry['task_id']]
            status_before = []
            if mode == 'chain':
                if parent_fixture is None:
                    parent_fixture = chain_parent(system, task)
                parent, implementations = parent_fixture
                plan = route.make_plan(task, parent, implementations, system.harness)
                plan.source, plan.source_composite_ref = 'atomic_composition', None
                system.planner.build_plan = lambda *args, **kwargs: plan
                if promoted_refs:
                    status_before = [system.skills.get_atomic(promoted_refs[0]).status.value,
                        system.skills.get_implementation(promoted_refs[1]).status.value,
                        system.tools.get(promoted_refs[2]).status.value]
                delegates = {stage: system._provider(stage) for stage in _SYSTEM_PROMPTS}
                root = output / 'provider_calls' / task.task_id
                provider_type = AgentReuseProvider if promoted_refs else DraftEntryProvider
                provider = provider_type(route.RouteCase(task.task_id), 'runtime', [],
                    delegate=delegates['runtime_preparation'], audit=lambda stage, i, kind, payload:
                    atomic_write_json(root / f'{stage}_{i:03d}_{kind}.json', payload))
                system._provider_override = {**delegates, 'runtime_preparation': provider, 'runtime_seeded': provider}
            started = time.monotonic()
            trace = system.run_task(task)
            system._provider_override = None
            artifact_audit_snapshot(system.database)
            row = row_for(trace, time.monotonic() - started)
            if mode == 'chain':
                if promoted_refs is None and row['promotions']:
                    promoted_refs = row['promotions'][-1]['refs']
                row.update(status_before=status_before, promoted_refs=promoted_refs)
            rows.append(row)
            atomic_write_json(output / 'progress.json', rows)
            print(json.dumps({k: v for k, v in row.items() if k not in {'llm_usage', 'runtime_tool_trials', 'support_transfers'}}, ensure_ascii=False), flush=True)
            if row['action_source_audit']['unattributed_world_actions']:
                raise RuntimeError('Diagnostic action provenance audit failed; see saved Trace and progress')
        finalization = finalize_diagnostic(system, output, rows)
        usage = [item.to_dict() for item in system.usage.events]
    result = {'mode': mode, 'complete': len(rows) == len(entries), 'cases': rows,
        'tasks': len(rows), 'official_successes': sum(row['official_won'] for row in rows),
        'code_hash': manifest['code_hash'], 'code_unchanged': hash_code(REPO) == manifest['code_hash'],
        'wall_seconds': time.monotonic() - start, 'formal_experiment': False,
        'finalization': finalization,
        'total_tokens': sum(int(event.get('total_tokens') or 0) for event in usage)}
    atomic_write_json(output / 'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/alfworld_train_full_120_r102_seed42.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['dev16', 'chain'], required=True)
    args = parser.parse_args()
    run(args.config, args.output, mode=args.mode)
