"""Locked Release4 training-side coverage, independent of formal Test134."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

from atomic_skillgraph.core.serialization import atomic_create_json, atomic_write_json
from atomic_skillgraph.deployment.release_protocol import ReleaseError, sha
from .protocol import ALFWORLD_FORMAL_TASK_TYPES, hash_code

REPO = Path(__file__).resolve().parents[1]


def read(path):
    return Path(path).read_text(encoding='utf-8')


def entries_for(root):
    reference = json.loads(read(REPO/'data/baseline_manifests/train_120.json'))
    prior = json.loads(read(root/'prior_dev_entries.json'))
    entries = []
    for family in ALFWORLD_FORMAL_TASK_TYPES:
        candidates = [e for e in prior if e['task_type'] == family]
        if not candidates:
            candidates = [e for e in reference['tasks'] if e['task_type'] == family]
        entries.append(min(candidates, key=lambda e: e['index']))
    return entries


def episode_config(root, seed, entry):
    prepared = root/f'seed{seed}'
    config = json.loads(read(prepared/'prepare_manifest.json'))['config']
    config.pop('_config_path', None)
    frozen = prepared/'frozen'
    manifest = json.loads(read(frozen/'release_manifest.json'))
    output = root/'dev'/f'seed{seed}'/'current'/entry['task_id']
    config['data_dir'], config['trace_data_dir'] = str(frozen), str(output)
    config['harness']['split'] = 'train'
    config['experiment'].update(name=output.name, experiment_kind='bank_release_eval', phase='test',
        runtime_mode='frozen', freeze_skills=True, allow_long_term_knowledge_writes=False,
        output_dir=str(output), task_manifest_path=str(output/'task_manifest.json'))
    config['bank_release'] = {'protocol_version': manifest['protocol_version'], 'source_seed': seed,
        'release_manifest': str(frozen/'release_manifest.json'), 'expected_bank_digest': manifest['knowledge_digest'],
        'provider_probe_dir': str(root/'dev/provider_probe')}
    config['deployment'] = {'presentation_profile': 'current', 'presentation_version': 'r103.frozen-expression.v2',
        'input_authorization_version': 'r103.caller-input.v1', 'programs': True, 'automatic_entry': True,
        'capture_final_request': True, 'repeat_index': 1}
    return config


def surface_audit(trace):
    """Audit every physical request, not one guessed request per session."""
    metadata = trace.get('metadata', {})
    surfaces = metadata.get('support_call_surfaces', [])
    checks = []
    for request in trace.get('provider_requests', []):
        matches = [s for s in surfaces if s['session_id'] == request['session_id']]
        if not matches:
            payload = request.get('final_payload_audit') or {}
            offered = [t for t in payload.get('tools', []) if t.get('function', {}).get('name') == 'invoke_support_atomic']
            if offered:
                checks.append({'request_id': request['request_id'], 'passed': False, 'reason': 'offered_support_without_surface'})
            continue
        payload = request.get('final_payload_audit') or {}
        native = [t['function']['parameters']['properties']['support_call_id']['enum']
            for t in payload.get('tools', []) if t.get('function', {}).get('name') == 'invoke_support_atomic']
        contexts = payload.get('policy_contexts', [])
        rows = [c.get('support_atomic_candidates', c.get('task_runtime_frame', {}).get('capability_candidates', []))
                for c in contexts]
        matches = [s for s in matches if rows and rows[-1] == s['options']]
        surface = matches[-1] if matches else {'options': []}
        expected = [o['support_call_id'] for o in surface['options']]
        passed = bool(matches) and (native == [expected] if expected else native == [])
        checks.append({'request_id': request['request_id'], 'session_id': request['session_id'],
            'option_ids': expected, 'passed': passed})
    for call in trace.get('native_tool_calls', []):
        if call['tool_name'] != 'invoke_support_atomic':
            continue
        rows = [r for r in metadata.get('support_call_resolutions', [])
                if r['call_id'] == call['call_id'] and r['session_id'] == call['session_id']]
        checks.append({'call_id': call['call_id'], 'passed': len(rows) == 1
            and rows[0]['support_call_id'] == call['arguments'].get('support_call_id')
            and 'support_atomic_ref' not in call['arguments'] and 'output_mapping' not in call['arguments']})
    return {'passed': all(c['passed'] for c in checks), 'checks': checks}


def verify_coverage(root, *, write=False):
    from .released_dev_checks import check_episode
    root = Path(root)
    code = hash_code(REPO)
    lock = json.loads(read(root/'dev/coverage_tasks.lock.json'))
    if lock['entries'] != entries_for(root) or lock['code_hash'] != code:
        raise ReleaseError('coverage task/code lock mismatch')
    rows, controlled = [], []
    for seed in (42, 43, 44):
        manifest = json.loads(read(root/f'seed{seed}/frozen/release_manifest.json'))
        if manifest['code_hash'] != code:
            raise ReleaseError('coverage release code differs')
        for entry in lock['entries']:
            output = root/'dev'/f'seed{seed}'/'current'/entry['task_id']
            row = check_episode(output, code_hash=code, bank_digest=manifest['knowledge_digest'], entry=entry, profile='current')
            trace = json.loads(read(output/'traces'/f"{row['trace_id']}.json"))
            audit = surface_audit(trace)
            if not audit['passed']:
                raise ReleaseError(f'HTTP/Support surface mismatch: {output}')
            if (root/f'seed{seed}/frozen/native_call_contract.json').is_file():
                from .native_contract_audit import audit_trace
                native = audit_trace(trace)
                if not native['passed']:
                    atomic_write_json(output/'native_contract_failure.json', native)
                    raise ReleaseError(f'HTTP/Session native schema or usage mismatch: {output}')
                row['native_contract_audit'] = native
            rows.append({'seed': seed, 'task_id': entry['task_id'], 'surface_audit': audit, **row})
        auto_path = root/'dev'/f'seed{seed}'/'automation/acceptance.json'
        auto = json.loads(read(auto_path))
        if (not auto.get('passed') or auto.get('code_hash') != code
                or auto['bank_digest_before'] != manifest['knowledge_digest']
                or auto['bank_digest_after'] != manifest['knowledge_digest']
                or {r['kind'] for r in auto.get('public_discovery_checks', [])} != {'takeable', 'non_takeable'}
                or sha(auto_path.parent/'traces'/f"{auto['trace_id']}.json") != auto['trace_sha256']):
            raise ReleaseError('controlled public discovery/dataflow acceptance failed')
        controlled.append({'seed': seed, 'path': str(auto_path), 'sha256': sha(auto_path)})
    if (root/'seed42/frozen/native_call_contract.json').is_file():
        graph_path = root/'dev/seed42/graph_revision/acceptance.json'
        graph = json.loads(read(graph_path))
        if (not graph['passed'] or graph['code_hash'] != code
                or graph['bank_digest'] != json.loads(read(root/'seed42/frozen/release_manifest.json'))['knowledge_digest']
                or not graph['trace_hashes'] or any(sha(graph_path.parent/p) != h for p, h in graph['trace_hashes'].items())):
            raise ReleaseError('controlled graph revision acceptance failed')
        controlled.append({'kind': 'seed42_graph_revision', 'path': str(graph_path), 'sha256': sha(graph_path)})
    result = {'passed': True, 'suite': 'oldfirst-coverage6', 'code_hash': code, 'episodes': rows}
    if write:
        atomic_write_json(root/'dev/coverage_acceptance.json', result)
        atomic_write_json(root/'dev/controlled_acceptance.json', {'passed': True, 'seeds': controlled})
        from .compiler_metrics import distribution
        from .release4_metrics import trace_metrics
        measurements = []
        for row in rows:
            output = Path(row['output'])
            summary = json.loads(read(output/'summary.json'))
            trace = json.loads(read(output/'traces'/f"{row['trace_id']}.json"))
            measurements.append({'seed': row['seed'], 'task_id': row['task_id'],
                'task_type': trace['task']['task_type'], 'success': trace['benchmark_success'],
                'costs': row['costs'], 'selected_route': (trace.get('runtime_plan') or {}).get('source'),
                'summary': summary, 'interfaces': trace_metrics(trace),
                'native_protocol': row.get('native_contract_audit', {'status': 'legacy/not_captured'})})
        report = {'passed': True, 'episode_count': len(rows),
            'successes': sum(m['success'] is True for m in measurements),
            'all_task_token_distribution': distribution(r['costs']['total_tokens'] for r in rows),
            'failed_task_recorded_tokens': sum(m['costs']['total_tokens'] for m in measurements if not m['success']),
            'rows': measurements, 'controlled_seeds': controlled,
            'scope': 'Fixed training-side development only; not Test134; no guarantee of 18/18 or 40K.'}
        atomic_write_json(root/'dev/coverage_report.json', report)
        lines = ['# Release4 六族三库验证', '',
            f"实现与协议检查通过；自然任务成功 {report['successes']}/{len(rows)}。", '',
            '成功率只用十八个固定自然 episode；专项执行不加入分母。费用包含全部 attempt，reasoning 已包含在 completion。', '',
            '| Seed | Task | 成功 | 全 attempt tokens | 请求数 | 路线 |', '|---|---|---|---:|---:|---|']
        lines.extend(f"| {m['seed']} | {m['task_id']} | {m['success']} | {m['costs']['total_tokens']} | {m['costs']['call_count']} | {m['selected_route']} |"
                     for m in measurements)
        lines += ['', '完整接口/关系/费用关联见 coverage_report.json；原始材料位于 seed*/current/<task>/。',
            '三库真实程序专项位于 seed*/automation/，三库静态准备覆盖位于 ../seed*/frozen/preparation_coverage.json。',
            '该检查不以全部任务成功或低于固定 token 阈值为放行条件；不是泛化效果或纯压缩收益的证明。', '']
        (root/'dev/coverage_report.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


def run_seed(root, seed, entries):
    from .run_v3_released_frozen import run
    from .run_v3_r103_validation import resolve_tasks
    from .oldfirst_bank_checks import run as controlled
    config = episode_config(root, seed, entries[0])
    output = root/'dev'/f'seed{seed}'/'automation'
    # A fixed training fixture, never a Runtime family-to-action rule.
    controlled_entry = next(e for e in entries if e['task_type'] == 'look_at_obj_in_light')
    if not output.exists():
        controlled(config, controlled_entry, output)
    if seed == 42 and (root/'seed42/frozen/native_call_contract.json').is_file():
        from .release5_graph_checks import run as graph_checks
        graph_output = root/'dev/seed42/graph_revision'
        if not graph_output.exists():
            entry = next(e for e in entries if e['task_type'] == 'pick_cool_then_place_in_recep')
            graph_ref = json.loads(read(root/'seed42_graph_revision_audit.json'))['target_ref']
            graph_checks(episode_config(root, seed, entry), entry, graph_ref, graph_output)
    for entry in entries:
        config = episode_config(root, seed, entry)
        path = root/'dev/configs'/f'seed{seed}_{entry["task_id"]}.json'
        if path.exists() and json.loads(read(path)) != config:
            raise ReleaseError('coverage config changed')
        if not path.exists():
            atomic_create_json(path, config)
        episode = Path(config['experiment']['output_dir'])
        if (episode/'progress.json').is_file() and json.loads(read(episode/'progress.json')).get('state') == 'completed':
            continue  # verified as a complete immutable episode by final acceptance
        print(json.dumps({'stage': 'coverage_episode', 'seed': seed, 'task_id': entry['task_id']}), flush=True)
        run(path, resume=episode.exists(), task_entries=lambda harness, e=entry: resolve_tasks(SimpleNamespace(harness=harness), [e]))


def run_coverage(root, *, workers=1, seeds=(42, 43, 44)):
    root = Path(root).resolve()
    entries = entries_for(root)
    lock = {'suite': 'oldfirst-coverage6', 'code_hash': hash_code(REPO), 'entries': entries}
    path = root/'dev/coverage_tasks.lock.json'
    if path.exists() and json.loads(read(path)) != lock:
        raise ReleaseError('coverage must restart in a new root for different code/tasks')
    if not path.exists():
        atomic_create_json(path, lock)
    # Complete the shared read-only capability result before worker processes start.
    from atomic_skillgraph.agents.provider_probe import ensure_provider_capability
    from .protocol import hash_config
    config = episode_config(root, 42, entries[0])
    ensure_provider_capability(config, output_dir=root/'dev/provider_probe',
        config_hash=hash_config({'llm': config['llm']}), code_hash=lock['code_hash'], run_if_missing=True)
    if workers == 1:
        for seed in seeds:
            run_seed(root, seed, entries)
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=min(workers, 3)) as pool:
            futures = [pool.submit(run_seed, root, seed, entries) for seed in seeds]
            for future in futures:
                future.result()
    if all((root/'dev'/f'seed{s}'/'current'/e['task_id']/'summary.json').is_file()
           for s in (42, 43, 44) for e in entries):
        return verify_coverage(root, write=True)
    return {'passed': False, 'pending_other_seeds': True, 'episodes': []}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    report = verify_coverage(args.release_root, write=True) if args.verify_only else run_coverage(args.release_root, workers=args.workers)
    print(json.dumps({'passed': report['passed'], 'episodes': len(report['episodes'])}), flush=True)
