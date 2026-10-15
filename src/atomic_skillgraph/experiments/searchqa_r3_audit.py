"""Read-only D0 Host audit. Its held-out diagnostics are never learning material."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3

from ..empirical.contracts import digest, validate_schema_instance
from ..empirical.searchqa_trace_learning import read, sha, validate_candidate
from ..empirical.searchqa_guidance import render_procedure
from ..empirical.searchqa_guidance_r3 import normalize_finish
from ..empirical.public_results import public_result, SEARCHQA_R3_EXECUTION
from ..empirical.answer_executor import STEP
from .run_empirical import write_json


def material(request):
    for message in request['messages']:
        if message.get('role') == 'user' and isinstance(message.get('content'), str):
            try: return json.loads(message['content'])
            except ValueError: continue
    return {}


def audit(r2, output, historical_manifest):
    r2, output = Path(r2).resolve(), Path(output).resolve()
    old = read(historical_manifest)['files']
    changed = [p for p, value in old.items() if sha(p) != value['sha256']]
    if changed: raise ValueError('Historical source changed: ' + str(changed[:3]))
    files = {str(p): sha(p) for p in sorted(r2.rglob('*')) if p.is_file()
             and not p.name.startswith('.denied_') and not p.name.endswith(('-wal', '-shm'))}
    exposed, inductions, builders, rejected, reuse = set(), [], [], [], []
    for path in sorted((r2 / 'learn').glob('*/requests.json')):
        audit_row = read(path); result = read(path.with_name('result.json'))
        for req in audit_row['requests']:
            if not req.get('http_attempts'): continue
            source = {s['case_id']: s for s in material(req).get('sources', [])}
            exposed.update(source)
            row = {'path': str(path), 'file_sha256': sha(path), 'request_id': req['id'],
                   'material_hash': digest(material(req)), 'source_ids': sorted(source),
                   'response_hash': digest(req.get('response')), 'decision': result['decision']}
            inductions.append(row)
            if result['decision'] == 'rejected' and req is audit_row['requests'][-1]:
                calls = req.get('response', {}).get('tool_calls', [])
                c = calls[0]['arguments'].get('candidate') if len(calls) == 1 else None
                try: validate_candidate(c, source)
                except (ValueError, TypeError, KeyError) as exc: reason = str(exc)
                else: reason = None
                rejected.append({**row, 'old_raw_candidate': c, 'original_error': result.get('reason'),
                    'replay_error': reason, 'rendered_chars': len(render_procedure(c)) if c else None,
                    'repair_available_in_r3': True, 'promoted': False})
    db = sqlite3.connect((r2 / 'train/bank/bank.sqlite3').as_uri() + '?mode=ro', uri=True)
    assets = {key: json.loads(payload) for key, payload in db.execute("SELECT id,payload FROM assets WHERE kind='skill'")}
    db.close()
    for folder in sorted((r2 / 'worker_probe').glob('skill_*')):
        result = read(folder / 'result.json')
        failures = []
        for path in sorted(folder.glob('generation_*/requests.json')):
            for req in read(path)['requests']:
                if not req.get('http_attempts'): continue
                source = material(req).get('sources', [])
                exposed.update(s['case_id'] for s in source)
                response = req.get('response', {})
                row = {'path': str(path), 'file_sha256': sha(path), 'request_id': req['id'],
                    'response': response, 'response_hash': digest(response), 'material_hash': digest(material(req)),
                    'finish_reason': response.get('finish_reason'), 'reasoning_tokens': response.get('usage', {}).get('reasoning_tokens'),
                    'reasoning_missing': response.get('usage', {}).get('reasoning_tokens') is None,
                    'http_attempt_ids': [a['request_id'] for a in req['http_attempts']],
                    'classification': 'completion_length' if response.get('finish_reason') == 'length' else 'other'}
                builders.append(row)
                if response.get('finish_reason') == 'length': failures.append(row)
        if failures:
            reuse.append({'old_asset': assets[folder.name], 'old_spec': read(folder / 'spec.json'),
                'old_result': result, 'old_failures': failures, 'parent_spec_hash': sha(folder / 'spec.json'),
                'old_asset_hash': digest(assets[folder.name]), 'qualification': 'historical_failed_spec_only'})
    projections, failures = [], []
    def project(value):
        if isinstance(value, list): return [project(v) for v in value]
        if not isinstance(value, dict): return value
        if 'execution_receipt' in value and 'outputs' in value:
            result = public_result(value, SEARCHQA_R3_EXECUTION)
            if result['outputs'] != value['outputs']: raise ValueError('Projection changed actual output')
            return result
        return {k: project(v) for k, v in value.items()}
    for phase in ('train_probe', 'val'):
        for path in sorted((r2 / phase).glob('*/traces/*.json')):
            trace = read(path)
            for req in trace['requests']:
                raw = req.get('messages', []); projected = deepcopy(raw)
                for m in projected:
                    if m.get('role') != 'user' or not isinstance(m.get('content'), str): continue
                    try: value = json.loads(m['content'])
                    except ValueError: continue
                    m['content'] = json.dumps(project(value), ensure_ascii=False, separators=(',', ':'))
                if 'execution_receipt' in json.dumps(raw):
                    raw_material = material(req); new_material = project(raw_material)
                    projections.append({'path': str(path), 'request_id': req['id'], 'raw_messages_hash': digest(raw),
                        'projected_messages_hash': digest(projected), 'receipt_removed': 'execution_receipt' not in json.dumps(new_material),
                        'outputs_unchanged': True, 'removed_chars': len(json.dumps(raw_material)) - len(json.dumps(new_material))})
                response = req.get('response', {})
                for call in response.get('tool_calls', []):
                    if {'unit', 'end'} & set(call.get('arguments', {})):
                        normalized, action = normalize_finish(call['arguments'], STEP)
                        try: validate_schema_instance(normalized, STEP)
                        except ValueError as exc: error = str(exc)
                        else: error = None
                        failures.append({'path': str(path), 'request_id': req['id'], 'raw_call': call,
                            'normalization': action, 'replay_error': error, 'old_result_unchanged': True})
                if response.get('finish_reason') == 'length' and not response.get('content') and not response.get('tool_calls'):
                    failures.append({'path': str(path), 'request_id': req['id'], 'finish_reason': 'length',
                        'raw_response_hash': digest(response), 'r3_eligible_for_one_finish': True, 'old_result_unchanged': True})
    if len(exposed) != 19 or len(rejected) != 4 or len(reuse) != 3 or sum(r['classification'] == 'completion_length' for r in builders) != 6 or len(projections) != 7:
        raise ValueError('Actual R2 counts differ from reviewed evidence; stop before paid work: ' + str({
            'exposed': len(exposed), 'rejected': len(rejected), 'failed_specs': len(reuse),
            'length_builders': sum(r['classification'] == 'completion_length' for r in builders), 'projections': len(projections)}))
    result = {'http': 0, 'historical_files_verified': len(old), 'historical_manifest_hash': sha(historical_manifest),
        'historical_files': {p: value['sha256'] for p, value in old.items()}, 'r2_files': files,
        'exposed_ids': sorted(exposed), 'induction_material_audit': inductions,
        'induction_rejections': rejected, 'builder_responses': builders,
        'receipt_projections': projections, 'protocol_replays': failures,
        'learning_boundary': 'Host-only diagnostics; Val/probe predictions and responses excluded from learning material'}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'd0_host_audit.json', result)
    write_json(output / 'd0_identity.json', {'host_audit_sha256': sha(output / 'd0_host_audit.json')})
    write_json(output / 'r2_reuse_material.json', reuse)
    write_json(output / 'r2_exposure.json', {'source_ids': sorted(exposed), 'r2_manifest_hash': digest(files)})
    return result
