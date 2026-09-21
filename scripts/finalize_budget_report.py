"""Audited report-only overlay for an already completed historical train run.

Never changes the historical checkout, manifests, or Trace bytes. The original
runner still enforces its config, provider, knowledge, usage and freeze gates.
"""
import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys


def report_function(source):
    return next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == '_r4_learning_metrics')


def checked_overlay(old_source, fixed_source):
    original = report_function(old_source)
    expected = copy.deepcopy(original)
    assignments = [n for n in ast.walk(expected) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'allowed_outcomes' for t in n.targets)]
    if len(assignments) != 1:
        raise ValueError('unexpected historical reporting function')
    allowed = assignments[0].value
    if not (isinstance(allowed, ast.Call) and isinstance(allowed.func, ast.Name)
            and allowed.func.id == 'frozenset' and len(allowed.args) == 1
            and isinstance(allowed.args[0], ast.Set)):
        raise ValueError('unexpected outcome declaration')
    if any(isinstance(n, ast.Constant) and n.value == 'budget_exhausted' for n in allowed.args[0].elts):
        raise ValueError('historical reporter is already fixed')
    allowed.args[0].elts.append(ast.Constant('budget_exhausted'))
    fixed = report_function(fixed_source)
    if ast.dump(expected) != ast.dump(fixed):
        raise ValueError('report overlay contains changes beyond the reviewed outcome addition')
    return ast.fix_missing_locations(ast.Module(body=[fixed], type_ignores=[]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--finalize', action='store_true')
    args = ap.parse_args()
    root = args.root.resolve()
    fixed_path = Path(__file__).resolve().parents[1] / 'experiments/report.py'
    old_path = root / 'experiments/report.py'
    overlay = checked_overlay(old_path.read_text(), fixed_path.read_text())
    sys.path[:0] = [str(root), str(root / 'src')]
    from atomic_skillgraph.system import load_config
    from experiments import report, run_v3_train as train
    from experiments.protocol import code_file_manifest, hash_code, hash_config, AttemptTraceLedger
    from atomic_skillgraph.agents.provider_probe import ensure_provider_capability
    config_path = root / args.config
    config = load_config(config_path)
    output = train._path(config['experiment']['output_dir'])
    execution = json.loads((output/'execution_code_manifest.json').read_text())
    if code_file_manifest(root) != execution['files'] or hash_code(root) != execution['code_hash']:
        raise RuntimeError('historical source inventory changed')
    if hash_config(config_path) != execution['config_hash']:
        raise RuntimeError('historical config changed')
    with sqlite3.connect(f'file:{output}/data_v3/state.sqlite3?mode=ro', uri=True) as connection:
        rows = connection.execute('select state from run_tasks where run_id=?', (config['experiment']['name'],)).fetchall()
        expected_count = train._train_protocol(config)[3]
        if len(rows) != expected_count or any(r[0] != 'completed' for r in rows):
            raise RuntimeError('report-only recovery requires all tasks completed')
        last = connection.execute('select result_json from run_tasks where run_id=? order by rowid desc limit 1', (config['experiment']['name'],)).fetchone()
        if not json.loads(last[0]).get('final_batch_maintenance'):
            raise RuntimeError('final maintenance evidence missing')
    if (output/'.task_checkpoint').exists():
        raise RuntimeError('unfinished checkpoint: not a report-only failure')
    ledger = AttemptTraceLedger(output/'attempt_history', output/'traces')
    if ledger.pending(run_id=config['experiment']['name']) or ledger.unresolved(run_id=config['experiment']['name']):
        raise RuntimeError('unsettled attempts: cannot waive missing evidence')
    ensure_provider_capability(config, output_dir=output, config_hash=hash_config(config_path), code_hash=hash_code(root), run_if_missing=False)
    trace_paths = sorted((output/'traces').glob('trace_*.json'))
    trace_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in trace_paths}
    scope = {}
    exec(compile(overlay, str(fixed_path), 'exec'), report.__dict__, scope)
    report._r4_learning_metrics = scope['_r4_learning_metrics']
    payloads = [json.loads(p.read_text()) for p in trace_paths]
    report.validate_formal_usage(payloads)
    receipt = dict(kind='report_only_budget_outcome_overlay', original_execution_code_hash=hash_code(root),
                   original_report_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
                   corrected_report_sha256=hashlib.sha256(fixed_path.read_bytes()).hexdigest(),
                   wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   change='accept budget_exhausted; all other reporting checks unchanged',
                   completed_tasks=len(rows), validated_trace_count=len(trace_paths), source_traces=trace_hashes,
                   created_at=datetime.now(timezone.utc).isoformat())
    if not args.finalize:
        print(json.dumps(dict(passed=True, completed_tasks=len(rows), validated_trace_count=len(trace_paths), api_requests=0, formal_started=False)))
        return 0
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup = root.parent / f'report_recovery_backup_{root.name}_{stamp}'
    shutil.copytree(output, backup)
    receipt['backup'] = str(backup)
    (output/'report_only_recovery.json').write_text(json.dumps(receipt, indent=2)+'\n')
    result = train.run(config_path, resume=True)
    for name, expected in trace_hashes.items():
        if hashlib.sha256((output/'traces'/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('original Trace changed during finalization')
    receipt['finalized'] = result == 0
    (output/'report_only_recovery.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return result


if __name__ == '__main__':
    raise SystemExit(main())
