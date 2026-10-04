"""Bounded preparation learning from two completed Train cases, then ordinary frozen use."""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path

import yaml

from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.checkpoint import TaskCheckpoint
from atomic_skillgraph.empirical.contracts import digest
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.core.errors import BudgetExhausted
from atomic_skillgraph.harness.registry import create_simple_harness
from .run_empirical import code_identity, load_env, resolve_alfworld_tasks, write_json


SUBJECTS = ('alfworld_train_1_pick_and_place_simple', 'alfworld_train_30_look_at_obj_in_light')
LOCAL_GOAL = ('Find and acquire the queried object using only the public scene, including necessary navigation/opening; '
              'return its concrete object_id. Use target_query as the ordinary category input. '
              'Preserve this preparation goal; final placement or finding a lamp is outside this capability.')


def public_experience(trace):
    if 'tools' in trace:
        return {key: deepcopy(trace[key]) for key in ('tools', 'score', 'execution')}
    return {'tools': [{'name': a['action_type'], 'arguments': a['arguments'],
        'result': {key: a.get(key) for key in ('accepted', 'observation', 'done', 'won')}}
        for a in trace['environment_actions']],
        'score': {'hard': trace['benchmark_success'], 'raw_score': trace.get('official_score')},
        'execution': {'reason': 'historical_train_experience', 'attempts': []}}


def verify(output):
    result = json.loads((output / 'acceptance.json').read_text())
    identity = json.loads((output / 'execution_manifest.json').read_text())
    with_bank = Bank(output / 'bank', readonly=True)
    frozen = Bank(output / 'frozen_bank', readonly=True)
    try:
        if with_bank.digest() != result['train_bank_digest'] or frozen.digest() != result['freeze']['digest']:
            raise ValueError('Acceptance Bank identity changed')
        if identity['code'] != code_identity():
            raise ValueError('Acceptance strategy/source identity changed')
        verified = {'code': identity['code'], 'passed': result['passed'], 'new_model_calls': 0,
                    'train_bank_digest': with_bank.digest(), 'frozen_digest': frozen.digest()}
        write_json(output / 'verification_manifest.json', verified)
        return verified
    finally:
        with_bank.close(); frozen.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/alfworld_empirical_seed42.yaml')
    parser.add_argument('--materials', required=True, help='Completed pilot root; selection may be referenced by train/config.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--env-file')
    parser.add_argument('--resume-verification', action='store_true', help='Read and verify existing acceptance artifacts; no model or environment calls')
    args = parser.parse_args()
    output = Path(args.output)
    if args.resume_verification:
        print(json.dumps(verify(output)), flush=True)
        return
    if args.env_file: load_env(args.env_file)
    source = Path(args.materials).resolve()
    selection_path = source / 'selection.json'
    if not selection_path.exists():
        selection_path = Path(json.loads((source / 'train/config.json').read_text())['manifest'])
    selection = json.loads(selection_path.read_text())
    by_id = {row['task_id']: row for row in selection['train']}
    if not set(SUBJECTS).issubset(by_id):
        raise ValueError('Fixed two Train subjects are missing; do not substitute tasks')
    traces = {}
    trace_root = source / 'train/traces'
    if not trace_root.exists(): trace_root = source / 'train12/traces'
    for path in trace_root.glob('*.json'):
        trace = json.loads(path.read_text())
        if trace['task']['task_id'] in SUBJECTS:
            traces[trace['task']['task_id']] = (path, trace)
    if set(traces) != set(SUBJECTS): raise ValueError('The two completed Train traces are required')
    output.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(Path(args.config).read_text())
    config['data_dir'] = str(output / 'bank')
    config['manifest'] = str(selection_path)
    config['experiment'].update(output_dir=str(output), runtime_mode='online')
    adapter = create_simple_harness(config)
    tasks = resolve_alfworld_tasks(adapter, [by_id[k] for k in SUBJECTS], mapping_path=output/'task_identity_resolution.json')
    if any(task.physical_key != traces[key][1]['task']['physical_key'] for key,task in zip(SUBJECTS,tasks)):
        raise ValueError('Completed experience and current physical Train identity differ')
    system = EmpiricalSystem(config, harness=adapter, adapter_factory=lambda: create_simple_harness(config))
    system.audit_path = output / 'requests/learning.json'
    identity = {'code': code_identity(), 'config': system.config, 'source': str(source),
                'tasks': [asdict(t) for t in tasks], 'formal_score': False,
                'source_traces': {key: {'path': str(path), 'digest': digest(trace)} for key,(path,trace) in traces.items()}}
    write_json(output / 'execution_manifest.json', identity)
    write_json(output / 'config.json', system.config)
    print(json.dumps({'event': 'resolved_config', 'config': system.config}, ensure_ascii=False), flush=True)
    learning = []
    job_snapshots = []
    try:
        for index, (key,task) in enumerate(zip(SUBJECTS,tasks)):
            system.adapter.reset(task)
            system.phase, system.budget_scope = 'train', task.physical_key
            system.checkpoint = TaskCheckpoint(output / 'checkpoints' / str(index))
            try:
                log = system.learner.learn(task, public_experience(traces[key][1]), focus=LOCAL_GOAL +
                    (' Request this Program and defer realization until the second applicable completed Train case is available.' if index == 0 else
                     ' Select the existing pending Skill and realize it with explicit fixed bindings for both completed Train cases.'))
            except (ValueError, SyntaxError, BudgetExhausted) as exc:
                log = {'error': str(exc), 'program': None}
            learning.append(log)
            job_snapshots.append(system.bank.jobs())
            write_json(output / 'learning.json', learning)
        jobs = system.bank.jobs()
        program_ids = {j['program_id'] for j in jobs if j.get('program_id')}
        programs = [system.bank.get(key) for key in sorted(program_ids)]
        pending_skills = {j['skill_id'] for j in job_snapshots[0] if not j.get('program_id')}
        missing_job_observed = bool(pending_skills and any(j['skill_id'] in pending_skills and j.get('program_id') for j in jobs))
        qualified = [p for p in programs if p['state'] == 'usable' and len({t['task_key'] for t in system.bank.attempts(p['id'])
            if t['origin'] == 'train_test' and t['outcome'] == 'positive' and t.get('result', {}).get('calls', 0) >= 2}) == 2]
        frozen = system.bank.freeze(output / 'frozen_bank')
        result = {'learning': learning, 'jobs': jobs, 'job_snapshots': job_snapshots, 'freeze': frozen, 'train_bank_digest': system.bank.digest(),
                  'missing_program_job_observed': missing_job_observed,
                  'state': qualified[0]['state'] if qualified else 'candidate' if programs else None,
                  'passed': False, 'formal_score': False,
                  'usage': [e.to_dict() for e in system.usage.events]}
        write_json(output / 'acceptance.json', result)
        if qualified:
            evaluation_config = deepcopy(system.config)
            evaluation_config['data_dir'] = str(output / 'frozen_bank')
            evaluation_config['experiment'].update(runtime_mode='frozen', output_dir=str(output / 'ordinary_use'))
            evaluation = EmpiricalSystem(evaluation_config, readonly=True)
            evaluation.audit_path = output / 'requests/ordinary_use.json'
            try:
                before = evaluation.bank.digest()
                trace = evaluation.run_task(replace(tasks[0], split='val'), learn=False)
                write_json(output / 'ordinary_use/trace.json', trace)
                selected = [a for a in trace['execution']['attempts'] if a['program_id'] in {p['id'] for p in qualified}]
                result.update(ordinary_execution=trace, frozen_unchanged=before == evaluation.bank.digest(),
                    mechanism_observed=bool(selected and any(a['status']=='ok' and a['calls']>=2 and a['outputs_consumed'] for a in selected)))
                result['passed'] = missing_job_observed and result['mechanism_observed'] and result['frozen_unchanged']
            except Exception as exc:
                result['ordinary_execution_error'] = str(exc)
                raise
            finally:
                result['usage'].extend(e.to_dict() for e in evaluation.usage.events)
                result['total_tokens'] = sum(u['total_tokens'] for u in result['usage'])
                write_json(output / 'acceptance.json', result)
                evaluation.close()
        result['total_tokens'] = sum(u['total_tokens'] for u in result['usage'])
        write_json(output / 'acceptance.json', result)
        print(json.dumps({k: result.get(k) for k in ('state','passed','mechanism_observed','frozen_unchanged','total_tokens')}), flush=True)
    finally: system.close()


if __name__ == '__main__': main()
