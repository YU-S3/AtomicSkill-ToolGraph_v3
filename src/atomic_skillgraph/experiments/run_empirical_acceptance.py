"""One genuine Learner/Builder preparation candidate from two existing Train cases."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import yaml

from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.harness.registry import create_simple_harness
from .run_empirical import code_identity, load_env, resolve_alfworld_tasks, write_json


def public_experience(trace):
    return {"tools": [{"name": action['action_type'], "arguments": action['arguments'],
        "result": {key: action.get(key) for key in ('accepted', 'observation', 'done', 'won')}}
        for action in trace['environment_actions']],
        "score": {"hard": trace['benchmark_success'], "raw_score": trace.get('official_score')},
        "execution": {"reason": "historical_train_experience", "attempts": []},
        "source_trace_id": trace['trace_id']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/alfworld_empirical_seed42.yaml')
    parser.add_argument('--materials', required=True, help='Existing pilot root containing selection/train12/traces')
    parser.add_argument('--output', required=True)
    parser.add_argument('--env-file')
    parser.add_argument('--resume-verification', action='store_true', help='Verify an already registered, fully tested candidate without new model calls')
    args = parser.parse_args()
    if args.env_file:
        load_env(args.env_file)
    output = Path(args.output)
    if not args.resume_verification:
        output.mkdir(parents=True, exist_ok=False)
    elif not (output / 'execution_manifest.json').is_file():
        raise ValueError('No previous acceptance manifest')
    config = yaml.safe_load(Path(args.config).read_text())
    config['data_dir'] = str(output / 'bank')
    config['manifest'] = str(Path(args.materials)/'selection.json')
    config['experiment']['output_dir'] = str(output)
    source = Path(args.materials)
    selection = json.loads((source / 'selection.json').read_text())
    subjects = [row for row in selection['train'] if row['task_id'] in {
        'alfworld_train_1_pick_and_place_simple', 'alfworld_train_30_look_at_obj_in_light'}]
    if len(subjects) != 2:
        raise ValueError('Fixed two Train subjects are missing; do not substitute tasks')
    adapter = create_simple_harness(config)
    tasks = resolve_alfworld_tasks(adapter, subjects, mapping_path=output/'task_identity_resolution.json')
    def factory():
        candidate = create_simple_harness(config)
        return candidate
    system = EmpiricalSystem(config, harness=adapter, adapter_factory=factory)
    system.audit_path = output / 'requests.json'
    identity = {'code': code_identity(), 'config': config,
        'source': str(source), 'tasks': [asdict(t) for t in tasks], 'formal_score': False}
    if args.resume_verification:
        original = json.loads((output / 'execution_manifest.json').read_text())
        if any(original[k] != identity[k] for k in ('config', 'source', 'tasks')):
            raise ValueError('Acceptance task/config identity changed')
        write_json(output / 'verification_manifest.json', identity)
    else:
        write_json(output / 'execution_manifest.json', identity)
    traces = {trace['task']['task_id']: trace for path in (source / 'train12/traces').glob('trace_*.json')
              if (trace := json.loads(path.read_text()))['task']['task_id'] in {t.task_id for t in tasks}}
    try:
        if args.resume_verification:
            requests = json.loads((output / 'requests.json').read_text())
            generated = [r['response']['tool_calls'][0]['arguments'] for r in requests['requests']
                if r['stage'] == 'tool_builder' and r.get('response', {}).get('tool_calls')]
            programs = [p for p in system.bank.all('program') if generated and p['source'] == generated[-1]['source']]
            if len(programs) != 1:
                raise ValueError('Cannot identify the actual registered Builder output')
            tests = system.bank.attempts(programs[0]['id'])
            if {t['task_key'] for t in tests} != {t.physical_key for t in tasks}:
                raise ValueError('Registered candidate lacks the fixed Train trials')
            learned = {'program': programs[0]['id'], 'tests': tests, 'trial_inputs': generated[-1]['trial_inputs'],
                       'recovered_registered_candidate': True}
            usage = requests['usage']
        else:
            first = public_experience(traces[tasks[0].task_id])
            system.learner.cases.append((tasks[0], {'task': {'goal': tasks[0].goal, 'inputs': tasks[0].inputs},
                'events': first['tools'], 'score': first['score'], 'result': first['execution']}))
            system.adapter.reset(tasks[1])
            system._learning_start = len(system.usage.events)
            learned = system.learner.learn(tasks[1], public_experience(traces[tasks[1].task_id]),
                focus='Find and acquire the queried object from the public scene, including necessary navigation/opening; '
                      'return its concrete object_id. Use target_query as the ordinary category input. '
                      'This acceptance test focuses on a multi-action preparation capability, not final task placement.')
            usage = [e.to_dict() for e in system.usage.events]
        frozen = system.bank.freeze(output / 'frozen_bank')
        program = system.bank.get(learned['program']) if learned.get('program') else None
        result = {'learning': learned, 'freeze': frozen, 'state': program.get('state') if program else None,
                  'usage': usage, 'passed': bool(program and program['state'] == 'usable' and
                    len([t for t in learned['tests'] if t['outcome'] == 'positive' and t['calls'] >= 2]) == 2)}
        if result['passed']:
            readonly = Bank(output / 'frozen_bank', readonly=True)
            test_adapter = factory()
            from atomic_skillgraph.harness.simple_protocol import Broker
            test_adapter.reset(tasks[0])
            before = readonly.digest()
            inputs = next((row['inputs'] for row in learned.get('trial_inputs', [])
                           if row['case_id'] == tasks[0].physical_key), None)
            if inputs is None:
                raise ValueError('Generated binding for this TrialCase was not logged')
            from atomic_skillgraph.empirical.executor import Executor
            from atomic_skillgraph.empirical.planner import Planner
            system.phase, system.budget_scope = 'val', 'acceptance_frozen_continuation'
            plan = {'goal':tasks[0].goal, 'nodes':[
                {'id':'prepare', 'goal':'Run the learned preparation', 'program_id':program['id'],
                 'args':{key:{'literal':value} for key,value in inputs.items()}},
                {'id':'continue', 'goal':tasks[0].goal,
                 'args':{key:{'from':'prepare','field':key} for key in program['output_schema'].get('properties', {})}}]}
            execution = Executor(readonly,system.agent,system.worker,Planner(readonly,system.agent),frozen=True).run(
                tasks[0],test_adapter,Broker(test_adapter,100,step_limit=100),plan)
            invoked = next(row['result'] for row in execution['history'] if row.get('program')==program['id'])
            result['ordinary_execution'] = execution
            result['continuation_score'] = test_adapter.evaluate(test_adapter.submit(execution['prediction']))
            result['frozen_invocation'] = invoked
            result['frozen_unchanged'] = before == readonly.digest()
            result['passed'] &= invoked['status'] == 'ok' and invoked['calls'] >= 2 and result['frozen_unchanged']
            readonly.close()
            test_adapter.close()
        result['usage'] = [e.to_dict() for e in system.usage.events] if not args.resume_verification else [*usage,*[e.to_dict() for e in system.usage.events]]
        write_json(output / 'acceptance.json', result)
        print(json.dumps({k: v for k, v in result.items() if k in {'state', 'passed', 'frozen_unchanged'}}), flush=True)
    finally:
        system.close()


if __name__ == '__main__':
    main()
