"""Import and independently validate a manual Bank, then run canonical Test102.

This is an authored reference control, not a naturally learned Ours Bank.
Qualification uses the unchanged Host local-validation checks and real Docker.
No training, evaluator access during qualification, or candidate bypass exists.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from ..empirical.bank import Bank
from ..empirical.budget_governor import BudgetGovernor
from ..empirical.contracts import PublicTask, digest
from ..empirical.local_validation import canonical_binding, evidence_from_trace, validate_on_source
from ..empirical.program_worker import ProgramWorker
from ..empirical.task_context import TaskContext
from ..harness.benchmarks import OfficeAdapter
from ..harness.simple_protocol import Broker
from .canonical_manifest import sha256, verify
from .formal_log import FormalLog, tree_identity, utc
from .office_reference_oracles import REFERENCES, cases
from .run_empirical import code_identity, load_env, run, write_json

VERSION = 'officeqa.manual-reference-test.v1'
TEST_TOKENS = 102 * 192000
V4_RESERVATION = {'tokens': 1800000, 'http': 120}


def read(path):
    return json.loads(Path(path).read_text())


def import_bank(source, destination, config):
    data = read(source / 'banks/officeqa/bank.json')
    if (data['schema'] != 'manual_reference_bank.v2' or data['benchmark'] != 'officeqa'
            or data['intended_use'] != 'manual_reference_oracle_free_not_learned_ours'
            or data['program_environment'] != config['program_environment']):
        raise ValueError('Unsupported manual Bank/environment identity')
    bank = Bank(destination, seed=42, environment=config['program_environment'])
    bank.bind_office_method(config['experiment']['officeqa_method_version'])
    try:
        names = {}
        sources = {p.read_text(): p.stem for p in (source / 'programs/officeqa').glob('*.py')}
        for kind, key in [('skill', 'skills'), ('program', 'programs'), ('implementation', 'implementations'), ('workflow', 'workflows')]:
            for asset in data[key]:
                if kind == 'program':
                    if asset['state'] != 'candidate' or asset.get('local_validation'):
                        raise ValueError('Imported code must be an unqualified candidate')
                    names[asset['id']] = sources[asset['source']]
                stored = bank.put(kind, asset)
                if stored['id'] != asset['id']:
                    raise ValueError('Import changed asset ID')
        if any(bank.program_eligible(p) for p in bank.all('program')):
            raise ValueError('Import must not establish execution qualification')
        return bank, names
    except BaseException:
        bank.close()
        raise


def probe(bank, program, case, config, folder):
    folder.mkdir(parents=True, exist_ok=False)
    task = PublicTask('reference:' + case['name'] + ':' + case['label'], digest(case),
                      'Manual public-input contract probe; not an official task', case['inputs'], 'reference_probe')
    cfg = deepcopy(config)
    cfg['harness']['corpus_root'] = case['corpus']
    cfg['experiment']['output_dir'] = str(folder)
    factory = lambda: OfficeAdapter({task.task_id: {}}, cfg)
    adapter = factory()
    adapter.reset(task)
    broker = Broker(adapter, 48, context=TaskContext(cfg['runtime']))
    worker = ProgramWorker({**cfg['program_worker'], 'record_execution_receipt': True})
    try:
        if case['status'] != 'ok':
            result = worker.execute(program, case['inputs'], broker)
            receipt = dict(case=case, task=asdict(task), result=result, tools=broker.events,
                           passed=result['status'] == case['status'], qualification=False)
        else:
            # Execute the independent diagnostic, then bind only its actual public output.
            structured = case['name'] == 'read_hit'  # Full 12K pages exceed temporary Python stdout.
            if structured:
                source = 'def run(ctx, inputs):\n' + '\n'.join('    '+line for line in REFERENCES[case['name']].splitlines())
                source += "\n    return {'status':'ok','outputs':out}\n"
                reference = worker.execute({**program, 'id':'reference_'+digest(source), 'source':source}, case['inputs'], broker)
                if reference['status'] != 'ok':
                    raise RuntimeError('Structured reference failed: ' + str(reference))
                reference_id = broker.context.register('manual-reference', reference)
                trace = {'tools':broker.events, 'result_store':{'program_results':{reference_id:reference}},
                    'execution':{'temporary_executions':[{'id':'manual-reference', 'status':'ok', 'source':source,
                        'arguments':case['inputs'], 'input_schema':program['input_schema'],
                        'output_schema':program['output_schema'], 'result_id':reference_id}]}}
            else:
                source = 'import json\ninputs = json.loads(' + repr(json.dumps(case['inputs'])) + ')\n'
                source += REFERENCES[case['name']] + '\nprint(json.dumps(out))\n'
                reference = broker.call('execute_python', {'source': source})
                if not reference.get('accepted') or not isinstance(reference.get('data', {}).get('computed'), dict):
                    write_json(folder / 'reference_error.json', {'result': reference, 'tools': broker.events})
                    raise RuntimeError('Independent reference did not execute: ' + case['name'])
                trace = {'tools': broker.events}
            evidence = evidence_from_trace(task, trace, config['program_environment'])
            experience = {'local_evidence': evidence, 'events': broker.events}
            selected = next(e for e in evidence if e['kind'] == 'executed_python') if structured else next(
                e for e in evidence if e['operation'].get('name') == 'execute_python')
            binding = canonical_binding({'local_evidence_ref': selected['id'], 'inputs': case['inputs'],
                'input_refs': {k: {'kind': 'public_json', 'path': ['inputs', k]} for k in case['inputs']},
                'reference_fields': {k: [k] if structured else ['data', 'computed', k]
                                     for k in program['output_schema']['required']}}, experience, program)
            system = SimpleNamespace(bank=bank, adapter=adapter, adapter_factory=factory, worker=worker,
                                     config=cfg, budget_governor=None)
            trial = validate_on_source(system, program, binding, task, experience, 'manual:' + digest(case),
                                       validation_origin='manual_reference_validation')
            receipt = dict(case=case, task=asdict(task), reference_trace=trace, binding=binding, trial=trial,
                           passed=trial['validation']['passed'], qualification=True)
        write_json(folder / 'receipt.json', receipt)
        if not receipt['passed']:
            raise RuntimeError('Manual reference contract failed: ' + str(folder))
        return {'name': case['name'], 'label': case['label'], 'passed': True, 'receipt_sha256': sha256(folder/'receipt.json')}
    finally:
        adapter.close()


def workflow_probe(bank, source, config, root):
    from ..empirical.executor import Executor
    from ..empirical.planner import Planner
    scenario = read(source / 'fixtures/scenarios/office_complete_table_growth.json')
    task = PublicTask('reference:workflow', digest(scenario), 'Fixture workflow', scenario, 'reference_probe')
    cfg = deepcopy(config)
    cfg['harness']['corpus_root'] = str(source / 'fixtures/office_corpus')
    cfg['experiment']['output_dir'] = str(root / 'workflow_probe')
    adapter = OfficeAdapter({task.task_id: {}}, cfg)
    adapter.reset(task)
    broker = Broker(adapter, 48, context=TaskContext(cfg['runtime']))
    def no_model(*args, **kwargs):
        raise AssertionError('Program-only fixture must not call a model')
    try:
        workflow = next(w for w in bank.all('workflow') if len(w['nodes']) == 4)
        executor = Executor(bank, no_model, ProgramWorker({**cfg['program_worker'], 'record_execution_receipt': True}),
                            Planner(bank, no_model), frozen=True)
        result = executor.run(task, adapter, broker, workflow)
        write_json(root / 'workflow_probe.json', {'execution': result, 'tools': broker.events})
        if result['prediction'] != '25.00%' or len(result['attempts']) != 4:
            raise RuntimeError('Production workflow/dataflow integration failed')
    finally:
        adapter.close()


def prepare(source, parent, root):
    if root.exists():
        raise ValueError('Reference Test requires a new independent directory')
    config = deepcopy(read(parent / 'resolved_config.json')['test'])
    ledger = read(parent / 'budget.json')
    if ledger['unknown_billing'] or any(a['status'] != 'metered' for a in ledger['attempts'].values()):
        raise ValueError('Unresolved historical billing')
    historical = sum(a['accounted_tokens'] for a in ledger['attempts'].values())
    if historical + TEST_TOKENS + V4_RESERVATION['tokens'] > ledger['limits']['token_limit']:
        raise ValueError('Test and the separate authorized V4 pilot exceed remaining parent authority')
    root.mkdir(parents=True)
    shutil.copytree(source, root / 'input', ignore=shutil.ignore_patterns('__pycache__', 'dependencies'))
    source = root / 'input'
    config['data_dir'] = str(root / 'frozen_bank')
    config['experiment'].update(output_dir=str(root / 'test'), reference_control=VERSION,
                                asset_origin='human_authored_not_naturally_learned')
    config['program_worker']['record_execution_receipt'] = True
    # Reserve both child maxima up front; no two-process writes to a shared ledger.
    limits = dict(token_limit=historical + TEST_TOKENS, finish_reserve=0,
                  request_limit=ledger['limits']['request_limit'] - V4_RESERVATION['http'])
    ledger['limits'] = limits
    config['budget'].update(limits)
    write_json(root / 'budget.json', ledger)
    write_json(root / 'cost_carryforward_manifest.json', {
        'parent': str(parent), 'parent_ledger_sha256': sha256(parent/'budget.json'),
        'historical_tokens': historical, 'historical_http': len(ledger['attempts']),
        'test_token_cap': TEST_TOKENS, 'reserved_for_separate_v4': V4_RESERVATION,
        'parent_limits': read(parent/'budget.json')['limits'],
        'aggregate_max_tokens': historical + TEST_TOKENS + V4_RESERVATION['tokens'],
        'accounting': 'Historical attempts copied unchanged; add each child delta once, never sum inherited totals.'})
    write_json(root / 'config.json', config)
    bank, names = import_bank(source, root / 'bank', config)
    try:
        inputs = cases(source / 'fixtures/office_corpus', config['harness']['corpus_root'])
        write_json(root / 'probe_plan.json', inputs)
        programs = {names[p['id']]: p for p in bank.all('program')}
        if set(programs) != set(REFERENCES):
            raise ValueError('Reference diagnostic coverage must match the complete imported Bank')
        receipts = [probe(bank, programs[c['name']], c, config, root / 'probes' / str(i)) for i, c in enumerate(inputs)]
        if not all(bank.program_eligible(p) for p in bank.all('program')):
            raise RuntimeError('At least one imported candidate lacks genuine Host qualification')
        workflow_probe(bank, source, config, root)
        frozen = bank.freeze(root / 'frozen_bank')
        counts = {kind: len(bank.all(kind)) for kind in ('skill', 'program', 'implementation', 'workflow')}
    finally:
        bank.close()
    verify(Path(config['manifest']).parent.parent)
    public_path = Path(config['harness']['evaluator_records']).with_name('test.json')
    official = read(config['manifest'])['tasks']
    tasks = read(public_path)['tasks']
    if len(tasks) != 102 or [r['task_id'] for r in tasks] != [r['task_id'] for r in official]:
        raise ValueError('Test must be the unchanged complete canonical 102 in manifest order')
    write_json(root / 'tasks.json', {'tasks': tasks})
    write_json(root / 'preflight.json', {'passed': True, 'http': 0, 'counts': counts, 'probes': receipts,
                                       'workflow': 'four_program_production_executor_passed'})
    plan = dict(version=VERSION, created_at=utc(), code=code_identity(), source_archive=read(source/'archive_import_identity.json'),
        config_hash=digest(config), frozen=tree_identity(root/'frozen_bank'), freeze=frozen, counts=counts,
        files={str(p): sha256(p) for p in [root/'config.json', root/'tasks.json', root/'probe_plan.json', root/'preflight.json',
               root/'cost_carryforward_manifest.json', Path(config['manifest']), public_path, Path(config['harness']['evaluator_records'])]},
        input_identity=tree_identity(source), corpus_identity=tree_identity(config['harness']['corpus_root']),
        user_authorization='2026-10-15: 将这个bank落地接入，运行一次真实test',
        stop_after='canonical_test102', natural_learning_claim=False)
    write_json(root / 'plan.json', plan)
    return {k: plan[k] for k in ('version', 'code', 'counts', 'stop_after')}


def evaluate(root):
    plan, config = read(root / 'plan.json'), read(root / 'config.json')
    if plan['code'] != code_identity() or plan['code']['tracked_dirty']:
        raise ValueError('Run source must be the exact clean prepared revision')
    if any(sha256(Path(p)) != expected for p, expected in plan['files'].items()):
        raise ValueError('Immutable input/config changed')
    if (tree_identity(root/'input') != plan['input_identity']
            or tree_identity(config['harness']['corpus_root']) != plan['corpus_identity']
            or tree_identity(root/'frozen_bank') != plan['frozen']):
        raise ValueError('Corpus or manual Bank changed')
    phase = root / 'test'
    tasks = [PublicTask(**row) for row in read(root/'tasks.json')['tasks']]
    log = FormalLog(phase, {'method': VERSION, 'asset_origin': 'human_authored_not_naturally_learned',
        'benchmark': 'officeqa', 'run_seed': 42, 'phase': 'test', 'source_commit': plan['code']['git_sha'],
        'plan_hash': digest(plan), 'frozen_hash': plan['frozen']['sha256']}, config,
        resume=(phase/'run_manifest.json').exists())
    governor = BudgetGovernor(root/'budget.json', **read(root/'budget.json')['limits'])
    try:
        result = run(config, tasks, phase, readonly=True, resume=(phase/'execution_manifest.json').exists(),
            formal_log=log, budget_governor=governor,
            task_metadata={r['task_id']: r['task_metadata'] for r in read(config['manifest'])['tasks']})
        log.finish('completed' if result['complete'] else 'paused')
        return {k: v for k, v in result.items() if k != 'cases'}
    except BaseException as exc:
        log.finish('paused' if isinstance(exc, SystemExit) and exc.code == 75 else 'failed')
        write_json(root/'last_error.json', {'timestamp': utc(), 'type': type(exc).__name__, 'error': str(exc)})
        raise
    finally:
        actual = tree_identity(root/'frozen_bank')
        write_json(phase/'frozen_check.json', {'before': plan['frozen']['sha256'], 'after': actual['sha256'],
                                              'unchanged': actual == plan['frozen']})
        if actual != plan['frozen']:
            raise RuntimeError('Frozen Bank changed during Test')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'test'])
    parser.add_argument('--source'); parser.add_argument('--parent'); parser.add_argument('--output', required=True)
    parser.add_argument('--env-file')
    args = parser.parse_args()
    root = Path(args.output).resolve()
    if args.stage == 'prepare':
        result = prepare(Path(args.source).resolve(), Path(args.parent).resolve(), root)
    else:
        import fcntl
        with (root/'.test.lock').open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.env_file: load_env(args.env_file)
            result = evaluate(root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
