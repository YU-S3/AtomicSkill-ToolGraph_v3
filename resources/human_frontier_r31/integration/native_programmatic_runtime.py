"""Execute authored Workflow references using native, isolated Programs."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import os

METHOD = 'human.frontier.livemath.programmatic.r3.1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def load_bundle(path):
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    if obj.get('method_version') != METHOD or obj.get('benchmark') != 'livemath':
        raise ValueError('R3.1 programmatic bank identity mismatch')
    return obj


def fixture_worker(program, arguments):
    """Explicit offline fixture only; the production bridge always uses Docker."""
    namespace = {}
    exec(compile(program['source'], '<offline-fixture>', 'exec'), namespace)
    return namespace['run'](None, deepcopy(arguments))


def docker_worker(adapter, settings=None):
    from atomic_skillgraph.empirical.program_worker import ProgramWorker
    from atomic_skillgraph.empirical.math_program_probe import PureBroker
    from atomic_skillgraph.empirical.task_context import TaskContext
    from atomic_skillgraph.empirical.contracts import validate_schema_instance
    worker = ProgramWorker({'record_execution_receipt': True, **(settings or {})})
    def execute(program, arguments):
        validate_schema_instance(arguments, program['input_schema'])
        if program.get('permission') != 'author_trusted' or program.get('allowed_tools'):
            raise ValueError('Only manual read-only Programs are permitted')
        pure = PureBroker(adapter, 1, context=TaskContext({}))
        try:
            result = worker.execute(program, arguments, pure)
        except OSError as exc:
            result = {'status': 'execution_error', 'error_code': 'docker_control_failed',
                      'detail': str(exc), 'program_id': program['id']}
        if pure.events:
            raise RuntimeError('Pure authored Program attempted RPC')
        return result
    return execute


def run_programmatic_graph(task, bank, *, variant='benchmark_prior', run_worker=None,
                           dynamic_answer=None, stage_dir=None):
    from atomic_skillgraph.empirical.contracts import ValueStore, validate_schema_instance, validate_workflow
    from atomic_skillgraph.empirical.checkpoint import save_json
    if variant not in {'benchmark_prior', 'theorem_reasoning'} or run_worker is None:
        raise ValueError('Explicit valid variant and Worker required')
    workflows = {w['id']: w for w in bank.all('workflow')}
    prior = workflows['workflow_lm_r31_meta_prior']
    reasoning = workflows['workflow_lm_r31_claim_reasoning']
    for workflow in (prior, reasoning):
        validate_workflow(workflow, bank=bank)
    if prior['nodes'][0] != reasoning['nodes'][0]:
        raise ValueError('Shared scan occurrence differs between authored workflows')
    programs = {p['id']: p for p in bank.all('program')}
    links = {i['program_id']: i for i in bank.all('implementation')}
    values = ValueStore(task)
    trace = []
    folder = Path(stage_dir) if stage_dir else None
    if folder:
        folder.mkdir(parents=True, exist_ok=True)

    def resolve(refs):
        arguments, missing = values.resolve(refs)
        if missing:
            raise ValueError('Missing actual Workflow dataflow: ' + json.dumps(missing))
        return arguments

    def consumed(refs, consumer):
        for ref in refs.values():
            if 'from' in ref:
                row = next(r for r in trace if r['node'] == ref['from'])
                row['outputs_consumed'] = True
                row.setdefault('consumers', []).append({'node': consumer, 'field': ref['field']})

    def stage(node):
        if not set(node.get('after', [])).issubset(values.results):
            raise ValueError('Workflow after dependency has no successful result')
        program = programs[node['program_id']]
        impl = links.get(program['id'])
        if impl is None:
            raise ValueError('Program occurrence has no real Implementation')
        args = resolve(node['args'])
        validate_schema_instance(args, program['input_schema'])
        key = digest([METHOD, task.__dict__, node, digest(program), impl, args])
        path = folder / (node['id'] + '.json') if folder else None
        claim = folder / (node['id'] + '.started.json') if folder else None
        cached = bool(path and path.exists())
        if cached:
            receipt = json.loads(path.read_text())
            if receipt['stage_key'] != key or receipt['receipt_hash'] != digest({k:v for k,v in receipt.items() if k != 'receipt_hash'}):
                raise ValueError('Saved Program stage changed')
            result = receipt['result']
        else:
            if claim:
                with claim.open('x') as handle:
                    handle.write(json.dumps({'stage_key': key})); handle.flush(); os.fsync(handle.fileno())
            result = run_worker(program, args)
            receipt = {'stage_key': key, 'program_id': program['id'], 'implementation_id': impl['id'],
                       'node': node, 'arguments': args, 'result': result}
            receipt['receipt_hash'] = digest(receipt)
            if path:
                save_json(path, receipt)
        row = {'node': node['id'], 'mode': 'program', 'program_id': program['id'],
               'implementation_id': impl['id'], 'status': result.get('status'), 'cached': cached,
               'outputs_consumed': False, 'receipt_hash': receipt['receipt_hash'],
               'arguments_hash': digest(args), 'result': result}
        trace.append(row)
        consumed(node['args'], node['id'])
        if result.get('status') != 'ok':
            return None
        validate_schema_instance(result['outputs'], program['output_schema'])
        values.publish(node['id'], result['outputs'], 'program')
        return result['outputs']

    scan = stage(reasoning['nodes'][0])
    selected = prior if variant == 'benchmark_prior' and scan and scan['meta_count'] == 1 else reasoning
    branch = 'meta_prior' if selected is prior else 'theorem_reasoning'
    origin = 'none'
    model_receipt = None
    answer = None
    fallback = scan is None
    if not fallback:
        result = stage(selected['nodes'][1])
        fallback = result is None
        if selected is prior and not fallback:
            answer = resolve(prior['outputs'])['answer']
            consumed(prior['outputs'], 'workflow.outputs')
            origin = 'programmatic_benchmark_prior'
    handoff = None
    if origin != 'programmatic_benchmark_prior':
        node = reasoning['nodes'][-1]
        if fallback:
            branch = 'dynamic_fallback'
            handoff = {'goal': task.goal, 'choices': deepcopy(task.inputs['choices']),
                       'features': [], 'contrasts': []}
        else:
            handoff = resolve(node['args'])
            consumed(node['args'], node['id'])
        handoff['limitation'] = 'surface_differences_only_no_truth_verification'
        if dynamic_answer:
            value = dynamic_answer(task, handoff, node)
            if isinstance(value, tuple):
                answer, model_receipt = value
            else:
                answer = value
            origin = 'dynamic_model'
            values.publish(node['id'], {'answer': answer}, 'dynamic')
            answer = resolve(reasoning['outputs'])['answer']
            trace.append({'node': node['id'], 'mode': 'dynamic',
                          'status': 'ok' if answer in {c['label'] for c in task.inputs['choices']} else 'invalid_answer',
                          'outputs_consumed': True, 'answer_label': answer,
                          'handoff_hash': digest(handoff), 'handoff_exposure': True,
                          'semantic_use': 'unknown'})
    valid = answer in {c['label'] for c in task.inputs['choices']}
    return {'status': 'ok' if valid else 'needs_dynamic' if not dynamic_answer else 'invalid_answer',
            'decision_origin': origin, 'branch': branch, 'answer': answer,
            'truth_authority': 'public_benchmark_prior_not_math_proof' if origin == 'programmatic_benchmark_prior' else 'dynamic_model',
            'graph': selected['id'], 'workflow_hash': digest(selected), 'fallback_to': reasoning['id'] if fallback else None,
            'program_nodes': trace, 'handoff': handoff,
            'worker_invocations_this_run': sum(r['mode'] == 'program' and not r['cached'] for r in trace),
            'program_invocations_total': sum(r['mode'] == 'program' for r in trace),
            'program_results_consumed': sum(r['mode'] == 'program' and r['outputs_consumed'] for r in trace),
            'dynamic_model_receipt': model_receipt}
