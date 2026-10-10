"""Host-bound local evidence and version-specific executable qualification."""
import ast
from pathlib import Path
import hashlib
import zipfile
import xml.etree.ElementTree as ET
from copy import deepcopy

from .contracts import digest, program_digest, validate_schema_instance
from . import LOCAL_VALIDATION_POLICY_VERSION, FILE_EFFECT_VERSION, BINDING_POLICY_VERSION

POLICY = LOCAL_VALIDATION_POLICY_VERSION


def artifact_identity(path):
    path = Path(path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            members = {n:archive.read(n) for n in sorted(archive.namelist())}
            if '[Content_Types].xml' in members and 'xl/workbook.xml' in members:
                types = ET.fromstring(members['[Content_Types].xml'])
                xlsx = any(t.attrib.get('PartName') == '/xl/workbook.xml' and t.attrib.get('ContentType') ==
                    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml' for t in types)
                if xlsx and 'docProps/core.xml' in members:
                    core = ET.fromstring(members['docProps/core.xml'])
                    for node in core.iter('{http://purl.org/dc/terms/}modified'):
                        node.text = 'NORMALIZED_SAVE_TIME'
                    members['docProps/core.xml'] = ET.tostring(core, encoding='utf-8')
            return digest({n:hashlib.sha256(data).hexdigest() for n,data in members.items()})
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_record(path):
    return {'algorithm': FILE_EFFECT_VERSION, 'effect_sha256': artifact_identity(path),
            'raw_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}


PRIVATE_FIELDS = {'task_id', 'physical_key', 'gold', 'correct_choice', 'answer_label', 'answers', 'correct_candidate'}


def source_literals(source):
    """Locatable literals; optional branch bodies are not evidence of execution."""
    result = []
    def walk(node, path):
        if isinstance(node, ast.Constant) and type(node.value) in {str, int, float, bool, type(None)}:
            result.append({'kind':'source_literal', 'path':path, 'value':node.value})
        for key,value in ast.iter_fields(node):
            if isinstance(node, (ast.If, ast.IfExp, ast.For, ast.While, ast.Try)) and key in {'body','orelse','finalbody','handlers'}:
                continue
            if isinstance(value, ast.AST): walk(value, [*path,key])
            elif isinstance(value, list):
                for i,item in enumerate(value):
                    if isinstance(item, ast.AST): walk(item, [*path,key,i])
    try: walk(ast.parse(source), [])
    except SyntaxError: pass
    return result


def input_sources(evidence):
    rows=[]
    def walk(value, path):
        if isinstance(value,dict):
            for key,item in value.items():
                if key not in PRIVATE_FIELDS: walk(item,[*path,key])
        elif isinstance(value,list):
            for i,item in enumerate(value): walk(item,[*path,i])
        elif not isinstance(value,str) or len(value)<=256:
            rows.append({'kind':'public_json','path':path,'value':value})
    walk(evidence.get('public_task',{}), [])
    op=evidence['operation']
    source=op.get('source') or op.get('arguments',{}).get('source','')
    rows += source_literals(source)
    # Native argument trees are public, actually executed inputs.
    def arguments(value,path):
        if isinstance(value,dict):
            for k,v in value.items():
                if k not in PRIVATE_FIELDS and k!='source': arguments(v,[*path,k])
        elif isinstance(value,list):
            for i,v in enumerate(value):arguments(v,[*path,i])
        else:rows.append({'kind':'operation_json','path':path,'value':value})
    arguments(op.get('arguments',op.get('inputs',{})),[])
    return rows


def reference_value(evidence, selector):
    if isinstance(selector,list): return at(evidence['reference'],selector)
    if not isinstance(selector,dict):raise ValueError('Output reference must be a typed selector or JSON path')
    if selector.get('kind')=='publication':
        files=sorted(evidence['reference'].get('local_artifact_identities',{}))
        if not files or evidence['reference'].get('accepted',True) is not True:
            raise ValueError('Selected operation has no verified publication')
        field=selector.get('field')
        if field=='files':return files
        if field=='file' and selector.get('name') in files:return selector['name']
        raise ValueError('Unknown publication projection')
    if selector.get('kind')=='json':return at(evidence['reference'],selector['path'])
    raise ValueError('Unknown output reference kind')


def bound_input(evidence, ref):
    if not isinstance(ref,dict):raise ValueError('Input source must be a typed selector')
    path=ref.get('path',[])
    if any(p in PRIVATE_FIELDS for p in path if isinstance(p,str)):
        raise ValueError('Private answer field is forbidden')
    kind=ref.get('kind')
    if kind=='source_literal':
        source=evidence['operation'].get('source') or evidence['operation'].get('arguments',{}).get('source','')
        row=next((r for r in source_literals(source) if r['path']==path),None)
        if row is None:raise ValueError('Literal is absent or in an unverified optional branch')
        return row['value']
    if kind=='operation_json':
        if 'source' in path:raise ValueError('Use a locatable source literal')
        return at(evidence['operation'].get('arguments',evidence['operation'].get('inputs',{})),path)
    value=at(evidence['public_task'],path)
    if kind=='public_json':return value
    if kind=='public_span':
        start,end=ref.get('start'),ref.get('end')
        if not isinstance(value,str) or type(start) is not int or type(end) is not int or not 0<=start<end<=len(value):
            raise ValueError('Invalid public string span')
        return value[start:end]
    raise ValueError('Unknown input source kind')


def canonical_binding(binding, experience, skill):
    """Aggregate preflight errors and freeze authority from the selected Host event."""
    evidence=next((e for e in experience.get('local_evidence',[]) if e['id']==binding.get('local_evidence_ref')),None)
    errors=[]
    def error(code,field,detail):errors.append({'code':code,'field':field,'detail':str(detail)})
    if not evidence:error('unknown_source','local_evidence_ref','Select an existing operation reference')
    for key in ('prefix','start_mode','source_trace_sha256','environment_identity','binding_hash'):
        if key in binding:error('host_authority_field',key,'Host supplies this field')
    try:validate_schema_instance(binding.get('inputs',{}),skill['input_schema'])
    except ValueError as exc:error('input_type','inputs',exc)
    refs=binding.get('input_refs',{})
    if evidence:
        for key,value in binding.get('inputs',{}).items():
            try:
                if key in PRIVATE_FIELDS or key not in refs or bound_input(evidence,refs[key])!=value:
                    raise ValueError('Value requires an exact public or executed-source reference')
            except (ValueError,TypeError,KeyError) as exc:error('unbound_input','inputs.'+key,exc)
    fields=binding.get('reference_fields',{})
    required=set(skill['output_schema'].get('required',[]))
    for key in sorted(required-set(fields)):error('missing_output_reference','reference_fields.'+key,'Required output needs a typed reference')
    values={}
    if evidence:
        for key,selector in fields.items():
            try:values[key]=reference_value(evidence,selector)
            except (ValueError,TypeError,KeyError) as exc:error('invalid_output_reference','reference_fields.'+key,exc)
        if skill.get('result_role')=='final_files' and not evidence['reference'].get('local_artifact_identities'):
            error('no_publication','local_evidence_ref','An inspection or declared filename is not a published file')
        try:validate_schema_instance(values,skill['output_schema'])
        except ValueError as exc:error('output_type','reference_fields',exc)
    if errors:
        failure=ValueError('trial_binding_invalid: '+str(errors))
        failure.feedback={'code':'trial_binding_invalid','errors':errors}
        raise failure
    result={**deepcopy(binding),'start_mode':'prefix_replay' if evidence['prefix'] else 'reset',
        'prefix':deepcopy(evidence['prefix']),'source_trace_sha256':evidence['source_trace_sha256'],
        'environment_identity':deepcopy(evidence['environment_identity']),'binding_policy':BINDING_POLICY_VERSION}
    result['binding_hash']=digest(result)
    return result


def evidence_from_trace(task, trace, environment):
    source = {'task': {'goal': task.goal, 'inputs': task.inputs}, 'events': trace.get('tools', []),
              'execution': trace.get('execution', {}), 'results': trace.get('result_store', {})}
    trace_hash = digest(source)
    evidence = []
    prefix = []
    for event in source['events']:
        if event.get('state') == 'finished' and event.get('backend_invoked') and event.get('result', {}).get('accepted'):
            evidence.append({'id': 'local:' + event['event_id'], 'source_physical_key': task.physical_key,
                'source_trace_sha256': trace_hash, 'event_ids': [event['event_id']], 'kind': 'native_operation',
                'operation': {'name': event['name'], 'arguments': deepcopy(event['arguments'])},
                'reference': deepcopy(event['result']), 'prefix': deepcopy(prefix),
                'public_task': source['task'], 'environment_identity': deepcopy(environment)})
        if event.get('backend_invoked'):
            prefix.append({'name': event['name'], 'arguments': deepcopy(event['arguments'])})
    results = source['results'].get('program_results', {})
    for attempt in [*source['execution'].get('attempts', []),*source['execution'].get('temporary_executions', [])]:
        result = results.get(attempt.get('result_id'), {})
        if attempt.get('status') == 'ok' and result.get('outputs') and attempt.get('source'):
            evidence.append({'id': 'local:' + attempt['id'], 'source_physical_key': task.physical_key,
                'source_trace_sha256': trace_hash, 'event_ids': [attempt['id']], 'kind': 'executed_python',
                'operation': {'source': attempt['source'], 'inputs': deepcopy(attempt['arguments']),
                              'input_schema': attempt['input_schema'], 'output_schema': attempt['output_schema']},
                'reference': deepcopy(result['outputs']), 'prefix': [], 'public_task': source['task'],
                'environment_identity': deepcopy(environment)})
    return evidence


def at(value, path):
    for part in path:
        if isinstance(value, dict) and isinstance(part, str) and part in value: value = value[part]
        elif isinstance(value, list) and type(part) is int and 0 <= part < len(value): value = value[part]
        else: raise ValueError('Local evidence path does not exist')
    return value


def contains(tree, value):
    return tree == value or (isinstance(tree, dict) and any(contains(v, value) for v in tree.values())) or (
        isinstance(tree, list) and any(contains(v, value) for v in tree))


def resolve_evidence(binding, experience, skill):
    ref = binding.get('local_evidence_ref')
    evidence = next((e for e in experience.get('local_evidence', []) if e['id'] == ref), None)
    if not evidence: raise ValueError('Program binding requires an existing host local_evidence_ref')
    if binding['prefix'] != evidence['prefix']:
        raise ValueError('Local validation start must match the real source operation prefix')
    if binding.get('binding_policy'):
        proposal={k:v for k,v in binding.items() if k not in {'start_mode','prefix','source_trace_sha256','environment_identity','binding_policy','binding_hash'}}
        if canonical_binding(proposal,experience,skill)!=binding:raise ValueError('Canonical binding identity changed')
    for key, value in binding['inputs'].items():
        if key in {'task_id', 'physical_key', 'gold', 'correct_choice', 'answer_label'}:
            raise ValueError('Program inputs cannot be answer lookup identities')
        if not binding.get('binding_policy') and not contains([evidence['public_task'], evidence['operation']], value):
            raise ValueError('Local input is not bound to public source or executed operation: ' + key)
    fields = binding.get('reference_fields', {})
    required = set(skill['output_schema'].get('required', []))
    if not fields or required - set(fields):
        raise ValueError('Local reference must bind the declared required outputs')
    for path in fields.values(): reference_value(evidence, path)
    return evidence


def qualification(program):
    validation = program.get('local_validation', {})
    return (program.get('state') == 'usable' and validation.get('passed') is True and
            validation.get('policy') == POLICY and validation.get('program_digest') == program_digest(program) and
            validation.get('environment_hash') == digest(program['environment']))


def validate_on_source(system, program, binding, task, experience, trial_id):
    """Never call a model or an evaluator for an intermediate local operation."""
    evidence = resolve_evidence(binding, experience, program)
    if evidence['environment_identity'] != program['environment']:
        raise ValueError('Local evidence environment changed')
    old = next((a for a in system.bank.attempts(program['id']) if a['id'] == trial_id), None)
    if old: return old
    from ..harness.simple_protocol import Broker
    from ..harness.simple_protocol import PROGRAM_FORBIDDEN_TOOLS
    from .task_context import TaskContext
    from .program_submission import normalize_program_result
    if system.adapter_factory is None: raise RuntimeError('Local validation requires an isolated Adapter factory')
    adapter = system.adapter_factory()
    try:
        inherit = getattr(adapter, 'inherit_discovery', None)
        if inherit: inherit(system.adapter)
        adapter.reset(task)
        broker = Broker(adapter, system.config['runtime']['global_action_budget'], context=TaskContext(system.config['runtime']))
        specs={t['name']:t for t in [*adapter.tool_definitions(),*broker.available_tools()]}
        allowed=set(specs)-PROGRAM_FORBIDDEN_TOOLS
        if set(program['allowed_tools'])-allowed:raise ValueError('Program permissions exceed the public Adapter surface')
        for action in evidence['prefix']:
            replay = broker.call(action['name'], action['arguments'])
            if not replay.get('accepted'): raise ValueError('Source prefix cannot be reconstructed')
        start = len(broker.events)
        if system.budget_governor: system.budget_governor.admit_validation(trial_id)
        fields = binding['reference_fields']
        expected = {key: reference_value(evidence, path) for key, path in fields.items()}
        file_refs = evidence['reference'].get('local_artifact_identities', {})
        def check_stage(stage, outputs):
            for name in outputs.get('files', []):
                path = (Path(stage)/name).resolve()
                identity=file_refs.get(name)
                if not isinstance(identity,dict) or identity.get('algorithm')!=FILE_EFFECT_VERSION:
                    raise ValueError('Source file identity requires an original-source replay under the current policy')
                if (not path.is_relative_to(Path(stage).resolve()) or not path.is_file()
                        or artifact_identity(path) != identity['effect_sha256']):
                    raise ValueError('Local file effect differs from the recorded source')
        result = normalize_program_result(adapter, program, system.worker.execute(
            program, binding['inputs'], broker,before_publish=check_stage))
        outputs = result.get('outputs', {})
        passed = result.get('status') == 'ok' and all(key in outputs and outputs[key] == value for key, value in expected.items())
        reason = 'source_output_match' if passed else 'source_output_mismatch'
        if passed and outputs.get('files'):
            passed = bool(file_refs)
            reason = 'source_file_effect_match' if passed else 'source_file_effect_mismatch'
        # An acceptance flag is not evidence that the intended operation ran.
        # Compare actual broker calls as well as values; allow extra read-only
        # checks, but not additional writes outside the recorded local contract.
        calls = [e for e in broker.events[start:] if e.get('backend_invoked')]
        operation = evidence['operation']
        native_match = None
        if passed and evidence['kind'] == 'native_operation':
            if operation['name'] not in PROGRAM_FORBIDDEN_TOOLS:
                native_match = any(e['name'] == operation['name'] and e['arguments'] == operation['arguments']
                    and e.get('result', {}).get('accepted') for e in calls)
                writes = [e for e in calls if specs.get(e['name'], {}).get('effect') != 'read_only']
                expected_writes = [] if specs.get(operation['name'], {}).get('effect') == 'read_only' else [operation]
                native_match = native_match and [{'name':e['name'],'arguments':e['arguments']} for e in writes] == expected_writes
                passed = native_match
                if not passed: reason = 'source_operation_mismatch'
            elif not outputs.get('files') and all(path and path[0] in {
                    'accepted','status','done','won','new_revision'} for path in fields.values() if isinstance(path,list)):
                passed = False
                reason = 'missing_local_effect_reference'
        # Pure JSON compilation additionally replays one schema-valid parameter change,
        # when the original executed operation supplies an independent executable reference.
        numeric = next((k for k,v in binding['inputs'].items() if type(v) in (int,float) and k in operation.get('inputs', {})), None)
        variation = None
        if passed and evidence['kind'] == 'executed_python' and numeric:
            changed = {**binding['inputs'], numeric: binding['inputs'][numeric]+1}
            try: validate_schema_instance(changed, program['input_schema'])
            except ValueError: pass
            else:
                reference = {**program, 'id': 'reference_'+digest(operation), 'source': operation['source'],
                             'input_schema': operation['input_schema'], 'output_schema': operation['output_schema']}
                if system.budget_governor: system.budget_governor.admit_validation(trial_id+':reference')
                original = system.worker.execute(reference, changed, broker)
                if system.budget_governor: system.budget_governor.admit_validation(trial_id+':variation')
                replay = system.worker.execute(program, changed, broker)
                variation = {'input_hash':digest(changed), 'reference_hash':digest(original), 'actual_hash':digest(replay),
                    'reference_cpu_seconds':original.get('cpu_seconds'), 'candidate_cpu_seconds':replay.get('cpu_seconds')}
                passed = original.get('status') == replay.get('status') == 'ok' and all(
                    replay['outputs'].get(k) == at(original['outputs'], path) for k,path in fields.items())
                if not passed: reason = 'parameterized_replay_mismatch'
        tree = ast.parse(program['source'])
        if passed and native_match is not True and not outputs.get('files'):
            returns = [n.value for f in tree.body if isinstance(f,ast.FunctionDef) and f.name == program['entry']
                       for n in ast.walk(f) if isinstance(n,ast.Return)]
            try: literals = [ast.literal_eval(value) for value in returns]
            except (ValueError,TypeError): pass
            else:
                if literals and all(value == literals[0] for value in literals):
                    passed = False
                    reason = 'constant_local_output'
        parameterized = any(isinstance(n,ast.Name) and n.id in {'inputs','ctx'} and isinstance(n.ctx,ast.Load) for n in ast.walk(tree))
        passed = passed and parameterized and bool(binding['inputs'] or program['allowed_tools'])
        validation = {'policy':POLICY, 'passed':passed, 'program_digest':program_digest(program),
            'environment_hash':digest(program['environment']), 'source_trace_sha256':evidence['source_trace_sha256'],
            'source_physical_key':task.physical_key, 'evidence_id':evidence['id'],
            'operation_contract_hash':digest([program['input_schema'],program['output_schema'],program.get('entry_constraints')]),
            'reference_hash':digest(expected), 'input_hash':digest(binding['inputs']), 'output_hash':digest(outputs),
            'reason':reason if parameterized else 'missing_parameterized_input', 'variation':variation,
            'native_operation_match':native_match,
            'binding_hash':binding.get('binding_hash'), 'file_effect_version':FILE_EFFECT_VERSION,
            'actual_operation_hash':digest([{'name':e['name'],'arguments':e['arguments']} for e in calls])}
        record = {'id':trial_id, 'program_id':program['id'], 'task_key':task.physical_key, 'origin':'train_test',
            'split':'train', 'outcome':'positive' if passed else 'execution_failure', 'basis':'local_check' if passed else None,
            'calls':len(broker.events)-start, 'result':result, 'tools':broker.events, 'validation':validation}
        system.bank.record(record)
        system.bank.record_validation(program['id'], validation)
        return record
    finally: adapter.close()
