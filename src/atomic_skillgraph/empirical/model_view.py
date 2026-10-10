"""Readable, public stage projections. Execution and scoring use original values."""
from copy import deepcopy
import json
from . import POLICY_DEFAULTS
from .contracts import PublicTask

MODEL_VIEW_VERSION = POLICY_DEFAULTS['runtime']['model_view_version']
CANDIDATE_VIEW_VERSION = 'candidate.v2'
CONTRACT_FIELDS = ('input_schema', 'output_schema', 'result_role', 'entry_constraints')


def pack_material(material):
    """Intern only identical contracts/jobs; every referenced value stays in this request."""
    groups = {'contracts': {}, 'jobs': {}}
    def candidates(v):
        if isinstance(v, dict):
            if 'input_schema' in v and 'output_schema' in v:
                yield 'contracts', {k:v[k] for k in CONTRACT_FIELDS if k in v}
            if all(k in v for k in ('id', 'skill_id', 'case_bindings', 'state', 'generation_count')):
                yield 'jobs', v
            for x in v.values(): yield from candidates(x)
        elif isinstance(v, list):
            for x in v: yield from candidates(x)
    def key(v): return json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    for kind, v in candidates(material):
        serial = key(v)
        groups[kind][serial] = groups[kind].get(serial, 0) + 1
    refs, tables = {}, {'contracts': {}, 'jobs': {}}
    for kind, counts in groups.items():
        for serial, count in counts.items():
            if count > 1:
                ref = kind[0] + str(len(tables[kind]))
                refs[(kind, serial)] = ref
                tables[kind][ref] = json.loads(serial)
    if not any(tables.values()): return deepcopy(material)
    def pack(v):
        if isinstance(v, list): return [pack(x) for x in v]
        if not isinstance(v, dict): return deepcopy(v)
        if all(k in v for k in ('id', 'skill_id', 'case_bindings', 'state', 'generation_count')):
            ref = refs.get(('jobs', key(v)))
            if ref: return {'job_ref': ref}
        result = dict(v)
        if 'input_schema' in v and 'output_schema' in v:
            contract = {k:v[k] for k in CONTRACT_FIELDS if k in v}
            ref = refs.get(('contracts', key(contract)))
            if ref:
                result = {k:x for k,x in v.items() if k not in contract}
                result['contract_ref'] = ref
        return {k:pack(x) for k,x in result.items()}
    return {**pack(material), 'material_tables': tables,
        'material_reference_note': 'contract_ref merges fields from material_tables.contracts; job_ref replaces the object with material_tables.jobs. All evidence is here; these are not read_result IDs.'}


def expand_material(material):
    tables = material.get('material_tables', {})
    def expand(v):
        if isinstance(v, list): return [expand(x) for x in v]
        if not isinstance(v, dict): return deepcopy(v)
        if set(v) == {'job_ref'}: return deepcopy(tables['jobs'][v['job_ref']])
        result = {k:expand(x) for k,x in v.items() if k != 'contract_ref'}
        if 'contract_ref' in v: result.update(deepcopy(tables['contracts'][v['contract_ref']]))
        return result
    return expand({k:v for k,v in material.items() if k not in {'material_tables','material_reference_note'}})


def canonical_bytes(value):
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8'))


def learning_preview(context, value):
    preview = context.preview(value)
    # Learner/Builder cannot dereference episode-local pointers; keep the readable window.
    if isinstance(preview, dict) and preview.get('truncated') and 'preview' in preview:
        preview = {k:v for k,v in preview.items() if k != 'ref'}
    return preview


def operation_directory(evidence, case_id):
    from .local_validation import input_sources
    reference=evidence.get('reference',{})
    files=sorted(reference.get('local_artifact_identities',{})) if reference.get('accepted',True) else []
    outputs=[{'kind':'json','path':[k],'type':type(v).__name__} for k,v in reference.items()
             if k not in {'local_artifact_identities','local_artifact_hashes','observation','error'}]
    outputs += [{'kind':'json','path':[k,child],'type':type(value).__name__}
                for k,v in reference.items() if k=='data' and isinstance(v,dict) for child,value in v.items()]
    if files:
        outputs += [{'kind':'publication','field':'files','value':files},
                    *[{'kind':'publication','field':'file','name':n,'value':n} for n in files]]
    sources=[r for r in input_sources(evidence) if canonical_bytes(r)<=512]
    return {'case_id':case_id,'local_evidence_ref':evidence['id'],'kind':evidence['kind'],
        'operation_name':evidence['operation'].get('name','executed_python'),
        'effect':'published_files' if files else 'inspection_or_json',
        'success':reference.get('accepted',True),'published_files':files,
        'input_sources':sources[:32], 'input_source_count':len(sources), 'output_references':outputs,
        'source_content_available_to_builder':True}


def learning_case(experience, case_id):
    """No replay prefix, raw result or repeated public task enters the directory."""
    task=experience.get('task',{})
    evidence=experience.get('local_evidence',[])
    selected=sorted(evidence,key=lambda e:(not bool(e['reference'].get('local_artifact_identities')),
        canonical_bytes(e['operation']),e['id']))[:1]
    return {'case_source':{'physical_case_id':case_id},'task':{'goal':task.get('goal','')},
        **{k:deepcopy(experience.get(k)) for k in ('score','termination_reason','answer_status','completion_truncated')},
        'local_evidence':[operation_directory(e,case_id) for e in selected],
        'directory_selection':'verified publication first, then complete operation bytes and ID',
        'projection_audit':{'original_event_count':len(experience.get('events',[])),
            'original_evidence_count':len(experience.get('local_evidence',[])),
            'original_utf8_bytes':canonical_bytes(experience)}, 'compact_learning_directory':True}


def builder_example(binding, evidence):
    return {'case_id':binding['case_id'],'binding_hash':binding['binding_hash'],
        'inputs':deepcopy(binding['inputs']), 'reference_fields':deepcopy(binding['reference_fields']),
        'operation':deepcopy(evidence['operation']), 'kind':evidence['kind'],
        'public_task':{'goal':evidence['public_task'].get('goal','')},
        'local_evidence_ref':evidence['id']}


def structure_repair_material(proposal, errors, directories):
    """Keep the proposed scope and contracts; remove explanatory duplication."""
    value=deepcopy(proposal)
    value.pop('rationale',None)
    if value.get('skill'):value['skill'].pop('guidance',None)
    def contract(v):
        if isinstance(v,dict):return {k:contract(x) for k,x in v.items() if k not in {'description','title','examples'}}
        if isinstance(v,list):return [contract(x) for x in v]
        return v
    if value.get('skill'):
        for k in ('input_schema','output_schema'):value['skill'][k]=contract(value['skill'][k])
    inputs=[v for b in value.get('realization_request',{}).get('case_bindings',[]) for v in b.get('inputs',{}).values()]
    operations=[]
    for rows in directories:
        for row in rows:
            operations.append({k:deepcopy(row[k]) for k in ('case_id','local_evidence_ref','kind','effect','published_files','output_references')})
            operations[-1]['input_sources']=[r for r in row['input_sources'] if r['value'] in inputs]
    short=[{'binding_index':row['binding_index'],'errors':[{'code':e['code'],'field':e.get('field'),
        **({'expected':e['detail'][:200]} if e['code'] in {'input_type','output_type'} else {})} for e in row['errors']]} for row in errors]
    return {'proposal':value,'errors':short,'legal_operations':operations,
            'note':'Host start/identity fields must be removed. No matching source value means omit or reject the unsupported input; do not invent provenance.'}


def project_related_candidates(related):
    """Project already selected candidates; never mutate assets or judge eligibility."""
    def fields(row, names):
        return {key: deepcopy(row[key]) for key in names.split() if key in row}
    def job(row):
        result=fields(row, 'id skill_id skill_version program_id kind state generation_count repair_used epoch last_error_kind contract_quarantine')
        result['case_bindings']=[fields(b,'case_id local_evidence_ref binding_hash') for b in row.get('case_bindings',[])]
        return result
    def failure(row):
        if isinstance(row, str): return {'message': row[:512]}
        if not isinstance(row, dict): return None
        result = fields(row, 'code error_code repair_target infrastructure_error stage operation cause_type trial_execution_id')
        for key in ('message', 'error', 'detail', 'reason'):
            if isinstance(row.get(key), str): result[key] = row[key][:512]
        return result
    value = deepcopy(related)
    for asset in value:
        if asset.get('current_program'):
            asset['current_program'] = fields(asset['current_program'],
                'id state entry backend input_schema output_schema result_role entry_constraints allowed_tools environment')
        if asset.get('current_job'):
            asset['current_job'] = job(asset['current_job'])
            bindings = asset['current_job'].get('case_bindings', [])
            used = set(asset.get('used_physical_tasks', []))
            asset['trial_slots'] = {'limit': 2, 'fixed_case_count': len(bindings),
                'unfilled_slot_count': 2 - len(bindings),
                'untried_case_ids': [b['case_id'] for b in bindings if b['case_id'] not in used]}
        if 'pending' in asset: asset['pending'] = [job(row) for row in asset['pending']]
        if 'independent_results' in asset:
            records = []
            for row in asset['independent_results']:
                summary = fields(row, 'id logical_trial_id trial_execution_id program_id task_key origin split outcome basis status local_check output_contract_status')
                result = row.get('result', {})
                summary.update(fields(result, 'status local_check output_contract_status contract_status error_code'))
                if isinstance(result.get('submission_preparation'), dict):
                    summary['submission_preparation'] = fields(result['submission_preparation'], 'status error_code repair_target')
                if isinstance(result.get('score'), dict):
                    summary['score'] = fields(result['score'], 'hard raw_score')
                for key, source in (('error', row.get('error')), ('result_error', result.get('error')),
                                    ('exception', row.get('exception')), ('diagnostic', result.get('diagnostic'))):
                    error = failure(source)
                    if error: summary[key] = error
                records.append(summary)
            asset['independent_results'] = records
    return value


def contains_exact(value, wanted):
    if value == wanted: return True
    if isinstance(value, dict): return any(contains_exact(v, wanted) for v in value.values())
    if isinstance(value, list): return any(contains_exact(v, wanted) for v in value)
    return False


def model_task(task, adapter):
    if hasattr(adapter, 'model_task'):
        return deepcopy(adapter.model_task(task))
    return {'goal': task.goal, 'inputs': deepcopy(task.inputs)}


def project(stage, material, *, task=None, adapter=None, context=None):
    value = deepcopy(material)
    if 'original_task' in value and task is not None:
        value['original_task'] = model_task(task, adapter)
        contract = getattr(adapter, 'answer_contract', lambda: '')()
        if contract: value['original_task']['answer_contract'] = contract
        if context and adapter.capabilities.interaction != 'single_answer':
            value['original_task']['inputs'] = {k: context.preview(v) for k, v in value['original_task']['inputs'].items()}
    if stage == 'runtime' and 'node_interface' in value:
        public_task = value.pop('original_task')
        interface = value.pop('node_interface')
        sources = interface.pop('args', {})
        node_goal = value.pop('node_goal')
        if interface.get('node_goal') == node_goal: interface.pop('node_goal')
        if node_goal == public_task['goal']: node_goal = {'task_field': 'goal'}
        inputs = value.pop('inputs')
        if context and task and inputs:
            source = context.reference({'goal': task.goal, 'inputs': task.inputs}, 'task_inputs')
            inputs = {k: {**source, 'path': ['inputs', k]} if k in task.inputs and v == task.inputs[k]
                      else context.preview(v) for k, v in inputs.items()}
        if context:
            sources = {k: {'literal': context.reference(v['literal'], 'literal_input')} if 'literal' in v else v
                       for k, v in sources.items()}
        state = value.pop('public_state')
        if task and state.get('inputs') == task.inputs: state.pop('inputs')
        if context and state.get('observation') and contains_exact(public_task, state['observation']):
            state['observation'] = context.reference(state['observation'], 'observation_already_shown')
        memory = value.pop('working_memory')
        recent = value.pop('recent')
        completed = value.pop('completed_results')
        if context:
            for row in recent:
                rid = row.get('result_id')
                if not rid: continue
                original = context.results[rid]
                if original.get('observation') and original.get('observation') == state.get('observation'):
                    row['observation'] = {'result_id': rid, 'path': ['observation']}
                for fields in completed.values():
                    for key, field in fields.items():
                        if key in original.get('outputs', {}) and field == original['outputs'][key]:
                            fields[key] = {'result_id': rid, 'path': ['outputs', key]}
        # Memory stores query/scope/acceptance and the result ID; previews live in recent_results.
        return pack_material({'model_view_version': MODEL_VIEW_VERSION, 'task': public_task,
            'node': {'id': value.pop('node_id'), 'goal': node_goal, 'interface': interface},
            'bindings': {'inputs': inputs, 'input_sources': sources, 'missing': value.pop('missing'),
                         'completed_results': completed},
            'handoff': {k: value.pop(k) for k in ('required_handoff_fields', 'handoff_consumers', 'return_example', 'pending_outputs', 'output_aliases')},
            'state': state,
            'calls': {k: value.pop(k) for k in ('tools', 'programs', 'allowed_calls')},
            'memory': {'guidance': value.pop('guidance'), 'operations': memory, 'recent_results': recent},
            'recovery': value})
    if stage == 'extractor' and 'related' in value:
        before = canonical_bytes(value['related'])
        value['related'] = project_related_candidates(value['related'])
        value['candidate_view_version'] = CANDIDATE_VIEW_VERSION
        value['candidate_view_audit'] = {'version': CANDIDATE_VIEW_VERSION, 'before_bytes': before,
            'after_bytes': canonical_bytes(value['related']), 'candidate_ids': [a['id'] for a in value['related']]}
    if stage == 'extractor' and task is not None and not value['experience'].get('compact_learning_directory'):
        value['experience']['task'] = model_task(task, adapter)
        if context and adapter.capabilities.interaction != 'single_answer':
            for case in value.get('completed_train_cases', []):
                original = case['task']
                case['task'] = learning_preview(context, model_task(PublicTask('', '', original['goal'], original.get('inputs', {})), adapter))
                if 'action_prefix' in case:
                    case['action_prefix'] = learning_preview(context, case['action_prefix'])
    return pack_material(value)


def callable_tools(material):
    material = expand_material(material)
    return material.get('calls', {}).get('tools', material.get('tools', []))
