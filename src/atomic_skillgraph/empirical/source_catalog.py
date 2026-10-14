"""Host-owned local IDs; models select sources, never author binding paths."""
from copy import deepcopy

from .contracts import digest, object_schema, validate_schema_instance
from .local_validation import bound_input, canonical_binding
from .model_view import expand_material, operation_directory
from . import OFFICE_SOURCE_METHOD_VERSION, OFFICE_SOURCE_MATERIAL_VERSION, OFFICE_TRACE_METHOD_VERSION, OFFICE_TRACE_MATERIAL_VERSION

ID_MAP = {'type': 'object', 'additionalProperties': {'type': 'string'}}
BINDING = object_schema({'operation_id': {'type': 'string'}, 'input_ids': ID_MAP, 'output_ids': ID_MAP},
                        ['operation_id', 'input_ids', 'output_ids'])


def enabled(config):
    return config.get('experiment', {}).get('officeqa_method_version') in {OFFICE_SOURCE_METHOD_VERSION,OFFICE_TRACE_METHOD_VERSION}


def trace_enabled(config):
    return config.get('experiment', {}).get('officeqa_method_version') == OFFICE_TRACE_METHOD_VERSION


def verified_sources(cases):
    """Only Host-recorded successful calls grant the native Python capacity extension."""
    result = {}
    for task, case in cases:
        for evidence in case.get('local_evidence', []):
            from .retrieval_chain import KIND, authentic
            if evidence.get('kind')==KIND and authentic(evidence,case):
                result[(task.physical_key,evidence['id'])]=operation_directory(evidence,task.physical_key)
                continue
            if evidence.get('kind') != 'native_operation': continue
            event = next((e for e in case.get('events', []) if e.get('event_id') in evidence.get('event_ids', [])), None)
            if (event and event.get('state') == 'finished' and event.get('backend_invoked') is True
                    and not event.get('parent_event_id') and event.get('name') == 'execute_python'
                    and event.get('result', {}).get('accepted') is True
                    and evidence.get('operation') == {'name': event['name'], 'arguments': event['arguments']}
                    and evidence.get('reference') == event['result']
                    and evidence.get('source_physical_key') == task.physical_key
                    and evidence.get('source_trace_sha256')):
                result[(task.physical_key,evidence['id'])]=operation_directory(evidence,task.physical_key)
    return result


def verified_card(card, trusted):
    original=trusted.get((card.get('case_id'),card.get('local_evidence_ref'))) if isinstance(trusted,dict) else None
    if original is None:return False
    compact=catalog_material({'experience':{'local_evidence':[original]}},trace=original['kind']=='verified_retrieval_chain.v1')['experience']['local_evidence'][0]
    fields=('kind','operation_name','success','source_excerpt','input_sources','output_references')
    return any(all(card.get(k)==expected.get(k) for k in fields) for expected in (original,compact))


def catalog_material(material, *, trace=False):
    """Replace verbose selectors with stable short IDs; exact selectors stay at the Host."""
    value = expand_material(material)
    for case in [value.get('experience', {}), *value.get('completed_train_cases', [])]:
        for card in case.get('local_evidence', []):
            card['operation_id'] = 'op_' + digest([card['case_id'], card['local_evidence_ref']])[:12]
            card['input_sources'] = [
                {'id': 'i' + str(i), 'kind': row['kind'], **({'value_preview':row['value'][:8], 'item_count':len(row['value']),
                    'full_value_available_to_builder':True} if trace and isinstance(row['value'],list) and len(row['value'])>8 else {'value':row['value']}),
                 **({'field':row.get('path'), 'type':type(row['value']).__name__} if trace else {})}
                for i, row in enumerate(card['input_sources'])]
            card['output_references'] = [
                {'id': 'o' + str(i), **{k: v for k, v in row.items() if k not in {'kind', 'value'}}}
                for i, row in enumerate(card['output_references'])]
        case['source_catalog_version'] = OFFICE_TRACE_MATERIAL_VERSION if trace else OFFICE_SOURCE_MATERIAL_VERSION
    return value


def proposal_schema(schema, *, trace=False):
    schema = deepcopy(schema)
    schema['properties']['realization_request']['properties']['case_bindings']['items'] = deepcopy(BINDING)
    if trace:
        skill=schema['properties']['skill']
        for key in ('input_schema','output_schema'):
            skill['properties'].pop(key);skill['required'].remove(key)
    return schema


def bind_proposal(proposal, catalog, cases, bank):
    """Return the existing canonical proposal format, checking every ID against shown sources."""
    proposal = deepcopy(proposal)
    request = proposal.get('realization_request')
    if not request: return proposal
    cards = [card for case in [catalog.get('experience', {}), *catalog.get('completed_train_cases', [])]
             for card in case.get('local_evidence', [])]
    selected = {card['operation_id']: card for card in cards}
    trace=catalog.get('experience',{}).get('source_catalog_version')==OFFICE_TRACE_MATERIAL_VERSION
    skill = proposal.get('skill') if request['skill_id'] == '$new' else bank.get(request['skill_id'])
    if not skill: raise ValueError('Select $new or an existing Skill ID')
    bindings = []
    for binding in request['case_bindings']:
        validate_schema_instance(binding, BINDING)
        card = selected.get(binding['operation_id'])
        if not card: raise ValueError('Unknown operation_id; select a shown Host operation')
        case_id = card['case_id']
        case = cases[case_id][1]
        evidence = next(e for e in case['local_evidence'] if e['id'] == card['local_evidence_ref'])
        original = operation_directory(evidence, case_id)
        inputs, refs, outputs = {}, {}, {}
        for key, source_id in binding['input_ids'].items():
            choices = {row['id']: i for i, row in enumerate(card['input_sources'])}
            if source_id not in choices: raise ValueError('Unknown parameter source ID: ' + source_id)
            source = original['input_sources'][choices[source_id]]
            refs[key] = {k: v for k, v in source.items() if k != 'value'}
            inputs[key] = bound_input(evidence, refs[key])
        for key, source_id in binding['output_ids'].items():
            choices = {row['id']: i for i, row in enumerate(card['output_references'])}
            if source_id not in choices: raise ValueError('Unknown output field ID: ' + source_id)
            outputs[key] = {k: v for k, v in original['output_references'][choices[source_id]].items()
                            if k not in {'type', 'content_hint', 'value'}}
        raw = {'case_id': case_id, 'local_evidence_ref': evidence['id'], 'inputs': inputs,
               'input_refs': refs, 'reference_fields': outputs}
        if trace:
            from .retrieval_chain import KIND, authentic, schema
            from .local_validation import reference_value
            if evidence['kind']==KIND:
                if not authentic(evidence,case):raise ValueError('Retrieval chain source mismatch')
                controls={r['path'][0] for r in refs.values() if r['kind']=='operation_json' and len(r['path'])==1}
                if controls!=set(evidence['operation']['arguments']) or len(refs)!=len(controls):
                    raise ValueError('Select every Host chain control exactly once')
                if skill.get('result_role','intermediate')!='intermediate':raise ValueError('Retrieval text supports intermediate only')
                if not any(r.get('path')==['data','text'] for r in outputs.values()):
                    raise ValueError('Retrieval chain must expose the actual acquired text')
            derived={'input_schema':schema(inputs), 'output_schema':schema({k:reference_value(evidence,r) for k,r in outputs.items()})}
            if evidence['kind']==KIND:
                for key,ref in outputs.items():
                    if ref.get('path')==['data','next_offset']:
                        derived['output_schema']['properties'][key]={'type':['integer','null']}
            if request['skill_id']=='$new' and not bindings:skill.update(derived)
            elif any(skill.get(k)!=v for k,v in derived.items()):raise ValueError('Source-derived Schema differs across bindings or existing Skill')
        canonical_binding(raw, case, skill)  # No bypass of schema, provenance, or final-answer checks.
        bindings.append(raw)
    request['case_bindings'] = bindings
    return proposal


def repair_material(proposal, error, catalog):
    selected={(b.get('operation_id')) for b in (proposal.get('realization_request') or {}).get('case_bindings',[])}
    cards=[card for case in [catalog.get('experience',{}),*catalog.get('completed_train_cases',[])]
           for card in case.get('local_evidence',[])]
    matching=[card for card in cards if card['operation_id'] in selected]
    operations=[{k:deepcopy(v) for k,v in card.items() if k not in
        {'source_excerpt','source_truncated','executed_calls','source_content_available_to_builder'}}
        for card in (matching or cards[:1])]
    return {'proposal':deepcopy(proposal),'errors':[error[:1600]], 'legal_operations':operations,
            'source_catalog_version':catalog.get('experience',{}).get('source_catalog_version',OFFICE_SOURCE_MATERIAL_VERSION),
            'note':'Preserve operation_id/input_ids/output_ids. Host owns values, paths and replay identity.'}


PROMPT = """Learn one reusable local procedure from completed Train operations. Submit one submit_learning ToolCall.
Tools describe future execution; do not call them now. Choose no_change when unsupported or already covered,
or reuse_existing with a real Skill ID. New Skill references use $new. Prefer intermediate retrieval, extraction,
sorting or numerical computation. A failed whole answer can still contain a successful reusable local operation.
For realization_request.case_bindings select a shown operation_id, input_ids mapping Skill input names to that
operation's i IDs, and output_ids mapping Skill output names to its o IDs. Host supplies exact values, selectors,
prefix and identity. Never invent paths, values or IDs. Keep every required output grounded in the same operation.
source_excerpt is executed source, possibly truncated; Builder receives full source. Inputs may be entire lists
or objects. Keep the selected local goal; do not add whole-task goals. intermediate numerical outputs are not
final_answer. final_answer requires an actual executed nonempty answer:string field, never relabeled text/numbers.
No private gold, hardcoded answers or fabricated results. No additional trial cases or validation requests.
guidance_only uses generate_program=false and no realization_request. Program requests need a real source binding.
Workflows are optional; omit unnecessary composition. Repairs preserve this ID contract and the local goal."""

TRACE_PROMPT = PROMPT + """
This version derives input_schema/output_schema at the Host: omit both from Skill proposals.
For verified_retrieval_chain.v1 select all displayed chain input controls and the acquired text output.
The chain proves recorded path/offset handoffs and text acquisition, not numeric parsing or whole-task success.
Keep result_role=intermediate. Only original glob/grep/read are proven in these historical chains.
Do not claim grep_page/read_context executed historically. A truncated grep is an incomplete search;
an existing hit can be read, but absent evidence must return needs_input/not_found/blocked.
Composite nodes must have verified Implementations and actual observed producer/consumer bindings.
"""
