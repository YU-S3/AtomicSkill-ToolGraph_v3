"""Readable, public stage projections. Execution and scoring use original values."""
from copy import deepcopy
import ast
import json
from . import POLICY_DEFAULTS
from .contracts import PublicTask
from .public_results import public_result, evidence_origin, TEXT_PAGE

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
    texts = material.get('p2_text_table', {})
    p2='p2_text_table' in material and 'p2_table_note' in material
    trace_values=material.get('trace_value_table',{})
    trace='trace_value_table' in material and 'trace_table_note' in material
    def expand(v):
        if isinstance(v, list): return [expand(x) for x in v]
        if not isinstance(v, dict): return deepcopy(v)
        if trace and set(v)=={'trace_value_ref'}:return expand(deepcopy(trace_values[v['trace_value_ref']]))
        if p2 and set(v)=={'p2_text_ref'}:return texts[v['p2_text_ref']]
        if set(v) == {'job_ref'}: return expand(deepcopy(tables['jobs'][v['job_ref']]))
        result = {k:expand(x) for k,x in v.items() if k != 'contract_ref'}
        if 'contract_ref' in v: result.update(expand(deepcopy(tables['contracts'][v['contract_ref']])))
        if p2 and 'p2_span_table' in result:
            table=result.pop('p2_span_table')
            result['spans']=[dict(span_id=row[0],task_path=table['paths'][row[1]],start=row[2],end=row[3]) for row in table['rows']]
        return result
    return expand({k:v for k,v in material.items() if k not in {'material_tables','material_reference_note','p2_text_table','p2_table_note','trace_value_table','trace_table_note'}})


def pack_trace_material(material):
    """V3 only: retain full repeated source strings/scopes once in the same request."""
    from collections import Counter
    value=expand_material(material);counts=Counter()
    def key(v):return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'))
    def count(v):
        if isinstance(v,(str,list)) and len(key(v))>=128:counts[key(v)]+=1
        if isinstance(v,dict):
            for child in v.values():count(child)
        elif isinstance(v,list):
            for child in v:count(child)
    count(value);refs={};table={}
    def pack(v):
        serial=key(v) if isinstance(v,(str,list)) else None
        if serial and counts[serial]>1:
            ref=refs.setdefault(serial,'v'+str(len(refs)));table[ref]=deepcopy(v)
            return {'trace_value_ref':ref}
        if isinstance(v,list):return [pack(x) for x in v]
        if isinstance(v,dict):return {k:pack(x) for k,x in v.items()}
        return v
    packed=pack(value)
    return pack_material({**packed,'trace_value_table':table,
        'trace_table_note':'trace_value_ref replaces the value with the exact trace_value_table entry. Full source, scopes and bindings remain in this request.'}) if table else pack_material(value)


def pack_capability_material(material):
    """Losslessly tabulate repeated span paths and exact strings for explicit P2 v3."""
    from collections import Counter
    value=reference_exact_span_quotes(expand_material(material));counts=Counter()
    def tabulate(v):
        if isinstance(v,list):return [tabulate(x) for x in v]
        if not isinstance(v,dict):return v
        result={k:tabulate(x) for k,x in v.items()}
        spans=result.get('spans')
        if spans and all(set(s)=={'span_id','task_path','start','end'} for s in spans):
            paths=[];rows=[]
            for s in spans:
                if s['task_path'] not in paths:paths.append(s['task_path'])
                rows.append([s['span_id'],paths.index(s['task_path']),s['start'],s['end']])
            result.pop('spans');result['p2_span_table']={'paths':paths,'rows':rows}
        return result
    value=tabulate(value)
    def strings(v):
        if isinstance(v,str):counts[v]+=1
        elif isinstance(v,dict):
            for child in v.values():strings(child)
        elif isinstance(v,list):
            for child in v:strings(child)
    strings(value)
    refs={text:str(i) for i,text in enumerate(text for text,n in counts.items() if n>1 and len(text)>=64)}
    def intern(v):
        if isinstance(v,str) and v in refs:return {'p2_text_ref':refs[v]}
        if isinstance(v,dict):return {k:intern(x) for k,x in v.items()}
        if isinstance(v,list):return [intern(x) for x in v]
        return v
    return {**intern(value),'p2_text_table':{ref:text for text,ref in refs.items()},
        'p2_table_note':'p2_text_ref is the exact string in p2_text_table. Each p2_span_table row is [span_id, path_index, start, end]; task_path is paths[path_index]. Resolve these lossless references before copying IDs or reading spans. All task text and evidence remain here.'}


def restore_capability_response(value, wire_material):
    """Resolve only identity slots against this exact request's bijective text table."""
    table=wire_material.get('p2_text_table',{})
    if not isinstance(table,dict) or any(not isinstance(v,str) for v in table.values()) or len(set(table.values()))!=len(table):
        raise ValueError('Ambiguous capability text table')
    def restore(v,slot=None):
        if isinstance(v,list):return [restore(x,slot) for x in v]
        if not isinstance(v,dict):return deepcopy(v)
        if 'p2_text_ref' in v:
            if set(v)!={'p2_text_ref'} or slot not in {'case_id','span_id','span_ids','capability_id','candidate_hash','task_hash','parent_ids','proposal_quote'}:
                raise ValueError('Compressed reference outside an identity slot')
            key=v['p2_text_ref']
            if not isinstance(key,str) or key not in table:raise ValueError('Unknown capability text reference')
            return table[key]
        return {k:restore(x,k) for k,x in v.items()}
    # Abstention is terminal; accompanying unused candidate fields never create an asset.
    if isinstance(value,dict) and value.get('decision')=='no_change':return {'decision':'no_change'}
    return restore(value)


def reference_exact_span_quotes(material):
    """Lossless references to text already present in the same public task."""
    value=deepcopy(material);changed=False
    def visit(row):
        nonlocal changed
        if isinstance(row,list):
            for child in row:visit(child)
        elif isinstance(row,dict):
            if isinstance(row.get('task'),dict) and isinstance(row.get('spans'),list):
                for span in row['spans']:
                    try:
                        text=row['task']
                        for field in span['task_path']:text=text[field]
                        if not isinstance(text,str) or text[span['start']:span['end']]!=span.get('quote'):continue
                    except (KeyError,IndexError,TypeError):continue
                    span.pop('quote');changed=True
            for child in list(row.values()):visit(child)
    visit(value)
    if changed:value['span_quote_reference_note']='A span quote is exactly task[task_path][start:end], with Unicode character offsets. Full source text remains in that task.'
    return value


def canonical_bytes(value):
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8'))


def shrink_learning_preview(material):
    """Reduce optional previews, preserving every source selector and output type."""
    cards=material.get('experience',{}).get('local_evidence',[])
    rows=[*cards,*material.get('legal_operations',[])]
    if len(cards)>1:return False
    for row in rows:
        for output in row.get('output_references',[]):
            if output.get('content_hint'):
                output.pop('content_hint');return True
    from . import OFFICE_SOURCE_MATERIAL_VERSION, OFFICE_TRACE_MATERIAL_VERSION
    if material.get('learning_material_version') in {OFFICE_SOURCE_MATERIAL_VERSION,OFFICE_TRACE_MATERIAL_VERSION}:
        return False  # Executed source is required evidence under this versioned contract.
    for row in rows:
        if len(row.get('source_excerpt',''))>512:
            row['source_excerpt']=row['source_excerpt'][:512];row['source_truncated']=True
            return True
    for row in rows:
        if row.get('source_excerpt'):
            row.pop('source_excerpt');row['source_truncated']=True
            return True
    return False


def learning_capacity_audit(payload, limit, stage, purpose):
    """Only a recognized, exhausted learning projection can be a controlled refusal."""
    from .budget_governor import input_token_bound
    unverified={'classification':'unverified_capacity_refusal','projection_version':'learning-capacity.v1'}
    try:
        if stage not in {'extractor','tool_builder'}:return unverified
        material=json.loads(payload['messages'][1]['content'])
        if not isinstance(material,dict):return unverified
        value=expand_material(material)
        experience=value.get('experience',{})
        if not isinstance(experience,dict):return unverified
        cards=experience.get('local_evidence',[])
        operations=value.get('legal_operations',[])
        if not isinstance(cards,list) or not isinstance(operations,list):return unverified
        refs=[r['local_evidence_ref'] for r in operations]
        if not all(isinstance(r,str) for r in refs):return unverified
    except (ValueError,TypeError,KeyError,IndexError):return unverified
    optional=any(value.get(k) for k in ('completed_train_cases','related','related_guidance'))
    optional=optional or len(cards)>1
    from . import OFFICE_SOURCE_MATERIAL_VERSION, OFFICE_TRACE_MATERIAL_VERSION
    protected=value.get('learning_material_version') in {OFFICE_SOURCE_MATERIAL_VERSION,OFFICE_TRACE_MATERIAL_VERSION}
    optional=optional or any((row.get('source_excerpt') and not protected) or any(r.get('content_hint') for r in row.get('output_references',[]))
                            for row in [*cards,*operations])
    if purpose=='learning_structure_repair':
        expected={'proposal','errors','legal_operations','note'}
        if value.get('source_catalog_version') in {OFFICE_SOURCE_MATERIAL_VERSION,OFFICE_TRACE_MATERIAL_VERSION}:expected.add('source_catalog_version')
        recognized=expected==set(value) and bool(value['errors'])
        optional=optional or len(refs)!=len(set(refs))
    elif stage=='tool_builder':
        recognized={'build_request','examples','future_program_api','workspace_capabilities','program_submission_contract'}<=set(value)
    else:
        recognized=experience.get('compact_learning_directory') is True and bool(cards)
    # Same-value interning and compact JSON must not turn a fitting request into a refusal.
    compact=deepcopy(payload)
    sizes=[]
    for material in (value,pack_material(value)):
        compact['messages'][1]['content']=json.dumps(material,ensure_ascii=False,separators=(',',':'),allow_nan=False)
        sizes.append(input_token_bound(compact))
    minimum=min(sizes)
    controlled=stage in {'extractor','tool_builder'} and recognized and not optional and minimum>limit
    return {'classification':'learning_capacity_rejected' if controlled else 'unverified_capacity_refusal',
        'recognized_learning_contract':bool(recognized),'optional_material_remaining':bool(optional),
        'minimum_equivalent_wire_bytes':minimum,'projection_version':'learning-capacity.v1'}


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
    outputs=[]
    def output(value,path):
        if isinstance(value,dict):
            if value and path==['data','computed']:
                outputs.append({'kind':'json','path':path,'type':'dict'})
            for key,item in value.items():output(item,[*path,key])
        else:
            row={'kind':'json','path':path,'type':type(value).__name__}
            if isinstance(value,str) and value:row['content_hint']=value[:256]
            outputs.append(row)
    for key,value in public_result(reference,evidence_origin(evidence)).items():
        if key not in {'accepted','done','error'} and not (key=='observation' and not value):output(value,[key])
    if files:
        outputs += [{'kind':'publication','field':'files','value':files},
                    *[{'kind':'publication','field':'file','name':n,'value':n} for n in files]]
    sources=[r for r in input_sources(evidence) if evidence['kind']=='verified_retrieval_chain.v1' or canonical_bytes(r)<=
             (2048 if r['kind']=='source_literal' and isinstance(r['value'],(list,dict)) else 512)]
    aggregates=[r['path'] for r in sources if r['kind']=='source_literal' and isinstance(r['value'],(list,dict))]
    sources=[r for r in sources if r['kind']!='source_literal' or not any(
        len(r['path'])>len(path) and r['path'][:len(path)]==path for path in aggregates)]
    unique={}
    for row in sources:
        key=(row['kind'],json.dumps(row['value'] if row['kind']=='source_literal' else row['path'],
            sort_keys=True,ensure_ascii=False,allow_nan=False))
        unique.setdefault(key,row)
    sources=list(unique.values())
    operation=evidence['operation']
    source=operation.get('source') or operation.get('arguments',{}).get('source','')
    if operation.get('source_context'):source='\n\n'.join(c['source'] for c in operation['source_context'])
    if source:
        try: source=ast.unparse(ast.parse(source))
        except SyntaxError: source=''
    return {'case_id':case_id,'local_evidence_ref':evidence['id'],'kind':evidence['kind'],
        'operation_name':evidence['operation'].get('name','executed_python'),
        'effect':'published_files' if files else 'inspection_or_json',
        'success':reference.get('accepted',True),'published_files':files,
        'input_sources':sources[:32], 'input_source_count':len(sources), 'output_references':outputs,
        'supported_roles': ['intermediate'] + (['final_files'] if files else []) +
            (['final_answer'] if any(r.get('path',[])[-1:]==['answer'] and r.get('type')=='str'
                                    and r.get('content_hint','').strip() for r in outputs) else []),
        **({'dataflow':operation['edges'], 'actions':[{k:a[k] for k in ('name','event_id','result_id')} for a in operation['actions']],
            'local_goal':operation['local_goal'],'handoff_proof':operation['handoff_proof']}
           if evidence['kind']=='verified_retrieval_chain.v1' else {}),
        **({'executed_calls': [{'name':e['name'],'accepted':e['result']['accepted']}
                              for e in evidence['executed_calls']]} if evidence.get('executed_calls') else {}),
        **({'source_excerpt':source[:2048], 'source_truncated':len(source)>2048} if source else {}),
        'source_content_available_to_builder':True}


def select_operation_directory(evidence, case_id):
    """Publication/code/page representatives survive before discovery-only cards."""
    available=[operation_directory(e,case_id) for e in evidence]
    selected, omitted, seen, names = [], [], set(), set()
    def code(row): return row['success'] and (row['kind']=='executed_python' or row['operation_name']=='execute_python')
    def mode(index,row):
        if not code(row):return row['operation_name']
        op=evidence[index]['operation'];source=op.get('source') or op.get('arguments',{}).get('source','')
        try:
            rpc=any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and
                    isinstance(n.func.value,ast.Name) and n.func.value.id=='ctx' for n in ast.walk(ast.parse(source)))
        except SyntaxError:rpc=False
        reference=evidence[index]['reference'];data=reference.get('data')
        typed=(isinstance(data,dict) and bool(data.get('computed'))) or (row['kind']=='executed_python' and bool(reference))
        if rpc:
            from .public_results import SCOPED_SEARCH
            protocols=evidence[index].get('context_result_protocols', [])
            if TEXT_PAGE in protocols:return 'python_page'
            if SCOPED_SEARCH in protocols:return 'python_search'
        return 'python_rpc' if rpc else 'python_compute' if typed else 'python_stdout'
    modes=[mode(i,row) for i,row in enumerate(available)]
    ordered=sorted(enumerate(available),key=lambda pair:(
        -1 if pair[1]['kind']=='verified_retrieval_chain.v1' else 0 if pair[1]['published_files'] else 1 if modes[pair[0]]=='python_compute' else 2 if code(pair[1]) else
        3 if evidence_origin(evidence[pair[0]])==TEXT_PAGE else 4,
        {'python_page':0, 'python_search':1}.get(modes[pair[0]],2),
        not bool(evidence[pair[0]]['reference'].get('data',{}).get('computed'))
            if isinstance(evidence[pair[0]]['reference'].get('data'),dict) else True,
        -pair[0] if code(pair[1]) else pair[0]))
    for index,row in ordered:
        identity=json.dumps([evidence[index]['operation'], public_result(evidence[index]['reference'],evidence_origin(evidence[index]))],
                            sort_keys=True,ensure_ascii=False,allow_nan=False)
        group=(row['operation_name'],modes[index])
        reason='duplicate_operation' if identity in seen else 'operation_mode_represented' if group in names else 'directory_card_limit'
        seen.add(identity)
        if len(selected)<4 and group not in names and reason!='duplicate_operation':
            selected.append(row);names.add(group)
        else:omitted.append({'local_evidence_ref':row['local_evidence_ref'],'reason':reason})
    return selected, {'available':[r['local_evidence_ref'] for r in available],
                      'included':[r['local_evidence_ref'] for r in selected], 'omitted':omitted}


def learning_case(experience, case_id):
    """No replay prefix, raw result or repeated public task enters the directory."""
    task=experience.get('task',{})
    evidence=experience.get('local_evidence',[])
    selected,audit=select_operation_directory(evidence,case_id)
    return {'case_source':{'physical_case_id':case_id},'task':{'goal':task.get('goal','')},
        **{k:deepcopy(experience.get(k)) for k in ('score','termination_reason','answer_status','completion_truncated')},
        'local_evidence':selected,
        'directory_selection':'verified publication and typed computation, then acquired pages, search and discovery; latest successful code represents each recorded operation mode; at most four cards',
        'projection_audit':{'original_event_count':len(experience.get('events',[])),
            'original_evidence_count':len(experience.get('local_evidence',[])),
            'original_utf8_bytes':canonical_bytes(experience), **audit}, 'compact_learning_directory':True}


def learning_payload(material):
    """Selection receipts are Host audit; selectable cards are the model's directory."""
    value=expand_material(material)
    for case in [value.get('experience',{}), *value.get('completed_train_cases',[])]:
        if case.get('compact_learning_directory'):
            case.pop('projection_audit',None)
            case.pop('directory_selection',None)
            for card in case.get('local_evidence',[]):
                if card.get('input_source_count')==len(card.get('input_sources',[])):
                    card.pop('input_source_count',None)
    return pack_material(value)


def executed_source_capacity(material, learning, default, *, trusted=()):
    """Opt-in Extractor capacity for a real executed Python source directory."""
    from .source_catalog import verified_card
    experience=expand_material(material).get('experience',{})
    cards=experience.get('local_evidence',[])
    source_rich=(experience.get('compact_learning_directory') is True and any(
        (row.get('kind')=='executed_python' or (row.get('kind')=='verified_retrieval_chain.v1' and verified_card(row,trusted)) or
         (row.get('kind')=='native_operation' and row.get('operation_name')=='execute_python'
          and row.get('success') is True and row.get('source_excerpt')
          and verified_card(row,trusted)))
        and row.get('source_content_available_to_builder') is True
        and row.get('local_evidence_ref') and row.get('input_sources') and row.get('output_references') for row in cards))
    return learning.get('source_program_payload_max_bytes',default) if source_rich else default


def capability_capacity(material, learning, purpose, default):
    from . import CHOICE_CAPABILITY_FULL_MATERIAL_VERSION,MATH_CAPABILITY_MATERIAL_VERSION
    if material.get('learning_material_version')==MATH_CAPABILITY_MATERIAL_VERSION and purpose=='math_consolidation':return 131072
    if (material.get('learning_material_version') in {CHOICE_CAPABILITY_FULL_MATERIAL_VERSION,MATH_CAPABILITY_MATERIAL_VERSION} and
        purpose in {'capability_induction','guidance_learning','guidance_grounding','math_consolidation'}):
        return learning.get('capability_payload_max_bytes',default)
    return default


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
    bindings=(value.get('realization_request') or {}).get('case_bindings',[])
    operations=[]
    referenced={b.get('local_evidence_ref') for b in bindings}
    rows=[row for group in directories for row in group]
    matching=[row for row in rows if row['local_evidence_ref'] in referenced]
    if matching:rows=matching
    provenance_fields={'input_sources','output_references','published_files','reference_fields','local_evidence_ref'}
    needs_details=bool(bindings) or any(
        e.get('code') in {'invalid_binding','unknown_source','unbound_input','missing_output_reference','invalid_output_reference','no_publication'} or
        (e.get('field') or '').split('.')[0] in provenance_fields
        for group in errors for e in group['errors'])
    for row in rows:
        if needs_details:
            operations.append({k:deepcopy(row[k]) for k in ('case_id','local_evidence_ref','operation_name','kind','effect','published_files','output_references')})
            operations[-1]['input_sources']=deepcopy(row['input_sources'])
        else:
            operations.append({k:row[k] for k in ('operation_name','local_evidence_ref')})
    short=[{'binding_index':row['binding_index'],'errors':[{'code':e['code'],'field':e.get('field'),
        **({'expected':e['detail'][:400]} if e.get('detail') else {}),
        **({'repair_target':e['repair_target']} if e.get('repair_target') else {})} for e in row['errors']]} for row in errors]
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
        if getattr(adapter,'trace_retrieval',False) and context:
            from .office_runtime import evidence_view, compact_recent
            state['retrieval_evidence']=evidence_view(context,task.goal)
            recent=compact_recent(recent,context)
        completed = value.pop('completed_results')
        if context:
            for row in recent:
                rid = row.get('result_id')
                if not rid: continue
                original = context.public_result(context.results[rid],origin=context.origin(rid))
                if original.get('observation') and original.get('observation') == state.get('observation'):
                    row['observation'] = {'result_id': rid, 'path': ['observation']}
                for fields in completed.values():
                    for key, field in fields.items():
                        if key in original.get('outputs', {}) and field == original['outputs'][key]:
                            fields[key] = {'result_id': rid, 'path': ['outputs', key]}
        # Memory stores query/scope/acceptance and the result ID; previews live in recent_results.
        calls = {k: value.pop(k) for k in ('tools', 'programs', 'allowed_calls')}
        temporary = getattr(adapter, 'task_program_contract', lambda tools: None)(calls['tools'])
        if temporary: calls['task_local_program'] = temporary
        return pack_material({'model_view_version': MODEL_VIEW_VERSION, 'task': public_task,
            'node': {'id': value.pop('node_id'), 'goal': node_goal, 'interface': interface},
            'bindings': {'inputs': inputs, 'input_sources': sources, 'missing': value.pop('missing'),
                         'completed_results': completed},
            'handoff': {k: value.pop(k) for k in ('required_handoff_fields', 'handoff_consumers', 'return_example', 'pending_outputs', 'output_aliases')},
            'state': state,
            'calls': calls,
            'memory': {'guidance': value.pop('guidance'), 'operations': memory, 'recent_results': recent,
                       'public_notes': context.public_notes if context else []},
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
