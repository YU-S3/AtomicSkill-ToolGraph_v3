"""P6's single operation contract. Public syntax is evidence of syntax, never truth."""
import ast
from copy import deepcopy
from fractions import Fraction
import json
import re

from .contracts import digest, object_schema
from .math_program_induction import IMAGE

VERSION = 'livemath.math-capability-program.p6'
CONTRACT = 'livemath.program-candidate.p6'
MATERIAL = VERSION + '.train-only.v1'
FINALIZATION = VERSION + '.bounded-finish'
TEXT = {'type':'string','minLength':1,'maxLength':512}
STRINGS = {'type':'array','items':TEXT,'minItems':1,'maxItems':8}
QUANTIFIER = re.compile(r'\\(forall|exists)\s*(?:\\(bar|vec)\s*)?([a-zA-Z])\b')


class ContractError(ValueError):
    def __init__(self, errors):
        self.diagnostics = errors
        super().__init__(json.dumps(errors,ensure_ascii=False))


def error(code, path, expected, value=None, source_ref=None):
    return {'code':code,'field_path':path,'expected':expected,
            'actual_type':type(value).__name__,'source_ref':source_ref}


def schema_errors(value, schema, path='$'):
    """Report all local fields, checking containers before any indexing or set operation."""
    errors=[];kind=schema.get('type')
    types={'object':dict,'array':list,'string':str,'integer':int,'boolean':bool,'number':(int,float),'null':type(None)}
    if kind and not any(isinstance(value,types[k]) and not (k in {'integer','number'} and isinstance(value,bool))
                        for k in (kind if isinstance(kind,list) else [kind])):
        return [error('field_type_invalid',path,kind,value)]
    if 'const' in schema and value!=schema['const'] or 'enum' in schema and value not in schema['enum']:
        errors.append(error('field_value_invalid',path,schema.get('const',schema.get('enum')),value))
    if isinstance(value,dict):
        props=schema.get('properties',{})
        for k in schema.get('required',[]):
            if k not in value:errors.append(error('required_field_missing',path+'.'+k,'required',None))
        for k,v in value.items():
            if k in props:errors+=schema_errors(v,props[k],path+'.'+k)
            elif schema.get('additionalProperties') is False:errors.append(error('unexpected_field',path+'.'+k,'not allowed',v))
            elif isinstance(schema.get('additionalProperties'),dict):errors+=schema_errors(v,schema['additionalProperties'],path+'.'+k)
    if isinstance(value,list):
        for i,v in enumerate(value):errors+=schema_errors(v,schema.get('items',{}),f'{path}[{i}]')
    if isinstance(value,(str,list)):
        lo,hi=('minLength','maxLength') if isinstance(value,str) else ('minItems','maxItems')
        if len(value)<schema.get(lo,0) or len(value)>schema.get(hi,float('inf')):
            errors.append(error('field_length_invalid',path,{lo:schema.get(lo),hi:schema.get(hi)},value))
    if type(value) in (int,float) and (value<schema.get('minimum',-float('inf')) or value>schema.get('maximum',float('inf'))):
        errors.append(error('field_range_invalid',path,'within declared range',value))
    return errors


def require(value, schema, path='$'):
    errors=schema_errors(value,schema,path)
    if errors:raise ContractError(errors)


def interface(kind):
    integer={'type':'integer','minimum':-1000000,'maximum':1000000}
    if kind=='quantifier_dependency_audit':
        inp=object_schema({'statement':{'type':'string','minLength':1,'maxLength':16000}},['statement'])
        token=object_schema({'operator':{'enum':['forall','exists']},'variable':TEXT,
            'start':{'type':'integer'},'end':{'type':'integer'}},['operator','variable','start','end'])
        out={'quantifiers':{'type':'array','items':token},'uninterpreted_predicates':{'const':True}}
        scope='syntactic_order_only_not_truth_or_scope_proof'
        semantics=("Scan statement with Python re pattern " + repr(QUANTIFIER.pattern) +
            ". Return operator=group(1), variable=('\\\\'+group(2)+' ' if group(2) else '')+group(3), "
            "and original match start/end offsets in order. No truth or predicate interpretation. tested_cases is token count; "
            "no matches means needs_input. Preserve exact original text; never infer missing quantifiers.")
    elif kind=='exact_rational_comparison':
        seq={'type':'array','minItems':1,'maxItems':64,'items':{'type':'array','items':integer,'minItems':2,'maxItems':2}}
        inp=object_schema({'left':seq,'right':seq},['left','right'])
        out={'comparisons':{'type':'array','items':{'enum':[-1,0,1]}}};scope='finite_explicit_pairs_only'
        semantics='Zip left/right positionally, equal positive lengths, denominators nonzero; compare exactly with Fraction. No Cartesian product, float or padding. tested_cases=len(left).'
    elif kind=='finite_relation_quantifiers':
        inp=object_schema({'relation':{'type':'array','minItems':1,'maxItems':32,'items':{'type':'array','minItems':1,'maxItems':32,'items':{'type':'boolean'}}}},['relation'])
        out={'forall_exists':{'type':'boolean'},'exists_forall':{'type':'boolean'},
            'common_witness_columns':{'type':'array','items':{'type':'integer'}}};scope='explicit_finite_relation_only'
        semantics='Nonempty rectangular Boolean relation, rows=complete supplied x domain, columns=complete supplied y domain. forall_exists=all rows have true; common_witness_columns=zero-based columns all true; exists_forall=bool(common_witness_columns). tested_cases=number of cells. No infinite-domain conclusion.'
    else:raise ContractError([error('operation_semantics_missing','operation_kind','registered operation',kind)])
    output=object_schema({**out,'tested_cases':{'type':'integer','minimum':0},'proof_scope':{'const':scope}},[*out,'tested_cases','proof_scope'])
    return {'input_schema':inp,'output_schema':output,'proof_scope':scope,'result_role':'intermediate',
        'allowed_tools':[],'probe_contract':{'kind':'independent_finite_oracle','output_field':next(iter(out))},
        'semantics':semantics+' Missing/invalid inputs: needs_input with outputs={}. Success: status=ok with exact output fields.'}


KINDS=('quantifier_dependency_audit','exact_rational_comparison','finite_relation_quantifiers')
REF=object_schema({'kind':{'enum':['public_span_parse','runtime_required']},'span_id':TEXT,'parser_id':TEXT,'reason':TEXT},['kind'])
SPEC=object_schema({'operation_kind':{'enum':list(KINDS)},'input_schema':{'type':'object'},'output_schema':{'type':'object'},
    'result_role':{'const':'intermediate'},'allowed_tools':{'type':'array','maxItems':0},'proof_scope':TEXT,
    'probe_contract':{'type':'object'},'input_bindings':{'type':'object','additionalProperties':REF},
    'demonstration_input':{'type':'object'}},['operation_kind','input_schema','output_schema','result_role','allowed_tools','proof_scope','probe_contract','input_bindings'])
ATOMIC=object_schema({'goal':TEXT,'applicability':TEXT,'preconditions':STRINGS,'operation':TEXT,'stop_conditions':STRINGS},
    ['goal','applicability','preconditions','operation','stop_conditions'])
PROPOSAL=object_schema({'contract_version':{'const':CONTRACT},'decision':{'enum':['program_candidate','guidance_only','theorem_precondition_only','no_change']},
    'source_case_ids':STRINGS,'public_source_refs':{'type':'array','items':TEXT,'minItems':1,'maxItems':16},
    'atomic':ATOMIC,'program_spec':SPEC,'reason':TEXT},['contract_version','decision'])
BUILD=object_schema({'source':{'type':'string','minLength':1,'maxLength':16000}},['source'])


def oracle(kind, inputs):
    io=interface(kind);require(inputs,io['input_schema'],'inputs')
    try:
        if kind=='quantifier_dependency_audit':
            tokens=[{'operator':m[1],'variable':('\\'+m[2]+' ' if m[2] else '')+m[3],'start':m.start(),'end':m.end()} for m in QUANTIFIER.finditer(inputs['statement'])]
            if not tokens:raise ValueError('No explicit formal quantifier token')
            result={'quantifiers':tokens,'uninterpreted_predicates':True,'tested_cases':len(tokens)}
        elif kind=='exact_rational_comparison':
            a,b=inputs['left'],inputs['right']
            if len(a)!=len(b):raise ValueError('Equal positive sequence lengths required')
            pairs=[(Fraction(*x),Fraction(*y)) for x,y in zip(a,b)]
            result={'comparisons':[(x>y)-(x<y) for x,y in pairs],'tested_cases':len(pairs)}
        else:
            rows=inputs['relation']
            if any(len(row)!=len(rows[0]) for row in rows):raise ValueError('Rectangular relation required')
            cols=[i for i in range(len(rows[0])) if all(row[i] for row in rows)]
            result={'forall_exists':all(any(r) for r in rows),'exists_forall':bool(cols),'common_witness_columns':cols,'tested_cases':sum(map(len,rows))}
    except (ValueError,ZeroDivisionError) as exc:
        if isinstance(exc,ContractError):raise
        raise ContractError([error('operation_input_invalid','inputs',str(exc),inputs)]) from exc
    return {**result,'proof_scope':io['proof_scope']}


def public_spans(task):
    """Host identities cover complete public fields; no response, score or gold is read."""
    texts=[(['goal'],task['goal'])]+[(['inputs','choices',i,'text'],c['text']) for i,c in enumerate(task.get('inputs',{}).get('choices',[]))]
    for path,text in texts:
        row={'case_id':task['physical_key'],'task_path':path,'start':0,'end':len(text),'quote':text,'text_hash':digest(text)}
        yield {**row,'span_id':'p6_span_'+digest(row)}


def parse_span(span, parser):
    text=span['quote']
    if parser=='formal_quantifier_text':
        oracle('quantifier_dependency_audit',{'statement':text});return text
    # Literal objects must be explicitly labelled with operation semantics in public text.
    if parser in {'explicit_finite_relation','explicit_rational_left','explicit_rational_right'}:
        label='finite_relation' if parser=='explicit_finite_relation' else 'rational_pairs'
        match=re.search(r'\b'+label+r'\s*=\s*',text)
        if not match:raise ContractError([error('unbound_semantics','span',label+' = literal JSON',text,span['span_id'])])
        try:value,end=json.JSONDecoder().raw_decode(text[match.end():])
        except json.JSONDecodeError as exc:raise ContractError([error('public_parse_invalid','span','literal JSON',text,span['span_id'])]) from exc
        if parser=='explicit_finite_relation':oracle('finite_relation_quantifiers',{'relation':value});return value
        oracle('exact_rational_comparison',value)
        return value['left' if parser.endswith('left') else 'right']
    raise ContractError([error('parser_unknown','parser_id','registered deterministic parser',parser,span['span_id'])])


PARSERS={'quantifier_dependency_audit':{'statement':'formal_quantifier_text'},
    'exact_rational_comparison':{'left':'explicit_rational_left','right':'explicit_rational_right'},
    'finite_relation_quantifiers':{'relation':'explicit_finite_relation'}}


def ready_bindings(task, kind):
    """A syntactic audit can bind a formula; a truth-matrix calculation cannot invent one."""
    for span in public_spans(task):
        bindings={k:{'kind':'public_span_parse','span_id':span['span_id'],'parser_id':p} for k,p in PARSERS[kind].items()}
        try:
            values={k:parse_span(span,p) for k,p in PARSERS[kind].items()}
            oracle(kind,values)
        except ContractError:continue
        return {'kind':'GroundedInput','operation_kind':kind,'grounded_task_inputs':values,'input_bindings':bindings,
            'public_source_refs':[span],'input_hash':digest(values),'physical_key':task['physical_key'],
            'relevance_checked':True,'proof_obligation':interface(kind)['semantics'],
            'qualification':'task_binding_checked_not_task_solved'}
    return None


def bind(spec, spans):
    require(spec,SPEC,'program_spec');io=interface(spec['operation_kind']);errors=[];values={};bound_spans=[]
    fields=io['input_schema']['properties']
    if set(spec['input_bindings'])!=set(fields):
        raise ContractError([error('binding_fields_mismatch','program_spec.input_bindings',list(fields),spec['input_bindings'])])
    for field,ref in spec['input_bindings'].items():
        path='program_spec.input_bindings.'+field
        if ref['kind']=='runtime_required':
            if set(ref)!={'kind','reason'}:errors.append(error('binding_contract_invalid',path,'kind and reason only',ref))
            continue
        if set(ref)!={'kind','span_id','parser_id'}:errors.append(error('binding_contract_invalid',path,'kind, span_id, parser_id',ref));continue
        span=spans.get(ref['span_id'])
        if not span:errors.append(error('source_reference_unknown',path,'Host source Span ID',ref,ref['span_id']));continue
        if ref['parser_id']!=PARSERS[spec['operation_kind']][field]:errors.append(error('parser_semantics_mismatch',path,'parser for declared field',ref));continue
        try:
            values[field]=parse_span(span,ref['parser_id']);bound_spans.append(span)
        except ContractError as exc:errors.extend(exc.diagnostics)
    if len({s['case_id'] for s in bound_spans})>1 or (spec['operation_kind']=='exact_rational_comparison' and len({s['span_id'] for s in bound_spans})>1):
        errors.append(error('cross_source_binding','program_spec.input_bindings','one physical task and one explicit paired object',spec['input_bindings']))
    if errors:raise ContractError(errors)
    if len(values)!=len(fields):return None
    oracle(spec['operation_kind'],values);return values


def validate_spec(value, unit, rows):
    if isinstance(value,dict) and isinstance(value.get('decision'),str) and value['decision'] in {'no_change','guidance_only','theorem_precondition_only'}:
        require({k:value[k] for k in ('contract_version','decision','reason') if k in value},PROPOSAL)
        return {k:value[k] for k in ('contract_version','decision','reason') if k in value}
    require(value,PROPOSAL)
    require(value,{'type':'object','required':['source_case_ids','public_source_refs','atomic','program_spec']})
    source=value['source_case_ids']
    if len(source)!=len(set(source)) or not set(source)<=set(unit['source_case_ids']):
        raise ContractError([error('source_identity_invalid','source_case_ids','assigned distinct Train sources',source)])
    spans={s['span_id']:s for r in rows if r['task']['physical_key'] in source for s in public_spans(r['task'])}
    for ref in value['public_source_refs']:
        if ref not in spans:raise ContractError([error('source_reference_unknown','public_source_refs','assigned public Span',ref,ref)])
    if {spans[ref]['case_id'] for ref in value['public_source_refs']}!=set(source):
        raise ContractError([error('source_evidence_missing','public_source_refs','at least one exact public reference per declared source',value['public_source_refs'])])
    spec=value['program_spec'];io=interface(spec['operation_kind']);errors=[]
    if spec['operation_kind']!=unit['operation_kind']:errors.append(error('operation_semantics_mismatch','program_spec.operation_kind',unit['operation_kind'],spec['operation_kind']))
    for key in ('input_schema','output_schema','result_role','allowed_tools','proof_scope','probe_contract'):
        if spec[key]!=io[key]:errors.append(error('schema_contract_mismatch','program_spec.'+key,io[key],spec[key]))
    if errors:raise ContractError(errors)
    grounded=bind(spec,spans)
    if 'demonstration_input' in spec:oracle(spec['operation_kind'],spec['demonstration_input'])
    if grounded is None and 'demonstration_input' not in spec:
        raise ContractError([error('operation_input_missing','program_spec','grounded input or explicitly synthetic demonstration',spec)])
    return deepcopy(value)


def validate_source(value):
    require(value,BUILD);source=value['source']
    try:tree=ast.parse(source)
    except SyntaxError as exc:raise ContractError([error('source_syntax_invalid','source',str(exc),source)]) from exc
    for n in ast.walk(tree):
        if isinstance(n,(ast.Import,ast.ImportFrom)):
            names=[a.name for a in n.names] if isinstance(n,ast.Import) else [n.module or '']
            if any(s.split('.')[0] not in {'math','fractions','itertools','functools','collections','re'} for s in names):
                raise ContractError([error('disallowed_import',f'source.line[{n.lineno}]','declared pure standard library',names)])
        if isinstance(n,ast.Name) and n.id in {'open','eval','exec','compile','__import__','globals','locals','getattr','setattr'} or isinstance(n,ast.Attribute) and (n.attr.startswith('_') or isinstance(n.value,ast.Name) and n.value.id=='ctx'):
            raise ContractError([error('disallowed_authority',f'source.line[{n.lineno}]','no files, dynamic code, private attributes or RPC',source)])
        if isinstance(n,ast.Constant) and isinstance(n.value,str) and re.search(r'<answer>|correct_choice|correct_answer|task_id|livemath:',n.value):
            raise ContractError([error('answer_content_forbidden','source','parameterized intermediate operation',n.value)])
    return source


def fingerprint(kind):
    return digest({'version':CONTRACT,'operation_kind':kind,**interface(kind)})


PROMPT='''Infer one local mathematical operation from these saved public Train experiences and feedback.
A wrong final answer does NOT identify a false intermediate step. PublicDemand and model-reported process
are hypotheses, not proofs. Use the assigned operation only if relevant; no_change/guidance_only/
theorem_precondition_only are normal. Do not solve the question or copy a theorem/option conclusion.
Use real source_case_ids and Host public Span IDs. At least two sources when their demands agree;
single-source hypotheses are explicitly labelled. Copy registry schemas/probe_contract exactly.
input_bindings is a field-name object of typed references. Missing fields remain runtime_required.
demonstration_input is synthetic, never evidence of real task binding or transfer. No invented finite
domain, truth matrix or theorem premise. All prose fields <=512 characters, target <=240.
Submit the exact supplied contract; abstention only needs contract_version, decision, optional reason.'''
BUILD_PROMPT='''Implement only the attached registry operation as def run(ctx, inputs).
Use the exact input/output contract and independent mathematical semantics. Return status/outputs;
missing or invalid inputs -> needs_input with empty outputs. Preserve proof_scope and tested_cases.
Only math, fractions, itertools, functools, collections and re; no files, network, RPC, model/oracle calls,
task IDs, option answers, default values or floats for exact arithmetic. Do not alter the interface.
Source target <=8000 characters, hard limit16000. Submit only source; no note or other fields.'''
