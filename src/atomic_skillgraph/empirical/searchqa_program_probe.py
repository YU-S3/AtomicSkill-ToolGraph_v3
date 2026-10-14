"""New text-derived code is qualified by real source and held-out Train Workers."""
from copy import deepcopy
import json
from pathlib import Path
import re
import time

from .contracts import PublicTask, digest, object_schema, program_digest, validate_program, validate_schema_instance
from .checkpoint import save_json
from .searchqa_guidance import (VERSION, MATERIAL, OUTPUT, QUALIFICATION, IMAGE, APPLICABILITY,
    applicability, documents, terms)
from .task_context import TaskContext
from ..harness.simple_protocol import Broker
from ..harness.benchmarks import AnswerAdapter

INPUT = object_schema({'question':{'type':'string'}, 'context':{'type':'string'}},['question','context'])
SPAN = object_schema({'doc_index':{'type':'integer','minimum':1},'start':{'type':'integer','minimum':0},
    'end':{'type':'integer','minimum':1},'quote':{'type':'string','minLength':1,'maxLength':600}},
    ['doc_index','start','end','quote'])
RESULT = object_schema({'matches':{'type':'array','items':SPAN,'minItems':1,'maxItems':8}},['matches'])
SPEC = object_schema({'goal':{'type':'string','minLength':1,'maxLength':160},
    'selection_basis':{'enum':['question_term','date_or_number','alias_surface','document_partition']},
    'applicability':APPLICABILITY,'source_case_ids':{'type':'array','items':{'type':'string'},'minItems':2,'maxItems':4},
    'proposal_origin':{'enum':['candidate_derived_from_text']},'result_role':{'enum':['intermediate']}},
    ['goal','selection_basis','applicability','source_case_ids','proposal_origin','result_role'])
PROPOSAL = object_schema({'decision':{'enum':['candidate','no_change']},'spec':{'type':'object'},
                         'reason':{'type':'string','maxLength':400}},['decision','reason'])
BUILD = object_schema({'source':{'type':'string','minLength':1,'maxLength':16000}},['source'])


def validate_spec(spec, asset):
    validate_schema_instance(spec,SPEC)
    ids=spec['source_case_ids']
    c=asset['searchqa_metadata']['candidate']
    if len(set(ids))<2 or len(ids)!=len(set(ids)) or set(ids)-set(c['source_case_ids']):
        raise ValueError('Program spec requires distinct original Atomic sources')
    if spec['applicability']!=c['applicability']: raise ValueError('Program cannot invent a different source applicability')


def inputs(task): return {'question':task.goal,'context':task.inputs['context']}


def prepare(source, spec, asset, environment, *, parent=None):
    validate_spec(spec,asset)
    program={'source':source,'entry':'run','input_schema':deepcopy(INPUT),'output_schema':deepcopy(RESULT),
        'allowed_tools':[],'environment':deepcopy(environment),'result_role':'intermediate',
        'method_version':VERSION,'proposal_origin':'candidate_derived_from_text',
        'entry_constraints':{'method_version':VERSION,'output_contract':OUTPUT,'spec':deepcopy(spec),
                             'atomic_skill_id':asset['id'],'parent_program_id':parent},
        'state':'candidate'}
    validate_program(program)
    validate_declaration(program)
    return program


def validate_declaration(program):
    c=program.get('entry_constraints',{})
    if (program.get('method_version')!=VERSION or program.get('proposal_origin')!='candidate_derived_from_text' or
            c.get('method_version')!=VERSION or c.get('output_contract')!=OUTPUT or
            program['input_schema']!=INPUT or program['output_schema']!=RESULT or program['allowed_tools'] or
            program['environment'].get('image_digest')!=IMAGE or program['environment'].get('adapter_abi')!='simple.v2' or
            program.get('result_role')!='intermediate'):
        raise ValueError('Invalid R2 Program authority/interface')
    validate_schema_instance(c.get('spec'),SPEC)
    if not c.get('atomic_skill_id','').startswith('skill_'): raise ValueError('Program requires a real Atomic source')


def check(program, arguments, result):
    """Proves bounded, truthful public localization, not answer correctness/completeness."""
    validate_declaration(program);validate_schema_instance(arguments,INPUT)
    if result.get('status')!='ok': return {'passed':False,'reason':result.get('status','missing_status')}
    validate_schema_instance(result.get('outputs'),RESULT)
    context=arguments['context'];docs=documents(context);seen=set()
    if not arguments['question'].strip() or not context.strip(): raise ValueError('Empty public input cannot be successful localization')
    basis=program['entry_constraints']['spec']['selection_basis']
    for span in result['outputs']['matches']:
        a,b=span['start'],span['end'];doc=next((d for d in docs if d['doc_index']==span['doc_index']),None)
        if (not doc or not doc['start']<=a<b<=doc['end'] or context[a:b]!=span['quote'] or (a,b) in seen):
            raise ValueError('Program output lacks exact public DOC provenance')
        seen.add((a,b));text=span['quote']
        if basis=='question_term' and not set(terms(arguments['question'])) & set(terms(text)):
            raise ValueError('Window does not contain a declared public question term')
        if basis=='date_or_number' and not re.search(r'\d',text): raise ValueError('No declared numeric surface')
        if basis=='alias_surface' and not re.search(r'\b(?:also known|aka|a\.k\.a\.|abbreviat\w*|nicknam\w*|called)\b|\([A-Z]{2,8}\)',text,re.I):
            raise ValueError('No declared alias surface')
        if basis=='document_partition' and (a,b)!=(doc['start'],doc['end']):
            raise ValueError('Declared whole-document output is not a complete DOC')
    return {'passed':True,'reason':'public_span_source_match','scope':'intermediate localization only; no gold/QA correctness assertion'}


def execute(bank, worker, program, task, directory, *, role, arguments=None, governor=None):
    """Durable claim precedes Docker. Interrupted invocation is never silently repeated."""
    directory=Path(directory);arguments=arguments if arguments is not None else inputs(task)
    identity=digest([program['id'],task.physical_key,role,arguments])
    path=directory/(identity+'.json');claim=directory/(identity+'.started.json')
    if path.exists():
        row=json.loads(path.read_text())
        if row['program_digest']!=program_digest(program) or row['arguments']!=arguments: raise ValueError('Worker replay identity changed')
        return row
    if claim.exists(): raise RuntimeError('Unresolved Worker invocation; explicit review required before recovery')
    validate_schema_instance(arguments,INPUT)
    if governor: governor.admit_validation(identity)
    save_json(claim,{'id':identity,'program_digest':program_digest(program),'task_key':task.physical_key,'role':role})
    adapter=AnswerAdapter('searchqa',{task.task_id:{}})
    adapter.reset(task)
    broker=Broker(adapter,1,context=TaskContext({}))
    begin=time.time()
    try:
        result=worker.execute(program,arguments,broker)
        try:validation=check(program,arguments,result)
        except (ValueError,TypeError,KeyError) as exc:validation={'passed':False,'reason':str(exc)}
        boundary=role.startswith('boundary:')
        if boundary:
            # Empty/missing question must decline; the Unicode parameter variation must locate real spans.
            expected_decline=role in {'boundary:empty_context','boundary:empty_question'}
            validation={'passed': result.get('status') in {'not_found','needs_input','blocked'},
                        'reason':'honest_empty_input_fallback'} if expected_decline else validation
        row={'id':identity,'policy':QUALIFICATION,'program_id':program['id'],'program_digest':program_digest(program),
             'environment_hash':digest(program['environment']),'task_key':task.physical_key,'task':asdict_task(task),
             'role':role,'arguments':arguments,'input_hash':digest(arguments),'result':result,
             'output_hash':digest(result.get('outputs',{})),'check':validation,'start_time':begin,'end_time':time.time(),
             'tools':broker.events,'historical_execution':False}
        receipt=result.get('execution_receipt',{})
        if not receipt or receipt.get('image_digest')!=IMAGE or '--network=none' not in receipt.get('command',[]):
            raise RuntimeError('R2 requires a real locked Docker execution receipt')
        save_json(path,row)
        return row
    finally:adapter.close()


def asdict_task(task):
    return {'task_id':task.task_id,'physical_key':task.physical_key,'goal':task.goal,'inputs':task.inputs,'split':task.split}


def valid_receipt(program,row):
    try:
        receipt=row['result']['execution_receipt']
        if (row['policy']!=QUALIFICATION or row['program_digest']!=program_digest(program) or
                row['task_key']!=row['task']['physical_key'] or
                row['id']!=digest([program['id'],row['task_key'],row['role'],row['arguments']]) or
                row['environment_hash']!=digest(program['environment']) or
                row['input_hash']!=digest(row['arguments']) or row['output_hash']!=digest(row['result'].get('outputs',{})) or
                receipt['source_sha256']!=__import__('hashlib').sha256(program['source'].encode()).hexdigest() or
                receipt['input_hash']!=digest(row['arguments']) or receipt['result_hash']!=digest(
                    {k:v for k,v in row['result'].items() if k!='execution_receipt'}) or
                receipt['image_digest']!=IMAGE or not receipt['container'].startswith('skillcompiler-') or
                '--network=none' not in receipt['command'] or '--read-only' not in receipt['command'] or
                row['task']['split']!='train' or row['historical_execution'] is not False): return False
        if row['role'] in {'source','transfer'}:
            task=PublicTask(**row['task'])
            return row['arguments']==inputs(task) and check(program,row['arguments'],row['result'])['passed']
        if row['role'] in {'boundary:empty_context','boundary:empty_question'}:
            empty='context' if row['role']=='boundary:empty_context' else 'question'
            return not row['arguments'][empty] and row['result']['status'] in {'not_found','needs_input','blocked'}
        return (row['role']=='boundary:unicode_parameter' and row['arguments']!=inputs(PublicTask(**row['task'])) and
                check(program,row['arguments'],row['result'])['passed'])
    except (KeyError,TypeError,ValueError):return False


def qualification(program, validation):
    rows=validation.get('receipts',[])
    if (validation.get('policy')!=QUALIFICATION or validation.get('passed') is not True or
            validation.get('program_digest')!=program_digest(program) or validation.get('environment_hash')!=digest(program['environment']) or
            len(rows)<5 or len({r.get('id') for r in rows})!=len(rows) or any(not valid_receipt(program,r) for r in rows)): return False
    source=[r for r in rows if r['role']=='source'];transfer=[r for r in rows if r['role']=='transfer']
    bounds={r['role'] for r in rows if r['role'].startswith('boundary:')}
    return bool(source and transfer and bounds=={'boundary:empty_context','boundary:empty_question','boundary:unicode_parameter'} and
        any(a['task_key']!=b['task_key'] and a['input_hash']!=b['input_hash'] for a in source for b in transfer) and
        all(r['task_key'] in program['entry_constraints']['spec']['source_case_ids'] for r in source))


def qualify(bank,program,rows):
    good=[r for r in rows if r['check']['passed'] and valid_receipt(program,r)]
    validation={'policy':QUALIFICATION,'program_digest':program_digest(program),'environment_hash':digest(program['environment']),
                'passed':True,'receipts':good,'reason':'new_source_and_distinct_train_worker_provenance'}
    if not qualification(program,validation): return {'passed':False,'reason':'coverage_insufficient'}
    roles={r['task']['physical_key']:r['experience'].get('searchqa_role') for r in bank.train_cases()}
    if any(roles.get(r['task_key'])!=('induction' if r['role']=='source' else 'worker_probe') for r in good if r['role'] in {'source','transfer'}):
        raise ValueError('Qualification crosses the preregistered Train partitions')
    attempt={'id':'r2_qualification:'+digest(validation),'program_id':program['id'],'task_key':good[0]['task_key'],
        'origin':'train_test','split':'train','outcome':'positive','basis':'local_check','calls':0,'validation':validation}
    bank.record(attempt)
    bank.record_validation(program['id'],validation)
    bank.put('implementation',{'skill_id':program['entry_constraints']['atomic_skill_id'],'program_id':program['id'],
                              'method_version':VERSION,'qualification_hash':digest(validation)})
    return {'passed':True,'program_id':program['id'],'receipt_count':len(good)}


def build(learner,spec,asset,sources,*,generation=0,previous=None):
    from .prompts import SEARCHQA_BUILDER_PROMPT
    from .program_worker import public_program_abi
    material={'proposal_origin':'candidate_derived_from_text','spec':spec,
        'sources':[sources[k] for k in spec['source_case_ids']], 'input_schema':INPUT,'output_schema':RESULT,
        'public_program_abi':public_program_abi(),'allowed_tools':[], 'environment':learner.system.config['program_environment'],
        'output_contract':OUTPUT,'previous_failure':previous}
    proposal=learner._receive('build_'+asset['id']+'_'+str(generation),'tool_builder',SEARCHQA_BUILDER_PROMPT,
        material,'submit_searchqa_code',BUILD,material_version=MATERIAL,repair_limit=0,
        decision_purpose='searchqa_builder',job_key=[asset['id'],0,generation])
    return prepare(proposal['source'],spec,asset,learner.system.config['program_environment'],
                   parent=previous.get('program_id') if previous else None)
