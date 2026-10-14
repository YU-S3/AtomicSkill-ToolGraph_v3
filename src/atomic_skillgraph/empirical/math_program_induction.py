"""P5 Train-only operation hypotheses. Neither outcomes nor old utility certify code."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import re

from .bank import Bank
from .contracts import PublicTask, digest, object_schema, validate_program, validate_schema_instance
from . import math_capability as math
from .math_trace_learning import trace_records

VERSION = 'livemath.math-program.p5'
MATERIAL = VERSION + '.train-only.v1'
IMAGE = 'sha256:4db520e20cd121f830731c2d0c0bcfbe9cacd254d1817262404cc18e1e757783'
KINDS = ('finite_predicates', 'finite_quantifiers', 'rational_compare')
TEXT = {'type':'string', 'minLength':1, 'maxLength':800}
ATOMIC = object_schema({'goal':TEXT, 'applicability':TEXT, 'prerequisites':{'type':'array','items':TEXT,'minItems':1,'maxItems':6},
    'operation':TEXT, 'outputs':{'type':'array','items':TEXT,'minItems':1,'maxItems':8},
    'stop_condition':TEXT, 'result_role':{'enum':['intermediate']}},
    ['goal','applicability','prerequisites','operation','outputs','stop_condition','result_role'])
SPEC = object_schema({'input_schema':{'type':'object'}, 'output_schema':{'type':'object'},
    'allowed_tools':{'type':'array','maxItems':0}, 'finite_scope':TEXT,
    'negative_cases':{'type':'array','items':TEXT,'minItems':1,'maxItems':5},
    'probe_kind':{'enum':list(KINDS)}, 'source_input':{'type':'object'},
    'input_derivation':TEXT}, ['input_schema','output_schema','allowed_tools','finite_scope',
        'negative_cases','probe_kind','source_input','input_derivation'])
PROPOSAL = object_schema({'decision':{'enum':['propose_program','guidance_only','no_change']},
    'source_opportunity_ids':{'type':'array','items':{'type':'string'},'minItems':1,'maxItems':1},
    'atomic':ATOMIC,'program_spec':SPEC,'limitations':TEXT}, ['decision'])
BUILD = object_schema({'source':{'type':'string','minLength':1,'maxLength':16000}},['source'])
PROMPT = '''Identify one finite, independently checkable local operation in the public Train experience,
including failures. It is a hypothesis, not a past verified execution or a known diagnosis of the wrong answer.
Produce one atomic capability and parameterized ProgramSpec, or no_change/guidance_only.
Do not solve the complete theorem, choose an option letter, or copy a specific theorem conclusion.
The optional old Atomic descriptions are Train-only hypotheses, not proofs. Use only listed opportunity IDs.
Select a supported probe_kind and copy its exact input/output schemas. Define a small source_input from
the public mathematical expressions; explicitly explain any finite instantiation and its limitation.
No missing theorem premise may be guessed. A finite instantiation tests only that instance, not the theorem.
Target each prose field <=300 characters (Host tolerance 800). No source code at this stage.
Use only submit_math_program_spec with the supplied schema. Abstention needs only decision.'''
BUILD_PROMPT = '''Write a complete parameterized def run(ctx, inputs) for the frozen Atomic/ProgramSpec.
Return {'status': 'ok', 'outputs': ...} or {'status':'needs_input','outputs':{}} for insufficient/invalid input.
Use explicit JSON parameters only; no tools, ctx calls, files, network, grader, task IDs, answer letters,
prefilled source results or unexplained task-specific constants. Python standard math/fractions/itertools
are available. Independently compute the finite operation; do not call a Host oracle.
Report tested domain and finite limitation exactly as specified. Finite checks never prove general theorems.
Keep code concise (target <=8000 characters, Host maximum 16000). Submit only source via submit_program.'''


def load_sources(source, p4, read_json=None):
    """An explicit allowlist: never traverse P4 Val, consolidation, candidates or Test."""
    from ..experiments.canonical_manifest import sha256
    read_json = read_json or (lambda p: json.loads(Path(p).read_text()))
    source, p4 = Path(source), Path(p4)
    files = {}
    def read(p):
        files[str(p.resolve())] = sha256(p)
        return read_json(p)
    original = read(source/'train/config.json'); manifest = read(source/'run_manifest.json')
    summary = read(source/'train/summary.json'); execution = read(source/'train/execution_manifest.json')
    if (not summary['complete'] or summary['tasks'] != 60 or manifest['benchmark'] != 'livemath'
            or manifest['run_seed'] != 42 or execution['config'] != original):
        raise ValueError('P5 requires complete original LiveMath seed42 Train60')
    bank_path = source/'train/bank/bank.sqlite3'; files[str(bank_path.resolve())] = sha256(bank_path)
    bank = Bank(bank_path.parent, readonly=True, environment=original['program_environment'])
    try:
        if bank.digest() != summary['knowledge_digest']: raise ValueError('Source Train Bank changed')
        cases = bank.train_cases()
    finally: bank.close()
    if [c['task'] for c in cases] != execution['tasks'] or execution['tasks'] != [c['task'] for c in summary['cases']]:
        raise ValueError('Train task/score/order identity differs')
    if any(c['task']['split'] != 'train' for c in cases): raise ValueError('Non-Train learning source')
    traces = {}
    for p in sorted((source/'train/traces').glob('*.json')):
        t = read(p); key = t['task']['physical_key']
        if key in traces: raise ValueError('Duplicate physical trace')
        traces[key] = t
    rows, _ = trace_records(cases, traces)
    if len(rows) != 60: raise ValueError('Missing Train traces')
    if any(c['score']!=traces[c['task']['physical_key']]['score'] for c in summary['cases']):
        raise ValueError('Summary and original trace score differ')
    # Compare the declared P4 source projection, but do not import its policy/outcomes.
    if read(p4/'source_cases.json') != cases or read(p4/'source_records.json') != rows:
        raise ValueError('P4 Train projection does not match raw Train')
    by_key = {r['task']['physical_key']:r for r in rows}; atoms = []
    for i in range(15):
        log = read(p4/'learning/units'/str(i)/'learning.json')
        if log['method_version'] != math.P4 or log['stage'] != 'learning': raise ValueError('Not a P4 Train unit')
        for c in log['candidates']:
            if not math.checked_candidate(c) or not set(c['source_case_ids']) <= by_key.keys():
                raise ValueError('Unchecked or non-Train Atomic')
            if any(r != by_key[r['task']['physical_key']] for r in c['source_records']):
                raise ValueError('Atomic source projection changed')
            atoms.append({'unit':i, **{k:deepcopy(c[k]) for k in
                ('id','category','operation','guidance','applicability','prerequisites','stop_condition','source_case_ids','references')}})
    if len(atoms) != 15: raise ValueError('P5 requires all fifteen pre-Val Train Atomic descriptions')
    public = []
    for row in rows:
        task = row['task']; trace = traces[task['physical_key']]; observation = row['observation']
        computations=deepcopy(observation['computations'])
        for computation in computations:
            execution=computation['execution']
            computation['evidence_kind']='observed_operation' if execution.get('status')=='ok' and computation.get('result') else 'failed_execution'
        public.append({'task':deepcopy(task), 'spans':deepcopy(row['spans']),
            'trace_sha256':files[str((source/'train/traces'/(task['task_id']+'.json')).resolve())],
            'parent_outcome':'correct' if trace['score']['hard'] else 'empty' if observation['answer_status']!='valid' else 'wrong',
            'saved_model_answer':observation['submission'], 'public_process':observation['public_process'],
            'computations':computations, 'observed_execution':False})
    return public, atoms, files, original, manifest


def catalog(rows, atoms, seed=42, maximum=6):
    """Demand matching and deterministic diversity, never Val-based ranking."""
    patterns = {
        'finite_predicates':r'congruen|modulo|\\pmod|\\bmod|divisib',
        'finite_quantifiers':r'for every|for each|there exists|uniform|unique|\\forall|\\exists',
        'rational_compare':r'rational|finite (?:sum|sequence)|integer|matrix|fraction'}
    opportunities = []
    for row in rows:
        key = row['task']['physical_key']
        for kind in KINDS:
            refs = [{'case_id':key,'span_id':s['span_id']} for s in row['spans'] if re.search(patterns[kind],s['quote'],re.I)]
            if not refs: continue
            item = {'schema':'math_operation_opportunity.v1','source_case_ids':[key],
                'origin':'answer_supported_hypothesis' if row['parent_outcome']=='correct' else 'problem_derived_hypothesis',
                'parent_outcome':row['parent_outcome'],'observed_execution':False,
                'public_source_refs':refs[:4], 'requested_local_operation':kind,
                'limits':{'finite_only':True,'max_enumerated_cases':100000},
                'source_trace_hashes':[row['trace_sha256']],
                'atomic_source_ids':[a['id'] for a in atoms if key in a['source_case_ids']],
                'qualification':'Public demand suggests a possible finite sub-operation; no theorem or historical execution claim'}
            item['opportunity_id']='op_'+digest(item); opportunities.append(item)
    selected=[];used=set()
    for outcome in ('correct','wrong_or_empty'):
        for kind in KINDS:
            options=[o for o in opportunities if o['requested_local_operation']==kind
                and (o['parent_outcome']=='correct')==(outcome=='correct') and o['source_case_ids'][0] not in used]
            if options:
                o=min(options,key=lambda v:digest([seed,v['opportunity_id']]))
                selected.append(o);used.update(o['source_case_ids'])
    for o in sorted(opportunities,key=lambda v:digest([seed,v['opportunity_id']])):
        if len(selected)>=maximum:break
        if o['source_case_ids'][0] not in used:selected.append(o);used.update(o['source_case_ids'])
    return opportunities, selected[:maximum]


def material(op, rows, atoms):
    from .math_program_probe import interface, semantics
    selected=[r for r in rows if r['task']['physical_key'] in op['source_case_ids']]
    return {'opportunity':op,'train_sources':selected,
        'train_atomic_hypotheses':atoms,
        'finite_interfaces':{op['requested_local_operation']:interface(op['requested_local_operation'])},
        'probe_semantics':semantics(op['requested_local_operation'])}


def validate_spec(value, op):
    from .math_program_probe import interface, oracle
    if isinstance(value,dict) and value.get('decision') in {'no_change','guidance_only'}:
        return {'decision':value['decision']}
    validate_schema_instance(value,PROPOSAL)
    if set(value) != set(PROPOSAL['properties']) or value['source_opportunity_ids'] != [op['opportunity_id']]:
        raise ValueError('Program proposal requires the exact assigned opportunity and complete Atomic/Spec')
    spec=value['program_spec'];kind=spec['probe_kind']
    if kind!=op['requested_local_operation']:raise ValueError('Operation differs from frozen public demand')
    io=interface(kind)
    if spec['input_schema']!=io['input_schema'] or spec['output_schema']!=io['output_schema']:
        raise ValueError('Spec must use the declared independent finite probe interface')
    if not value['limitations'].strip():raise ValueError('Finite limitation required')
    oracle(kind,spec['source_input'])  # Well-defined finite input, not a source theorem proof.
    return deepcopy(value)


def atomic(value, op):
    result={**deepcopy(value['atomic']), 'method_version':VERSION, 'state':'candidate',
        'knowledge_type':'procedural', 'execution_intent':'program_candidate',
        'generation_basis':'hypothesis_from_train','source_case_ids':op['source_case_ids'],
        'public_source_refs':op['public_source_refs'],'parent_outcome':op['parent_outcome'],
        'observed_execution':False,'limitations':value['limitations']}
    result['id']='skill_'+digest(result);return result


def program(source, spec, asset, environment, parent=None):
    """Static admission supplements, but never replaces, the real container boundary."""
    try:tree=ast.parse(source)
    except SyntaxError as exc:raise ValueError('Generated Python syntax: '+str(exc)) from exc
    for n in ast.walk(tree):
        if isinstance(n,(ast.Import,ast.ImportFrom)):
            names=[a.name for a in n.names] if isinstance(n,ast.Import) else [n.module or '']
            if any(s.split('.')[0] not in {'math','fractions','itertools','functools','collections'} for s in names):
                raise ValueError('P5 pure computation permits only the stated standard libraries')
        if isinstance(n,ast.Name) and n.id in {'open','eval','exec','compile','__import__','globals','locals','getattr','setattr'}:
            raise ValueError('P5 code cannot inspect files, evaluators or dynamic execution')
        if isinstance(n,ast.Attribute) and (n.attr.startswith('_') or isinstance(n.value,ast.Name) and n.value.id=='ctx'):
            raise ValueError('P5 pure computation cannot use RPC or private attributes')
        if isinstance(n,ast.Constant) and isinstance(n.value,str) and (
                re.search(r'<answer>|correct_choice|task_id|livemath:|^(?:[A-E])$',n.value)):
            raise ValueError('Program must return local parameterized results, not a task/option answer')
    p={'source':source,'entry':'run','input_schema':deepcopy(spec['input_schema']),
        'output_schema':deepcopy(spec['output_schema']),'allowed_tools':[],
        'environment':deepcopy(environment),'result_role':'intermediate','method_version':VERSION,
        'generation_basis':'hypothesis_from_train','observed_execution':False,
        'entry_constraints':{'atomic_skill_id':asset['id'],'spec':deepcopy(spec),'parent_program_id':parent},'state':'candidate'}
    validate_program(p)
    if environment.get('image_digest')!=IMAGE or environment.get('adapter_abi')!='simple.v2':
        raise ValueError('P5 requires its fixed Docker image and Adapter ABI')
    return p
