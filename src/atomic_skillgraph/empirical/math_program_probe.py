"""Independent, bounded mathematical oracles and Docker-only development trials."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import time

from .checkpoint import save_json
from .contracts import PublicTask, digest, object_schema, program_digest, validate_program, validate_schema_instance
from .task_context import TaskContext
from ..harness.benchmarks import AnswerAdapter
from ..harness.simple_protocol import Broker

INTEGER={'type':'integer','minimum':-1000000,'maximum':1000000}
ARRAY={'type':'array','items':INTEGER,'minItems':1,'maxItems':256}
LIMITATION='finite_explicit_input_only_not_general_proof'


def interface(kind):
    if kind=='finite_predicates':
        inp=object_schema({'x_values':ARRAY,'y_values':ARRAY,'left':{'type':'array'},'right':{'type':'array'}})
        point=object_schema({'x':INTEGER,'y':INTEGER},['x','y'])
        out=object_schema({'tested_cases':{'type':'integer'},'left_implies_right':{'type':'boolean'},
            'right_implies_left':{'type':'boolean'},'left_counterexamples':{'type':'array','items':point,'maxItems':1},
            'right_counterexamples':{'type':'array','items':point,'maxItems':1},'limitation':{'enum':[LIMITATION]}},
            ['tested_cases','left_implies_right','right_implies_left','left_counterexamples','right_counterexamples','limitation'])
    elif kind=='finite_quantifiers':
        inp=object_schema({'relation':{'type':'array','minItems':1,'maxItems':32,
            'items':{'type':'array','minItems':1,'maxItems':32,'items':{'type':'boolean'}}}})
        out=object_schema({'tested_cases':{'type':'integer'},'forall_exists':{'type':'boolean'},
            'exists_forall':{'type':'boolean'},'common_witness_columns':{'type':'array','items':{'type':'integer'}},
            'limitation':{'enum':[LIMITATION]}},['tested_cases','forall_exists','exists_forall','common_witness_columns','limitation'])
    elif kind=='rational_compare':
        fractions={'type':'array','minItems':1,'maxItems':64,'items':{'type':'array','items':INTEGER,'minItems':2,'maxItems':2}}
        inp=object_schema({'left':fractions,'right':fractions})
        out=object_schema({'tested_cases':{'type':'integer'},'comparisons':{'type':'array','items':{'enum':[-1,0,1]}},
            'limitation':{'enum':[LIMITATION]}},['tested_cases','comparisons','limitation'])
    else:raise ValueError('Unsupported local oracle')
    # Missing fields deliberately reach the Worker and must produce needs_input.
    return {'input_schema':inp,'output_schema':out}


def semantics(kind):
    return {
        'finite_predicates': 'Enumerate the Cartesian product in supplied order. left/right are prefix expressions: integer or x/y leaves; '
            '[add|sub|mul|mod, a,b], [eq|ne|lt|le|gt|ge,a,b], [and|or,p,q], [not,p]. '
            'Predicates must be boolean, arithmetic integer; no division by zero. Report both implications and the first '
            'counterexample in each direction. Max depth 12, nodes 128, cases 100000, integer intermediates abs<=1000000000.',
        'finite_quantifiers': 'relation is a nonempty rectangular Boolean matrix. Rows are the explicit finite x domain; columns the y domain. '
            'forall_exists means every row has true; exists_forall means at least one column is all true. Return all such zero-based columns.',
        'rational_compare': 'left/right have equal positive length. Each entry is [numerator,denominator], denominator nonzero. '
            'Compare corresponding rational values exactly: -1,0,1; no floating-point approximation.'}[kind]+(
                ' All input fields are semantically required. For missing/invalid data return needs_input/blocked/not_found. '
                'tested_cases is Cartesian product size, matrix cell count, or pair count respectively. '
                'limitation must equal '+LIMITATION+'. This does not establish any source theorem or final option.')


def expression(node, x, y, depth=0):
    if depth>12:raise ValueError('Expression depth exceeds finite limit')
    if type(node) is int and abs(node)<=1000000:return node
    if node=='x':return x
    if node=='y':return y
    if not isinstance(node,list) or not node or not isinstance(node[0],str) or node[0] not in {'add','sub','mul','mod','eq','ne','lt','le','gt','ge','and','or','not'}:
        raise ValueError('Unsupported finite expression')
    op=node[0]
    if len(node)!=(2 if op=='not' else 3):raise ValueError('Expression arity')
    a=expression(node[1],x,y,depth+1)
    b=expression(node[2],x,y,depth+1) if op!='not' else None
    if op in {'and','or','not'}:
        if type(a) is not bool or op!='not' and type(b) is not bool:raise ValueError('Boolean operands required')
        return not a if op=='not' else a and b if op=='and' else a or b
    if type(a) is not int or type(b) is not int:raise ValueError('Integer operands required')
    if op in {'eq','ne','lt','le','gt','ge'}:
        return {'eq':a==b,'ne':a!=b,'lt':a<b,'le':a<=b,'gt':a>b,'ge':a>=b}[op]
    if op=='mod' and b==0:raise ValueError('Zero modulus')
    value=a+b if op=='add' else a-b if op=='sub' else a*b if op=='mul' else a%b
    if abs(value)>1000000000:raise ValueError('Integer intermediate exceeds finite limit')
    return value


def oracle(kind, inputs):
    validate_schema_instance(inputs,interface(kind)['input_schema'])
    def nodes(v):
        pending=[v];count=0
        while pending and count<=128:
            node=pending.pop();count+=1
            if isinstance(node,list):pending.extend(node)
        return count
    try:
        if kind=='finite_predicates':
            xs,ys=inputs['x_values'],inputs['y_values']
            if len(xs)*len(ys)>100000 or any(nodes(inputs[k])>128 for k in ('left','right')):raise ValueError('Finite input bound exceeded')
            losses=[[],[]]
            for x in xs:
                for y in ys:
                    a,b=(expression(inputs[k],x,y) for k in ('left','right'))
                    if type(a) is not bool or type(b) is not bool:raise ValueError('Boolean predicates required')
                    for i,fail in enumerate((a and not b,b and not a)):
                        if fail and not losses[i]:losses[i].append({'x':x,'y':y})
            return {'tested_cases':len(xs)*len(ys),'left_implies_right':not losses[0],
                'right_implies_left':not losses[1],'left_counterexamples':losses[0],
                'right_counterexamples':losses[1],'limitation':LIMITATION}
        if kind=='finite_quantifiers':
            rows=inputs['relation']
            if any(len(row)!=len(rows[0]) for row in rows):raise ValueError('Rectangular relation required')
            witnesses=[j for j in range(len(rows[0])) if all(row[j] for row in rows)]
            return {'tested_cases':sum(map(len,rows)),'forall_exists':all(any(row) for row in rows),
                'exists_forall':bool(witnesses),'common_witness_columns':witnesses,'limitation':LIMITATION}
        left,right=inputs['left'],inputs['right']
        if len(left)!=len(right):raise ValueError('Equal sequence lengths required')
        values=[(Fraction(*a),Fraction(*b)) for a,b in zip(left,right)]
        return {'tested_cases':len(values),'comparisons':[(a>b)-(a<b) for a,b in values],'limitation':LIMITATION}
    except (KeyError,ZeroDivisionError) as exc:raise ValueError('Missing premise or zero denominator') from exc


def probe_inputs(spec):
    kind=spec['probe_kind'];source=deepcopy(spec['source_input']);oracle(kind,source)
    variant=deepcopy(source)
    if kind=='finite_predicates':
        for k in ('x_values','y_values'):
            options=[v for v in range(1,33) if v not in variant[k]]
            for value in options:
                proposed={**variant,k:[value]+variant[k][:8]}
                try:oracle(kind,proposed)
                except ValueError:continue
                variant=proposed;break
            else:variant[k]=list(reversed(variant[k]))
    elif kind=='finite_quantifiers':
        variant['relation']=[[not cell for cell in row] for row in source['relation']]
    else:
        variant['left']=[[a+1,b] for a,b in source['left']]
        variant['right']=[[a-1,b] for a,b in source['right']]
    oracle(kind,variant)
    if variant==source:raise ValueError('No distinct admissible parameter variation')
    return [('source_local_input',source),('parameter_variation',variant),('missing_premise',{})]


class PureBroker(Broker):
    def rpc(self,*args,**kwargs):
        raise ValueError('P5 pure computation forbids all context RPC')


def execute(worker, program, arguments, directory, role):
    """Persist invocation claims; never silently repeat an interrupted container."""
    from .math_program_induction import VERSION, IMAGE
    directory=Path(directory);identity=digest([program_digest(program),arguments,role])
    path=directory/(identity+'.json');claim=directory/(identity+'.started.json')
    if path.exists():
        row=json.loads(path.read_text())
        if row['id']!=identity or row['receipt_hash']!=digest({k:v for k,v in row.items() if k!='receipt_hash'}):
            raise ValueError('Worker receipt content changed')
        return row
    if claim.exists():raise RuntimeError('Unresolved Worker invocation; review required before any repeat')
    validate_program(program)
    if program['environment']['image_digest']!=IMAGE or program['allowed_tools']:raise ValueError('Invalid P5 Worker authority')
    save_json(claim,{'id':identity,'role':role,'program_digest':program_digest(program)})
    adapter=AnswerAdapter('livemath',{identity:{}});adapter.reset(PublicTask(identity,identity,'Finite local development probe'))
    broker=PureBroker(adapter,1,context=TaskContext({}));start=time.time()
    row={'id':identity,'method_version':VERSION,'program_id':program['id'],'program_digest':program_digest(program),
        'role':role,'arguments':arguments,'historical_execution':False,'start_time':start}
    try:
        result=worker.execute(program,arguments,broker);receipt=result.get('execution_receipt',{})
        if not receipt or receipt.get('image_digest')!=IMAGE or '--network=none' not in receipt.get('command',[]):
            raise RuntimeError('Missing real locked Docker receipt')
        if role=='missing_premise':passed=result['status'] in {'needs_input','not_found','blocked'}
        else:
            expected=oracle(program['entry_constraints']['spec']['probe_kind'],arguments)
            passed=result['status']=='ok' and result.get('outputs')==expected
            row['expected_local_outputs']=expected
        row.update(result=result,passed=passed,status='probe_passed' if passed else 'worker_failed',
            docker_exit_code=receipt.get('exit_code'))
    except (RuntimeError,OSError,ValueError) as exc:
        row.update(passed=False,status='blocked' if isinstance(exc,(RuntimeError,OSError)) else 'probe_inconclusive',reason=str(exc))
    finally:adapter.close()
    row.update(end_time=time.time(),rpc_calls=broker.events);row['receipt_hash']=digest(row);save_json(path,row)
    return row


def probe(worker,program,directory):
    try:inputs=probe_inputs(program['entry_constraints']['spec'])
    except (ValueError,TypeError) as exc:return {'state':'probe_inconclusive','reason':str(exc),'receipts':[]}
    receipts=[]
    for role,arguments in inputs:
        row=execute(worker,program,arguments,directory,role);receipts.append(row)
        if row['status']=='blocked':break
    passed=len(receipts)==3 and all(r['passed'] for r in receipts)
    return {'state':'train_trial_only' if passed else 'blocked' if any(r['status']=='blocked' for r in receipts) else 'probe_failed',
        'probe_passed':passed,'usable':False,'consumed':False,'receipts':receipts,
        'qualification':'Only finite local inputs checked; no historical execution, theorem proof or official promotion'}
