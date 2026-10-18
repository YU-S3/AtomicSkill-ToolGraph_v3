"""Find a concrete counterexample to a supplied arithmetic implication."""
import ast
import itertools
from fractions import Fraction


def calc(n, env):
    if isinstance(n,ast.Expression): return calc(n.body,env)
    if isinstance(n,ast.Constant) and type(n.value) in (int,float,bool):
        return n.value if type(n.value) is bool else Fraction(str(n.value))
    if isinstance(n,ast.Name) and n.id in env: return Fraction(str(env[n.id]))
    if isinstance(n,ast.UnaryOp):
        v=calc(n.operand,env)
        if isinstance(n.op,ast.USub): return -v
        if isinstance(n.op,ast.UAdd): return +v
        if isinstance(n.op,ast.Not): return not v
    if isinstance(n,ast.BinOp):
        a,b=calc(n.left,env),calc(n.right,env)
        if max(getattr(a,'numerator',0).bit_length(),getattr(b,'numerator',0).bit_length())>4096: raise ValueError('operand too large')
        if isinstance(n.op,ast.Add): return a+b
        if isinstance(n.op,ast.Sub): return a-b
        if isinstance(n.op,ast.Mult): return a*b
        if isinstance(n.op,ast.Div): return a/b
        if isinstance(n.op,ast.FloorDiv): return Fraction(a//b)
        if isinstance(n.op,ast.Mod): return a%b
        if isinstance(n.op,ast.Pow) and b.denominator==1 and abs(b)<=20:return a**int(b)
    if isinstance(n,ast.BoolOp):
        if isinstance(n.op,ast.And):return all(bool(calc(x,env)) for x in n.values)
        if isinstance(n.op,ast.Or):return any(bool(calc(x,env)) for x in n.values)
    if isinstance(n,ast.Compare):
        left=calc(n.left,env)
        for op,right in zip(n.ops,n.comparators):
            right=calc(right,env)
            if isinstance(op,ast.Eq):ok=left==right
            elif isinstance(op,ast.NotEq):ok=left!=right
            elif isinstance(op,ast.Lt):ok=left<right
            elif isinstance(op,ast.LtE):ok=left<=right
            elif isinstance(op,ast.Gt):ok=left>right
            elif isinstance(op,ast.GtE):ok=left>=right
            else:raise ValueError('unsupported comparison')
            if not ok:return False
            left=right
        return True
    raise ValueError('Unsupported syntax: use arithmetic, comparisons, and/or/not only')


def run(ctx,inputs):
    try:
        domain=inputs['domains'];premise=inputs['premise'];conclusion=inputs['conclusion']
        if not isinstance(domain,dict) or not 1<=len(domain)<=5:raise ValueError('1..5 variables required')
        if any(not isinstance(k,str) or not k.isidentifier() or k.startswith('_') for k in domain):raise ValueError('invalid variable')
        total=1
        for values in domain.values():
            if not isinstance(values,list) or not values or any(type(v) not in (int,float,str) for v in values):raise ValueError('nonempty numeric domains required')
            for v in values:Fraction(str(v))
            total*=len(values)
        if total>20000:raise ValueError('bounded enumeration limit is 20000 assignments')
        if any(not isinstance(x,str) or not x.strip() or len(x)>1500 for x in (premise,conclusion)):raise ValueError('bounded premise/conclusion required')
        p,q=ast.parse(premise,mode='eval'),ast.parse(conclusion,mode='eval')
        if len(list(ast.walk(p)))+len(list(ast.walk(q)))>400:raise ValueError('formula too complex')
        tested=admissible=undefined=0;witness=None
        for row in itertools.product(*domain.values()):
            env=dict(zip(domain,row));tested+=1
            try:
                pv=calc(p,env)
                if type(pv) is not bool:raise ValueError('premise must be a Boolean comparison')
                if not pv:continue
                admissible+=1;qv=calc(q,env)
                if type(qv) is not bool:raise ValueError('conclusion must be Boolean')
                if not qv:witness=env;break
            except ZeroDivisionError:undefined+=1
        result={'counterexample_found':witness is not None,'witness':witness,'tested':tested,
                'admissible':admissible,'undefined':undefined,
                'scope':'Counterexample search in supplied finite domains. No witness does NOT prove a general theorem.'}
        return {'status':'ok','outputs':result}
    except (KeyError,ValueError,TypeError,SyntaxError,OverflowError) as e:
        return {'status':'needs_input','outputs':{},'detail':str(e)}
