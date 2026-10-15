"""Human-authored exact arithmetic. No I/O, eval(), or model calls."""
import ast
from fractions import Fraction


def _value(node, names=None):
    names = names or {}
    if isinstance(node, ast.Expression): return _value(node.body, names)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return Fraction(str(node.value))
    if isinstance(node, ast.Name) and node.id in names: return Fraction(str(names[node.id]))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        x = _value(node.operand, names); return -x if isinstance(node.op, ast.USub) else x
    if isinstance(node, ast.BinOp):
        a, b = _value(node.left, names), _value(node.right, names)
        if max(a.numerator.bit_length(), a.denominator.bit_length(), b.numerator.bit_length(), b.denominator.bit_length()) > 8192:
            raise ValueError('operand too large')
        if isinstance(node.op, ast.Add): return a+b
        if isinstance(node.op, ast.Sub): return a-b
        if isinstance(node.op, ast.Mult): return a*b
        if isinstance(node.op, ast.Div): return a/b
        if isinstance(node.op, ast.FloorDiv): return Fraction(a//b)
        if isinstance(node.op, ast.Mod): return a%b
        if isinstance(node.op, ast.Pow) and b.denominator == 1 and abs(b) <= 100:
            return a**int(b)
    raise ValueError('Only numeric arithmetic and supplied variables are supported')


def run(ctx, inputs):
    try:
        expression = inputs['expression']
        if not isinstance(expression,str) or not expression.strip() or len(expression)>2000:
            raise ValueError('Provide a bounded arithmetic expression')
        tree=ast.parse(expression, mode='eval')
        if len(list(ast.walk(tree)))>300: raise ValueError('expression too complex')
        val = _value(tree, inputs.get('variables', {}))
        digits = inputs.get('digits', 12)
        if type(digits) is not int or not 0<=digits<=30: raise ValueError('digits must be 0..30')
        from decimal import Decimal, localcontext
        with localcontext() as c:
            c.prec=max(50,digits+20)
            dec=Decimal(val.numerator)/Decimal(val.denominator)
            decimal=format(dec, '.'+str(digits)+'f')
        return {'status':'ok','outputs':{'exact':str(val),'decimal':decimal,
            'scope':'Exact arithmetic of the supplied expression; not a proof of its applicability.'}}
    except (KeyError, ValueError, TypeError, SyntaxError, ZeroDivisionError, OverflowError) as e:
        return {'status':'needs_input','outputs':{},'detail':str(e)}
