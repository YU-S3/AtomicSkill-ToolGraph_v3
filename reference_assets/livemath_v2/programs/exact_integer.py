"""Bounded exact gcd and remainder on explicit integer operands, not a general proof."""
from math import gcd


def run(ctx, inputs):
    try:
        a,b,op = inputs['a'],inputs['b'],inputs['operation']
        if any(type(v) is not int or abs(v)>1000000 for v in (a,b)):
            raise ValueError('bounded integers required')
        if op=='gcd': value=gcd(a,b)
        elif op=='mod' and b>0: value=a % b
        else: raise ValueError('unsupported operation or modulus')
        return {'status':'ok','outputs':{'value':value,'scope':'exact_supplied_integers_only'}}
    except (KeyError,TypeError,ValueError):
        return {'status':'needs_input','outputs':{}}
