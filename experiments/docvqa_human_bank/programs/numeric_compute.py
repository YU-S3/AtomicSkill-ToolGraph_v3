"""Deterministic arithmetic on values already visually obtained by the model.

No source OCR, no image parsing, no answer lookup and no statement that model-
transcribed operands are correct. result_role must remain intermediate.
"""
from decimal import Decimal, InvalidOperation, localcontext, ROUND_HALF_UP


def run(ctx, inputs):
    try:
        op=inputs['operation']
        data=inputs['values']
        if not isinstance(data,list) or not 1 <= len(data) <= 256:
            raise ValueError('1..256 observed values required')
        if any(type(x) not in (str,int,float) for x in data):
            raise ValueError('Only numeric inputs')
        with localcontext() as precision:
            precision.prec=44
            nums=[Decimal(str(x).replace(',','')) for x in data]
            if any(not x.is_finite() for x in nums):raise ValueError('Nonfinite operand')
            if op=='count':value=Decimal(len(nums));formula='count(values)'
            elif op=='sum':value=sum(nums);formula='sum(values)'
            elif op=='mean':value=sum(nums)/Decimal(len(nums));formula='sum(values)/n'
            elif op=='max':value=max(nums);formula='max(values)'
            elif op=='min':value=min(nums);formula='min(values)'
            elif op in ('difference','absolute_difference','ratio','percent_change'):
                if len(nums)!=2:raise ValueError('Exactly two named values are required')
                a,b=nums
                if op=='difference':value=b-a;formula='to_value - from_value'
                elif op=='absolute_difference':value=abs(b-a);formula='abs(to_value - from_value)'
                elif a==0:raise ValueError('Denominator/base value is zero')
                elif op=='ratio':value=b/a;formula='numerator / denominator'
                else:value=(b-a)/a*100;formula='100*(new_value-base_value)/base_value'
            else:raise ValueError('Unsupported operation')
            digits=inputs.get('decimals')
            if digits is not None:
                if type(digits) is not int or not 0<=digits<=12:raise ValueError('decimals must be 0..12')
                answer=format(value.quantize(Decimal(1).scaleb(-digits),rounding=ROUND_HALF_UP),f'.{digits}f')
            else:
                answer=format(value,'f')
                if '.' in answer:answer=answer.rstrip('0').rstrip('.')
            if answer in ('','-0'):answer='0'
            return {'status':'ok','outputs':{'answer':answer,'value':str(value),
                'formula':formula,'source_values':[str(x) for x in nums],
                'unit':inputs.get('unit','unspecified'),
                'scope':'local_arithmetic_only_operands_must_be_verified_against_image'}}
    except (KeyError,InvalidOperation,ValueError,TypeError,ArithmeticError) as exc:
        return {'status':'needs_input','outputs':{},'detail':str(exc)[:400]}
