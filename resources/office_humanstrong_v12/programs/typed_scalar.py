"""Convert an acquired, typed quantity once; round only for final display."""
from decimal import Decimal, localcontext, ROUND_HALF_UP


def render(value, source_scale, output_scale, style, decimals, unit, suffix=None):
    scales={'quantity','fraction','percent','percentage_points'}
    if source_scale not in scales or output_scale not in scales:
        raise ValueError('Explicit known source/output scale required')
    if type(value) not in (str,int,float) or type(decimals) is not int or not 0<=decimals<=18:
        raise ValueError('Numeric value and decimals 0..18 required')
    if not isinstance(unit,str) or not unit.strip():raise ValueError('Source unit required')
    if style not in ('number','percent','unit_suffix'):raise ValueError('Explicit display required')
    if style=='percent' and output_scale!='percent':raise ValueError('Percent symbol requires percent scale')
    if style=='unit_suffix' and (suffix!=unit or '%' in suffix):
        raise ValueError('Suffix must equal the acquired unit; percent uses percent style')
    if style!='unit_suffix' and suffix is not None:raise ValueError('Unexpected suffix')
    with localcontext() as precision:
        precision.prec=60;number=Decimal(str(value))
        if not number.is_finite():raise ValueError('Finite unrounded value required')
        if source_scale!=output_scale:
            if source_scale in ('percent','percentage_points') and output_scale=='fraction':number/=100
            elif source_scale=='fraction' and output_scale in ('percent','percentage_points'):number*=100
            else:raise ValueError('Unsupported unit/scale conversion; calculate explicitly first')
        number=number.quantize(Decimal(1).scaleb(-decimals),rounding=ROUND_HALF_UP)
        if not number:number=abs(number)
        answer=format(number,f'.{decimals}f')
    return answer+('%' if style=='percent' else ' '+suffix if style=='unit_suffix' else '')


def run(ctx,inputs):
    try:
        if inputs['submit_answer'] is not True:raise ValueError('Explicit final submission required')
        ref=inputs['quantity_ref'];quantity={}
        for field in ('value','scale','unit'):
            result=ctx.call('read_result',{**ref,'path':[*ref['path'],field],'offset':0,'limit':256})
            if not result.get('accepted') or result.get('next_offset') is not None:
                raise ValueError('Use an exact acquired quantity_ref with complete value, scale and unit fields')
            quantity[field]=result['data']
        fmt=inputs['format']
        answer=render(quantity['value'],quantity['scale'],fmt['output_scale'],fmt['style'],fmt['decimals'],
                      quantity['unit'],fmt.get('suffix'))
        return {'status':'ok','outputs':{'answer':answer,'quantity':quantity,'format':fmt,
            'evidence':[{'source_ref':ref}],
            'proof_scope':'Producer-declared quantity, explicit scale conversion and rounding; not proof of task semantics.'}}
    except (KeyError,TypeError,ValueError,ArithmeticError) as error:
        return {'status':'needs_input','outputs':{},'detail':str(error)}
