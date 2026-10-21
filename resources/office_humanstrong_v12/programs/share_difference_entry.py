"""Difference of two shares of the same total, at an explicit output scale."""


def run(ctx,inputs):
    try:
        if inputs['submit_answer'] is not True:raise ValueError('Explicit final submission required')
        cache,evidence={},[]
        values={k:Decimal(read_operand(ctx,inputs[k],cache,evidence)) for k in ('first','second','total')}
        if values['total']==0:raise ValueError('Nonzero common total required')
        with localcontext() as precision:
            precision.prec=50
            value=(values['first']-values['second'])/values['total']
            if inputs['absolute']:value=abs(value)
            unrounded=str(value)
            if inputs['output_scale']=='percent':value*=100
            elif inputs['output_scale']!='fraction':raise ValueError('Choose fraction or percent output explicitly')
            fmt=inputs['format']
            if fmt['scale']!='as_is':raise ValueError('Share calculator already sets scale; do not rescale twice')
            if fmt['style']=='percent' and inputs['output_scale']!='percent':raise ValueError('Percent display requires percent scale')
            answer=format_answer(str(value),fmt)
        return {'status':'ok','outputs':{'answer':answer,
            'quantity':{'value':unrounded,'scale':'fraction','unit':'dimensionless'},
            'formula':'(first-second)/total','output_scale':inputs['output_scale'],
            'evidence':evidence,'proof_scope':'Actual selected cells and share arithmetic; caller selects category semantics.'}}
    except (KeyError,TypeError,ValueError,ArithmeticError) as error:
        return {'status':'needs_input','outputs':{},'detail':str(error)}
