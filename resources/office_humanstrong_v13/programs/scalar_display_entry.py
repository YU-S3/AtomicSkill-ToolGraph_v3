def run(ctx,inputs):
    try:
        if inputs['submit_answer'] is not True:raise ValueError('Explicit final submission required')
        ref=inputs['value_ref']
        result=ctx.call('read_result',{**ref,'offset':0,'limit':256})
        if not result.get('accepted') or result.get('next_offset') is not None:
            raise ValueError('Use one complete acquired numeric value_ref')
        value=result['data'];fmt=inputs['format']
        answer=render(value,inputs['source_scale'],fmt['output_scale'],fmt['style'],fmt['decimals'],
                      inputs['unit'],fmt.get('suffix'))
        return {'status':'ok','outputs':{'answer':answer,'value':str(value),'format':fmt,
            'source_scale':inputs['source_scale'],'unit':inputs['unit'],'evidence':[{'source_ref':ref}],
            'proof_scope':'Exact acquired scalar and explicit caller-declared scales; conversion/rounding only, not semantic validation.'}}
    except (KeyError,TypeError,ValueError,ArithmeticError) as error:
        return {'status':'needs_input','outputs':{},'detail':str(error)}
