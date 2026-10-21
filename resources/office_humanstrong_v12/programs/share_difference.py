"""Read selected public cells or acquired scalar results without retyping values."""
import re
from decimal import Decimal


def read_operand(ctx, selector, cache, evidence):
    ref = selector['source_ref']
    key = (ref['result_id'], tuple(ref['path']), selector.get('offset', 0))
    if key not in cache:
        result = ctx.call('read_result', {**ref, 'offset': key[2], 'limit': 12000})
        if not result.get('accepted'):
            raise ValueError('Source read refused: '+str(result.get('error')))
        cache[key] = result
    result = cache[key]
    value = result['data']
    receipt = {'source_ref': ref}
    if 'row' in selector:
        if not isinstance(value, str): raise ValueError('Cell selector needs acquired source text')
        lines = value.splitlines(keepends=True)
        header = selector['header'].strip()
        anchors = [i for i, line in enumerate(lines) if line.strip() == header]
        if len(anchors) != 1: raise ValueError('Use one exact, unique table header line in the source window')
        start = anchors[0]
        matches = []
        for i in range(start+1, len(lines)):
            line = lines[i]
            if not line.lstrip().startswith('|'): break
            if i == len(lines)-1 and result.get('next_offset') is not None and not line.endswith('\n'):
                continue  # Never parse a truncated numeric row.
            cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
            row_column=selector.get('row_column',0)
            if row_column < len(cells) and cells[row_column] == selector['row'].strip() and (
                    'row_text' not in selector or line.strip()==selector['row_text'].strip()):
                matches.append((i, cells))
        if len(matches) != 1:
            raise ValueError('Requested row is absent/repeated; supply the exact full row_text and row_column for its date group')
        index, cells = matches[0]
        column = selector['column']
        if type(column) is not int or not 1 <= column < len(cells):
            raise ValueError('column counts all cells from 0, including blank cells; cell 0 is the row label')
        raw = cells[column]
        token = re.fullmatch(r'\s*(\(?[+-]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)\)?)(%)?(?:\s*[rp])?(?:\s+\d+/)?\s*', raw)
        if not token: raise ValueError('Selected cell is not a plain numeric observation: '+raw)
        value = token[1].replace(',', '')
        if token[2]: receipt['source_scale']='percent_number; no automatic division by 100'
        if value.startswith('(') and value.endswith(')'): value = '-'+value[1:-1]
        receipt.update(header=header, row=selector['row'], column=column, cell=raw,
            quote=lines[index].rstrip('\n'), start=key[2]+sum(map(len, lines[:index])))
    elif result.get('next_offset') is not None:
        raise ValueError('Numeric result reference must be complete')
    if type(value) not in (str, int, float): raise ValueError('Select one numeric result leaf, not a container')
    number = Decimal(str(value))
    if not number.is_finite(): raise ValueError('Nonfinite acquired operand')
    receipt['value'] = str(number)
    evidence.append(receipt)
    return str(number)


"""Render an explicit answer contract; never infer formatting from gold or task ID."""
import re
from decimal import Decimal, localcontext, ROUND_HALF_UP


def format_answer(value, contract):
    allowed = {'style', 'decimals', 'scale', 'suffix'}
    if set(contract)-allowed or not {'style', 'decimals', 'scale'} <= set(contract):
        raise ValueError('format requires style, decimals, scale; suffix only for unit_suffix')
    style, digits, scale = contract['style'], contract['decimals'], contract['scale']
    if style not in ('number', 'percent', 'unit_suffix') or type(digits) is not int or not 0 <= digits <= 18:
        raise ValueError('style number|percent|unit_suffix and decimals 0..18 required')
    if scale not in ('as_is', 'fraction_to_percent'):
        raise ValueError('Explicit scale as_is|fraction_to_percent required')
    if style != 'unit_suffix' and 'suffix' in contract:
        raise ValueError('Only unit_suffix accepts a suffix; number never appends units')
    suffix = contract.get('suffix', '')
    if style == 'unit_suffix' and (not isinstance(suffix, str) or not suffix.strip() or
            len(suffix) > 60 or '\n' in suffix or '%' in suffix):
        raise ValueError('unit_suffix requires a short literal unit; use percent for %')
    if type(value) not in (str, int, float):
        raise ValueError('A numeric result leaf is required; do not pass prose/reason or an entire object')
    if isinstance(value, str) and not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', value.strip()):
        raise ValueError('Result must be numeric without units, commas or explanation')
    with localcontext() as context:
        context.prec = 60
        number = Decimal(str(value))
        if not number.is_finite(): raise ValueError('nonfinite result')
        if scale == 'fraction_to_percent': number *= 100
        number = number.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
        if number == 0: number = abs(number)
        answer = format(number, f'.{digits}f')
    if style == 'percent': answer += '%'
    if style == 'unit_suffix': answer += ' '+suffix.strip()
    return answer


def read_numeric(ctx, ref):
    if ctx.remaining_calls() <= 0: raise ValueError('No tool budget left to read computed result')
    result = ctx.call('read_result', {**ref, 'offset': 0, 'limit': 256})
    if not result.get('accepted') or result.get('next_offset') is not None:
        raise ValueError('Computed result is unavailable or incomplete; select a numeric leaf')
    return result['data']


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
