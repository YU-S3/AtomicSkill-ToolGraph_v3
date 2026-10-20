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


"""Ordered final list, independently formatted fields and no model rewrite."""


def run(ctx, inputs):
    try:
        if inputs['submit_answer'] is not True: raise ValueError('Explicit final submission required')
        items = inputs['items']
        if not 1 <= len(items) <= 32: raise ValueError('Ordered list requires 1..32 items')
        values, evidence = [], []
        for item in items:
            value = read_numeric(ctx, item['value_ref'])
            values.append(format_answer(value, item['format']))
            evidence.append({'source_ref': item['value_ref'], 'value': value, 'format': item['format']})
        layout = inputs['layout']
        if layout not in ('brackets', 'comma', 'lines'): raise ValueError('Explicit list layout required')
        answer = ('\n' if layout == 'lines' else ', ').join(values)
        if layout == 'brackets': answer = '['+answer+']'
        return {'status': 'ok', 'outputs': {'answer': answer, 'evidence': evidence,
            'proof_scope': 'ordered formatting of acquired numeric results; caller selects requested order'}}
    except (KeyError, ValueError, TypeError, ArithmeticError) as exc:
        return {'status': 'needs_input', 'outputs': {}, 'detail': str(exc)}
