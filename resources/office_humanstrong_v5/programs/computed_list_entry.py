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
