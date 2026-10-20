"""Final scalar submission from an already acquired numeric result."""


def run(ctx, inputs):
    try:
        if inputs['submit_answer'] is not True: raise ValueError('Explicit final submission required')
        ref, contract = inputs['value_ref'], inputs['format']
        value = read_numeric(ctx, ref)
        return {'status': 'ok', 'outputs': {'answer': format_answer(value, contract),
            'evidence': [{'source_ref': ref, 'value': value, 'format': contract}],
            'proof_scope': 'formatting an acquired numeric result, not verifying its question semantics'}}
    except (KeyError, ValueError, TypeError, ArithmeticError) as exc:
        return {'status': 'needs_input', 'outputs': {}, 'detail': str(exc)}
