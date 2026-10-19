"""Polynomial OLS over explicit public observations; no implicit imputation.

Inputs are public, explicitly aligned observations. This computes a polynomial
OLS forecast with an intercept; it cannot certify source/column semantics.
"""
from decimal import Decimal, localcontext, ROUND_HALF_UP


def run(ctx, inputs):
    try:
        degree = inputs['degree']
        if type(degree) is not int or not 1 <= degree <= 5:
            raise ValueError('degree must be 1..5')
        rows = inputs['observations']
        expected = inputs['required_periods']
        if not degree < len(expected) <= 240 or len(set(expected)) != len(expected):
            raise ValueError('distinct required periods and at least degree+1 observations required')
        if len(rows) != len(expected) or [r['period'] for r in rows] != expected:
            raise ValueError('observations must cover every required period once, in specified order')
        if any(r['unit'] != inputs['unit'] or not r.get('source_ref') for r in rows):
            raise ValueError('consistent units and source references required')
        operation = inputs['operation']
        if operation not in ('predict', 'absolute_error'):
            raise ValueError('choose predict or absolute_error')
        actual = inputs.get('actual')
        if operation == 'absolute_error' and actual is None:
            raise ValueError('absolute_error requires an independently acquired actual value')
        if actual is not None and (actual['unit'] != inputs['unit'] or not actual.get('source_ref')):
            raise ValueError('actual value needs matching units and a source reference')
        digits = inputs['decimals']
        if type(digits) is not int or not 0 <= digits <= 18:
            raise ValueError('decimals must be 0..18')
        with localcontext() as context:
            context.prec = 60
            x = [Decimal(str(r['x'])) for r in rows]
            y = [Decimal(str(r['value'])) for r in rows]
            target = Decimal(str(inputs['prediction_x']))
            observed = Decimal(str(actual['value'])) if actual is not None else None
            if not all(v.is_finite() for v in x+y+[target]+([] if observed is None else [observed])):
                raise ValueError('finite numeric values required')
            if len(set(x)) < degree+1:
                raise ValueError('rank deficient design')
            center = sum(x)/len(x)
            scale = max(abs(v-center) for v in x)
            z = [(v-center)/scale for v in x]
            design = [[Decimal(1)]+[v**j for j in range(1, degree+1)] for v in z]
            width = degree+1
            matrix = [[sum(r[i]*r[j] for r in design) for j in range(width)]
                      +[sum(r[i]*v for r, v in zip(design, y))] for i in range(width)]
            for i in range(width):
                pivot = max(range(i, width), key=lambda j: abs(matrix[j][i]))
                matrix[i], matrix[pivot] = matrix[pivot], matrix[i]
                divisor = matrix[i][i]
                if divisor == 0:
                    raise ValueError('rank deficient design')
                matrix[i] = [v/divisor for v in matrix[i]]
                for j in range(width):
                    if j != i:
                        factor = matrix[j][i]
                        matrix[j] = [a-factor*b for a, b in zip(matrix[j], matrix[i])]
            coefficients = [r[-1] for r in matrix]
            target_z = (target-center)/scale
            prediction = coefficients[-1]
            for c in reversed(coefficients[:-1]):
                prediction = prediction*target_z+c
            error = abs(prediction-observed) if observed is not None else None
            value = prediction if operation == 'predict' else error
            answer = format(value.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP), f'.{digits}f')
            return {'status': 'ok', 'outputs': {
                'count': len(rows), 'degree': degree, 'unit': inputs['unit'],
                'center': str(center), 'scale': str(scale),
                'coefficient_basis': 'ascending powers of (x-center)/scale, including intercept',
                'coefficients': [str(v) for v in coefficients],
                'prediction': str(prediction), 'actual': str(observed) if observed is not None else None,
                'absolute_difference': str(error) if error is not None else None, 'answer': answer,
                'formula': 'unweighted polynomial OLS with intercept; '+('prediction' if operation == 'predict' else 'abs(prediction-actual)'),
                'proof_scope': 'calculation and explicit coverage only; source semantics not certified'}}
    except (KeyError, ValueError, TypeError, ArithmeticError) as exc:
        return {'status': 'needs_input', 'outputs': {}, 'detail': str(exc)}
