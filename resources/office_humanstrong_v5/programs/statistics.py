"""Explicit statistical conventions, with unrounded operands and intermediates."""


def calculate_statistics(ctx, inputs):
    try:
        op = inputs['operation']
        values = [Decimal(str(value)) for value in inputs['values']]
        if not values or any(not value.is_finite() for value in values):
            raise ValueError('Finite nonempty series required')
        transform = inputs['transform']
        if transform not in ('none', 'first_difference'): raise ValueError('Unknown series transform')
        if transform == 'first_difference':
            values = [b-a for a, b in zip(values, values[1:])]
        n = len(values)
        with localcontext() as context:
            context.prec = 50
            if not n: raise ValueError('No observations after transform')
            mean = sum(values)/n
            if op == 'mean_spread':
                if transform != 'none': raise ValueError('mean_spread expects two original aligned series')
                other = [Decimal(str(value)) for value in inputs['other_values']]
                if len(other) != n or any(not value.is_finite() for value in other):
                    raise ValueError('Two finite period-aligned series of equal length required')
                value = sum(a-b for a, b in zip(values, other))/n
                formula = 'mean(values - other_values)'
            else:
                ddof = inputs['ddof']
                if type(ddof) is not int or ddof not in (0, 1) or n <= ddof:
                    raise ValueError('Explicit ddof=0 population or ddof=1 sample and enough observations required')
                std = (sum((v-mean)**2 for v in values)/(n-ddof)).sqrt()
                if op == 'coefficient_variation':
                    if mean == 0: raise ValueError('CV requires nonzero mean')
                    value = std/mean; formula = 'std(values, ddof)/mean(values)'
                elif op == 'z_score':
                    observed = Decimal(str(inputs['observation']))
                    if not observed.is_finite() or std == 0: raise ValueError('Finite observation and nonzero std required')
                    value = (observed-mean)/std; formula = '(observation-mean(values))/std(values, ddof)'
                else: raise ValueError('Unknown statistic')
            return {'status': 'ok', 'outputs': {'value': str(value), 'formula': formula, 'count': n,
                'transform': transform, 'ddof': inputs.get('ddof'), 'unit': inputs['unit'],
                'proof_scope': 'named arithmetic; caller selects statistic/convention and public observation'}}
    except (KeyError, ValueError, TypeError, ArithmeticError) as exc:
        return {'status': 'needs_input', 'outputs': {}, 'detail': str(exc)}
