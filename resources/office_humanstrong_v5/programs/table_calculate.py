METRIC_RULES = {'difference': {'required': ['operation', 'from_value', 'to_value', 'absolute', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'from_value', 'to_value', 'absolute']}, 'ratio': {'required': ['operation', 'numerator', 'denominator', 'output_percent', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'numerator', 'denominator']}, 'percent_change': {'required': ['operation', 'base_value', 'new_value', 'output_percent', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'base_value', 'new_value', 'periods']}, 'log_return': {'required': ['operation', 'base_value', 'new_value', 'output_percent', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'base_value', 'new_value', 'periods']}, 'cagr': {'required': ['operation', 'base_value', 'new_value', 'output_percent', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'base_value', 'new_value', 'periods']}, 'annual_log_growth': {'required': ['operation', 'base_value', 'new_value', 'output_percent', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'base_value', 'new_value', 'periods']}, 'sum': {'required': ['operation', 'values', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'values', 'other_values']}, 'mean': {'required': ['operation', 'values', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'values', 'other_values']}, 'median': {'required': ['operation', 'values', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'values', 'other_values']}, 'std_population': {'required': ['operation', 'values', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'values', 'other_values']}, 'std_sample': {'required': ['operation', 'values', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'values', 'other_values']}, 'variance_population': {'required': ['operation', 'values', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'values', 'other_values']}, 'variance_sample': {'required': ['operation', 'values', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'values', 'other_values']}, 'gini_population': {'required': ['operation', 'values', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'values', 'other_values']}, 'gini_sample_adjusted': {'required': ['operation', 'values', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'values', 'other_values']}, 'correlation': {'required': ['operation', 'values', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'values', 'other_values']}, 'ols_slope': {'required': ['operation', 'values', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'values', 'other_values']}, 'ols_intercept': {'required': ['operation', 'values', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'values', 'other_values']}, 'geometric_growth': {'required': ['operation', 'values', 'unit'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'values', 'other_values']}, 'symmetric_difference': {'required': ['operation', 'values', 'unit', 'output_percent'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'values']}, 'share_difference': {'required': ['operation', 'values', 'unit', 'output_percent', 'absolute'], 'allowed': ['unit', 'decimals', 'operand_refs', 'output_percent', 'operation', 'values', 'absolute']}}

"""Literal table selection: reject conflicting headers instead of guessing columns."""
import re
from decimal import Decimal


class SelectionError(ValueError):
    def __init__(self, code, detail, **context):
        super().__init__(detail)
        self.recovery = {'code': code, **context}


def norm(value):
    return ' '.join(value.casefold().split())


def label_key(value):
    # A terminal punctuation mark is presentation, not a different month/category.
    # Footnotes, words, years and internal punctuation are not stripped.
    return norm(value).rstrip('.')


def cells(line):
    value = line.strip()
    if '|' in value:
        if value.startswith('|'): value = value[1:]
        if value.endswith('|'): value = value[:-1]
        return [cell.strip() for cell in value.split('|')]
    return [cell.strip() for cell in re.split(r'\t+| {2,}', value)]


def number(value):
    value = re.sub(r'(?<=\d)[*†‡]+$', '', value.strip()).replace(',', '')
    if not re.fullmatch(r'\(?[+-]?(?:\d+(?:\.\d*)?|\.\d+)\)?', value):
        raise ValueError('not a complete numeric cell')
    if value.startswith('('):
        if not value.endswith(')'): raise ValueError('unbalanced numeric parentheses')
        value = '-'+value[1:-1]
    elif value.endswith(')'): raise ValueError('unbalanced numeric parentheses')
    return str(Decimal(value))


def extract_table(ctx, inputs):
    try:
        text, wanted, column = inputs['text'], inputs['rows'], inputs['column']
        if not isinstance(text, str) or not text.strip() or not wanted:
            raise SelectionError('missing_table', 'Supply a complete table window, including headers and units.')
        if len({label_key(row) for row in wanted}) != len(wanted):
            raise SelectionError('duplicate_requested_row', 'Requested rows must be distinct.')
        raw = text.splitlines(keepends=True)
        parsed = [cells(line) for line in raw]
        offsets, cursor = [], 0
        for line in raw: offsets.append(cursor); cursor += len(line)
        separator = lambda row: len(row) > 1 and all(re.fullmatch(r':?-{2,}:?', c) for c in row)
        def numeric_row(row):
            for value in row[1:]:
                try: number(value); return True
                except ValueError: pass
            return False
        def data_row(index):
            # Markdown's delimiter identifies a header even when its labels are years.
            return numeric_row(parsed[index]) and not (
                index+1 < len(parsed) and separator(parsed[index+1]))
        candidates = []
        for index, row in enumerate(parsed):
            if len(row) < 2 or data_row(index) or separator(row): continue
            for col, value in enumerate(row[1:], 1):
                if norm(value) == norm(column): candidates.append((index, col))
        blocks = {}
        for index, col in candidates:
            lo, hi = index, index+1
            while lo and len(parsed[lo-1]) > 1 and not data_row(lo-1): lo -= 1
            while hi < len(raw) and len(parsed[hi]) > 1 and not data_row(hi): hi += 1
            if inputs.get('header_hint') and norm(inputs['header_hint']) not in norm(' '.join(raw[max(0, lo-6):hi])):
                continue
            blocks[(lo, hi, col)] = None
        if len(blocks) > 1 and inputs.get('header_path'):
            selected = {}
            for lo, hi, col in blocks:
                path = []
                for row in parsed[lo:hi]:
                    if not separator(row) and col < len(row) and row[col] and (not path or norm(row[col]) != norm(path[-1])):
                        path.append(row[col])
                if [norm(p) for p in path] == [norm(p) for p in inputs['header_path']]:
                    selected[(lo, hi, col)] = None
            blocks = selected
        if len(blocks) != 1:
            raise SelectionError('ambiguous_header', 'Supply one table with a unique requested column and header_hint.',
                                 column=column, matches=len(blocks))
        (lo, hi, col), = blocks
        headers = [row for row in parsed[lo:hi] if not separator(row)]
        if any(len(row) != len(headers[0]) for row in headers):
            raise SelectionError('header_width_conflict', 'Header widths differ; obtain an unambiguous source.',
                                 headers=raw[lo:hi])
        path = []
        for row in headers:
            if row[col] and (not path or norm(row[col]) != norm(path[-1])): path.append(row[col])
        # A parent heading must never be mistaken for a different child column.
        if not path or norm(path[-1]) != norm(column):
            raise SelectionError('header_conflict', 'Requested heading conflicts with the leaf column; do not guess an index.',
                                 column=column, column_path=path, headers=raw[lo:hi])
        supplied_path = inputs.get('header_path')
        if len(path) > 1 and [norm(p) for p in (supplied_path or [])] != [norm(p) for p in path]:
            raise SelectionError('header_path_required', 'Bind the full observed parent-to-leaf header_path.',
                                 column_path=path, headers=raw[lo:hi])
        unit = inputs['unit']
        if not unit or norm(unit) not in norm(text):
            raise SelectionError('unit_missing', 'Quote the source unit literally; display units are a separate field.',
                                 unit=unit)
        scope = inputs.get('scope_quote')
        if scope and scope not in ''.join(raw[max(0, lo-6):hi]):
            raise SelectionError('scope_missing', 'scope_quote must occur in this table title/header.')
        end = hi
        while end < len(raw) and len(parsed[end]) > 1 and not separator(parsed[end]): end += 1
        matches = []
        for label in wanted:
            found = []
            for index in range(hi, end):
                row = parsed[index]
                if not row or label_key(row[0]) != label_key(label): continue
                if len(row) != len(headers[0]):
                    raise SelectionError('row_width_conflict', 'Data row width differs from its header.', row=label)
                try: value = number(row[col])
                except ValueError:
                    raise SelectionError('nonnumeric_cell', 'Requested cell is not a complete number.', row=label, column=column)
                found.append({'row': label, 'matched_row': row[0], 'column': column, 'value': value, 'unit': unit,
                    'source_line': index+1, 'start': offsets[index], 'end': offsets[index]+len(raw[index].rstrip('\r\n')),
                    'quote': raw[index].rstrip('\r\n'), 'header': ''.join(raw[lo:hi]).rstrip(),
                    'column_path': path, 'scope_quote': scope})
            if len(found) != 1:
                raise SelectionError('row_missing_or_ambiguous', 'Select one literal row; never substitute a broader category.',
                    row=label, matches=len(found), available_rows=[row[0] for row in parsed[hi:end]][:30])
            matches.extend(found)
        return {'status': 'ok', 'outputs': {'cells': matches, 'values': [x['value'] for x in matches],
            'header': ''.join(raw[lo:hi]).rstrip(), 'unit': unit, 'requested_rows': wanted,
            'table_header_bound': True, 'target_periods_complete': True,
            'scope': 'Literal source selector only; caller must match question category and period.',
            'proof_scope': 'literal columns/rows only; caller binds question scope and convention'}}
    except (KeyError, ValueError, TypeError, ArithmeticError) as exc:
        return {'status': 'needs_input', 'outputs': {}, 'detail': str(exc),
                'recovery': getattr(exc, 'recovery', {'code': 'invalid_selector'})}


"""Named formulas with explicit conventions; computation, not operand provenance."""
import math
from decimal import Decimal, localcontext, ROUND_HALF_UP


def calculate_metrics(ctx,inputs):
    try:
        op=inputs['operation'];direction='as_supplied';formula=op
        named={'difference','ratio','percent_change','log_return','cagr','annual_log_growth'}
        if 'output_scale' in inputs:raise ValueError('output_scale is unsupported; specify output_percent explicitly')
        if op in named:
            if 'values' in inputs:raise ValueError('This operation requires named operands, not positional values')
            if op=='difference':
                raw=[inputs['from_value'],inputs['to_value']]
                if type(inputs['absolute']) is not bool:raise ValueError('absolute must be explicit boolean')
                formula='abs(to_value-from_value)' if inputs['absolute'] else 'to_value-from_value'
                direction='absolute' if inputs['absolute'] else 'to_minus_from'
            elif op=='ratio':
                raw=[inputs['denominator'],inputs['numerator']];formula='numerator/denominator';direction='numerator_over_denominator'
            else:
                raw=[inputs['base_value'],inputs['new_value']];direction='base_to_new'
                formula={'percent_change':'(new_value-base_value)/base_value','log_return':'ln(new_value/base_value)',
                         'cagr':'exp(ln(new_value/base_value)/periods)-1',
                         'annual_log_growth':'ln(new_value/base_value)/periods'}[op]
        else:raw=inputs['values']
        if not isinstance(raw,list) or not raw or len(raw)>5000:raise ValueError('1..5000 values required')
        if any(type(v) not in (str,int,float) for v in raw):raise ValueError('numeric strings/numbers required')
        with localcontext() as context:
            context.prec=50
            x=[Decimal(str(v)) for v in raw]
            if any(not v.is_finite() for v in x):raise ValueError('nonfinite operand')
            n=Decimal(len(x));mean=sum(x)/n;convention=op
            if op=='sum':v=sum(x)
            elif op=='mean':v=mean
            elif op=='symmetric_difference':
                if len(x)!=2:raise ValueError('values must be [first,second]')
                v=abs(x[1]-x[0])/((abs(x[0])+abs(x[1]))/2)
                formula='abs(second-first)/((abs(first)+abs(second))/2)';direction='absolute_symmetric'
            elif op=='share_difference':
                if len(x)!=4:raise ValueError('values must be [first_part,first_total,second_part,second_total]')
                if type(inputs.get('absolute')) is not bool:raise ValueError('absolute must be explicit boolean')
                v=x[2]/x[3]-x[0]/x[1]
                formula='second_part/second_total-first_part/first_total';direction='second_share_minus_first_share'
                if inputs['absolute']:v=abs(v);formula='abs('+formula+')';direction='absolute_share_difference'
            elif op=='median':
                a=sorted(x);mid=len(a)//2;v=a[mid] if len(a)%2 else (a[mid-1]+a[mid])/2
            elif op in {'difference','ratio','percent_change','log_return','cagr','annual_log_growth'}:
                if len(x)!=2:raise ValueError('exactly two values in [start,end] order required')
                a,b=x
                if op=='difference':v=abs(b-a) if inputs['absolute'] else b-a
                elif op=='ratio':v=b/a
                elif op=='percent_change':v=(b-a)/a
                else:
                    if a<=0 or b<=0:raise ValueError('growth operands must be positive')
                    ratio=b/a
                    if op=='log_return':v=ratio.ln()
                    else:
                        periods=Decimal(str(inputs.get('periods','0')))
                        if periods<=0:raise ValueError('positive elapsed interval count required, not observation count')
                        v=ratio.ln()/periods if op=='annual_log_growth' else (ratio.ln()/periods).exp()-1
            elif op in {'std_population','std_sample','variance_population','variance_sample'}:
                denominator=n-(1 if op.endswith('sample') else 0)
                if denominator<=0:raise ValueError('sample requires at least two observations')
                v=sum((a-mean)**2 for a in x)/denominator
                if op.startswith('std_'):v=v.sqrt()
            elif op in {'gini_population','gini_sample_adjusted'}:
                if min(x)<0 or sum(x)<=0:raise ValueError('Gini requires nonnegative values and positive sum')
                a=sorted(x);v=sum(Decimal(2*i-len(a)-1)*z for i,z in enumerate(a,1))/(n*sum(a))
                if op=='gini_sample_adjusted':
                    if n<=1:raise ValueError('sample correction requires n>1')
                    v=v*n/(n-1)
            elif op in {'correlation','ols_slope','ols_intercept'}:
                raw_y=inputs.get('other_values',[])
                if len(raw_y)!=len(x) or len(x)<2:raise ValueError('paired x/y arrays of equal length required')
                y=[Decimal(str(z)) for z in raw_y]
                if any(not z.is_finite() for z in y):raise ValueError('nonfinite y')
                my=sum(y)/n;sxx=sum((a-mean)**2 for a in x);sxy=sum((a-mean)*(b-my) for a,b in zip(x,y))
                if sxx==0:raise ValueError('zero x variance')
                slope=sxy/sxx
                if op=='ols_slope':v=slope
                elif op=='ols_intercept':v=my-slope*mean
                else:
                    syy=sum((b-my)**2 for b in y)
                    if syy==0:raise ValueError('zero y variance')
                    v=sxy/(sxx*syy).sqrt()
            elif op=='geometric_growth':
                if any(1+a<=0 for a in x):raise ValueError('growth factors must be positive')
                v=(sum((1+a).ln() for a in x)/n).exp()-1
            else:raise ValueError('unsupported named operation')
            percent=inputs.get('output_percent',False)
            if type(percent) is not bool:raise ValueError('output_percent must be boolean')
            if percent:
                if op not in {'ratio','percent_change','cagr','geometric_growth','symmetric_difference','share_difference'}:raise ValueError('percent output is not supported for this operation')
                v*=100;formula='100*('+formula+')'
            digits=inputs.get('decimals')
            if digits is not None:
                if type(digits) is not int or not 0<=digits<=18:raise ValueError('decimals must be 0..18')
                answer=format(v.quantize(Decimal(1).scaleb(-digits),rounding=ROUND_HALF_UP),'.'+str(digits)+'f')
            else:answer=format(v,'f').rstrip('0').rstrip('.') if '.' in format(v,'f') else format(v,'f')
            if answer in ('-0',''):answer='0'
            return {'status':'ok','outputs':{'value':str(v),'answer':answer,'formula':formula,'direction':direction,
                'operand_refs':inputs.get('operand_refs',[]),'periods':str(inputs.get('periods','not_applicable')),
                'unit':inputs['unit'],'output_percent':percent,'proof_scope':'local_computation_only',
                'convention':convention,'count':len(x)}}
    except (KeyError,ValueError,TypeError,ArithmeticError) as e:return {'status':'needs_input','outputs':{},'detail':str(e)}


"""Polynomial OLS over explicit public observations; no implicit imputation.

Inputs are public, explicitly aligned observations. This computes a polynomial
OLS forecast with an intercept; it cannot certify source/column semantics.
"""
from decimal import Decimal, localcontext, ROUND_HALF_UP


def calculate_polynomial(ctx, inputs):
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


"""Bound real table references to a named calculation and a final answer.

Uses safe literal extraction, named arithmetic and an explicit output contract.
"""


def source_window(ctx, table):
    ref = table['source_ref']
    start = table.get('offset', 0)
    length = table.get('length', 12000)
    if type(start) is not int or start < 0 or type(length) is not int or not 1 <= length <= 24000:
        raise ValueError('source window requires offset>=0 and length 1..24000')
    # Read one preceding character to identify a partial first line.
    cursor = max(0, start-1)
    end = start+length
    chunks = []
    last = None
    while cursor < end:
        if ctx.remaining_calls() <= 0:
            raise ValueError('tool budget exhausted before requested source window was read')
        last = ctx.call('read_result', {**ref, 'offset': cursor, 'limit': min(3000, end-cursor)})
        if not last.get('accepted'):
            raise ValueError('source read refused: '+str(last.get('error')))
        if not isinstance(last.get('data'), str) or last.get('unit') != 'unicode_codepoint_0_based':
            raise ValueError('source_ref must identify acquired public text')
        if cursor >= last.get('size', cursor+1):
            raise SelectionError('offset_out_of_bounds', 'Use a window inside the acquired source, not a guessed document offset.',
                                 requested_offset=start, source_size=last.get('size'), source_ref=ref)
        if last['offset'] != cursor or last['end'] < cursor:
            raise SelectionError('source_offset_conflict', 'Source read offsets do not match the request.', source_ref=ref)
        chunks.append(last['data'])
        if last['next_offset'] is None or last['end'] == cursor:
            break
        cursor = last['end']
    text = ''.join(chunks)
    base = max(0, start-1)
    if start:
        if not text:
            raise ValueError('source window is empty')
        previous, text = text[0], text[1:]
        base = start
        if previous != '\n':
            newline = text.find('\n')
            if newline < 0:
                raise ValueError('no complete source line in requested window')
            base += newline+1
            text = text[newline+1:]
    if last and last['next_offset'] is not None and not text.endswith('\n'):
        # A partial numeric row must not be parsed as a smaller complete value.
        text = text[:text.rfind('\n')+1] if '\n' in text else ''
    if not text.strip():
        raise ValueError('no complete source lines in requested window')
    return text, base


def run(ctx, inputs):
    try:
        if inputs.get('submit_answer') is not True:
            raise ValueError('Choose this final-answer Program only for the complete requested calculation')
        contract = inputs['format']
        format_answer('0', contract)  # Reject invalid display contracts before any tool calls.
        tables = inputs['tables']
        if not 1 <= len(tables) <= 8:
            raise ValueError('1..8 table selectors required')
        groups, evidence, units, windows = {}, [], set(), {}
        for table in tables:
            key = table['key']
            if key in groups:
                raise ValueError('table keys must be distinct')
            ref = table['source_ref']
            window_key = (ref['result_id'], tuple(ref['path']), table.get('offset', 0), table.get('length', 12000))
            if window_key not in windows:
                windows[window_key] = source_window(ctx, table)
            text, base = windows[window_key]
            extracted = extract_table(ctx, {'text': text, 'rows': table['rows'],
                'column': table['column'], 'unit': table['unit'],
                'header_hint': table.get('header_hint', ''),
                'header_path': table.get('header_path'), 'scope_quote': table.get('scope_quote')})
            if extracted['status'] != 'ok':
                recovery = extracted.get('recovery', {})
                raise SelectionError(recovery.get('code', 'invalid_selector'),
                    key+': '+extracted.get('detail', 'ambiguous table'),
                    table_key=key, selector_feedback=recovery)
            cells = extracted['outputs']['cells']
            for cell in cells:
                cell.update(source_ref=table['source_ref'], start=base+cell['start'],
                            end=base+cell['end'], table_key=key)
            groups[key] = cells
            evidence.extend(cells)
            units.add(table['unit'])
        if len(units) != 1:
            raise ValueError('mixed source units: normalize explicitly before using this Program')
        bindings = inputs['bindings']
        if not bindings or set(k for names in bindings.values() for k in names) != set(groups):
            raise ValueError('bindings must use every extracted table key and no unknown keys')
        selected = {}
        for field, names in bindings.items():
            if not names or len(set(names)) != len(names):
                raise ValueError('each binding needs distinct ordered table keys')
            selected[field] = [cell for key in names for cell in groups[key]]
        options = dict(inputs['calculation'])
        method = options.pop('method')
        source_unit = next(iter(units))
        # Actual source unit and display suffix are independent. No conversion.
        options['unit'] = source_unit
        if method == 'statistics':
            op = options['operation']
            expected = {'operation', 'transform', 'unit'} | (
                {'ddof', 'observation'} if op == 'z_score' else
                {'ddof'} if op == 'coefficient_variation' else set())
            fields = {'values', 'other_values'} if op == 'mean_spread' else {'values'}
            if set(options) != expected or set(selected) != fields:
                raise ValueError('Statistics requires exact operands, transform and explicit ddof/observation where applicable')
            if op == 'mean_spread':
                labels = [[label_key(row['matched_row']) for row in selected[field]] for field in ('values', 'other_values')]
                if labels[0] != labels[1]: raise ValueError('mean_spread rows must have identical period order')
            else:
                options['unit'] = 'dimensionless'
            options.update({field: [row['value'] for row in rows] for field, rows in selected.items()})
            computed = calculate_statistics(ctx, options)
        elif method == 'series_metrics':
            options['decimals'] = None
            if set(options) - {'operation','decimals','unit','absolute','output_percent','periods'}:
                raise ValueError('metric calculation contains polynomial-only options')
            op = options['operation']
            absolute_relative = op == 'absolute_relative_change'
            if absolute_relative:
                options['operation'] = 'percent_change'
            if op in {'variance_population','variance_sample'}: options['unit'] = source_unit+'^2'
            elif op in {'ratio','percent_change','absolute_relative_change','log_return','cagr',
                        'annual_log_growth','geometric_growth','symmetric_difference','share_difference',
                        'correlation','ols_slope','gini_population','gini_sample_adjusted'}:
                options['unit'] = ('percentage points' if op == 'share_difference' else 'percent') if options.get('output_percent') else 'dimensionless'
            scalar = {'from_value','to_value','base_value','new_value','numerator','denominator'}
            if set(selected) - scalar - {'values','other_values'}:
                raise ValueError('unknown metric operand binding')
            for field, rows in selected.items():
                if field in scalar and len(rows) != 1:
                    raise ValueError(field+' requires exactly one extracted cell')
                options[field] = rows[0]['value'] if field in scalar else [r['value'] for r in rows]
            options['operand_refs'] = [r['source_ref'] for r in evidence]
            rule = METRIC_RULES.get(options['operation'])
            if not rule or set(options)-set(rule['allowed']) or set(rule['required'])-set(options):
                raise ValueError('calculation fields do not match the exact named metric contract')
            computed = calculate_metrics(ctx, options)
            if absolute_relative and computed['status'] == 'ok':
                result = computed['outputs']
                with localcontext() as precision:
                    precision.prec = 50
                    value = abs(Decimal(result['value']))
                    result.update(value=str(value), answer=str(value),
                        formula='abs('+result['formula']+')',direction='absolute_base_relative',
                        convention='absolute_relative_change')
        elif method == 'polynomial_forecast':
            options['decimals'] = 18
            if set(options) - {'operation','degree','decimals','unit','first_x','x_step','prediction_x'}:
                raise ValueError('polynomial calculation contains unsupported options')
            if set(selected) - {'observations','actual'} or 'observations' not in selected:
                raise ValueError('polynomial requires observations and optional actual bindings')
            rows = selected['observations']
            first = Decimal(str(options.pop('first_x', 0)))
            step = Decimal(str(options.pop('x_step', 1)))
            if not first.is_finite() or not step.is_finite() or step == 0:
                raise ValueError('finite first_x and nonzero x_step required')
            options['required_periods'] = [r['row'] for r in rows]
            options['observations'] = [{'period': r['row'], 'x': str(first+i*step),
                'value': r['value'], 'unit': source_unit, 'source_ref': r['source_ref']}
                for i, r in enumerate(rows)]
            if 'actual' in selected:
                if len(selected['actual']) != 1:
                    raise ValueError('actual binding requires exactly one observed target value')
                r = selected['actual'][0]
                options['actual'] = {'value': r['value'], 'unit': source_unit,
                                     'source_ref': r['source_ref']}
            computed = calculate_polynomial(ctx, options)
        else:
            raise ValueError('unknown calculation method')
        if computed['status'] != 'ok':
            raise ValueError(computed.get('detail', 'calculation needs input'))
        result = computed['outputs']
        result['source_unit'] = source_unit
        if result.get('output_percent') and contract['scale'] != 'as_is':
            raise ValueError('Calculation already multiplied by 100; format scale must be as_is')
        scalable = (method == 'statistics' and options['operation'] == 'coefficient_variation' or
                    method == 'series_metrics' and op in {'ratio', 'percent_change', 'absolute_relative_change',
                        'cagr', 'geometric_growth', 'symmetric_difference', 'share_difference'})
        if contract['scale'] == 'fraction_to_percent' and not scalable:
            raise ValueError('This operation retains its scale; use format scale=as_is')
        value = result.get('value', result.get('answer'))
        if method == 'polynomial_forecast':
            value = result['prediction'] if options['operation'] == 'predict' else result['absolute_difference']
        return {'status': 'ok', 'outputs': {'answer': format_answer(value, contract),
            'calculation': result, 'evidence': evidence, 'format': contract,
            'proof_scope': 'literal source extraction and named computation; caller selects question semantics'}}
    except (KeyError, ValueError, TypeError, ArithmeticError) as exc:
        return {'status': 'needs_input', 'outputs': {}, 'detail': str(exc),
            'recovery': getattr(exc, 'recovery', {'code': 'calculation_contract'}),
            'next_action': 'Change the indicated selector/contract using observed evidence. If ambiguous, use an explicit source-bound computation; do not repeat guesses.'}
