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
