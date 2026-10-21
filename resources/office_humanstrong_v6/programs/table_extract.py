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


def run(ctx, inputs):
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
            row = parsed[index]
            if index+1 < len(parsed) and separator(parsed[index+1]): return False
            # Empty leading cells identify continuation headers, including year labels.
            if not row[0] and any(row[1:]): return False
            # Section labels and missing-value data rows end a header block too.
            missing = {'', '-', '–', '—', '...', '…', 'n/a', 'na'}
            return numeric_row(row) or (
                bool(row[0]) and all(norm(value) in missing for value in row[1:]))
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
