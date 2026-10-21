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
