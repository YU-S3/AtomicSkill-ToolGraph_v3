"""Independent public-input checks for the supplied manual reference Bank.

These are diagnostic specifications, never task answers or runtime guidance.
The source is executed by the existing temporary Python Docker interface before
the imported Program is checked against its actual Host result.
"""

REFERENCES = {
    'scoped_glob': """
paths = ctx.call('glob', {'pattern': inputs['pattern']})['data']
out = dict(paths=paths, count=len(paths), scope_complete=True)
""",
    'find_table_windows': """
files = ctx.call('glob', {'pattern': inputs['path_glob']})['data']
hits = ctx.call('grep', {'pattern': inputs['heading_pattern'], 'paths': files})['data']
out = dict(snippets=[], searched_files=len(files), hit_count=len(hits), complete_scan=True)
for hit in hits:
    start = max(0, hit['offset'] - 200)
    text = ctx.call('read', {'path': hit['path'], 'offset': start})['data']['text']
    out['snippets'].append(dict(path=hit['path'], hit_offset=hit['offset'], read_offset=start, text=text[:2500]))
""",
    'grep_page_windows': """
args = dict(pattern=inputs['pattern'], paths=inputs['paths'], limit=inputs.get('limit', 40))
if inputs.get('cursor') is not None: args['cursor'] = inputs['cursor']
page = ctx.call('grep_page', args)['data']
out = dict(hits=page['hits'], next_cursor=page['next_cursor'], complete=page['complete'], scope='authenticated_page_only')
""",
    'read_hit': """
start = max(0, inputs['offset'] - inputs.get('before', 160))
page = ctx.call('read', dict(path=inputs['path'], offset=start))['data']
out = dict(path=inputs['path'], start_offset=start, anchor_offset=inputs['offset'], text=page['text'], next_offset=page['next_offset'])
""",
    'extract_simple_table': """
rows = [list(map(str.strip, line.strip().strip('|').split('|'))) for line in inputs['text'].splitlines() if '|' in line]
column = rows[0].index(inputs['period'])
row = next(row for row in rows[1:] if row[0] == inputs['metric'])
raw = row[column]
value = float(raw.strip('()').replace(',', '')) * (-1 if raw.startswith('(') else 1)
out = dict(value=value, row_label=row[0], period=inputs['period'], source_path=inputs['source_path'], source_offset=inputs['start_offset'], raw_cell=raw, scope='strict_pipe_table_only')
""",
    'extract_aligned_pair': """
record = sorted(inputs['snippets'], key=lambda r: r['path'])[0]
text = record['text']
unit = next(line.split(':', 1)[1].strip() for line in text.splitlines() if line.startswith('Unit:'))
rows = [list(map(str.strip, line.strip().strip('|').split('|'))) for line in text.splitlines() if '|' in line]
row = next(row for row in rows[1:] if row[0] == inputs['metric'])
values = [float(row[rows[0].index(inputs[p])].replace(',', '')) for p in ('earlier_period', 'later_period')]
offset = record['read_offset'] + text.index(inputs['metric'])
sources = [dict(path=record['path'], offset=offset, period=inputs[p], metric=inputs['metric'], value=value)
           for p, value in zip(('earlier_period', 'later_period'), values)]
out = dict(earlier=values[0], later=values[1], unit=unit, sources=sources, aligned=True,
           evidence_count=len(inputs['snippets']), scope='public_read_pipe_table_only')
""",
    'normalize_unit': """
units = ['USD', 'thousand_USD', 'million_USD', 'billion_USD']
value = inputs['amount'] * (1000 ** units.index(inputs['from_unit'])) / (1000 ** units.index(inputs['to_unit']))
out = dict(value=value, unit=inputs['to_unit'], source_unit=inputs['from_unit'], scope='explicit_USD_scales_only')
""",
    'growth_from_verified_values': """
delta = inputs['later'] - inputs['earlier']
out = dict(delta=delta, percentage_change=delta * 100.0 / inputs['earlier'])
""",
    'stats_from_values': """
from decimal import Decimal
values = list(map(lambda x: Decimal(str(x)), inputs['values']))
total = sum(values)
mean = total / len(values)
op = inputs['operation']
value = {'sum': lambda: total, 'mean': lambda: mean, 'population_variance': lambda: sum((v-mean)**2 for v in values)/len(values)}[op]()
out = dict(value=str(value), n=len(values), operation=op, scope='verified_equal_unit_values_only')
""",
    'growth_final_answer': """
change = (inputs['later'] - inputs['earlier']) * 100 / inputs['earlier']
assert abs(change - inputs['percentage_change']) < 1e-8
out = dict(answer=format(change, '.'+str(inputs['precision'])+'f')+'%', percentage_change=change,
           unit=inputs['unit'], source_refs=inputs['sources'], scope='explicit_source_bound_values_only')
""",
}


def cases(fixture_root, corpus_root):
    """Fixed synthetic contracts plus two real public reports; no evaluator load."""
    import re
    from pathlib import Path
    fixture_root, corpus_root = Path(fixture_root), Path(corpus_root)
    snippets = [dict(path=p.name, read_offset=0, text=p.read_text()) for p in sorted(fixture_root.glob('*.txt'))]
    sources = [dict(path=s['path'], offset=s['text'].index('Receipts')) for s in snippets]
    f = str(fixture_root)
    values = {
        'extract_simple_table': [dict(text=snippets[0]['text'], metric='Receipts', period=p,
                                     source_path=snippets[0]['path'], start_offset=0) for p in ('1951', '1952')],
        'extract_aligned_pair': [dict(snippets=snippets[:n], metric='Receipts', earlier_period='1951',
                                     later_period='1952', complete_scan=True) for n in (1, 2)],
        'normalize_unit': [dict(amount=a, from_unit='million_USD', to_unit='thousand_USD') for a in (2, 7)],
        'growth_from_verified_values': [dict(earlier=a, later=b) for a, b in ((80, 100), (120, 90))],
        'stats_from_values': [dict(values=[2, 4, 6], operation=op) for op in ('sum', 'population_variance')],
        'growth_final_answer': [dict(earlier=80, later=100, unit='million_USD', sources=sources,
                                     precision=p, percentage_change=25) for p in (2, 3)],
    }
    result = [dict(name=name, label='fixture'+str(i), corpus=f, inputs=x, status='ok')
              for name, rows in values.items() for i, x in enumerate(rows)]
    for path in sorted(corpus_root.glob('*.txt'))[:2]:
        text = path.read_text()
        line = next(line for line in text.splitlines() if 10 < len(line) < 120 and text.splitlines().count(line) == 1)
        offset = text.index(line)
        requests = {
            'scoped_glob': dict(pattern=path.name),
            'find_table_windows': dict(path_glob=path.name, heading_pattern='^'+re.escape(line)+'$'),
            'grep_page_windows': dict(paths=[path.name], pattern='^'+re.escape(line)+'$'),
            'read_hit': dict(path=path.name, offset=offset),
        }
        result.extend(dict(name=name, label=path.stem, corpus=str(corpus_root), inputs=x, status='ok') for name, x in requests.items())
    negative = {
        'scoped_glob': (dict(pattern='*.txt', max_paths=1), 'needs_input'),
        'find_table_windows': (dict(path_glob='*.txt', heading_pattern='absent_metric'), 'not_found'),
        'grep_page_windows': (dict(paths=['../outside.txt'], pattern='Receipts'), 'blocked'),
        'read_hit': (dict(path='../outside.txt', offset=0), 'blocked'),
        'extract_simple_table': ({**values['extract_simple_table'][0], 'metric':'Missing'}, 'needs_input'),
        'extract_aligned_pair': ({**values['extract_aligned_pair'][0], 'complete_scan':False}, 'needs_input'),
        'normalize_unit': (dict(amount=1, from_unit='unknown', to_unit='USD'), 'needs_input'),
        'growth_from_verified_values': (dict(earlier=0, later=1), 'needs_input'),
        'stats_from_values': (dict(values=['NaN'], operation='sum'), 'needs_input'),
        'growth_final_answer': ({**values['growth_final_answer'][0], 'percentage_change':99}, 'blocked'),
    }
    result.extend(dict(name=name, label='refusal', corpus=f, inputs=x, status=status) for name, (x, status) in negative.items())
    return result
