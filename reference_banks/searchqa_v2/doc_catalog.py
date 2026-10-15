"""Preserve public DOC blocks/offsets; parse TLE separately from PAR text."""
import re


def run(ctx, inputs):
    context = inputs.get('context', '')
    if not isinstance(context, str) or not context.strip():
        return {'status': 'needs_input', 'outputs': {}}
    markers = list(re.finditer(re.escape('[DOC]'), context))
    bounds = [(m.end(), markers[i+1].start() if i+1 < len(markers) else len(context))
              for i, m in enumerate(markers)] if markers else [(0, len(context))]
    docs = []
    for i, (a, b) in enumerate(bounds, 1):
        raw = context[a:b]
        if not raw.strip():
            continue
        title = re.search(r'\[TLE\](.*?)(?:\[PAR\]|$)', raw, re.S)
        title = title.group(1).strip() if title else next(line.strip() for line in raw.splitlines() if line.strip())
        docs.append({'doc_index': i, 'start': a, 'end': b, 'title': title, 'text': raw})
    return {'status': 'ok' if docs else 'not_found', 'outputs': {'documents': docs, 'count': len(docs)} if docs else {}}
