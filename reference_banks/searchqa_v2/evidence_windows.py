"""Return diverse exact evidence spans, including titles; ranking is not entailment."""
import re

STOP = set('a an the of in at to by for from with is are was were be been being on who what when where which how why its his her their did does do name named and or as it this that these those about doc tle par'.split())


def words(text):
    return {w.casefold() for w in re.findall(r'\w+', text) if len(w) > 2 and w.casefold() not in STOP}


def run(ctx, inputs):
    docs, question = inputs.get('documents'), inputs.get('question')
    if not isinstance(docs, list) or not isinstance(question, str) or not words(question):
        return {'status': 'needs_input', 'outputs': {}}
    query, found = words(question), []
    for d in docs:
        if not isinstance(d, dict) or not all(k in d for k in ('doc_index', 'start', 'end', 'text')):
            return {'status': 'blocked', 'outputs': {}}
        text = d['text']
        if not isinstance(text, str) or d['start'] + len(text) != d['end']:
            return {'status': 'blocked', 'outputs': {}}
        # Newlines in snippets are formatting, not sentence boundaries. Use the whole
        # short document; for long text center windows on actual query occurrences.
        starts = {0} if len(text) <= 600 else {
            max(0, min(m.start()-220, len(text)-600)) for m in re.finditer(r'\w+', text)
            if m.group().casefold() in query}
        for start in sorted(starts):
            quote = text[start:start+600]
            shared = words(quote) & query
            if shared:
                found.append({'doc_index': d['doc_index'], 'start': d['start']+start,
                    'end': d['start']+start+len(quote), 'quote': quote,
                    'score': len(shared), 'matched_terms': sorted(shared)})
    ranked = sorted(found, key=lambda r: (-r['score'], r['doc_index'], r['start']))
    chosen, seen = [], set()
    # First pass represents distinct documents; second pass fills unused slots
    # without near-duplicate windows. The complete public input stays available.
    for diverse in (True, False):
        for row in ranked:
            if len(chosen) >= 8:
                break
            if diverse and row['doc_index'] in seen:
                continue
            if any(old['doc_index'] == row['doc_index'] and
                   max(0, min(old['end'], row['end'])-max(old['start'], row['start'])) >
                   0.5*min(old['end']-old['start'], row['end']-row['start']) for old in chosen):
                continue
            chosen.append(row); seen.add(row['doc_index'])
    if not chosen:
        return {'status': 'not_found', 'outputs': {}}
    return {'status': 'ok', 'outputs': {'windows': chosen,
        'covered_terms': sorted({term for row in chosen for term in row['matched_terms']})}}
