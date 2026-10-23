"""Pure answer-surface normalization; no corpus, task IDs, facts or gold lookup."""
import html
import re

ONES = dict(zip('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split(), range(20)))
TENS = dict(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(), range(20, 100, 10)))


def small_number(text):
    parts = text.lower().replace('-', ' ').split()
    if len(parts) == 1:
        return ONES.get(parts[0], TENS.get(parts[0]))
    if len(parts) == 2 and parts[0] in TENS and parts[1] in ONES and 0 < ONES[parts[1]] < 10:
        return TENS[parts[0]] + ONES[parts[1]]
    return None


def normalize(answer, question):
    changes = []
    body = answer.strip()
    match = re.fullmatch(r'<answer>(.*?)</answer>', body, re.S)
    if match:
        body = match.group(1).strip()
    if not body or re.search(r'</?answer\b', body, re.I):
        raise ValueError('Exactly one nonempty answer is required')
    decoded = html.unescape(body)
    if decoded != body:
        changes.append('html_entities')
        body = decoded
    # Typography only: preserve accent letters and all lexical words.
    plain = body.translate(str.maketrans({'‘': "'", '’': "'", '“': '"', '”': '"', '–': '-', '—': '-'}))
    if plain != body:
        changes.append('typographic_punctuation')
        body = plain
    quantity = bool(re.search(r'\b(?:how many|how much|this many|this number|number of|this age|what age|how old|what percent|what percentage|this percent|this percentage)\b', question, re.I))
    if quantity:
        numeric = small_number(body)
        if numeric is not None:
            body = str(numeric)
            changes.append('small_quantity_digits')
        suffix = None
        if re.search(r'\b(?:age|old)\b', question, re.I):
            suffix = r'years? old'
        elif re.search(r'\b(?:percent|percentage)\b', question, re.I):
            suffix = r'(?:percent|per cent|percentage points|%)'
        if suffix:
            match = re.fullmatch(r'([+-]?\d+(?:\.\d+)?)\s*' + suffix, body, re.I)
            if match:
                body = match.group(1)
                changes.append('unit_already_in_question')
    if not body or re.search(r'</?answer\b', body, re.I):
        raise ValueError('Decoded nested/empty answer is not allowed')
    return body, changes


def run(ctx, inputs):
    answer, question = inputs.get('answer'), inputs.get('question')
    if not isinstance(answer, str) or not isinstance(question, str):
        return {'status': 'needs_input', 'outputs': {}, 'error': 'answer and question strings required'}
    try:
        value, changes = normalize(answer, question)
    except ValueError as error:
        return {'status': 'needs_input', 'outputs': {}, 'error': str(error)}
    return {'status': 'ok', 'outputs': {'answer': '<answer>' + value + '</answer>',
        'surface_audit': {'raw_answer': answer, 'normalized_answer': value, 'changes': changes,
                          'semantic_validation': False}}}
