"""Strict, idempotent normalization of the pinned LiveMathematician MCQ."""
NORMALIZATION_VERSION = 'livemath.cf4.v1'
UPSTREAM_REVISION = 'SkillOpt@fa4ca184573e42ec11472959dd57422381418096'
LABELS = 'ABCDEFG'


def normalize_label(value):
    return str(value).strip().upper().rstrip('.):')


def normalize_livemath_item(raw_item: dict) -> dict:
    item_id = raw_item.get('id', str(raw_item.get('month', '')) + ':' + str(raw_item.get('no', '')))
    def invalid(reason):
        raise ValueError(f'LiveMath {item_id}: {reason}')
    mcq = raw_item.get('mcq', raw_item)
    if not isinstance(mcq, dict): invalid('mcq must be an object')
    question = mcq.get('question')
    if not isinstance(question, str) or not question.strip(): invalid('empty question')
    source = mcq.get('choices')
    if isinstance(source, dict):
        source = [{'label': k, 'text': v} for k, v in sorted(source.items())]
    if not isinstance(source, list) or not source: invalid('unparseable choices')
    choices, seen = [], set()
    for index, choice in enumerate(source):
        if isinstance(choice, str):
            if index >= len(LABELS): invalid('unlabeled choice exceeds A-G')
            label, text = LABELS[index], choice
        elif isinstance(choice, dict):
            label = choice.get('label', LABELS[index] if index < len(LABELS) else '')
            text = choice.get('text', choice.get('content'))
        else: invalid('unparseable choice')
        label = normalize_label(label)
        if not label or not isinstance(text, str) or not text.strip(): invalid('choice label/text missing')
        if label in seen: invalid('duplicate or conflicting label ' + label)
        seen.add(label)
        choices.append({'label': label, 'text': text.strip()})
    correct = mcq.get('correct_choice')
    label = normalize_label(correct.get('label', '') if isinstance(correct, dict) else correct or '')
    text = correct.get('text', correct.get('content')) if isinstance(correct, dict) else None
    if not label: invalid('correct label missing')
    matching = next((c for c in choices if c['label'] == label), None)
    if text is not None and (not isinstance(text, str) or not text.strip()): invalid('correct text missing')
    text = text.strip() if text is not None else None
    if matching:
        if text is not None and text != matching['text']: invalid('conflicting correct text')
        text = matching['text']
    else:
        if text is None: invalid('missing correct choice has no original text')
        choices.append({'label': label, 'text': text})
        choices.sort(key=lambda c: LABELS.index(c['label']) if c['label'] in LABELS else len(LABELS))
    return {'id': item_id, 'question': question.strip(), 'choices': choices,
            'correct_choice': {'label': label, 'text': text}}
