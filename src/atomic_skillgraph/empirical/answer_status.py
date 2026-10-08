"""Side classification; it never replaces an official score."""
def classify_answer(answer, finish_reason, *, valid_format=True):
    empty = isinstance(answer, str) and not answer.strip() or answer is None
    truncated = finish_reason == 'length'
    status = 'truncated_empty' if empty and truncated else 'empty' if empty else 'valid' if isinstance(answer, str) and valid_format else 'invalid_format'
    return {'provider_finish_reason': finish_reason, 'answer_status': status,
            'empty_answer': bool(empty), 'completion_truncated': truncated}
