"""Format a caller-selected value and submit it; no question, OCR or answer lookup."""
import re


def run(ctx, inputs):
    kind, value = inputs['kind'], inputs['value']
    answer = value
    if kind == 'phone' and re.fullmatch(r'\+?\d[\d ]*\d', value):
        answer = value.replace(' ', '')
    elif kind == 'page':
        match = re.fullmatch(r'[-–—]\s*(\d+)\s*[-–—]', value)
        if match:
            answer = match.group(1)
    elif kind == 'count':
        match = re.fullmatch(r'[Nn]\s*=\s*(\d+)', value)
        if match:
            answer = match.group(1)
    return {'status': 'ok', 'outputs': {'answer': answer, 'original': value,
        'kind': kind, 'format_applied': answer != value}}
