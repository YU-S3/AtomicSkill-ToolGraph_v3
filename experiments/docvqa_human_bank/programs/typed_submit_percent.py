"""Explicit value formatting; no question, gold, OCR, arithmetic or unit conversion."""
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
    elif kind == 'percentage' and value.count('%') == 1:
        answer = re.sub(r'(?<=\d)[ \t\u00a0]+%', '%', value)
    return {'status': 'ok', 'outputs': {'answer': answer, 'original': value,
        'kind': kind, 'format_applied': answer != value}}
