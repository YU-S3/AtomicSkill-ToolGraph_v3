"""Opt-in final-answer transport; never extract answers from explanations."""
import re

from ..agents.protocol import NativeToolSpec

VERSION = 'atomic.scalar-or-tool.v1'
LIST_VERSION = 'atomic.scalar-list-or-tool.v2'
SCHEMA = {'type': 'object', 'properties': {'answer': {'type': 'string', 'minLength': 1}},
          'required': ['answer'], 'additionalProperties': False}
INSTRUCTION = ('Submit the final answer with finish_answer(answer). This is a submission interface, '
    'not an environment tool: it cannot retrieve or compute. Put only the requested answer in answer, '
    'without explanation or intermediate values. A complete bare numeric answer is also accepted unchanged.')
SCALAR = re.compile(r'[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?%?')


def enabled(settings):
    return settings.get('final_submission_version') in (VERSION, LIST_VERSION)


def instruction(settings):
    if settings.get('final_submission_version') == LIST_VERSION:
        return INSTRUCTION.replace('bare numeric answer', 'bare numeric answer or flat bracketed answer list')
    return INSTRUCTION


def tools(settings):
    return [NativeToolSpec('finish_answer', 'Submit only the final answer', SCHEMA)] if enabled(settings) else []


def prompt(text, settings):
    # The legacy finalizer prohibits every tool; the new interface only submits.
    if not enabled(settings): return text
    text = text.replace('Do not call tools, execute programs, read more results or replan.',
                        'Do not execute programs, read more results or replan.')
    guide = instruction(settings)
    return text if guide in text else text + '\n' + guide


def scalar_answer(turn):
    if turn.tool_calls or turn.finish_reason != 'stop' or not isinstance(turn.content, str): return None
    value = turn.content.strip()
    return value if len(value) <= 128 and SCALAR.fullmatch(value) else None


def plain_answer(turn, settings):
    """Accept a whole final value, without extracting or reformatting its contents."""
    scalar = scalar_answer(turn)
    if scalar is not None or settings.get('final_submission_version') != LIST_VERSION: return scalar
    if turn.tool_calls or turn.finish_reason != 'stop' or not isinstance(turn.content, str): return None
    value = turn.content.strip()
    if len(value) > 512 or not value.startswith('[') or not value.endswith(']'): return None
    items = [item.strip() for item in value[1:-1].split(',')]
    # Flat numeric fields and short categorical labels, e.g. [z, unusual].
    # No prose outside the list, nested containers, code, or truncated outputs.
    if not 1 <= len(items) <= 16: return None
    if any(item.lower() in ('nan', 'inf', 'infinity') for item in items): return None
    if all(SCALAR.fullmatch(item) or re.fullmatch(r'[A-Za-z][A-Za-z _-]{0,31}', item) for item in items):
        return value
    return None
