"""Opt-in final-answer transport; never extract answers from explanations."""
import re

from ..agents.protocol import NativeToolSpec

VERSION = 'atomic.scalar-or-tool.v1'
SCHEMA = {'type': 'object', 'properties': {'answer': {'type': 'string', 'minLength': 1}},
          'required': ['answer'], 'additionalProperties': False}
INSTRUCTION = ('Submit the final answer with finish_answer(answer). This is a submission interface, '
    'not an environment tool: it cannot retrieve or compute. Put only the requested answer in answer, '
    'without explanation or intermediate values. A complete bare numeric answer is also accepted unchanged.')
SCALAR = re.compile(r'[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?%?')


def enabled(settings):
    return settings.get('final_submission_version') == VERSION


def tools(settings):
    return [NativeToolSpec('finish_answer', 'Submit only the final answer', SCHEMA)] if enabled(settings) else []


def prompt(text, settings):
    # The legacy finalizer prohibits every tool; the new interface only submits.
    if not enabled(settings): return text
    text = text.replace('Do not call tools, execute programs, read more results or replan.',
                        'Do not execute programs, read more results or replan.')
    return text if INSTRUCTION in text else text + '\n' + INSTRUCTION


def scalar_answer(turn):
    if turn.tool_calls or turn.finish_reason != 'stop' or not isinstance(turn.content, str): return None
    value = turn.content.strip()
    return value if len(value) <= 128 and SCALAR.fullmatch(value) else None
