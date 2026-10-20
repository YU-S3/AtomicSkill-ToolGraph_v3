"""Versioned public input, shared by Runtime and isolated source trials."""
from copy import deepcopy

from .contracts import digest
from .public_results import TEXT_CONTEXT

VERSION = 'office.external-minimax.public-pages.v1'
ANSWER_CONTRACT = ('Return only the requested answer. For a numeric result, output the '
    'number at the requested scale and precision; add a unit or percent-sign suffix '
    'only if an explicit answer template requires it. For dates or lists follow '
    'the requested form. Do not add explanation.')


def answer_prompt(prompt):
    return prompt if prompt.startswith(ANSWER_CONTRACT) else ANSWER_CONTRACT + '\n' + prompt


def enabled(config):
    return config.get('harness', {}).get('office_public_input_version') == VERSION


def seed(task, context, *, recorded=None):
    if task.inputs.get('office_public_input_version') != VERSION:
        raise ValueError('Frozen public page input version differs')
    pages = task.inputs['source_pages']
    # Trials reconstruct only the original supplied pages before replaying tools.
    # They must not import the future outputs being tested.
    if recorded is not None:
        supplied = {rid: value for rid, value in recorded.items()
                    if isinstance(value, dict) and value.get('provenance', {}).get('kind') == 'provided_public_input'}
        if len(supplied) != len(pages):
            raise ValueError('Source trial lacks its exact supplied-page snapshot')
        if supplied:
            scopes = {rid.rsplit(':', 1)[0] for rid in supplied}
            if len(scopes) != 1:
                raise ValueError('Supplied page scopes differ')
            context.scope = scopes.pop()
    refs = []
    for i, page in enumerate(pages):
        if digest(page['text']) != page['text_digest']:
            raise ValueError('Provided page content differs from its frozen identity')
        raw = {'accepted': True, 'data': {'text': page['text'], 'next_offset': len(page['text'])},
               'provenance': {'kind': 'provided_public_input', 'source_url': page['source_url'],
                              'document': page['document'], 'page': page['page']}}
        rid = context.register('public_input:' + digest([task.physical_key, i, page]), raw,
            name='provided_source_page', arguments={'path': page['document'] + '#page=' + page['page']},
            origin=TEXT_CONTEXT, verify_source=True)
        if recorded is not None and supplied.get(rid) != raw:
            raise ValueError('Source trial public page identity changed')
        refs.append({'result_id': rid, 'path': ['data', 'text']})
    return refs


def runtime_material(system, material):
    refs = seed(system.adapter.task, system.task_context)
    value = deepcopy(material)
    for key in ('task', 'original_task'):
        if isinstance(value.get(key), dict):
            value[key]['inputs'] = deepcopy(system.adapter.task.inputs)
    state = value.get('state', value.get('public_state'))
    if isinstance(state, dict) and 'inputs' in state:
        state['inputs'] = {k: v for k, v in state['inputs'].items() if k != 'source_pages'}
        state['inputs']['source_page_refs'] = refs
    value['provided_source_page_refs'] = refs
    return value
