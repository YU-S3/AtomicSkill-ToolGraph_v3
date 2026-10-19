"""Explicit public input for BOTH arms; supplied pages are not tool executions."""
from copy import deepcopy

from atomic_skillgraph.empirical.contracts import digest
from atomic_skillgraph.empirical.public_results import TEXT_CONTEXT

VERSION = 'office.external-minimax.public-pages.v1'


def seed_context(system):
    task = system.adapter.task
    if system.config['harness'].get('office_public_input_version') != VERSION:
        raise ValueError('HumanStrong V2 requires the external public input configuration')
    inputs = task.inputs
    if inputs.get('office_public_input_version') != VERSION:
        raise ValueError('External public inputs required; refusing old full-corpus-only tasks')
    refs = []
    for i, page in enumerate(inputs['source_pages']):
        if digest(page['text']) != page['text_digest']:
            raise ValueError('Provided page content differs from its frozen input identity')
        identity = digest([task.physical_key, i, page])
        rid = system.task_context.register('public_input:' + identity,
            {'accepted': True, 'data': {'text': page['text'], 'next_offset': len(page['text'])},
             'provenance': {'kind': 'provided_public_input', 'source_url': page['source_url'],
                            'document': page['document'], 'page': page['page']}},
            name='provided_source_page', arguments={'path': page['document'] + '#page=' + page['page']},
            origin=TEXT_CONTEXT, verify_source=True)
        refs.append({'result_id': rid, 'path': ['data', 'text']})
    return refs


def runtime_material(system, material):
    refs = seed_context(system)
    value = deepcopy(material)
    # Native projection normally previews file inputs. This protocol explicitly
    # supplies complete pages, so restore the frozen public input before admission.
    for key in ('task', 'original_task'):
        if isinstance(value.get(key), dict):
            value[key]['inputs'] = deepcopy(system.adapter.task.inputs)
    # original_task carries every original page. Avoid a second full copy via observe().
    state = value.get('state', value.get('public_state'))
    if isinstance(state, dict) and 'inputs' in state:
        state['inputs'] = {k: v for k, v in state['inputs'].items() if k != 'source_pages'}
        state['inputs']['source_page_refs'] = refs
    value['provided_source_page_refs'] = refs
    return value
