"""Immutable manual guidance with optional inspection and typed submission."""
from contextlib import contextmanager
from copy import deepcopy
import json
import hashlib
from pathlib import Path
import runpy

METHOD = 'docvqa.human.single-node.v6'
ROOT = Path(__file__).resolve().parents[1]
# Four page rows and three columns, with overlap to retain nearby labels.
TILES = {f'{column}{row + 1}': [
    max(0, col * 1000 // 3 - 50), max(0, row * 250 - 50),
    min(1000, ((col + 1) * 1000 + 2) // 3 + 50), min(1000, (row + 1) * 250 + 50)]
    for row in range(4) for col, column in enumerate('ABC')}


def row_context(task, output):
    """Keep distant row labels with the detail; derived only from the verified source."""
    for path in task.inputs.get('images', []):
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() == output['source_sha256']:
            bbox = [0, output['bbox_normalized'][1], 1000, output['bbox_normalized'][3]]
            render = runpy.run_path(str(ROOT / 'programs/image_region.py'))['crop_bytes']
            band = render(raw, bbox, max_edge=1600)
            return {'source_bbox': bbox, 'image_sha256': band['image_sha256'],
                'authority': 'host_derived_same_source_row_context_not_program_output_or_ocr',
                'part': {'type': 'image_url', 'image_url': {'detail': 'high',
                    'url': 'data:' + band['image_mime'] + ';base64,' + band['crop_base64']}}}
    raise ValueError('Verified crop source is unavailable')


def prepare(repo, config_path, destination, *, crop=False, skill_revision='v6', principle_ids=()):
    from atomic_skillgraph.empirical.bank import Bank
    from atomic_skillgraph.empirical.contracts import digest, program_digest
    config = json.loads(Path(config_path).read_text())
    sources = {'v6': 'single_node.json', 'v7': 'single_node_v7.json'}
    if skill_revision in ('v10', 'v11', 'v12'):
        from rule_search import skills
        source = skills(principle_ids, revision=skill_revision)
    elif skill_revision == 'v9':
        if principle_ids:
            raise ValueError('v9 uses its frozen public-intent clauses')
        from typed_answer import skills
        source = skills()
    elif skill_revision == 'v8':
        from principles import compile_skill
        source = compile_skill(principle_ids)
    else:
        if principle_ids:
            raise ValueError('Principle selection requires v8')
        source = json.loads((ROOT / 'bank' / sources[skill_revision]).read_text())
    output = Path(destination)
    if output.exists():
        raise ValueError('Fresh destination required')
    output.mkdir(parents=True)
    bank = Bank(output, seed=42, environment=config['program_environment'])
    try:
        assets = [bank.put('skill', item) for item in (source if skill_revision in ('v9', 'v10', 'v11', 'v12') else [source])]
        asset = assets[0]
        if crop:
            bundle = json.loads((ROOT / 'bank/bank.json').read_text())
            program = next(p for p in bundle['program'] if p['source_file'] == 'programs/image_region.py')
            program.update(environment=config['program_environment'], method_version=METHOD,
                           source=(ROOT / program['source_file']).read_text())
            program['id'] = 'program_' + program_digest(program)
            program = bank.put('program', program)
            bank.put('implementation', {'skill_id': asset['id'], 'program_id': program['id'],
                'method_version': METHOD, 'origin': 'human_authored_unqualified'})
        if skill_revision in ('v9', 'v10', 'v11', 'v12'):
            from typed_answer import program as typed_program
            formatter = bank.put('program', typed_program(config['program_environment'], METHOD,
                percentage=skill_revision == 'v11' and 'percent_render' in principle_ids))
            bank.put('implementation', {'skill_id': asset['id'], 'program_id': formatter['id'],
                'method_version': METHOD, 'origin': 'human_authored_unqualified'})
        manifest = {'method_version': METHOD, 'origin': 'human_authored_only_not_learned',
            'bank_digest': bank.digest(), 'source_bundle_hash': digest(source),
            'counts': {k: len(bank.all(k)) for k in ('skill', 'program', 'implementation', 'workflow')},
            'program_status': 'optional_image_crop' if crop else 'disabled', 'seed': 42,
            'skill_revision': skill_revision}
        if skill_revision in ('v9', 'v10', 'v11', 'v12'):
            manifest.update(typed_submission=True, routing='public_intent_first_match.v1')
        if skill_revision in ('v10', 'v11', 'v12'):
            manifest['principle_ids'] = [a['principle_id'] for a in assets if a['principle_id'] != 'core']
            if skill_revision == 'v11':
                manifest['percentage_submission'] = 'percent_render' in principle_ids
        if skill_revision == 'v8':
            manifest['principle_ids'] = source['principle_ids']
        (output / 'manual_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (output / 'imported_bank.json').write_text(json.dumps(
            {k: bank.all(k) for k in manifest['counts']}, indent=2) + '\n')
    finally:
        bank.close()


class SingleNodeDocVQAView:
    def __init__(self, bank, manifest, **kwargs):
        from atomic_skillgraph.empirical.contracts import digest
        self.crop_enabled = manifest['program_status'] == 'optional_image_crop'
        self.formatter_enabled = manifest.get('typed_submission', False)
        self.percentage_enabled = manifest.get('percentage_submission', False)
        if self.percentage_enabled != (manifest.get('skill_revision') == 'v11' and 'percent_render' in manifest.get('principle_ids', [])):
            raise ValueError('Percentage submission differs from the frozen rule selection')
        self.formatter_name = 'typed_submit_percent' if self.percentage_enabled else 'typed_submit'
        count = int(self.crop_enabled) + int(self.formatter_enabled)
        expected = {'skill': 4 if self.formatter_enabled else 1, 'program': count, 'implementation': count, 'workflow': 0}
        if manifest.get('skill_revision') in ('v10', 'v11', 'v12'):
            expected['skill'] = 1 + len(manifest['principle_ids'])
            if sorted(a['principle_id'] for a in bank.all('skill')) != sorted(['core'] + manifest['principle_ids']):
                raise ValueError('Frozen rule variants differ from the manifest')
        if (not bank.readonly or manifest['method_version'] != METHOD
                or manifest['bank_digest'] != bank.digest()
                or {k: len(bank.all(k)) for k in expected} != expected):
            raise ValueError('Manual Bank content does not match its frozen view')
        self.bank = bank
        self.programs = {Path(p['source_file']).stem: p for p in bank.all('program')}
        if self.formatter_enabled != (manifest.get('skill_revision') in ('v9', 'v10', 'v11', 'v12')):
            raise ValueError('Typed submission requires a matching Bank')
        self.view_hash = digest([METHOD, manifest,
            'public_intent_typed_submission.v9' if self.formatter_enabled else 'one_skill_optional_image_crop'])

    def __getattr__(self, key):
        return getattr(self.bank, key)

    def retrieve_guidance(self, query, limit=5):
        if self.formatter_enabled:
            from typed_answer import select
            return select(self.bank.all('skill'), query)[:limit]
        return self.bank.all('skill')[:limit]

    retrieve = retrieve_guidance

    def guidance_retrieval_audit(self, query, assets):
        return [{'rank': i, 'skill_id': a['id'], 'manual_source': METHOD,
            **({'public_intent_clause': a['principle_id']} if self.formatter_enabled else {})} for i, a in enumerate(assets)]

    def program_options(self, *args, **kwargs): return []
    def workflow_options(self): return []
    def planning_cards(self, query): return []
    def routes(self, *args, **kwargs): return []
    def program_eligible(self, program):
        return bool(program and program in self.programs.values())


@contextmanager
def single_node_bridge(*, guidance_role='user', evidence_interface='legacy', submission_contract='legacy'):
    """Keep production metering/checkpoints/scoring; expose only the image and Skill."""
    if guidance_role not in ('user', 'system'):
        raise ValueError('guidance_role must be user or system')
    if evidence_interface not in ('legacy', 'scoped', 'compact', 'typed'):
        raise ValueError('Unknown evidence interface')
    if submission_contract not in ('legacy', 'single_tool') or (submission_contract == 'single_tool' and evidence_interface != 'typed'):
        raise ValueError('Single-tool submission requires the typed interface')
    from atomic_skillgraph.empirical.system import EmpiricalSystem
    old = EmpiricalSystem.agent

    def agent(s, stage, prompt, materials, name, schema, **kwargs):
        if stage == 'runtime' and isinstance(s.bank, SingleNodeDocVQAView):
            if s.bank.formatter_enabled != (evidence_interface == 'typed'):
                raise ValueError('Typed interface and v9 Bank must be selected together')
            answer_step = name == 'answer_step'
            if name not in ('answer_step', 'finish_answer', None):
                raise ValueError('Unexpected single-node decision')
            prompt = ('Answer the question from the supplied document image. The manual Skill is '
                'procedural guidance, not document evidence. Apply its relevant rules together. '
                'Return the requested source text faithfully, without a plan or explanation. '
                + ('Submit the answer only in answer_step.answer; do not also write an XML or prose answer.'
                   if submission_contract == 'single_tool' and answer_step and materials['remaining_local_executions'] > 0
                   else s.adapter.answer_contract()))
            material = deepcopy({k: materials[k] for k in ('goal', 'guidance', 'content_parts') if k in materials})
            if not material.get('guidance'):
                material['guidance'] = s.bank.retrieve_guidance(material.get('goal', ''))
            if guidance_role == 'system':
                prompt += '\n\nManual Skill:\n' + '\n\n'.join(
                    item['guidance'] for item in material.pop('guidance', []))
            if not material.get('content_parts'):
                material['content_parts'] = s.adapter.content_parts()
            if s.bank.crop_enabled or s.bank.formatter_enabled:
                from human_view import safe_crop_parts
                contexts = []
                failures = [{k: r[k] for k in ('result_id', 'status', 'error', 'reason') if k in r}
                    for r in materials.get('local_results', []) if r.get('status') not in (None, 'ok')]
                if failures:
                    material['inspection_failures'] = failures
                for crop in safe_crop_parts(s.task_context, s.adapter.task, s.bank):
                    if len(material['content_parts']) >= s.config['llm']['image_input']['max_images']:
                        break
                    output = s.task_context.results[crop['result_id']]['outputs']
                    material['content_parts'].append(crop['part'])
                    material.setdefault('crop_evidence', []).append({'result_id': crop['result_id'],
                        'tile': next(tile for tile, bbox in TILES.items() if bbox == crop['bbox']),
                        'source_bbox': crop['bbox'], 'image_index': len(material['content_parts']) - 1,
                        'resize_scale': output['resize_scale'],
                        'source_crop_size': [output['source_crop_width'], output['source_crop_height']],
                        'delivered_size': [output['crop_width'], output['crop_height']]})
                    contexts.append((material['crop_evidence'][-1], row_context(s.adapter.task, output)))
                # Original and actual Program crops take priority over derived context.
                for evidence, context in contexts:
                    if len(material['content_parts']) >= s.config['llm']['image_input']['max_images']:
                        break
                    material['content_parts'].append(context.pop('part'))
                    evidence['row_context'] = {**context, 'image_index': len(material['content_parts']) - 1}
                # Selectable evidence handles are issued only for images actually delivered.
                # A page/tile label is never a Program result ID.
                sources = {'page': None}
                if evidence_interface != 'legacy':
                    material['answer_sources'] = [{'handle': 'page', 'kind': 'original_page'}]
                    for index, evidence in enumerate(material.get('crop_evidence', []), 1):
                        handle = f'crop_{index}'
                        sources[handle] = evidence.pop('result_id')
                        evidence['source_handle'] = handle
                        material['answer_sources'].append({'handle': handle,
                            'kind': 'verified_program_crop',
                            'image_index': evidence['image_index']})
                        if 'row_context' in evidence:
                            context_handle = f'row_{index}'
                            sources[context_handle] = None
                            evidence['row_context']['source_handle'] = context_handle
                            material['answer_sources'].append({'handle': context_handle,
                                'kind': 'host_derived_row_context_not_program_output',
                                'image_index': evidence['row_context']['image_index']})
                if answer_step and materials['remaining_local_executions'] > 0:
                    schema = {'type': 'object', 'properties': {
                        'action': {'type': 'string', 'enum': ['inspect', 'finish']},
                        'tile': {'type': 'string', 'enum': list(TILES),
                            'description': 'Original page: columns A/B/C left-to-right, rows 1/2/3/4 top-to-bottom. Tiles overlap.'},
                        'evidence_text': {'type': 'string', 'description': 'Visible supporting field or row, including its label; transcription, not reasoning.'},
                        'answer_kind': {'type': 'string', 'enum': ['entity', 'value', 'identifier', 'title', 'text', 'list']},
                        'answer': {'type': 'string'},
                        'used_results': {'type': 'array', 'items': {'type': 'string'}}},
                        'required': ['action'], 'additionalProperties': False,
                        'allOf': [
                            {'if': {'properties': {'action': {'const': 'inspect'}}},
                             'then': {'required': ['tile']}},
                            {'if': {'properties': {'action': {'const': 'finish'}}},
                             'then': {'required': ['evidence_text', 'answer_kind', 'answer']}}]}
                    if evidence_interface != 'legacy':
                        del schema['properties']['used_results']
                        schema['properties']['evidence_sources'] = {
                            'type': 'array', 'items': {'type': 'string', 'enum': list(sources)},
                            'minItems': 1, 'uniqueItems': True,
                            'description': 'Select only supplied answer_sources handles: page for the original image, crop_N for an actual Program crop, row_N for its host-derived context. Tile names are inspect locations, never source handles.'}
                        schema['allOf'][1]['then']['required'].append('evidence_sources')
                    if evidence_interface == 'typed':
                        from typed_answer import contract, instructions
                        can_inspect = s.bank.crop_enabled and materials['remaining_local_executions'] > 1
                        percentage = s.bank.percentage_enabled and any(a.get('principle_id') == 'percent_render'
                            for a in s.bank.retrieve_guidance(material.get('goal', '')))
                        schema = contract(schema, can_inspect=can_inspect, percentage=percentage)
                        prompt += ' ' + instructions(can_inspect, percentage=percentage)
                    elif evidence_interface == 'compact':
                        for key in ('evidence_text', 'answer_kind'):
                            del schema['properties'][key]
                        schema['allOf'][1]['then']['required'] = ['answer', 'evidence_sources']
                    if evidence_interface == 'compact':
                        prompt += (' Answer directly when clear. For uncertain detail, inspect its tile: '
                            'A/B/C are page columns; 1/2/3/4 are rows. At most two inspections. '
                            'Check the returned crop against the original label and row context; '
                            'completed tiles already contain the available pixels. '
                            'Use one answer_step call: inspect with tile, or finish with answer and '
                            'evidence_sources from answer_sources. page is the original; crop_N is a '
                            'Program crop; row_N is host context. Tile names are not source handles. '
                            'An inspection locates no target and verifies no text by itself.')
                    elif evidence_interface != 'typed':
                        prompt += (' For small print, handwriting or ambiguous characters affecting the answer, '
                        'inspect the tile containing the field before committing to a reading. '
                        'Return exactly one answer_step call per response, with one action; never parallel calls. '
                        'Divide the ORIGINAL page into four equal rows (1 top through 4 bottom) and '
                        'three equal columns (A left, B center, C right). Select one overlapping tile. '
                        'The Program determines the pixel coordinates; it does not identify or transcribe the target. '
                        'Check that the returned region actually contains the requested field and its relevant labels. '
                        'The attached crop_evidence images are inspections ALREADY COMPLETED, named by tile. '
                        'Read those images directly. Repeating a tile gives the same pixels and consumes another inspection. '
                        'Use a different tile when context is missing or the field was misplaced. '
                        'Each detail has a same-source full-width row_context to retain distant labels; '
                        'match the requested label to its value in that context before reading the detail. '
                        'Otherwise answer directly. At most two inspections; no mandatory second pass. '
                        'For finish, copy the supporting field/row into evidence_text, select answer_kind, '
                        'then put ONLY the requested constituent in answer, without introductory prose. '
                        'The evidence transcription is your reading, not independently verified OCR. ')
                    if evidence_interface not in ('compact', 'typed'):
                        prompt += ('For finish, select evidence_sources from answer_sources. The original page is page; '
                        'crop_N and row_N are available only after a successful inspection. A1/B2/etc. select where to inspect, '
                        'not evidence provenance. Never invent a handle or copy a tile into evidence_sources.'
                        if evidence_interface == 'scoped' else
                        'Use actual used_results when a crop supports the answer.')
                    material['remaining_inspections'] = (int(can_inspect) if evidence_interface == 'typed'
                                                         else materials['remaining_local_executions'])
                    step = deepcopy(old(s, stage, prompt, material, name, schema, **kwargs))
                    if step['action'] == 'inspect':
                        return {'action': 'call_program', 'name': s.bank.programs['image_region']['id'],
                            'arguments': {'image_path': '/workspace/inputs/image_0.png',
                                'bbox': TILES[step['tile']], 'max_edge': 1600}}
                    if evidence_interface != 'legacy':
                        # Schema validation already rejects unknown handles; do not drop invalid
                        # references or infer consumption from execution alone.
                        selected = step.pop('evidence_sources', [])  # Plain-text finish claims no usage.
                        step['used_results'] = [sources[k] for k in selected
                                                if sources[k] is not None]
                        s.requests[-1]['evidence_binding'] = {'version': 'scoped_image_sources.v1',
                            'available': sources, 'selected': selected, 'used_results': step['used_results']}
                        s._save_requests()
                    if evidence_interface == 'typed' and step.get('answer_kind', 'verbatim') != 'verbatim':
                        return {'action': 'call_program', 'name': s.bank.programs[s.bank.formatter_name]['id'],
                            'arguments': {'kind': step['answer_kind'], 'value': step['answer']},
                            'used_results': step.get('used_results', [])}
                    return step
            answer = old(s, stage, prompt, material, None if answer_step else name, None, **kwargs)
            return {'action': 'finish', 'answer': answer or ''} if answer_step else answer
        return old(s, stage, prompt, materials, name, schema, **kwargs)

    EmpiricalSystem.agent = agent
    try:
        yield
    finally:
        EmpiricalSystem.agent = old
