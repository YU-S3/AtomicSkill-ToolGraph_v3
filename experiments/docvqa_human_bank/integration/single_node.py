"""One immutable manual Skill; optional source-image crop, no graph routing."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path

METHOD = 'docvqa.human.single-node.v5'
ROOT = Path(__file__).resolve().parents[1]
# Four page rows and three columns, with overlap to retain nearby labels.
TILES = {f'{column}{row + 1}': [
    max(0, col * 1000 // 3 - 50), max(0, row * 250 - 50),
    min(1000, ((col + 1) * 1000 + 2) // 3 + 50), min(1000, (row + 1) * 250 + 50)]
    for row in range(4) for col, column in enumerate('ABC')}


def prepare(repo, config_path, destination, *, crop=False):
    from atomic_skillgraph.empirical.bank import Bank
    from atomic_skillgraph.empirical.contracts import digest, program_digest
    config = json.loads(Path(config_path).read_text())
    source = json.loads((ROOT / 'bank/single_node.json').read_text())
    output = Path(destination)
    if output.exists():
        raise ValueError('Fresh destination required')
    output.mkdir(parents=True)
    bank = Bank(output, seed=42, environment=config['program_environment'])
    try:
        asset = bank.put('skill', source)
        if crop:
            bundle = json.loads((ROOT / 'bank/bank.json').read_text())
            program = next(p for p in bundle['program'] if p['source_file'] == 'programs/image_region.py')
            program.update(environment=config['program_environment'], method_version=METHOD,
                           source=(ROOT / program['source_file']).read_text())
            program['id'] = 'program_' + program_digest(program)
            program = bank.put('program', program)
            bank.put('implementation', {'skill_id': asset['id'], 'program_id': program['id'],
                'method_version': METHOD, 'origin': 'human_authored_unqualified'})
        manifest = {'method_version': METHOD, 'origin': 'human_authored_only_not_learned',
            'bank_digest': bank.digest(), 'source_bundle_hash': digest(source),
            'counts': {k: len(bank.all(k)) for k in ('skill', 'program', 'implementation', 'workflow')},
            'program_status': 'optional_image_crop' if crop else 'disabled', 'seed': 42}
        (output / 'manual_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (output / 'imported_bank.json').write_text(json.dumps(
            {k: bank.all(k) for k in manifest['counts']}, indent=2) + '\n')
    finally:
        bank.close()


class SingleNodeDocVQAView:
    def __init__(self, bank, manifest, **kwargs):
        from atomic_skillgraph.empirical.contracts import digest
        self.crop_enabled = manifest['program_status'] == 'optional_image_crop'
        expected = {'skill': 1, 'program': int(self.crop_enabled), 'implementation': int(self.crop_enabled), 'workflow': 0}
        if (not bank.readonly or manifest['method_version'] != METHOD
                or manifest['bank_digest'] != bank.digest()
                or {k: len(bank.all(k)) for k in expected} != expected):
            raise ValueError('Expected the immutable single-Skill manual Bank')
        self.bank = bank
        self.programs = {'image_region': bank.all('program')[0]} if self.crop_enabled else {}
        self.view_hash = digest([METHOD, manifest, 'one_skill_optional_image_crop'])

    def __getattr__(self, key):
        return getattr(self.bank, key)

    def retrieve_guidance(self, query, limit=5):
        return self.bank.all('skill')[:limit]

    retrieve = retrieve_guidance

    def guidance_retrieval_audit(self, query, assets):
        return [{'rank': i, 'skill_id': a['id'], 'manual_source': METHOD} for i, a in enumerate(assets)]

    def program_options(self, *args, **kwargs): return []
    def workflow_options(self): return []
    def planning_cards(self, query): return []
    def routes(self, *args, **kwargs): return []
    def program_eligible(self, program):
        return bool(self.crop_enabled and program and program == self.programs['image_region'])


@contextmanager
def single_node_bridge(*, guidance_role='user'):
    """Keep production metering/checkpoints/scoring; expose only the image and Skill."""
    if guidance_role not in ('user', 'system'):
        raise ValueError('guidance_role must be user or system')
    from atomic_skillgraph.empirical.system import EmpiricalSystem
    old = EmpiricalSystem.agent

    def agent(s, stage, prompt, materials, name, schema, **kwargs):
        if stage == 'runtime' and isinstance(s.bank, SingleNodeDocVQAView):
            answer_step = name == 'answer_step'
            if name not in ('answer_step', 'finish_answer', None):
                raise ValueError('Unexpected single-node decision')
            prompt = ('Answer the question from the supplied document image. The manual Skill is '
                'procedural guidance, not document evidence. Apply its relevant rules together. '
                'Return the requested source text faithfully, without a plan or explanation. '
                + s.adapter.answer_contract())
            material = deepcopy({k: materials[k] for k in ('goal', 'guidance', 'content_parts') if k in materials})
            if not material.get('guidance'):
                material['guidance'] = s.bank.all('skill')
            if guidance_role == 'system':
                prompt += '\n\nManual Skill:\n' + '\n\n'.join(
                    item['guidance'] for item in material.pop('guidance', []))
            if not material.get('content_parts'):
                material['content_parts'] = s.adapter.content_parts()
            if s.bank.crop_enabled:
                from human_view import safe_crop_parts
                inspected = set()
                failures = [{k: r[k] for k in ('result_id', 'status', 'error', 'reason') if k in r}
                    for r in materials.get('local_results', []) if r.get('status') not in (None, 'ok')]
                if failures:
                    material['inspection_failures'] = failures
                for crop in safe_crop_parts(s.task_context, s.adapter.task, s.bank):
                    inspected.update(tile for tile, bbox in TILES.items() if bbox == crop['bbox'])
                    if len(material['content_parts']) >= s.config['llm']['image_input']['max_images']:
                        break
                    output = s.task_context.results[crop['result_id']]['outputs']
                    material['content_parts'].append(crop['part'])
                    material.setdefault('crop_evidence', []).append({'result_id': crop['result_id'],
                        'source_bbox': crop['bbox'], 'image_index': len(material['content_parts']) - 1,
                        'resize_scale': output['resize_scale'],
                        'source_crop_size': [output['source_crop_width'], output['source_crop_height']],
                        'delivered_size': [output['crop_width'], output['crop_height']]})
                if answer_step and materials['remaining_local_executions'] > 0:
                    schema = {'type': 'object', 'properties': {
                        'action': {'enum': ['inspect', 'finish']},
                        'tile': {'enum': [tile for tile in TILES if tile not in inspected],
                            'description': 'Original page: columns A/B/C left-to-right, rows 1/2/3/4 top-to-bottom. Tiles overlap.'},
                        'answer': {'type': 'string'},
                        'used_results': {'type': 'array', 'items': {'type': 'string'}}},
                        'required': ['action'], 'additionalProperties': False,
                        'allOf': [
                            {'if': {'properties': {'action': {'const': 'inspect'}}},
                             'then': {'required': ['tile']}},
                            {'if': {'properties': {'action': {'const': 'finish'}}}, 'then': {'required': ['answer']}}]}
                    prompt += (' For small print, handwriting or ambiguous characters affecting the answer, '
                        'inspect the tile containing the field before committing to a reading. '
                        'Divide the ORIGINAL page into four equal rows (1 top through 4 bottom) and '
                        'three equal columns (A left, B center, C right). Select one overlapping tile. '
                        'The Program determines the pixel coordinates; it does not identify or transcribe the target. '
                        'Check that the returned region actually contains the requested field and its relevant labels. '
                        'A second inspection must use a different tile for missing context or a misplaced field. '
                        'Otherwise answer directly. At most two inspections; no mandatory second pass. '
                        'Use finish with actual used_results when a crop supports the answer.')
                    material['remaining_inspections'] = materials['remaining_local_executions']
                    step = old(s, stage, prompt, material, name, schema, **kwargs)
                    if step['action'] == 'inspect':
                        return {'action': 'call_program', 'name': s.bank.programs['image_region']['id'],
                            'arguments': {'image_path': '/workspace/inputs/image_0.png',
                                'bbox': TILES[step['tile']], 'max_edge': 1600}}
                    return step
            answer = old(s, stage, prompt, material, None if answer_step else name, None, **kwargs)
            return {'action': 'finish', 'answer': answer or ''} if answer_step else answer
        return old(s, stage, prompt, materials, name, schema, **kwargs)

    EmpiricalSystem.agent = agent
    try:
        yield
    finally:
        EmpiricalSystem.agent = old
