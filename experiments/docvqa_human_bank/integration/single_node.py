"""One immutable manual Skill, one text response, no tool or graph exposure."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path

METHOD = 'docvqa.human.single-node.v1'
ROOT = Path(__file__).resolve().parents[1]


def prepare(repo, config_path, destination):
    from atomic_skillgraph.empirical.bank import Bank
    from atomic_skillgraph.empirical.contracts import digest
    config = json.loads(Path(config_path).read_text())
    source = json.loads((ROOT / 'bank/single_node.json').read_text())
    output = Path(destination)
    if output.exists():
        raise ValueError('Fresh destination required')
    output.mkdir(parents=True)
    bank = Bank(output, seed=42, environment=config['program_environment'])
    try:
        asset = bank.put('skill', source)
        manifest = {'method_version': METHOD, 'origin': 'human_authored_only_not_learned',
            'bank_digest': bank.digest(), 'source_bundle_hash': digest(source),
            'counts': {'skill': 1, 'program': 0, 'implementation': 0, 'workflow': 0},
            'program_status': 'disabled', 'seed': 42}
        (output / 'manual_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (output / 'imported_bank.json').write_text(json.dumps({'skill': [asset]}, indent=2) + '\n')
    finally:
        bank.close()


class SingleNodeDocVQAView:
    def __init__(self, bank, manifest, **kwargs):
        from atomic_skillgraph.empirical.contracts import digest
        expected = {'skill': 1, 'program': 0, 'implementation': 0, 'workflow': 0}
        if (not bank.readonly or manifest['method_version'] != METHOD
                or manifest['bank_digest'] != bank.digest()
                or {k: len(bank.all(k)) for k in expected} != expected):
            raise ValueError('Expected the immutable single-Skill manual Bank')
        self.bank = bank
        self.view_hash = digest([METHOD, manifest, 'one_text_response_no_tools'])

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
    def program_eligible(self, program): return False


@contextmanager
def single_node_bridge():
    """Keep production metering/checkpoints/scoring; expose only the image and Skill."""
    from atomic_skillgraph.empirical.system import EmpiricalSystem
    old = EmpiricalSystem.agent

    def agent(s, stage, prompt, materials, name, schema, **kwargs):
        if stage == 'runtime' and isinstance(s.bank, SingleNodeDocVQAView):
            if name != 'answer_step':
                raise ValueError('Single-node evaluation permits one answer decision only')
            prompt = ('Answer the question from the supplied document image. The manual Skill is '
                'procedural guidance, not document evidence. Apply its relevant rules together. '
                'Return only the requested answer, without a plan or explanation. '
                + s.adapter.answer_contract())
            material = deepcopy({k: materials[k] for k in ('goal', 'guidance', 'content_parts')})
            answer = old(s, stage, prompt, material, None, None, **kwargs)
            return {'action': 'finish', 'answer': answer or ''}
        return old(s, stage, prompt, materials, name, schema, **kwargs)

    EmpiricalSystem.agent = agent
    try:
        yield
    finally:
        EmpiricalSystem.agent = old
