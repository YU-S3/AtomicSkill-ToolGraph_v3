"""Real Docker programs and original+crop wire delivery; synthetic model replies only."""
import base64
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'experiments/docvqa_human_bank'
sys.path.insert(0, str(ASSETS / 'integration'))
sys.path.insert(0, str(ASSETS / 'tools'))
from human_view import HumanDocVQAView, image_result_bridge, safe_crop_parts
from import_native import prepare
from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.contracts import PublicTask
from atomic_skillgraph.empirical.system import EmpiricalSystem
from atomic_skillgraph.harness.benchmarks import AnswerAdapter
from atomic_skillgraph.harness.simple_protocol import Broker
from atomic_skillgraph.empirical.task_context import TaskContext
from atomic_skillgraph.empirical.program_worker import ProgramWorker
from atomic_skillgraph.experiments.run_formal import model_settings, resolved_config
from test_cf2_contracts import http, response


@pytest.fixture
def imported(tmp_path):
    import yaml
    spec = yaml.safe_load((ROOT / 'configs/docvqa_deepseek_v41.yaml').read_text())
    model = json.loads((ROOT / spec['model_lock']).read_text())['formal_models'][0]
    base = model_settings(yaml.safe_load((ROOT / spec['base_config']).read_text()), model)
    profile = json.loads((ROOT / spec['benchmark_profiles']).read_text())['profiles']['docvqa']
    c = resolved_config(base, profile, 'docvqa', 42, 'test', tmp_path, tmp_path, ROOT / 'data/main_experiment_v1', None)
    c['data_dir'] = str(tmp_path / 'bank')
    c['program_worker']['record_execution_receipt'] = True
    path = tmp_path / 'config.json'; path.write_text(json.dumps(c))
    prepare(ROOT, path, tmp_path / 'bank')
    manifest = json.loads((tmp_path / 'bank/manual_manifest.json').read_text())
    return c, manifest


def test_native_ids_readonly_and_reference_scope(imported):
    c, manifest = imported
    bank = Bank(c['data_dir'], readonly=True, seed=42, environment=c['program_environment'])
    try:
        view = HumanDocVQAView(bank, manifest)
        assert [s['title'] for s in view.retrieve_guidance('What date?', limit=3)][:2] == ['DV01', 'DV05']
        assert view.retrieve_guidance('Which region has the highest attendance?', limit=3)[-1]['title']=='DV03'
        assert [s['title'] for s in view.retrieve_guidance('Read this handwritten address', limit=3)]==['DV01','DV05','DV02']
        assert view.program_options('Where was the meeting?')[0]['id'] == view.programs['image_region']['id']
        for p in bank.all('program'):
            assert p['state'] == 'candidate' and not bank.program_eligible(p)
            assert view.program_eligible(p)
        for link in bank.all('implementation'):
            assert bank.get(link['skill_id']) and bank.get(link['program_id'])
        assert not HumanDocVQAView(bank, manifest, arm='off').program_options('total')
    finally:
        bank.close()


def test_real_worker_programs_and_crop_wire(imported, monkeypatch):
    c, manifest = imported
    adapter = AnswerAdapter('docvqa', {'docvqa:fixture': {'answers': ['74216']}}, c)
    system = EmpiricalSystem(c, harness=adapter, readonly=True,
        bank_view_factory=lambda b: HumanDocVQAView(b, manifest))
    image = ASSETS / 'fixtures/synthetic_form.png'
    task = PublicTask('docvqa:fixture', 'synthetic-physical', 'What is the account number?', {'images': [str(image)]}, 'test')
    crop_id = system.bank.programs['image_region']['id']
    sent = http(monkeypatch, [response([{'action': 'call_program', 'name': crop_id,
        'arguments': {'image_path': '/workspace/inputs/image_0.png', 'region': 'top'}}], name='answer_step'),
        response([{'action': 'call_program', 'name': crop_id,
        'arguments': {'image_path': '/workspace/inputs/image_0.png', 'bbox': [250, 250, 500, 500]}}], name='answer_step'),
        response(content='74216', finish='stop')])
    try:
        before = system.bank.digest()
        with image_result_bridge(): trace = system.run_task(task, learn=False)
        assert len(sent) == 3
        attempt = trace['execution']['attempts'][0]
        assert attempt['status'] == 'ok', attempt
        assert safe_crop_parts(system.task_context, task, system.bank), {
            'origins': system.task_context.result_origins, 'memory': system.task_context.memory,
            'results': {k: {f: v for f, v in r.items() if f != 'outputs'} for k, r in system.task_context.results.items()}}
        first = sent[0]['messages'][1]['content']; second = sent[1]['messages'][1]['content']
        images = lambda parts: [x for x in parts if x['type'] == 'image_url']
        assert len(images(first)) == 1 and len(images(second)) == 2
        assert len(images(sent[2]['messages'][1]['content']))==3
        assert images(first)[0] == images(second)[0]
        text = next(x['text'] for x in second if x['type'] == 'text')
        assert 'crop_visual_evidence' in text
        crops = safe_crop_parts(system.task_context, task, system.bank)
        assert len(crops)==2
        crop = crops[0]
        assert hashlib.sha256(base64.b64decode(images(second)[1]['image_url']['url'].split(',')[1])).hexdigest() == crop['image_sha256']
        rid = crop['result_id']
        record = system.task_context.results[rid]
        out = record['outputs']
        assert out['bbox_normalized']==[0,0,1000,400]
        enlarged=system.task_context.results[crops[1]['result_id']]['outputs']
        assert enlarged['crop_width']==enlarged['source_crop_width']*3 and enlarged['resize_scale']==3
        assert out['image_mime']=='image/png'
        record['outputs']['source_sha256'] = '0' * 64
        assert len(safe_crop_parts(system.task_context, task, system.bank))==1
        assert system.bank.digest() == before
        broker = Broker(adapter, 1, context=TaskContext(c['runtime']))
        worker = ProgramWorker(c['program_worker'])
        numeric = worker.execute(system.bank.programs['numeric_compute'], {'operation': 'difference', 'values': ['5', '9']}, broker)
        assert numeric['status'] == 'ok' and numeric['outputs']['answer'] == '4'
        text_check = worker.execute(system.bank.programs['transcription_check'], {'observed_text': 'Account 74216', 'candidate': '74216'}, broker)
        assert text_check['status'] == 'ok' and text_check['outputs']['candidate_present']
        refused = worker.execute(system.bank.programs['image_region'], {'image_path': '/etc/passwd', 'bbox': [0, 0, 1000, 1000]}, broker)
        assert refused['status'] == 'needs_input'
        broad = worker.execute(system.bank.programs['image_region'],
            {'image_path': '/workspace/inputs/image_0.png', 'region': 'bottom'}, broker)
        assert broad['status']=='ok' and broad['outputs']['bbox_normalized']==[0,600,1000,1000]
        from atomic_skillgraph.empirical.contracts import validate_schema_instance
        with pytest.raises(ValueError):
            validate_schema_instance({'image_path':'/workspace/inputs/image_0.png',
                'region':'bottom','bbox':[0,0,1000,1000]},system.bank.programs['image_region']['input_schema'])
    finally:
        system.close()
