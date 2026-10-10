"""Single-cell dispatch and pause/resume control; intercepted campaign execution."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.contracts import digest
from atomic_skillgraph.experiments.canonical_manifest import BENCHMARKS, sha256
from atomic_skillgraph.experiments.formal_log import tree_identity
from atomic_skillgraph.experiments.run_empirical import write_json
from atomic_skillgraph.experiments import run_formal, run_single_cell

ROOT = Path(__file__).resolve().parents[1]


def test_t36_real_campaign_pause_resume_test_once_no_repeated_phase_cost(tmp_path, monkeypatch):
    base = yaml.safe_load((ROOT / 'configs/default.yaml').read_text())
    base['budget']={'token_limit':6000000,'finish_reserve':0,'request_limit':600}
    monkeypatch.setattr(run_formal,'code_identity',lambda:{'git_sha':'fixture','tracked_dirty':False,'source_sha256':'fixture'})
    profiles = json.loads((ROOT / 'benchmark_profiles.json').read_text())['profiles']
    datasets = Path(os.environ['CF4_DATASETS'])
    calls = []
    def run(config, tasks, phase, **kwargs):
        split = config['experiment']['runtime_mode']
        calls.append(phase.name)
        if split == 'online':
            bank = Bank(config['data_dir'])
            bank.freeze(phase / 'frozen_bank'); bank.close()
            kwargs['formal_log'].freeze(phase / 'frozen_bank', newly_created=True)
        summary = {'complete': True, 'tasks': len(tasks), 'successes': 0, 'known_total_tokens': 123,
            'unknown_billing_attempts': 0, 'knowledge_digest': 'fixture', 'cases': [{'score': {'raw_score': 0}} for _ in tasks]}
        write_json(phase / 'summary.json', summary)
        return summary
    monkeypatch.setattr(run_formal, 'run', run)
    root = tmp_path / 'cell'
    arguments = (base, profiles, 'searchqa', 42, root, datasets, ROOT / 'data/main_experiment_v1')
    first = run_formal.campaign(*arguments, stop_after_val=True)
    assert calls == ['train', 'val'] and set(first) == {'train', 'val'}
    completion = json.loads((root / 'completion.json').read_text())
    frozen = json.loads((root / 'final_frozen_manifest.json').read_text())
    assert completion['status'] == 'awaiting_test' and completion['completed_splits'] == ['train', 'val']
    assert not frozen['test_started_after_freeze'] and frozen['artifact_hash_after_test'] is None and not (root / 'test').exists()
    assert json.loads((root / 'run_manifest.json').read_text())['run_status'] == 'awaiting_test'
    before = {split: (root / split / 'summary.json').read_bytes() for split in ('train', 'val')}
    final = run_formal.campaign(*arguments, resume=True)
    run_formal.campaign(*arguments, resume=True)
    assert calls == ['train', 'val', 'test'] and set(final) == {'train', 'val', 'test'}
    frozen = json.loads((root / 'final_frozen_manifest.json').read_text())
    assert frozen['test_started_after_freeze'] and frozen['hash_before_test'] == frozen['artifact_hash_after_test']
    assert json.loads((root / 'completion.json').read_text())['status'] == 'completed'
    assert all((root / split / 'summary.json').read_bytes() == before[split] for split in before)
    different = deepcopy(base); different['llm']['model'] = 'other-model'
    with pytest.raises(ValueError, match='identity mismatch'):
        run_formal.campaign(different, *arguments[1:], resume=True)
    with pytest.raises(ValueError, match='relabel'):
        run_formal.campaign(*arguments, resume=True, stop_after_val=True)


@pytest.mark.parametrize('benchmark', BENCHMARKS)
def test_t36_single_cell_dispatches_only_selected_model_benchmark_seed(tmp_path, monkeypatch, benchmark):
    spec = yaml.safe_load((ROOT / 'configs/main_experiment_v1.yaml').read_text())
    for key in ('base_config', 'benchmark_profiles', 'model_lock', 'authority'): spec[key] = str(ROOT / spec[key])
    config = tmp_path / 'spec.yaml'; config.write_text(yaml.safe_dump(spec))
    calls = []
    monkeypatch.setattr(run_single_cell, 'campaign', lambda *a, **k: calls.append((a, k)))
    args = ['--config', str(config), '--benchmark', benchmark, '--seed', '43', '--datasets', os.environ['CF4_DATASETS'],
        '--output', str(tmp_path / 'cell'), '--stop-after-val']
    if benchmark == 'officeqa': args += ['--corpus-root', str(tmp_path)]
    run_single_cell.main(args)
    if benchmark == 'docvqa':
        assert not calls and json.loads((tmp_path / 'cell/completion.json').read_text())['status'] == 'unsupported'
    else:
        assert len(calls) == 1 and calls[0][0][2:4] == (benchmark, 43)
        assert calls[0][1]['stop_after_val'] and calls[0][0][0]['llm']['model'] == 'deepseek-v4-flash'


def test_t37_version_and_public_material_change_cannot_resume_old(tmp_path):
    from atomic_skillgraph.empirical.system import validate_config
    base = yaml.safe_load((ROOT / 'configs/default.yaml').read_text())
    original = deepcopy(base); original['experiment']['implementation_revision'] = 'empirical-v3.1-CF3'
    with pytest.raises(ValueError, match='implementation_revision'): validate_config(original)
    original = deepcopy(base); original['runtime']['model_view_version'] = 'changed'
    with pytest.raises(ValueError, match='model_view_version'): validate_config(original)
    profiles = json.loads((ROOT / 'benchmark_profiles.json').read_text())['profiles']
    with pytest.raises(ValueError, match='new public materialization'):
        run_formal.campaign(base, profiles, 'searchqa', 42, tmp_path / 'cell',
            '/home/yangchengyu/main_experiment_v1_resources_20261004', ROOT / 'data/main_experiment_v1')
    corpus = tmp_path / 'corpus'; corpus.mkdir(); file = corpus / 'text.txt'; file.write_text('original public corpus')
    before = run_formal.corpus_identity(corpus)
    file.write_text('changed public corpus')
    assert run_formal.corpus_identity(corpus)['sha256'] != before['sha256']


def test_t38_repository_and_installed_module_help_are_real_entrypoints():
    for entry in [['experiments/run_single_cell.py'], ['-m', 'atomic_skillgraph.experiments.run_single_cell']]:
        result = subprocess.run([sys.executable, *entry, '--help'], cwd=ROOT, capture_output=True, text=True, check=True)
        assert all(flag in result.stdout for flag in ('--benchmark', '--seed', '--model-key', '--stop-after-val', '--resume'))
