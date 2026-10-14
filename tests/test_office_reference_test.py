"""Real Docker integration of externally authored assets, without model requests."""
import json
import os
from pathlib import Path

import pytest

from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.experiments.run_office_reference_test import import_bank, probe, workflow_probe
from atomic_skillgraph.experiments.office_reference_oracles import cases


def test_manual_bank_qualification_and_production_workflow(tmp_path):
    source = os.environ.get('OFFICE_REFERENCE_ASSETS')
    config_path = os.environ.get('OFFICE_REFERENCE_CONFIG')
    if not source or not config_path:
        pytest.skip('Requires supplied manual archive and the locked local Office Docker environment')
    source = Path(source)
    config = json.loads(Path(config_path).read_text())['test']
    config['experiment']['output_dir'] = str(tmp_path)
    bank, names = import_bank(source, tmp_path/'bank', config)
    try:
        assert len(bank.all('program')) == 10
        assert not any(bank.program_eligible(p) for p in bank.all('program'))
        programs = {names[p['id']]: p for p in bank.all('program')}
        fixture = source/'fixtures/office_corpus'
        plan = cases(fixture, fixture)
        for i, case in enumerate(plan):
            probe(bank, programs[case['name']], case, config, tmp_path/'probes'/str(i))
        assert all(bank.program_eligible(p) for p in bank.all('program'))
        for program in bank.all('program'):
            assert not bank.program_eligible({**program, 'source': program['source'] + '\n# altered'})
            records = bank.attempts(program['id'])
            assert len(records) == 2
            assert all(r['origin'] == 'manual_reference_validation' and r['split'] == 'reference_probe' for r in records)
            assert all(r['validation']['passed'] and r['result']['execution_receipt']['image_digest'] ==
                       config['program_environment']['image_digest'] for r in records)
        workflow_probe(bank, source, config, tmp_path)
        bank.freeze(tmp_path/'frozen')
        frozen = Bank(tmp_path/'frozen', readonly=True, seed=42, environment=config['program_environment'])
        try:
            assert len(frozen.all('program')) == 10
            assert len(frozen.all('workflow')) == 5
            assert not frozen.db.execute("SELECT 1 FROM sqlite_master WHERE name='train_cases'").fetchone()
            for p in frozen.all('program'):
                assert frozen.routes({'execution_mode':'program', 'program_id':p['id']})
        finally:
            frozen.close()
    finally:
        bank.close()


def test_full_read_reference_uses_structured_worker_output(tmp_path):
    source = os.environ.get('OFFICE_REFERENCE_ASSETS')
    config_path = os.environ.get('OFFICE_REFERENCE_CONFIG')
    if not source or not config_path:
        pytest.skip('Requires supplied manual archive and locked Docker environment')
    config = json.loads(Path(config_path).read_text())['test']
    bank, names = import_bank(Path(source), tmp_path/'bank', config)
    corpus = tmp_path/'corpus'
    corpus.mkdir()
    (corpus/'long.txt').write_text('Header\n' + 'Long public line\n' * 2000)
    case = {'name':'read_hit', 'label':'full12k', 'corpus':str(corpus),
            'inputs':{'path':'long.txt', 'offset':170}, 'status':'ok'}
    try:
        program = next(p for p in bank.all('program') if names[p['id']] == 'read_hit')
        probe(bank, program, case, config, tmp_path/'probe')
        receipt = json.loads((tmp_path/'probe/receipt.json').read_text())
        assert len(receipt['trial']['result']['outputs']['text']) == 12000
        assert receipt['trial']['validation']['variation'] is not None
        assert receipt['trial']['validation']['passed']
        assert all(e['name'] == 'read' for e in receipt['reference_trace']['tools'])
    finally:
        bank.close()
