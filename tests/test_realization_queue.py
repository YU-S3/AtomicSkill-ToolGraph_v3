import pytest
from atomic_skillgraph.knowledge.database import StateDatabase
from atomic_skillgraph.evolution.realization_queue import RealizationQueue


def test_once_per_evidence_terminal_and_frozen_readonly(tmp_path):
    path = tmp_path / 'state.sqlite3'
    with StateDatabase(path) as db:
        queue = RealizationQueue(db)
        identity = {'contract_hash': 'contract', 'source': 'one', 'builder_input_hash': 'input'}
        key, claimed, row = queue.claim(identity)
        assert claimed and not row['online_execution_credit']
        queue.finish(key, status='no_tool')
        assert queue.claim(identity)[1:] == (False, {**row, 'status': 'no_tool',
                                                    'failure_codes': [], 'result_refs': []})
        assert queue.claim({**identity, 'source': 'two'})[1]
        queue.finish_interrupted()
    with StateDatabase(path) as db:
        assert not RealizationQueue(db).claim(identity)[1]
    with StateDatabase(path, readonly=True) as db:
        with pytest.raises(RuntimeError, match='read-only'):
            RealizationQueue(db).claim({**identity, 'source': 'new'})
