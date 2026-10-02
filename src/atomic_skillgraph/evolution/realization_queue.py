"""Bounded evidence-keyed realization attempts in the existing durable ledger.

Metadata participates in existing snapshots/digests/rollback. These records
are scheduling facts only, never execution observations or admission credit.
No migration, bank-wide scan, new LLM pool or Frozen writes are performed.
"""
import json
from ..core.refs import content_hash

VERSION = 'skillcompiler.realization.v1'
PREFIX = VERSION + ':'


class RealizationQueue:
    def __init__(self, database):
        self.database = database

    def claim(self, identity):
        key = PREFIX + content_hash(identity)
        with self.database.transaction() as connection:
            previous = connection.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()
            if previous is not None:
                return key, False, json.loads(previous[0])
            record = {'version': VERSION, 'identity': identity, 'status': 'running',
                      'initial_generation_limit': 1, 'protocol_repair_limit': 1,
                      'online_execution_credit': False}
            connection.execute('INSERT INTO metadata(key,value) VALUES(?,?)',
                               (key, json.dumps(record, sort_keys=True)))
        return key, True, record

    def finish(self, key, *, status, failure_codes=(), result_refs=()):
        if status in {'', 'pending', 'running'}:
            raise ValueError('realization requires a terminal status')
        with self.database.transaction() as connection:
            row = connection.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()
            if row is None:
                raise ValueError('unknown realization attempt')
            record = json.loads(row[0])
            if record['status'] != 'running':
                raise ValueError('realization attempt already terminal')
            record.update(status=status, failure_codes=list(failure_codes), result_refs=list(result_refs))
            connection.execute('UPDATE metadata SET value=? WHERE key=?',
                               (json.dumps(record, sort_keys=True), key))

    def finish_interrupted(self):
        """At a quiescent final drain, terminate incomplete attempts, never rerun."""
        rows = self.database.rows('SELECT key,value FROM metadata WHERE key LIKE ?', (PREFIX + '%',))
        for row in rows:
            if json.loads(row['value'])['status'] == 'running':
                self.finish(row['key'], status='interrupted', failure_codes=['generation_interrupted'])
