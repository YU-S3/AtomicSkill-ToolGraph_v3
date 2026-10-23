"""Durable task stages and response recovery, separate from native side effects."""
import json
import os
from pathlib import Path
from uuid import uuid4


def native_event_restorable(event, specs, sealed_workspace=False):
    """Completed local refusals never reached a backend; unknown effects still block."""
    result = event.get('result', {})
    data = result.get('data')
    effect = specs.get(event.get('name'), {}).get('effect')
    unchanged_compute = (effect == 'sandbox_compute' and event.get('backend_invoked') is True and result.get('accepted') is True
        and isinstance(data, dict) and data.get('files') == [] and data.get('deleted_files') == []
        and result.get('local_artifact_hashes') == {} and result.get('local_artifact_identities') == {})
    return event.get('state') == 'finished' and (
        sealed_workspace or effect == 'read_only' or unchanged_compute
        or (event.get('backend_invoked') is False and
            (event.get('result', {}).get('accepted') is False or event.get('local_result_read') is True)))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class TaskCheckpoint:
    def __init__(self, root, *, response_namespace=None):
        self.root = Path(root)
        self.path = self.root / 'state.json'
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            'schema': 'empirical.checkpoint.v2', 'stage': 'task_started', 'decisions': {}}
        if self.state.get('schema') != 'empirical.checkpoint.v2':
            raise ValueError('Checkpoint policy changed; use an independent run directory')
        self.response_namespace = Path(response_namespace or self.state.get('recovery_identity',{}).get('response_namespace') or self.root)

    def advance(self, stage, **values):
        updated = {**self.state, 'stage': stage, **values}
        save_json(self.path, updated)
        self.state = updated

    def prepare_decision(self, scope, purpose, owner_state_version):
        identity = [scope, purpose, owner_state_version]
        for decision in reversed(list(self.state['decisions'].values())):
            if decision['identity'] == identity and decision['status'] in {'prepared', 'response_received'}:
                return decision['logical_decision_id']
        key = uuid4().hex
        decisions = {**self.state['decisions'], key: {'scope': scope, 'logical_decision_id': key,
            'purpose': purpose, 'owner_state_version': owner_state_version, 'identity': identity,
            'repair_index': 0, 'status': 'prepared'}}
        self.advance(self.state['stage'], decisions=decisions)
        return key

    def commit_decision(self, key, status, *, repair_index=None, **values):
        decisions = dict(self.state['decisions'])
        if key:
            decision = {**decisions[key], 'status': status}
            if repair_index is not None:
                decision['repair_index'] = repair_index
            decisions[key] = decision
        self.advance(values.pop('stage', self.state['stage']), decisions=decisions, **values)

    def native_events(self, events):
        save_json(self.root / 'native_events.json', events)

    def response(self, key):
        path = self.root / 'responses' / (key + '.json')
        return json.loads(path.read_text()) if path.exists() else None

    def save_response(self, key, value):
        path = self.root / 'responses' / (key + '.json')
        save_json(path, value)
        path.chmod(0o600)
