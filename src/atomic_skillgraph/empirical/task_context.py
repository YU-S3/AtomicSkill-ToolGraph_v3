"""Episode-local public results and readable memory, never a learned Bank."""
import json
from uuid import uuid4

from . import POLICY_DEFAULTS
from .contracts import object_schema, digest
from ..harness.tool_spec import ToolSpec, result_schema


class TaskContext:
    def __init__(self, settings=None):
        self.settings = {**POLICY_DEFAULTS['runtime'], **(settings or {})}
        self.scope = uuid4().hex
        self.results, self.sources, self.memory, self.local_reads = {}, {}, {}, []
        self.pending_outputs = {}
        self.active_node = None
        self.progress_observations, self.progress_contents, self.progress_states = [], set(), set()
        self.loop_hits, self.loop_feedback = {}, None

    def activate_node(self, key):
        if self.active_node != key:
            self.progress_observations, self.loop_hits, self.loop_feedback = [], {}, None
            self.active_node = key

    def mark_progress(self, kind, value):
        key = digest([kind, value])
        if key in self.progress_contents:
            return False
        self.progress_contents.add(key)
        self.progress_observations, self.loop_hits, self.loop_feedback = [], {}, None
        return True

    def observe_progress(self, event):
        result = event['result']
        content = stable_public({key: result[key] for key in ['observation', 'data', 'outputs', 'error'] if key in result})
        state = event.get('progress_after')
        changed = state not in self.progress_states
        self.progress_states.add(state)
        if self.mark_progress('public_result', content) or changed:
            self.progress_observations, self.loop_hits, self.loop_feedback = [], {}, None
        signature = digest([event['name'], event['arguments'], event.get('progress_before'), state, content])
        if self.progress_observations and self.progress_observations[-1]['node'] != self.active_node:
            self.progress_observations = []
        self.progress_observations.append({'node': self.active_node, 'signature': signature, 'name': event['name']})
        self.progress_observations = self.progress_observations[-16:]
        sequence = [x['signature'] for x in self.progress_observations]
        for length in range(1, 5):
            if len(sequence) >= 2 * length and sequence[-length:] == sequence[-2*length:-length]:
                key = digest(sequence[-length:])
                self.loop_hits[key] = self.loop_hits.get(key, 0) + 1
                self.loop_feedback = {'kind': 'no_progress_loop', 'signature': key, 'pattern_length': length,
                                      'hits': self.loop_hits[key], 'operations': [x['name'] for x in self.progress_observations[-length:]]}
                break

    def register(self, event_id, result, *, name='', arguments=None):
        if event_id in self.sources:
            return self.sources[event_id]
        result_id = self.scope + ':' + str(len(self.results))
        self.results[result_id] = result
        self.sources[event_id] = result_id
        if name:
            key = json.dumps([name, arguments], sort_keys=True, ensure_ascii=False)
            self.memory[key] = {'name': name, 'arguments': arguments, 'result_id': result_id,
                                'accepted': result.get('accepted'), 'status': result.get('status')}
        return result_id

    def resolve(self, ref):
        if not isinstance(ref, dict) or set(ref) != {'result_id', 'path'}:
            raise ValueError('ResultRef needs result_id and path')
        if ref['result_id'] not in self.results:
            raise ValueError('Unknown result_id in this episode')
        if not isinstance(ref['path'], list):
            raise ValueError('ResultRef.path must be a list')
        value = self.results[ref['result_id']]
        for part in ref['path']:
            if isinstance(part, str) and isinstance(value, dict) and part in value:
                value = value[part]
            elif type(part) is int and part >= 0 and isinstance(value, list) and part < len(value):
                value = value[part]
            else:
                raise ValueError('Invalid ResultRef.path')
        return value

    def bind(self, values, refs):
        if set(values) & set(refs):
            raise ValueError('Explicit values and result references conflict')
        return {**values, **{key: self.resolve(ref) for key, ref in refs.items()}}

    def reference(self, value, source='model_value'):
        result_id = self.register('model_source:' + digest([source, value]), value)
        return {'result_id': result_id, 'path': []}

    def preview(self, value, ref=None):
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False)
        if len(raw) <= self.settings['result_inline_max_chars'] and (not isinstance(value, list) or len(value) <= self.settings['result_preview_max_items']):
            return value
        preview = value[:self.settings['result_preview_max_items']] if isinstance(value, list) else value
        text = json.dumps(preview, ensure_ascii=False, allow_nan=False)
        return {'ref': ref or self.reference(value), 'type': type(value).__name__, 'size': len(value) if isinstance(value, (list, str, dict)) else len(raw),
                'preview': text[:self.settings['result_inline_max_chars']], 'truncated': True}

    def view(self, result_id):
        result = self.results[result_id]
        return {'result_id': result_id, **{key: self.preview(value, {'result_id': result_id, 'path': [key]})
                                         for key, value in result.items()}}

    def model_memory(self):
        return [{**row, 'arguments': self.preview(row['arguments'])} for row in self.memory.values()]

    def read(self, result_id, offset=0, limit=None, path=None):
        value = self.resolve({'result_id': result_id, 'path': path or []})
        if type(offset) is not int or offset < 0:
            raise ValueError('read_result.offset must be non-negative')
        array = isinstance(value, list)
        cap = self.settings['result_preview_max_items'] if array else self.settings['result_read_window_chars']
        if limit is None: limit = cap
        if type(limit) is not int or not 1 <= limit <= cap:
            raise ValueError('read_result.limit exceeds its window')
        content = value if array or isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        end = min(len(content), offset + limit)
        self.local_reads.append({'result_id': result_id, 'path': path or [], 'offset': offset, 'limit': limit})
        return {'accepted': True, 'observation': '', 'data': content[offset:end], 'error': None, 'done': False,
                'unit': 'array_item_0_based' if array else 'unicode_codepoint_0_based',
                'size': len(content), 'next_offset': end if end < len(content) else None}

    def tool(self):
        schema = object_schema({'result_id': {'type': 'string'}, 'offset': {'type': 'integer', 'minimum': 0},
            'limit': {'type': 'integer', 'minimum': 1}, 'path': {'type': 'array', 'items': {
                'oneOf': [{'type': 'string'}, {'type': 'integer', 'minimum': 0}]}}}, ['result_id'])
        return ToolSpec('read_result', 'Read an already obtained result from this episode; text characters or array items.',
                        schema, result_schema(), effect='read_only').view()


def public_view(adapter):
    view = getattr(adapter, 'model_state', adapter.observe)()
    return view


def progress_key(adapter):
    from .contracts import digest
    state = getattr(adapter, 'progress_key', None)
    if state:
        return state()
    return digest([stable_public(adapter.observe()), stable_public(adapter.available_tools())])


def stable_public(value):
    if isinstance(value, dict):
        return {k: stable_public(v) for k,v in value.items() if k not in {
            'revision','source_ref','timestamp','remaining_calls','version','event_id','result_id',
            'calls','elapsed_seconds','environment_step','environment_steps'}}
    if isinstance(value, list):
        return [stable_public(v) for v in value]
    return value
