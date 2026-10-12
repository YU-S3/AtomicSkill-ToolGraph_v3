"""Episode-local public results and readable memory, never a learned Bank."""
import json
from uuid import uuid4

from . import POLICY_DEFAULTS
from .contracts import object_schema, digest
from ..harness.tool_spec import ToolSpec, result_schema
from .public_results import public_result, public_reference, SCOPED_SEARCH, FILE_COMPUTE


class TaskContext:
    def __init__(self, settings=None):
        self.settings = {**POLICY_DEFAULTS['runtime'], **(settings or {})}
        self.scope = uuid4().hex
        self.results, self.sources, self.memory, self.local_reads = {}, {}, {}, []
        self.result_origins, self.window_sources, self.sent_windows = {}, {}, []
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

    def register(self, event_id, result, *, name='', arguments=None, verify_source=False, origin=None):
        if event_id in self.sources:
            if verify_source and self.results[self.sources[event_id]] != result:
                from ..core.errors import AtomicSkillGraphError, FailureLayer
                raise AtomicSkillGraphError('learner_case_source_conflict',
                    'Immutable learning source has conflicting content', layer=FailureLayer.INFRASTRUCTURE)
            return self.sources[event_id]
        result_id = self.scope + ':' + str(sum(rid.removeprefix(self.scope + ':').isdigit() for rid in self.results))
        self.results[result_id] = result
        self.result_origins[result_id] = origin
        if name == 'read_result' and result.get('source_ref'):
            self.window_sources[result_id] = {k: result[k] for k in
                ('source_ref', 'unit', 'offset', 'end', 'size', 'next_offset', 'content_hash')}
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
        return {**values, **{key: self.resolve_public(ref) for key, ref in refs.items()}}

    def origin(self, rid):
        if rid in self.result_origins: return self.result_origins[rid]
        # Used only for read-only migration of recorded pre-protocol contexts.
        row = next((r for r in self.memory.values() if r['result_id'] == rid), {})
        if row.get('name') == 'grep' and 'scope_audit' in self.results[rid]: return SCOPED_SEARCH
        if row.get('name') == 'execute_python': return FILE_COMPUTE
        return None

    def public_result(self, raw_result, *, origin=None):
        return public_result(raw_result, origin)

    def resolve_public(self, ref):
        self.resolve(ref)  # Preserve strict ResultRef validation and raw Host lookup.
        return public_reference(self.results[ref['result_id']], ref['path'], self.origin(ref['result_id']))

    def business_refs(self, ref):
        value = self.resolve_public(ref)
        if isinstance(value, dict):
            return [leaf for key in value for leaf in self.business_refs(
                {'result_id': ref['result_id'], 'path': [*ref['path'], key]})]
        return [ref]

    def result_window(self, ref, offset=0, limit=None):
        value = self.resolve_public(ref)
        if isinstance(value, dict): raise ValueError('Read a business field path instead of serializing a dictionary root')
        array = isinstance(value, list)
        cap = self.settings['result_preview_max_items'] if array else self.settings['result_read_window_chars']
        limit = cap if limit is None else limit
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= cap:
            raise ValueError('read_result range exceeds its window')
        content = value if array or isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False)
        end = min(len(content), offset + limit)
        return {'source_ref': deepcopy_ref(ref), 'unit': 'array_item_0_based' if array else 'unicode_codepoint_0_based',
                'offset': offset, 'end': end, 'size': len(content), 'content': content[offset:end],
                'next_offset': end if end < len(content) else None, 'content_hash': digest(content[offset:end])}

    def fit_window(self, window, fits):
        """Reduce by whole source units; never misreport a half JSON element as delivered."""
        window = dict(window)
        while True:
            window['truncated'] = window['end'] < window['size']
            if fits(window) or not window['content']: break
            window['content'] = window['content'][:len(window['content']) // 2]
            window['end'] = window['offset'] + len(window['content'])
            window['next_offset'] = window['end'] if window['end'] < window['size'] else None
            window['content_hash'] = digest(window['content'])
        if not window['content'] and window['offset'] < window['size']:
            ref = window['source_ref']
            if window['unit'] == 'array_item_0_based':
                window['field_refs'] = self.business_refs({'result_id': ref['result_id'],
                    'path': [*ref['path'], window['offset']]})
            window['omission_reason'] = 'source_unit_exceeds_input_limit'
        return window

    def displayed_windows(self, material):
        windows = []
        def walk(value):
            if isinstance(value, dict):
                if {'source_ref', 'unit', 'offset', 'end', 'content_hash', 'content'} <= set(value):
                    windows.append({k: v for k, v in value.items() if k not in {'content', 'field_refs'}})
                else:
                    for child in value.values(): walk(child)
            elif isinstance(value, list):
                for child in value: walk(child)
        walk(material)
        return list({digest(w): w for w in windows}.values())

    def reference(self, value, source='model_value'):
        key = 'model_source:' + digest([source, value])
        result_id = self.sources.get(key, self.scope + ':source:' + digest([source, value]))
        self.sources[key], self.results[result_id] = result_id, value
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
        result = self.public_result(self.results[result_id], origin=self.origin(result_id))
        def field(value, path):
            ref = {'result_id': result_id, 'path': path}
            if isinstance(value, dict): return {k: field(v, [*path, k]) for k, v in value.items()}
            if isinstance(value, (list, str)) and len(json.dumps(value, ensure_ascii=False)) > self.settings['result_inline_max_chars']:
                window = self.result_window(ref)
                return self.fit_window(window, lambda w: len(json.dumps(w['content'], ensure_ascii=False)) <= self.settings['result_inline_max_chars'])
            return value
        if result_id in self.window_sources:
            # A bounded read page is already the requested window, not a new 2048-character preview.
            return {'result_id': result_id, 'data': {**self.window_sources[result_id], 'content': result['data']},
                    **{k: result[k] for k in ('accepted', 'error', 'done')}}
        return {'result_id': result_id, **{key: field(value, [key]) for key, value in result.items()}}

    def model_memory(self):
        return [{**row, 'arguments': self.preview(row['arguments'])} for row in list(self.memory.values())[-16:]]

    def read(self, result_id, offset=0, limit=None, path=None):
        window = self.result_window({'result_id': result_id, 'path': path or []}, offset, limit)
        self.local_reads.append({k: v for k, v in window.items() if k != 'content'})
        return {'accepted': True, 'observation': '', 'data': window.pop('content'), 'error': None, 'done': False, **window}

    def tool(self):
        schema = object_schema({'result_id': {'type': 'string'}, 'offset': {'type': 'integer', 'minimum': 0},
            'limit': {'type': 'integer', 'minimum': 1}, 'path': {'type': 'array', 'items': {
                'oneOf': [{'type': 'string'}, {'type': 'integer', 'minimum': 0}]}}}, ['result_id'])
        return ToolSpec('read_result', 'Read an obtained public business field path; offset/limit use Unicode text characters or array items. Dictionary roots expose field references; Host metadata is unavailable.',
                        schema, result_schema(), effect='read_only').view()


def deepcopy_ref(ref):
    return {'result_id': ref['result_id'], 'path': list(ref['path'])}


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
