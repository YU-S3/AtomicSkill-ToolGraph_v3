"""ScienceWorld 1.2.3 public grammar and exact catalog authority.

No fuzzy matching or private object lookup. Keep all non-free policy actions;
reset is a controller operation, never a policy affordance.
"""
from __future__ import annotations
import re
from ..core.refs import content_hash
from .protocol import HarnessActionSpec

GRAMMAR = (
    ('OPEN', r'open (.+)', ('container',)),
    ('CLOSE', r'close (.+)', ('container',)),
    ('TELEPORT', r'teleport to (.+)', ('destination',)),
    ('GO_TO', r'go to (.+)', ('destination',)),
    ('LOOK_IN', r'look in (.+)', ('container',)),
    ('EXAMINE', r'look at (.+)', ('entity',)),
    ('ACTIVATE', r'activate (.+)', ('device',)),
    ('DEACTIVATE', r'deactivate (.+)', ('device',)),
    ('CONNECT', r'connect (.+?) to (.+)', ('left', 'right')),
    ('DISCONNECT', r'disconnect (.+)', ('entity',)),
    ('USE', r'use (.+?) on (.+)', ('instrument', 'target')),
    ('READ', r'read (.+)', ('entity',)),
    ('MOVE', r'move (.+?) to (.+)', ('entity', 'container')),
    ('PICK_UP', r'pick up (.+)', ('entity',)),
    ('PUT_DOWN', r'put down (.+)', ('entity',)),
    ('POUR', r'pour (.+?) into (.+)', ('source', 'destination')),
    ('DUNK', r'dunk (.+?) into (.+)', ('entity', 'container')),
    ('MIX', r'mix (.+)', ('container',)),
    ('FOCUS', r'focus on (.+)', ('entity',)),
    ('EAT', r'eat (.+)', ('entity',)),
    ('FLUSH', r'flush (.+)', ('entity',)),
    ('WAIT1', r'wait1', ()),
    ('WAIT', r'wait', ()),
    # Official parser clarification commands, exposed only while the public
    # catalog explicitly contains these strings. This is not an internal ID.
    ('CLARIFY', r'(\d+)', ('option',)),
)
FREE = {'look around', 'inventory', 'task'}
CONTROL = {'reset task'}

def primitive_schema():
    return [{'action_type': name, 'argument_roles': list(roles)} for name, _, roles in GRAMMAR]

def parse(raw):
    for action_type, pattern, roles in GRAMMAR:
        matched = re.fullmatch(pattern, raw)
        if matched:
            return action_type, dict(zip(roles, matched.groups()))
    raise ValueError(f'Unrecognized ScienceWorld public action: {raw!r}')

def catalog(rows, revision):
    result, seen = [], set()
    for row in rows:
        raw = row['action']
        if raw in FREE | CONTROL:
            continue
        action_type, arguments = parse(raw)
        identity = {'revision': revision, 'template_id': row['template_id'], 'raw_action': raw}
        action_id = 'sw_' + content_hash(identity)[:24]
        if action_id in seen:
            continue
        seen.add(action_id)
        result.append(HarnessActionSpec(action_id, revision, action_type, arguments, raw, raw,
            {'template_id': row['template_id'], 'obj_ids': row.get('obj_ids', []),
             'policy_surface': 'exact_tuple'}))
    return sorted(result, key=lambda spec: spec.action_id)

def resolve(items, action_type, arguments, revision):
    matches = [a for a in items if a.revision == revision and a.action_type == action_type
               and a.arguments == arguments]
    if len(matches) != 1:
        raise ValueError('ambiguous_action_tuple' if len(matches) > 1 else 'illegal_or_stale_action_tuple')
    return matches[0]

def compact(items):
    result = {}
    for name, _, roles in GRAMMAR:
        values = sorted({tuple(a.arguments[r] for r in roles) for a in items if a.action_type == name})
        if values:
            result[name] = ({roles[0]: [v[0] for v in values]} if len(roles) == 1
                            else {'roles': list(roles), 'tuples': [list(v) for v in values]})
    return result

def action_schema(items):
    # The discriminated branches are generated from exactly this request's catalog.
    branches = []
    for name, _, roles in GRAMMAR:
        entries = [a for a in items if a.action_type == name]
        if not entries:
            continue
        branches.append({'type': 'object', 'additionalProperties': False,
            'required': ['action_type', 'arguments'], 'properties': {
                'action_type': {'type': 'string', 'enum': [name]},
                'arguments': {'type': 'object', 'additionalProperties': False,
                    'required': list(roles), 'properties': {r: {'type': 'string',
                        'enum': sorted({a.arguments[r] for a in entries})} for r in roles}}}})
    return {'type': 'object', 'oneOf': branches}
