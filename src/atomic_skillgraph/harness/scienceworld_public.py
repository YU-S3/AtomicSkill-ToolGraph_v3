"""Public-only, conservative ScienceWorld evidence. No scientific answer tables."""
from __future__ import annotations
import re
from .protocol import PredicateSpec
from .public_discovery import PublicDiscoveryFrame, DiscoveryRecord, InspectedScope
from ..core.refs import content_hash

VERSION = 'scienceworld.public-evidence.v1'
WORLD = {
    'agent.at_location': ('location',), 'agent.holds': ('entity',),
    'container.open': ('container',), 'entity.in_container': ('entity', 'container'),
    'device.active': ('device',), 'device.inactive': ('device',),
    'electrical.connected': ('left', 'right'), 'electrical.disconnected': ('entity',),
    'entity.focused': ('entity',), 'matter.state_changed': ('entity', 'target_state'),
}
EVIDENCE = {
    'container.inspected': ('container', 'evidence'),
    'entity.discovered_at': ('entity', 'location'), 'scope.inspected': ('location', 'evidence'),
    'entity.examined': ('entity', 'evidence'), 'entity.read': ('entity', 'evidence'),
    'instrument.used_on': ('instrument', 'target', 'evidence'),
    'measurement.observed': ('subject', 'instrument', 'evidence'),
    'experiment.trial_observed': ('subject', 'apparatus', 'evidence'),
    'biology.life_stage_observed': ('entity', 'stage', 'evidence'),
    'chemistry.product_observed': ('entity', 'evidence'),
    'container.mixed': ('container', 'evidence'),
    'liquid.poured': ('source', 'destination', 'evidence'),
    'time.progressed': ('evidence',),
}
GOALS = ('matter', 'measurement', 'electricity', 'classification', 'growth',
         'chemistry', 'lifespan', 'life_stage', 'forces', 'genetics')

def normalize_listing(text):
    """Only unordered tab-indented sibling listings are order-insensitive.

    Preserve headers, prose, nested depth, quantities and duplicate objects.
    ScienceWorld 1.2.x iterates room contents in JVM hash order.
    """
    output, pending, depth = [], [], None
    for line in text.splitlines():
        current_depth = len(line) - len(line.lstrip('\t'))
        if not current_depth or current_depth != depth:
            output.extend(sorted(pending))
            pending = []
        if current_depth:
            pending.append(line.rstrip())
        else:
            output.append(line.rstrip())
        depth = current_depth
    output.extend(sorted(pending))
    return '\n'.join(output).strip()

def predicate_schema():
    return [PredicateSpec(p, domain, roles, {r: 'evidence_ref' if r == 'evidence' else 'entity' for r in roles},
                          VERSION) for domain, table in [('world', WORLD), ('evidence', EVIDENCE)]
            for p, roles in table.items()] + [PredicateSpec(f'scienceworld.{p}_goal_satisfied', 'world',
                ('task',), {'task': 'task'}, 'official_score') for p in GOALS]

def fact(predicate, arguments, revision):
    return {'predicate': predicate, 'args': arguments,
        'effect_domain': 'world' if predicate in WORLD else 'evidence',
        'observed_at_revision': revision,
        'witness_ref': 'sw_fact:' + content_hash([revision, predicate, arguments]),
        'evidence_status': 'observed', 'source_kind': VERSION}


def action_accepted(observation):
    """Public parser/action rejection messages, not task success or score.

    ScienceWorld can offer USE pairs for which the device rejects useWith at
    execution time. Presence in the catalog is therefore not an effect proof.
    """
    return not bool(re.search(
        r"(?:^|\n)(?:Ambiguous request|Which one|I don.t understand|I.m not sure|"
        r"That is not|You can.t|You cannot)", observation, re.I))


def listed_identity(line, entities):
    """Resolve the head of an official public listing, not a substring mention.

    Descriptions after ', ', '. ' or ' (' belong to the explicitly named head.
    Nested contents are not attributed to the room as direct children. Multiple
    valid heads are unresolved; an alias in the global catalog is not a witness.
    """
    clean = re.sub(r'^(?:a substance called |an |a |the )', '', line.strip())
    candidates = {e for e in entities if clean == e or any(clean.startswith(e + sep) for sep in (', ', '. ', ' ('))}
    return next(iter(candidates)) if len(candidates) == 1 else None

def frame_evidence(frame, catalog, revision, episode):
    """Only names actually printed in the current local listing are relations.

    The catalog can include remote destinations; availability alone never places
    an object in the current room. Descriptive/unparsed listing lines remain
    incomplete rather than being silently discarded as an empty room.
    """
    look = frame['look']
    matched = re.match(r'This room is called the (.+?)\. In it, you see:', look)
    location = matched[1] if matched else None
    source = 'sw_public:' + content_hash([episode, revision, frame])
    facts, records, scopes = [], [], []
    entities = {v for a in catalog for r, v in a.arguments.items() if r != 'destination'}
    if location:
        facts.append(fact('agent.at_location', {'location': location}, revision))
        lines = look.split('In it, you see:', 1)[1].split('You also see:', 1)[0].splitlines()
        complete = True
        for line in (s.strip() for s in lines if s.strip()):
            clean = re.sub(r'^(?:a substance called |an |a |the )', '', line)
            if clean == 'agent':
                continue
            entity = listed_identity(line, entities)
            if entity is None:
                complete = False
                continue
            records.append(DiscoveryRecord(entity, location, 'public_observation_relation',
                source, 'in_room', revision))
        scopes.append(InspectedScope(location, 'complete_listing' if complete else 'unparsed', source))
        facts.append(fact('scope.inspected', {'location': location, 'evidence': source}, revision))
    for line in frame['inventory'].splitlines()[1:]:
        entity = listed_identity(line, entities)
        if entity is not None:
            facts.append(fact('agent.holds', {'entity': entity}, revision))
    discovery = PublicDiscoveryFrame(VERSION, episode, revision, content_hash(frame), tuple(records), tuple(scopes), ())
    facts.extend(fact('entity.discovered_at', {'entity': r.entity, 'location': r.location}, revision) for r in records)
    return facts, discovery

def action_evidence(action, observation, revision, episode):
    name, a = action.action_type, action.arguments
    reference = 'sw_observation:' + content_hash([episode, revision, name, a, observation])
    items = []
    direct = {
        'OPEN': ('container.open', a), 'ACTIVATE': ('device.active', a),
        'DEACTIVATE': ('device.inactive', a), 'CONNECT': ('electrical.connected', a),
        'DISCONNECT': ('electrical.disconnected', a), 'FOCUS': ('entity.focused', a),
        'MOVE': ('entity.in_container', a), 'PICK_UP': ('agent.holds', a),
    }
    if name in direct:
        p, args = direct[name]
        # Broken devices accept the interaction but explicitly do not change
        # state. Only the official public state-change response proves it.
        device_change = {'ACTIVATE':'activated', 'DEACTIVATE':'deactivated'}
        if name not in device_change or re.search(
                r'^The .+ is now ' + device_change[name] + r'\.[\s]*$', observation, re.I):
            items.append(fact(p, dict(args), revision))
    events = {'EXAMINE': ('entity.examined', a), 'READ': ('entity.read', a),
        'USE': ('instrument.used_on', a), 'MIX': ('container.mixed', a),
        'POUR': ('liquid.poured', a), 'WAIT': ('time.progressed', {}), 'WAIT1': ('time.progressed', {})}
    if name in events:
        p, args = events[name]
        items.append(fact(p, {**args, 'evidence': reference}, revision))
    # A use command alone is NOT a measurement. Require the public response.
    if name == 'USE' and re.fullmatch(r'.+? measures a temperature of -?\d+(?:\.\d+)? degrees celsius[.\s]*', observation, re.I):
        items.append(fact('measurement.observed', {'subject': a['target'], 'instrument': a['instrument'],
                                                  'evidence': reference}, revision))
    return items, {'reference': reference, 'observation': observation, 'revision': revision,
                   'action_type': name, 'arguments': a}


def container_evidence(action, observation, catalog, room_frame, revision, episode):
    """Parse only this LOOK_IN response, aligned to the *new* public catalog.

    A partial listing may expose definite relations, but never proves absence.
    Container scopes are not represented as room InspectedScope objects.
    """
    container = action.arguments['container']
    source = 'sw_container:' + content_hash([episode, revision, container, observation])
    location = next((r.location for r in room_frame.records if r.entity == container), None)
    row = {'container':container, 'status':'unparsed', 'source_ref':source,
           'entities':[], 'unresolved':[], 'revision':revision, 'location':location}
    facts = []
    if re.search(r'^(?:Ambiguous request|Which one)', observation, re.I):
        row['status'] = 'ambiguous'
    elif re.search(r"(?:closed|can.t see inside|cannot see inside|not accessible)", observation, re.I):
        row['status'] = 'inaccessible'
    elif observation.strip() == f'There is nothing in the {container}.':
        row['status'] = 'empty_listing'
    elif observation.startswith(f'Inside the {container} is:'):
        body = observation[len(f'Inside the {container} is:'):]
        lines = [s.strip() for s in body.splitlines() if s.strip()]
        entities = {v for a in catalog for k,v in a.arguments.items() if k != 'destination'}
        # Official 1.2.3 LOOK_IN uses this exact empty-list body.  An empty
        # response or prose merely containing "nothing" does not prove absence.
        if lines == ['nothing']:
            row['status'] = 'empty_listing'
            lines = []
        else:
            row['status'] = 'complete_listing' if lines else 'unparsed'
        for line in lines:
            clean = re.sub(r'^(?:a substance called |an |a |the )', '', line)
            candidates = {e for e in entities if clean == e or any(clean.startswith(e+s) for s in (', ', '. ', ' ('))}
            if len(candidates) != 1:
                row['unresolved'].append(line)
                row['status'] = 'ambiguous' if len(candidates) > 1 else (
                    'ambiguous' if row['status'] == 'ambiguous' else 'unparsed')
                continue
            entity = next(iter(candidates))
            row['entities'].append(entity)
            facts.append(fact('entity.in_container', {'entity':entity,'container':container}, revision))
            if location:
                facts.append(fact('entity.discovered_at', {'entity':entity,'location':location}, revision))
    if row['status'] in {'complete_listing','empty_listing'}:
        facts.append(fact('container.inspected', {'container':container,'evidence':source}, revision))
    for item in facts:
        item['source_kind'] = VERSION + '/container_listing'
        item['witness_ref'] = source + ':' + content_hash([item['predicate'],item['args']])
    return facts, row

def retain_action_evidence(facts, action):
    """Invalidate affected relations, not every relation after every action.

    Evidence-domain records describe historical observations. They are not live
    state assertions. World relations are revoked when the corresponding public
    operation can change them; unrelated entities keep their last known state.
    """
    a, name = action.arguments, action.action_type
    def revoked(f):
        p, v = f['predicate'], f['args']
        if p == 'agent.holds':
            return True  # Re-established by the current inventory.
        if name in {'MOVE', 'PICK_UP', 'PUT_DOWN', 'EAT', 'FLUSH'}:
            if p == 'entity.in_container' and v.get('entity') == a.get('entity'):
                return True
        if name in {'OPEN', 'CLOSE'} and p == 'container.open':
            return v.get('container') == a['container']
        if name in {'ACTIVATE', 'DEACTIVATE'} and p in {'device.active', 'device.inactive'}:
            return v.get('device') == a['device']
        if name == 'DISCONNECT' and p == 'electrical.connected':
            return a['entity'] in (v.get('left'), v.get('right'))
        if name == 'CONNECT' and p == 'electrical.disconnected':
            return v.get('entity') in (a['left'], a['right'])
        if name == 'FOCUS' and p == 'entity.focused':
            return True
        return False
    return [f for f in facts if not revoked(f)]
