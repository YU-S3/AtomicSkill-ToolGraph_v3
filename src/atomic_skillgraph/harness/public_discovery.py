"""Conservative projection of the installed ALFWorld public feedback grammar.

No world snapshot, goal interpretation, navigation inference or model output
is accepted here. Relations expire at the next feedback revision.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

VERSION = 'alfworld.public-discovery.v1'
PARSER_VERSION = 'flat-listing.v1'
_ENTITY = r'[A-Za-z][A-Za-z0-9_-]*(?: [A-Za-z][A-Za-z0-9_-]*)* [0-9]+'
_LISTING = re.compile(r'(?:^|(?<=\.)\s+)(?:On the (?P<on>' + _ENTITY + r'), you see '
    r'|The (?P<inside>' + _ENTITY + r') is open\. In it, you see )(?P<items>[^.]+)\.')
_CLOSED = re.compile(r'(?:^|(?<=\.)\s+)The (?P<location>' + _ENTITY + r') is closed\.')
_ITEM = re.compile(r'(?:a|an) (' + _ENTITY + r')')


@dataclass(frozen=True)
class DiscoveryRecord:
    entity: str
    location: str
    source_kind: str
    source_ref: str
    relation_kind: str
    observed_revision: int
    raw_span: tuple[int, int] | None = None
    action_id: str | None = None


@dataclass(frozen=True)
class InspectedScope:
    location: str
    status: str
    source_ref: str


@dataclass(frozen=True)
class PublicDiscoveryFrame:
    version: str
    episode_id: str
    revision: int
    observation_hash: str
    records: tuple[DiscoveryRecord, ...]
    inspected_scopes: tuple[InspectedScope, ...]
    conflicts: tuple[tuple[str, tuple[str, ...]], ...]

    def to_dict(self):
        return asdict(self)

    def relation_facts(self):
        # The detailed frame retains corroborating sources; the public relation
        # view emits each joint identity once (observation precedes catalogue).
        unique = {(r.entity, r.location): r for r in reversed(self.records)}
        return [{'predicate': 'entity.discovered_at', 'effect_domain': 'evidence',
            'args': {'entity': r.entity, 'location': r.location},
            'observed_at_revision': r.observed_revision, 'source_kind': r.source_kind,
            'public_evidence_ref': r.source_ref, 'evidence_status': 'observed'} for r in unique.values()]


def project_discovery(*, observation, action_signature, accepted, revision, catalog, episode_id):
    from .alfworld import normalize_entity
    observation_hash = hashlib.sha256(observation.encode()).hexdigest()
    episode_hash = hashlib.sha256(episode_id.encode()).hexdigest()[:16]
    source = f'public_observation:{episode_hash}:{observation_hash}:revision:{revision}'
    # A goal may itself contain entity/relation words; never parse that section.
    text = re.split(r'Your task is to:', observation, maxsplit=1, flags=re.IGNORECASE)[0]
    offset = len(text) - len(text.lstrip())
    text = text.strip()
    records, scopes = [], []
    if accepted:
        for match in _LISTING.finditer(text):
            location = normalize_entity(match['on'] or match['inside'])
            listing = match['items']
            items = [] if listing == 'nothing' else re.split(r', and |, | and ', listing)
            parsed = [_ITEM.fullmatch(item) for item in items]
            valid = all(parsed)
            scopes.append(InspectedScope(location, 'complete_listing' if valid else 'unparsed', source))
            if valid:
                records.extend(DiscoveryRecord(normalize_entity(item[1]), location,
                    'public_observation_relation', source, 'on' if match['on'] else 'in',
                    revision, (offset + match.start(), offset + match.end())) for item in parsed)
        scopes.extend(InspectedScope(normalize_entity(m['location']), 'inaccessible', source)
                      for m in _CLOSED.finditer(text))
    for spec in catalog:
        if spec.revision != revision or spec.action_type != 'TAKE':
            continue
        entity, location = spec.arguments.get('object'), spec.arguments.get('source')
        if entity and location:
            records.append(DiscoveryRecord(normalize_entity(entity), normalize_entity(location),
                'public_action_catalog', f'action_catalog:{episode_hash}:{spec.action_id}:revision:{revision}',
                'catalog_take', revision, action_id=spec.action_id))
    # Multiple locations for one identity are ambiguity, not an invitation to
    # pick whichever source happened to be last in the iteration.
    locations = {}
    for row in records:
        locations.setdefault(row.entity, set()).add(row.location)
    conflicts = tuple(sorted((entity, tuple(sorted(values))) for entity, values in locations.items() if len(values) > 1))
    ambiguous = {entity for entity, _ in conflicts}
    records = list(dict.fromkeys(row for row in records if row.entity not in ambiguous))
    signature = action_signature or {}
    # Identity-only scope is useful diagnostics, never a complete inspection.
    if signature.get('action_type') == 'GO_TO':
        location = normalize_entity(signature.get('arguments', {}).get('destination', ''))
        if location and not any(s.location == location for s in scopes):
            scopes.append(InspectedScope(location, 'identity_only' if accepted else 'inaccessible', source))
    return PublicDiscoveryFrame(VERSION, episode_id, revision, observation_hash,
        tuple(records), tuple(dict.fromkeys(scopes)), conflicts)


def resource_contract(grammar_path):
    grammar = Path(grammar_path)
    return {'version': VERSION, 'parser_version': PARSER_VERSION,
        'grammar_sha256': hashlib.sha256(grammar.read_bytes()).hexdigest(),
        'parser_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'supported_relations': ['flat_on_listing', 'explicit_open_in_listing', 'catalog_take'],
        'effect_domain': 'evidence', 'predicate': 'entity.discovered_at'}
