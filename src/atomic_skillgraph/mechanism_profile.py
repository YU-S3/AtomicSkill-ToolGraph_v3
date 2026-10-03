"""Explicit opt-in policy boundary; legacy R10.3 readers remain unchanged."""
import json

VERSION = 'skillcompiler.mechanism.v2'
KEY = 'mechanism_effective_profile'


def enabled(config):
    return config.get('mechanism_profile') == VERSION


def resolve(config):
    version = config.get('mechanism_profile')
    if version not in (None, 'legacy', VERSION):
        raise ValueError('unsupported mechanism_profile')
    if not enabled(config):
        return None
    from .runtime.support_call_surface import VERSION as support
    from .harness.public_discovery import VERSION as discovery
    if config.get('repair_revision') != 'R10.3':
        raise ValueError('mechanism.v2 requires the R10.3 execution protocol')
    if config.get('runtime', {}).get('support_interface_version') != support:
        raise ValueError('mechanism.v2 requires explicit current Support interface')
    if config.get('harness', {}).get('adapter') == 'alfworld_v3' and (
            config.get('harness', {}).get('public_discovery_version') != discovery):
        raise ValueError('mechanism.v2 requires explicit public discovery')
    return {'version': VERSION, 'decision_frame': 'skillcompiler.decision-frame.v2',
            'support_interface_version': support,
            'public_discovery_version': config['harness'].get('public_discovery_version'),
            'generalization': 'skillcompiler.shared-contract.v2',
            'qualification': 'skillcompiler.train-qualification.v2'}


def bind(system):
    profile = resolve(system.config)
    row = system.database.execute('SELECT value FROM metadata WHERE key=?', (KEY,)).fetchone()
    stored = json.loads(row['value']) if row else None
    if stored is not None and stored != profile:
        raise ValueError('Bank effective mechanism profile does not match the reader')
    if profile is not None and stored is None:
        if system.readonly or system.database.execute('SELECT 1 FROM artifact_index LIMIT 1').fetchone():
            raise ValueError('mechanism.v2 requires a fresh Bank; legacy credit cannot be relabeled')
        system.database.execute('INSERT INTO metadata(key,value) VALUES(?,?)',
                                (KEY, json.dumps(profile, sort_keys=True)))
        system.database.connection.commit()
    return profile
