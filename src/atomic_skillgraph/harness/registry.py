"""Benchmark construction stays outside Planner and Runtime policy logic."""

def resolve_benchmark_identity(config):
    adapter = (config.get('harness') or {}).get('adapter', 'alfworld_v3')
    expected = {'alfworld_v3':'alfworld', 'scienceworld_v1':'scienceworld'}
    if adapter not in expected:
        raise ValueError(f'Unknown harness adapter: {adapter}')
    benchmark = (config.get('experiment') or {}).get('benchmark')
    if benchmark is None and adapter == 'alfworld_v3':
        benchmark = 'alfworld'  # Historical config compatibility only.
    if benchmark != expected[adapter]:
        raise ValueError(f'Benchmark/adapter identity mismatch: {benchmark!r}/{adapter}')
    return {'benchmark':benchmark, 'adapter':adapter}
def create_harness(config):
    resolve_benchmark_identity(config)
    settings = dict(config.get('harness') or {})
    experiment = config.get('experiment') or {}
    name = settings.get('adapter', 'alfworld_v3')
    split = str(experiment.get('split', settings.get('split', 'train')))
    if name == 'alfworld_v3':
        from .alfworld import AlfWorldAdapter
        return AlfWorldAdapter(split=split, max_steps=int(settings.get('max_steps', 100)),
            alfworld_data=settings.get('alfworld_data') or None,
            public_discovery_version=settings.get('public_discovery_version'))
    if name == 'scienceworld_v1':
        from .scienceworld import ScienceWorldAdapter
        return ScienceWorldAdapter(split=split, max_steps=int(settings.get('max_steps', 100)),
            simplification=settings.get('simplification', 'easy'),
            version=settings.get('scienceworld_version', '1.2.3'))
    raise ValueError(f'Unknown harness adapter: {name}')
