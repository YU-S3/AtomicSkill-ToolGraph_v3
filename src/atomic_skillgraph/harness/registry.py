"""Benchmark construction stays outside Planner and Runtime policy logic."""
def create_simple_harness(config):
    settings = config.get('harness', {})
    name = settings.get('adapter', 'alfworld_v3')
    if name == 'alfworld_v3':
        from .alfworld import AlfWorldAdapter
        from .alfworld_simple import SimpleAlfWorld
        return SimpleAlfWorld(AlfWorldAdapter(split=settings.get('split', 'train'),
            max_steps=settings.get('max_steps', 100), alfworld_data=settings.get('alfworld_data'),
            public_discovery_version=settings.get('public_discovery_version')))
    from .benchmarks import create_adapter
    return create_adapter(config)
