"""Immutable ScienceWorld resource identity, independent of policy observations."""
from functools import lru_cache
import hashlib
import importlib.metadata
from pathlib import Path
import subprocess
from ..core.refs import content_hash

@lru_cache(maxsize=1)
def resource_contract():
    from scienceworld.constants import JAR_PATH
    distribution = importlib.metadata.distribution('scienceworld')
    if distribution.version != '1.2.3':
        raise ValueError('ScienceWorld 1.2.3 is required')
    package_files = sorted((str(p), hashlib.sha256(distribution.locate_file(p).read_bytes()).hexdigest())
        for p in distribution.files if str(p).startswith('scienceworld/') and str(p).endswith(('.py', '.json', '.jar')))
    return {'kind': 'scienceworld_variation', 'scienceworld_version': distribution.version,
        'jar_sha256': hashlib.sha256(Path(JAR_PATH).read_bytes()).hexdigest(),
        'package_sha256': content_hash(package_files),
        'java_version': subprocess.run(['java', '-version'], capture_output=True, text=True, check=True).stderr.strip(),
        'simplification': 'easy', 'env_step_limit': 100}

def verify_resource(expected):
    actual = resource_contract()
    differences = {k: (expected.get(k), actual.get(k)) for k in actual if expected.get(k) != actual[k]}
    if differences:
        raise ValueError(f'ScienceWorld resource identity changed: {differences}')
    return actual
