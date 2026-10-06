"""Public benchmark contracts, independent of any learning method."""
from hashlib import sha256
from pathlib import Path
import json

VERSION = '3.1.4'


def source_identity():
    root = Path(__file__).parent
    files = {p.name: sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.iterdir()) if p.suffix == '.py' or p.name == 'LICENSE'}
    return sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
