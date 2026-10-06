"""Build the public contracts alone, from their unique source tree."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def build(output):
    source = Path(__file__).resolve().parents[1] / 'src/skillcompiler_bench_contracts'
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.iterdir())
              if p.suffix == '.py' or p.name == 'LICENSE'}
    with tempfile.TemporaryDirectory(prefix='benchmark-contracts-build-') as temporary:
        root = Path(temporary)
        package = root / 'src/skillcompiler_bench_contracts'
        package.mkdir(parents=True)
        for name in hashes: shutil.copyfile(source / name, package / name)
        shutil.copyfile(source / 'LICENSE', root / 'LICENSE')
        (root / 'pyproject.toml').write_text('''[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"
[project]
name = "skillcompiler-bench-contracts"
version = "3.1.4"
requires-python = ">=3.10"
dependencies = []
license = {file = "LICENSE"}
[tool.setuptools.packages.find]
where = ["src"]
namespaces = false
[tool.setuptools.package-data]
skillcompiler_bench_contracts = ["LICENSE"]
''')
        subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
                        '--wheel-dir', str(output), str(root)], check=True)
    wheel = output / 'skillcompiler_bench_contracts-3.1.4-py3-none-any.whl'
    manifest = {'distribution': 'skillcompiler-bench-contracts', 'version': '3.1.4',
        'source_files_sha256': hashes,
        'benchmark_contracts_sha256': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        'wheel': wheel.name, 'wheel_sha256': hashlib.sha256(wheel.read_bytes()).hexdigest()}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    print(json.dumps(build(parser.parse_args().output)))
