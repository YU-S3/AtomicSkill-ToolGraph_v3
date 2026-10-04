"""Public input staging and atomic publication of complete output versions."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import hashlib
from uuid import uuid4


class Workspace:
    def __init__(self, root, inputs=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.inputs = {name: Path(path).resolve(strict=True) for name, path in (inputs or {}).items()}

    def stage(self):
        directory = Path(tempfile.mkdtemp(prefix='stage-', dir=self.root))
        directory.chmod(0o777)
        manifest = self.root / 'manifest.json'
        if manifest.exists():
            previous = self.root / json.loads(manifest.read_text())['version']
            for path in previous.iterdir():
                if path.name == 'inputs': continue
                if path.is_symlink(): raise ValueError('Published workspace contains a symlink')
                target = directory / path.name
                if path.is_dir(): shutil.copytree(path, target)
                else: shutil.copyfile(path, target)
                for item in [target, *target.rglob('*')] if target.is_dir() else [target]:
                    item.chmod(0o777 if item.is_dir() else 0o666)
        public = directory / 'inputs'
        public.mkdir(mode=0o755)
        for name, source in self.inputs.items():
            if Path(name).name != name or not source.is_file():
                raise ValueError('Input handle must name an authorized file')
            shutil.copyfile(source, public / name)
            (public / name).chmod(0o444)
        return directory

    def publish(self, stage, declared, deleted=()):
        stage = Path(stage).resolve(strict=True)
        if stage.parent != self.root or not stage.name.startswith('stage-'):
            raise ValueError('Not an owned staging directory')
        if any(path.is_symlink() for path in stage.rglob('*')):
            raise ValueError('Output symlinks are not publishable')
        declared, deleted = list(declared), list(deleted)
        if set(declared) & set(deleted): raise ValueError('Conflicting output declarations')
        for relative in declared + deleted:
            if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or '..' in Path(relative).parts or Path(relative).parts[0] == 'inputs':
                raise ValueError('Output path must be relative and outside inputs')
        pointer = self.root / 'manifest.json'
        previous = json.loads(pointer.read_text()) if pointer.exists() else None
        inherited = []
        for relative in (previous or {}).get('outputs', []):
            old = self.root / previous['version'] / relative
            current = stage / relative
            if relative in deleted:
                if current.exists(): raise ValueError('Deleted output still exists')
            elif relative not in declared:
                if not current.is_file() or current.read_bytes() != old.read_bytes():
                    raise ValueError('undeclared_output_change: ' + relative)
                inherited.append(relative)
        for relative in deleted:
            if relative not in (previous or {}).get('outputs', []):
                raise ValueError('Cannot delete an unregistered output')
        for relative in declared:
            candidate = stage / relative
            resolved = candidate.resolve(strict=True)
            if not resolved.is_relative_to(stage) or not resolved.is_file():
                raise ValueError('Declared output escapes stage or is not a file')
            if any(part.is_symlink() for part in [candidate, *candidate.parents] if part.is_relative_to(stage)):
                raise ValueError('Output symlinks are not publishable')
        hashes = {p.relative_to(stage).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(stage.rglob('*')) if p.is_file() and 'inputs' not in p.relative_to(stage).parts[:1]}
        outputs = sorted(set(inherited + declared))
        from .contracts import digest
        content_hash = digest({'files': hashes, 'outputs': outputs})
        if previous and previous.get('content_hash') == content_hash:
            self.discard(stage)
            return previous
        version = 'version-' + uuid4().hex
        os.rename(stage, self.root / version)
        for path in (self.root / version).rglob('*'):
            path.chmod(0o555 if path.is_dir() else 0o444)
        manifest = {'version': version, 'outputs': outputs, 'hashes': hashes, 'content_hash': content_hash}
        temporary = self.root / ('manifest-' + uuid4().hex + '.tmp')
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(manifest, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.root / 'manifest.json')
        return manifest

    def discard(self, stage):
        stage = Path(stage).resolve()
        if stage.parent != self.root or not stage.name.startswith('stage-'):
            raise ValueError('Not an owned staging directory')
        if stage.exists(): shutil.rmtree(stage)
