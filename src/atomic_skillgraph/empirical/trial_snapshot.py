"""Persist public trial artifacts before an isolated Adapter is cleaned."""
import hashlib
import json
from pathlib import Path
import shutil

from .contracts import digest
from ..core.errors import AtomicSkillGraphError, BudgetExhausted, FailureLayer
from .program_submission import ProgramContractError


def host_call(operation, function, *args, trial_context=None, **kwargs):
    """Classify faults only at an explicit host operation boundary."""
    try:
        return function(*args, **kwargs)
    except (BudgetExhausted, ProgramContractError):
        raise
    except AtomicSkillGraphError as exc:
        if exc.layer == FailureLayer.INFRASTRUCTURE and not hasattr(exc, 'trial_context'):
            exc.trial_context = {**(trial_context or {}), 'operation': operation,
                                 'cause_type': type(exc.__cause__ or exc).__name__}
        raise
    except Exception as exc:
        failure = AtomicSkillGraphError('infrastructure_failure', str(exc), layer=FailureLayer.INFRASTRUCTURE)
        failure.trial_context = {**(trial_context or {}), 'operation': operation, 'cause_type': type(exc).__name__}
        raise failure from exc


def seal_trial_workspace(adapter, destination):
    workspace = getattr(adapter, 'workspace', None)
    if not workspace: return None
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    pointer = workspace.root / 'manifest.json'
    manifest = json.loads(pointer.read_text()) if pointer.exists() else {}
    if manifest:
        shutil.copytree(workspace.root / manifest['version'], destination / manifest['version'])
        shutil.copyfile(pointer, destination / 'manifest.json')
    hashes = {p.relative_to(destination).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(destination.rglob('*')) if p.is_file()}
    receipt = {'root': str(destination), 'files_sha256': hashes, 'sha256': digest(hashes), 'manifest': manifest}
    (destination / 'snapshot.json').write_text(json.dumps(receipt))
    return receipt


def restore_trial_workspace(adapter, receipt):
    root = Path(receipt['root'])
    for name, expected in receipt['files_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Trial public artifact hash mismatch')
    if digest(receipt['files_sha256']) != receipt['sha256']:
        raise ValueError('Trial artifact manifest hash mismatch')
    workspace = getattr(adapter, 'workspace', None)
    if not workspace: raise ValueError('Trial requires a reconstructible public workspace')
    manifest = receipt['manifest']
    if manifest:
        shutil.copytree(root/manifest['version'], workspace.root/manifest['version'])
        shutil.copyfile(root/'manifest.json', workspace.root/'manifest.json')


def exception_details(exc, stage):
    context = getattr(exc, 'trial_context', {})
    infrastructure = (exc.layer == FailureLayer.INFRASTRUCTURE if isinstance(exc, AtomicSkillGraphError)
                      else not isinstance(exc, (ValueError, SyntaxError))) and not isinstance(exc, BudgetExhausted)
    return {'code': getattr(exc, 'code', type(exc).__name__), 'message': str(exc),
            'stage': stage, 'repair_target': 'host' if infrastructure else None,
            'infrastructure_error': infrastructure, 'outer_stage': stage, **context}
