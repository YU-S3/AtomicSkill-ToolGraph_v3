"""Deterministic base-plus-delta compilation; no Runtime task selection rules."""
import copy
import json
from pathlib import Path

from ..core.serialization import atomic_write_json
from .release_protocol import ReleaseError, sha

OPERATIONS = {
    'version_existing_discovery_contract': 'atomic',
    'version_existing_bounded_search_implementation': 'program_realization',
    'version_existing_workflow': 'composite',
    'version_existing_composite': 'composite',
}


def compile_plan(base_path, delta_path, output):
    base_path, delta_path, output = map(Path, (base_path, delta_path, output))
    if output.exists():
        raise FileExistsError(output)
    base, delta = (json.loads(p.read_text(encoding='utf-8')) for p in (base_path, delta_path))
    if delta.get('schema') == 'r103.release5.graph-delta.v1':
        if base['seed'] != delta['seed'] or delta['operation'] != 'version_existing_composite':
            raise ReleaseError('graph delta identity mismatch')
        if delta['source_ref'].split('@')[0] != delta['target_ref'].split('@')[0]:
            raise ReleaseError('graph delta may not change logical identity')
        sources = {j['target_ref']: j for j in workflow_jobs(base)}
        if delta['source_ref'] not in sources or delta['target_ref'] in sources:
            raise ReleaseError('graph delta source/target not available')
        result = copy.deepcopy(base)
        job = {**copy.deepcopy(sources[delta['source_ref']]), **copy.deepcopy(delta),
            'job_id': 'release5_existing_graph_revision', 'kind': 'composite', 'action': 'revise'}
        result.setdefault('derived_revision_jobs', []).append(job)
        result.setdefault('deployment_preferences_patch', {}).setdefault('additional_preferred_workflow_refs', []).append(delta['target_ref'])
        result['graph_delta_provenance'] = {'base_sha256': sha(base_path), 'delta_sha256': sha(delta_path)}
        atomic_write_json(output, result)
        return result
    if delta.get('schema') != 'r103.oldfirst-preparation-delta.v1' or base['seed'] != delta['seed']:
        raise ReleaseError('base/delta identity mismatch')
    jobs = copy.deepcopy(delta['derived_revision_jobs'])
    if len({j['job_id'] for j in jobs}) != len(jobs):
        raise ReleaseError('duplicate derived job')
    available = {a['ref'] for a in base['asset_dispositions']}
    for key in ('atomic_revision_jobs', 'new_atomic_jobs', 'workflow_targets'):
        available.update(j['target_ref'] for j in base[key])
    for job in base['program_realization_jobs']:
        available.update(job[k] for k in ('tool_ref', 'implementation_ref') if job.get(k))
    ordered = []
    while jobs:
        advanced = False
        for job in list(jobs):
            if OPERATIONS.get(job.get('operation')) != job.get('kind'):
                raise ReleaseError(f'unknown or mismatched derived operation: {job.get("operation")}')
            dependencies = {job[k] for k in ('source_ref', 'source_atomic_ref', 'source_tool_ref',
                'source_implementation_ref', 'atomic_ref') if job.get(k)}
            dependencies.update(n['atomic_ref'] for n in job.get('target_nodes', []))
            if not dependencies <= available:
                continue
            targets = {job[k] for k in ('target_ref', 'tool_ref', 'implementation_ref') if job.get(k)}
            if targets & available:
                raise ReleaseError('derived revision collides with existing asset')
            for source_key, target_key in (('source_ref', 'target_ref'), ('source_tool_ref', 'tool_ref'),
                                           ('source_implementation_ref', 'implementation_ref')):
                if job.get(source_key) and job.get(target_key) and job[source_key].split('@')[0] != job[target_key].split('@')[0]:
                    raise ReleaseError('derived revision may not introduce a logical ID')
            ordered.append(job); available.update(targets); jobs.remove(job); advanced = True
        if not advanced:
            raise ReleaseError('derived job dependency cycle or missing reference')
    if not set(delta['preserve_base_publications']) <= set(base['manual_publications']):
        raise ReleaseError('delta drops a required base publication')
    result = {**copy.deepcopy(base), 'derived_revision_jobs': ordered,
        'required_protocols': delta['required_protocols'],
        'deployment_preferences_patch': delta['deployment_preferences_patch'],
        'preparation_delta_provenance': {'base_sha256': sha(base_path), 'delta_sha256': sha(delta_path),
            'baseline_commit': delta['baseline_commit']}}
    atomic_write_json(output, result)
    return result


def program_jobs(plan):
    return [*plan['program_realization_jobs'],
            *(j for j in plan.get('derived_revision_jobs', []) if j['kind'] == 'program_realization')]


def workflow_jobs(plan):
    jobs = copy.deepcopy(plan['workflow_targets'])
    for revision in plan.get('derived_revision_jobs', []):
        if revision['kind'] == 'composite':
            source = next((j for j in jobs if j['target_ref'] == revision['source_ref']), None)
            if source is None:
                raise ReleaseError('workflow delta source is not a base workflow')
            source.update(revision, action='revise')
    return jobs
