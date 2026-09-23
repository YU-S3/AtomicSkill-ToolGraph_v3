"""Rebuild archived plans, add one proved graph version, compare every old ref."""
import argparse
import copy
import json
import shutil
import sqlite3
from pathlib import Path
from atomic_skillgraph.core.serialization import atomic_create_json
from atomic_skillgraph.deployment.release_protocol import ReleaseSpec, ReleaseError, sha
from atomic_skillgraph.deployment.oldfirst_plan import compile_plan
from atomic_skillgraph.deployment.oldfirst_release import prepare_oldfirst, verify_oldfirst
from atomic_skillgraph.deployment.bank_release import freeze_release, _bank_files


def read(path):
    return json.loads(Path(path).read_text())


def inventory(bank):
    with sqlite3.connect(f'file:{bank / "state.sqlite3"}?mode=ro', uri=True) as db:
        return {ref: {'status': status, 'payload': read(path), 'sha256': sha(path)}
            for ref, status, path in db.execute('SELECT artifact_ref,status,file_path FROM artifact_index')}


def graph_proof(bank, delta):
    assets = inventory(bank)
    source = assets[delta['source_ref']]
    if source['sha256'] != delta['source_asset_sha256']:
        raise ReleaseError('source graph bytes differ from reviewed delta')
    nodes = delta['target_nodes']
    original = source['payload']
    if [o['node_ref'] for o in original['occurrences']] != [
            assets[n['atomic_ref']]['payload']['ref'] for n in nodes]:
        raise ReleaseError('graph revision changes original Atomic sequence')
    # Prove the exact two transitive identity rewrites from the old contracts.
    by_step = {n['step_id']: assets[n['atomic_ref']]['payload'] for n in nodes}
    proofs = []
    for step, output, role in [('station_open', 'container', 'container'),
                               ('station_nav', 'location', 'destination'), ('cool', 'object', 'object')]:
        atomic = by_step[step]
        derivation = atomic['validator_spec']['output_derivations'][output]
        if derivation != {'kind': 'input_identity', 'input_role': role}:
            raise ReleaseError('identity rewrite lacks exact original declaration')
        inp = next(p for p in atomic['inputs'] if p['name'] == role)
        out = next(p for p in atomic['outputs'] if p['name'] == output)
        if inp['semantic_type'] != out['semantic_type']:
            raise ReleaseError('identity rewrite changes role types')
        proofs.append({'step': step, 'atomic_ref': nodes[[n['step_id'] for n in nodes].index(step)]['atomic_ref'],
                       'output': output, 'input': role, 'derivation': derivation})
    old_nodes = {o['step_id']: o for o in original['occurrences']}
    old_sequence = original['control_sequence']
    expected_edges = [(old_sequence[1], 'location', old_sequence[2], 'container'),
        (old_sequence[2], 'container', old_sequence[3], 'station'),
        (old_sequence[0], 'object', old_sequence[3], 'object'),
        (old_sequence[0], 'object', old_sequence[4], 'object')]
    for source_step, source_role, target_step, target_role in expected_edges:
        b = old_nodes[target_step]['binding_specs'][target_role]
        if (b['kind'], b['source_step'], b['source_role']) != ('data_flow', source_step, source_role):
            raise ReleaseError('source path differs from reviewed identity rewrite')
    return {'passed': True, 'source_ref': delta['source_ref'], 'target_ref': delta['target_ref'],
            'source_sha256': source['sha256'], 'identity_proofs': proofs, 'old_paths': expected_edges}


def prepare(previous, output, graph_delta):
    previous, output, graph_delta = map(lambda p: Path(p).resolve(), (previous, output, graph_delta))
    if output.exists():
        raise FileExistsError(output)
    delta = read(graph_delta)
    # Archive validation only: no bypass of the runtime code guard.
    old_assets = {}
    for seed in (42, 43, 44):
        root = previous/f'seed{seed}'
        manifest = read(root/'frozen/release_manifest.json')
        if manifest['files'] != _bank_files(root/'frozen') or sha(root/'source/input.zip') != manifest['source_zip_hash']:
            raise ReleaseError('old archive or frozen inventory changed')
        old_assets[seed] = inventory(root/'frozen')
    proof = graph_proof(previous/'seed42/frozen', delta)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(previous/'prior_dev_entries.json', output/'prior_dev_entries.json')
    atomic_create_json(output/'seed42_graph_revision_audit.json', proof)
    from .native_contract_audit import audit_bank, historical_regression
    atomic_create_json(output/'native_protocol_regression.json', historical_regression(
        Path(__file__).resolve().parents[1]/'tests/fixtures/release5/schema_failure_evidence.json'))
    audits = []
    for seed in (42, 43, 44):
        old = previous/f'seed{seed}'
        plan_path = output/'plans'/f'seed{seed}.json'
        if seed == 42:
            compile_plan(old/'frozen/edit_plan.lock.json', graph_delta, plan_path)
        else:
            atomic_create_json(plan_path, read(old/'frozen/edit_plan.lock.json'))
        config = copy.deepcopy(read(old/'prepare_manifest.json')['config'])
        config.pop('_config_path', None)
        atomic_create_json(output/'configs'/f'seed{seed}_base.json', config)
        prepared = prepare_oldfirst(ReleaseSpec(seed, old/'source/input.zip', output/f'seed{seed}', config), plan_path)
        bank = prepared.root/'work/data_v3'
        new = inventory(bank)
        changed = [ref for ref, row in old_assets[seed].items() if new.get(ref) != row]
        added = sorted(new.keys() - old_assets[seed].keys())
        allowed = [delta['target_ref']] if seed == 42 else []
        if changed or added != allowed:
            raise ReleaseError(f'unexpected asset changes: seed{seed}: changed={changed}, added={added}')
        if seed == 42 and new[delta['target_ref']]['payload']['goal_contract'] != old_assets[seed][delta['source_ref']]['payload']['goal_contract']:
            raise ReleaseError('new graph changed goal contract')
        atomic_create_json(prepared.root/'reseal_comparison.json', {'passed': True, 'unchanged_refs': sorted(old_assets[seed]),
            'added_refs': added, 'changed_refs': changed, 'asset_count': len(new), 'source_root': str(old)})
        audit = audit_bank(bank)
        atomic_create_json(bank/'native_call_contract.json', audit)
        audits.append({'seed': seed, **audit})
        checks = verify_oldfirst(prepared)
        freeze_release(prepared, checks)
        print(json.dumps({'seed': seed, 'stage': 'frozen', 'assets': len(new)}), flush=True)
    atomic_create_json(output/'native_contract_audit.json', {'passed': True, 'seeds': audits})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-release-root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--graph-delta', required=True)
    args = parser.parse_args()
    prepare(args.source_release_root, args.out, args.graph_delta)
