"""One fresh 12-Train/6-valid_seen pilot, selected before any model request."""
import argparse
import hashlib
import json
from pathlib import Path

from atomic_skillgraph.core.serialization import atomic_create_json
from experiments.run_v3_r103_validation import train_dev16, deploy
from experiments.protocol import ALFWORLD_FORMAL_TASK_TYPES

REPO = Path(__file__).resolve().parents[1]


def select(path, split, per_family):
    path = Path(path)
    raw = path.read_bytes()
    manifest = json.loads(raw)
    counts = dict.fromkeys(ALFWORLD_FORMAL_TASK_TYPES, 0)
    selected = []
    ids = set()
    for row in manifest['tasks']:
        if row['task_id'] in ids or row['source_split'] != split:
            raise ValueError('source manifest identity/split mismatch')
        ids.add(row['task_id'])
        family = row['task_type']
        if family not in counts:
            raise ValueError('unexpected task family in source manifest')
        if counts[family] < per_family:
            selected.append(row)
            counts[family] += 1
    if any(n != per_family for n in counts.values()):
        raise ValueError('source manifest cannot supply the fixed pilot')
    return selected, {'source_path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest(),
                      'source_manifest_id': manifest['manifest_id'], 'counts': counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/alfworld_train_full_120_r103_seed42.yaml')
    parser.add_argument('--train-manifest', default=str(REPO / 'data/baseline_manifests/train_120.json'))
    parser.add_argument('--val-manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    train, train_source = select(args.train_manifest, 'train', 2)
    val, val_source = select(args.val_manifest, 'valid_seen', 1)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    selection = {'protocol': 'skillcompiler.mechanism-pilot.v1', 'train': train, 'valid_seen': val,
                 'sources': {'train': train_source, 'valid_seen': val_source},
                 'formal_score': False, 'test_tuning': False}
    from atomic_skillgraph.agents.node_context import VERSION as node_version
    from atomic_skillgraph.evolution.extraction_view import VERSION as extraction_version
    from atomic_skillgraph.evolution.realization_queue import VERSION as realization_version
    from atomic_skillgraph.runtime.scope_diagnostics import OUTCOME_VERSION
    selection['protocol_versions'] = {'node_context': node_version, 'e1': extraction_version,
        'realization': realization_version, 'search_outcome': OUTCOME_VERSION}
    atomic_create_json(output / 'selection.json', selection)
    training = train_dev16(args.config, output / 'train12', fixed_entries=train,
                          selection_identity=selection['sources'])
    validation = deploy(args.config, output / 'valid6', output / 'train12', 'C11', validation_entries=val)
    atomic_create_json(output / 'pilot_result.json', {'completed': True, 'selection': selection,
        'train': training, 'valid_seen': validation})


if __name__ == '__main__':
    main()
