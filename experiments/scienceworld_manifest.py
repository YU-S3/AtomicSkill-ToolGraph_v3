"""Official-split, hash-ranked manifests. Selection never consults task score."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from atomic_skillgraph.core.refs import content_hash
from atomic_skillgraph.core.serialization import atomic_write_json
from atomic_skillgraph.harness.protocol import HarnessTask
from atomic_skillgraph.harness.scienceworld_resource import resource_contract

ALGORITHM = 'scienceworld_official_split_hashrank_v1'

def rank(split, task_type, variation):
    return hashlib.sha256(f'scienceworld|{split}|42|{task_type}|{variation}'.encode()).hexdigest()

def build(output):
    from scienceworld import ScienceWorldEnv
    output = Path(output)
    env = ScienceWorldEnv(envStepLimit=100)
    resource = resource_contract()
    candidates, memberships = {s: [] for s in ('train', 'dev', 'test')}, {}
    try:
        for task_type, name in env.tasks.items():
            env.load(name, 0, 'easy', generateGoldPath=False)
            memberships[task_type] = {}
            for split in candidates:
                variations = sorted(getattr(env, f'get_variations_{split}')())
                memberships[task_type][split] = variations
                for variation in variations:
                    entry = {'benchmark': 'scienceworld', 'macro_type': task_type.split('-')[0],
                        'task_type': task_type, 'task_name': name, 'variation_idx': variation,
                        'source_split': split, 'resource_identity': resource,
                        'task_id': f'scienceworld_{split}_{task_type}_{name}_var{variation:04d}'}
                    entry['task_signature'] = content_hash(entry)
                    candidates[split].append(entry)
        for split, count in [('train', 4), ('test', 3), ('dev', 1)]:
            group_key = 'macro_type' if split == 'dev' else 'task_type'
            selected = []
            for group in sorted({e[group_key] for e in candidates[split]}, key=lambda x: tuple(map(int, x.split('-')))):
                entries = sorted([e for e in candidates[split] if e[group_key] == group],
                    key=lambda e: rank(split, e['task_type'], e['variation_idx']))
                if len(entries) < count:
                    raise ValueError(f'Insufficient official {split} variations for {group}: {len(entries)} < {count}')
                selected.extend(entries[:count])
            for index, entry in enumerate(selected):
                entry['index'] = index
            payload = {'schema_version': 2, 'benchmark': 'scienceworld', 'selection_algorithm': ALGORITHM,
                'selection_seed': 42, 'split': split, 'tasks': selected,
                'official_membership': memberships, 'resource_identity': resource}
            payload['digest'] = content_hash(payload)
            target = output / f'{split}_{len(selected)}.json'
            if target.exists() and json.loads(target.read_text()) != payload:
                raise FileExistsError(f'Manifest differs; refusing overwrite: {target}')
            atomic_write_json(target, payload)
            print(json.dumps({'manifest': str(target), 'count': len(selected), 'digest': payload['digest']}), flush=True)
    finally:
        env.close()

def load(path):
    value = json.loads(Path(path).read_text())
    if value['digest'] != content_hash({k:v for k,v in value.items() if k != 'digest'}):
        raise ValueError('Manifest digest mismatch')
    if (value['schema_version'], value['benchmark'], value['selection_algorithm'], value['selection_seed']) != (2, 'scienceworld', ALGORITHM, 42):
        raise ValueError('Unsupported manifest protocol')
    seen, counts = set(), {}
    for index, row in enumerate(value['tasks']):
        identity = row['task_name'], row['variation_idx']
        if identity in seen or row['variation_idx'] not in value['official_membership'][row['task_type']][value['split']]:
            raise ValueError('Duplicate or nonmember variation')
        seen.add(identity)
        if row['index'] != index or row['source_split'] != value['split'] or row['resource_identity'] != value['resource_identity']:
            raise ValueError('Manifest entry resource/order mismatch')
        if row['task_signature'] != content_hash({k:v for k,v in row.items() if k not in {'index', 'task_signature'}}):
            raise ValueError('Manifest task signature mismatch')
        group = row['macro_type'] if value['split'] == 'dev' else row['task_type']
        counts[group] = counts.get(group, 0) + 1
    expected_groups, expected_count = {'dev': (10, 1), 'train': (30, 4), 'test': (30, 3)}[value['split']]
    if len(counts) != expected_groups or set(counts.values()) != {expected_count}:
        raise ValueError('Unbalanced or incomplete official manifest')
    for groups in value['official_membership'].values():
        splits = [set(groups[s]) for s in ('train', 'dev', 'test')]
        if any(splits[i] & splits[j] for i in range(3) for j in range(i)):
            raise ValueError('Official split overlap')
    return value

def task_from_entry(entry, goal):
    return HarnessTask(entry['task_id'], goal, 'scienceworld', entry['task_type'],
        context={k: entry[k] for k in ('task_name', 'variation_idx', 'source_split')}, metadata=dict(entry))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/scienceworld_manifests')
    build(parser.parse_args().output)
