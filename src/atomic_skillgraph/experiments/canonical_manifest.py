"""Generate the public split authority once; materialization never resplits it."""
import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import random

from .prepare_benchmarks import upstream_ids
from .run_empirical import write_json
from ..empirical.contracts import digest


BENCHMARKS = ('searchqa', 'spreadsheetbench', 'officeqa', 'docvqa', 'livemath', 'alfworld')
ADAPTER_NAMES = {'spreadsheetbench': 'spreadsheet'}
COUNTS = {'searchqa': (300, 24, 1400, 276), 'spreadsheetbench': (200, 20, 180, 0),
          'officeqa': (120, 24, 102, 0), 'docvqa': (180, 22, 201, 131),
          'livemath': (60, 17, 100, 0), 'alfworld': (120, 24, 134, 0)}
SPLITS = ('train', 'val', 'test', 'reserve')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ordered_train(tasks, run_seed):
    if run_seed not in (42, 43, 44):
        raise ValueError('Formal run_seed must be 42, 43 or 44')
    tasks = list(tasks)
    random.Random(run_seed).shuffle(tasks)
    return tasks


def frozen_write(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Frozen authority differs: ' + str(path))
        return
    write_json(path, value)


def rank(prefix, value):
    return hashlib.sha256((prefix + str(value)).encode()).hexdigest()


def quota_split(rows, field, quotas, benchmark):
    groups = {name: [] for name in SPLITS}
    if set(quotas) != {row['task_metadata'][field] for row in rows}:
        raise ValueError('Unexpected stratum')
    for label, sizes in quotas.items():
        pool = sorted((r for r in rows if r['task_metadata'][field] == label),
                      key=lambda r: rank(f'{benchmark}|{label}|42|', r['source_id']))
        if len(pool) != sum(sizes):
            raise ValueError('Stratum size mismatch: ' + label)
        offset = 0
        for name, size in zip(SPLITS, sizes):
            groups[name].extend(pool[offset:offset + size])
            offset += size
    return groups


def multilabel_split(rows, label_field, stages):
    import numpy as np
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    labels = sorted({label for r in rows for label in r['task_metadata'][label_field]})
    matrix = np.array([[int(label in r['task_metadata'][label_field]) for label in labels] for r in rows])
    remaining, groups = list(range(len(rows))), {}
    for name, target, seed in stages:
        splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=target, random_state=seed)
        rest, chosen = next(splitter.split(np.zeros((len(remaining), 1)), matrix[remaining]))
        groups[name] = [rows[remaining[i]] for i in chosen]
        remaining = [remaining[i] for i in rest]
    groups['train'] = [rows[i] for i in remaining]
    return {name: groups.get(name, []) for name in SPLITS}


def entry(benchmark, source_id, metadata=None, **identity):
    stable_id = rank('', source_id) if benchmark == 'alfworld' else str(source_id)
    return {'task_id': ADAPTER_NAMES.get(benchmark, benchmark) + ':' + stable_id,
            'source_id': str(source_id), 'task_metadata': metadata or {}, **identity}


def generate(resources, output, alfworld_data, legacy_manifests):
    import iterstrat
    import numpy as np
    resources, output = Path(resources), Path(output)
    upstream = resources / 'upstream/SkillOpt-fa4ca184573e42ec11472959dd57422381418096'
    source_paths = []
    datasets, rules = {}, {}
    search = upstream_ids(upstream, 'searchqa')
    source_paths += list((upstream / 'data/searchqa_id_split').glob('*/items.json'))
    train = sorted(search['train'], key=lambda r: rank('searchqa|train|42|', r['id']))
    val = sorted(search['val'], key=lambda r: rank('searchqa|val|42|', r['id']))
    datasets['searchqa'] = {name: [entry('searchqa', r['id']) for r in rows] for name, rows in {
        'train': train[:300], 'val': val[:24], 'test': search['test'], 'reserve': train[300:] + val[24:]}.items()}
    rules['searchqa'] = {'train_rank': 'SHA256(searchqa|train|42|id)',
                         'val_rank': 'SHA256(searchqa|val|42|id)', 'test': 'original SkillOpt order and IDs'}
    sheet = resources / 'extracted/spreadsheetbench_verified_400/dataset.json'
    source_paths.append(sheet)
    rows = [entry('spreadsheetbench', r['id'], {'instruction_type': r['instruction_type']})
            for r in json.loads(sheet.read_text())]
    quotas = {'Cell-Level Manipulation': (138, 14, 123, 0), 'Sheet-Level Manipulation': (62, 6, 57, 0)}
    datasets['spreadsheetbench'] = quota_split(rows, 'instruction_type', quotas, 'spreadsheet')
    rules['spreadsheetbench'] = {'rank': 'SHA256(spreadsheet|instruction_type|42|id)', 'quotas': quotas}
    office = resources / 'raw/officeqa/officeqa_full.csv'
    source_paths.append(office)
    with office.open(encoding='utf-8-sig') as stream:
        rows = [entry('officeqa', r['uid'], {'difficulty': r['difficulty']}) for r in csv.DictReader(stream)]
    quotas = {'easy': (55, 11, 47, 0), 'hard': (65, 13, 55, 0)}
    datasets['officeqa'] = quota_split(rows, 'difficulty', quotas, 'officeqa')
    rules['officeqa'] = {'category': 'difficulty', 'rank': 'SHA256(officeqa|difficulty|42|uid)', 'quotas': quotas}
    documents = upstream_ids(upstream, 'docvqa')
    source_paths += list((upstream / 'data/docvqa_id_split').glob('*/items.json'))
    unique = {str(r['questionId']): r for rows in documents.values() for r in rows}
    rows = [entry('docvqa', key, {'topic': unique[key]['topic'].split('|'), 'doc_id': str(unique[key]['docId'])})
            for key in sorted(unique, key=int)]
    stages = [('reserve', 130, 42), ('test', 200, 43), ('val', 24, 44)]
    datasets['docvqa'] = multilabel_split(rows, 'topic', stages)
    rules['docvqa'] = {'order': 'numeric questionId', 'labels': 'topic', 'stages': stages,
                       'quantity_correction': False, 'accepted_actual_counts': [180, 22, 201, 131]}
    rows = []
    paths = sorted((resources / 'raw/livemath/data').glob('*/*.json'))
    source_paths += paths
    for path in paths:
        for r in json.loads(path.read_text()):
            labels = r.get('theorem_type') or ['unknown']
            if isinstance(labels, str):
                labels = [labels]
            rows.append(entry('livemath', str(r['month']) + ':' + str(r['no']),
                              {'month': str(r['month']), 'theorem_type': labels}))
    rows.sort(key=lambda r: r['source_id'])
    stages = [('test', 100, 42), ('val', 17, 43)]
    datasets['livemath'] = multilabel_split(rows, 'theorem_type', stages)
    rules['livemath'] = {'order': 'lexicographic month:no', 'labels': 'theorem_type only', 'stages': stages,
                         'month': 'metadata only', 'quantity_correction': False}
    from ..harness.alfworld import AlfWorldAdapter, TASK_TYPE_IDS
    import alfworld.agents.environment as alf_env
    harness = AlfWorldAdapter(split='train', alfworld_data=str(alfworld_data))
    config = harness._build_config()
    config['dataset']['data_path'] = str(Path(alfworld_data) / 'json_2.1.1/train')
    native = alf_env.get_environment('AlfredTWEnv')(config, train_eval='train')
    alfworld_data = Path(alfworld_data).resolve()
    rows = []
    for file in native.game_files:
        file = Path(file).resolve()
        relative = file.relative_to(alfworld_data).as_posix()
        task_type = json.loads(file.with_name('traj_data.json').read_text())['task_type']
        rows.append(entry('alfworld', relative, {'task_type': task_type}, source_split='train',
                          gamefile_rel=relative, gamefile_sha256=sha256(file), task_type=task_type,
                          env_index=None, task_signature=None))
    selected = []
    for task_type in sorted(TASK_TYPE_IDS):
        pool = sorted((r for r in rows if r['task_type'] == task_type),
                      key=lambda r: rank('alfworld|train|42|', r['gamefile_rel']))
        if len(pool) < 20:
            raise ValueError('Insufficient official Train tasks')
        selected.extend(pool[:20])
    datasets['alfworld'] = {'train': selected, 'reserve': []}
    for name, filename in [('val', 'validation_24.json'), ('test', 'test_ood_full_134.json')]:
        path = Path(legacy_manifests) / filename
        source_paths.append(path)
        datasets['alfworld'][name] = [entry('alfworld', r['gamefile_rel'], {'task_type': r['task_type']},
            **{k: v for k, v in r.items() if k not in {'task_id', 'index'}}) for r in json.loads(path.read_text())['tasks']]
    rules['alfworld'] = {'train_rank': 'SHA256(alfworld|train|42|gamefile_rel)', 'train_per_type': 20,
                        'universe': 'official AlfredTWEnv collect_game_files',
                        'val_test': 'preserve existing physical members and order, explicit user decision',
                        'unused_official_games': 'outside experiment; not materialized'}
    sources = {str(p.relative_to(resources)) if p.is_relative_to(resources) else 'legacy/' + p.name: sha256(p)
               for p in sorted(source_paths)}
    for benchmark, groups in datasets.items():
        seen = set()
        for name, expected in zip(SPLITS, COUNTS[benchmark]):
            ids = [r['source_id'] for r in groups[name]]
            if len(ids) != expected or len(set(ids)) != len(ids) or seen.intersection(ids):
                raise ValueError('Canonical count/disjointness mismatch: ' + benchmark + '/' + name)
            seen.update(ids)
    manifest = {'schema': 'public.main_experiment.v1', 'authority': 'main_experiment_v1',
                'split_seed': 42, 'run_seeds': [42, 43, 44], 'upstream': 'SkillOpt@fa4ca184573e42ec11472959dd57422381418096',
                'libraries': {'numpy': np.__version__, 'iterative-stratification': iterstrat.__version__},
                'sources_sha256': sources, 'benchmarks': {}}
    for benchmark, groups in datasets.items():
        splits = {}
        seen = set()
        for name, expected in zip(SPLITS, COUNTS[benchmark]):
            rows = groups[name]
            ids = [r['source_id'] for r in rows]
            if len(rows) != expected or len(set(ids)) != len(ids) or seen.intersection(ids):
                raise ValueError('Canonical count/disjointness mismatch: ' + benchmark + '/' + name)
            seen.update(ids)
            path = output / benchmark / (name + '.json')
            frozen_write(path, {'schema': 'public.main_experiment.split.v1', 'benchmark': benchmark,
                               'split': name, 'split_seed': 42, 'tasks': rows})
            splits[name] = {'count': len(rows), 'path': benchmark + '/' + name + '.json', 'sha256': sha256(path)}
        manifest['benchmarks'][benchmark] = {'rules': rules[benchmark], 'splits': splits}
    frozen_write(output / 'manifest.json', manifest)
    return verify(output)


def verify(authority):
    authority = Path(authority)
    manifest = json.loads((authority / 'manifest.json').read_text())
    if manifest['authority'] != 'main_experiment_v1' or manifest['split_seed'] != 42:
        raise ValueError('Unsupported public authority')
    for benchmark in BENCHMARKS:
        seen = set()
        for name, expected in zip(SPLITS, COUNTS[benchmark]):
            lock = manifest['benchmarks'][benchmark]['splits'][name]
            path = authority / lock['path']
            if not path.resolve().is_relative_to(authority.resolve()) or sha256(path) != lock['sha256']:
                raise ValueError('Public split hash/path mismatch')
            rows = json.loads(path.read_text())['tasks']
            ids = [r['source_id'] for r in rows]
            if len(ids) != expected or lock['count'] != expected or len(set(ids)) != len(ids) or seen.intersection(ids):
                raise ValueError('Public split membership mismatch')
            seen.update(ids)
    return manifest


def materialize(authority, prepared_pool, output):
    authority, prepared_pool, output = Path(authority), Path(prepared_pool), Path(output)
    manifest = verify(authority)
    from skillcompiler_bench_contracts.livemath import (permute_livemath_choices, NORMALIZATION_VERSION,
        UPSTREAM_REVISION, CHOICE_PROJECTION_VERSION, CHOICE_SEED)
    lock = json.loads((prepared_pool / 'dataset_lock.json').read_text())
    if lock.get('livemath_normalization_version') != NORMALIZATION_VERSION:
        raise ValueError('CF4 requires a freshly normalized LiveMath resource pool')
    integrity = json.loads((prepared_pool / 'livemath_integrity.json').read_text())
    if lock.get('livemath_choice_projection_version') != CHOICE_PROJECTION_VERSION or lock.get('livemath_choice_seed') != CHOICE_SEED:
        raise ValueError('LiveMath choice projection lock differs')
    if integrity.get('choice_projection_version') != CHOICE_PROJECTION_VERSION or integrity.get('choice_seed') != CHOICE_SEED:
        raise ValueError('LiveMath choice integrity differs')
    integrity.update(split_counts={}, public_choices_sha256={}, evaluator_choices_sha256={})
    external = {}
    for benchmark in BENCHMARKS:
        internal = ADAPTER_NAMES.get(benchmark, benchmark)
        records = {} if benchmark == 'alfworld' else json.loads((prepared_pool / internal / 'evaluator_records.json').read_text())
        pool = {}
        if benchmark != 'alfworld':
            source_files = ('all',) if (prepared_pool / internal / 'all.json').exists() else SPLITS
            for name in source_files:
                path = prepared_pool / internal / (name + '.json')
                if not path.exists():
                    continue
                for row in json.loads(path.read_text())['tasks']:
                    if row['task_id'] in pool:
                        raise ValueError('Duplicate resource-pool task')
                    pool[row['task_id']] = row
            canonical_ids = {r['task_id'] for name in SPLITS for r in json.loads(
                (authority / benchmark / (name + '.json')).read_text())['tasks']}
            if set(pool) != canonical_ids or set(records) != canonical_ids:
                raise ValueError('Resource pool does not cover the canonical universe exactly')
            for task_id, task in pool.items():
                for filename in [*task['inputs'].get('images', []), *records[task_id].get('public_files', {}).values()]:
                    external[str(Path(filename).resolve())] = sha256(filename)
            if benchmark == 'livemath':
                for task_id, task in pool.items():
                    record = records[task_id]
                    projection = record.get('choice_projection')
                    if not projection: raise ValueError('LiveMath lacks canonical projection provenance')
                    normalized = permute_livemath_choices(projection['canonical'], choice_seed=CHOICE_SEED,
                        stable_item_id=task_id.removeprefix('livemath:'))
                    if normalized['choice_projection'] != projection:
                        raise ValueError('LiveMath projection identity differs: ' + task_id)
                    if normalized['choices'] != task['inputs']['choices'] or record['choices'] != normalized['choices']:
                        raise ValueError('LiveMath public/evaluator choices differ: ' + task_id)
                    if record['correct_choice'] != normalized['correct_choice'] or task['goal'] != normalized['question']:
                        raise ValueError('LiveMath question/gold projection differs: ' + task_id)
                    integrity['public_choices_sha256'][task_id] = digest(task['inputs']['choices'])
                    integrity['evaluator_choices_sha256'][task_id] = digest(record['choices'])
        for name in SPLITS:
            source = authority / manifest['benchmarks'][benchmark]['splits'][name]['path']
            rows = json.loads(source.read_text())['tasks']
            if benchmark == 'livemath': integrity['split_counts'][name] = len(rows)
            if benchmark != 'alfworld':
                tasks = []
                for r in rows:
                    task = deepcopy(pool[r['task_id']])
                    task['split'] = name
                    tasks.append(task)
                value = {'schema': 'empirical.tasks.v1', 'benchmark': internal, 'split_seed': 42,
                         'authority_sha256': sha256(authority / 'manifest.json'), 'split_sha256': sha256(source), 'tasks': tasks}
            else:
                value = {'schema': 'empirical.physical-tasks.v1', 'benchmark': benchmark,
                         'authority_sha256': sha256(authority / 'manifest.json'), 'split_sha256': sha256(source), 'tasks': rows}
            frozen_write(output / benchmark / (name + '.json'), value)
            if benchmark != 'alfworld':
                frozen_write(output / benchmark / ('evaluator_records_' + name + '.json'),
                             {r['task_id']: records[r['task_id']] for r in rows})
    frozen_write(output / 'livemath_integrity.json', integrity)
    frozen_write(output / 'materialization.json', {'authority_sha256': sha256(authority / 'manifest.json'),
        'livemath_normalization_version': NORMALIZATION_VERSION, 'livemath_upstream_revision': UPSTREAM_REVISION,
        'livemath_choice_projection_version': CHOICE_PROJECTION_VERSION, 'livemath_choice_seed': CHOICE_SEED,
        'integrity_sha256': sha256(output / 'livemath_integrity.json'),
        'external_files_sha256': external,
        'files_sha256': {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.glob('*/*.json'))}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    generate_parser = sub.add_parser('generate')
    generate_parser.add_argument('--resources', required=True)
    generate_parser.add_argument('--output', required=True)
    generate_parser.add_argument('--alfworld-data', required=True)
    generate_parser.add_argument('--legacy-manifests', required=True)
    verify_parser = sub.add_parser('verify')
    verify_parser.add_argument('--authority', required=True)
    materializer = sub.add_parser('materialize')
    materializer.add_argument('--authority', required=True)
    materializer.add_argument('--prepared-pool', required=True)
    materializer.add_argument('--output', required=True)
    args = vars(parser.parse_args())
    command = args.pop('command')
    {'generate': generate, 'verify': verify, 'materialize': materialize}[command](**args)
    print(json.dumps({'command': command, 'completed': True, 'counts': COUNTS}), flush=True)


if __name__ == '__main__':
    main()
