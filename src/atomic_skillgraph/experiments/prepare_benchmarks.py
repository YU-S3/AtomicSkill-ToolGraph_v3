"""Build fixed Ours manifests from authorized local resources, with private gold separate."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tarfile

from atomic_skillgraph.empirical.contracts import PublicTask, digest
from atomic_skillgraph.harness.benchmarks import truncate_context
from .run_empirical import write_json


def split_exact(rows, quotas, *, seed=42):
    import numpy as np
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    labels = sorted({label for row in rows for label in row['labels']})
    matrix = np.array([[int(label in row['labels']) for label in labels] for row in rows])
    remaining, groups = list(range(len(rows))), {}
    names = list(quotas)
    for name in names[:-1]:
        if not quotas[name]:
            groups[name] = []
            continue
        splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=quotas[name]/len(remaining), random_state=seed)
        rest, chosen = next(splitter.split(np.zeros((len(remaining),1)), matrix[remaining]))
        groups[name] = [remaining[i] for i in chosen]
        remaining = [remaining[i] for i in rest]
    groups[names[-1]] = remaining
    totals = matrix.sum(axis=0)
    def error(group, target):
        counts = matrix[group].sum(axis=0) if group else np.zeros(len(labels))
        return float(np.abs(counts-totals*target/len(rows)).sum())
    while any(len(groups[name]) != quotas[name] for name in names):
        source = next(name for name in names if len(groups[name]) > quotas[name])
        target = next(name for name in names if len(groups[name]) < quotas[name])
        def movement(index):
            src = [i for i in groups[source] if i != index]
            gain = error(src, quotas[source]) + error(groups[target]+[index], quotas[target])
            return gain, hashlib.sha256(f'{seed}|{rows[index]["id"]}'.encode()).hexdigest()
        index = min(groups[source], key=movement)
        groups[source].remove(index)
        groups[target].append(index)
    return {name: sorted((rows[i] for i in group), key=lambda row: hashlib.sha256(
        f'{seed}|{row["id"]}'.encode()).hexdigest()) for name, group in groups.items()}


def upstream_ids(upstream, dataset):
    return {split: json.loads((upstream/'data'/f'{dataset}_id_split'/split/'items.json').read_text())
            for split in ['train','val','test']}


def spreadsheet_cases(directory, override=None):
    if override is not None:
        if any(Path(value).name != value for case in override for value in case.values()):
            raise ValueError('Resource mapping must stay in the task directory')
        return [{key:str(directory/value) for key,value in case.items()} for case in override]
    cases = []
    for source, gold in [('_input', '_answer'), ('_init', '_golden')]:
        for path in sorted(directory.glob('*' + source + '.xlsx')):
            answer = path.with_name(path.name.replace(source + '.xlsx', gold + '.xlsx'))
            if answer.is_file(): cases.append({'input':str(path), 'gold':str(answer)})
    if not cases and (directory/'initial.xlsx').is_file() and (directory/'golden.xlsx').is_file():
        cases.append({'input':str(directory/'initial.xlsx'), 'gold':str(directory/'golden.xlsx')})
    return cases


def prepare(resources, output, case_map=None):
    import pyarrow.parquet as pq
    resources, output = Path(resources).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    upstream = resources/'upstream/SkillOpt-fa4ca184573e42ec11472959dd57422381418096'
    datasets = {}
    search_ids = upstream_ids(upstream,'searchqa')
    chosen = {str(row['id']) for rows in search_ids.values() for row in rows}
    search = {}
    for path in sorted((resources/'raw/searchqa').rglob('*.parquet')):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                if str(row['key']) in chosen:
                    search[str(row['key'])] = {'id':str(row['key']), 'goal':row['question'],
                        'inputs':{'context':truncate_context(row['context'])}, 'answers':row['answers'], 'labels':['searchqa']}
    if set(search) != chosen: raise ValueError('SearchQA selected ID coverage mismatch')
    rank = lambda row: hashlib.sha256(f'42|{row["id"]}'.encode()).hexdigest()
    train = sorted((search[str(row['id'])] for row in search_ids['train']),key=rank)
    val = sorted((search[str(row['id'])] for row in search_ids['val']),key=rank)
    datasets['searchqa'] = {'train':train[:300],'val':val[:24],
        'test':[search[str(row['id'])] for row in search_ids['test']], 'reserve':train[300:]+val[24:]}
    document_ids = {str(row['id']):row for group in upstream_ids(upstream,'docvqa').values() for row in group}
    documents = []
    images = output/'public_images'
    images.mkdir(exist_ok=True)
    for path in sorted((resources/'raw/docvqa').rglob('*.parquet')):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
            for row in batch.to_pylist():
                task_id = str(row['questionId'])
                if task_id not in document_ids: continue
                image = images/(task_id+'.png')
                from PIL import Image
                import io
                raw = row['image']['bytes']
                with Image.open(io.BytesIO(raw)) as pixels: pixels.convert('RGB').save(image)
                documents.append({'id':task_id,'goal':row['question'],'inputs':{'images':[str(image)]},
                    'answers':row['answers'],'labels':document_ids[task_id]['topic'].split('|')})
    if len(documents) != 534: raise ValueError('DocVQA Universe must contain the fixed 534 questions')
    datasets['docvqa'] = split_exact(documents, {'train':180,'val':24,'test':200,'reserve':130})
    office = [{'id':row['uid'],'goal':row['question'],'inputs':{'corpus':'full offline corpus; glob/read/grep'},
               'answer':row['answer'],'labels':[row['difficulty']]} for row in csv.DictReader(
               (resources/'raw/officeqa/officeqa_full.csv').open(encoding='utf-8-sig'))]
    if len(office) != 246: raise ValueError('OfficeQA Full must contain 246 tasks')
    datasets['officeqa'] = split_exact(office, {'train':120,'val':24,'test':102})
    mathematics = []
    for path in sorted((resources/'raw/livemath/data').glob('*/*.json')):
        for row in json.loads(path.read_text()):
            mathematics.append({'id':str(row['month'])+':'+str(row['no']), 'goal':row['mcq']['question'],
                'inputs':{'choices':row['mcq']['choices']}, 'choices':row['mcq']['choices'],
                'correct_choice':row['mcq']['correct_choice'], 'labels':[str(row['month']), *row['theorem_type']]})
    if len(mathematics) != 177: raise ValueError('LiveMath Universe must contain 177 tasks')
    datasets['livemath'] = split_exact(mathematics, {'train':60,'val':17,'test':100})
    extracted = resources/'extracted/spreadsheetbench_verified_400'
    if not (extracted/'dataset.json').exists():
        archive_path = next((resources/'raw/spreadsheetbench').glob('*.tar.gz'))
        destination = extracted.parent
        destination.mkdir(parents=True,exist_ok=True)
        with tarfile.open(archive_path) as archive:
            for member in archive.getmembers():
                target = (destination/member.name).resolve()
                if not target.is_relative_to(destination) or member.issym() or member.islnk():
                    raise ValueError('Unsafe workbook archive member')
            archive.extractall(destination,filter='data')
    spreadsheets = []
    case_map = json.loads(Path(case_map).read_text()) if case_map else {}
    for row in json.loads((extracted/'dataset.json').read_text()):
        directory = extracted/row['spreadsheet_path']
        cases = spreadsheet_cases(directory, case_map.get(str(row['id'])))
        if not cases or not all(Path(case['gold']).is_file() for case in cases):
            raise ValueError('Spreadsheet cases missing: '+str(row['id']))
        position = row['answer_position']
        if '!' not in position and row.get('answer_sheet'): position = row['answer_sheet']+'!'+position
        spreadsheets.append({'id':str(row['id']),'goal':row['instruction'],
            'inputs':{'input_file':'/workspace/inputs/input.xlsx','output_file':'/workspace/case1_result.xlsx',
                      'answer_position':position,'solution_contract':'Write solution.py and case1_result.xlsx. solution.py uses INPUT_PATH and OUTPUT_PATH.'},
            'public_files':{'input.xlsx':cases[0]['input']},'cases':cases,
            'instruction_type':row['instruction_type'],'answer_position':position,'labels':[row['instruction_type']]})
    datasets['spreadsheet'] = split_exact(spreadsheets, {'train':200,'val':20,'test':180})
    counts = {}
    for benchmark, groups in datasets.items():
        path = output/benchmark
        records = {}
        counts[benchmark] = {name:len(rows) for name,rows in groups.items()}
        for split, rows in groups.items():
            tasks = []
            for row in rows:
                task_id = benchmark+':'+row['id']
                records[task_id] = {key:value for key,value in row.items() if key not in {'goal','inputs','labels'}}
                task = PublicTask(task_id,digest([benchmark,row['id']]),row['goal'],row['inputs'],split)
                from dataclasses import asdict
                tasks.append(asdict(task))
            manifest = {'schema':'empirical.tasks.v1','benchmark':benchmark,'split_seed':42,'tasks':tasks}
            target = path/(split+'.json')
            if target.exists() and json.loads(target.read_text()) != manifest:
                raise ValueError('Existing frozen manifest differs; do not silently resplit '+str(target))
            write_json(target,manifest)
        write_json(path/'evaluator_records.json',records)
    write_json(output/'dataset_lock.json',{'upstream':'SkillOpt@fa4ca184573e42ec11472959dd57422381418096',
        'split_seed':42,'seeds':[42,43,44],'counts':counts,'docvqa_split_unit':'question',
        'officeqa_information':'full_offline_no_oracle','livemath_use_theorem':False,'livemath_use_sketch':False})
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resources',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--case-map',help='Explicit approved spreadsheet resource pairing JSON')
    args=parser.parse_args()
    print(json.dumps(prepare(args.resources,args.output,args.case_map)),flush=True)


if __name__=='__main__': main()
