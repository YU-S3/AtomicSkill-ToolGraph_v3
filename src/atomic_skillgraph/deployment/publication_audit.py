"""Read-only publication evidence. No learning credit or lifecycle mutation."""
import json
from pathlib import Path
from collections import defaultdict

from ..core.serialization import to_primitive, atomic_write_json
from ..evolution.identity_matching import raw_hash
from ..knowledge.identity_index import IdentityIndex
from .release_protocol import ReleaseError, sha, contained


def json_lines(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(to_primitive(row), ensure_ascii=False, sort_keys=True) + '\n'
                                 for row in rows), encoding='utf-8')
    temporary.replace(path)


def inventory(assets, refs):
    return [{'ref': row['artifact_ref'], 'kind': row['artifact_kind'],
             'source_status': row['status'], 'file_sha256': sha(path),
             'raw_payload_hash': raw_hash(payload), 'authored_source': str(obj.ref) in refs,
             'source_relative_path': path.as_posix().split('/source/unpacked/', 1)[-1]}
            for row, obj, payload, path in assets]


def verify_preservation(database, bank, source_inventory):
    rows=[]
    for source in source_inventory:
        indexed=database.execute('SELECT * FROM artifact_index WHERE artifact_ref=?', (source['ref'],)).fetchone()
        if source['authored_source']:
            if indexed: raise ReleaseError('authored source version still indexed')
            continue
        if indexed is None or indexed['status'] != source['source_status']:
            raise ReleaseError('old asset status changed during publication')
        if sha(indexed['file_path']) != source['file_sha256']:
            raise ReleaseError('old immutable asset bytes changed')
        rows.append({'ref':source['ref'],'file_sha256':source['file_sha256'],'status':indexed['status']})
    return rows


def identity_and_support(database, bank, root):
    """Verify complete proofs; retain ordered positive AND negative history.

    A source Trace absent from the delivery remains unresolved. We report the
    union, but cannot use these events to promote a historical Candidate.
    """
    groups=defaultdict(list);index=IdentityIndex(database,bank)
    for row in database.rows('SELECT * FROM artifact_identity_index ORDER BY artifact_ref'):
        index.verify(row['artifact_ref'])
        proof=json.loads(contained(bank,row['proof_path']).read_text())
        if proof.get('proof') is None and row['equivalence_id'] != row['artifact_ref']:
            raise ReleaseError('unproved identity shares another ref group')
        groups[(row['artifact_kind'],row['equivalence_id'])].append({
            'ref':row['artifact_ref'],'proof_hash':row['proof_hash'], 'status':proof['status']})
    group_rows=[];support=[]
    for (kind,key),members in groups.items():
        group_rows.append({'kind':kind,'equivalence_id':key,'members':members})
        refs={m['ref'] for m in members}
        events=[dict(e) for e in database.rows('SELECT rowid AS source_order,* FROM evidence_events ORDER BY rowid')
                if e['artifact_ref'] in refs]
        unique={e['event_id']:e for e in events}
        support.append({'kind':kind,'equivalence_id':key,'source_events':list(unique.values()),
            'independent_task_ids':sorted({e['task_id'] for e in unique.values()}),
            'status':'unresolved_source_trace' if events else 'no_execution_evidence',
            'new_execution_credit':0,'lifecycle_projection_changed':False})
    json_lines(root/'identity_groups.jsonl',group_rows)
    atomic_write_json(root/'identity_verification.json',{'passed':True,'rows':sum(map(len,groups.values())),
        'exact_multi_ref_groups':[g for g in group_rows if len(g['members'])>1]})
    atomic_write_json(root/'support_union_report.json',{'groups':support,'promotion_from_union':[],
        'note':'Historical status inherited; unresolved sources do not create new positive or negative credit.'})
    return group_rows
