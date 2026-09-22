"""Authenticated authored-bank publication; not a learning evidence channel."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

PROTOCOL_VERSION='r103.bank-release.v1'
OLDFIRST_PROTOCOL_VERSION = 'r103.oldfirst-release.v1'
OLDFIRST_INPUT_HASHES = {
    42: 'e7a09b35d4192b6c02c071b07a968343aa56c3e5214a8ca51f7c4d9ce499d23e',
    43: 'bfbd04ef672a1dd3fc080691a913d68f4de05d110a5d7711dba485392d300de3',
    44: 'eca900051bf7f8eba128a247600d8384aaa5426799feefcf28b74c79f3e2ff07',
}
# Byte-verified original final Frozen snapshot; 169 refs/payloads/statuses
# exactly match the supplied seed43 plan. The run-root ZIP is an older snapshot.
OLDFIRST_SOURCE_ARCHIVES = {seed: {digest} for seed, digest in OLDFIRST_INPUT_HASHES.items()}
OLDFIRST_SOURCE_ARCHIVES[43].add('1138732dc30e3501e7efcfc12401549ab990d27d08e4924d3fccf8fd2be595ca')
INPUT_HASHES={
    42:'83e470df82cf2c65206641cfb1cc772fb6984cd0069a875c331ff1d8fc2c63bf',
    43:'f6a731d0b2b8fa3e4ba57681125fb9ef0f647fbcaaad8b8f6c3c0630e10bac88',
    44:'317e86954797c31e8ac6d05770627bfd7495fdf65fcaf897e0c17164d72654c5',
}
DDL='''CREATE TABLE release_deployments (
 artifact_ref TEXT PRIMARY KEY, artifact_kind TEXT NOT NULL,
 effective_status TEXT NOT NULL, basis TEXT NOT NULL,
 source_ref TEXT NOT NULL, source_payload_hash TEXT NOT NULL,
 released_payload_hash TEXT NOT NULL, checks_path TEXT NOT NULL,
 checks_hash TEXT NOT NULL,
 FOREIGN KEY(artifact_ref) REFERENCES artifact_index(artifact_ref));'''

class ReleaseError(ValueError):
    pass

@dataclass(frozen=True)
class ReleaseSpec:
    seed:int
    input_zip:Path
    output_dir:Path
    base_config:dict
    resume:bool=False

@dataclass(frozen=True)
class PreparedRelease:
    root:Path
    seed:int

@dataclass(frozen=True)
class ReleaseChecks:
    passed:bool
    checks_hash:str

@dataclass(frozen=True)
class FrozenRelease:
    root:Path
    seed:int
    knowledge_digest:str

_SOURCE_TOKEN=object()
@dataclass(frozen=True,init=False)
class VerifiedFrozenSource:
    manifest:dict
    bank:Path
    def __init__(self,manifest,bank,*,_token=None):
        if _token is not _SOURCE_TOKEN:
            raise ReleaseError('verified source can only be constructed by a source validator')
        object.__setattr__(self,'manifest',manifest)
        object.__setattr__(self,'bank',Path(bank))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def contained(root,relative):
    root=Path(root).resolve();path=(root/relative).resolve()
    if not path.is_relative_to(root):raise ReleaseError('release path escape')
    return path

def verify_deployments(database,root):
    if not database.execute("SELECT 1 FROM sqlite_master WHERE name='release_deployments'").fetchone():return
    from ..evolution.identity_matching import raw_hash
    expected_columns = ['artifact_ref','artifact_kind','effective_status','basis','source_ref',
                        'source_payload_hash','released_payload_hash','checks_path','checks_hash']
    columns = database.rows('PRAGMA table_info(release_deployments)')
    if [r['name'] for r in columns] != expected_columns or columns[0]['pk'] != 1:
        raise ReleaseError('invalid release deployment schema')
    foreign_keys = database.rows('PRAGMA foreign_key_list(release_deployments)')
    if not any(r['table'] == 'artifact_index' and r['from'] == 'artifact_ref' and r['to'] == 'artifact_ref' for r in foreign_keys):
        raise ReleaseError('release deployment foreign key missing')
    for row in database.rows('SELECT * FROM release_deployments'):
        if row['basis'] not in {'inherited_registry','evidence_union','authored_revision','curated_existing'}:
            raise ReleaseError('unknown deployment basis')
        indexed=database.execute('SELECT * FROM artifact_index WHERE artifact_ref=?',(row['artifact_ref'],)).fetchone()
        if (indexed is None or indexed['status']!=row['effective_status']
                or indexed['artifact_kind']!=row['artifact_kind']
                or raw_hash(json.loads(Path(indexed['file_path']).read_text()))!=row['released_payload_hash']):
            raise ReleaseError('deployment does not match artifact index')
        check=contained(root,row['checks_path'])
        if sha(check)!=row['checks_hash'] or json.loads(check.read_text())['passed'] is not True:
            raise ReleaseError('deployment checks missing or altered')
        if row['basis'] == 'curated_existing':
            report = json.loads(check.read_text())
            plan_path = Path(root) / 'edit_plan.lock.json'
            plan = json.loads(plan_path.read_text())
            permission = plan.get('manual_publications', {}).get(row['artifact_ref'], {})
            source = next((item for item in plan.get('asset_dispositions', [])
                           if item['ref'] == row['artifact_ref']), {})
            if (report.get('protocol_version') != OLDFIRST_PROTOCOL_VERSION
                    or report.get('edit_plan_hash') != sha(plan_path)
                    or permission.get('basis') != 'curated_existing'
                    or permission.get('old_status') != 'candidate'
                    or permission.get('target_effective_status') != row['effective_status']
                    or permission.get('execution_credit_delta') != 0
                    or source.get('source_sha256') != report.get('source_file_hash')
                    or row['source_ref'] != row['artifact_ref']
                    or row['source_payload_hash'] != row['released_payload_hash']
                    or report.get('source_status') != 'candidate'
                    or report.get('target_status') != row['effective_status']
                    or report.get('source_ref') != row['source_ref']
                    or report.get('source_file_hash') != sha(indexed['file_path'])):
                raise ReleaseError('curated publication source/check mismatch')
