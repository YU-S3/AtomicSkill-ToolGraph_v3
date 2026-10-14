"""Public fields of registered native result protocols; raw Host records stay intact."""
from copy import deepcopy

SCOPED_SEARCH = 'native.scoped-search.v1'
TEXT_PAGE = 'native.text-page.v1'
SEARCH_PAGE = 'native.search-page.v3'
TEXT_CONTEXT = 'native.text-context.v3'
FILE_COMPUTE = 'native.file-compute.v1'
PROGRAM_EXECUTION = 'program.execution-result.v1'
MANUAL_PROGRAM_EXECUTION = 'manual-reference.program-result.v1'
MANUAL_GRAPH_EXECUTION = 'manual-reference.graph-result.v1'
HOST_FIELDS = {
    SCOPED_SEARCH: {'scope_audit'},
    SEARCH_PAGE: {'scope_audit'},
    FILE_COMPUTE: {'local_artifact_identities', 'local_artifact_hashes',
                   'worker_cpu_seconds', 'worker_seconds'},
    PROGRAM_EXECUTION: {'cpu_seconds', 'elapsed_seconds', 'program_id'},
    MANUAL_PROGRAM_EXECUTION: {'cpu_seconds', 'elapsed_seconds', 'program_id', 'execution_receipt'},
    MANUAL_GRAPH_EXECUTION: {'nodes'},
}


def public_result(raw, origin=None):
    if not isinstance(raw, dict): return deepcopy(raw)
    hidden = HOST_FIELDS.get(origin, set())
    return {k: deepcopy(v) for k, v in raw.items() if k not in hidden}


def public_reference(raw, path, origin=None):
    if path and path[0] in HOST_FIELDS.get(origin, set()):
        raise ValueError('Host metadata is not a public output reference')
    value = public_result(raw, origin)
    for part in path:
        if isinstance(value, dict) and isinstance(part, str) and part in value: value = value[part]
        elif isinstance(value, list) and type(part) is int and 0 <= part < len(value): value = value[part]
        else: raise ValueError('Local evidence path does not exist')
    return value


def evidence_origin(evidence):
    if 'result_protocol' in evidence: return evidence['result_protocol']
    # Legacy native evidence has no protocol tag. Only known envelope producers
    # qualify; computed JSON and nested business fields are never filtered.
    if evidence.get('kind') == 'native_operation':
        name = evidence.get('operation', {}).get('name')
        if name == 'grep' and 'scope_audit' in evidence.get('reference', {}): return SCOPED_SEARCH
        if name == 'execute_python': return FILE_COMPUTE
    return None
