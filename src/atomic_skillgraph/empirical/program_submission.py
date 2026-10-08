"""Shared Program output and host publication contracts."""
from copy import deepcopy
import json
from .contracts import validate_schema_instance

CONTRACT_VERSION = 'empirical.program-submission.v1'
PUBLICATION_FIELDS = {'files': {'type': 'array', 'items': {'type': 'string'}},
                      'deleted_files': {'type': 'array', 'items': {'type': 'string'}}}


class ProgramContractError(ValueError):
    def __init__(self, code, detail, repair_target='source'):
        self.code, self.repair_target = code, repair_target
        self.feedback = {'code': code, 'detail': detail, 'repair_target': repair_target}
        super().__init__(json.dumps(self.feedback, ensure_ascii=False))


def submission_contract(adapter):
    getter = getattr(adapter, 'program_submission_contract', None)
    return getter() if getter else {'version': CONTRACT_VERSION,
        'final_submission_kind': adapter.capabilities.final_submission_kind,
        'publication_contract': {'supported': False, 'required_files': []}}


def effective_output_schema(declared_schema, *, result_role, publication_contract):
    schema = deepcopy(declared_schema)
    if not isinstance(schema, dict) or schema.get('type') != 'object':
        raise ProgramContractError('program_output_schema_invalid', 'Output must be an object schema', 'declaration')
    supported = publication_contract.get('supported', False)
    def normalize(node):
        if not isinstance(node, dict):
            raise ProgramContractError('program_output_schema_invalid', 'Composition needs schema objects', 'declaration')
        if node.get('type') == 'object' or any(k in node for k in ('properties', 'required', 'additionalProperties')):
            props = node.setdefault('properties', {})
            for name, reserved in PUBLICATION_FIELDS.items():
                if name in props:
                    field = props[name]
                    if supported and field.get('type') == 'array' and 'items' not in field:
                        field['items'] = {'type':'string'}
                    if supported and (field.get('type') != 'array' or field.get('items', {}).get('type') != 'string'):
                        raise ProgramContractError('program_publication_field_conflict', name, 'declaration')
                elif supported:
                    props[name] = deepcopy(reserved)
            if supported and any(k in node for k in ('const', 'enum', 'not')):
                raise ProgramContractError('program_publication_schema_unsupported', 'Object const/enum/not cannot extend publication fields', 'declaration')
        for key in ('allOf', 'anyOf', 'oneOf'):
            for branch in node.get(key, []): normalize(branch)
    normalize(schema)
    if result_role == 'final_files':
        schema['required'] = sorted(set(schema.get('required', [])) | {'files'})
    return schema


def validate_program_declaration(asset, capabilities):
    if asset.get('execution_intent') == 'guidance_only' and 'source' not in asset:
        return deepcopy(asset)
    contract = capabilities if isinstance(capabilities, dict) else {
        'final_submission_kind': capabilities.final_submission_kind,
        'publication_contract': {'supported': False, 'required_files': []}}
    role, kind = asset.get('result_role', 'intermediate'), contract['final_submission_kind']
    if role not in ('intermediate', 'final_answer', 'final_files'):
        raise ProgramContractError('program_result_role_invalid', role, 'declaration')
    if role == 'final_answer' and kind not in ('text', 'single_answer'):
        raise ProgramContractError('program_submission_kind_mismatch', 'final_answer requires a text Adapter', 'declaration')
    if role == 'final_files' and kind != 'files':
        raise ProgramContractError('program_submission_kind_mismatch', 'final_files requires a file submission Adapter', 'declaration')
    output = effective_output_schema(asset['output_schema'], result_role=role,
                                    publication_contract=contract['publication_contract'])
    if role == 'final_answer' and ('answer' not in output.get('required', []) or
            output.get('properties', {}).get('answer', {}).get('type') != 'string'):
        raise ProgramContractError('program_answer_schema_invalid', 'Required string answer is missing', 'declaration')
    return {**deepcopy(asset), 'output_schema': output}


def prepare_program_submission(adapter, program, result, *, workspace_before=None):
    status = {'status': 'not_applicable', 'payload': None, 'error_code': None,
              'repair_target': None, 'diagnostics': {}, 'output_contract_status': 'not_checked',
              'contract_version': CONTRACT_VERSION}
    if result.get('status') != 'ok': return status
    try:
        normalized = validate_program_declaration(program, submission_contract(adapter))
        outputs = result.get('outputs')
        json.dumps(outputs, allow_nan=False)
        validate_schema_instance(outputs, normalized['output_schema'])
        role = program.get('result_role', 'intermediate')
        if role == 'final_answer':
            if not outputs['answer'].strip():
                raise ProgramContractError('program_answer_empty', 'answer must be nonempty')
            status.update(status='ready', payload=outputs['answer'])
        elif role == 'final_files':
            required = set(submission_contract(adapter)['publication_contract']['required_files'])
            receipt = result.get('publication_receipt', {})
            before, after = receipt.get('before', {}), receipt.get('after', {})
            current = adapter.observe().get('workspace', {})
            if not receipt.get('invocation_id') or not receipt.get('host_verified'):
                raise ProgramContractError('program_publication_incomplete', 'Missing host invocation publication receipt')
            if workspace_before is not None and before != workspace_before:
                raise ProgramContractError('program_publication_receipt_mismatch', 'Publication before state differs')
            if not required.issubset(receipt.get('declared', [])) or not required.issubset(after.get('outputs', [])):
                raise ProgramContractError('program_publication_incomplete', 'Required files must all be published by this invocation')
            if after != current or not all(after.get('hashes', {}).get(n) for n in required):
                raise ProgramContractError('program_publication_receipt_mismatch', 'Current sealed workspace differs from receipt')
            if not any(before.get('hashes', {}).get(n) != after.get('hashes', {}).get(n) for n in required):
                raise ProgramContractError('program_publication_unchanged', 'No required file changed in this invocation')
            if not adapter.submission_ready(outputs, result_role=role, previous_workspace=before):
                raise ProgramContractError('program_publication_incomplete', 'Published required files are unavailable')
            status.update(status='ready', payload=outputs)
        status['output_contract_status'] = 'valid'
    except (ValueError, TypeError) as exc:
        status.update(status='contract_error', error_code=getattr(exc, 'code', 'program_output_contract_invalid'),
                      repair_target=getattr(exc, 'repair_target', 'source'),
                      diagnostics={'detail': str(exc)}, output_contract_status='invalid')
    return status


def normalize_program_result(adapter, program, result, *, workspace_before=None):
    if result.get('status') != 'ok':
        return {**result, 'worker_status': result.get('worker_status', result.get('status')),
                'output_contract_status': result.get('output_contract_status', 'not_checked')}
    prepared = prepare_program_submission(adapter, program, result, workspace_before=workspace_before)
    value = {**result, 'worker_status': result.get('worker_status', result.get('status')),
             'output_contract_status': prepared['output_contract_status'], 'submission_preparation': prepared,
             'compatibility_contract_version': CONTRACT_VERSION}
    if prepared['status'] == 'contract_error':
        value.update(status='execution_error', error_code=prepared['error_code'],
                     repair_target=prepared['repair_target'], worker_result=deepcopy(result),
                     detail=prepared['diagnostics']['detail'])
    return value


def positive_eligible(result):
    return result.get('status') == 'ok' and result.get('output_contract_status') == 'valid'
