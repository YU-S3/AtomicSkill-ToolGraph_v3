"""Explicit host qualification fixture for dispatch-only tests with inert Workers."""
from atomic_skillgraph.empirical.contracts import digest, program_digest
from atomic_skillgraph.empirical.local_validation import POLICY


def qualify(bank, program):
    validation={'policy':POLICY,'passed':True,'program_digest':program_digest(program),
        'environment_hash':digest(program['environment']),'source_trace_sha256':'inert-worker-unit-fixture',
        'source_physical_key':'unit-fixture','operation_contract_hash':digest([
            program['input_schema'],program['output_schema'],program.get('entry_constraints')])}
    bank.record({'id':'qualification:'+program['id'],'program_id':program['id'],'task_key':'unit-fixture',
        'origin':'train_test','outcome':'positive','basis':'local_check','validation':validation})
    return bank.record_validation(program['id'],validation)
