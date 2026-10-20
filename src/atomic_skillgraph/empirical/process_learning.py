"""Full-Train process learning; no inner Train membership split or authored assets."""
from copy import deepcopy

from .contracts import digest

VERSION = 'officeqa.process-learning.v1'
MATERIAL = 'learner.public-processes.v1'
POLICY = {'material_version': MATERIAL, 'extractor_payload_max_bytes': 131072,
          'builder_payload_max_bytes': 131072, 'structural_repair_payload_max_bytes': 16384}

PROMPT = """
Learn reusable procedures and executable operations from this completed Train experience.
The public task, source pages and actual executed processes are evidence, not an answer key.
The whole-task score is feedback: a failed answer does not invalidate an independently
executed retrieval or computation, and a successful answer does not prove every step correct.
Learn complementary, scoped Skills over successive tasks: when to apply, the procedure,
required evidence and stop/fallback conditions. Reuse an existing real ID when already covered.
Keep independently useful capabilities separate; do not replace the Bank with one global summary.
Programs may retrieve, extract tables, bind periods/units, calculate, or format a supported answer.
Prefer parameterized deterministic work that can replace repeated model steps. Do not manufacture
an operation merely to fill a capability list. Avoid copying example-specific numbers into guidance.
operation_id/input_ids/output_ids refer only to this request's Host directory. The Builder receives
the complete executed source, its inputs, supplied pages and output references.
Use guidance_only for useful advice that lacks an executable source; it is advisory, not verified code.
For a Program, select a real source binding. intermediate is appropriate for a local result.
final_answer requires a nonempty executed answer:string AND a correct source task. Do not relabel
unverified arithmetic or page text as the answer. Keep repair count and validation rules unchanged.
No private gold, test examples, authored Bank content, task-ID lookup or invented evidence.
Do not request extra benchmark tasks; the Host runs the existing bounded source tests.
Submit one submit_learning call. A single task may produce one new/revised Skill and one realization
job; the Bank accumulates these across ALL Train tasks. Use $new for the new Skill.
"""

RUNTIME = """
Supplied source pages below are complete public inputs with readable result references.
Temporary Python can read them through ctx.read_result; this does not require another corpus search.
When the source has a regular table, prefer extracting rows/columns from that text and calculating
in one parameterized operation. Check the requested labels and units; do not guess missing rows.
For deterministic processing, expose structured JSON with the input/source references, operation,
operands, units and computed result. If the requested answer is fully established, include its exact
formatted answer:string; otherwise return an intermediate result and identify missing evidence.
Use eligible learned Programs when their interface and entry conditions fit. They may fail or ask
for input; inspect that result and continue normally. A verified local operation is not proof of a
whole-task answer. Submit an established answer without redundant model recomputation.
"""


def enabled(config):
    return config.get('experiment', {}).get('officeqa_method_version') == VERSION


def validate(config):
    exp = config['experiment']
    if any(exp.get(k) for k in ('choice_selection_physical_keys', 'office_effect_version',
                               'manual_reference_version', 'office_human_bank_version')):
        raise ValueError('Process learning uses all Train and an independent natural Bank')
    from .public_pages import enabled as supplied
    if not supplied(config):
        raise ValueError('Process learning requires the approved public input protocol')
    if exp.get('shared_answer_contract') != 'office.scalar-or-explicit-template.v1':
        raise ValueError('Process learning requires the same shared scalar answer contract')
    if config['learning'].get('source_program_payload_max_bytes') is not None:
        raise ValueError('Process learning has one explicit complete-material capacity')
    settings = {**config['llm'], **config['llm'].get('runtime', {})}
    if (config['runtime'].get('global_action_budget') != 48
            or config['runtime'].get('evidence_handoff_version') != 'public-evidence-spans.v1'
            or config['runtime'].get('finish_input_max_bytes') != 65536
            or settings.get('max_completion_tokens') != 32768
            or settings.get('max_total_tokens_per_task') != 216576
            or settings.get('reasoning_effort') != 'high'
            or settings.get('protocol', {}).get('thinking_type') != 'enabled'
            or config['llm']['purpose_overrides'].get('finish_only') !=
               {'protocol': {'thinking_type': 'disabled'}, 'max_completion_tokens': 32768}):
        raise ValueError('Process Runtime must retain the frozen public-input V4 inference settings')


def case_view(experience, case_id):
    """Keep real pages/code and typed selectors; never a guessed replacement summary."""
    from .model_view import select_operation_directory
    from .public_results import public_result, evidence_origin
    evidence_by_id = {e['id']: e for e in experience.get('local_evidence', [])}
    cards, audit = select_operation_directory(list(evidence_by_id.values()), case_id)
    for card in cards:
        evidence = evidence_by_id[card['local_evidence_ref']]
        op = evidence['operation']
        source = op.get('source') or op.get('arguments', {}).get('source')
        if source:
            card.update(source_excerpt=source, source_truncated=False)
        card['observed_output'] = public_result(evidence['reference'], evidence_origin(evidence))
    return {'case_source': {'physical_case_id': case_id, 'experience_sha256': digest(experience)},
            'task': deepcopy(experience['task']), 'score': {'hard': bool((experience.get('score') or {}).get('hard'))},
            'submission': deepcopy(experience.get('submission')),
            'termination_reason': experience.get('termination_reason'),
            'local_evidence': cards, 'compact_learning_directory': True,
            'process_selection': audit,
            'process_directory': [{'event_id': e.get('event_id'), 'name': e['name'],
                                   'accepted': e.get('result', {}).get('accepted')}
                                  for e in experience.get('events', [])],
            'failed_actions': [{'name': e['name'], 'arguments': deepcopy(e['arguments']),
                               'result': deepcopy(e.get('result'))}
                              for e in experience.get('events', [])
                              if e.get('result', {}).get('accepted') is False]}


def record_skill(asset, old, task, experience):
    asset = deepcopy(asset)
    asset['evidence_source'] = {'source_task_id': task.task_id, 'physical_key': task.physical_key,
        'experience_sha256': digest(experience), 'material_version': MATERIAL, 'split': 'train',
        'qualification': 'advisory_until_executable_validation'}
    if old:
        asset['parent_skill_id'] = old['id']
    return asset


def parameter_probe(system, program, binding, evidence, broker, trial_id):
    """Compare a changed numeric source literal with independently re-executed Python."""
    import ast
    from .local_validation import reference_value
    from .contracts import validate_schema_instance
    op = evidence['operation']
    if evidence['kind'] != 'native_operation' or op['name'] != 'execute_python':
        return None
    selected = None
    for key, ref in binding.get('input_refs', {}).items():
        value = binding['inputs'][key]
        if ref['kind'] != 'source_literal':
            continue
        if type(value) in (int, float):
            selected = (key, ref, value + 1)
        elif isinstance(value, list) and value and all(type(v) in (int, float) for v in value):
            selected = (key, ref, [v + 1 for v in value])
        if selected:
            break
    if selected is None:
        return None
    key, ref, value = selected
    inputs = {**binding['inputs'], key: value}
    try:
        validate_schema_instance(inputs, program['input_schema'])
    except ValueError:
        return None
    tree = ast.parse(op['arguments']['source'])
    parent = tree
    for part in ref['path'][:-1]:
        parent = parent[part] if isinstance(part, int) else getattr(parent, part)
    replacement = ast.parse(repr(value), mode='eval').body
    field = ref['path'][-1]
    if isinstance(field, int):
        parent[field] = replacement
    else:
        setattr(parent, field, replacement)
    source = ast.unparse(ast.fix_missing_locations(tree))
    if system.budget_governor:
        system.budget_governor.admit_validation(trial_id + ':source_variation')
    reference = broker.call('execute_python', {**op['arguments'], 'source': source})
    if system.budget_governor:
        system.budget_governor.admit_validation(trial_id + ':candidate_variation')
    actual = system.worker.execute(program, inputs, broker)
    passed = reference.get('accepted') is True and actual.get('status') == 'ok'
    if passed:
        expected = {k: reference_value({'reference': reference, 'result_protocol': evidence.get('result_protocol')}, r)
                    for k, r in binding['reference_fields'].items()}
        passed = all(actual.get('outputs', {}).get(k) == v for k, v in expected.items())
    return {'passed': passed, 'input_hash': digest(inputs), 'source_sha256': digest(source),
            'reference': reference, 'candidate': actual, 'kind': 'executed_numeric_literal_variation'}
