"""Deterministic new authored Bank; never edit or import historical result labels."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'searchqa_accuracy_v7_20261019/packages/v7_text_compiler'
DEST = HERE / 'packages/minimax_r1'
VERSION = 'human.searchqa.minimax.r1.20261023'
SUFFIX = 'minimax_r1'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def build():
    assert not (DEST / 'FILE_SHA256.json').exists(), 'Never overwrite a frozen package'
    for folder in ['bank', 'guidance', 'programs', 'skills', 'program_assets', 'implementations', 'workflows']:
        (DEST / folder).mkdir(parents=True, exist_ok=True)
    shutil.copytree(PARENT / 'licenses', DEST / 'licenses')
    shutil.copy2(PARENT / 'THIRD_PARTY_NOTICES.md', DEST / 'THIRD_PARTY_NOTICES.md')
    facts = json.loads((HERE / 'fact_cards.json').read_text())
    write(DEST / 'fact_cards.json', facts)
    for name in ['solve', 'evidence', 'surface']:
        text = (HERE / 'guidance' / (name + '.txt')).read_text()
        text = text.replace('{{FACT_CARDS}}', '\n\n'.join(x['fact'] for x in facts))
        (DEST / 'guidance' / (name + '.txt')).write_text(text)
    for source in (PARENT / 'programs').glob('*.py'):
        shutil.copy2(source, DEST / 'programs' / source.name)
    shutil.copy2(HERE / 'submit.py', DEST / 'programs/submit.py')
    bundle = json.loads((PARENT / 'bank/bank.json').read_text().replace('_v7_text_compiler', '_' + SUFFIX))
    bundle.update(method_version=VERSION, parent_method_version='human.searchqa.accuracy.v7.text_compiler',
        development_exposure='Human-authored after inspecting MiniMax Test1400. Public government facts are explicitly sourced. No Test ID, question-to-gold mapping, scorer changes, or claims of natural learning. Results are post-Test development.',
        measured_performance=None,
        runtime='Native solve -> submit. One model solve; pure submit normalizes declared surface forms and records original/normalized values. Public fact cards are part of the evidence Atomic Skill.',
        sources=[{'url': x['source'], 'card_id': x['id'], 'checked_on': x['checked_on']} for x in facts])
    for kind in ['skill', 'program', 'implementation', 'workflow']:
        for asset in bundle[kind]:
            asset['method_version'] = VERSION
            asset['origin'] = 'human_authored_post_test_development'
    for asset in bundle['skill']:
        name = asset['id'].removeprefix('skill_hs_searchqa_').removesuffix('_' + SUFFIX)
        if name in ['solve', 'evidence', 'surface']:
            asset['guidance'] = (DEST / 'guidance' / (name + '.txt')).read_text().strip()
        if name == 'evidence':
            asset['reference_cards'] = facts
        if name == 'submit':
            asset['goal'] = 'Normalize supported surface encoding and numeric units, then seal one answer with an audit.'
            asset['guidance'] = 'Execute the declared pure Program with the current public question and the solve answer. It records original and normalized values; it cannot select identities, remove names, or infer semantic correctness.'
            asset['input_schema']['properties']['question'] = {'type': 'string'}
            asset['input_schema']['required'].append('question')
            asset['output_schema']['properties']['surface_audit'] = {'type': 'object'}
            asset['output_schema']['required'].append('surface_audit')
            submit_skill = asset
    for program in bundle['program']:
        if program['id'] == 'program_hs_searchqa_submit_' + SUFFIX:
            program.update(source=(DEST / 'programs/submit.py').read_text(), title=submit_skill['goal'],
                input_schema=deepcopy(submit_skill['input_schema']), output_schema=deepcopy(submit_skill['output_schema']))
    bundle['workflow'][0]['nodes'][1]['args']['question'] = {'task': 'goal'}
    write(DEST / 'bank/bank.json', bundle)
    for kind, folder in [('skill', 'skills'), ('program', 'program_assets'), ('implementation', 'implementations'), ('workflow', 'workflows')]:
        for asset in bundle[kind]:
            write(DEST / folder / (asset['id'] + '.json'), asset)
    shutil.copy2(PARENT / 'environment.lock.json', DEST / 'environment.lock.json')
    integration = (PARENT / 'integration.py').read_text().replace('human.searchqa.accuracy.v7.text_compiler', VERSION).replace('_v7_text_compiler', '_' + SUFFIX)
    integration = integration.replace('Use the attached authored strategies when relevant, not as factual evidence.',
        'Use the attached Atomic Skills. Their sourced reference cards are background facts only when the requested relation matches; dated public evidence and explicit question wording take precedence.')
    integration = integration.replace('seals the selected answer without changing words', 'normalizes only its declared surface encoding and quantity forms, preserving an audit')
    integration = integration.replace('b1305aeb7075757be2888813e54e3f7aa7e88683', '28ac1ca31cc5f7f1b3d5772fedd3f84c489cebf9')
    (DEST / 'integration.py').write_text(integration)
    write(DEST / 'candidate_provenance.json', {'parent_bank_sha256': sha(PARENT / 'bank/bank.json'),
        'parent_integration_sha256': sha(PARENT / 'integration.py'), 'version': VERSION,
        'fact_sources': facts, 'author_trusted_not_learned': True, 'program_import_state': 'candidate',
        'normal_model_calls': 1, 'test_label_runtime_input': False, 'post_test_development': True,
        'scope': 'General target/surface skills plus sourced domain relationships; no task ID or per-question answer lookup.'})
    (DEST / 'README.md').write_text('# MiniMax R1 authored SearchQA Bank\n\n原生 Atomic / Workflow / Implementation / Program；solve → submit，每题正常一次模型求解。人工事后开发，不是自然学习或未见 Test 的评测。事实卡片使用公开政府资料；提交 Program 仅处理字符表示和有条件数字格式，保留修改收据，不做姓名截短或答案查表。\n')
    write(DEST / 'FILE_SHA256.json', {str(p.relative_to(DEST)): sha(p) for p in sorted(DEST.rglob('*')) if p.is_file() and p.name != 'FILE_SHA256.json'})
    return {'package': str(DEST), 'counts': {k: len(bundle[k]) for k in ['skill', 'program', 'implementation', 'workflow']},
            'guidance_chars': sum(len(s['guidance']) for s in bundle['skill'] if s['execution_intent'] == 'guidance_only')}


if __name__ == '__main__':
    print(json.dumps(build()))
