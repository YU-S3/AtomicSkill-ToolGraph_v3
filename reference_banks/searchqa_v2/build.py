"""Version the supplied manual Bank. No task IDs, predictions or gold are inputs."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from atomic_skillgraph.empirical.bank import Bank
from atomic_skillgraph.empirical.contracts import program_digest

HERE = Path(__file__).resolve().parent
VERSION = 'human.searchqa.reference.v2'
GUIDANCE = {
    'SQ01': ('Identify the missing target of a complete or elliptical clue',
        'Treat a clue as a request for its missing relation, not a request to repeat a supplied name or number. '
        'Resolve the target noun, grammatical number, date and relation before selecting a span. '
        'For a fragment, infer the strongest supported missing relation from the public titles and snippets. '
        'A list can ask for a shared class, location, group or work; consider all listed items together. '
        'Do not arbitrarily select one listed item or invent a missing category. Explicit wording overrides these possibilities.'),
    'SQ02': ('Read document titles and bodies with exact source boundaries',
        'Titles often identify the subject, work or category that a snippet omits. Read [TLE] and [PAR] together. '
        'DOC segmentation preserves the original text and offsets. Programs are optional; submit only one action per turn.'),
    'SQ03': ('Locate diverse evidence and preserve relational context',
        'Use evidence windows to inspect several distinct documents and the full relation around matching terms. '
        'Line wraps and ellipses are not new facts. A ranked match is neither truth nor a final answer; '
        'check subject, predicate and requested object against the original full context, which remains available.'),
    'SQ04': ('Use explicit aliases without changing the answer target',
        'The alias program recognizes explicit aka/also-known-as/uppercase parenthesized acronyms only. '
        'It cannot resolve arbitrary common names, synonyms or spelling variants. For these, read the public evidence directly; '
        'not_found means the pattern is unsupported, not that a proposed answer is false.'),
    'SQ05': ('Use the narrow parent/birthplace extractor only on supported questions',
        'This Program ONLY accepts Who is/was the mother/father of X? or Where was X born? '
        'It does NOT handle membership, performers, countries, authors or general identification. '
        'Do not rewrite an unrelated clue to force a call. On not_found, continue from the context; '
        'do not call it again on the same unsupported relation.'),
    'SQ06': ('Recognize the numeric parser input contract',
        'Use this parser only for literal subject|year|value|unit lines already present in context. '
        'Ordinary prose and dates are not this format. Read those directly; never invent rows to make the parser applicable.'),
    'SQ09': ('Render a unique answer only after explicit relation extraction',
        'This renderer is only the last step of the narrow parent/birthplace workflow. Its verified flag means '
        'the exact source quote was checked, not that an arbitrary candidate has semantic proof. '
        'Other question types can be answered directly without this Program.'),
    'SQ10': ('Disambiguate omitted relations across documents',
        'For short clues, compare the same entity across titles and snippets. Distinguish a work from its author, '
        'performer, character or source work; distinguish a location from its containing region and a member from its group. '
        'Choose the relation supported by the clue and corroborating text, not merely the most frequent copied name. '
        'Do not keep searching for a hidden category absent from the public input; if genuinely unresolved, report uncertainty rather than fabricating evidence.'),
    'SQ11': ('Respect historical scope and numerical direction',
        'Match the clue date, tenure, version and geographical scope. Do not replace historical officeholders with current ones. '
        'For comparisons, establish the same entity and units first; distinguish publication timestamps from event dates.'),
    'SQ12': ('Return exactly the requested constituent',
        'Return one concise answer in <answer>...</answer>. Fill only the missing part of a quoted title or phrase; '
        'do not repeat words the clue already supplies. Return the requested place, person, number or modifier, '
        'not its surrounding organization, location or description. Preserve disambiguating name parts and source spelling; '
        'do not mechanically reduce every person to a surname, swap aliases, or strip meaningful words. '
        'Match singular/plural to the requested target and omit explanations or alternative answers.'),
}


def build(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError('Use a new package directory')
    original = json.loads((source/'bank/bank.json').read_text())
    data = deepcopy(original)
    names = {p.read_text(): p.stem for p in (source/'programs').glob('*.py')}
    output.mkdir(parents=True); (output/'programs').mkdir()
    mapping = {}
    for p in data['programs']:
        old = p['id']; name = names[p['source']]
        if (HERE/(name+'.py')).exists():
            p['source'] = (HERE/(name+'.py')).read_text()
        if name == 'explicit_relation':
            p['source'] = p['source'].replace('Program: match strongly explicit surface relations; abstain on ambiguity.',
                'ONLY Who is/was the mother/father of X? or Where was X born? Not a general relation solver.')
            p['entry_constraints']['preconditions'] = ['Who is/was the mother/father of X?', 'Where was X born?']
        p['id'] = 'program_'+program_digest(p)
        if p['id'] != old: p['parent_id'] = old
        mapping[old] = p['id']
        (output/'programs'/(name+'.py')).write_text(p['source'])
    for skill in data['skills']:
        old = skill['id']; key = old.rsplit('_', 1)[-1]
        skill.update(id=old+'_v2', parent_id=old, method_version=VERSION)
        if key in GUIDANCE: skill['goal'], skill['guidance'] = GUIDANCE[key]
        if key in {'SQ01', 'SQ10'}:
            skill['applicability'] = 'Complete or elliptical public clue and supplied context'
            skill['stop_condition'] = 'No supported unique relation after considering all clue terms'
        mapping[old] = skill['id']
    for impl in data['implementations']:
        impl.update(parent_id=impl['id'], id=impl['id']+'_v2',
            skill_id=mapping[impl['skill_id']], program_id=mapping[impl['program_id']])
    for w in data['workflows']:
        w.update(parent_id=w['id'], id=w['id']+'_v2')
        for n in w['nodes']:
            if n.get('program_id'): n['program_id'] = mapping[n['program_id']]
            if n['execution_mode'] == 'dynamic':
                n['goal'] += (' Resolve the missing relation of elliptical clues using titles and all clue items; '
                    'do not merely repeat a clue entity. Select only the requested short constituent, '
                    'not adjacent facts. No invented category or evidence.')
    data.update(bank_version=VERSION, origin='human_authored_test_feedback_revision_not_ours_learned',
        parent_bank_sha256=hashlib.sha256((source/'bank/bank.json').read_bytes()).hexdigest(),
        revision_basis='Public failure patterns from reference Test; no per-task answer rules or gold in assets')
    bank = Bank(output/'bank', environment=data['program_environment'])
    try:
        for kind, key in [('skill','skills'),('program','programs'),('implementation','implementations'),('workflow','workflows')]:
            for asset in data[key]:
                assert bank.put(kind, asset)['id'] == asset['id']
    finally: bank.close()
    (output/'bank/bank.json').write_text(json.dumps(data, ensure_ascii=False, indent=2))
    (output/'README.md').write_text((HERE/'README.md').read_text())
    sums = {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.rglob('*')) if p.is_file()}
    (output/'SHA256SUMS.json').write_text(json.dumps(sums, indent=2))
    return {key:len(data[key]) for key in ('skills','programs','implementations','workflows')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True); parser.add_argument('--output', required=True)
    print(build(**vars(parser.parse_args())))
