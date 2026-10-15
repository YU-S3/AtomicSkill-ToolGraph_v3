"""Compile fixed upstream Markdown into lossless, ordered manual Bank assets."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from ..empirical.bank import Bank
from ..empirical.contracts import digest, program_digest
from .run_empirical import write_json

VERSION = 'human.frontier.livemath.r2.20261016'
PACKAGE = Path(__file__).parents[3] / 'resources/human_frontier_r2'
VARIANTS = {'upstream_skillopt': 'skillopt', 'upstream_skillopt_lite': 'skillopt_lite',
            'benchmark_prior': 'skillopt_lite'}


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def compile_variant(name, package=PACKAGE):
    package = Path(package)
    source_name = VARIANTS[name]
    manifest = json.loads((package/'sources/source_manifest.json').read_text())
    source = next(x for x in manifest if x['name'] == source_name)
    raw = (package/'sources'/f'{source_name}_original.md').read_bytes()
    if (hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() != source['git_blob']
            or hashlib.sha256(raw).hexdigest() != source['sha256']):
        raise ValueError('Fixed upstream source changed')
    text = raw.decode('utf-8')
    # Section boundaries keep each paragraph, its exceptions and whitespace intact.
    boundaries = sorted({0, len(text), *[m.start() for m in re.finditer(r'^## ', text, re.M)]})
    skills = []
    for start, end in zip(boundaries, boundaries[1:]):
        segment = text[start:end]
        asset = {'kind':'skill', 'goal':segment.splitlines()[0].lstrip('# '),
            'guidance':segment, 'execution_intent':'guidance_only',
            'content_variant':'upstream_'+source_name, 'source_blob':source['git_blob'],
            'source_segment':{'start':start,'end':end,'unit':'unicode_codepoints'},
            'author_execution_permission':'manual_eval', 'origin':'external_human_reference',
            'rule_basis':'benchmark_prior', 'method_version':VERSION}
        asset['id'] = 'skill_'+digest(asset)
        skills.append(asset)
    order = [s['id'] for s in skills]
    composite = {'id':'workflow_'+digest([source['git_blob'],order]),
        'goal':'Execute the original authored strategy in source order',
        'interface_version':'empirical.workflow.v2', 'assembly_order':order,
        'workflow_execution':'prompt_compiled', 'author_execution_permission':'manual_eval',
        'nodes':[{'id':'section_'+str(i),'execution_mode':'dynamic','goal':s['goal'],
                  'reference_skill_ids':[s['id']],'after':['section_'+str(i-1)] if i else [],
                  'args':{'goal':{'task':'goal'}}} for i,s in enumerate(skills)],
        'outputs':{'answer':{'from':'section_'+str(len(skills)-1),'field':'answer'}}}
    bundle = {'benchmark':'livemath','method_version':VERSION,'variant':name,
        'source_manifest':source,'raw_source_sha256':source['sha256'],
        'assembled_policy_sha256':text_hash(text),'content_variant':'upstream_'+source_name,
        'workflow_execution':'prompt_compiled','assembly_order':order,
        'skill':skills,'program':[],'implementation':[],'workflow':[composite],
        'default_workflow_id':composite['id'], 'external_reference_not_learned':True}
    if ''.join(s['guidance'] for s in skills).encode('utf-8') != raw:
        raise ValueError('Lossless source compilation failed')
    return bundle, raw


def build(output, package=PACKAGE):
    output, package = Path(output), Path(package)
    if output.exists():
        raise ValueError('Use a new Bank export directory')
    output.mkdir(parents=True)
    variants = {}
    for name in VARIANTS:
        bundle, raw = compile_variant(name, package)
        dest = output/'variants'/name
        bank = Bank(dest)
        try:
            for asset in bundle['skill']:bank.put('skill',asset)
            for asset in bundle['workflow']:bank.put('workflow',asset)
            bank.db.execute('INSERT INTO metadata VALUES(?,?)',('manual_authority',VERSION))
            bank.db.commit()
            variants[name] = {'native_bank_digest':bank.digest(),'bundle_hash':digest(bundle),
                'source_hash':bundle['raw_source_sha256'],'assembly_hash':bundle['assembled_policy_sha256'],
                'atomic_count':len(bundle['skill']),'workflow_execution':'prompt_compiled'}
        finally:bank.close()
        write_json(dest/'bank.json',bundle)
        (dest/'SKILL.md').write_bytes(raw)
    legacy = json.loads((package/'legacy_humanfrontier/bank/bank.json').read_text())
    optional = deepcopy(legacy)
    for p in optional['program']:
        if p['source_file'].endswith('exact_compute.py'):
            p['input_schema']['properties']['digits'].update(minimum=0,maximum=30)
        else:
            p['title'] = ('Search finite task-related domains for a counterexample. State the actual task/subproblem '
                'relation and scope; unrelated toy implications are not evidence. No witness is not a general proof.')
            p['entry_constraints']['description'] = p['title']
        old = p['id'];p['id'] = 'program_'+program_digest(p)
        for i in optional['implementation']:
            if i['program_id']==old:i['program_id']=p['id']
    write_json(output/'optional_programs/bank.json',optional)
    write_json(output/'variant_manifest.json',{'method_version':VERSION,'variants':variants,
        'benchmark_prior_equivalent_to':'upstream_skillopt_lite','programs':'off','graph':False,
        'provenance':'external human reference; upstream optimization may overlap the exposed Test',
        'legacy_retained_at':'legacy_humanfrontier/'})
    return variants


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True)
    print(json.dumps(build(ap.parse_args().output),indent=2))
