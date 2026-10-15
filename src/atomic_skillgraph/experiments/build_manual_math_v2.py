"""Build an independent authored catalog, preserving V1 sources and qualification state."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

from ..empirical.contracts import digest, object_schema, program_digest
from ..empirical.manual_math_v2 import VERSION
from .canonical_manifest import sha256
from .run_empirical import write_json

ASSETS = Path(__file__).parents[3]/'reference_assets/livemath_v2'
MATHLIB = 'f0b4c8150aa0dbc758aff1291410c5ae335aff3a'


def obj(fields): return object_schema(fields,list(fields))


def registry():
    rows = [
        ('banach_contraction','functional_analysis','Contracting.lean','Mathlib/Topology/MetricSpace/Contracting.lean',
         'ContractingWith.fixedPoint_isFixedPt; ContractingWith.fixedPoint_unique, lines 236-284',
         'A contraction self-map of a nonempty complete metric space has exactly one fixed point.',
         ['Nonempty complete metric space X','f maps X to X','A single constant 0 <= K < 1 bounds d(f(x),f(y)) <= K*d(x,y) for all x,y'],
         'There exists a unique x in X with f(x)=x.',
         ['contraction mapping','contracting map','complete metric space'],
         'Not applicable to a merely continuous or nonexpansive map; self-map and completeness must be established.'),
        ('borel_cantelli_first','probability','BorelCantelli.lean','Mathlib/MeasureTheory/OuterMeasure/BorelCantelli.lean',
         'MeasureTheory.measure_limsup_atTop_eq_zero, lines 55-64',
         'For a sequence of measurable events with finite sum of probabilities, the probability of infinitely many events occurring is zero.',
         ['A probability measure and measurable events E_n indexed by natural numbers','The series sum_n P(E_n) is finite'],
         'P(limsup E_n)=0; independence is not required for this direction.',
         ['Borel-Cantelli','Borel–Cantelli','infinitely often'],
         'Divergent sum alone gives no converse; independence conditions for the second lemma are outside this entry.'),
        ('lagrange_subgroup','group_theory','Card.lean','Mathlib/GroupTheory/Coset/Card.lean',
         'Subgroup.card_subgroup_dvd_card, lines 24-70',
         'The order of a subgroup of a finite group divides the order of the group.',
         ['G is a finite group','H is a subgroup of G'],
         '|H| divides |G|.', ['finite group','subgroup'],
         'The converse is false in general; this entry says nothing about representation dimensions or infinite cardinal divisibility.'),
    ]
    result=[]
    for key,domain,file,path,location,statement,assumptions,conclusion,triggers,limitations in rows:
        result.append({'theorem_id':'reference.math.'+key+'.v2','domain':domain,'statement':statement,
            'assumptions':assumptions,'conclusion':conclusion,'quantifier_scope':'Universal conditional statement; no task premises certified',
            'source_title':'mathlib4 formalized mathematics','source_identifier':'https://github.com/leanprover-community/mathlib4/blob/'+MATHLIB+'/'+path,
            'source_location':location,'source_file':'sources/'+file,'source_sha256':sha256(ASSETS/'reference_knowledge/sources'/file),
            'verified_by':'Codex source inspection of named declaration and surrounding hypotheses; not a local Lean rebuild or expert sign-off',
            'verification_status':'source_checked','allowed_transformations':['rename bound variables','apply only after all listed premises are independently established'],
            'known_limitations':limitations,'version':VERSION,'public_triggers':triggers})
    return result


def build(original, output):
    original,root=Path(original).resolve(),Path(output).resolve()
    if root.exists():raise ValueError('Use a new V2 asset directory')
    source=original/'banks/livemath/bank.json';data=json.loads(source.read_text())
    data=deepcopy(data);data['method_revision']=VERSION;data['origin']='human_authored_reference'
    root.mkdir(parents=True)
    shutil.copytree(original/'programs/livemath',root/'programs/livemath')
    shutil.copytree(ASSETS/'reference_knowledge',root/'banks/livemath/reference_knowledge')
    names=['finite_entailment','finite_quantifiers','exact_rational','truth_table','finite_choice','binary_answer']
    programs=dict(zip(names,data['programs']))
    for s in data['skills']:
        s['id'] += '_v2';s['reference_version']=VERSION
    skills={s['id'].split('_')[-2]:s for s in data['skills']}
    for i in data['implementations']:
        i['id']+='_v2';i['skill_id']+='_v2'
    integer={'type':'integer'};text={'type':'string'}
    new=[('exact_integer',obj({'a':integer,'b':integer,'operation':{'enum':['gcd','mod']}}),
          obj({'value':integer,'scope':text}),'Compute bounded gcd or remainder from explicit integers'),
         ('exact_matrix',obj({'matrix':{'type':'array','minItems':1,'maxItems':4,'items':{'type':'array','items':text}}}),
          obj({'determinant':text,'dimension':integer,'scope':text}),'Compute an exact determinant of a supplied rational matrix')]
    for n,inp,out,goal in new:
        p={'source':(ASSETS/'programs'/(n+'.py')).read_text(),'entry':'run','input_schema':inp,'output_schema':out,
           'allowed_tools':[],'environment':data['program_environment'],'result_role':'intermediate',
           'entry_constraints':'Exact publicly grounded bounded operands only; no general theorem or option verdict', 'state':'candidate'}
        p['id']='program_'+program_digest(p);programs[n]=p;data['programs'].append(p)
        shutil.copyfile(ASSETS/'programs'/(n+'.py'),root/'programs/livemath'/(n+'.py'))
        code='LM13' if n=='exact_integer' else 'LM14'
        s=deepcopy(skills['LM08']);s.update(id='skill_reference_livemath_'+code+'_v2',goal=goal,
            guidance='Check the exact operands and bounds. Compute only this local operation; stop on missing data. A finite value is not a proof of the surrounding theorem.',input_schema=inp,output_schema=out)
        data['skills'].append(s);skills[code]=s
        data['implementations'].append({'id':'implementation_reference_livemath_'+code+'_v2','skill_id':s['id'],'program_id':p['id'],'qualification':'manual_reference_unqualified'})
    mapping={'LM03':'truth_table','LM06':'finite_entailment','LM07':'finite_quantifiers','LM08':'exact_rational','LM09':'truth_table','LM10':'finite_choice','LM13':'exact_integer','LM14':'exact_matrix'}
    capabilities=[]
    for code,s in skills.items():
        name=mapping.get(code);p=programs.get(name)
        capabilities.append({'capability_id':'reference.math.'+code+'.v2','code':code,'skill_id':s['id'],
            'domain_tags':['finite_exact' if p else 'mathematical_obligation'], 'operation_kind':name or code,
            'applicability':s['preconditions'] or [s['goal']], 'required_public_evidence':['exact public expression or typed public operands' if p else 'explicit obligation spans'],
            'required_verified_inputs':p['input_schema']['required'] if p else [],
            'forbidden_assumptions':['invented_relation','guessed_option_truth','unproved_domain_completeness'],
            'expected_output_role':'intermediate','stop_conditions':s['failure_modes']+['unverified_inputs'],
            'source_reference_ids':['human_reference_v1:'+sha256(source)],'execution_mode':'program' if p else 'guidance',
            'origin':'human_authored_reference', **({'program_id':p['id']} if p else {})})
    workflows=[{'id':w['id'],'routable':False,'reason':'not_routable_from_raw_question'} for w in data['workflows']]
    for name in ('exact_rational','exact_integer','exact_matrix','finite_entailment','finite_quantifiers'):
        p=programs[name]
        w={'goal':'Calculate a verified public '+name+' obligation and interpret only its local scope',
           'interface_version':'empirical.workflow.v2','nodes':[
               {'id':'calculate','execution_mode':'program','program_id':p['id'],'after':[],
                'args':{key:{'task':key} for key in p['input_schema']['required']}},
               {'id':'scope','execution_mode':'dynamic','goal':'Report what the actual finite calculation establishes and what remains unproved. Do not output an answer label or certify extra premises.',
                'after':['calculate'],'args':{key:{'from':'calculate','field':key} for key in p['output_schema']['required']}}],
           'outputs':{'analysis':{'from':'scope','field':'analysis'}}}
        w['id']='workflow_'+digest(w);data['workflows'].append(w)
        workflows.append({'id':w['id'],'program_id':p['id'],'routable':True,'input_readiness':'exact_public_operands','proof_scope':'local_computation_only'})
    index={'version':VERSION,'capabilities':capabilities,'workflows':workflows,
           'program_contracts':{p['id']:{'required_input_provenance':'exact public expression or explicit typed public operands; result refs must resolve to those same values',
                'input_schema':p['input_schema'],'output_schema':p['output_schema'],
                'proof_scope':'exact_symbolic' if name in {'exact_matrix','exact_rational','exact_integer'} else 'finite',
                'unknown_handling':'needs_input; final-answer tools are not routable without independent all-choice certificates',
                'negative_tests':['missing_input','invalid_domain_or_shape','invented_provenance'],
                'runtime_applicability':p['entry_constraints']} for name,p in programs.items()},
           'policy':{'guidance_max':2,'program_max':2,'workflow_max':1,'final_answer_programs':'not_routable_without_independent_all_choice_certificates'}}
    write_json(root/'banks/livemath/bank.json',data)
    write_json(root/'banks/livemath/capability_index.json',index)
    rows=registry()
    (root/'banks/livemath/reference_knowledge/theorem_registry.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    write_json(root/'IMPORT_SOURCE.json',{'version':VERSION,'original':str(original),'original_bank_sha256':sha256(source),
        'original_programs':{p.name:sha256(p) for p in (original/'programs/livemath').glob('*.py')},
        'knowledge_provenance':'Independent public mathlib sources, no task gold or historical predictions', 'test_history':'previously_viewed_developmental_retest'})
    write_json(root/'SHA256SUMS.json',{p.relative_to(root).as_posix():sha256(p) for p in root.rglob('*') if p.is_file()})
    return {'version':VERSION,'counts':{k:len(data[k]) for k in ('skills','programs','implementations','workflows')},'source_checked_theorems':len(rows)}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--original',required=True);parser.add_argument('--output',required=True)
    print(json.dumps(build(**vars(parser.parse_args()))))
