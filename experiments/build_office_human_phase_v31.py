"""Keep the actual authored Bank JSON, Program IDs and readable cards in sync."""
import json
from pathlib import Path

from atomic_skillgraph.empirical.contracts import digest, program_digest, object_schema as obj
from atomic_skillgraph.empirical.office_human_policy_v31 import VERSION, REF

ROOT=Path(__file__).resolve().parents[1]/'resources/office_human_frontier_v31'
CARDS={
'OA-Target': ('Type the target before retrieving',
    'Lock the exact measure, observation periods, fiscal/calendar basis, reporting vintage, requested statistic, '
    'output unit and format; a percentage numeric scale is distinct from its percent symbol. Source amounts only from acquired evidence. Search hits locate a source; they are not '
    'automatically the correct table, column or value. Focus on the single missing target field.'),
'OA-Locate': ('Acquire the relevant source with bounded search',
    'Use a narrow file scope and distinctive measure. The scoped_search Program can batch search and read real '
    'hit paths/offsets within this task budget. A returned ok proves only acquired source windows; it '
    'does not establish table/period alignment or a final answer. After empty hits vary one query dimension. Read the exact hit '
    'and nearby header/unit; do not repeatedly expand to the whole corpus or treat an exhausted search as complete target coverage.'),
'OA-Align': ('Bind quantities to the actual header, period and unit',
    'Align the title, header, row, footnotes, period and unit before using adjacent OCR numbers. Use table_extract '
    'only with a complete relevant table window and exact column/row labels. Ambiguous OCR needs native read_context, '
    'not guessed cells. Keep each operand with its source_ref and exact quote in evidence_state; name it by its '
    'computation field (e.g. numerator, denominator, from_value, to_value, values.0). Preserve period order and missing '
    'observations; never fill gaps. Accumulate operands across rounds; repeated reports are deduplicated. Retrieve only missing observation periods in human_office_progress.period_coverage. Conflicting values need explicit source disambiguation; to correct a record, supply replaces_source_ref for that exact old source. Gross/net, offered/accepted, old/new securities, currency quotation direction and '
    '32nds are different quantities, not interchangeable units.'),
'OA-Compute': ('Compute from explicitly named operands',
    'Use series_metrics for its listed formulas when it fits; otherwise use existing temporary Python. Difference '
    'names FROM and TO plus absolute; ratio names numerator and denominator; change/growth names base_value and '
    'new_value. Set output_percent explicitly, never multiply twice. periods means elapsed intervals, not number '
    'of observations. Programs prove local arithmetic only. Persist the computed output via answer_candidate.result_ref '
    'with matching operation, output_unit and covers_task. Choose an actual scalar path from recent_scalar_result_refs, never its parent dictionary. Only the current executed operands need bindings; use answer_candidate.operand_bindings by operand name to disambiguate saved sources. Temporary Python may return operand_values alongside answer '
    'to retain its binding. If the target is solved, finish now rather than search again for the same quantity.'),
'OA-Commit': ('Submit the best supported result through the native protocol',
    'If the exact requested target has been computed from the required observations, immediately use native '
    'runtime_step(action="finish", answer=<only the requested numeric/list/date payload>). No XML tags. For percentages keep the numeric scale and omit a trailing percent symbol unless the question explicitly requires it in a template. '
    'Do not restart a broad search or replace a grounded value with an unverified one. A candidate is not semantic '
    'certification: check its target, direction, period, unit and precision against its actual evidence. When forced '
    'to finish, preserve the best genuinely supported result and remaining uncertainty; do not invent a completion receipt.')}


def build():
    path=ROOT/'bank/bank.json';bank=json.loads(path.read_text())
    old_skills={s['id']:s for s in bank['skill']};mapping={};skills=[]
    for title,(goal,text) in CARDS.items():
        asset={'kind':'skill','title':title,'goal':goal,'guidance':text,'execution_intent':'guidance_only',
               'input_schema':obj({},[]),'output_schema':obj({},[]),'origin':'external_human_reference',
               'author_execution_permission':'manual_eval','method_version':VERSION}
        asset['id']='skill_'+digest(asset);skills.append(asset)
    by_title={s['title']:s for s in skills}
    aliases={'oq01':'OA-Target','oq02':'OA-Locate','oq03':'OA-Align','oq04':'OA-Align',
             'oq05':'OA-Compute','oq06':'OA-Align','oq07':'OA-Align','oq08':'OA-Compute','oq09':'OA-Commit',
             'scoped_search':'OA-Locate','table_extract':'OA-Align','series_metrics':'OA-Compute'}
    for old,s in old_skills.items():
        title=s.get('title') or next((v for k,v in aliases.items() if old.endswith(k)),None)
        if title not in by_title:raise ValueError('Unmapped legacy Skill '+old)
        mapping[old]=by_title[title]['id']
    bank['skill']=skills
    number={'type':['number','string']};nums={'type':'array','items':number,'minItems':1,'maxItems':5000}
    common={'unit':{'type':'string','minLength':1},'decimals':{'type':'integer','minimum':0,'maximum':18},
            'operand_refs':{'type':'array','items':REF},'output_percent':{'type':'boolean'}}
    for p in bank['program']:
        old=p['id'];name=Path(p['source_file']).stem;p['source']=(ROOT/p['source_file']).read_text()
        p['method_version']=VERSION
        if name=='series_metrics':
            other=[v for v in p['input_schema'].get('properties',{}).get('operation',{}).get('enum',[])
                   if v not in {'difference','ratio','percent_change','log_return','cagr','annual_log_growth'}]
            if not other:other=['sum','mean','median','std_population','std_sample','variance_population','variance_sample',
                'gini_population','gini_sample_adjusted','correlation','ols_slope','ols_intercept','geometric_growth']
            branches=[obj({**common,'operation':{'const':'difference'},'from_value':number,'to_value':number,
                'absolute':{'type':'boolean'}},['operation','from_value','to_value','absolute','unit']),
                obj({**common,'operation':{'const':'ratio'},'numerator':number,'denominator':number},
                    ['operation','numerator','denominator','output_percent','unit']),
                obj({**common,'operation':{'enum':['percent_change','log_return','cagr','annual_log_growth']},
                     'base_value':number,'new_value':number,'periods':number},
                    ['operation','base_value','new_value','output_percent','unit']),
                obj({**common,'operation':{'enum':other},'values':nums,'other_values':nums},['operation','values','unit'])]
            p['input_schema']={'type':'object','oneOf':branches}
            p['output_schema']=obj({**{k:{'type':'string'} for k in ['value','answer','formula','direction','periods','unit','proof_scope','convention']},
                'operand_refs':{'type':'array','items':REF},'count':{'type':'integer'},'output_percent':{'type':'boolean'}},
                ['value','answer','formula','direction','operand_refs','periods','unit','proof_scope','convention','count','output_percent'])
            p['title']='Named local arithmetic: explicit numerator/denominator, from/to+absolute, base/new+output_percent. Never certify question semantics.'
        elif name=='scoped_search':
            props=p['output_schema']['properties'];props.pop('search_complete',None)
            props.update(search_exhausted={'type':'boolean'},target_periods_complete={'type':['boolean','null']},table_header_bound={'type':'boolean'})
            p['output_schema']['required']=['windows','search_exhausted','target_periods_complete','table_header_bound','next_cursor','scope']
            p['title']='Batch scoped search and real hit reads. Acquired windows are usable intermediate evidence; ok is local retrieval success only; table/period binding remains.'
        else:
            p['input_schema']['required']=['text','rows','column','unit']
            props=p['output_schema']['properties'];props.update(header={'type':'string'},unit={'type':'string'},
                requested_rows={'type':'array','items':{'type':'string'}},table_header_bound={'type':'boolean'},
                target_periods_complete={'type':'boolean'},proof_scope={'type':'string'})
            p['output_schema']['required']=list(props)
            p['title']='Extract unique requested rows under an exact header and actual unit from an acquired table window. Ambiguous OCR needs native read.'
        p['id']='program_'+program_digest(p);mapping[old]=p['id']
        p['entry_constraints']['description']=p['title']
        # entry_constraints participates in identity; compute after all edits.
        p['id']='program_'+program_digest(p);mapping[old]=p['id']
    def remap(v):
        if isinstance(v,str):return mapping.get(v,v)
        if isinstance(v,list):return [remap(x) for x in v]
        if isinstance(v,dict):return {k:remap(x) for k,x in v.items()}
        return v
    bank['implementation']=remap(bank['implementation'])
    for a in bank['implementation']:a['id']='implementation_'+digest({k:v for k,v in a.items() if k!='id'})
    old_default=bank['default_workflow_id'];workflows=[]
    # Legacy fixed-argument helper graphs no longer describe the named ABI.
    # Preserve only the dynamic phase graph; callable operations live in Implementation.
    for w in remap([w for w in bank['workflow'] if w['id']==old_default]):
        for n in w['nodes']:
            if 'reference_skill_ids' in n:n['reference_skill_ids']=list(dict.fromkeys(n['reference_skill_ids']))
        old=w['id'];w['id']='workflow_'+digest({k:v for k,v in w.items() if k!='id'});workflows.append(w)
        if old==old_default:bank['default_workflow_id']=w['id']
    bank.update(workflow=workflows,method_version=VERSION,measured_performance=None,
        revision_basis='Public interface/evidence audit; no task IDs, golds or answer lookup rules',
        runtime_mode='native_dynamic_tool_loop; authored stage cards, graph not forced')
    path.write_text(json.dumps(bank,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'skills').mkdir(exist_ok=True)
    for s in skills:(ROOT/'skills'/f"{s['title']}.md").write_text('# '+s['title']+'\n\n'+s['guidance']+'\n')
    (ROOT/'SKILL.md').write_text('# Authored Office Bank V3.1\n\nRuntime reads bank.json guidance; this file mirrors it. '
        'At most two phase cards are visible. Frozen manual assets, not learned usable claims.\n\n'+
        '\n\n'.join('## '+s['title']+'\n\n'+s['guidance'] for s in skills)+'\n')
    return bank


if __name__=='__main__':build()
