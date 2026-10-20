"""Build separate authored policy and Program variants; never mutate older Banks."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent/'searchqa_accuracy_v7_20261019/packages/v7_text_compiler'
NAMES = ('solve','evidence','conflict','span','surface','check')
GROUPS = {'solve':('solve','conflict'), 'evidence':('evidence',), 'surface':('span','surface','check')}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
write = lambda p,v: p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
schema = lambda props, required: dict(type='object',properties=props,required=required,additionalProperties=False)

def build(variant):
    dest = HERE/'packages'/variant
    dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'FROZEN').exists(), 'Frozen candidate cannot be edited'
    version = 'human.searchqa.accuracy.'+variant
    for folder in ('guidance','programs','skills','program_assets','implementations','workflows','bank'):
        (dest/folder).mkdir(exist_ok=True)
    for name in NAMES: shutil.copy2(HERE/'guidance'/f'{name}.txt',dest/'guidance'/f'{name}.txt')
    shutil.copy2(PARENT/'programs/submit.py',dest/'programs/submit.py')
    # Reuse the tested explicit field parser, not its lexical window ranking.
    old=(PARENT/'programs/evidence_windows.py').read_text()
    parser=old[old.index('def _norm'):old.index('def run(ctx, inputs):')]
    source='"""Extract literal matching fields from current public input; no answer memory."""\nimport re\n\n'+parser
    source+='''def run(ctx, inputs):
    question, context = inputs.get("question"), inputs.get("context")
    if not isinstance(question, str) or not isinstance(context, str):
        return {"status": "needs_input", "outputs": {}, "error": "question/context strings required"}
    fields = _explicit_fields(context, question, _documents(context))
    for field in fields:
        field["clue_text"] = context[slice(*field["clue_span"])]
    return {"status": "ok", "outputs": {"source_fields": {
        "fields": fields[:4], "omitted": max(0,len(fields)-4),
        "status": "matched_literal_fields" if fields else "no_matching_field",
        "offset_unit": "unicode_character_end_exclusive",
        "semantic_status": "unverified; check relation, qualifiers and requested scope"}}}
'''
    (dest/'programs/source_fields.py').write_text(source)
    text={'type':'string'}; report={'type':'object'}; answer={'type':'string','minLength':1}
    qc=schema({'question':text,'context':text},['question','context'])
    inp=schema({'question':text,'context':text,'source_fields':report},['question','context'])
    out=schema({'answer':answer},['answer'])
    env=json.loads((PARENT/'environment.lock.json').read_text())
    skills=[]; programs=[]; impl=[]
    sid=lambda name: f'skill_hs_searchqa_{name}_{variant}'
    for name in (*GROUPS,'source_fields','submit'):
        procedural=name in ('source_fields','submit')
        ins=out if name=='submit' else inp if name=='solve' else qc
        outs=out if name in ('solve','submit') else schema({'source_fields':report},['source_fields']) if name=='source_fields' else schema({},[])
        guidance='\n\n'.join((dest/'guidance'/f'{part}.txt').read_text().strip() for part in GROUPS[name]) if not procedural else (
            'Execute the mapped pure Program on the original public question/context; pass its actual report to solve. No match is normal. Literal matches do not decide the answer.'
            if name=='source_fields' else 'Seal the selected answer verbatim using the native submit Program; do not normalize, expand or shorten its lexical content.')
        skill=dict(id=sid(name),goal=guidance.split('\n')[0],guidance=guidance,input_schema=ins,output_schema=outs,
                   execution_intent='procedural' if procedural else 'guidance_only',result_role='final_answer' if name=='submit' else 'intermediate',
                   method_version=version,origin='human_authored_post_test_development',state='authored',permission='author_trusted',
                   applicability={'selector':'always' if name!='source_fields' else 'public_structural_markers'})
        skills.append(skill)
        if procedural:
            pid=f'program_hs_searchqa_{name}_{variant}'
            programs.append(dict(id=pid,title=skill['goal'],entry='run',backend='sandbox_python_v1',
                source=(dest/'programs'/f'{name}.py').read_text(),input_schema=ins,output_schema=outs,
                allowed_tools=[],environment=env,result_role=skill['result_role'],permission='author_trusted',origin='human_authored',method_version=version))
            impl.append(dict(id=f'implementation_hs_searchqa_{name}_{variant}',skill_id=sid(name),program_id=pid,permission='author_trusted',origin='human_authored'))
    graph=dict(id=f'workflow_hs_searchqa_answer_{variant}',interface_version='empirical.workflow.v2',
        goal='Apply concrete conditional experience to the public question and submit one supported answer',
        method_version=version,origin='human_authored',nodes=[
        dict(id='solve',execution_mode='skill',skill_id=sid('solve'),reference_skill_ids=[sid(n) for n in GROUPS if n!='solve'],
             args={'question':{'task':'goal'},'context':{'task':'context'}},after=[]),
        dict(id='submit',execution_mode='skill',skill_id=sid('submit'),args={'answer':{'from':'solve','field':'answer'}},after=['solve'])],
        outputs={'answer':{'from':'submit','field':'answer'}})
    with_fields=deepcopy(graph);with_fields['id']=f'workflow_hs_searchqa_source_fields_{variant}'
    with_fields['nodes'].insert(0,dict(id='source_fields',execution_mode='skill',skill_id=sid('source_fields'),
        args={'question':{'task':'goal'},'context':{'task':'context'}},after=[]))
    with_fields['nodes'][1]['args']['source_fields']={'from':'source_fields','field':'source_fields'}
    with_fields['nodes'][1]['after']=['source_fields']
    bundle=dict(schema='human.frontier.bank.v1',benchmark='searchqa',method_version=version,parent_method_version='human.searchqa.accuracy.v7.text_compiler',
        trust='author_trusted',default_workflow_id=graph['id'],source_fields_workflow_id=with_fields['id'],
        enable_source_fields=variant=='v8_programmatic',measured_performance=None,
        development_exposure='Prior public Test comparisons and supplied GEPA artifacts informed general conditional skills. No task IDs, dataset entities, gold mappings or factual flashcards in executable assets. All examples are fictional. Post-Test development, not held-out evaluation or natural learning.',
        runtime='Native optional source_fields -> solve -> submit; one text model solve. Program output delivered as an actual bound input; delivery is not proof of reasoning benefit.',
        skill=skills,program=programs,implementation=impl,workflow=[graph,with_fields])
    write(dest/'bank/bank.json',bundle);write(dest/'environment.lock.json',env)
    for kind,folder in [('skill','skills'),('program','program_assets'),('implementation','implementations'),('workflow','workflows')]:
        for asset in bundle[kind]:write(dest/folder/(asset['id']+'.json'),asset)
    integration=(PARENT/'integration.py').read_text().replace('human.searchqa.accuracy.v7.text_compiler',version).replace('_v7_text_compiler','_'+variant)
    old_guidance='GUIDANCE_IDS = (SOLVE_SKILL, "skill_hs_searchqa_evidence_'+variant+'", "skill_hs_searchqa_surface_'+variant+'")'
    assert old_guidance in integration
    integration=integration.replace(old_guidance,'GUIDANCE_IDS = '+repr(tuple(sid(n) for n in GROUPS)))
    selector='''def has_source_fields(context, question):
    import re
    norm = lambda text: " ".join(re.findall(r"[^\\W_]+", text.casefold()))
    shape = "|" in context or "\\t" in context or bool(re.search(r"(?im)^\\s*(?:Q(?:uestion)?|Clue)\\s*:", context))
    return bool(shape and norm(question) and norm(question) in norm(context))

'''
    integration=integration.replace('def read(path):',selector+'def read(path):',1)
    integration=integration.replace('        graph, compiled = compile_graph(system.bank)',
        '''        use_fields = system.bank.bundle["enable_source_fields"] and has_source_fields(task.inputs.get("context", ""), task.goal)
        system.bank.graph = system.bank.bundle["source_fields_workflow_id" if use_fields else "default_workflow_id"]
        graph, compiled = compile_graph(system.bank)
        compiled["source_fields_selected"] = use_fields
        compiled["selection_basis"] = "public_structural_markers"''',1)
    integration=integration.replace('value.get("node", {}).get("id") == "solve" and not value.get("memory", {}).get("recent_results") and len(value["task"]["inputs"].get("context", "")) <= 7500',
                                    'value.get("node", {}).get("id") == "solve"')
    old='''                call_options = dict(kwargs)'''
    new='''                report = None
                source = value.get("bindings", {}).get("completed_results", {}).get("source_fields", {}).get("source_fields")
                if source is not None:
                    report = system.task_context.resolve_public(source) if isinstance(source, dict) and "result_id" in source and "path" in source else source
                    if not isinstance(report, dict) or "fields" not in report:
                        raise ValueError("Source-field producer did not provide its declared report")
                    text_material["source_fields"] = report
                call_options = dict(kwargs)'''
    assert integration.count(old)==1
    integration=integration.replace(old,new)
    integration=integration.replace('"used_inputs": ["question", "context"]}', '"used_inputs": ["question", "context"] + (["source_fields"] if report is not None else [])}')
    integration=integration.replace('"response_kind": "text", "synthesized_model_tool_call": False, "lexical_content_unchanged": True}',
        '"response_kind": "text", "synthesized_model_tool_call": False, "lexical_content_unchanged": True,\n                    "source_fields_delivered": report is not None, "source_field_count": len(report["fields"]) if report else 0,\n                    "consumption_semantics": "actual bound Program output delivered in solve payload; no claim of model reliance",\n                    "guidance_ids": list(GUIDANCE_IDS)}')
    # No menu of hidden optional capabilities: the selected native graph handles pure Programs.
    start=integration.index('    def program_options(');end=integration.index('\n\ndef compile_graph',start)
    integration=integration[:start]+'    def program_options(self, *args, **kwargs):\n        return []\n'+integration[end:]
    integration=integration.replace('Optional Programs are offered only when\\nthe original context is long; use only a displayed real ID, then consume its real result before claiming use.','The selected native graph executes pure Programs when public input structure calls for them.')
    (dest/'integration.py').write_text(integration)
    write(dest/'candidate_provenance.json',dict(version=version,parent_bundle_sha256=sha(PARENT/'bank/bank.json'),
        source_integration_sha256=sha(PARENT/'integration.py'),source_parser_sha256=sha(PARENT/'programs/evidence_windows.py'),
        no_gold_input=True,no_answer_memory=True,fictional_examples=True,normal_model_requests=1,
        source_fields_enabled=bundle['enable_source_fields'],author_trusted_not_learned=True))
    (dest/'README.md').write_text('# '+variant+'\n\n人工经验候选；六类具体经验归入三个原生 Atomic Skill（目标与冲突、证据、范围与表达及检查），满足原生引用上限。programmatic 版本增加按公开结构触发的原生字段提取节点。完整 context 保留；每题正常一次模型求解；没有事实卡片、答案记忆或学习晋升声明。\n')
    write(dest/'FILE_SHA256.json',{str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob('*'))
        if p.is_file() and p.name!='FILE_SHA256.json' and '__pycache__' not in p.parts})
    return {'variant':variant,'guidance_chars':sum(len((dest/'guidance'/f'{n}.txt').read_text()) for n in NAMES),
            'counts':{k:len(bundle[k]) for k in ('skill','program','implementation','workflow')}}

if __name__=='__main__':
    print(json.dumps([build('v8_guidance'),build('v8_programmatic')]))
