"""Transfer the provided full GEPA strategy into three native authored Atomic Skills."""
import hashlib
import json
from pathlib import Path
import shutil
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'searchqa_target_accuracy_20261024'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

def build():
    assert not (HERE/'packages').exists()
    shutil.copytree(OLD/'packages/scope',HERE/'packages/scope')
    dest=HERE/'packages/gepa';shutil.copytree(OLD/'packages/scope',dest)
    source=HERE.parent/'searchqa_gemini_gepa_audit_20261020/actual_system_prompts.json'
    prompt=json.loads(source.read_text())['gemini_gepa_seed44']
    sections={}
    for i in range(1,7):
        start=prompt.index('### '+str(i)+'.')
        end=prompt.index('### '+str(i+1)+'.') if i<6 else prompt.index('## Domain Reference Facts')
        sections[i]=prompt[start:end].strip().removesuffix('---').strip()
    opening=prompt[:prompt.index('## Core Strategy')].strip().removesuffix('---').strip()
    facts=prompt[prompt.index('## Domain Reference Facts'):prompt.index('## Task Format')].strip()
    policies={'solve':opening+'\n\n'+sections[3]+'\n\n'+sections[4],
        'evidence':sections[1]+'\n\n'+facts,
        'surface':sections[2]+'\n\n'+sections[5]+'\n\n'+sections[6]+'\n\n'+prompt[prompt.index('## Task Format'):].strip()}
    before='target_scope_20261024';after='target_gepa_20261024';version='human.searchqa.'+after
    bundle=json.loads((dest/'bank/bank.json').read_text().replace(before,after))
    bundle.update(method_version=version,parent_method_version='human.searchqa.target_scope_20261024',
        development_exposure='Post-Test development. Full user-supplied GEPA seed44 strategy and Train-derived fact reminders transferred verbatim into three Ours Atomic Skills; no Test gold, task IDs or per-task answer lookup in runtime.')
    for kind in ('skill','program','implementation','workflow'):
        for a in bundle[kind]:a['method_version']=version
    for a in bundle['skill']:
        name=a['id'].removeprefix('skill_hs_searchqa_').removesuffix('_'+after)
        if name in policies:
            a['guidance']=policies[name];(dest/'guidance'/f'{name}.txt').write_text(a['guidance']+'\n')
        if name=='evidence':
            a['reference_cards']=[]
            a['embedded_reference_blocks']=[{'id':'supplied_gepa_seed44_train_facts','source_sha256':sha(source),'injection':'complete block inside evidence Atomic guidance'}]
    integration=(dest/'integration.py').read_text().replace(before,after)
    integration=integration.replace('"reference_card_ids": [card["id"] for card in selected_cards]', '"reference_card_ids": ["supplied_gepa_seed44_train_facts"]')
    (dest/'integration.py').write_text(integration)
    write(dest/'bank/bank.json',bundle)
    for kind,folder in [('skill','skills'),('program','program_assets'),('implementation','implementations'),('workflow','workflows')]:
        for p in (dest/folder).glob('*.json'):p.unlink()
        for a in bundle[kind]:write(dest/folder/(a['id']+'.json'),a)
    write(dest/'fact_cards.json',{'id':'supplied_gepa_seed44_train_facts','source_sha256':sha(source),'text':facts})
    write(dest/'candidate_provenance.json',{'version':version,'source_prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
        'source_file_sha256':sha(source),'normal_model_calls':1,'raw_policy_sections_preserved':True,'author_trusted_not_learned':True,
        'post_test_development':True,'test_gold_runtime_input':False,'counts':{k:len(bundle[k]) for k in ('skill','program','implementation','workflow')},
        'purpose':'Test substantive full reference strategy transfer; avoid a sparse-card comparison with otherwise identical prompts.'})
    (dest/'README.md').write_text('# Full GEPA strategy in Ours\n\n用户提供的seed44最终策略原文分配给solve/evidence/surface三条Atomic；事实块完整常驻。原生solve→submit，真实Program执行与评分。继承scope的纯提交格式Program。人工参考迁移，不伪装自然学习；不得作为未见Test泛化证明。\n')
    write(dest/'FILE_SHA256.json',{str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='FILE_SHA256.json'})
    print(json.dumps({'package':str(dest),'bundle':sha(dest/'bank/bank.json'),'guidance_chars':{k:len(v) for k,v in policies.items()}}))
if __name__=='__main__':build()
