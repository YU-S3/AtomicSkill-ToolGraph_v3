"""Verify completed accounting, write a Chinese review and package raw evidence."""
import hashlib
import json
from pathlib import Path
import zipfile
import analyze

x=analyze.run
r,c,HERE,ROOT,STAGE=x.r,x.c,x.HERE,x.ROOT,x.STAGE


def finalize():
    result=analyze.analyze()
    assert result['status']=='completed' and result['common_tasks']==128
    assert set(result['new_attempt_statuses'])=={'metered'} and result['new_physical_http']<=300
    assert result['old_reservations_unchanged']
    assert result['measured_new_tokens']==sum(v['tokens'] for v in result['metrics_same_completed_tasks'].values())
    assert all(r.read(ROOT/STAGE/'freeze_audit.json').values())
    assert result['remaining_token_admission']>=0
    old=r.read(HERE/'baseline_metrics.json')
    replay=r.read(HERE/'surface_replay_host_only.json')
    plus=sum(not v['old_correct'] and v['score']['em']==1 for v in replay)
    minus=sum(v['old_correct'] and v['score']['em']==0 for v in replay)
    stats=result['metrics_same_completed_tasks']
    ratio=result['r1_v7_token_ratio']
    conclusion=('新版在这128道固定题上有净提升。' if result['net']>0 else
                '新版在这128道固定题上没有净提升。' if result['net']==0 else
                '新版在这128道固定题上退步，不应替换旧Bank。')
    text=f'''# MiniMax SearchQA R1 准确率审查

{conclusion} 本轮未启动新版完整 Test1400，不能把128题准确率当作完整测试集达到82%或85%的证明。

|版本|正确数|EM|F1|Tokens|HTTP|
|---|---:|---:|---:|---:|---:|
'''
    for a,label in [('v7','同期V7'),('r1','新版R1')]:
        v=stats[a];text+=f"|{label}|{v['correct']}/128|{v['em']:.2%}|{v['f1']:.2%}|{v['tokens']:,}|{v['http']}|\n"
    text+=f'''
同题改对 {result['gained']}、改错 {result['lost']}、净增 {result['net']}。新/旧 token 比为 {ratio:.3f}，{'符合' if ratio<=1.5 else '超过'}用户此前最高增加50%的成本偏好。精确配对 McNemar p={result['paired_mcnemar_exact_two_sided_p']:.4f}；配对 bootstrap 95% 区间为 {result['paired_bootstrap95_em_delta']}（EM差值比例）。样本较小且为接触旧Test后的开发实验，不是盲测。

原完整V7有效结果仍为 {old['correct']}/1400（{old['em']:.4%}），F1 {old['f1']:.4%}，{old['tokens']:,} tokens / {old['http']} HTTP。旧Docker200题的费用仍排除于有效结果，完整物理账本保留。达到82%需要在旧基线上净增32题，85%需要净增74题。

## 实际问题与修改

284道旧错题中，123道包含正确答案的连续词片段但输出更长，15道相反；这些是字符串诊断，不能等同于123道都可安全截短。另有6道HTML/弯引号等字符差异。官方评分已删除ASCII百分号，旧1400个最终回答也没有百分号，因此仅删百分号不能解释本轮原准确率。

R1保持原生 Atomic / Workflow / Implementation / Program，7个Skill、4个Program、4个Implementation、1个Workflow。目标关系推断、姓名/类别范围和最终表达分属三个常驻指导Skill；公开事实卡片属于证据Skill。正常每题一次模型求解后执行真实submit Program。没有改通用方法代码或官方评分器。

submit只处理HTML实体、弯引号/破折号、严格数量问法下的小数字，以及问题已明确的年龄/百分数单位。它保存原答案、处理后答案和规则列表；不会删除人名或查询Test答案。零HTTP回放旧回答改对{plus}题、改错{minus}题，假想分数为{old['correct']+plus-minus}/1400；该回放没有替换旧成绩，也不是新版模型重跑结果。

事实卡片记录内政部机构隶属、1913—1973副总统与总统关系、参议院宪法数量，来源分别为[美国内政部](https://www.doi.gov/bureaus)、[美国参议院历史表](https://www.senate.gov/about/officers-staff/vice-president/vice-presidents.htm)、[参议院宪法说明](https://www.senate.gov/artandhistory/history/common/briefing/Constitution_Senate.htm)。卡片有适用关系与年代边界，没有Test题号或逐题gold映射。资产为人工author_trusted，Program导入为candidate，无自然学习或晋升声明。

## 控制条件与验收

128题在canonical1400中用seed20261023均匀抽样、未按正确率筛选，并保留canonical顺序。两臂逐题轮转、MiniMax-M3 adaptive thinking、stream、65536 cap、每题128K和公开context一致。generation seed不支持。完整题单和plan已在付费前冻结；执行中未改Bank或源码。

实际源码28ac1ca31cc5f7f1b3d5772fedd3f84c489cebf9，两臂一致。它相对之前MiniMax源码增加了独立Gemini适配和缓存恢复审计补丁，MiniMax请求参数在离线测试中核对。零HTTP测试覆盖原生编译、Docker执行、输出消费、公开输入/模型payload、完成resume、Docker中断缓存续跑和传输失败停止。实际两臂共256题均计分。原生Workflow完整完成数：V7 {stats['v7']['native_graphs_completed']}/128，R1 {stats['r1']['native_graphs_completed']}/128；finish-only收尾题数：V7 {stats['v7']['finish_only_tasks']}，R1 {stats['r1']['finish_only_tasks']}；截断请求：V7 {stats['v7']['truncated_requests']}，R1 {stats['r1']['truncated_requests']}。这些普通模型失败保留原评分和全部费用，不能算作Program成功。实际Program错误数为{sum(v['program_errors'] for v in stats.values())}，Frozen前后hash未变。

本轮实际新增 {result['measured_new_tokens']:,} tokens / {result['new_physical_http']} HTTP，全部计量。原154,912 tokens未知预留原样保留。未提高原8M累计token额度；剩余可准入 {result['remaining_token_admission']:,} tokens。预算文件包含历史物理成本，不能把它全部当作本轮成本。

逐题变化见paired_changes_host_only.json，真实提交处理见actual_surface_changes_host_only.json，完整请求/轨迹/Bank/config/plan/账本在审查包。所有gold及旧错误诊断仅在Host审查文件；它们不属于运行Bank输入。
'''
    (HERE/'准确率改进审查报告.md').write_text(text)
    archive=HERE.parent/'SearchQA_MiniMax_R1_准确率审查_20261023.zip'
    assert not archive.exists(),'Preserve earlier review packages'
    files={}
    for base,prefix in [(HERE,'delivery'),(ROOT,'run')]:
        for p in sorted(base.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or p.name in ['runner.log','pid','budget.json.lock','.lock']:continue
            name=prefix+'/'+str(p.relative_to(base))
            files[name]=p
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        sums={}
        for name,p in files.items():
            data=p.read_bytes();z.writestr(name,data);sums[name]=hashlib.sha256(data).hexdigest()
        z.writestr('SHA256SUMS.json',json.dumps(sums,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(archive) as z: assert z.testzip() is None
    receipt={'archive':str(archive),'sha256':r.sha(archive),'bytes':archive.stat().st_size,'files':len(files),
             'report':str(HERE/'准确率改进审查报告.md'),'completed':True,'results':result}
    r.write_json(HERE/'delivery_artifacts.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':finalize()
