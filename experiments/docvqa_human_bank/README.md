# DocVQA 人工 Bank 配对评测

当前 v1.12 **未通过完整 Test 验收，不能作为稳定提分版本推荐**。
它通过固定 Train36、其余 Train144 和 Val22 的推进门槛后，在首对 Test201 中
得到178/201（88.56%），同期旧 v1.1 对照为182/201（90.55%）：纠正8题、退步12题。
首对失败后自动取消第二对，没有重试替换原结果。95%目标至少需要191/201，尚未达到。

本目录是用户提供人工 Bank 的独立对照，不是 Ours 自然学习产物。
通用学习、晋升和评分规则没有因这些人工资产而放宽。
原包来源与 SHA256 见 `source_manifest.json`，公开来源见 `sources/SOURCES.md`。

## 资产和边界

10 Skill、3 Program、3 Implementation、3 Workflow 通过生产 `Bank.put` 导入并重算引用。
Program 保持 candidate，由只读人工视图明确授权调用；不伪造 usable 或输出消费收据。
单答案执行器使用可选 Program，Workflow 仅保存，不能声称执行了整张图。
Train 阶段也是只读评估，不发生自然学习。原图、canonical 成员和评分器保持固定。

- 常驻 Skill 负责问题目标，专用 Skill 给出对应来源的读取和边界规则，收尾保持该范围。
- `no.` 等编号缩写能实际路由到相应 Skill；多值字段不会被指导为只取第一项。
- 跨裁剪 `selection_target` 是 Agent 的未验证计划，不是 OCR 或已验证证据。
- 最多两次局部执行；保留原图和最多两张真实裁剪。小区域最多插值放大3倍，优先 PNG。
- 裁剪编码超限时从原区域逐级缩小，保持原680,000字符 base64 上限。
- v1.12 在图像额度允许时补充整页定位框，明确标记为 host 绘制的位置提示。
  原图和真实裁剪优先；定位框不属于 Program 输出，也不产生新事实。
- 定位图复用既有请求，不增加强制模型步骤或提高执行次数上限；实际费用照常记录。

## 已完成的对照证据

| 候选 | 范围 | 结果 | 判断 |
|---|---|---|---|
| v1.7 | 两次 Test201 | 182、175；对照181、173 | 净增很小，翻转21题，对照16题；未证明稳定收益 |
| v1.8 | 两次 Train36 | 30、30；对照33、32 | 未通过门槛 |
| v1.9 | 两次 Train36 | 31、34；对照33、32 | 合计持平且有一轮退步 |
| v1.10 | 两次 Train36 | 31、33；对照30、34 | 合计持平且有一轮退步 |
| v1.11 | 两次 Train36 → Train144确认 | 35、34；随后122/144，对照128/144 | 确认失败，停止后续评测 |
| v1.12 | 两次 Train36 | 34、34；对照33、32 | 通过开发门槛 |
| v1.12 | 两次 Train144确认 | 127、124；对照125、124 | 通过推进门槛；平均增益区间跨0 |
| v1.12 | 两次 Val22 | 20、20；对照20、20 | 持平 |
| v1.12 | 首对 Test201 | 178；对照182 | 失败，第二对取消 |

v1.12 在 Train144 的重复对错翻转为11题，对照17题；这没有转化为 Test 提升。
Test 的候选/对照分别消耗1,062,315/688,188 tokens、241/220 HTTP，各有1次空答，
无基础设施错误。更多裁剪和更高费用没有证明更好的结果。
原 v1.7 首对曾纠正的5题中，v1.12本轮答对4题；不能将这种局部保留当成整体稳定收益。

每个开发/Train确认候选要求两轮都不低于对应对照、合计严格提高、无额外空答或基础设施错误。
已失败的首对不会再购买第二对来覆盖失败。Val 在已通过正向 Train确认后允许持平。
稳定性另看逐题退步、重复对错翻转和配对任务簇区间；95% EM 意味着 Test 至少191/201。
重复同题不是独立新样本；多次查看和复测的 Test 是探索性结果，不再声称未接触留出集。

## 固定身份运行

本轮使用 `evaluate.py`，显式固定源码、题单、只读 Bank 和累计账本。
v1.12 付费受测源码为 `0076f304b189f813323d46708df4fad8ec4b2c90`，模式为 `guidance`。
合并上游后的 `571a9bc0f47bd9b9c8256210df65521fa575cb8d` 通过3项聚焦零 HTTP 检查，
付费结果仍只归属上述冻结源码，不改称为合并版的付费验证。
旧 `run.py` 默认的 `grounded` 是另一种消融，不等于本轮受测设置。

```bash
python experiments/docvqa_human_bank/evaluate.py \
  --repo "$PINNED_REPO" --sha "$EXPECTED_SHA" \
  --config "$SOURCE_CONFIG" --selection "$FROZEN_SELECTION" \
  --output "$NEW_OUTPUT" --campaign "$CAMPAIGN" \
  --env-file "$ENV_FILE" --crop-mode guidance --check
```

`--check` 只做预检。移除它才发出请求。候选不传 `--bank`，按固定源码重新导入资产；
`--bank` 仅用于匹配旧版源码的对照 Bank。同版本同目录恢复使用 `--resume`。
每个原始 run 的 `plan.json`、`result.json`、`episodes.jsonl`、`final_frozen_manifest.json`
及对应 split 的原始 traces/requests 保存实际身份、结果、费用和冻结前后哈希。

使用 `deepseek-flash`，thinking enabled、effort high、输出上限32768，每题96,000 tokens。
run_seed42 固定题单，Provider 不支持 generation seed；响应不是确定性复现。
累计上限19,488,000 tokens /9,135 HTTP，结转旧人工对照819,414 tokens /270 HTTP。
账本排他锁序列化评测，失败与所有对照费用均保留，不因换候选扩大预算。
