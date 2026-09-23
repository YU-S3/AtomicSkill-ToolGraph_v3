# Release4 v2.0 修复与放行报告

日期：2026-09-23。执行代码：`061e66d`。文档：用户提供的 Release4 v2.0 完整实施包。

## 结论

本轮修复及要求的验证已完成，可以启动预先声明的五轮 Frozen Test134。
正式实验尚未启动，`eval/` 尚不存在。

- 全部 pytest：**1807 passed**，121.82 秒。
- 三库真实无模型程序专项：**3/3 通过**。每库分别覆盖公开非可拿取实体定位、可拿取对象定位、先无匹配并回滚再匹配、真实多动作执行及显式数据流后继。专项不计入自然成功率。
- 固定 current、相同六个训练侧任务、三个独立 seed bank：**18/18 official won**。
- 逐 episode 的不可变 Trace、最终 HTTP、全部 attempt 用量、SQLite、来源任务、代码/资源/配置及 bank 摘要检查通过。
- 正式矩阵只读预检查通过；五份配置已经生成。

## 真实结果与费用

以下为每题全部 attempt 的 tokens；reasoning 已包含在 completion，不再次相加。

| 任务 | seed42 | seed43 | seed44 |
|---|---:|---:|---:|
| train_0 / simple | 22,546 | 18,572 | 19,926 |
| train_34 / look | 23,701 | 33,810 | 27,733 |
| train_4 / clean | 35,249 | 41,120 | 45,341 |
| train_36 / heat | 62,567 | 64,750 | 63,486 |
| train_6 / cool | 45,172 | 39,126 | 38,129 |
| train_2 / pick-two | 67,955 | 53,307 | 35,497 |
| 合计 | 257,190 | 250,685 | 230,112 |

自然任务合计 **737,987 tokens / 84 次物理请求**；每题均值 40,999.28，中位数 38,627.5，p90 63,865.2，p95 65,230.75。未知用量请求为 0。
31 次 Support 原始选择全部保留，解码/映射/grounding 拒绝为 0，请求—用量—调用关联缺项为 0。

自然 episode 的最早开始到最晚结束为 UTC 01:48:37.673—01:59:20.235（北京时间 09:48:37—09:59:20），并行墙钟约 **10 分 43 秒**。此区间不含此前构建、pytest、能力探针与独立无模型专项。

这不是正式 Test134 成功率，也不是纯提示词压缩的因果对照。本轮同时修订程序和图；没有通过 40K 截断、降 reasoning、换题或重复抽样获得成绩。加热和 pick-two 仍存在较高成本尾部，完整数据如实保留。

## 机制核对

三库 look 均选择各自 `@1.1.0` 的 stored Composite，并实际执行新版 scoped discovery。

- seed42、44：take → locate_light → observe；定位输出传入后继，observe 为 `direct_autonomous_success`。
- seed43：take → locate_light → light_nav（already_satisfied）→ light_on（自动执行）；USE 达到 official terminal 后，observe 为 `skipped_goal_terminal`，没有多执行环境动作。
- 最终 HTTP 的 native IDs、卡片 IDs 与解码表一致；无方案时不发空 enum；原始 `support_call_id` 没有改写成模型手填映射。
- 新发现来自当前公开 On/In 联合关系或 TAKE 目录，与 validator 同源；没有把 USE、目标文本或历史关系当作当前位置证据。
- 无模型专项各发生一次真实无匹配回滚；每库记录两个 dataflow consumption，且存在零 provider 增量的多动作程序及自动后继。
- 十八题前后冻结 bank 摘要一致，未产生长期学习写入。

## 三库与原始材料

完整新发布目录：

```text
/home/yangchengyu/asg_oldfirst_20260923_release4/
```

Windows 资源管理器地址：

```text
\\wsl.localhost\Ubuntu\home\yangchengyu\asg_oldfirst_20260923_release4\
```

seed42/43/44 当前分别 202/185/199 项资产。185/169/183 项原始资产、逐 ref 许可、失败历史及严格等价关系按基础计划保留；每库新增的是原 logical ID 的发现 Atomic / Implementation / Tool / look Composite 四个版本，不是跨 seed 移植理想库。

- `seed*/frozen/`：完整实际冻结库。
- `seed*/frozen/derived_revision_comparisons.json`、`resolved_ref_map.json`：源正文哈希、具体变更与身份匹配。
- `seed*/frozen/preparation_coverage.json`：全部部署 Composite 的真实输入来源和准备边界；42/43/44 分别审查 11/10/11 张图。Agent 明确选择不记为自动程序覆盖。
- `seed*/frozen/public_discovery_contract.json`：本机语法、parser 版本和 SHA256。
- `dev/coverage_report.md`、`coverage_report.json`、`coverage_acceptance.json`、`controlled_acceptance.json`：汇总和门禁。
- `dev/seed*/current/<task>/`：完整 Trace、provider payload、all_usage、attempt_history、run_state、实际配置和 manifests。
- `dev/seed*/automation/`：三库真实无模型专项的报告和原始 Trace。
- `configs/seed42_rep01.json` 至 rep03，以及 seed43/44_rep01：正式五轮配置。
- `evaluation_plan.json`：固定矩阵；`dev_validation.log`：本轮完整验证日志。

Release3 和原始训练库未改动。build01/build02 是修复阶段的独立构建/专项调试记录，不混入上述十八题结果。

## 正式启动

当前本地代码已是验证版本，不需要 git pull、重新安装或重跑开发验证。

```bash
cd /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3
bash scripts/run_released_frozen_parallel.sh /home/yangchengyu/asg_oldfirst_20260923_release4
```

脚本自动读取现有 `.env`，启动三个流：seed42 rep01→02→03 顺序；seed43/44 各 rep01。完成轮会跳过；seed 锁阻止重入。若只先跑 seed42 rep01，在命令末尾加 `42-first`。

```bash
tail -F /home/yangchengyu/asg_oldfirst_20260923_release4/seed{42,43,44}.log
```

同版本中断续跑：在启动命令末尾加 `42 --resume`（或 43/44）。原完整性和未知用量守卫不豁免。

全部五轮后：

```bash
/home/yangchengyu/asg_alfworld_venv/bin/python -m experiments.run_v3_released_frozen aggregate \
  --plan /home/yangchengyu/asg_oldfirst_20260923_release4/evaluation_plan.json \
  --output /home/yangchengyu/asg_oldfirst_20260923_release4/report
```
