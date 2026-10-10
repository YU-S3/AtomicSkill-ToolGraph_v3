# R1 引用兼容补丁及四 benchmark seed42 启动

依据 `AtomicSkill_Unified_R1_引用兼容补丁与四基准单seed启动说明_20261010.md`，本轮生产代码仅修改 `empirical/local_validation.py`，包含前序 Office Runtime 收尾修复。

JSON 输出引用统一兼容旧列表与 `kind=json` 的 typed selector。参数变体读取独立参考结果时使用同一解析入口；缺字段、值不一致和非法结果结构仍失败，记录真实 trial，不逃逸为 Host 路径异常。无文件的 native execute_python 仅成功标志仍拒绝，真实计算结果不再被空集合判断误拒。publication 保持独立，声称文件名但未实际发布仍拒绝。未降低晋升标准、增加 repair 或 Worker 次数，也未改变方法、模型、cap、数据划分或 XLSX 效果政策。

## 无新增模型请求回归

四个受影响模块共 **74 用例通过，0 失败，0 跳过**，收费模型请求 **0**。四模块首轮为 73 passed／1 failed，失败位于新增夹具的诊断 stdout 精确值断言；改用原生计算夹具调用生产 Docker Worker 的结构化结果通道后，定向四例全部通过。70 个未改动用例与最终 4 例去重计为 74，不把前后批次相加，也不计文档附录的进程内 17 例。

正例包含生产 operation_directory 提供的 typed selector（去除显示 type），经过实际提议校验、Host canonical binding、Builder、参数变体、usable 与 Frozen；旧引用同链验证。另验证非源消费及 Frozen 不变、native 无文件计算／仅成功标志的新旧引用、候选／参考缺字段负向 trial、真实 XLSX 发布、无实际发布的文件名反例与错误原生动作反例。

Docker 固定镜像为 `sha256:4db520e20cd121f830731c2d0c0bcfbe9cacd254d1817262404cc18e1e757783`。机器结果见 `atomic_unified_r1_selector_verification_20261010.json`；JUnit 和逐例证据在 `D:/T3S_exp/deliveries/outputs/atomic_unified_r1_selector_20261010/`。

## 连续正式启动

本轮只准备指令，未执行收费启动。四项均用包含本补丁和 Office 修复的同一干净提交，在独立 WSL checkout：

`/home/yangchengyu/asg_atomic_unified_r1_selector_20261010`

新根目录：`/home/yangchengyu/atomic_unified_r1_selector_seed42_formal_20261010`。实际 SHA、旧账本身份及哈希由该目录 `launch_preparation.json` 固定；输入与配置核验见交付目录 `launch_checks.json`。四项为 SearchQA、LiveMath、OfficeQA、SpreadsheetBench，单 DeepSeek、seed42，从各自空 Bank 连续 Train→Frozen→Val→Test；无 max-new-tasks、stop-after-val 或额外诊断臂。

| Benchmark | Train / Val / Test | 原授权 tokens / HTTP | 旧前缀 tokens / HTTP | 实际剩余 tokens / HTTP |
| --- | --- | --- | --- | --- |
| SearchQA | 300 / 24 / 1400 | 196704000 / 83580 | 31294 / 12 | 196672706 / 83568 |
| LiveMath | 60 / 17 / 100 | 23232000 / 9165 | 90221 / 11 | 23141779 / 9154 |
| OfficeQA | 120 / 24 / 102 | 36096000 / 284070 | 73749 / 9 | 36022251 / 284061 |
| SpreadsheetBench | 200 / 20 / 180 | 59200000 / 558000 | 375746 / 47 | 58824254 / 557953 |

旧 19 题全部保留为 R1 调试前缀，不迁移为新版本训练结果。新 run 会重新执行这些成员。仅把旧已计量 HTTP 账本按原请求 ID 原样复制到新预算文件，用既有 Governor 扣在同一授权总额内；不复制旧 Bank、训练记录或模型响应。原封顶 315232000 tokens／934815 HTTP，旧费用 571010／79 后剩余 314660990／934736。旧目录全文件哈希在准备与启动前核对，不能换目录恢复整份额度。

在 Ubuntu WSL 启动：

```bash
nohup bash /home/yangchengyu/start_atomic_unified_r1_selector_seed42_full_20261010.sh --execute \
  > /home/yangchengyu/atomic_unified_r1_selector_seed42_full_20261010.launch.log 2>&1 &
```

只查看四条完整命令可将 `--execute` 换为 `--print`。启动器使用现有 single_cell CLI，为四项分配独立日志、锁、临时目录与账本；拒绝源码变化、账本变化或重复初次启动。不使用旧四项一键 resume，也未改旧 `run_atomic_seed42.sh` 的前五题含义。

```bash
tail -f /home/yangchengyu/atomic_unified_r1_selector_seed42_formal_20261010/logs/*.seed42.log
```

逐题进度、费用和状态以每项 `seed42/episodes.jsonl`、`llm_calls.jsonl`、`run_manifest.json`、分阶段 `summary.json` 与 `completion.json` 为准。旧 Office 保持暂停，执行上述新指令时才创建新 Office run。正常失败、低分或暂未生成 Program 沿原规则记录；合同、账本或基础设施错误停止受影响 cell；额度不自动扩大。

本报告取代上一份报告中旧三个源码副本可直接续跑的启动建议；旧报告作为当时记录保留。
