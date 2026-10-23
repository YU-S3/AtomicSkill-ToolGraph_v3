# Gemini-3.8-flash × V7：SearchQA Test1400

根据用户“启动配置好的 gemini-3.8-flash 的 test”，已于 2026-10-23 14:47（北京时间）启动一次独立完整 Test1400。沿用原 V7 text compiler Bank、Atomic/Workflow/Implementation/Program、模型可见消息、完整公开输入、canonical 顺序和官方 EM/F1 评分。

## 实际配置

| 项目 | 值 |
| --- | --- |
| 接口 | RootFlow OpenAI-compatible relay |
| Base URL | `https://api.rootflowai.com/v1` |
| 请求／首批响应 model | `gemini-3.8-flash` |
| Key 环境变量 | `ROOTFLOWAI_API_KEY`，未归档密钥 |
| reasoning_effort | `high` |
| 输出上限 | `max_tokens=65536`；finish_only 为 16384 |
| 单题父预算 | 128,000 tokens |
| Stream / timeout | true / 600 秒 |
| Provider 内部重试 | 0 |
| 运行 seed | 42；generation seed 不支持，未发送 |
| 本次执行保护上限 | 独立 8,000,000 tokens / 3,000 HTTP，无自动扩容 |
| 运行目录 | `/home/yangchengyu/searchqa_v7_gemini38_test1400_seed42_20261023` |
| 冻结运行源码 | `28ac1ca31cc5f7f1b3d5772fedd3f84c489cebf9` |
| 源码 SHA256 | `ffe84c8a1718ae7286d479f07b4828a9613a0f1bb0c7cd9f8326c2ed805a9486` |

配置来自已存在的 `rootflow_gemini38_configuration_20261023/rootflow_gemini38_flash.yaml`。本次是 SearchQA 的独立账本；已有 Gemini 接口检查与 LiveMath 的原始费用留在各自账本，未挪用其额度或重复计费。请求参数与中转返回的 model 均有原始记录；这不是对上游内部实现的独立认证。

## 适配和核验

基于此前 SearchQA 冻结源码 `5ff29b0`，只移植已实测的 Gemini Provider 适配（上游提交 `1d7814b`、`4005f1a`），保留 SearchQA 已有拒答处理。另修复离线复现的摘要记录问题：中断后复用付费回答时，将原始请求审计记录与 usage 一并带回逐题摘要。方法逻辑、Bank、公开输入和评分不变。

沿用已验证的 Docker 保护：模型请求前检查镜像；Docker 连接失败时暂停，不交给 Agent 补救或评分。只有明确未执行的纯 Program 才允许保存重建收据后复用已付费回答，其他不明副作用继续阻止盲目续跑。

零 HTTP 验收已通过 6 组生产路径：正常执行、带 usage 的拒答、缺失 usage 的明确拒答、断流、缺失 usage、收到回答后 Docker 失败并恢复。覆盖全部 1400 题请求容量，最高保守准入值 78,188，低于单题 128,000。真实 Docker / 官方评分 / 完成后 resume 零新增调用均验证通过；通用 transport 测试另有 3 passed。

启动核验覆盖首 3 道真实 Test 请求，未新增诊断调用：模型 ID、high、输出上限、逐字相同的模型输入、raw_usage 与费用账本、Program 执行和输出消费均通过。核验时已完成 12/1400，见 `startup_verification.json`；这是启动时快照，不是完整 Test 结果。

## 监控

在 WSL Ubuntu 运行：

```bash
watch -n 30 '/home/yangchengyu/asg_alfworld_venv/bin/python -B /mnt/d/T3S_exp/deliveries/searchqa_v7_gemini38_test1400_20261023/monitor.py'
```

日志：`/home/yangchengyu/searchqa_v7_gemini38_test1400_seed42_20261023/run.log`。

结果与逐题审计位于运行目录的 `fulltest1400/v7_text_compiler/test/`，总进度位于 `fulltest1400/result.json`，累计费用位于根目录 `budget.json`。

源码补丁、Git bundle、Provider 配置、plan、Frozen/input hash、离线与启动收据一并归档。归档提交号与运行源码提交号分开记录。原 MiniMax、Qwen 及其他运行目录未修改。
