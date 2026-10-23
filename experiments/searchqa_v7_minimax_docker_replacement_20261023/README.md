# MiniMax-M3 × V7：Docker 故障 200 题替换与 Test 续跑

用户要求：第 518–717 题提交 Program 因 Docker 无法连接而失败，旧成本不计，重新运行这 200 题。此前完整 Test1400 授权继续有效。

## 固定范围与已有结果

这 200 题按原 canonical 顺序一次性确定，逐题存在真实 Docker daemon 连接失败证据，未按正确或错误结果筛选。它们原有的 150/200 分数、1,659,185 tokens 和 439 HTTP 从**有效实验结果及成本**中剔除；原始轨迹与物理费用账本保持不变。

| 部分 | 正确 / 完成 | EM | 已计量 tokens | HTTP |
| --- | ---: | ---: | ---: | ---: |
| Qwen-3.8-flash × 同一 V7，完整 Test | 1191 / 1400 | 85.0714% | 5,075,412 | 1400 |
| MiniMax 保留的无 Docker 故障部分 | 715 / 897 | 79.7101% | 3,056,694 | 898 |
| MiniMax 被替换的旧 200 题 | 150 / 200 | 75.0000% | 1,659,185（剔除） | 439（剔除） |

MiniMax 原运行共完成 1097 题，第 1098 题出现 `Invalid provider SSE event`：收到 739 个流式事件、10185 字符 reasoning，未得到最终 content 或 usage。现有记录不足以确认具体服务端原因。本轮保留该请求的 77,453 token 未知费用预留，并为这道未评分题记录一次明确续跑授权；更早的 77,459 token 预留同样保留。未知预留合计 154,912，不能当作已计量使用量。

## 执行与费用口径

新目录：`/home/yangchengyu/searchqa_v7_minimax_docker_replacement_20261023`。

1. `replacement200`：重跑原第 518–717 题。
2. `remaining303`：自动接着运行原第 1098–1400 题。
3. 完成后有效结果为 897 道保留题 + 200 道替换题 + 303 道剩余题，按 task ID 去重，共 1400 题。

`run_replacement.py status` / `monitor.py` 同时显示有效实验成本和物理账本。物理累计额度仍为 8,000,000 tokens / 3,000 HTTP，未扩容；原记录、已计量费用、未知预留未删除或改写。完成后的结果写入新根目录 `effective_summary.json`。

模型仍为 `MiniMax-M3`，adaptive thinking、stream=true、completion cap=65536，原单题配置保持不变。运行源码固定于 `5ff29b0a785642aa5ba54918843ebcdc586afae0`，源码 SHA256 `52c3e294dbc76ab1548cf14b152c757c7c5aa4c92b19e12a858059cb94c0e295`。Bank、公开输入、原生图与官方评分器均未修改。

## 本轮基础设施保护

新增版本化 `infrastructure_guard.py`，只用于本次新执行目录：每次模型请求前检查锁定 Docker 镜像是否可用；Program 返回 daemon 连接故障时停止执行，不将基础设施异常交给 Agent 修复或计入评分。

只有退出码 125、明确无法连接 daemon、零工具调用、无 CPU 执行记录、无 workspace、无允许工具、Broker 事件和终止状态均未改变的纯 Program，才记录完整原 checkpoint 收据并清除未启动标记，以允许复用已付费的 solve 回答。无法证明未执行的中断继续保留原副作用恢复屏障，不做盲目重试。

零 HTTP 验收覆盖正常运行、请求前断开 Docker、收到模型回答后断开 Docker、恢复后复用回答、完成后重复 resume 零新增请求、旧账本不变及精确题单替换。真实启动核验首 3 题的最终请求参数、逐字相同的模型可见输入、raw_usage 与账本一致、提交 Program 成功且输出被消费，见 `offline_acceptance.json` 和 `startup_verification.json`。

## 监控（WSL Ubuntu）

```bash
/home/yangchengyu/asg_alfworld_venv/bin/python -B /mnt/d/T3S_exp/deliveries/searchqa_v7_minimax_docker_replacement_20261023/monitor.py
```

持续观察可加 `watch -n 30`；不会新增模型请求。日志为新运行根目录的 `run.log`。

本目录是实际本机执行与审计归档。完整公开题目保留在交付目录的两个 `*_tasks.json`；Git 归档提供 task ID / physical key / SHA256 清单，避免重复提交数据集正文。归档提交号与冻结运行源码提交号分别记录，不伪装为同一个身份。
