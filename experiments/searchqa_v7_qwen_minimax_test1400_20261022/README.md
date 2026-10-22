# SearchQA V7：Qwen3.8-Flash / MiniMax-M3 Test1400

2026-10-22，按用户要求各启动一次完整 Test1400。两个进程、目录和模型账本独立；无 Train、无额外付费冒烟、不继承旧题目的预测。

使用原 DeepSeek 85.00% 实验的完整 `v7_text_compiler` Bank。输入包、Frozen Bank、canonical 1400 题成员和顺序、公开 context、提示文本、原生 solve→submit 图与官方评分保持一致。人工 Bank 身份仍为 author_trusted；不伪造学习或晋升证据。

| 配置 | Qwen | MiniMax |
|---|---|---|
| 实际 model ID | qwen3.8-flash | MiniMax-M3 |
| API | maas.qianwenaiapi.com/compatible-mode/v1 | api.minimax.cn/v1 |
| 密钥环境变量 | DASHSCOPE_API_KEY | MINIMAX_API_KEY |
| 思考 | enable_thinking=true；thinking_budget=131072 | thinking.type=adaptive |
| 主请求输出 cap | max_tokens=65536，正文与思考分开 | max_completion_tokens=65536，包含思考 |
| effort | 配置 high；显式 thinking_budget 生效，不重复发送 effort | 不发送不受支持的 effort |
| 单题总上限 | 400000 tokens | 128000 tokens |
| generation seed | 42，不保证确定性 | unsupported，不发送 |
| timeout / 内部重试 | 600 秒 / 0 | 600 秒 / 0 |
| 本次整段上限 | 8000000 tokens / 3000 HTTP | 8000000 tokens / 3000 HTTP |

这是按已配置模型参数进行的迁移，不能称为各模型具有完全相同的输出预算。Qwen 配置来自已修正的正文/思考独立预算版本，账本同时预留两部分；不是把思考费用漏计。现有 LiveMath 与 Provider 验收账本独立保留，不改写也不重复并入本次 SearchQA。

冻结源码：`/home/yangchengyu/asg_searchqa_qwen_minimax_release_20261022`，commit `5ff29b0a785642aa5ba54918843ebcdc586afae0`。以先前 SearchQA 冻结版本为基础，仅移植已验证的两模型 Provider、Qwen 输出预算计量，以及 EmpiricalSystem 参数转交；保留既有显式拒绝处理，不引入新 LiveMath/Office 方法逻辑。完整差异见 `source_changes.patch`，Git 对象见 `source.bundle`，来源见 `source_provenance.json`。

验收：10 组零 HTTP 生产链路检查通过，含实际 Docker Program、官方评分、流式事件时间戳、MiniMax 正常 EOF、显式拒绝、缺失 usage、断流与完成后续跑零新增调用；2800 个待发公开输入预算检查通过。既有 Provider transport 回归 3 passed。首批真实请求核验见 `startup_verification.json`：每个模型至少 3 道完整任务，检查响应模型、最终参数、完整 messages、raw_usage、Program 消费与账本。前缀结果不代表完整准确率。

运行根目录：

```text
/home/yangchengyu/searchqa_v7_qwen_minimax_test1400_seed42_20261022/
  qwen38_flash/
  minimax_m3/
```

每个目录的 `fulltest1400/v7_text_compiler/test/` 保存 SQLite、逐题轨迹和物理请求；根目录有 `budget.json`、`run.log`、`pid`。Bank 与 plan 文件运行期间冻结。中断或未知费用会保留现场停止，不能通过伪造零 usage 或把基础设施错误计作答案继续。

WSL 监控：

```bash
watch -n 20 /home/yangchengyu/asg_alfworld_venv/bin/python -B /mnt/d/T3S_exp/deliveries/searchqa_v7_qwen_minimax_test1400_20261022/monitor.py
```

`admitted=1` 通常表示当前在途请求；`unknown_billing` 是另一个状态。以 process_alive、当前题开始时间、run.log 和终止错误共同判断。

仅在进程已经结束且核对中断记录后，才使用单模型续跑入口；正常运行时不得重复启动：

```bash
/home/yangchengyu/asg_alfworld_venv/bin/python -B /mnt/d/T3S_exp/deliveries/searchqa_v7_qwen_minimax_test1400_20261022/run_models.py start --model qwen38_flash --resume
# MiniMax 则使用 --model minimax_m3
```

交付不含密钥，不修改其他正在运行的实验。
