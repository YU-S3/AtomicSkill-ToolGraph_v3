# MiniMax SearchQA V7：HTTP 529 后续跑

2026-10-23 00:56（北京时间）启动。用户要求“minimax能续跑吗”，据此前启动完整 Test1400 的授权，在原累计额度内恢复。

原 run 已完成 254/1400，216 题正确。第 255 题 HTTP 529 返回 `overloaded_error`，没有答案或 usage。此次保留父 run 所有文件、254 题评分和全部物理请求账目；只对这次已保存的明确过载错误建立恢复授权，重新执行中断题，再按 canonical 顺序运行后续任务。不会重跑已经答错的完成题，也不会把基础设施错误计作答错。

旧请求的 **77,459 tokens 保持未知费用预留**，不改为实际使用量或零费用。所有旧物理请求 ID 只结转一次。累计上限保持 **8,000,000 tokens / 3,000 HTTP**，没有扩容。子 run 为恢复尝试提供独立任务执行 scope，单次执行仍用原 128,000 tokens / 65,536 completion 配置；旧过载请求预留继续占用共享累计额度。对后续未知账目没有授予无限重试：再遇到未知费用错误会保留现场停止。

方法、模型、公开输入、策略提示、V7 Bank、原生 solve→submit 图、Docker 和官方评分均未修改。执行源码仍为 `5ff29b0a785642aa5ba54918843ebcdc586afae0`。父 run 保留 stopped 历史状态；进度应查询本目录的新监控脚本，而非旧 MiniMax 目录。

零 HTTP 验收覆盖真实 Docker Program 消费、原请求 messages/参数不变、账本和累计上限不变、完成后重入零新增调用，以及新过载出现时停止且不评分。恢复后至少三道真实题通过同项核验，首个中断题已经完成。详见 `offline_acceptance.json`、`startup_verification.json`。

当前运行目录：

```text
/home/yangchengyu/searchqa_v7_minimax_test1400_resume_20261023
```

父 run：

```text
/home/yangchengyu/searchqa_v7_qwen_minimax_test1400_seed42_20261022/minimax_m3
```

WSL 监控（同时显示 Qwen 最终结果和 MiniMax 合并进度）：

```bash
watch -n 20 /home/yangchengyu/asg_alfworld_venv/bin/python -B /mnt/d/T3S_exp/deliveries/searchqa_v7_minimax_resume_20261023/monitor.py
```

`new request states` 中的 admitted 是在途请求；历史 unknown 预留独立列出。无需重复启动进程。

GitHub 归档保留配置、代码、原始过载证据、身份与验收记录。1146 题正文仅保留在本地 `remaining_tasks.json` 及实际 run 内，未向仓库重复复制；对应文件 hash 在 plan 和归档清单中冻结，`specs` 可从不变的父题单重新生成。
