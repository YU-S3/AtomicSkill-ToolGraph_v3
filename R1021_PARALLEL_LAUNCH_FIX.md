# R10.2.1 seed43/44 启动修复（2026-09-20）

旧批次 `/home/yangchengyu/asg_r1021_parallel_MHo43n` 两个 seed 在正式协议入口拒绝，未开始任务。之前只增加配置和 API 探测，遗漏 train/frozen-test 的协议注册，这是启动适配遗漏，非 API 或并发故障。旧失败日志保留。

代码提交 `10a052e` 为 R10.2.1 两端明确注册 seed42/43/44；未放宽名称、seed、预算、冻结来源配对或题集检查。算法和现有 seed42 配置不变，运行中的 seed42 checkout 未修改。

启动脚本在两路 API 探测之前调用正式 runner 的 selection、formal config 和固定 manifest 校验，检查各自 train/test 配对。任一路失败均不启动正式训练。每个 seed 独立目录、空 bank，训练成功后才执行对应 test。

验证：新增 8 项真实配置入口与错配拒绝测试通过；全量 **1549 passed / 58.13s**。更新脚本 `--probe` 实际运行 exit=0，两路正式配置检查与真实 provider capability 探测均通过。证据：`/home/yangchengyu/asg_r1021_parallel_oIcLTO/seed43/provider_probe.log` 与同级 `seed44/provider_probe.log`。没有启动正式训练，也未将短请求探测声称为完整并发负载验收。

新启动命令（WSL，只执行一次，无需 git pull）：

```bash
bash /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/scripts/run_r1021_remaining_seeds.sh --launch
```

脚本建立新的 Linux 批次目录，打印监督 PID 和目录；不会续用失败批次或改动 seed42。启动后日志为每个 seed 目录的 `train.log`、`test.log` 和 `launcher.log`。
