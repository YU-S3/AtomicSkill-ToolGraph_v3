# OfficeQA fresh seed42 累计预算更新

用户本轮明确要求提高总额度。本轮仅更新费用结转、预算变更记录和启动核查，不改变 Learner/Builder/Agent 提示、晋升、试用、repair、工具权限或评分。没有模型 HTTP，也没有启动正式实验。源码提交 `80d6404d65074e853c8018c236839b2cbf318bb9`；固定 checkout 为 `/home/yangchengyu/asg_officeqa_budget85m_fixed_20261013`。

| 项目 | 冻结值 |
| --- | ---: |
| Train | 120 × 400,000 = 48,000,000 tokens |
| Val＋Test | (24＋102) × 96,000 = 12,096,000 tokens |
| 新实验理论 token 封顶 | 60,096,000 |
| 旧费用结转 | 24,567,954 tokens／3,021 HTTP |
| 累计理论 token 封顶 | 84,663,954 |
| 新累计封顶，取整 | 85,000,000 tokens／480,000 HTTP |
| 新实验剩余额度 | 60,432,046 tokens／476,979 HTTP |

请求按现有循环计算：Train 原生 64 次、Val/Test 24 次；每题求解逻辑请求上限 `2 * max(16, 4 * native_calls + 16) + 5`，Train 学习另外 4 次，HTTP retry 最多 4 次。新的请求理论上限为 `(120 * (549 + 4) + 126 * 229) * 5 = 476,070`；加旧费用后 479,091，取整为 480,000。这些是封顶，不是预期消耗；不再自动扩额。

Train 的 400,000 仍分为求解 320,000、学习 80,000；Val/Test 每题 96,000，单题预算和原生调用上限没有再次增加。原账本字节不变，新账本精确去重结转费用，写入 `budget-revision.v1` 授权记录；不继承 Bank 或已完成题目。新 gate 接受这个明确的预算版本，同时拒绝原费用账本改变。

固定提交的预算／gate 回归：20 passed、2 skipped（两项依赖未指定旧归档）；另有原统一配置回归 15 passed、29 skipped。真实费用账本的生产启动准备与 Governor 身份一致性检查通过，旧账本未修改；完整 Office 120/24/102 资源预检、Docker 和 shell 语法／`--print` 检查通过，均为零模型 HTTP。与此前 `eb7ee3d` 比较，生产 Python 只变化了预算处理与工程 gate 两个文件，方法实现字节保持一致。

准备文件及冻结 hash 在 `D:/T3S_exp/deliveries/outputs/officeqa_fresh_budget85m_20261013/`，包括 `budget_revision.json`、`preflight.json`、`launcher_budget_check.json`、`resolved_config.json` 和费用来源映射。旧审查包及旧固定版本保留。

在 WSL 使用本次新启动指令：

```bash
bash /mnt/d/T3S_exp/deliveries/outputs/officeqa_fresh_budget85m_20261013/start_officeqa_seed42.sh
```

新目录：`/home/yangchengyu/officeqa_budget85m_fresh_seed42_formal_20261013/officeqa/seed42`。从空 Bank 开始，Train120 → Frozen → Val24 → Test102；前六题工程 gate 是正式 Train 的组成部分，通过后同目录自动继续。

```bash
tail -f /home/yangchengyu/officeqa_budget85m_fresh_seed42_formal_20261013/runner.log
```

```bash
cd /home/yangchengyu/asg_officeqa_budget85m_fixed_20261013
PYTHONPATH=src /home/yangchengyu/asg_alfworld_venv/bin/python -m atomic_skillgraph.experiments.monitor_cells /home/yangchengyu/officeqa_budget85m_fresh_seed42_formal_20261013/officeqa/seed42
```

不能据现有 0 个自然 usable Program 预测第几题出现复用或降本。若第一题末产生 usable，后续第一道兼容题即可有调用机会。建议在已完成第 6/12/24 题时读取既有记录，观察 usable 数量、实际调用和同类任务的求解 tokens；学习成本单独列出。出现 Program 调用并不自动证明整题总 tokens 已下降，也不添加强制生成、强制调用或性能放行规则。
