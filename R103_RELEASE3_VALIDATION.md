# Release3 修复与验收报告

代码：`82f9cb1`（main）。正式 670 题尚未启动。旧 Release2、原 ZIP 和训练库未修改。

## 修改范围

- task/attempt 内的真实搜索检查历史独立于世界回滚保留；只作 Agent 历史，不授权实体输出或当前缺失事实。
- exact failure cache 引用原观察，不增加环境执行。预算中断调用恰好一次记录，异常与费用保持原边界。
- 完成帧补 Agent-before-invocation，terminal effect 单列；current/lean 共用原 Atomic 摘要，完整相同 preview 才去重。
- 最终 HTTP payload 分块字节/hash 可查；报告按最终任务选择分母、按全部 attempt 去重计费。
- 有限 dev 收尾验身份、不可变 Trace 哈希、费用、历史投影和两级报告；正式脚本防重复启动。

## 测试与技能库一致性

- 最终全量 pytest：1787 passed；专项 15 项通过；bash -n 通过。
- S01–S12：tests/test_release3_history.py；S13：dev/automation/acceptance.json（真实环境、既有生产执行器、非首入口、多动作、多 Tool、DATA_FLOW、无 provider 请求区间）。
- S14：新旧逐 ref 正文哈希 + 全数据库表对账（仅根路径和发布时间归一）；现有 source/code/resource guards 保留。
- 旧 40 请求只读复算：365114 tokens；参考历史投影 current 第3次含前2批、lean 第4次含前3批。该项不声称新模型降本。

| Seed | 冻结资产数 | 正文相同 | 数据库语义差异表数 |
|---|---:|---|---:|
| 42 | 198 | True | 0 |
| 43 | 181 | True | 0 |
| 44 | 195 | True | 0 |

## 固定六题结果

| 任务 | Profile | 成功 | Tokens（旧→新） | 请求（旧→新） | 搜索启动 | 检查访问/重访 | 预算调用漏记 |
|---|---|---|---:|---:|---:|---:|---:|
| alfworld_train_2_pick_two_obj_and_place | current | True | 51112 → 40324 | 5 → 5 | 2 | 3/0 | 0 |
| alfworld_train_2_pick_two_obj_and_place | lean | True | 48410 → 43010 | 5 → 6 | 2 | 3/0 | 0 |
| alfworld_train_34_look_at_obj_in_light | current | True | 68734 → 109308 | 6 → 9 | 1 | 1/0 | 0 |
| alfworld_train_34_look_at_obj_in_light | lean | True | 35225 → 145659 | 4 → 11 | 1 | 1/0 | 0 |
| alfworld_train_36_pick_heat_then_place_in_recep | current | True | 72782 → 62658 | 9 → 7 | 4 | 24/0 | 0 |
| alfworld_train_36_pick_heat_then_place_in_recep | lean | True | 88851 → 66951 | 11 → 8 | 4 | 21/0 | 0 |

### 完整输入与费用变化

| Profile | Tokens旧→新 | 实际messages bytes旧→新 | tools bytes旧→新 |
|---|---:|---:|---:|
| current | 192628 → 212290 | 389095 → 413162 | 130495 → 131477 |
| lean | 172486 → 255620 | 324867 → 444683 | 132216 → 156941 |

共享能力探针另计：4 次请求、1800 tokens，保留 dev/provider_probe 原始记录。
六题全部 attempt 合计 467,910 tokens；上一轮 365,114，本轮增加 28.15%。
全部 46 次最终请求的历史投影校验通过，摘要缺失 0；本轮移除 14 条完全相同 preview，保留原始列表审计。
新历史与摘要的输入成本已包含，不按局部去重字节宣称净节省；不同模型决策/请求次数存在随机性。这六题不是因果降本实验。
current look 的高成本轨迹包含两次 support_atomic_output_mapping_invalid 拒绝及一次相同调用负缓存；模型随后通过原生探索和合法照明辅助完成。拒绝未产生搜索执行/实体输出，没有因降本而放宽映射权限。此项是保留的策略低效样本，不应表述为所有任务都已降本。
lean look 有三次映射拒绝、两次 support_not_execution_ready；其中一次模型响应 completion 达31462 tokens。最终使用已观察到的具体实体完成，未绕过 grounding 边界。
普通失败保留，不追加抽样；预先固定 current，不按任务择优选 profile。

## 产物

- WSL 根：`/home/yangchengyu/asg_oldfirst_20260922_release3`
- `dev/`：六个 episode 的 traces、最终 payload、all_usage、attempt、SQLite、manifest 以及比较/自动专项/最终验收。
- `seed42/43/44/frozen/`：三个独立重新封存库；`reseal_comparison.json`：逐资产哈希与对账。
- `configs/` + `evaluation_plan.json`：seed42 3轮顺序、seed43/44各1轮，三条流并行。

## 正式启动（用户执行）

```bash
bash /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/scripts/run_released_frozen_parallel.sh /home/yangchengyu/asg_oldfirst_20260922_release3
```

```bash
tail -F /home/yangchengyu/asg_oldfirst_20260922_release3/seed{42,43,44}.log
```
