# OfficeQA Runtime R2 修复与验收（2026-10-12）

工程链路已通过离线回归与有界付费检查，可以提供新 fresh seed42 启动入口。准确率仍弱：固定六题 Train 为 0/6，两题 Val 为 1/2；这不是完整正式实验的成绩，也没有证明自然 Program 的学习与复用效果。

## 受测版本与改动

- 受测 commit：`60324e7828d9e0e550f5d9520d8341389e22e0bd`。
- Python 源码身份：`d0d76da0c22969fafb55dad7cd4aa1763f48a519829b67d91e0bdb37914991b5`。
- Runtime 投影：`empirical.model-view.public-windows.v5`。
- 保留实际 public Runtime content/detail，最多四条、每条 1024 字符，标为 model_reported；这些工作记录不作为独立来源证据或晋升依据。
- 修正 output_aliases 的实际输出示例，Producer 的待交付结果保留后可修正交接；提示明确要求使用实际字段、公开检索结果和可用计算工具。
- Office 原生 read 注册 `native.text-page.v1`：已取得的 12,000 字符页面直接参与模型请求；请求预算不足时按源单位收缩并记录实际窗口。一般结果的 2,048 字符预览不再把该页面提前缩到约 1,500 字符。
- Runtime read_result Schema 明确要求 path。字典根仍拒绝，错误指出可读的公开字段路径；Host 字段、越界目录、未知结果 ID 仍不可读。已取得的独立只读页可按现有三调用上限批处理。
- finish_trigger 独立记录预算／工具上限的收尾原因，便于区分正常有界收尾与工程错误；没有扩大每题预算或修复次数。
- 有界检查入口在任务边界停止时保留停止收据，不再继续尝试未产生的冻结库。

这次在隔离分支修改，已合并此前 LiveMath P2 提交；付费运行使用独立固定的 Linux checkout。没有向公共算法写入题目 ID、gold、文件名配对或答案规则，没有继承旧 Bank，也没有手工晋升任何资产。

## 验证事实

离线回归：253 passed，21 skipped，0 failed，0 errors。跳过项需要另外指定历史归档，不是本轮新回归或 Docker 测试。覆盖真实 Docker Worker、公共窗口、引用与绑定、预算／重试账本、学习材料、Program 验证标准、冻结、只读评分与断点恢复；Office execute_python 的实际注册 Schema 和 Docker 计算已有生产回归通过。

完整资源预检：Train120、Val24、Test102 共 246 题均通过实际适配器 reset；公共 canonical ID 和物化顺序一致、split 互斥、公共字段不包含 gold、模型可用四个文件工具；Val/Test 使用 train/frozen_bank。资源和 corpus 的 hash 已保存。本轮没有支付完整 246 题，完整 Test 的控制流和冻结不变性由离线回归覆盖。

付费检查均使用 canonical seed42 Train 前缀。第一次检查复现原生 read 投影和 read_result Schema 不一致，4 题后停下，费用和失败原始现场保留，未伪装成通过。

| 检查 | 题数／正确 | Tokens | HTTP |
|---|---:|---:|---:|
| 初次检查，停止 | Train 4／0 | 328,047 | 52 |
| 修复后的固定版本 | Train 6／0 | 536,897 | 73 |
| 同次固定版本冻结验证 | Val 2／1 | 122,242 | 17 |

最终检查：从空 Bank 完成六题学习与逐题持久化，自动 Frozen 后执行两题 readonly Val；冻结前后 tree hash 均为 `022d4aff1009f27bd5f9002ac97d91c67b4fae322d2b1b77e5375a6e7b48b5f8`。没有基础设施错误、learning_error、unknown_billing 或漏记 HTTP。

六题学习为四次 no_change、两次 proposal_invalid；后两次在原有一次结构修复后仍返回拼接 JSON（Extra data），按原规则拒绝。这些是模型提案失败。最后 Bank 没有 Skill/Program/Workflow；不能声称付费检查已经证明自然 Program 复用。模型仍会选错检索范围、重复搜索、用显示包装字段组成错误路径、缺少实际计算；这些原始失败均保留，没有自动纠正模型输出或改评分。

## 记录、预算与启动

固定 checkout：`/home/yangchengyu/asg_officeqa_runtime_r2_page_check_20261012`。

通过的付费检查：`/home/yangchengyu/officeqa_runtime_r2_page_paid_check_20261012`。

本机审查材料：`D:/T3S_exp/deliveries/outputs/officeqa_runtime_r2_20261012/`，含两批真实 Trace、请求／响应、配置、选择清单、Bank、Frozen、评分／学习日志、账本、XML 回归报告、完整资源预检和源码差异。旧正式 run 已停止在 56 个已完成任务，记录与原费用保留。

原 Office 累计额度仍是 36,096,000 tokens／284,070 HTTP。旧费用与本轮两批检查去重后累计 17,925,950 tokens／2,270 HTTP，剩余 18,170,050 tokens／281,800 HTTP。启动时结转这些费用，仅结转账本，fresh Bank 仍为空。额度不足按原规则停止，未保证剩余额度必然覆盖完整正式运行。

在 Ubuntu WSL 执行：

```bash
cd /home/yangchengyu/asg_officeqa_runtime_r2_page_check_20261012
tmux new-session -d -s officeqa42 \
  "cd $PWD && OFFICE_R1_VALIDATION_COST_SOURCE=/home/yangchengyu/officeqa_runtime_r2_validation_costs_20261012 bash scripts/run_officeqa_r1_seed42.sh --execute 60324e7828d9e0e550f5d9520d8341389e22e0bd > /home/yangchengyu/officeqa_runtime_r2_fresh_seed42_formal_20261012.log 2>&1"
```

运行范围是 Train120 → Frozen → Val24 → Test102。前六题属于此次正式 Train，工程 gate 通过后同版本、同目录继续，没有额外重复前缀。该正式目录尚未启动。

监控：

```bash
tail -f /home/yangchengyu/officeqa_runtime_r2_fresh_seed42_formal_20261012.log
```

输出位于 `/home/yangchengyu/officeqa_runtime_r2_fresh_seed42_formal_20261012/officeqa/seed42`。若需要恢复同版本运行，使用相同脚本和 commit 将 `--execute` 改为 `--resume`；人为设置 STOP_AFTER_TASK 后须显式撤销停止标记，不跨版本 resume。
