# LiveMath HumanFrontier R3.1 — Programmatic Atomic SkillBank

**人工资产参考组。不是 Ours 自然进化；不是已经实测提分的 Test 上界。**

这个交付纠正原 R3 只有 `skill=3, program=0, implementation=0, workflow=2` 且仅 `prompt_compiled` 的问题。R3.1 把可确定的公共选项解析交给真正有源码的 Docker Program，把不可确定的研究数学定理判断留给 Dynamic Agent。

## 原生资产清单

- `bank.json` / `bank.sqlite3`：**4 Atomic / 3 Program / 3 Implementation / 2 Composite**，每个 Atomic 含类型化 I/O、前提、effects、provenance 和说明性验证状态。SQLite 为可浏览人工快照；运行时以 `tools/import_native.py` 走你仓库真实 `Bank.put` 导入。
- `programs/`：3份完整 Python 源码，均为 `def run(ctx,inputs)`、无网络、无外部模型/金标、`allowed_tools=[]`。
- `workflows/`：`workflow_lm_r31_meta_prior` 含 `scan → prior_commit` 两个 Program occurrence；`workflow_lm_r31_claim_reasoning` 含 `scan → contrast → theorem_decide`（最后一步 Dynamic）。两者均有 `after/args/from/field/outputs`。
- `integration/native_programmatic_runtime.py`：**实际执行程序节点和数据流**，可接你原生 `ProgramWorker`，不是 Prompt Graph；`integration/episode_bridge.py`：可调用原 R2 单次模型解题，生成独立 R3.1 结果和 Program Trace，不写旧账本/旧结果。
- `tools/native_smoke.py`：Linux + Docker 的真实 Worker 小测试，0模型HTTP；`tests/`：纯离线检验。
- `reference/`：此前 R3 继承的公开 SkillOpt / SkillOpt-Lite 原始参考正文与许可证，用于人工作者内容溯源；不冒充 Ours 学习产物。

## 两条运行分支

```text
Task(public goal, labeled choices)
            |
            v
  [Atomic1: public_choice_scan]  ← Program 真实执行
            |
            +── benchmark_prior && unique meta ──> [Atomic2: meta_prior_commit]
            |                                      Program; benchmark prior, not theorem proof
            |                                                  |
            |                                               final label
            |
            +── no unique meta / theorem_reasoning ──> [Atomic3: claim_clause_features]
                                                         Program; surface hints only
                                                                  |
                                                         [Atomic4: theorem_decide]
                                                          Dynamic real target LLM
                                                                  |
                                                               final label
```

**两个版本含义不同**：

- `benchmark_prior`：对**唯一且可从公开选项文本识别**的元选项，允许程序化直接选择其真实显示标签（当前 LiveMath 构造规律的得分导向先验）；其余任务交由一次目标模型 Dynamic 解题。它不能代表一般数学证明能力。
- `theorem_reasoning`：禁用先验直接提交，所有题经 `claim_clause_features → Dynamic`，结构化词面差异只是提示，不宣称数学证明。

在元选项重复、无匹配、Program返回 `needs_input`、Docker失败等情况下，不编造最终标签，退回完整题面 Dynamic。不要将已观测 Test 的正确选项、题号或隐藏 proof 放入 Bank。

## 本地只读检测（不产生模型请求）

```bash
cd LiveMath_HumanFrontier_R3_Programmatic_R31_20261017
python -m unittest discover -s tests -v
python -m py_compile programs/*.py integration/*.py tools/*.py
```

## 使用你的真实仓库导入及 Worker 检验

```bash
python tools/import_native.py \
  --repo-root /path/to/AtomicSkill-ToolGraph_v3 \
  --output /new/isolated/r31_native_bank

python tools/native_smoke.py \
  --repo-root /path/to/AtomicSkill-ToolGraph_v3 \
  --bank-root /new/isolated/r31_native_bank
```

**特别注意：**`Bank.freeze()` 会滤掉未经过 *Ours* 训练资格的 candidate Program，**人工路线不调用它后假装仍保留3个 Program**。人工独立新目录使用真实 `Bank.put`、只读 Bank 和 `author_trusted` ProgramWorker 执行授权，不修改共享系统的 `program_eligible`。没有实际 Docker 不得宣称本机原生 Program 已验收。

## 真正执行（需要你本地既有运行器接线）

已提供可调用的 Python 接口：

```python
from integration.episode_bridge import solve_one

# 由现有同协议 campaign runner 提供 task、adapter、只读 bank、bundle、
# llm、root、真实累积 governor。单题另开目录；禁止热替换旧 R2/R3 运行。
receipt = solve_one(
    task, adapter, bank, bundle, llm,
    root=episode_dir, governor=shared_governor,
    variant="benchmark_prior", phase="train_development"
)
```

`solve_one` 在 meta 分支运行真实 Worker 并用 Host scorer 提交结果，**不调用目标模型**；Dynamic 分支使用项目已有 `human_frontier_runtime.solve`，保留原本 65,536 主 completion、128,000 parent token 和收尾策略，另外保存 graph Program receipts。必须通过原生 `native_smoke` 后，再由你当前 campaign runner 调用；不能只用这个函数替代任务列表、评测私有记录、累计账本、运行断点控制。

原来的 `HumanFrontier R2 57/100` 是旧文字版实测，不是 R3.1。R3.1 不是当前官方新结果；之前所说 `77/100` 也只是已见 Test 上的离线反事实分解。人工 Bank 内容编辑应使用 Train 开发题进行新实验，旧 Test 不作为新盲测。

## 2026-10-17 本地生产接入版本

`run_human_frontier_r31` 已接入独立 campaign：固定公开哈希排序的 Train4（唯一元选项与非元选项各 2 题）接口验收，之后同一版本只读 Test100。本次只运行 `benchmark_prior` 一臂。Train4 不设准确率门槛，也不据得分修改内容。Test 已有开发暴露历史。

生产桥接按实际 Workflow 的 `after/args/from/field/outputs` 执行，绑定真实 Implementation，保存 Docker 执行收据；Dynamic 注入本包新 Atomic 的 guidance。`reference/` 是溯源材料，不作为隐藏替代提示。原始 bank.json、SQLite 和三份 Program 源码未修改。原包校验清单保存为 `UPSTREAM_FILE_SHA256.json`；`FILE_SHA256.json` 是当前接入版本清单，`INTEGRATION_PROVENANCE.json` 记录差异与原 ZIP 身份。

固定预算：development 512,000 tokens / 8 HTTP，Test 8,000,000 / 200 HTTP，继续受累计 45,000,000 / 9,165 约束。旧费用及保守预留按前序已完成 R2 账本逐项结转。阶段预算不借用，不补跑错误题。主解题 65,536 completion、enabled/high、600 秒、父预算 128,000；一次既有 2,048 completion 收尾，HTTP retry=0。

在干净且已提交的 Linux checkout 中：

```bash
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
PY=/home/yangchengyu/asg_alfworld_venv/bin/python
RUN=/home/yangchengyu/livemath_human_frontier_r31_seed42_20261017
"$PY" -m atomic_skillgraph.experiments.run_human_frontier_r31 prepare --root "$RUN" --previous /home/yangchengyu/livemath_human_frontier_r2_seed42_20261016
"$PY" -m atomic_skillgraph.experiments.run_human_frontier_r31 check --root "$RUN"
"$PY" -m atomic_skillgraph.experiments.run_human_frontier_r31 execute --root "$RUN" --env-file /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env
"$PY" -m atomic_skillgraph.experiments.run_human_frontier_r31 monitor --root "$RUN"
```

`check` 对全部 104 个公开输入执行真实 Docker，并拦截构造 208 份 Provider 请求与预算准入，零模型 HTTP。`execute` 使用进程锁及哈希校验；已完成响应和 Program 收据可原样恢复，缺结束收据的 Program 或计费未知的请求阻止盲目重放。

整题的权威记录为 `r31_result.json`。`dynamic/trace.json` 仅描述数学模型子调用，其中 program_calls=0 不代表整题未执行程序。Program 输出交接、实际图执行、CPU/wall 与模型原始 usage 分开记录；提示暴露不等同于证明模型语义采纳。
