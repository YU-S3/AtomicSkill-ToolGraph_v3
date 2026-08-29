# AtomicSkillGraph v3

AtomicSkillGraph v3 是一个独立的、基于 native ToolCall 的集中式原子
Skill/Tool 联合进化实验系统。它使用 Planner 构建严格线性的 Runtime 控制序列，
用 Runtime Agent 在真实 Harness 中认证参数并调用 Implementation，通过分层验证、
append-only EvidenceLedger 和生命周期投影完成可审计的长期进化。

完整语义与不变量见
[v3.0 设计文档](AtomicSkillGraph_集中式原子SkillGraph与Tool联合进化_完整设计文档_v3.0_独立重构版.md)。

## 独立性边界

v3 的运行时实现只来自本仓库的 `src/atomic_skillgraph/`：

- 不把 v3 定位为 FlowEvo 扩展；
- 不 import `D:\T3S_exp\AtomicSkill-ToolGraph` 的 v2 代码；
- 不 import `D:\T3S_exp\FlowEvo-main`；
- 不加载或迁移 v2 bank；正式训练从空的 schema v3 bank 开始；
- ALFWorld 只通过 v3 Harness 边界接入，Core 不解析任意文本动作。

v2 和 FlowEvo 中经 contract 审计后可复用的思想已经被独立重写；它们不是 v3
安装、导入或运行所需的路径。

## 环境要求与安装

- Python 3.10 或更高版本；
- 真实 ALFWorld 实验需要 ALFWorld 数据；
- deterministic smoke 不需要模型 API 或 ALFWorld。

在仓库根目录创建独立环境：

```bash
python -m venv .venv
```

PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[alfworld,dev]"
```

Linux/WSL：

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[alfworld,dev]'
```

本机可直接复用 v2 已有的 WSL 实验环境；其 Python 和 ALFWorld 数据路径已经验证：

```bash
cd /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3
/home/yangchengyu/asg_alfworld_venv/bin/python -m pip install -e '.[dev]'
export ALFWORLD_DATA=/home/yangchengyu/.cache/alfworld
```

该环境没有 `bin/activate`，所以本机命令始终直接调用它的 Python 绝对路径。这里
不创建新的 ALFWorld 环境，也不从 v2 目录导入代码；editable install 只把 v3 当前
仓库安装到同一 Python 环境中。

如果只运行无 ALFWorld 的静态检查或 deterministic smoke，可使用：

```bash
python -m pip install -e '.[dev]'
```

## API 填写

三个配置文件只保存环境变量名：

```yaml
llm:
  provider: openai_compatible
  base_url: "https://openrouter.ai/api/v1"
  model: "REPLACE_WITH_MODEL_ID"
  api_key_env: MODEL_API_KEY
```

运行真实 API 前：

1. 在要使用的配置文件中填写实际的 OpenAI-compatible `base_url` 和 `model`；
   所选模型/端点必须原生支持 Chat Completions `tools`、单次 native ToolCall、
   `response_format.type=json_schema`、完整 `usage` 计量，以及配置中的
   `reasoning_effort`；静态 preflight 会检查本地 Provider 请求能力、模型配置和
   API 来源，远端端点的真实协议能力在首次请求时验证，任一不满足都会 fail closed；
2. 把密钥放入 `api_key_env` 指定的环境变量；默认变量名为
   `MODEL_API_KEY`；
3. train 和 frozen eval 必须使用相同的 provider、base URL、模型和 API key
   来源，保证实验公平性。

PowerShell 当前会话：

```powershell
$env:MODEL_API_KEY = "填写真实密钥"
$env:ALFWORLD_DATA = "填写 ALFWorld 数据目录"
```

Linux/WSL 当前会话：

```bash
export MODEL_API_KEY='填写真实密钥'
export ALFWORLD_DATA='/path/to/alfworld/data'
```

[`.env.example`](.env.example) 只列出变量名，便于人工配置；当前运行时不承诺
自动加载 `.env`。仅复制 `.env.example` 而不向进程导出变量不会生效。

不要在 YAML、命令行参数、README 或源码中增加 `api_key` 值。v3 Provider
只调用 `os.environ[api_key_env]` 对应的环境来源；缺失时 fail closed，且不会退回
配置文件明文或 v2/FlowEvo 的本地配置。

## 配置文件

| 文件 | 用途 |
|---|---|
| `configs/default.yaml` | 设计文档中的通用 v3 Agent、Planner、Runtime、生命周期和抽取预算 |
| `configs/alfworld_train_full_30.yaml` | 固定 full-method 在线训练：`train` split，六类各 5 题，共 30 题 |
| `configs/alfworld_frozen_eval.yaml` | 固定 held-out 冻结评测：`eval_out_of_distribution`（ALFWorld `valid_unseen`），六类各 10 题，共 60 题 |

训练和评测共同覆盖以下六类，顺序也是固定 manifest 的选择顺序：

1. `pick_and_place_simple`
2. `look_at_obj_in_light`
3. `pick_clean_then_place_in_recep`
4. `pick_heat_then_place_in_recep`
5. `pick_cool_then_place_in_recep`
6. `pick_two_obj_and_place`

`full` 表示完整 AtomicSkillGraph v3 方法，不是 ablation。训练配置要求空 v3 bank；
冻结评测配置要求读取训练完成后生成的 frozen snapshot，并拒绝与训练 manifest
重叠的 held-out 任务。

## 实验启动命令

设计冻结的实验模块名为：

- `experiments.run_v3_smoke`
- `experiments.run_v3_train`
- `experiments.run_v3_frozen_eval`

以下命令均为当前仓库已经实现并通过命令级检查的正式入口。

### 1. 静态 preflight

最小命令：

```bash
python -m experiments.run_v3_smoke --preflight
```

显式配置形式：

```bash
python -m experiments.run_v3_smoke --preflight --config configs/default.yaml
```

Preflight 必须检查 import、配置/API 来源、SQLite schema、空 bank、Harness、
Provider 的 native ToolCall 请求接口、真实物化的任务 manifest，以及 Artifact/Trace
输出可写性。静态 preflight 不向模型端点发付费探测请求，也不应打印密钥。

### 2. Deterministic no-API full-chain smoke

命令：

```bash
python -m experiments.run_v3_smoke --deterministic --config configs/default.yaml
```

该 smoke 使用 Fake Agent 和 Fake Harness，至少覆盖四个 episode：Full Dynamic
学习、下一题 autonomous Direct、错误参数 preflight 后 Fresh Seeded，以及图完成后
task rescue。验收包括 Trace 完整、Ledger 幂等、token 守恒、started 归因、
validated output DataFlow、Candidate 在线可用和 frozen digest 不变。

### 3. 真实 ALFWorld smoke

命令：

```bash
python -m experiments.run_v3_smoke --real-alfworld --config configs/default.yaml
```

固定开发门禁是 3 个 cold `pick_and_place_simple` train task、2 个未见过的 warm
同类 task，可选 1 个 heat-then-place 多节点 task。必须至少得到一条成功 Trace、
一个 Candidate、一次 Learned Invocation preflight，以及一次 started Direct 或明确的
`agent_completed_before_invocation`；仅“不崩溃”不算通过。

### 4. Full-30 在线训练

命令：

```bash
python -m experiments.run_v3_train --config configs/alfworld_train_full_30.yaml
```

在本机复用 v2 环境时，从 PowerShell 进入 WSL 后可完整执行：

```bash
cd /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3
/home/yangchengyu/asg_alfworld_venv/bin/python -m pip install -e '.[dev]'
export ALFWORLD_DATA=/home/yangchengyu/.cache/alfworld
read -rsp 'MODEL_API_KEY: ' MODEL_API_KEY && export MODEL_API_KEY && echo
# 先按“API 填写”一节修改 configs/alfworld_train_full_30.yaml 的 base_url/model
/home/yangchengyu/asg_alfworld_venv/bin/python -m experiments.run_v3_smoke --preflight --config configs/alfworld_train_full_30.yaml
/home/yangchengyu/asg_alfworld_venv/bin/python -m experiments.run_v3_train --config configs/alfworld_train_full_30.yaml
```

Runner 必须先固化精确任务 manifest，再从空 v3 bank 按该清单运行。配置中的
`tasks_per_type: 5`、`total_tasks: 30` 和六个 task type 是正式协议，不能被
“取前 30 题”替代。fresh bank 检查覆盖全部长期知识表以及 `artifacts/` 下的文件；
仅有 schema-v3 初始化行和空 artifact 目录才视为真正的空 bank。

### 5. 任务边界 resume

命令：

```bash
python -m experiments.run_v3_train --config configs/alfworld_train_full_30.yaml --resume
```

Resume 只跳过 manifest 中已经 `completed`，且 task signature、config hash、代码
commit 和 knowledge milestone 全部一致的题目。它不从任意 ALFWorld world revision
恢复；中途失败的 episode 从相同初始任务状态重跑，且基础设施/API 失败不产生长期
Skill/Tool 负面证据。`experiment.max_task_attempts` 是跨进程累计的严格正整数；正式
train 配置设为 `3`，因此每题最多执行 3 个 attempt，下一次 resume 会在启动第 4 个
attempt 前 fail closed。

30 题全部完成后，Runner 还会在冻结前执行一次配置批次的 final maintenance；只有
结构化结果满足 `pending_count == 0` 才允许 freeze。该边界同样受 knowledge checkpoint
保护；若 maintenance 改变知识摘要，Runner 会以 compare-and-update 事务把新摘要写入
最后一个 completed task 的 `knowledge_digest_after`，保持 source train digest chain
与 frozen provenance 一致。周期维护和 final maintenance 使用独立、先落盘的 immutable
maintenance Trace；不会覆盖已保存的 task Trace。

### 6. Frozen held-out 评测

命令：

```bash
python -m experiments.run_v3_frozen_eval --config configs/alfworld_frozen_eval.yaml
```

冻结评测固定使用 `eval_out_of_distribution/valid_unseen` 的六类各 10 题，共 60
题。开始前必须验证 train manifest、frozen snapshot 和 held-out manifest；结束后必须
满足：

```text
knowledge_digest_before == knowledge_digest_after
```

Frozen 模式只允许 Active/Preferred 资产。它可以写独立 eval Trace 和 metrics，但
禁止创建或修改 Skill/Implementation/Tool/Composite、更新长期状态/utility、写入训练
EvidenceLedger，或在题目间传播测试期新知识。

若评测中断，使用以下任务边界恢复命令；不要手工跳题或复用半完成 episode：

```bash
python -m experiments.run_v3_frozen_eval --config configs/alfworld_frozen_eval.yaml --resume
```

冻结评测同样固定 `experiment.max_task_attempts: 3`；计数保存在独立 eval run-state
数据库中，不写入或改变 frozen knowledge snapshot。

## 输出与审计要求

正式运行应至少保存：

- 精确 task manifest 及其 hash；
- config hash、代码 commit、运行 phase 和 knowledge milestone；
- 原子写入的逐题 Trace；
- NativeToolCall、EnvironmentAction、Implementation/Tool execution 和 Validation；
- `planner_p1`、`planner_p1_repair`、`planner_p2`、`planner_p2_repair`、
  `runtime_preparation`、`runtime_seeded`、`runtime_dynamic`、`extractor_e1`、
  `extractor_e2`、`evolution_repair` 的逐轮 usage；
- per-agent token/latency 报告和知识增长/生命周期报告。

每题的 artifact growth/lifecycle 不从文本日志反推：Runner 在任务边界读取
`artifact_index` 的版本/状态和 `lifecycle_projection` 的权威 checkpoint/完整投影，保存
带 digest 的 before/after snapshot 与 delta，并和 completed task 的 `result_json` 同一
事务提交。报告只对不可变 task Trace 做内存 overlay，不回写 Trace。收尾还会验证本次
进程 `UsageLedger` 的每个 event id 恰好存在于已落盘 task Trace 或本进程新建的
maintenance Trace，且每条 Trace 自身 token reconciliation 为零。

正式结果要求 `token_mismatch = 0` 且 `unattributed_total_tokens = 0`。Reasoning
token 只按 Provider metadata 计量，不读取 reasoning text，也不把 reasoning token
再次加到 Provider 已报告的 `total_tokens`。

## 常见失败

- `MODEL_API_KEY` 缺失：在启动 Runner 的同一个进程环境中导出变量；不要把 key
  写进 YAML。
- `model` 仍为 `REPLACE_WITH_MODEL_ID`：在三个正式使用的配置中填入同一个真实
  model id。
- ALFWorld 初始化失败：检查 `ALFWORLD_DATA` 下是否存在 `logic/` 和
  `json_2.1.1/`，并确认安装了 `.[alfworld]`。
- Resume 拒绝：不要覆盖旧 run directory；核对 manifest、配置、commit 和 frozen
  milestone 是否完全一致。
- `max_task_attempts` 已耗尽：该题不会再自动或通过 `--resume` 启动新 attempt；保留
  run directory 作为失败证据，定位基础设施问题后启动新的、独立命名的正式 run。
- Frozen digest 改变：该次评测无效，必须定位写路径并从原 frozen snapshot 重跑。
