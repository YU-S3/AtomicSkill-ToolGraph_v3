# Baseline 对比实验基础设施（Baseline Comparison）

本目录实现冻结版 external-baseline 协议的公共底座，以及 **B3 SkillOpt**、
**B4 EmbodiSkill** 和 **B5 GEPA**。
正式数据身份为 Train-120 / Validation-24 / Frozen Test-134，运行 seed 为
42、43、44。

## 架构

```text
Controller（主实验环境: asg_alfworld_venv）
  experiments/baselines/run_method.py
  experiments/baselines/common/           公共协议（manifest/usage/trace/freeze/authority）
  experiments/baselines/b3_skillopt/      B3 driver / freeze
  experiments/baselines/b4_embodiskill/    B4 campaign / controller / driver
  experiments/baselines/b5_gepa/          B5 campaign / controller / driver
        │  subprocess（worker wire JSON；密钥只经环境变量传递）
        ├──────────────────────────┬──────────────────────────┐
        ▼                          ▼                          ▼
.venv_b3_skillopt            .venv_b4_embodiskill          .venv_b5_gepa
SkillOpt worker              EmbodiSkill worker/Python3.12 GEPA worker/Python 3.12
        │                          │                          │
.external/skillopt           .external/embodiskill          .external/gepa
完整源码树 SHA-256 校验       commit 7601260...           tag v0.1.4 / 8b0ce6c...
                                                           + .external/skillopt
```

- 公共 manifest 使用 selection seed 42：各 family 先按 canonical gamefile
  path 排序，再用同一固定 RNG 依次 shuffle；Train-120 每类 20 题，Val-24
  每类 4 题。Test-134 是 `valid_unseen` 的全部 134 个 game，按 canonical
  path 存储，不做平衡抽样。
- 模型统一：`openai_compatible + https://api.deepseek.com + deepseek-v4-flash`，
  所有生成角色显式使用 `reasoning_effort=high`。密钥只经
  `MODEL_API_KEY` 环境变量进入 worker，绝不落盘（有 fail-closed 扫描）。
- SkillOpt 学习算法（trainer/reflect/aggregate/optimizer/slow/meta/gate/prompts）
  全部原样复用 `.external/skillopt`；我方只新写 EnvAdapter 与调度。
- Hidden reference 关闭：`build_reference_text() == ""`（设计文档 §16.6）。
- 正式训练阶段不读取 Test，也不做 Train resubstitution；训练完成后冻结
  `best_skill.md`，再由独立 test worker 只读执行 Test-134。冻结 digest 前后
  必须一致。
- ALFWorld 独立 episode 与 analyst 最多 16 路并行；每个环境、会话及
  provider 证据仍逐
  独立，结果在交给 SkillOpt 前恢复为 manifest 顺序，不改变训练批次或更新逻辑。
  当前适配器使用 thread pool，并在 `run_manifest.json` 中如实记录；没有伪装成
  冻结设计建议的 spawn process pool。
- Provider 传输禁用 SDK 内部隐藏重试，由适配层按 2/5/10/20 秒退避执行最多
  5 次总尝试；仅
  连接、超时、429/5xx、空响应和无效 usage 等瞬态失败重试，永久性 4xx
  立即失败。每次逻辑调用记录实际 attempt 数、恢复状态和非敏感失败码，
  最终失败仍 fail-closed，绝不作为普通 hard=0 样本进入 SkillOpt。
- B4 保留官方 TeamSolver、手册反思修订和 trajectory retrieval；Train120 恰好一次，epoch 后只读 Val24 选择完整 best snapshot。
- B5 从与 B1/B3 相同的 SkillOpt `initial.md` 出发，只优化
  `{"skill_text": ...}`。GEPA proposal/iteration/candidate state transition 串行，
  只并行同一 evaluation batch 内彼此独立的 ALFWorld episode；Val-24 只用于
  candidate scoring，不进入 reflection dataset，冻结后 Test-134 不构造 optimizer。
- B4/B5 正式 campaign 内 seeds42/43/44 并行，seed 内知识更新串行；每 seed 独立 evaluation 最多16路，共用global48 gate。不同方法不能重叠。B3 已完成的正式结果冻结，不重跑。

## B3 命令（§36）

```bash
cd /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3_baseline
set -a
. /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env
set +a
export PYTHONPATH="$PWD/src:$PWD"

# 1) 首次环境准备（拷贝并校验上游源码 + 创建 worker venv）
/home/yangchengyu/asg_alfworld_venv/bin/python -m experiments.baselines.bootstrap_external \
    --lock experiments/baselines/baseline_lock.yaml \
    --local-source '/mnt/d/T3S_exp/SkillOpt-main (2)/SkillOpt-main' \
    --setup-worker-venv

# 2) 构建公共 manifest（已提交的 data/baseline_manifests 可直接复用，重复构建结果一致）
/home/yangchengyu/asg_alfworld_venv/bin/python -m experiments.baselines.build_manifests \
    --alfworld-data "$ALFWORLD_DATA" --selection-seed 42 \
    --out data/baseline_manifests

# 3) 两题并发真实 API + ALFWorld smoke（每题最多 2 个环境动作，不要求解题成功）
/home/yangchengyu/asg_alfworld_venv/bin/python -m experiments.baselines.run_method \
    --method b3_skillopt --phase smoke \
    --seed 42 \
    --train-manifest data/baseline_manifests/train_120.json \
    --validation-manifest data/baseline_manifests/validation_24.json \
    --config configs/baselines/b3_skillopt.yaml

# 4) B3 SkillOpt 正式训练：Train120 -> Val24 gate -> freeze
/home/yangchengyu/asg_alfworld_venv/bin/python -m experiments.baselines.run_method \
    --method b3_skillopt --phase train \
    --seed 42 \
    --train-manifest data/baseline_manifests/train_120.json \
    --validation-manifest data/baseline_manifests/validation_24.json \
    --test-manifest data/baseline_manifests/test_ood_full_134.json \
    --config configs/baselines/b3_skillopt.yaml

# 5) 对相同 seed 的冻结最佳 Skill 做只读 held-out Test134
/home/yangchengyu/asg_alfworld_venv/bin/python -m experiments.baselines.run_method \
    --method b3_skillopt --phase test \
    --seed 42 \
    --source-run runs/baselines/protocol_faithful_matched_train_v2/b3_skillopt/<train目录> \
    --train-manifest data/baseline_manifests/train_120.json \
    --validation-manifest data/baseline_manifests/validation_24.json \
    --test-manifest data/baseline_manifests/test_ood_full_134.json \
    --config configs/baselines/b3_skillopt.yaml
```

### B3 legacy controller-code 单项豁免恢复

仅当既有三 seed Test 在创建输出目录之前全部只因
`controller_code` 失败时，才允许使用该恢复入口。它会暂时切换到 campaign
锁定的 controller commit，通过仓库 `runs/` 下的审计 bootstrap 只替换旧
`run_method.hash_code(REPO_ROOT)` 的两次返回值；commit、clean tree、Python、
SkillOpt runtime、模型、manifest、ALFWorld gamefile 和 Frozen digest 仍由旧正式
runner 原样校验。恢复结束或被中断后，控制器会切回启动时的 branch/commit。

先执行不调用 API 的 bootstrap-authority preflight。它只证明受控豁免、旧
checkout、Frozen 和 exactly-once 边界可用；模型、运行时、ALFWorld 与完整
manifest preflight 仍由随后启动的旧正式 runner 执行。只有该检查返回
`passed: true` 才启动三个并行 Test134：

```bash
set -euo pipefail
REPO=/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3_baseline
B3_PY="$REPO/.venv_b3_skillopt/bin/python"
CAMPAIGN="$REPO/runs/baselines/protocol_faithful_matched_train_v2/b3_skillopt/formal_3seed_authorityfix_20260912T052314Z"
FAILED="$CAMPAIGN/recovered_test_20260912T212811Z"

cd "$REPO"
set -a
. /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env
set +a
export ALFWORLD_DATA=/home/yangchengyu/.cache/alfworld
export PYTHONPATH="$REPO/src:$REPO"
export PYTHONUNBUFFERED=1

"$B3_PY" -m experiments.baselines.recover_b3_test \
    --campaign-root "$CAMPAIGN" \
    --failed-attempt-root "$FAILED" \
    --preflight-only \
    --acknowledge-controller-code-waiver

"$B3_PY" -m experiments.baselines.recover_b3_test \
    --campaign-root "$CAMPAIGN" \
    --failed-attempt-root "$FAILED" \
    --acknowledge-controller-code-waiver
```

该入口不执行 Train，也不自动重试任何已经生成 Test `run_manifest.json` 的
seed。结果目录会包含中央 waiver receipt、每 seed application receipt、
`campaign_report.json` 和 `recovered_test_report.json`；报告身份明确标记为
`accepted_with_controller_code_waiver`。

## B4 EmbodiSkill (v2.1)

当前真实 API smoke 尚未放行：high reasoning 在 512-token 上限内未输出动作。
证据、已通过检查及待确认预算调整见 [验证记录](B4_EMBODISKILL_VALIDATION.md)。

直接调用 air-embodied-brain/EmbodiSkill commit
`760126030eab1d33ec6a6f30988f0f1fb58df3a7`。不再需要 subgoal JSONL。
保留 TeamSolver、stuck recovery、ALFWorld 1-shot、all-MiniLM-L6-v2 检索、
skill-aware reflection 和版本化手册。每 seed 从空手册与空轨迹库开始，
Train120 按该 seed 的单个 permutation 分成 4×30，每题只训练一次；
每 epoch 后官方 revise_manual，再只读 Val24，严格提高才替换 best。
最后冻结完整 best persistent state，在 task-local 副本上只读 Test134。

初始化隔离 Python3.12 环境（只需一次）：

```bash
cd /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3_baseline
/home/yangchengyu/asg_alfworld_venv/bin/python -m experiments.baselines.bootstrap_external \
  --method embodiskill --setup-worker-venv
```

真实 smoke → 三 seed 正式实验（smoke 非零退出就不会启动 formal）：

```bash
cd /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3_baseline
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
bash experiments/baselines/launch_embodiskill.sh smoke "runs/baselines/b4_smoke_$STAMP" &&
bash experiments/baselines/launch_embodiskill.sh formal "runs/baselines/b4_formal_$STAMP"
```

入口自动从 v3/.env 读取 MODEL_API_KEY，API key 不进入 job JSON。
smoke 是 2 Train + 2 Val + 2 Test（固定 smoke manifests 的前两题），保留正式
max_trials 和全部方法参数，检查真实 reflection/revision/version advance，
不把 smoke 正确率作为正式指标。formal 固定 seeds42/43/44 并行；
Train 和手册更新在每 seed 内严格串行。正式创建前实际加载
ALFWorld+Chroma+embedding 并调用模型做并发/内存验证，
只能在建锁前从48降至36或24；不能安全容纳24则拒绝启动。

恢复已有 formal（失败 attempt 保留计费证据，仅重做尚未提交的状态边界）：

```bash
bash experiments/baselines/launch_embodiskill.sh resume runs/baselines/原来的目录
```

结果在 campaign 根目录的 `campaign_summary.json`、`paper_report.json`、
`REPORT.md`。逐 seed 的 `train/validation/test` 保存 `task_rows.jsonl`
和 `evaluated_common_episodes.jsonl`；`attempts/操作/attempt-id/`
保存 `provider_calls.jsonl`（每个物理调用）、`model_responses.jsonl`、
`environment_actions.jsonl`、`method_events.jsonl` 和 `result.json`。
失败时有 `rollout_failure.json`。API价格未冻结，金额明确为 null；
未知 token 保留 null 和已知小计。Frozen 位于 `seed_*/frozen/`，
包含整个 Chroma/graph/manual/versions/reflections 以及 source/config/manifests。

## B5 GEPA

### 隔离环境 bootstrap 与 verify

B5 锁定 GEPA tag `v0.1.4`、commit
`8b0ce6cd99a234f6b74daf37558a2ac0ce18f975`，并只读复用 pinned SkillOpt
源码中的 `initial.md` 和 common text executor 语义。worker 使用独立 Python
3.12 venv；不得在其中安装 `atomic-skillgraph` 或 `skillopt` distribution。

```bash
set -euo pipefail
REPO=/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3_baseline
ASG_PY=/home/yangchengyu/asg_alfworld_venv/bin/python
cd "$REPO"
export PATH="/home/yangchengyu/.local/bin:$PATH"
export PYTHONPATH="$REPO/src:$REPO"

"$ASG_PY" -m experiments.baselines.bootstrap_external \
    --lock experiments/baselines/baseline_lock.yaml \
    --method gepa \
    --skillopt-local-source '/mnt/d/T3S_exp/SkillOpt-main (2)/SkillOpt-main' \
    --setup-worker-venv

# 只重验 pinned GEPA source；不调用模型 API。
"$ASG_PY" -m experiments.baselines.bootstrap_external \
    --lock experiments/baselines/baseline_lock.yaml \
    --method gepa
```

```bash
"$ASG_PY" - <<'PY'
from pathlib import Path
from experiments.baselines.bootstrap_external import (
    verify_worker_environment,
    verify_worker_python,
    worker_expected_distributions,
)

python = Path.cwd() / ".venv_b5_gepa/bin/python"
print(verify_worker_python(python, expected_version="3.12"))
print(verify_worker_environment(
    python,
    expected_versions=worker_expected_distributions("b5_gepa"),
    forbidden_distributions=("atomic-skillgraph", "skillopt"),
    forbidden_modules=("atomic_skillgraph", "skillopt"),
))
PY
```

### 真实 smoke 与三 seed formal campaign

先确认 B3/B4 及其它真实 API 实验已经结束。B5 smoke 与 formal campaign 都从
固定 `.venv_b5_gepa/bin/python` 启动；`&&` 保证 smoke 失败时不会启动正式
实验。campaign 入口强制 seeds 必须恰为 `42 43 44` 且并行，seed 内只允许独立 task evaluation 并发，优化器迭代仍串行。

```bash
set -euo pipefail
REPO=/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3_baseline
B5_PY="$REPO/.venv_b5_gepa/bin/python"
cd "$REPO"
set -a
. /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env
set +a
export PYTHONPATH="$REPO/src:$REPO"
: "${MODEL_API_KEY:?MODEL_API_KEY is missing}"
: "${ALFWORLD_DATA:?ALFWORLD_DATA is missing}"

"$B5_PY" -m experiments.baselines.b5_gepa.controller \
    --phase smoke \
    --seed 42 \
    --train-manifest data/baseline_manifests/train_6_smoke.json \
    --validation-manifest data/baseline_manifests/validation_6_smoke.json \
    --test-manifest data/baseline_manifests/test_6_smoke.json \
    --config configs/baselines/b5_gepa_smoke.yaml &&
"$B5_PY" -m experiments.baselines.b5_gepa.run_seed_campaign \
    --seeds 42 43 44 \
    --train-manifest data/baseline_manifests/train_120.json \
    --validation-manifest data/baseline_manifests/validation_24.json \
    --test-manifest data/baseline_manifests/test_ood_full_134.json \
    --config configs/baselines/b5_gepa.yaml
```

正式 B4/B5 controller 会拒绝 dirty `experiments/`/`configs/` source tree。正式
运行前只需形成一个本地 clean commit 作为 provenance authority；本轮修改不推送
GitHub，也不需要 PR。一个 method 的 campaign 完成并释放
`runs/baselines/.formal_method_campaign.lock` 后，才能启动另一个 method。

每个 phase 使用不可复用的唯一目录。B3 主要产物包括：
`run_manifest.json`、`config_resolved.json`、`source_lock.json`、
`task_manifest.json`、`train/`（含 `best_skill.md`、`summary.json`、
`usage.json`、逐题 common sidecar）、`frozen/{digest.json,artifact/best_skill.md}`、
训练目录根部的 `report.json`，以及 test 目录的
`test/{task_rows.jsonl,summary.json}` 和 `test_report.json`。Test 的
official 成功来自 SkillOpt 环境的
`infos["won"]`；strict 成功由 controller 用 Ours Harness 边界后验重放计算
（`common/task_authority.py`），不回馈给 baseline Agent。

B4 campaign 根目录保存 `campaign_lock.json`、`campaign_summary.json`、
`REPORT.md`，完成后还有 `paper_report.json`；各 seed 的 `summary.json`
含全量尝试成本、Train/Val/修订成本和方法资产计数，失败则保存 `campaign_failure.json`。
B5 根目录继续保存 `campaign_report.json` 和 `completion.json`。
各 seed lane 保存自己的 common episode sidecar、usage、freeze digest 和
Train/Test 报告。B4 冻结完整 EmbodiSkill persistent state；B5 冻结 `best_skill.md`、
`gepa_result.json` 与 `candidate_lineage.json`。所有最终 task row 均按
`manifest_index` 恢复顺序。

## 测试

```bash
# controller/common/B3/B4/B5 确定性测试；真实 API smoke 需按上文单独执行。
cd /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3_baseline
export PYTHONPATH="$PWD/src:$PWD"
ALFWORLD_DATA=/home/yangchengyu/.cache/alfworld \
    /home/yangchengyu/asg_alfworld_venv/bin/python -m pytest -q \
    experiments/baselines/tests
```

确定性覆盖：seed-42 canonical/shuffle 选择、Train/Val 嵌套、完整 Test134、
split 不重叠、freeze digest 篡改检测、密钥落盘扫描、
usage 缺失 fail-closed、以及带脚本化 LLM 的完整上游 ReflACT 链路
（seed→rollout→reflect→patch→gate→best_skill）；另覆盖 B4 task/chunk/checkpoint/readonly/freeze 边界和 B5 GEPA candidate/Validation/reflection/freeze
边界。
