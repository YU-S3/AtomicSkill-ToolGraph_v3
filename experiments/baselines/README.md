# Baseline 对比实验基础设施（Baseline Comparison）

本目录实现冻结版 external-baseline 协议的公共底座，以及 **B3 SkillOpt**、
**B4 SkillGen-S** 和 **B5 GEPA**。
正式数据身份为 Train-120 / Validation-24 / Frozen Test-134，运行 seed 为
42、43、44。

## 架构

```text
Controller（主实验环境: asg_alfworld_venv）
  experiments/baselines/run_method.py
  experiments/baselines/common/           公共协议（manifest/usage/trace/freeze/authority）
  experiments/baselines/b3_skillopt/      B3 driver / freeze
  experiments/baselines/b4_skillgen_s/    B4 campaign / controller / driver
  experiments/baselines/b5_gepa/          B5 campaign / controller / driver
        │  subprocess（worker wire JSON；密钥只经环境变量传递）
        ├──────────────────────────┬──────────────────────────┐
        ▼                          ▼                          ▼
.venv_b3_skillopt            .venv_b4_skillgen          .venv_b5_gepa
SkillOpt worker              SkillGen worker/Python 3.9 GEPA worker/Python 3.12
        │                          │                          │
.external/skillopt           .external/skillgen          .external/gepa
完整源码树 SHA-256 校验       commit 816c91f...           tag v0.1.4 / 8b0ce6c...
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
- B4 保留 SkillGen 的 trajectory sampling、subgoal progress、TaskGraph、
  TD(lambda) credit、step-wise skill/golden segment extraction 和冻结后检索推理；
  不接收 Validation。每个 seed 对 Train-120 产生 `120 x 6 = 720` 个独立
  sampling episode，完成 barrier 后才允许 extraction，最后只读 Test-134。
- B5 从与 B1/B3 相同的 SkillOpt `initial.md` 出发，只优化
  `{"skill_text": ...}`。GEPA proposal/iteration/candidate state transition 串行，
  只并行同一 evaluation batch 内彼此独立的 ALFWorld episode；Val-24 只用于
  candidate scoring，不进入 reflection dataset，冻结后 Test-134 不构造 optimizer。
- B4/B5 的正式 campaign 都严格串行执行 seed `42 -> 43 -> 44`，每个 seed
  内最多 16 个独立 episode 并发，并共享 campaign 级 provider `global16`
  gate。禁止同时运行多个 method campaign；只要 B3 仍在运行，就不得启动
  B4/B5 real-API smoke 或正式 campaign。

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

## B4 SkillGen-S

### Supervision 前置条件

B4 的 `subgoal_progress` 必须来自 SkillGen 自己的、与 Train gamefile 对应的
upstream-format label 文件。用户必须将该文件的绝对路径写入
`SKILLGEN_TRAIN_LABELS`。不得从 Ours TaskContract、hidden PDDL、expert plan、
Test label 或其它未来信息生成或补齐监督。

当前 pinned `.external/skillgen/data/alfworld/all.jsonl` 只覆盖
`valid_unseen`/Test-134，对 `train_6_smoke.json` 和 `train_120.json` 的覆盖均为
0。因此它只能作为诊断文件，不能填入 `SKILLGEN_TRAIN_LABELS`。在 authentic
Train label 到位前，B4 必须 fail-closed，**不要运行真实 B4 smoke 或 formal
campaign**。现有 Train-6 诊断证据在：

```text
runs/baselines/preflight/b4_skillgen_shipped_label_check/preflight_label_coverage.json
```

### 隔离环境 bootstrap 与 verify

bootstrap 会锁定 SkillGen commit
`816c91f458ddb87be32657c882cb0431b9cfea01`，创建 Python 3.9 worker venv，
并校验源码树、关键文件、精确依赖和禁止分发包。若 `.external/skillgen` 尚不
存在，命令会从 lock 中的远程仓库 clone；已有目录则只接受完全匹配的干净
pinned checkout。

```bash
set -euo pipefail
REPO=/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3_baseline
ASG_PY=/home/yangchengyu/asg_alfworld_venv/bin/python
cd "$REPO"
export PATH="/home/yangchengyu/.local/bin:$PATH"
export PYTHONPATH="$REPO/src:$REPO"

"$ASG_PY" -m experiments.baselines.bootstrap_external \
    --lock experiments/baselines/baseline_lock.yaml \
    --method skillgen \
    --setup-worker-venv

# 只重验 pinned source；不调用模型 API。
"$ASG_PY" -m experiments.baselines.bootstrap_external \
    --lock experiments/baselines/baseline_lock.yaml \
    --method skillgen
```

环境级 verify 会在剥离 `PYTHONPATH` 和 user-site 后检查 Python 3.9、全部固定
distribution 版本，以及 `atomic-skillgraph`/`skillopt` 分发包与模块均不存在：

```bash
"$ASG_PY" - <<'PY'
from pathlib import Path
from experiments.baselines.bootstrap_external import (
    verify_worker_environment,
    verify_worker_python,
    worker_expected_distributions,
)

python = Path.cwd() / ".venv_b4_skillgen/bin/python"
print(verify_worker_python(python, expected_version="3.9"))
print(verify_worker_environment(
    python,
    expected_versions=worker_expected_distributions("b4_skillgen_s"),
    forbidden_distributions=("atomic-skillgraph", "skillopt"),
    forbidden_modules=("atomic_skillgraph", "skillopt"),
))
PY
```

### 真实 smoke 与三 seed formal campaign

以下命令只在 authentic Train label 已提供后执行。B4 controller/campaign 必须
由 ASG controller Python 启动；实际 sampling、extraction 和 inference worker
由 config 中的 `.venv_b4_skillgen/bin/python`（Python 3.9）启动。smoke 通过后
才允许 formal campaign；formal campaign 自身固定串行运行 42、43、44。

```bash
set -euo pipefail
REPO=/mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3_baseline
ASG_PY=/home/yangchengyu/asg_alfworld_venv/bin/python
cd "$REPO"
set -a
. /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env
set +a
export PYTHONPATH="$REPO/src:$REPO"
: "${MODEL_API_KEY:?MODEL_API_KEY is missing}"
: "${ALFWORLD_DATA:?ALFWORLD_DATA is missing}"
: "${SKILLGEN_TRAIN_LABELS:?set this to authentic upstream-format Train labels}"

"$ASG_PY" -m experiments.baselines.b4_skillgen_s.controller \
    --phase smoke \
    --seed 42 \
    --train-manifest data/baseline_manifests/train_6_smoke.json \
    --test-manifest data/baseline_manifests/test_6_smoke.json \
    --supervision "$SKILLGEN_TRAIN_LABELS" \
    --config configs/baselines/b4_skillgen_s_smoke.yaml &&
"$ASG_PY" -m experiments.baselines.b4_skillgen_s.campaign \
    --train-manifest data/baseline_manifests/train_120.json \
    --test-manifest data/baseline_manifests/test_ood_full_134.json \
    --supervision "$SKILLGEN_TRAIN_LABELS" \
    --config configs/baselines/b4_skillgen_s.yaml
```

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
实验。campaign 入口强制 seeds 必须恰为 `42 43 44` 且依次串行，只允许每个
seed 内的独立 task evaluation 并发。

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

B4/B5 的 campaign 根目录额外保存不可变 `campaign_lock.json`、
`campaign_report.json` 和 `completion.json`（失败则为 `campaign_failure.json`）；
各 seed lane 保存自己的 common episode sidecar、usage、freeze digest 和
Train/Test 报告。B4 冻结完整 retrieval library；B5 冻结 `best_skill.md`、
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
（seed→rollout→reflect→patch→gate→best_skill）；另覆盖 B4 label authority、
sampling/extraction/freeze 边界和 B5 GEPA candidate/Validation/reflection/freeze
边界。
