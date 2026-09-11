# Baseline 对比实验基础设施（Baseline Comparison）

本目录实现冻结版 external-baseline 协议的公共底座与 **B3 SkillOpt**。
正式数据身份为 Train-120 / Validation-24 / Frozen Test-134，运行 seed 为
42、43、44。

## 架构

```text
Controller（主实验环境: asg_alfworld_venv）
  experiments/baselines/run_method.py
  experiments/baselines/common/           公共协议（manifest/usage/trace/freeze/authority）
  experiments/baselines/b3_skillopt/      B3 driver / freeze
        │  subprocess（worker wire JSON，不含任何密钥）
        ▼
Worker（独立 venv: .venv_b3_skillopt，gitignored）
  experiments/baselines/b3_skillopt/worker.py
  experiments/baselines/b3_skillopt/common_alfworld_adapter.py
  .external/skillopt/                     上游 SkillOpt（完整源码树 SHA-256 校验，0 patch）
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

## 命令（§36）

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

每个 phase 使用不可复用的唯一目录。主要产物包括：
`run_manifest.json`、`config_resolved.json`、`source_lock.json`、
`task_manifest.json`、`train/`（含 `best_skill.md`、`summary.json`、
`usage.json`、逐题 common sidecar）、`frozen/{digest.json,artifact/best_skill.md}`、
训练目录根部的 `report.json`，以及 test 目录的
`test/{task_rows.jsonl,summary.json}` 和 `test_report.json`。Test 的
official 成功来自 SkillOpt 环境的
`infos["won"]`；strict 成功由 controller 用 Ours Harness 边界后验重放计算
（`common/task_authority.py`），不回馈给 baseline Agent。

## 测试

```bash
# 全量（均在 worker venv 中验证；真实 API smoke 需另行执行）
cd /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3_baseline
ALFWORLD_DATA=/home/yangchengyu/.cache/alfworld .venv_b3_skillopt/bin/python -m pytest -q
```

确定性覆盖：seed-42 canonical/shuffle 选择、Train/Val 嵌套、完整 Test134、
split 不重叠、freeze digest 篡改检测、密钥落盘扫描、
usage 缺失 fail-closed、以及带脚本化 LLM 的完整上游 ReflACT 链路
（seed→rollout→reflect→patch→gate→best_skill）。
