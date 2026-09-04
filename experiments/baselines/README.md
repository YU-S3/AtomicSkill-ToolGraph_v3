# Baseline 对比实验基础设施（Baseline Comparison）

本目录实现 `AtomicSkillGraph_External_Baseline_Comparison_Implementation_Design.md`
的公共底座与 **B3 SkillOpt** 这一条 baseline。其余方法（B0/B1/B2/B4/B5/B6/Ours
scaled）在 `baseline_lock.yaml` 中标记为 `not_implemented`。

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
  .external/skillopt/                     上游 SkillOpt（key-file hash 校验，0 patch）
```

- 公共 Train/Validation/Test manifest：`data/baseline_manifests/*.json`（已提交）。
  `train_30` 与主实验 Full-30 是同一批 30 题（由 `test_manifest_builder.py`
  中从主实验不可变 run manifest 提取的 64 位签名冻结核对）。
- 模型统一：`openai_compatible + https://api.deepseek.com + deepseek-v4-flash`，
  密钥只经 `MODEL_API_KEY` 环境变量进入 worker，绝不落盘（有 fail-closed 扫描）。
- SkillOpt 学习算法（trainer/reflect/aggregate/optimizer/slow/meta/gate/prompts）
  全部原样复用 `.external/skillopt`；我方只新写 EnvAdapter 与调度。
- Hidden reference 关闭：`build_reference_text() == ""`（设计文档 §16.6）。
- Test 阶段只加载 `frozen/artifact/best_skill.md`，结束要求 frozen digest
  前后一致，否则 run 无效（§12/§16.8）。

## 命令（§36）

```bash
cd /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3_baseline
export ALFWORLD_DATA=/home/yangchengyu/.cache/alfworld
read -rsp 'MODEL_API_KEY: ' MODEL_API_KEY && export MODEL_API_KEY && echo

# 1) 首次环境准备（拷贝并校验上游源码 + 创建 worker venv）
/home/yangchengyu/asg_alfworld_venv/bin/python -m experiments.baselines.bootstrap_external \
    --lock experiments/baselines/baseline_lock.yaml \
    --local-source /mnt/d/T3S_exp/SkillOpt-main \
    --setup-worker-venv

# 2) 构建公共 manifest（已提交的 data/baseline_manifests 可直接复用，重复构建结果一致）
/home/yangchengyu/asg_alfworld_venv/bin/python -m experiments.baselines.common.manifest_builder \
    --alfworld-data "$ALFWORLD_DATA" --output data/baseline_manifests

# 3) B3 SkillOpt 训练（同一批 30 题 + 公共 Validation 30 题）
/home/yangchengyu/asg_alfworld_venv/bin/python -m experiments.baselines.run_method \
    --method b3_skillopt --phase train \
    --train-manifest data/baseline_manifests/train_30.json \
    --validation-manifest data/baseline_manifests/validation_30.json

# 4) 冻结评测（held-out 60 题，只读 best_skill.md）
/home/yangchengyu/asg_alfworld_venv/bin/python -m experiments.baselines.run_method \
    --method b3_skillopt --phase test \
    --train-manifest data/baseline_manifests/train_30.json \
    --validation-manifest data/baseline_manifests/validation_30.json \
    --test-manifest data/baseline_manifests/test_ood_60.json
```

产物在 `runs/baselines/pilot/b3_skillopt/42/`：
`run_manifest.json`、`config_resolved.json`、`source_lock.json`、
`task_manifest.json`、`train/`（含 `best_skill.md`、`summary.json`、
`usage.json`、逐题 common sidecar）、`frozen/{digest.json,artifact/best_skill.md}`、
`test/{task_rows.jsonl,summary.json}`。Test 的 official 成功来自 SkillOpt 环境的
`infos["won"]`；strict 成功由 controller 用 Ours Harness 边界后验重放计算
（`common/task_authority.py`），不回馈给 baseline Agent。

## 测试

```bash
# 全量（主实验 458 项 + baseline 27 项，均在 worker venv 中验证）
cd /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3_baseline
ALFWORLD_DATA=/home/yangchengyu/.cache/alfworld .venv_b3_skillopt/bin/python -m pytest -q
```

确定性覆盖：manifest 一致性（六类数量、train30⊂train120⊂train300、split 不重叠、
与主实验同批 30 题签名一致）、freeze digest 篡改检测、密钥落盘扫描、
usage 缺失 fail-closed、以及带脚本化 LLM 的完整上游 ReflACT 链路
（seed→rollout→reflect→patch→gate→best_skill）。
