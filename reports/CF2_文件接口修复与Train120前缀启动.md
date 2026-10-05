# 文件接口修复与 seed42 Train120 前缀运行

受测源码固定为 `5462c70f8b3bb9424dec8b23f0d848acb3343559`，无 LLM 回归 **126/126 通过**。日志在 `/home/yangchengyu/alfworld_seed42_train120_cf2_20261005/engineering/verified_pytest.log`。

- 新 Skill 在 realization_request 和 Workflow 中只能用 `$new`；已有 Skill 只能引用 Bank 中真实存在的 Skill ID。规则进入实际 HTTP schema 和提交前校验，未知名称不自动映射，原一次结构修复上限不变。
- Builder/Learner 及文件工具明确区分程序内 `/workspace/...` 路径与 `files/deleted_files` 的相对发布名。保留绝对发布路径、`..`、inputs 越界及 symlink 拒绝。未改写旧生成源码或晋升记录。
- 用本轮 Office 原提议和 Spreadsheet 原生成源码做生产路径回归；普通模型失败仍如实拒绝/记录。无额外文件类 LLM 请求，不降低两个独立真实 Train 正向的 usable 标准。

这次不是独立 Train12 pilot。完整运行身份仍是唯一公共清单的 Train120，按原 `ordered_train(..., 42)` 顺序运行；后续计划为 Train120 完成后 Frozen，再执行原 Val24。新 Bank 的 assets/attempts/jobs/train_cases 全部从零开始，没有导入旧 Frozen 或准备能力诊断库。

本次只授权该运行前 **12 个 Train**。生产 runner 的 `stop_after_tasks=12` 在提交第 12 题后停止，写入 `STOP_AFTER_TASK` 和 `complete=false` 的部分 summary；不会冻结这 12 题，不启动 Val、Test、其他 seed 或模型矩阵。停止控制不改变任务成员、单题预算或运行身份。无 LLM 回归已验证继续同身份时不会重跑已完成题。

输出根：`/home/yangchengyu/alfworld_seed42_train120_cf2_20261005`。固定 WSL checkout：`/home/yangchengyu/asg_alfworld_seed42_train120_20261005`。保留实际 `config.json`、execution manifest、selection、initial_bank、旁路正式日志、traces/requests 与本次单组合 `matrix.json`。旧已停止矩阵不变。

实际启动命令由 `process.json` 记录：在固定 checkout 下以 `PYTHONPATH=src` 调用该输出目录的 `run_seed42_prefix.py`。脚本 SHA 进入运行身份；脚本只调用已有 canonical/config/runner/FormalLog 模块，并在 12 题边界结束。源码、配置、数据或运行目录变化都不能冒充同身份 resume。

启动状态以本机 `matrix.json`、`run.log` 和 `train/traces` 为准；本记录不报告性能判断。等待用户后续指示再分析结果或继续剩余 Train108/Val24。
