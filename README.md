# SkillCompiler empirical v3.1

唯一生产入口为 `skillcompiler.empirical.v1`。流程是普通 Skill 接口与指导 → Python Program → 固定真实 Train 试用 → usable 版本 → Workflow/动态执行 → 冻结库与独立评分。

本轮范围是 Ours。其他方法复现和更多模型 API 接入不在本轮范围；现有 DeepSeek 配置维持 high reasoning。正式矩阵尚未启动，小样本结果不能作为正式性能结论。

规范：[核心机制](docs/specs/01_核心机制精简重构实施文档_v3.1.md)、[六 Benchmark](docs/specs/02_六Benchmark适配实施文档_v3.1.md)、[删除与迁移](docs/specs/04_旧代码删除与迁移清单_v3.1.md)。发生范围冲突时，以用户明确要求为准。

## 本地 WSL

使用 Ubuntu WSL、Python 3.12 和已启用 WSL 集成的 Docker Desktop。模型密钥只通过环境变量或明确指定的 `--env-file` 加载。

```bash
python -m pip install '.[alfworld,benchmarks,dev]'
docker build -t skillcompiler-program:v3.1 containers/program
docker image inspect --format='{{.Id}}' skillcompiler-program:v3.1
```

镜像必须与 `configs/default.yaml`、`containers/program/image.lock.json` 中的 digest 一致。重新构建得到不同 digest 时，先核对环境和依赖，不能静默沿用旧身份。Program 与动态 Python 使用同一个非 root、禁网络、只读根文件系统的容器；只挂载本题公开输入和工作区。容器内不联网安装依赖。

```bash
PROGRAM_IMAGE_DIGEST=$(docker image inspect --format='{{.Id}}' skillcompiler-program:v3.1) python -m pytest -q
skillcompiler-prepare --resources /path/to/SkillCompiler_resources_20261003 --output /path/to/prepared --case-map data/resource_mappings/spreadsheet_cases.json
skillcompiler-acceptance --config configs/alfworld_empirical_seed42.yaml --materials /path/to/original/pilot --output /path/to/new/acceptance --env-file .env
skillcompiler-pilot --config configs/alfworld_empirical_seed42.yaml --materials /path/to/original/pilot --acceptance /path/to/new/acceptance/acceptance.json --output /path/to/new/pilot --env-file .env
skillcompiler-multibench --config configs/default.yaml --datasets /path/to/prepared --corpus-root /path/to/treasury_bulletins_parsed/transformed --output /path/to/new/smoke --env-file .env
```

单个 ALFWorld manifest 使用 `skillcompiler --config ... --manifest ... --output ...`；评估追加 `--frozen-bank ...`。新 Benchmark 使用 `skillcompiler-multibench` 的固定 2 Train＋1 Val smoke。DocVQA 需要锁定模型的真实图像能力，当前文本 DeepSeek 组合记录 unsupported，分数留空。

`--resume` 要求源码、配置和任务身份一致。已完成执行只恢复未完成学习；缓存响应与版本注册幂等。未知环境副作用停止尝试，不自动重放。`STOP_AFTER_TASK` 文件在下一题前停止。费用保留所有尝试，未知计费不会写成零。

ALFWorld 清单按数据根目录下的物理文件和 SHA256 解析，扫描到全部目标后停止；旧 `env_index` 不作为扫描上限。`task_identity_resolution.json` 保存原清单与当前环境 ID 的对应关系，reset 仍校验当前文件、序号、目标和签名。重复 discovery 从起点开始，缺题或身份变化时停止，不替换题目。

已完成 Train 后仅修复上述加载入口，可显式使用 `skillcompiler-pilot ... --continue-val` 继续原 6 个 Val。此入口核对原 Train 配置、物理题目集合、源码差异及 Train／冻结 Bank digest，保存 `continuation_manifest.json` 和分阶段源码版本；不重新执行 Train，也不放宽普通 `--resume` 的身份要求。

文件 Adapter 的 `tool_definitions()` 返回实际公开工具列表，供 Learner／Builder 使用；Program 仍禁止递归调用 `execute_python`。修复后已从空 Bank 完成 OfficeQA、Spreadsheet 各 2 Train＋1 Val，实际成绩和费用见 [文件类冒烟报告](reports/file_adapter_smoke_result.json)。43 项回归（含真实 Docker）与安装包验证见 [工程验证](reports/file_adapter_fix_verification.json)。

主实验共享划分、run seed 与统一日志尚未完全对齐，具体差异及待确认事项见 [主实验核对报告](reports/main_experiment_alignment_v11.json)。现有 pilot 和 smoke 保留为诊断结果，不作为正式主实验数据。

## 项目结构与边界

- `src/atomic_skillgraph/empirical/`：通用 Bank、普通参数引用、Planner/Learner、执行器、Docker worker、恢复与预算。
- `src/atomic_skillgraph/harness/`：ALFWorld 和五个新增 Benchmark 的公开输入、工具和独立评分器。数据集特定规则只在此层与数据准备层。
- `src/atomic_skillgraph/experiments/`：可安装的固定清单、运行、验收和 smoke 入口；根 `experiments/` 是兼容入口。
- `benchmark_profiles.json`、`models.lock.json`、`splits/`：资源条件、模型能力与固定 ID 清单；完整公开输入和私有 evaluator 记录留在本地资源目录。
- `cleanup_manifest.json`：逐文件删除与迁移分类；[历史版本](docs/history/README.md)保存旧实现与结果来源。

每个 seed 独立空 Bank；两个不同物理 Train 任务正向实测才成为 usable，repair 不继承旧版本成功。冻结仅包含 usable Program 和普通指导/策略；Val/Test 不更新 Bank。模型自报成功、文件存在与正式评分分别记录。

用户在本轮明确补充了整题终止分支：Program 正常返回并实际使环境由未终止转为终止，独立评分通过且没有 Agent 补做时，允许记为 `task_outcome` 正向；无需虚构下游输出消费。旧运行记录保持原样，修改后从空 Bank 重跑固定 pilot。

第三方评分来源与许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
