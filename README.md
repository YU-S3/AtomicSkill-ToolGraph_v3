# SkillCompiler empirical v3.1-CF3

唯一生产入口为 `skillcompiler.empirical.v1`。流程是普通 Skill 接口与指导 → Python Program → 固定真实 Train 试用 → usable 版本 → Workflow/动态执行 → 冻结库与独立评分。

本轮范围是 Ours。其他方法复现和更多模型 API 接入不在本轮范围；现有 DeepSeek 配置维持 high reasoning，实际模型 ID 如实记录。小样本 pilot 与正式实验使用独立 Bank，pilot 没有问题，也不需要为了正式划分重新跑。

本轮唯一实施入口为 [CF3 修改规范](docs/specs/SkillCompiler_CF3.md)，CF2 已完成的公开接口、状态与文件修复继续保留。公共 split、scorer、模型能力锁、high 推理和原预算维持原值。

CF3 节点明确区分 `dynamic/skill/program`：参考 Skill 不授予自动路线或输出约束；执行绑定读取真实资产接口，参数 ready 的 usable Program 继续自动接管并连续交接。局部 `patch_node` 只改本题实例，显式与系统重规划共用一次额度，之后最多一次剩余任务 Dynamic。逻辑决策 checkpoint 区分未完成响应恢复与新决策，未知在途副作用仍停止。

新配置记录 `empirical-v3.1-CF3`、`empirical.recoverable-takeover.v1` 和 checkpoint v2。使用独立输出目录；不得续写旧策略 checkpoint 或自行启动新的收费试验。新版学习与成本效果需要按用户授权另行测量。

[CF3 交付报告](reports/CF3_交付报告.md)与[逐项核验](reports/cf3_verification.json)记录 160 项通过的最终回归，以及原自然学出的 Heat／双对象程序在真实 ALFWorld 的只读执行层核验。fresh CF3 学习和费用效果尚未测量。

[CF2 交付报告](reports/CF2_交付报告.md)记录最终工程检查与固定真实诊断。120 项回归通过；冻结六题为 6/6、375,234 tokens，ALFWorld 和文件类成本仍超过目标上界，质量对照尚未测量。逐项测试和计费见 [机器记录](reports/cf2_verification.json)。

CF2 已统一工具定义、公开状态、结果引用和增量文件发布；Office 的独立纯读批次最多 3 项，Builder 首次与所有恢复合计最多两次生成。其历史配置记录 `empirical-v3.1-CF2` 与 `simple.v2` ABI；旧 Frozen 开发对照仅经只读兼容视图执行，不改原程序或正向记录。

旧正式运行保持用户停止状态，不随补丁自动恢复。新策略正式训练使用新输出目录和独立空 Bank。工程回归、真实诊断和正式效果分别记录；回归通过不代表成本或质量目标已达标。

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
skillcompiler-prepare --authority data/main_experiment_v1 --resources /path/to/SkillCompiler_resources_20261003 --output /path/to/prepared --case-map data/resource_mappings/spreadsheet_cases.json
skillcompiler-acceptance --config configs/alfworld_empirical_seed42.yaml --materials /path/to/original/pilot --output /path/to/new/acceptance --env-file .env
skillcompiler-multibench --config configs/default.yaml --datasets /path/to/prepared --corpus-root /path/to/treasury_bulletins_parsed/transformed --output /path/to/new/smoke --env-file .env
```

单个 ALFWorld manifest 使用 `skillcompiler --config ... --manifest ... --output ...`；评估追加 `--frozen-bank ...`。新 Benchmark 使用 `skillcompiler-multibench` 的固定 2 Train＋1 Val smoke。DocVQA 需要锁定模型的真实图像能力，当前文本 DeepSeek 组合记录 unsupported，分数留空。

`--resume` 要求源码、配置和任务身份一致。已完成执行只恢复未完成学习；缓存响应与版本注册幂等。未知环境副作用停止尝试，不自动重放。`STOP_AFTER_TASK` 文件在下一题前停止。费用保留所有尝试，未知计费不会写成零。

ALFWorld 清单按数据根目录下的物理文件和 SHA256 解析，扫描到全部目标后停止；旧 `env_index` 不作为扫描上限。`task_identity_resolution.json` 保存原清单与当前环境 ID 的对应关系，reset 仍校验当前文件、序号、目标和签名。重复 discovery 从起点开始，缺题或身份变化时停止，不替换题目。

历史加载修复的 `skillcompiler-pilot ... --continue-val` 入口核对原 Train 配置、物理题目集合、源码差异及 Train／冻结 Bank digest。本轮 CF2 改动执行与学习策略，不能通过该入口继承旧执行 checkpoint。固定旧 Frozen 的开发对照使用新目录，单独标记 diagnostic。

文件 Adapter 的 `tool_definitions()` 返回实际公开工具列表，供 Learner／Builder 使用；Program 仍禁止递归调用 `execute_python`。修复后已从空 Bank 完成 OfficeQA、Spreadsheet 各 2 Train＋1 Val，实际成绩和费用见 [文件类冒烟报告](reports/file_adapter_smoke_result.json)。43 项回归（含真实 Docker）与安装包验证见 [工程验证](reports/file_adapter_fix_verification.json)。

正式实验唯一数据 authority 是 [main_experiment_v1](data/main_experiment_v1/manifest.json)。SearchQA 为 300/24/1400，SpreadsheetBench 为 200/20/180，OfficeQA 为 120/24/102，DocVQA 正式采用自然分层的 180/22/201（Reserve131），LiveMath 为 60/17/100，ALFWorld 为 120/24/134。OfficeQA 按 difficulty，LiveMath 只按 theorem_type 分层；ALFWorld Val/Test 保留原物理成员和顺序。旧 `splits/` 与旧核对报告保留历史用途，正式 runner 不读取它们。

```bash
skillcompiler-manifest verify --authority data/main_experiment_v1
# 已有旧资源池也可直接按 canonical ID materialize，绝不重新切分：
skillcompiler-prepare --authority data/main_experiment_v1 --prepared-pool /path/to/old/prepared --output /path/to/formal/prepared
skillcompiler-formal --config configs/main_experiment_v1.yaml --datasets /path/to/formal/prepared --corpus-root /path/to/treasury_bulletins_parsed/transformed --output /path/to/new/formal --env-file .env
```

正式 runner 仅用 run_seed=42/43/44 打乱 Train，Val/Test 保持 canonical 顺序。每个 model/benchmark/seed 从独立空 Bank 开始，Train 结束后冻结并以只读方式执行 Val/Test；Ours 不用 Val 选择候选。已有方法参数与各角色预算不变，Provider seed 不支持时记录 unsupported/null。尚未配置的正式模型保持空值，当前模型的 DocVQA 组合明确 unsupported，不用 OCR 或文本替代真实图像。

每个正式 Run 旁路保存规范要求的八类 JSONL、resolved config、run manifest、实际 artifact 版本与最终 frozen manifest，并保留原生日志。实际 HTTP 请求、原始 usage、重试、环境交互和评分器原始输出均直接记录，不额外请求模型。私有 reasoning_content 按原有边界移除，完整请求的哈希与该移除标记保留；评分器原始输出只进入审计文件，不进入 Learner 输入。训练单位区分实际 trajectory、learning_update 与 program_trial，重复消费按真实单位记录。旧 pilot 不回填缺失字段。

[历史工程对齐验收](reports/main_experiment_v1_alignment.json)已通过，含 51 项回归、三 seed 顺序、原 pilot 335 文件不变与安装验证。[历史启动记录](reports/main_experiment_v1_launch.json)对应现有 DeepSeek 的 15 个可执行组合和 3 个 DocVQA unsupported 组合；该矩阵已被用户停止，当前状态以本机 `matrix.json` 为准。六个指定正式模型的接口仍按用户要求留空，当前实际 `deepseek-v4-flash` 不冒充 `DeepSeek-V4.1-flash`。

## 项目结构与边界

- `src/atomic_skillgraph/empirical/`：通用 Bank、普通参数引用、Planner/Learner、执行器、Docker worker、恢复与预算。
- `src/atomic_skillgraph/harness/`：ALFWorld 和五个新增 Benchmark 的公开输入、工具和独立评分器。数据集特定规则只在此层与数据准备层。
- `src/atomic_skillgraph/experiments/`：可安装的固定清单、运行、验收和 smoke 入口；根 `experiments/` 是兼容入口。
- `benchmark_profiles.json`、`models.lock.json`、`splits/`：资源条件、模型能力与固定 ID 清单；完整公开输入和私有 evaluator 记录留在本地资源目录。
- `cleanup_manifest.json`：逐文件删除与迁移分类；[历史版本](docs/history/README.md)保存旧实现与结果来源。

每个 seed 独立空 Bank；两个不同物理 Train 任务正向实测才成为 usable，repair 不继承旧版本成功。冻结仅包含 usable Program 和普通指导/策略；Val/Test 不更新 Bank。模型自报成功、文件存在与正式评分分别记录。

用户在本轮明确补充了整题终止分支：Program 正常返回并实际使环境由未终止转为终止，独立评分通过且没有 Agent 补做时，允许记为 `task_outcome` 正向；无需虚构下游输出消费。旧运行记录保持原样，CF2 只执行规范第 18 节的固定诊断，不新增 Train12 pilot。

第三方评分来源与许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
