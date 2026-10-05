# ALFWorld 隔离试用映射修复与独立 12 Train + 6 Val

旧运行 `/home/yangchengyu/alfworld_seed42_train120_cf2_20261005` 已提交 5 题，第 6 题停在首次 Program 试用前。对应 15 次模型请求均已返回，Extractor/Builder 已完成，没有原生试用事件。进程仍在消耗 CPU：新隔离适配器没有继承主适配器已核验的题目映射，reset 再次从原生题库开头扫描到目标索引。实际问题是重复题目发现，尚未执行的试用不能算普通模型失败或正向证据。

已对核对过命令行的旧 PID 14859 发送 SIGINT，确认 `completion.json`、`matrix.json` 已持久化 interrupted 后，再终止仍在扫描的线程。原请求、轨迹、费用、Bank 和部分学习记录均保留，新运行不导入它们。

受测源码固定为 `d046c1de3c7488a30bbe1edc1e0397b695041ba4`。隔离适配器复用主适配器已核验的不可变 file/index 映射，先检查配置身份一致；主环境和试用环境各自创建世界，不共享 episode 或状态。原文件/索引、goal、signature 校验仍执行。通用试用入口仅调用适配器的可选能力，没有 benchmark 分支。未修改学习逻辑、晋升标准、repair 次数或模型预算。

无 LLM 回归 **128/128 通过**，日志在新运行目录的 `engineering/verified_pytest.log` 和 `verified_pytest.xml`。新增回归覆盖生产 `test_program` 路径首次试用不重扫、主环境状态不变、错误 file/index 与配置拒绝。原首次冷启动题库发现流程不变，准备阶段仍可能较长；修复针对每次隔离试用重复发现。

最新用户要求替代此前“前 12 题后停止”的控制：本次为独立空 Bank 的 **seed42 → 固定 12 Train → Frozen → 原固定 6 Val**。12 Train 与旧运行所选 canonical Train120 前 12 题及顺序完全一致；6 Val 使用原 valid_seen6 的物理任务及顺序，全部属于公共 canonical Val24。公共清单 authority SHA256 为 `b85e0c2e442ca6b97ef98ece895719d4231f18db29682b0fcb6fdf3f91c37c95`。selection 文件记录父清单哈希、实际任务及顺序。

新目录：`/home/yangchengyu/alfworld_seed42_12train6val_trialfix_20261005`。
固定干净 WSL checkout：`/home/yangchengyu/asg_alfworld_trial_fix_20261005`。
驱动：新目录内 `run_12_6.py`，使用现有 canonical、config、runner、FormalLog 模块，脚本哈希进入身份。启动前拒绝已有 Bank 或 execution manifest，初始化后记录空 Bank 的四张表计数和未导入资产。模型仍为既有 deepseek-v4-flash/high。

Train12 完成后由原 runner 冻结 Bank，随后自动只读运行原 6 Val；记录并要求冻结目录 Val 前后哈希一致。运行失败或中断则保留记录并停止，不自动重复实验。此运行标记 `formal_score=false`，不冒充完整 Train120/Val24 正式实验，不继承旧 Bank，不启动其他 seed 或模型矩阵。原正式矩阵不变。

实际状态以新目录 `process.json`、`matrix.json`、`run.log`、Train/Val 的 traces、summary 和 completion 为准。本记录不作性能判断。
