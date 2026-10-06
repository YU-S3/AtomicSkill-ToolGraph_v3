# CF4 工程交付（2026-10-06）

基线 `bc63d02bc3bb0988cdbd15a203a068fc162f0b9b`；受测源码 `8550285`。最终交付在该源码上补充报告，源码指纹以 [核验记录](cf4_verification.json) 为准。实施规范为 [CF4](../docs/specs/SkillCompiler_CF4.md)。

本轮只修改 Ours。比较方法按用户此前“对比方法的部分不用管”的要求未接入，48 个 method×benchmark cell 的真实状态见 [公共契约矩阵](cf4_public_contract_matrix.json)。提供独立公共包供接入，不声称八方法已公平接通。没有启动 CF4 收费实验；现有 DeepSeek、high、预算、一次 Extractor/Builder 修复与两个独立 Train 正向晋升标准保持。

## 实际改动

- LiveMath 通过公共正常化函数补全原始候选，严格拒绝冲突和缺失文本；公开候选、私有评分候选、最终 HTTP 候选一致。
- SearchQA／LiveMath／DocVQA／OfficeQA 的公开答案格式进入实际请求。单轮 QA 仍只解一次，原图和上下文保留，独立评分器不变。
- Office `grep(pattern, paths?)` 接通 Planner、Runtime、Learner、Builder ABI。省略范围搜索全部授权文本；空数组不搜索；限定范围预先完整校验、去重排序，共享 40 个总命中及 Unicode 字符偏移。原始范围、执行范围、截断状态进入被动记录；检索失败键含规范范围。
- Learner 先纯分类，再验证、持久化与合并绑定。同版本 usable 的 build/trial 为幂等 no-op，继续保存经验、Workflow 和处理其他相关待办；真正候选仍最多两槽，已执行绑定不改。usable no-op 的多余绑定不被提前数量校验挡住，候选数量约束在语义接收边界保持。
- 真正请求的案例、必填参数／类型和动作前缀在原有一次 Extractor 修复内验证。没有适用案例可 defer；再次非法为 rejected，保留原始提议和错误，不改任务分数、不重跑任务。
- Planner 使用真实绑定的只读能力卡片；Workflow 统计动态、带／不带 usable 的 Skill 绑定和显式 usable Program，不添加资格证明或强制选路。
- 共用 `model_view.py` 提供可读分区和来源引用。TaskContext 的材料来源使用独立命名空间，原生结果编号不变；请求前保存引用。真实参数、原始结果、全部合法调用和实际进度判定保留。
- 版本为 `empirical-v3.1-CF4`／`empirical.model-view.cf4`。配置与 run 身份记录并核对公共包、材料、Office 授权语料 hash。新单 cell 入口复用原 campaign，Val 后可正常暂停为 `awaiting_test`，同版本恢复只补 Test。

替换与保留清单见核验 JSON 的 `removal_replacement`；没有重建学习架构、自动删除旧 Workflow 或改写旧实验记录。

## 核验与边界

最终源码完整 **214 passed，0 failed，0 skipped**。原 160 项覆盖继续保留；仅将模型视图的字段路径断言及获批的学习拒绝语义更新为 CF4，并增加 Spreadsheet 三变体检查。T01–T38 的实际 pytest 节点、T01/T16/T20 的旧行为与新行为差分见核验记录；原始 stdout 和 JUnit 在审查包。

真实 Docker／ALFWorld 只读执行了既有自然生成的 Heat 和双对象 Program，Planner 的返回由夹具控制。这证明执行接管与隔离保持，不能证明真实 CF4 模型已经愿意选程序，也不是新训练证据。文件程序检查包含独立评分、三变体、只读发布和不补做动态求解。

全量 LiveMath 审计确认 177 题均缺原正确候选 A，原文本均存在，已补全；重复、冲突、未解决为零。Train60／Val17／Test100 的成员、顺序和 physical_key 保持；全部六 Benchmark 的 canonical authority 仍为 `b85e0c2e442ca6b97ef98ece895719d4231f18db29682b0fcb6fdf3f91c37c95`。详情见 [177 题完整性报告](cf4_livemath_integrity.json)。

原固定六题的全部 82 次 HTTP 请求已离线投影，目标和合法调用保留，引用逐个解析；旧 run 与 Frozen 前后 hash 相同。使用 CF4 提示与投影后，消息 1,963,587 → 1,888,065 字节，工具 Schema 379,288 字节不变。这里是序列化字节，**不等于测得 token、费用或成功率改善**。DeepSeek 的 DocVQA 继续 unsupported，不引入 OCR 替代。

## 本机材料与入口

已生成的新六基准材料：

```text
/home/yangchengyu/main_experiment_v1_resources_cf4_20261006_v1
```

独立公共 wheel 为 `skillcompiler_bench_contracts-3.1.4-py3-none-any.whl`，只含标准库契约和 MIT 许可，不含 Ours；源码 hash 与 wheel hash 在核验记录。主包和公共包均已安装核验，单 cell 的仓库、模块、安装 CLI `--help` 与零 LLM 分派通过。

用户确认启动后，在固定干净源码目录使用现有 WSL Python（或从该目录 editable 安装 CLI），例如：

```bash
cd /home/yangchengyu/asg_cf4_20261006
PYTHONPATH=src /home/yangchengyu/asg_alfworld_venv/bin/python -m atomic_skillgraph.experiments.run_single_cell --config configs/main_experiment_v1.yaml --benchmark alfworld --seed 42 --datasets /home/yangchengyu/main_experiment_v1_resources_cf4_20261006_v1 --output /home/yangchengyu/alfworld_cf4_seed42_fresh --env-file /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3/.env --stop-after-val
```

以上是完整 Train120→Frozen→Val24 的启动命令，本轮未执行；没有再增加 12＋6 pilot。审查后同 commit/config/材料去掉 `--stop-after-val` 并增加 `--resume`，只运行 Test。其他基准替换 canonical 名称；OfficeQA 必须另传现有语料根目录。三个 seed 的新 Bank 各自独立，不能续用 CF3 checkpoint 或旧 Frozen 初始化。
