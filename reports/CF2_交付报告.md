# CF2 交付报告

实施依据：[CF2 最终规范](../docs/specs/SkillCompiler_CF2.md)。基线 `77959e89`，最终受测源码 `57ad1ed`。本轮修改 Ours，实际模型为既有 `deepseek-v4-flash`，各角色保持 high；数据成员、评分器、模型能力锁和原任务预算没有修改。结果均为开发诊断，`formal_score=false`。

工程回归通过，真实准备能力的小链已形成；冻结执行成本目标尚未全部达成。同条件 baseline 未运行，质量目标为 `unmeasured`。逐项测试、计费及源码身份见 [机器记录](cf2_verification.json)，原始材料见交付 ZIP。

## 修改和删除的旧分支

| 编号 | 生产落点 | 改动及被替代分支 |
|---|---|---|
| F01 | `SimpleAlfWorld._apply_public_result/_inventory/check_local` | Agent 和 Program 经 Broker 写入同一公开状态；MOVE/PUT 移除持有对象，明确空库存覆盖旧值，解析失败为 unknown。删除分散的持有状态更新。 |
| F02 | `ToolSpec`、`SimpleAlfWorld.tool_definitions/available_tools` | 名称、参数、返回类型、作用和单位来自同一注册定义；原生 HEAT/MOVE/PUT 语义直接传递。删除重复工具说明。 |
| F03 | `TaskContext`、`Broker.call/rpc`、`ValueStore` | 完整结果保存在本题，模型使用预览及 `result_id/path`；本地读取也计调用。替换每步重复完整反馈的投影。 |
| F04 | `Learner.learn/_realize`、`Bank.save_job/train_cases` | 缺 Program 的 Skill 与实现作业分别持久化；reuse/no_change 后仍可调度合法待办。删除仅按 propose 决策进入 Builder 的条件。 |
| F05 | `Learner._preflight_binding/_realize`、`EmpiricalSystem.agent` | 先固定真实 Train 绑定，最多两个物理案例；Builder 首次 32,768，length 恢复 65,536，共享一次修复和 262,144 学习预算。删除分别累加的修复机会。 |
| F06 | `BUILDER_PROMPT`、`EmpiricalSystem.test_program` | 导航参数必须来自公开工具允许值；试用使用固定案例/真实前缀，并实际执行和评分。未手改模型生成程序或验证题程序。 |
| F07 | `Bank.program_options`、`Planner.plan`、`Executor.run` | 相关程序卡片共用查询；目标、指导和状态可见；检查全部授权路线。删除两份按 ID 截取前 8 的菜单。 |
| F08 | `validate_workflow/ValueStore`、`Executor.run`、`progress_key` | 显式字段映射后自动接续；稳定失败 key 不含 revision 噪声。普通字典不被当作引用解包。 |
| F09 | `Workspace.publish` | 增量发布继承未改动产物，明确提交修改/删除，拒绝未声明变化并回滚。删除以本次 files 替换所有登记的语义。 |
| F10 | `Capabilities.final_submission_kind`、`Executor.run`、`EmpiricalSystem.test_program` | 最终 answer/bundle 就绪即提交，文件试用直接交独立 scorer；缺完整产物不记正向。删除提交型图结束后无条件整题重做。 |
| F11 | `OfficeAdapter._call`、`ToolSpec` | grep/read 共享 CRLF 规范化后的 Unicode 字符偏移；行号与 offset 分别明确标注。 |
| F12 | `EmpiricalSystem.agent`、`RuntimeDecision`、`Executor.run`、`TaskContext` | 1–3 项独立纯读先整体校验再顺序执行，各自计调用；结果用引用传递。删除 Runtime 无条件只允许一个 ToolCall 的分支。 |
| F13 | `Executor.run`、`OfficeAdapter.call/_path`、`run_multibench.run_smoke` | 原生额度耗尽后最多一次 finish-only；普通输入错误返回失败反馈，根目录/编码/未知在途操作仍停止；停止 Train 后不启动 Val。 |

Spreadsheet 的完整 bundle 与新产物规则保留在 Adapter；通用 empirical 代码不含题号或 Benchmark 名称特判。`FileAdapter.model_state` 保留原始 public inputs，包括输出位置与 solution contract。公共接口增加工具事实、结果引用、执行意图/结果用途、实现作业及提交类型；新 ABI 为 `simple.v2`。旧 `simple.v1` Frozen 仅只读兼容，原程序、信用和哈希未迁移。

## 验证口径

最终 clean WSL checkout 的测试为 **120/120**，T01–T40 每项均有生产模块映射。涉及 Program、文件计算和变体评分的检查使用实际 Docker，镜像 `sha256:4db520e20cd121f830731c2d0c0bcfbe9cacd254d1817262404cc18e1e757783`。固定反馈/拦截 HTTP 的测试与真实模型诊断分开标注。

旧源码与新源码的 F01/F09/F11/F13 四个确定性反例分别为旧版失败、新版通过；其余条目由生产路径回归覆盖，没有声称全部做过旧版实测。另用真实 ALFWorld 回放公开动作，并实际执行 TAKE→Program MOVE→INVENTORY 空的小链，当前持有缓存为空、known，局部检查未误报；无模型请求，测试程序未登记为 learned Program。

最终 wheel 安装在独立 venv；从 `/tmp` 调用七个真实入口的 `--help` 全部成功，模块来自安装后的 site-packages。Provider 请求构造测试检查最终 HTTP payload；原始 DocVQA 图像 SHA 和像素数据在请求中保留，当前文本模型的实际视觉诊断为 unsupported，没有替换为 OCR。

真实请求首次暴露 Planner 根 schema 的 HTTP 400，随后修为 object 根并完成实际请求验证。单轮 QA 的显式 guidance_only 与 realization_request 冲突已修复。文件投影遗漏 public inputs 已补回并增加最终 HTTP 断言。所有修复前请求、费用和失败仍保留，不回填旧日志。

## 固定真实诊断

| 诊断 | Train / 学习 | 冻结执行 | 成本状态 |
|---|---|---|---|
| 原 ALFWorld Frozen 六题 | 不重跑 Train12 | **6/6，375,234 tokens，均值 62,539** | 超过 300k 目标及 360k 上界；旧版同六题 690,135 |
| 新准备能力小链 | 两个真实物理案例正向；学习与试用 188,003 tokens | **1/1，12,873 tokens；Program GO_TO＋TAKE，输出被后继使用** | 机制已观察；不作为六题对照或正式分数 |
| SearchQA | 1/2，19,124 tokens | 1/1，1,509 tokens | ≤50k |
| LiveMath | 0/2，165,484 tokens | 0/1，12,412 tokens | ≤50k；普通答题失败保留 |
| OfficeQA | 0/2，257,194 tokens | **0/1，165,409 tokens** | >60k；finish-only length，无 ToolCall，按原规则结束 |
| SpreadsheetBench | 2/2，551,010 tokens | **1/1，108,301 tokens** | >60k；原变体 scorer 通过 |
| DocVQA | 未调用当前文本模型 | unsupported，score=null | 未测量 |

上述费用只分别表示各条链，不含被工程错误中断的初始请求。**本轮全部实际诊断已报告 usage 合计 2,175,853 tokens，另有 2 次计费未知的 HTTP 400；总费用不能据此填为确定值。** 完整去重 usage 与每次尝试见 ZIP 的 `all_usage.json`。文件冻结 Val 中 Office 原生调用 24 次、LLM 请求 20 次，Spreadsheet 原生调用 7 次、LLM 请求 12 次，两者 usable Program 调用均为 0；文件训练尚未形成 usable 复用链，未为降分/高成本另开训练。

旧 Frozen 六题使用原物理 ID、原预算和只读副本；没有重付 Train12，没有手改 Bank。两次初始 Planner 400 无已报告 usage，计费未知次数保留，不能填零。

准备能力来自两个已完成且物理不同的真实 Train 公开经验。两次 Learner 调用依次形成缺 Program 待办、reuse 后实现；Builder 一次 length 后按原共享机会恢复，版本取得两个真实正向。普通 Planner/Runtime 实际选择该程序，程序内部执行 GO_TO、TAKE 两次调用，输出被后继消费；Agent 接续 GO_TO、MOVE 后官方 won=True。该小链与旧 Frozen 六题是不同知识状态，不能混称正式泛化结果。

SearchQA/LiveMath 保持一次正式 solver；默认 guidance_only，显式 guidance_only 不启动 Builder。当前仍允许 Learner 明确声明 program_requested，未采用“全部单轮 QA 强制 guidance_only”的额外规则。DocVQA 当前锁不支持 image，实际 smoke 记 unsupported、score=null；图像 HTTP 构造检查不是付费视觉模型成绩。

文件类原固定两题 Train 保留，最终 Val 在修复公共输入投影后使用只读 Frozen 副本。Office 修复前 1/1、134,856 tokens 与最终结果分别记录，未择优替代。Train 的正确解答不意味着产生 usable 程序；本轮 Spreadsheet 两个生成版本仍为 candidate/deferred，Frozen 没有人工晋升。

## 运行身份、历史和入口

真实输出根为 `/home/yangchengyu/cf2_diagnostic_20261005`。学习及首次诊断源码为 `b452cee`，修复后六题/准备能力普通执行/QA 为 `9556f13`，文件 Train 为 `8bf1bef`，恢复公共输入后的文件 Val 和最终工程检查为 `57ad1ed`。各目录保存 config、execution manifest、实际任务清单、原始 traces/requests；跨修复衔接另有 continuation manifest，没有伪造同版本 resume。

固定清单 authority SHA256 仍为 `b85e0c2e442ca6b97ef98ece895719d4231f18db29682b0fcb6fdf3f91c37c95`。旧 pilot、文件 smoke 和已停止正式矩阵的 1,155 个文件保持原 SHA256；旧正式矩阵 interrupted=1、queued=14、unsupported=3。没有启动新正式矩阵。

运行入口均已安装验证。以下是入口和原路径说明，不是启动新的收费实验：

```bash
skillcompiler --help
skillcompiler-acceptance --help
skillcompiler-multibench --help
skillcompiler-formal --help
skillcompiler-manifest verify --authority /path/to/repo/data/main_experiment_v1
```

ALFWorld runner 的 `--resume` 要求相同任务/config/源码身份；通过输出目录 `STOP_AFTER_TASK` 在任务边界停止。未知在途副作用不能盲重放。跨源码不续写原运行；固定文件 Val 的实际 API 驱动脚本随材料提供。`skillcompiler-acceptance --resume-verification` 仅核验本入口已完整生成的结果，不增加模型/环境调用；首次旧 helper 因 HTTP 400 未产生完整 acceptance，不伪造其可恢复状态。

成本未达标的组合如实保留；没有通过删掉昂贵题、混合便宜 QA、扩大训练或更换模型掩盖。质量对照尚未测量，后续正式实验应按用户指令单独启动。
