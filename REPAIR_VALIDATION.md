# R9.2.1 修复与验收记录

日期：2026-09-14。基线 main：`64fd4b2768a36f4190385f560aa47108bf305449`。
本报告对应包含它的提交；live manifest 另记录工作区源码指纹、配置及场景哈希。
当前状态：文档 §12.2 所需工程验收及真实自主链证据已具备，可启动新的单 seed42 空库 Full120/Frozen134。
这不表示原七题 smoke 的 passed=true，也不是方法准确率已验证；原失败、预算耗尽及模型路线问题全部保留如下。未启动正式实验。

## 实现与边界

- P1：Runtime Preparation/Seeded、native draft schema 和公共接口共享输出推导说明。明确 input_identity 类型/required/resolution 边界和 fresh output 唯一 (predicate, argument_role) 来源。R0 算法未修改。
- P2：IF/STOP_WHEN 增加只读 action_catalog match。native schema、嵌套 static、local scope、实例常量检查和执行器采用一致规则。使用 Harness matcher 及当前 revision；空匹配 false，非法引用/状态明确报错，matcher 异常传播。旧七个 operator、strict FOR_EACH、required RETURN、finally 恢复和资源设置保持原行为。
- 修复已复现的语义输入/具体见证冲突：最终校验仅在局部使用 Harness 已确认的具体赋值；不改写原输入、RETURN、父 binding 或 Repeat。具体输入冲突、缺失 witness、错误输出和过期见证仍拒绝。ToolRunner 原直接 final-effect 校验失败时，通过同一 Harness resolver 验证见证，未修改 passed authority。
- automation 回复补齐当前公共 observation/catalog，包括 cached/rejected draft；避免多动作 trial 后缺少当前动作含义。
- ToolRunner 已解析的 step/final Effect 不再附带原绑定表做第二次替换。真实 EXAMINE 循环的 object=cabinet_1 不会被无关输入 object=egg 覆盖；缺失 local、错误实体和无动作见证仍拒绝。语义 witness 也先完成局部引用解析再做最终验证。
- Preparation 在 trial 返回后的下一次模型调用超预算时，若同一父 occurrence 的新 Seeded 会话实际取得有效 turn，补记父 Runtime 接续；仅在独立父验证通过后记完成。失败的新会话、其他 occurrence、terminal trial 不获此标记。没有新增重试或预算。
- 真实请求审查发现 Builder 缺少当前输入/动作目录以及完整证据筛选说明。补入原本已授权的公开信息和目标停止/实例参数规则，保留 NO_TOOL 和所有失败。
- 将 Harness 已有 public catalog→predicate 参数映射公开为接口契约，解释当前事实来源与有效期。ALFWorld 公开说明与原投影共享同一数据定义，投影行为与基线精确等价。没有新增 benchmark 解题路线、任务族分支、事实推断或成功规则。
- 定向 driver 只在 experiments/tests，正式入口不导入。仅固定首个 native draft，live 后续由真实 Runtime/Builder 决策；T1 才使用脚本 proposal，未把其程序塞入真实 Builder prompt。独立 bank、非正式标记、禁止覆盖输出，NO_TOOL 不计正例通过。

生产文件位于 src/atomic_skillgraph：tooling/{ir,validator,runtime_interface}.py、
agents/{structured_submission,runtime_prompt_texts,context_builder}.py、
runtime/{automation,tool_runner,node_executor}.py、validation/atomic_validator.py、
harness/alfworld.py。新增独立 runner 和 tests 随本提交交付。

## T0/T1 与完整回归

```bash
PYTHONPATH=src:. /home/yangchengyu/asg_alfworld_venv/bin/python -m pytest -q \
  --junitxml=runs/r921_scope_review/pytest_final7.xml
```

结果：**1263 passed，0 failed，0 skipped，39.54 秒**。
日志：runs/r921_scope_review/pytest_final7.log。无 xfail，未删除原行为断言。
三个旧 unit fixture 只补齐新读取的空 action_catalog/revision，验证断言不变。
最终 Builder instruction 11618 字符，SHA256：
7c1cc67f3b10700a795dafba0595e02aeb29d98367d7cdfc4050768ef9496edd。

新增测试覆盖真实 Preparation 缺输入 preflight、无实现 Seeded、多候选/干扰物/关闭候选、
IF/STOP_WHEN、嵌套 local、R0/NO_TOOL/static/缺目标/错 RETURN 拒绝、原动作预算、
won/done-without-won、Repeat 不提前提交、全 System Frozen stored_composite 和下一题隔离。
R0/static/ToolRunner/R1 不被 mock 成功。
Online R1→E1/admission/reuse 复用原 test_v32_r3_full_system_self_tool_reuse.py 完整回归。

确定性双案例完整证据：runs/r921_scope_review/scripted_core_two_delivery/。

| 场景 | 入口 | Trial 动作 | R1 | 父独立完成 |
|---|---|---:|---|---|
| locate_across_empty_and_distractor | Preparation | 3 GO_TO | 通过 | 是 |
| locate_inside_openable_candidate | Seeded | 3 GO_TO + OPEN | 通过 | 是 |

外部 tokens=0。每例有安全请求/回复、trace.json、timeline.json、case_result.json。
修复前失败证据仍在 runs/r921_scope_review/scripted_core_two/；初始四项 pytest 失败见 pytest.log。
该次真实 resolver 已找到 apple_1/drawer_2，但最终 concrete 校验仍用 apple，已用正负测试锁定。

基线旧 ToolRunner/IR 在内存加载，旧程序的新旧输出/动作/revision/budget 完全相同：
runs/r921_scope_review/legacy_replay_comparison.json。
原公共关系投影在五个变化 revision 上完全相同：
runs/r921_scope_review/public_projection_comparison.json。

## T2 真实 Builder：保留失败与全部成本

所有尝试用 configs/alfworld_train_full_120_r92_seed42.yaml 的原模型/reasoning/cap/预算，
同一 core-two-v1 场景。每次均因具体接口修改才新建目录，不随机换题、不覆盖。
目录前缀 runs/r921_targeted_live_builder_20260914_：

| 后缀 | 实际结果 | 暴露问题 |
|---|---|---|
| fix1 | 0/2，89070 外部 tokens | 预设当前 EXAMINE 可用；另一例 NO_TOOL，无法确认 discovery 来源和目标筛选 |
| fix2 | 1/2，66653 外部 tokens | 第二例在干扰物 discovery 上提前停止，R1 拒绝 |
| fix3 | 1/2，89120 外部 tokens | 第二例只导航，缺少后来可用的交互；父 Runtime 接管成功，但 trial 未通过 |
| fix4 | 1/2，78639 外部 tokens | 第二例仍遗漏访问候选内容的交互，trial 失败；未算通过 |
| fix5 | **2/2，59895 外部 tokens，314.47 秒** | Preparation 3 动作、Seeded 4 动作；两例 R1 和父独立完成都通过 |

fix5 没有换 core-two 场景、添加解题程序或修改执行/准入规则。补充的是所有 Builder 共用的
严格空集合失败及可访问性说明。真实请求/回复可核对。其后只修改定向报告：两例未全部
完成不提前报 suite passed；异常退出保留当例已发生用量。最终 pytest 包含这两项回归。

每例 tool_builder_001_request.json 是实际安全请求，tool_builder_001_reply.json 包含原始
proposal/NO_TOOL、rationale、usage 和 request id，不含 API headers/key。
internal_000_usage.json、trace.json、timeline.json、case_result.json、根 summary.json
记录真实 Runtime/Builder 成本与阶段结果。fixture forced draft 为零 token，不伪装外部调用。

真实 ALFWorld 场景在生成前固定为 train 索引 0：
runs/r921_scope_review/fixed_train_zero.json 保存真实 TaskManifest、game 路径/SHA256 和初始公开观察。
未用 Frozen134。runs/r921_targeted_real_alfworld_20260914_fix3/ 中 trial 为 7 次 GO_TO；
第一步已看见 cellphone_1，但继续走完，返回时没有当前 witness，R1 正确拒绝。
父 Runtime 返回 bed_1 后 TAKE 成功；该次不计 targeted 通过。

原 train0 的 fix4 trial/R1/父完成均成功，但只执行 1 动作，仍按 >=2 的原门槛记失败；
用量 22142 tokens，235.06 秒。此场景第一候选即见目标，不足以验证多动作路径。
随后显式更新开发 fixture scene_revision=2，固定 train 索引 1，生成前保存
runs/r921_scope_review/fixed_train_one_v2.json；旧场景/失败均保留，未循环换题挑成功。
真实环境新证据：/home/yangchengyu/asg_r921_validation/real_train_one/。
完整平行备份：runs/r921_scope_review/real_train_one/。
真实 Builder 生成程序，**7 个 accepted trial 动作、R1 通过、父独立完成**；
23873 外部 tokens，138.76 秒。未提供隐藏目标位置或手写程序，未使用 Frozen134。

所有八次真实定向运行合计 **462109 外部 tokens**，含失败和 NO_TOOL；不与七题自主成本混算。
完整汇总：runs/r921_scope_review/acceptance_audit.json。

## T3 原七任务自主 smoke

原七题选择/评分代码未改。之前的 6/7、0 trial 仍见 R92_REVIEW_VALIDATION.md，不冒充本轮结果。
本轮首次启动在调用 Runtime 前因旧 provider capability 指纹与新代码不符而退出，
见 runs/r921_scope_review/autonomous7_fix3.log；该次没有七题结果，也未覆盖旧缓存。

三轮使用相同原七题；配置副本只改变独立输出路径，资源和任务选择不变。每个新代码指纹
重新 provider probe 后启动原 --real-alfworld。原始结果不覆盖，定向计数不混入自主 smoke。

| 版本/目录 | 原始结果 | 具体证据 |
|---|---|---|
| fix4，runs/r921_autonomous7_final/ | passed=false，6/7；1534702 tokens / 96 calls | Direct=false、dataflow=false；4 次 trial，其中一次 23 动作 R1 通过，但同一父节点先前已消耗 9 次 trial 动作，随后原 node action budget 耗尽；不计父完成链 |
| fix5，/home/yangchengyu/asg_r921_validation/autonomous7_release/ | passed=false，7/7 严格成功；1196528 tokens / 84 calls | dataflow=true、Direct=false；一次 26 动作 trial 正确找到 egg_1，但 draft 缺少正式目标约束，R1 歧义拒绝；父 Runtime 接管完成 |
| 契约说明补充，/home/yangchengyu/asg_r921_validation/autonomous7_contract/ | passed=false，6/7；1970077 tokens / 120 calls | 3 动作 trial 暴露重复绑定误拒；第二次自主 trial 25 动作、R1 真通过，同一父节点随后 Seeded 独立完成；全题稍后在另一个节点 token 耗尽 |

三个已完成版本的 UNKNOWN=0、won/contract mismatch=0；全部 Trace 用量完整，无基础设施失败。
tokens 含维护 Trace，按 usage event id 去重；provider capability probe 另行保存，不计七题分母。
对应日志为 runs/r921_scope_review/autonomous7_{final,release,contract}.log。
fix5 完整副本：runs/r921_scope_review/autonomous7_release/。

fix5 自主提案的准确根因（Trace trace_35be4ec9057942f3b1795c5016c3fbec）：
输入 object_class=egg，两个 fresh 输出 object/source，Effect 只有
entity.discovered_at(entity=$object, location=$source)，没有引用 object_class。
Tool RETURN 的 egg_1/diningtable_1 本身正确，Tool final-effect 检查也通过；但 R1 不允许
把 RETURN 当作见证选择 authority，多个当前物体仍对应多个正式 witness assignment，故拒绝。
这不是 matcher/IR 找不到目标，也不应让 R1 自动选一个结果。补充 Runtime 共享输出契约说明：
唯一 predicate/role 推导不等于唯一具体见证；未被 Effect 引用的输入不提供目标约束；
RETURN 和描述文字不替代正式约束。R0/R1/Resolver 算法未因此修改。
新增同现场含目标与干扰物的真实 validator 回归：未引用目标的 draft 保持拒绝，正确引用
目标输入的 Effect 通过，原输入/输出/世界快照都不变。

fix5 warm train18/train22 均为 stored_composite → no_compatible_implementation → Seeded 成功。
源放置 Tool 的 replay 触发真实终止，tool_ir_replay_terminal_interrupted 保持拒绝完整准入；
没有修改该边界来制造 Direct=true，也没有把 Seeded 成功重命名为 Direct。

### 最终自主链与两处实际缺陷的复核

完整原始 Trace（原文件未修改）：
runs/r921_scope_review/autonomous7_contract/run_20260914T085233.430611Z_595803/traces/trace_a354b5a403d448759c34fe9cfea7e5f4.json。

自主 draft `search_egg_location_v1` → 真实 ToolBuilder create → static 通过 →
环境 event 10..34，共 25 个 accepted 动作，revision 10→35 → R1 真通过，
输出 container=diningtable_1，见证为 revision 35 的 entity.discovered_at(egg_1,diningtable_1)。
随后同一父 occurrence `_001` 的 SeededSession 发出 TAKE，revision 35→36；
父 Atomic 独立验证 agent.holds(egg_1) 通过，节点状态 seeded_success。
这条链没有强制 Runtime 提案、手写 Builder 程序或改环境。

原 trial 的 parent_resumed/completed=false 是统计漏记，不能直接推出父节点失败：
Preparation 的第 8 次 provider call 将 session 用量从 97298 增至 114897，
超出原 100000 额度，所以 submit_tool_result 抛异常后没执行恢复标记；
同一父节点新 SeededSession 实际完成，但旧统计又要求该标记已存在。
新增正负集成回归重现真实预算异常及新 Seeded 接续；未改变 Session 预算或执行路线。
acceptance_audit.json 另列 independent_parent_statuses/later_independent_atomic_validations，
保留旧 parent_completed 原值，不篡改历史记录来制造通过。

更早的 `locate_required_object_v1` 在 EXAMINE cabinet_1 后误拒的根因确认为：
局部变量已解析成 cabinet_1，ToolRunner 却再传入 object=egg 的输入表，
Harness 旧兼容替换按参数名覆盖了实际值。只在 ToolRunner 的已解析 Effect 调用边界修复，
未改 Harness 对其他原始合同调用的兼容行为。真实 ALFWorld 重放原六动作前缀，
原双绑定调用 false、修复后的 step check true，环境快照及原输入不变。
证据：runs/r921_scope_review/real_effect_collision_replay.json（0 外部 tokens）；
对应脚本与 log 在同目录。最终新旧旧程序 replay 对比再次通过，动作/输出/revision/预算一致。

整题最终失败发生在另一个 occurrence `_003`，实际合同是 **close container**，不是搜索父节点或 heat 本身。
上游 `_002` 未锚定导航目标，Runtime 选择 garbagecan_1，输出通过 data_flow 进入 `_003.container`。
当前 catalog 没有相应 CLOSE affordance，preflight 拒绝；随后 Runtime 转向 microwave 并关闭它，
却不能用另一个实例偷换已绑定的垃圾桶角色。Preparation 用量 145784、Seeded 105362，
均触发原 session token gate；没有 HEAT，最终整题失败。预算耗尽后返回的候选动作未执行。
未发现这里需要用 benchmark 规则硬改角色或放宽身份校验的证据，因此保留失败作为模型路线风险。
这不是“25 动作 helper 用光父动作额度”的那一轮；前一轮 action exhaustion 单独列在上表。

最后两处代码修复后执行了完整 1263 项回归、真实原前缀复现及旧程序对比。
没有付费重复整个七题直到抽到全对；上述三轮自主结果与各自指纹、失败和全部费用仍可核查。

## 放行状态

T0/T1 通过；真实 Builder core-two 2/2、真实 ALFWorld 定向多动作/R1/父节点通过；
原自主七题已出现真实自主多动作 R1→独立父完成链；全回归、Frozen/Repeat/terminal 隔离和旧 replay 对比通过。
据冻结文档 §12.2，可以放行新的单 seed42 正式 run。不能声称 smoke 全绿或正式实验必然高成功率。
最新自主七题仍为 6/7、原 passed=false；其终题模型路线/预算风险必须由正式结果如实计入。
仅向 main 提交本轮生产/测试/报告，不操作 baseline，不提交用户其他草稿/凭据，不启动正式实验。
