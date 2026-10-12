# OfficeQA 当前来源绑定修复、真实验证与 fresh seed42 启动核查

本轮完成了最小提示修复、固定源码离线回归、有界真实学习检查和 fresh 配置核查。工程检查通过；真实模型尚未生成 usable Program，因此“自然学习 → Frozen 复用”的效用验收没有通过。以下启动入口用于继续观察完整正式 Train120 → Frozen → Val24 → Test102，不以模拟生成的资产替代自然学习，不承诺准确率或 Program 增长。

## 固定身份与修改

- 受测并用于新 fresh 的提交：`eb7ee3de83cf52b916e854b2e691134acc564288`。
- Python 源码 SHA256：`30c00d8f9309625e0fdbe68dce56cc3bba57ff0ece38e13f4171df55fe2f1eb5`。
- 干净固定 checkout：`/home/yangchengyu/asg_officeqa_current_binding_fixed_20261013`。
- 实现版本：`empirical-v3.4-current-source-binding`；材料版本仍为 `learner.material.public-directory.v12`；公共视图仍为 v8。
- 模型：既有 `deepseek-v4-flash` 配置。完整请求、公开响应、实际 model 和 raw_usage 在本轮审查包中。
- Docker 镜像：`sha256:4db520e20cd121f830731c2d0c0bcfbe9cacd254d1817262404cc18e1e757783`。

Learner 说明现在明确：`related` 是 Bank 资产；`local_evidence` 是临时／原生执行记录，不等于持久化 Program。新 Program 可以绑定当前 `experience.local_evidence`，不要求已有 job 或第二条 Train。保留合理的 `no_change`，参数必须来自精确公共／执行来源，输出必须有真实类型引用。没有改晋升、source trial、repair 次数、权限、评分或程序资格。

启动脚本仅增加 `OFFICE_R1_CONFIG` 选项，默认行为保持原样。新的 Office 配置在独立交付目录，不改变其他 benchmark 的配置，不热改旧实验。

## 原始记录核查

UID0208 原始执行有七次临时 Python，其中包含真实 `ctx.call` 检索和纯数值计算。最终学习材料受原 16 KiB 完整 payload 上限约束，只留下优先级最高的完整 OLS 计算卡。检索脚本未留在最终请求中，不能把本轮 OLS 检查声称为自然检索 Program 的学习或复用成功。

旧 Learner 把“原生代码已经计算过”当成无需持久化的理由，并称没有 completed Train bindings。但当前真实卡已有 case_id、local_evidence_ref、源代码及输入引用。提示末尾关于已有 job／completed cases 的表述存在歧义，本轮已澄清。

期间完整来源回归发现提示变长会挤掉源代码尾部，已在发出该版模型请求前缩短提示。失败的零 HTTP fixture 保留；最终固定版请求为 16,360 B，OLS 源码完整，未截断。没有增加材料上限或伪造引用。

## 真实有界检查

两次均使用现成的真实 Train 轨迹与此前实际发送并正确评分的独立 finish 回答，不重新求解，不改写原错误轨迹。每次使用新的空诊断 Bank；所有诊断资产不作为正式资产。

| 固定提交 | 实际目录 | 结果 | tokens | HTTP |
| --- | --- | --- | ---: | ---: |
| `b12d61f` | `/home/yangchengyu/officeqa_durable_gap_paid_check_20261013` | no_change，0 usable Program | 4,798 | 1 |
| `eb7ee3d` | `/home/yangchengyu/officeqa_current_binding_paid_check_20261013` | 提出 Program；一次原结构 repair 后 rejected，0 usable Program | 9,534 | 2 |

最终提案的具体错误：把三段数组拼成 `series`，却将输入引用指向第一段数组；`n_train=30` 的来源实际指向值 31；还声明未绑定的 `answer`、`pred_next`、`actual_next`、`abs_diff` 为必需输出。Host 按原规则拒绝，没有放宽来源、补造答案或手工晋升。

这次是普通模型提议不合法，不是超预算、材料超限、缺工具或 Docker/检索 I/O 异常。结构 repair 仍最多一次。由于没有自然 usable Program，自动跳过依赖该资产的 Frozen 参数检查与新 Val 求解，新增 Val/Test 请求为零。没有为了取得资产反复重采样同一版本。

本轮新增实际费用：14,332 tokens、3 HTTP。前后所有旧账本和来源轨迹哈希一致，无未知计费。已有学习诊断累计使用 32,912 / 80,000 tokens；这两次没有耗尽学习预算，也没有依据这些记录扩大预算。

## 离线核验

固定 `eb7ee3d` 的 13 个相关测试文件：199 passed、19 skipped，耗时 159.66 秒。JUnit：`D:/T3S_exp/deliveries/officeqa_current_binding_offline_20261013.xml`。跳过项需要额外指定旧归档，不将 skipped 算作通过。另指定真实 Office 语料与原有 fresh 来源记录完成检索／材料生产回归，结果见单独 JUnit。

真实 Train 来源回归使用模拟 Learner/Builder 响应、实际生产 Bank/绑定校验和 Docker 执行：合法提案获得 usable，随后冻结；改变一个实际绑定的数值输入，原临时代码和 Frozen Program 分别实际执行，类型化输出完全一致，冻结目录前后字节哈希一致。实际模型 HTTP 为零。该回归只证明生产接口及参数化执行链路，不能替代自然模型学习成功或跨题复用成绩。

Fixture：`/home/yangchengyu/officeqa_current_binding_frozen_fixture_20261013`。本轮未把其 Program 放进正式 Bank。

fresh 全量离线预检还确认了：公共 canonical Office Train120/Val24/Test102 互斥，materialization 和外部文件 hash 一致；246 题逐一通过真实适配器 reset，公开模型视图只有原 goal/inputs，私有评分记录与题集对应；Train 使用 seed42 排序，Val/Test 保持 canonical 顺序；既有 corpus hash 和 Docker 镜像健康检查通过。预检不发模型 HTTP，不运行性能 Test。

预检：`D:/T3S_exp/deliveries/outputs/officeqa_fresh_readiness_20261013/preflight.json`。完整配置与费用来源 SHA256 一并保存，并在启动时复核。

## fresh 运行配置与费用

- 新目录：`/home/yangchengyu/officeqa_current_binding_fresh_seed42_formal_20261013/officeqa/seed42`，交付时不存在，尚未启动。
- Train：每题 400,000 tokens，求解 320,000、学习 80,000；原生调用 64 次。沿用此前批准的冷启动设置。
- Val/Test：每题 96,000 tokens、24 次原生调用。
- 累计 Office 总上限保持 36,096,000 tokens / 284,070 HTTP，不自动扩大。
- 旧正式和所有相关付费诊断按原 HTTP attempt ID 精确去重结转：24,567,954 tokens / 3,021 HTTP。
- 剩余累计额度：11,528,046 tokens / 281,049 HTTP。该额度不保证完成所有题目的理论最大消耗；触顶按原 governor 停止，不能误记为模型错误，也不能自动放大。
- 不继承任何旧 Bank、诊断 Bank、旧 completed tasks。旧正式的 `STOP_AFTER_TASK` 保留。
- 最初六题就是新正式 Train 的前六题。工程 gate 检查通过后，在同源码、同目录、同 Bank 自动继续剩余 Train，再 Frozen → Val → Test。正常 no_change/rejected 或低准确率不被伪造成基础设施故障，也不强制晋升资产。

在 WSL 启动（后台运行）：

```bash
bash /mnt/d/T3S_exp/deliveries/outputs/officeqa_fresh_readiness_20261013/start_officeqa_seed42.sh
```

该脚本的 `--print` 已零模型调用验收；重复 fresh 启动会拒绝已有目录。需要续跑时必须显式同版本 resume，不能删除目录或伪装换版本续跑。

查看日志：

```bash
tail -f /home/yangchengyu/officeqa_current_binding_fresh_seed42_formal_20261013/runner.log
```

查看逐阶段任务数和新增费用：

```bash
cd /home/yangchengyu/asg_officeqa_current_binding_fixed_20261013
PYTHONPATH=src /home/yangchengyu/asg_alfworld_venv/bin/python -m atomic_skillgraph.experiments.monitor_cells /home/yangchengyu/officeqa_current_binding_fresh_seed42_formal_20261013/officeqa/seed42
```

原包与旧运行保留。本轮审查包单独保存实际新请求、原来源、空诊断 Bank、明确标记的零模型 fixture、固定源码、JUnit、预检、准备好的启动配置和费用来源映射；不包含密钥或完整语料库。
