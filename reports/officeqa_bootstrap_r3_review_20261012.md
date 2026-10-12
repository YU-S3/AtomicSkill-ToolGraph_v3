# OfficeQA 冷启动、学习材料与任务内临时检索核验

## 实测结论

已授权的一次固定 6 Train＋2 Val 在 `d245e4e` 完成：Train 1/6，Val 1/2；1 个 Skill，0 个 Program，Frozen 前后不变。该结果没有证明方法已能自然学习或降本。正式 OfficeQA 仍暂停。

本轮唯一答对的 Train `UID0115` 原学习请求为 17,529 B，超过既定 16,384 B，发送前拒绝、Learner HTTP=0。它不是学习 token 用尽。`cb2451b` 在保留完整 Schema、输入来源和输出选择器的前提下压缩冗余提示与可选预览，同记录完整请求为 16,292 B。

经用户批准，对原记录只做一次新版学习诊断，不重解、不重评分、无 Val/Test、空诊断 Bank：实际 Learner 请求成功、4,261 tokens／1 HTTP，无预算或材料拒绝；模型选择 `no_change`，未发 Builder，没有自然 Program。其理由包括“无历史 Train 样例”和“原生工具已覆盖任务”；这些并非当前规则对有用、可绑定局部程序的排除条件。不得将该结果改写为学习成功，也不自动追加请求。

## 新版执行入口

原 Office `execute_python` 无 ctx 且 `allowed_tools=[]`，只有纯计算能力；未有持久化 Program 时，Agent 无法用临时脚本批量调用语料工具。用户已明确授权补齐。

新实现 `empirical-v3.3-task-python-rpc`／`learner.material.public-directory.v10`：

- Office 的临时 Python 暴露 `ctx.call`，通过同一 Broker 调用现有 glob/read/grep/read_result。脚本启动及每个内部调用均计入当前题工具额度，保留父子事件和完整实际结果。
- 容器仍无网络、无直接 corpus 挂载；越界路径和递归 execute_python 仍拒绝。Worker 同时受自身及当前 owner 的更早 deadline 限制。
- Runtime 能读取正确的父程序结果；临时脚本不创建 Bank 资产。来源验证以父脚本的真实源代码和输出为证据，原始内部事件保留，prefix 不重复重放其子调用。
- 提示明确：空 Bank 可运行有界临时检索；第一条真实案例也可绑定；原生工具覆盖不自动排除有用的复用程序。仍允许合理 no_change，不强制每题生成资产。
- `$new`／真实 ID、发布相对路径、输入与输出来源、独立 Worker 试用、晋升标准及 repair 次数保持原规则。其它文件适配器未启用新的检索权限。

## 零 HTTP 验收

新入口及原 R3 回归 20 项通过。共享执行、文件、预算及合同回归 171 项通过、8 项跳过；另两项资源测试初次缺少 CF4_DATASETS，设置真实既有资源路径后均通过。共 193 个不同测试通过、8 个依赖旧归档的测试跳过。所有 Provider 响应夹具均明确为模拟，不作为自然效果证据。

真实 UID0115 的完整生产学习链路采用模拟提案、真实原记录和真实 Docker 试用通过；新版完整 Extractor 为 16,250 B，Builder 为 16,287 B，均≤16,384 B。参数化计算在新输入下执行并被模拟消费者使用，Frozen 未改变。

另以真正执行的临时检索脚本构成整题答错的来源：完整 Learner→Builder→独立来源匹配通过，Program 按正常规则 usable，并在不同检索参数下成功执行。该测试验证工程接口，不代表自然模型已提取此 Program。所有夹具/诊断 Bank 都不能进入正式实验。

## 费用和上限

| 事件 | 已计量 tokens | HTTP |
| --- | ---: | ---: |
| 已暂停的原正式前缀 | 22,201,318 | 2,847 |
| 固定 6 Train＋2 Val | 1,698,851 | 129 |
| 单条真实学习诊断 | 4,261 | 1 |
| 本轮后累计 | 23,904,430 | 2,977 |

所有实际请求已计量，unknown billing=false，两个原账本 hash 未改变。原 Office 累计上限仍为 36,096,000 tokens／284,070 HTTP。固定验收加诊断共 1,703,112 tokens／130 HTTP，仍在已批准的 2,592,000／240 内。

学习 pool 仍为 80,000 tokens。对上述新版真实记录，正常 Extractor＋Builder 按完整输入字节上界和输出 cap 的保守预留合计 57,113，能容纳正常两阶段。该额度不保证所有截断、结构修复和 Builder 修复分支均能完成；发生真正额度不足时按实记为预算延后，不加 repair、不伪造成功。当前实际 `no_change` 只耗 4,261，不能归因于 token 不够。

请求材料的 16,384 B 上限与 token pool 独立。原始大轨迹完整归档，只向模型投影真实必要材料；必需来源材料仍不能容纳时明确拒绝，不删掉出处或扩大为未经批准的调用。

## 尚未实证的部分

新版临时检索入口和新版学习提示尚未接受自然模型的新求解观察。原 6＋2、`cb2451b` 诊断和本次新入口是不同源码身份；不热修改旧 run、不称同版本 resume，不继承诊断 Bank，不自动重开正式实验。

原始结果和诊断分别位于：

```
/home/yangchengyu/officeqa_bootstrap_r3_paid_check_20261012
/home/yangchengyu/officeqa_bootstrap_r3_learning_diagnostic_20261012
```

交付审查包保留配置、源码身份、题单、逐题 traces/requests、Bank/Frozen、账本和本轮离线证据，排除语料全集、项目全集与密钥。源文件路径中冒号仅在 ZIP 条目名中替换，字节和 SHA256 保持不变。
