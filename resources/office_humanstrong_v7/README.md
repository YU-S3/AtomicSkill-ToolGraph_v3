# OfficeQA HumanStrong V7

版本 `office.humanstrong.programmatic.v7.20261021`。V6 的独立修订，仍为 14 Atomic、9 Program、9 Implementation、4 存档 Composite。保留原表格、数值、格式及权限检查；不修改自然学习算法、题面、答案、预算或旧运行。

## 实际依据与修改

V6 Val24 为 Bank 11/24、NoSkill 13/24，Bank tokens 多 24.82%。本轮定位和回归使用原运行 `officeqa_humanstrong_v6_val24_pairs_seed42_20261021`；这套 Val 已用于开发，后续成绩不能称为未见验证。

| 原始记录暴露的问题 | V7 修改 |
| --- | --- |
| UID0024、UID0233 的检索 Python 返回行号、偏移量、`[行号, 文本]`，却被列为计算结果，触发提交提示 | 从候选数字中排除检索元数据及混合文本行；运行时与收尾使用同一候选引用逻辑。仍不根据数值推断整题完成 |
| UID0047 读取 `schema_ref` 得到 `Dictionary roots are unavailable` | Schema 保存为 JSON 文本，通过原生 `read_result` 分页读取；不开放字典根或 Host 元数据 |
| UID0047 将行名填入 `bindings`，读取源页之后才发现 key 不合法 | Schema 明确引用 `tables[].key`；读取前校验，拒绝收据给出未知、未使用及合法 key，不自动猜测映射 |
| UID0213 的 `data.text['content']` 类型错误后转而扩大检索 | 公共工具说明直接说明文本为字符串，两臂一致；Bank 恢复提示保留 Python 错误，要求优先修正具体访问方式，不把类型错误当缺失证据 |
| 常用但未调用的大 Schema 每轮重复注入；已有单元格仍被强行包成求和调用 | 目录保留所有真实 ID 与能力，使用时读取完整合同；指导明确支持直接查表后提交，不强制调用 Program |

UID0024、UID0233 所需数值均出现在原始最终请求中，不能归因为输入截断。当前 NoSkill 不加载 Bank，但保留公共检索、临时 Python、证据保留和收尾能力；不是外部旧 NoSkill 的复现。

## 验证与边界

测试使用真实保存记录、生产模块和 Docker，模型响应由夹具提供，真实模型 HTTP 为零。所有 9 个 Schema 可分页无损读回；校验合同读取成功后再执行 Program。原错误 binding 零读取拒绝，修正 binding 后得到原表 29265；仅修正 UID0213 的文本访问方式即可运行原计算并得到 -550.3。UID0144 原成功计算仍保留为候选结果。

这些是接口和执行回归，不是模型自然成功率。未把修改后的参数回填旧轨迹，未自动重跑 Val/Test；V7 的 `measured_performance` 保持 null。实际提升需在冻结此版本后做同设置观察，不能用离线修正结果宣称恢复了几道题。

本轮 V5/V6/V7 共 48 项检查通过。UID0024 同输入首轮完整请求：V6 36,026 B，V7 32,000 B，减少 11.18%；这是请求体字节量，不是实测 token 节省率。原始拦截请求和修复回放收据位于交付目录 `deliveries/outputs/officeqa_humanstrong_v7_20261021/`（工作区外的审查产物，不纳入 Bank）。

```bash
PYTHONPATH=src python resources/office_humanstrong_v7/tools/build_bundle.py
PYTHONPATH=src python -m pytest tests/test_office_humanstrong_v7.py -q
```

`integration/configuration.py`、`frontier.py`、`evaluate.py` 为 V7 实际入口。评估仍要求独立目录、冻结 Bank、外部对齐资源和已有 campaign budget governor；不复用旧版本启动器冒充续跑。
