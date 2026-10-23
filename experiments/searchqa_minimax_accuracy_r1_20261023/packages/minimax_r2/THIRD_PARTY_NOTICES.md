# 第三方来源与许可说明

本包结合用户已有研究资产与公开论文，重新编写了原生指导、纯字符串程序和接入代码。没有将任何公开论文的成绩声称为本包实测成绩。

- Microsoft SkillOpt 的公开 SearchQA guidance 是实质性研究参考，部分通用策略作了改写。其 MIT copyright/permission notice 完整保存在 `licenses/SkillOpt_MIT.txt`。原仓库与精确commit、blob见 `RESEARCH_AND_DECISIONS.md`。
- SkillOpt-Lite 的 MIT 成品用于比较通用策略及发现冲突，没有完整复制任何checkpoint。原文仍在作者仓库；来源链接与身份在研究说明中。
- GraphSkillEvo 仅用于阅读和概括步骤组织思想，本包没有复制或再分发其完整checkpoint。
- 用户原生仓库仍是运行依赖；本包不再分发其完整源码、评测题、gold、凭据或模型权重。

新增 helper 不是任何论文的神经重排器复现。接入代码承接用户此前作者可信 Bank 的原生导入与执行机制；新增协议适配的准确率影响尚待本次实跑。
