# 研究来源与本次编辑

本 Bank 以可直接获取的公开优化 Skill 为基础，不是只从论文标题臆测方法。下表是本次实际阅读的文件和 blob 身份；策略已经重写/拆分，所以其原论文成绩**不能**作为本 Bank 的成绩。

| 来源 | 实际文件 | blob SHA |
|---|---|---|
| EvolvingLMMs-Lab/SkillOpt-Lite | [skillopt_lite_ckpt/officeqa/gpt5.5/skill.md](https://github.com/EvolvingLMMs-Lab/SkillOpt-Lite/blob/main/skillopt_lite_ckpt/officeqa/gpt5.5/skill.md) | `b320a2dae089c7186e69b753a951a3beba116fb8` |
| microsoft/SkillOpt | [ckpt/officeqa/gpt5.5_skill.md](https://github.com/microsoft/SkillOpt/blob/main/ckpt/officeqa/gpt5.5_skill.md) | `869c1d4865ce75713825ab7b53fa3a4410371bf3` |

## 保留与修订

以全语料检索接口为准，删除 oracle-page 假设、固定外部路径和网络下载要求。保留有效查询、完整期间、表格行列、单位、交易语义、公式及最终格式。修正两点 Gini 的总体/样本修正口径混用；不因一个检查点用词而强制错误公式。新增可直接调用的检索、表格提取、统计程序。

## 如何解释“强”

SkillOpt、SkillOpt-Lite 的公开页面和检查点提供了这些策略的实验背景，但目标模型、划分、工具和输入条件不同。SkillOpt-Lite 项目页报告 GPT-5.5 的 LiveMath 为73.6、OfficeQA为76.2；这不是本库在你的 DeepSeek 或 full-corpus OfficeQA 中的成绩。我们也查阅了 SkillBoost、WikiSkill 和 GraphSkillEvo；未获得的最终检查点没有声称已装入此 Bank。

研究链接：[SkillOpt](https://arxiv.org/html/2605.23904v2)，[SkillOpt-Lite](https://evolvinglmms-lab.github.io/SkillOpt-Lite/)，[SkillBoost](https://github.com/HQ-Lin/SkillBoost)，[WikiSkill](https://arxiv.org/html/2608.27454)，[GraphSkillEvo](https://arxiv.org/html/2609.21749v1)。

学习晋升门槛已从本人工运行线路中移除；数学真假、来源文本和统计口径不能因为移除准入就凭空正确。本次没有采用按测试题号/答案字母的查表，也没有读对应论文的隐藏 theorem/proof sketch 后伪装成标准输入。
