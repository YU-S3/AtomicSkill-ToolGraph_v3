# 研究来源与本次编辑

本 Bank 以可直接获取的公开优化 Skill 为基础，不是只从论文标题臆测方法。下表是本次实际阅读的文件和 blob 身份；策略已经重写/拆分，所以其原论文成绩**不能**作为本 Bank 的成绩。

| 来源 | 实际文件 | blob SHA |
|---|---|---|
| EvolvingLMMs-Lab/SkillOpt-Lite | [skillopt_lite_ckpt/livemath/gpt5.5/skill.md](https://github.com/EvolvingLMMs-Lab/SkillOpt-Lite/blob/main/skillopt_lite_ckpt/livemath/gpt5.5/skill.md) | `6ddea929aa6068501f5f4e7b07dc881b6206bc78` |
| microsoft/SkillOpt | [ckpt/livemath/gpt5.5_skill.md](https://github.com/microsoft/SkillOpt/blob/main/ckpt/livemath/gpt5.5_skill.md) | `2d1ac842841f771f4867aea5bfb2fc257fa806d3` |

## 保留与修订

保留定理精度、最强可证结论、条件/量词/等价方向、端点与附加子句检查。保留元选项是重要候选的公开经验，但去掉“出现就无条件选择”“不知道是否最优就证明还有更强结果”的错误逻辑和最后字母偏置。默认文字策略直接解题；精确计算与有限反例程序只做真正的局部运算，不把未知命题伪造为布尔矩阵。

## 如何解释“强”

SkillOpt、SkillOpt-Lite 的公开页面和检查点提供了这些策略的实验背景，但目标模型、划分、工具和输入条件不同。SkillOpt-Lite 项目页报告 GPT-5.5 的 LiveMath 为73.6、OfficeQA为76.2；这不是本库在你的 DeepSeek 或 full-corpus OfficeQA 中的成绩。我们也查阅了 SkillBoost、WikiSkill 和 GraphSkillEvo；未获得的最终检查点没有声称已装入此 Bank。

研究链接：[SkillOpt](https://arxiv.org/html/2605.23904v2)，[SkillOpt-Lite](https://evolvinglmms-lab.github.io/SkillOpt-Lite/)，[SkillBoost](https://github.com/HQ-Lin/SkillBoost)，[WikiSkill](https://arxiv.org/html/2608.27454)，[GraphSkillEvo](https://arxiv.org/html/2609.21749v1)。

学习晋升门槛已从本人工运行线路中移除；数学真假、来源文本和统计口径不能因为移除准入就凭空正确。本次没有采用按测试题号/答案字母的查表，也没有读对应论文的隐藏 theorem/proof sketch 后伪装成标准输入。
