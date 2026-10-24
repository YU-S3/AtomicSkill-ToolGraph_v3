# 同一高准确率 Bank：DeepSeek / Kimi 各一轮 Test1400

用户授权：“同样的高准确率bank启动一轮deepseek、一轮kimi的test”。两组各运行一次独立、完整的canonical SearchQA Test1400，seed42、原始顺序。直接复制 Gemini 得到1246/1400（89.00%）的实际输入包及Frozen，不训练、不重新筛选Bank、不继承旧任务预测，也不自动追加实验。

- 输入包 SHA256：`71c1b8258e54eae6bbbc72f903fc5d3b90ff5591fc85b05b4297ed4f78b407fb`
- Frozen SHA256：`e5a4cd9d9fd486e820c3c64944b31ae130f640bac6f3c42f73c182d6b126eeff`
- Test题单 SHA256：`bd4e9bda35f94fb40aa469e0bc4cdd3d496703bbc58dee419cef5214a70e748c`
- 冻结运行源码：`28ac1ca31cc5f7f1b3d5772fedd3f84c489cebf9`

| 配置 | DeepSeek | Kimi |
|---|---|---|
| 实际模型 | deepseek-v4-flash | kimi-k2.6 |
| Thinking | enabled / high | enabled；无effort参数 |
| 输出上限 | 32768 | 65536 |
| 单题上限 | 96000 | 128000 |
| 流式请求 | 沿用原非流式配置 | 沿用原流式配置 |
| 本轮新增额度 | 1000万tokens / 2000HTTP | 1000万tokens / 2000HTTP |

沿用各模型此前实际运行配置的正常生成参数。DeepSeek原Provider内部重试4次改为0，统一由已经验证的外层持久化重试处理（初次+最多3次重试），避免两层重试叠加。Kimi原内部重试已为0。本轮不同时改变模型生成参数来追求结果。

真实请求中DeepSeek发送模型名为`deepseek-v4-flash`，服务端响应model为`deepseek-flash`，与此前V7重复实验记录一致；两者分别原样保存。Kimi请求及响应model均为`kimi-k2.6`。

旧请求、真实费用和未知费用预留完整结转；本轮新增请求独立核算。DeepSeek继续保留历史并行V8额度及R3预留，并检查总包络仍在原SearchQA累计权限内。每组独立账本和上限，不互相借用、不自动扩额。

正常答题与已确认传输失败的单题预算分开，但失败费用始终完整计入累计总账。重试输入及输出cap不变，退避30/90/180秒，次数跨resume保留；拒答按原官方空答评分，不重试也不从分母删除。保留Docker可证明未执行时的恢复机制。

`offline_acceptance.json`记录9项零真实HTTP检查（20个拦截请求）：两种实际Provider序列化、原生图及真实Docker提交、连续504恢复、有限重试耗尽并resume不重置、拒答评分和Kimi断流恢复。两个新Frozen与Gemini原Frozen逐字节一致。`live_acceptance.json`记录真实启动后的请求及提交核验。

运行根目录：`/home/yangchengyu/searchqa_gepa_deepseek_kimi_test1400_20261024`

WSL汇总监控：

```bash
/home/yangchengyu/asg_alfworld_venv/bin/python -B /mnt/d/T3S_exp/deliveries/searchqa_gepa_deepseek_kimi_test1400_20261024/run_models.py status
```

分别查看日志：

```bash
tail -f /home/yangchengyu/searchqa_gepa_deepseek_kimi_test1400_20261024/deepseek/run.log
tail -f /home/yangchengyu/searchqa_gepa_deepseek_kimi_test1400_20261024/kimi/run.log
```

如需在任务边界暂停某组，在对应模型目录创建`STOP_AFTER_TASK`。不要编辑运行中的冻结文件或重复启动。

完成后各模型写入`completion_report.json`，包含EM/F1、原生图/Program调用、已计量tokens、真实请求数、未知预留及Frozen核验。口径为人工Bank经Test反馈迭代后的跨模型复测。
