# MiniMax R2 拒答与有限重试续跑

用户已要求补齐修复后续跑，并增加自动重试。原完整Test完成228题、答对199题；第229题返回HTTP422、`input new_sensitive (1026)`，没有usage。本次在独立子run中续跑，原目录不改。第229题直接使用已保存拒答，按官方空答记0分、保留在1400题分母内，不再次请求或改变题目。旧228题的结果不重跑。

MiniMax 1026/1027为内容拒答，参照[官方错误码说明](https://platform.minimax.io/docs/api-reference/errorcode)。同时兼容HTTP错误封套和原生流式base_resp。其他422、认证错误、余额不足和未知错误不伪装为拒答，不通过换词或改提示规避审核。

临时HTTP408/429/500/502/503/504/529、明确的MiniMax临时错误码、连接中断及超时，最多追加3次请求；退避10/30/60秒。每次物理请求和未知费用预留均完整保存，重启不重置重试次数。相同messages/tools/model保持不变；仅当原单题128K余额不足时，下调重试输出cap以适配余额，实际cap逐次记录。正常请求仍为65536。余额不足或重试耗尽时停止，不改成模型答错，也不无限付费。

Docker可用性故障保留检查点和已付费回答，控制器最多恢复3次，每次等待最多3分钟；只检查服务恢复，不重置Docker或WSL。认证、未知计费、未知错误或持续基础设施故障仍需停下处理，不能承诺任何服务故障下都无条件跑完。

累计13,821,713 tokens／3795 HTTP上限不提高；旧未知预留和本次78,672 tokens拒答预留均保留，不填造成0用量。最终汇总区分实测tokens与未知预留，不因存在合法拒答而在收尾断言失败。

原生源码仍是28ac1ca31cc5f7f1b3d5772fedd3f84c489cebf9；变更仅在外层拒答/重试和续跑控制文件。R2 Bank、公开题目、官方评分及正常模型参数保持一致。`continuation_manifest.json`和plan记录新旧执行身份与父子关系，前后Frozen hash一致。

新运行目录：`/home/yangchengyu/searchqa_minimax_r2_test1400_resume_20261024`。

WSL监控：

```bash
watch -n 20 /home/yangchengyu/asg_alfworld_venv/bin/python -B /mnt/d/T3S_exp/deliveries/searchqa_minimax_r2_resilience_20261024/continue_test.py status
```

暂停：

```bash
touch /home/yangchengyu/searchqa_minimax_r2_test1400_resume_20261024/STOP_AFTER_TASK
```

零真实HTTP验收包括真实拒答记录重放、流式输出拒答、临时过载和超时恢复、未知422及认证错误停止、Docker中断后缓存恢复、重试次数/预算上限、重复启动不重复请求。详细收据见`offline_acceptance.json`。
