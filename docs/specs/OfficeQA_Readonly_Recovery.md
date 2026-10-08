# OfficeQA 只读检索超时恢复

2026-10-09 的 seed43/44 训练停止发生在已评分原题的 Program 试用。
697 个语料文件位于 WSL 的 `/mnt/d`；逐文件完整解析同一父目录产生重复 I/O，
多次全库 grep 耗尽原定 60 秒 Worker 时限。

`select_paths` 在单次选择内复用父目录解析，仍解析文件符号链接并检查最终路径、
文件类型和授权范围；不缓存跨请求语料内容，不改变排序、Unicode offset、40 hit 上限或正则语义。
恢复运行使用逐文件 SHA256 相同的 WSL 本地语料副本，物理路径变化进入实际配置。

仅在显式恢复子 run 的逻辑试用 checkpoint 中放置 `new_execution_authorization`。
已有记录正常返回；缺结果试用的授权在启动新 execution 时消费一次。
普通 resume 仍拒绝未知执行；再次中断不会自动追加 execution。
新的 execution 独立记录 ID 和原生交互，旧 exception、请求及费用保留。

Learner checkpoint 保留请求时的动态 schema，与原材料一起核验缓存响应语义。
旧 checkpoint 的 schema 只能从对应实际请求记录恢复并匹配原 semantics hash。
这样应用响应后新增的 Skill 不会改变该响应原本的 ID 枚举。
当前结果清除已解决的 learning_error，原异常留在旧 run 和 continuation 记录。

恢复准备不执行 Worker、模型或评分器，不改 Bank、不提交 pending task。
只有生产续跑执行已授权的新试用；原题求解和既有 Builder 响应复用。
保留晋升标准、repair 次数、题单、评分器和执行预算。
