# Release4 v2.0 实施与运行

基线：`45d59d14597e6417dad0a810525d4b9601cf796d`。仅 main；不改 Release3 和原始训练 bank。

## 本轮实现

- 请求内不可变 Support surface：native ID、公开卡片和解码同源；完整方案先证明再取前三个不同 Atomic。Node 空映射与无方案不同；Task 无父转交。原始提交不改写，解析后再查原状态缓存，真实执行仍经原 prove/resolver/transaction/transfer。
- 在真实参数 resolver 产生结构化拒绝说明；公开反馈与缓存保留具体角色、分辨率和当前证据要求。
- 可配置的 `alfworld.public-discovery.v1` 只解释本机公开平面 On/In 列表和 TAKE 目录。联合关系在公共投影与 validator 同源，保持 evidence domain；revision、checkpoint、replay、拒绝和终局边界有回归。
- 新程序沿用原有界范围、顺序和开容器授权，以同一 `entity.discovered_at` 同时匹配 query 与实际检查位置；没有新增 opcode、自动搜索策略或 benchmark 分支。累计搜索历史区分完整检查、未完成和回滚。
- 实际 base lock + 增量拓扑编译；保留原资产及逐 ref 发布许可，版本化三库发现能力与各自 look 图。保存 resolved refs、具体差异、身份匹配和全库准备依赖审查；Agent 选择不记为程序覆盖。
- 三库各自的真实无模型专项及固定 current × 六族，共十八个自然 episode。后者全部完成且审计通过才生成五份正式配置。正式脚本有 seed 锁，跳过已完成轮，未完成轮须显式 `--resume`。

## 验证与产物

新增回归：`tests/test_release4_interfaces.py`、`tests/test_release4_accounting_and_plan.py`。
原有效测试保留；没有删除失败反例或调整预算。

完整运行时，最终结论以新发布目录下这些文件为准，不能把 build01/build02 调试记录当成同版本十八题结果：

- `dev/coverage_report.md`、`dev/coverage_report.json`：自然结果、全 attempt 成本与逐请求关联。
- `dev/coverage_acceptance.json`、`dev/controlled_acceptance.json`：放行审计。
- `dev/seed{42,43,44}/current/<task>/`：全部原始 Trace、payload、用量、配置和台账。
- `dev/seed{42,43,44}/automation/`：真实无模型专项，单列、不加入成功率分母。
- `seed*/frozen/preparation_coverage.json`、`derived_revision_comparisons.json`、`resolved_ref_map.json`、`public_discovery_contract.json`：库内准备边界、变化和来源。

不承诺十八题全成功，不以 40K 为截断门槛；不把静态程序覆盖等同于实际选择或纯 token 压缩收益。

## 启动

开发验证（自动生成正式配置，但不启动 Test134）：

```bash
cd /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3
bash scripts/validate_release4.sh /home/yangchengyu/asg_oldfirst_20260923_release4
```

开发门禁通过后，三个流并行；42 内部三轮顺序，43/44 各一轮：

```bash
cd /mnt/d/T3S_exp/AtomicSkill-ToolGraph_v3
bash scripts/run_released_frozen_parallel.sh /home/yangchengyu/asg_oldfirst_20260923_release4
```

可先将 `42-first` 作为第二个参数，只运行 42 rep01。之后再次使用无参数矩阵启动，已完成轮会跳过。某 seed 中断时用 `42 --resume`（或 43/44）；原代码、配置、库、任务和资源一致性校验仍执行，不自动豁免未知 API 费用。

```bash
tail -F /home/yangchengyu/asg_oldfirst_20260923_release4/seed{42,43,44}.log
```

全部五轮后汇总：

```bash
/home/yangchengyu/asg_alfworld_venv/bin/python -m experiments.run_v3_released_frozen aggregate \
  --plan /home/yangchengyu/asg_oldfirst_20260923_release4/evaluation_plan.json \
  --output /home/yangchengyu/asg_oldfirst_20260923_release4/report
```
