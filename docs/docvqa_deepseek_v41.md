# DocVQA / DeepSeek V4.1（2026-10-18）

官方 [模型说明](https://api-docs.deepseek.com/) 和 [视觉接口](https://api-docs.deepseek.com/guides/vision/)
确认 `deepseek-flash` 当前由 DeepSeek-V4.1-Flash-1017 提供，支持 user 消息中的原图。
服务名是可变别名；运行记录同时保存请求名、响应 `model`、实际 payload 和 raw usage，不能把显示名称当成服务端不可变快照。

## 固定实验契约

- 只读取 `data/main_experiment_v1`：Train180 / Val22 / Test201 / Reserve131；不重新切分、不补题。Reserve 不进入运行。
- split_seed=42；run_seed=42/43/44 只改变 Train 顺序，成员完全相同；Val/Test 保持 canonical 顺序。
- 每个 seed 从独立空 Bank 开始，Train 完成后 Frozen，Val/Test 只读同一 Frozen。诊断 Bank 不继承。
- 公共问题和原始图片直接发送，`detail=high`。没有 OCR 替代、改用文本模型、人工注入资产或强制 Program。
- 使用现有 AnswerExecutor、自然 guidance 学习和局部计算入口。原晋升条件、一次结构修复、Docker 隔离和评分不变。
- ANLS 使用已冻结 scorer；hard 是 ANLS >= 0.999，soft 是原始 ANLS。记录原始答案与评分，识别错误不在适配器中修正。
- Runtime thinking=enabled、effort=high、输出上限32768；guidance_learning thinking=disabled、输出4096；结构修复 disabled、输出2048。
- Train 每题200,000 tokens（solve120,000 / learning80,000），Val/Test 每题96,000。
  每 seed 总封顶57,408,000 tokens /21,735 HTTP（既有循环及4次重试的保守理论上限），不跨 seed 借用。
  这是封顶，不是预期消耗。真实费用按返回 raw usage 记账；generation seed 不支持，记录 null。

图片传输与文本材料分开计量：完整请求<=48 MiB、单图<=32 MiB，原图检查格式/维度/哈希；
文本材料仍沿用既有限额，token 准入采用文本字节保守上界加官方每图1024 tokens上界。
实际 HTTP 保留完整原图，日志同时记录完整请求字节、文本材料字节和图像审计。
默认文本模型配置保持原有行为，新增配置只允许 DocVQA 单 cell 入口。

## 实测验收

付费受测提交 `2cca7584fa5671409eac6afecfea571ae3eb5900`，固定 seed42 Train 前两题
`docvqa:64547`、`docvqa:43719`，canonical Val 第一题 `docvqa:3116`。
同一生产 solver/learner/Bank/Frozen/readonly Val，独立诊断额度496,000 tokens /175 HTTP。

| 阶段 | 完全正确 | 平均 ANLS | 实际 tokens | HTTP |
|---|---:|---:|---:|---:|
| Train2 | 2/2 | 1.0 | 12,162 | 6 |
| Val1 | 0/1 | 0.8 | 3,439 | 1 |

两题自然生成2条 guidance，均在原一次结构修复后合法入库；第2题注入第1条，Val 注入冻结的2条。
本次没有生成或调用 Program，不据此声称 Program 降本成立。
Val 将手写 `63744` 读成 `C3744`，原图字节已核验、答案提取与 ANLS 正常；保持原始失败结果。
本验收证明链路可运行，不估计正式准确率或声称 guidance 已带来因果增益。

7次 HTTP 全部有 raw usage，无未知账单；无接口异常、材料误拒绝。Frozen 前后哈希
`e2caea5dc1675617160bc01c864a1612bcdb32a6e6cf447836849cb0a3560e51` 一致。
完成后对同一诊断执行 resume，仍为7次 HTTP /15,601 tokens，没有重复请求。
534张原图全部通过传输检查；最大4,288,412 B、最大边7184像素。Docker 已实跑。

## 正式运行

在 WSL 的已固定 clean checkout 中运行；设置 `EXPECTED_SHA` 为交付提交号。入口会核验源码、数据与图片、Docker，
以文件锁防止同目录同 seed 重复启动。使用 `--run` 是 fresh，已有目录必须显式 `--resume`，且源码/配置身份必须一致。

```bash
cd /home/yangchengyu/asg_docvqa_v41_release_20261018
export EXPECTED_SHA=<交付的完整提交号>
export OUTPUT_ROOT=/home/yangchengyu/runs/docvqa_v41_formal_20261018
mkdir -p "$OUTPUT_ROOT"
nohup bash scripts/run_docvqa_v41.sh --run 42 > "$OUTPUT_ROOT/seed42.log" 2>&1 < /dev/null &
```

一次命令连续运行 Train180 → Frozen → Val22 → Test201。43、44仅把末尾seed和日志名改为对应值；
各自的 Bank、预算、目录独立。正式结果写入 `$OUTPUT_ROOT/seed42/{train,val,test}`。
不用诊断目录作为 OUTPUT_ROOT，不从诊断 Bank 续训。

```bash
tail -f "$OUTPUT_ROOT/seed42.log"
cat "$OUTPUT_ROOT/seed42/completion.json"
# 主动暂停：在当前正在运行的 phase 目录创建 STOP_AFTER_TASK，当前题完成后退出。
# 仅恢复同版本：先确认停止原因、移除自己创建的 STOP_AFTER_TASK，再用 --resume 42。
```

完整请求、响应/raw_usage、trace、SQLite、task顺序、学习记录、source/config/split身份与Frozen哈希沿用正式 logger。
监控时不要把暂未生成的 summary.json 当作失败；逐题完成日志和 SQLite 可反映当前进度。
