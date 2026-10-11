# OfficeQA P0：非绑定结构修复材料

本补丁只改变无 case_bindings 且无 Host 来源错误的操作目录，保留真实 operation_name/local_evidence_ref。绑定、输出引用和 publication 错误继续发送完整操作来源，Host/Worker 准入、唯一 repair、8,192 B 上限均保持。不可删材料仍超限时保存拟发 payload、字节构成、必需字段和拒绝原因，不发 HTTP。

版本：empirical-v3.2-office-r2；learner.material.public-directory.v5。旧配置/响应不能冒名 resume。启动器默认费用来源改为最后 v4 账本，新输出为独立 v5 空 Bank。

原 UID0115 的生产函数及真实 Provider 构包回放：8,662 → 7,490 B；proposal/errors/tools/模型参数不变。原 missing args 仍拒绝；独立模拟模型的完整 args 提交通过 Host，生产没有自动填字段。v4 保存的两份 learning_structure_repair 均重新构包通过。必要材料超限负例保留 local_refusal，HTTP=0，无虚构 response/usage。

WSL 零模型回归：68 passed / 7 skipped。覆盖 Office 4 个旧预算失败、旧正式三题来源重放、真实 Docker Worker 资格正反例、预算/持久化/原生协议/快照；7 个依赖另一历史有限包的夹具未配置，未计为通过。证据保存在 D:/T3S_exp/deliveries/outputs/ours_p0_p1_20261012/p0_tests.log、p0_tests.xml、p0_receipts/。所有 HTTP 夹具均拦截 requests.post，无新模型请求。

旧 v4 六题及评分、Bank、费用原样保留。新的 Office 正式 run 需在本补丁的 clean pinned commit 启动，费用从最新 v4 Governor 唯一继承，不继承 Bank。先完成同一 Train 的固定六题工程 gate，再同版本同目录继续 Train120 → Frozen → Val24 → Test102；不以得分或 Program 数量判断放行。该启动的实际提交和费用收据另存外部交付目录。
