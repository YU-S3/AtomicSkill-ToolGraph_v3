# CF4-R1 工程交付

本轮按 CF4-R1 实施三个接口修复，并提供四个非 ALFWorld 正式组合的启动器。没有启动收费模型、额外 pilot 或正式实验，没有修改历史结果、旧 Bank、公共划分、模型锁或科学预算。

文本收尾使用一次无工具的 Runtime 请求，原样交给原评分器；空文本和意外工具调用结束为明确拒绝，length 有文本则保留。purpose、响应身份、逻辑决策和最终答案一起持久化；收到响应及提交答案后的冷启动恢复均不会补请求。

Builder 当前只能提交一个 submit_program；未来 Program 的授权名单、工具定义、当前形状示例与 ABI 来自同一 helper。首次、length、结构、语法与试用失败恢复使用相同材料分区和原有恢复额度。Broker 保留实际拒绝、日志和预算守卫，Runtime 的 execute_python 保留，Program RPC 不展示或允许递归调用它。持久 run 的输入/工作区与 solution.py 重执行的 INPUT_PATH/OUTPUT_PATH 契约分开，越界发布仍拒绝。

single_answer 保存完成经验后只调用专用 guidance Extractor，不排队 Program、Workflow 或旧 pending job。合法 guidance 以内容 ID 写入普通 Skill，修订保留父子版本，重复提交幂等；查询先过滤再取前 8/3 条，由原来的同一次 solver 消费。父子 artifact、检索/注入 ID 和拒绝原因进入旁路日志；中断的未完成学习不会提前写成 completed。

最终受测提交为 cdeae026593235b5d799de8840e42861840387ac，源码 SHA256 为 3ab48c8b3699775f6fccf351e159725d79b9c43c2d9bc2a3f52709a061b13391。之后仅追加核验报告；最终 main 身份在交付启动脚本中固定，WSL 使用独立干净 checkout。

252 项完整回归通过，0 失败、0 跳过，耗时 271.95 秒。原有 214 项保留，仅按新契约更新受影响断言；新增 38 项生产路径回归。R01–R22 对应实际测试节点、JUnit 和证据哈希见 [机器核验记录](cf4_r1_verification.json)。用户包内 22 项参考示例不计入生产回归数量。

验证使用拦截的真实 Provider HTTP 构造和锁定 Docker，包括录制的 Office Builder 错误响应、文件生产/封存、冷启动响应恢复、guidance 版本写入与 Frozen 消费、四个隔离伪 campaign 并行及单项失败、同版本暂停/恢复 Test。既有自然 ALFWorld Program 仅用于执行层只读回归，原 Frozen 哈希不变。新版本的自然学习、成本和质量尚未测量：implementation_ready=true，natural_learning_observed=false，cost_quality_measured=false。

源码模块、根目录入口与隔离安装 wheel 的单 cell CLI 均验证 --help、--stop-after-val 和 --resume；安装模块 revision 为 empirical-v3.1-CF4-R1。公共包和 model view 保持 CF4 版本。

公共 authority SHA256：b85e0c2e442ca6b97ef98ece895719d4231f18db29682b0fcb6fdf3f91c37c95。材料 SHA256：2e7fa5aef40bffcb96c1c1bc3be3b129a090d9c3652abbb47664c010d4cc1c1d，934 个外部输入逐项校验。Office 语料 SHA256：4883f8d199215ab95a904f0769e2e9c7dd636444d72498ca247d6feba561daeb。公共包 SHA256：e63911fe52c16518a7a1e0dd5647c1c53e52020e6275a198461c0706ebbb20c7。Docker digest：sha256:4db520e20cd121f830731c2d0c0bcfbe9cacd254d1817262404cc18e1e757783。

本轮启动计划为 DeepSeek seed42，四个独立空 Bank，各自完整 Train → Frozen → Val → awaiting_test：SearchQA 300/24、LiveMath 60/17、OfficeQA 120/24、SpreadsheetBench 200/20。ALFWorld 为 excluded_by_user_cost，文本模型 DocVQA 为 unsupported。总规范保留六基准及 seed42/43/44；不使用旧 matrix，不继承 pilot，不自动跑 Test。恢复 Test 时使用同源码/配置/材料/输出的单 cell 命令，移除 --stop-after-val，添加 --resume。

生产薄启动器为 [run_non_alfworld_formal.sh](../scripts/run_non_alfworld_formal.sh)：默认 dry-run；显式 --execute 启动四个进程，各有独立 Bank、checkpoint、TMPDIR、锁和日志。拒绝错误提交、脏源码、未跟踪执行代码、同 checkout 已运行的 seed42 cell 和既有输出根；失败不自动重试或重启其他组合。交付的环境包装脚本通过 systemd 用户服务运行监督进程，关闭终端后仍持续运行；重启/停止 WSL 会中断运行，之后按原 checkpoint 规则显式恢复。

原始核验材料位于 /home/yangchengyu/cf4_r1_review_20261007：pytest_final.log/xml、最终 HTTP 夹具、R1 生产请求/决策/父子版本证据、只读环境审计和 wheel/安装入口记录。初次回归中的旧契约断言及证据夹具问题单独留在 pytest_pre_fix.log/xml；最终成绩仅取最终完整套件。
