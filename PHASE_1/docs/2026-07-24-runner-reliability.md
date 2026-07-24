# Phase 1 运行可靠性与实验反馈周期改造记录

本次改造完成于 **2026 年 7 月 24 日**，开发分支为
`codex/phase1-runner-reliability`。它延续上一轮本地评测链路建设，但不修改 ReAct
Prompt、Agent 分析策略、工具协议或评分规则，目标是先解决少数异常任务拖慢整轮实验、
超时后无法定位原因，以及中断后必须重新运行全部任务的问题。

## 改造动机

上一轮 50 题 baseline 的墙钟时间约为 32 分钟。31 个成功任务的累计耗时约 898 秒，
16 个跑满 `max_steps` 的任务约 602 秒，而 3 个任务各自精确运行到 600 秒后被终止，
累计消耗约 1800 秒。它们占全部任务累计耗时约 55%，在两个并发 worker 下贡献了约
20 分钟等待时间。更严重的是，这三个超时任务的 `trace.json` 都只有统一的 timeout
信息和空步骤，无法区分模型请求无响应、工具执行过慢或子进程结果传输阻塞。

问题来自运行边界的叠加：OpenAI Python Client 没有显式请求预算，默认读取超时与任务
硬超时同为 600 秒；Client 在每次 ReAct step 中重新创建，无法复用连接；完整运行结果
通过 `multiprocessing.Queue` 返回，但父进程先等待子进程退出、之后才读取 Queue，大
Trace 存在互相等待风险；所有 Trace 又只在任务结束后一次性写入。因此，一次异常请求
既可能耗尽整个任务预算，也会抹去之前已经完成的诊断信息。

## 实现后的运行链路

模型 Adapter 现在在初始化时创建并复用 Client，关闭 SDK 隐式重试，改为应用层显式
控制。单次模型请求默认最多等待 20 秒，只对连接错误、超时、408、409、429 和 5xx
重试一次；退避从 1 秒开始，加入随机抖动，并把服务端 `Retry-After` 限制在最多 5 秒。
认证失败、无效请求和其他确定性 4xx 不会重试。每次尝试的 step、attempt、耗时、异常
类型和重试决定都会进入事件记录。

任务默认硬超时缩短为 120 秒。无论 `max_workers` 是 1 还是更高，正式运行都通过可
终止子进程执行。子进程不再把完整 Trace 放入 Queue，而是在任务目录原子写入结果文件；
父进程只负责监控生命周期并读取文件，从而消除未读取大 Queue 阻塞子进程退出的条件。
`events.jsonl` 在模型请求、工具调用和每个 ReAct step 完成后立即刷新。任务被强制终止
时，Runner 会从事件文件恢复已完成步骤，并把最后事件、step 和 attempt 写入 timeout
诊断，而不是生成无法解释的空 Trace。

批量运行新增任务清单、恢复和失败重跑能力。每个新 run 会先写入不含密钥的
`manifest.json` 固定任务集合；`--resume` 跳过已经完成的任务并补跑缺失任务，
`--retry-failed` 则归档已有失败产物后重新执行。固定的
`configs/regression_tasks.example.txt` 包含 11 个任务，覆盖全部难度、不同评分状态、
协议循环和上一轮三个 600 秒超时任务，作为日常开发的快速反馈集。

## 验证与脱敏结果

自动化测试覆盖首次超时后成功、重试耗尽、不可重试 4xx、429 受限退避、大于传统 Queue
缓冲区的结果传输、子进程硬超时、部分步骤恢复、成功任务跳过和失败任务归档重跑。原有
评测测试与新增测试共 41 项，全部通过；Ruff 静态检查、格式检查和补丁空白检查也保持
通过。

真实 11 题快速回归使用与上一轮相同的百炼模型，设置 `max_steps=10`、
`max_workers=2`、20 秒请求超时和 120 秒任务超时。整轮耗时 **128.967 秒**，低于
5 分钟目标。上一轮的三个 600 秒任务分别在 18.174、28.712 和 21.558 秒结束，并保留
10 个完整步骤；本轮没有模型请求超时或重试，最慢单次模型请求为 10.521 秒。随后对同一
run 执行恢复只用 0.007 秒便跳过全部 11 个完成任务。

快速回归中 4 个任务提交了答案，7 个任务在 10 步内未提交。这个结果只验证运行可靠性和
反馈速度，不能与上一轮使用 `max_steps=16` 的 50 题总分直接比较，也不宣称提高答案
正确率。后续原生工具调用 PR 将针对反复生成无效 `action_input` 的逻辑失败单独优化。

文档和版本控制中不包含 API Key、本地 endpoint 配置、原始数据、模型响应、逐步 Trace、
预测文件或完整运行产物。提交内容只保留公开任务 ID、聚合耗时和脱敏后的状态统计。

## 使用方式

快速回归：

```bash
cd PHASE_1
uv run dabench run-benchmark \
  --config configs/react_baseline.local.yaml \
  --task-file configs/regression_tasks.example.txt
```

恢复同一 `run.run_id`，或归档并重跑失败任务：

```bash
uv run dabench run-benchmark \
  --config configs/react_baseline.local.yaml \
  --task-file configs/regression_tasks.example.txt \
  --resume

uv run dabench run-benchmark \
  --config configs/react_baseline.local.yaml \
  --task-file configs/regression_tasks.example.txt \
  --resume --retry-failed
```
