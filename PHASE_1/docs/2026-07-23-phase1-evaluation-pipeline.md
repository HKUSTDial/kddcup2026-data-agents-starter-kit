# Phase 1 本地评测链路改造记录

本次改造完成于 **2026 年 7 月 23 日**，开发分支为
`codex/phase1-evaluation`，起点是官方 Phase 1 Starter Kit 的提交
`069ee5b`。该起点同时保留为标签
`official-phase1-baseline-069ee5b`。本次提交只建设评测基础设施，不修改
ReAct Prompt、Agent 决策逻辑或现有工具行为，目的是先建立一个稳定、透明且可复现的
度量基线，为后续原生工具调用、答案验证、数据探索和文档 ETL 改造提供可信对照。

## 背景与改造动机

官方 Starter Kit 的定位是教学 baseline。模型调用 `answer` 后，程序只检查
`columns` 是否为非空字符串列表、`rows` 是否为列表，以及每一行的字段数量是否与模型
自己声明的列数一致。满足这些结构条件后，Runner 就会生成 `prediction.csv` 并把任务
记为成功。这个“成功”只表示 Agent 顺利结束，不表示预测内容与 `gold.csv` 一致；错误
答案、多余列、缺失行或重复数据都不会在运行阶段受到惩罚。

仓库此前也没有本地 scorer，公开数据虽然提供了 50 份 `gold.csv`，但缺少把预测转换为
任务分数和总体指标的实现。与此同时，根目录和 `PHASE_1` 目录的 `.gitignore` 都忽略了
`tests/`、`evaluation/` 和 `docs/`，导致即便在本地补充评测与测试，这些成果也无法正常
进入版本控制。没有固定评分口径和回归测试时，后续任何“Agent 变得更好”的结论都无法
被可靠证明。

因此，本次工作的重点不是提升模型回答能力，而是把“程序运行完成”和“答案确实正确”
拆成两个独立、可观测的结果，并用公开 gold 建立第一份真实 baseline。

## 实现后的评测链路

改造后新增了独立的 `data_agent_baseline.evaluation` 模块。评分器按照
[DataAgent-Bench 官方技术规则](https://dataagent.top/rules)进行列级匹配：每列的单元格
先完成空值、数字、日期时间和文本归一化，再对列内值排序并生成签名；预测列与 gold 列
按签名一对一匹配，列名、列顺序和行顺序不参与评分，重复列则按照出现次数匹配。任务
得分遵循以下公式：

```text
Recall = Matched Columns / Gold Columns
Penalty = λ × Extra Columns / Predicted Columns
Score = max(0, Recall - Penalty)
```

当前默认 `λ=0.1`，同时作为显式 CLI 参数和报告元数据保存。额外预测列会受到软惩罚；
某列只要增加、缺失或重复了一个值，完整列签名就会变化，不会因为“包含了部分正确值”
而被误判为匹配。评分器还明确处理缺失预测、损坏 CSV、重复列和无效 gold：缺失或损坏
的预测在对应任务上记 0 分，而受信任的 gold 缺失或不可解析时直接终止评测，避免输出
看似合法但实际无效的报告。

CLI 新增了 `dabench score-run` 命令。它读取一次 Runner 输出目录，为所有公开 gold
任务生成逐题结果，汇总总分、Mean Recall、缺失预测、读取错误和各难度平均分，并通过
原子写入生成 `scores.json`。部分任务运行仍按完整 gold 集合聚合，未执行的任务记 0，
从而避免把小规模 smoke test 的局部分数误当成完整 benchmark 成绩。中英文 README
和 `PHASE_1/docs/evaluation.md` 同步记录了命令、公式、归一化口径及评分限制。

改造前后的关键区别可以概括为：

| 方面 | 改造前 | 改造后 |
|---|---|---|
| 任务成功含义 | 只要 `answer` 结构合法并写出 CSV | 运行状态与答案得分分开记录 |
| Gold 对比 | 无 | 对全部公开任务进行列签名匹配 |
| 冗余与缺失 | 不检查 | 缺失列降低 Recall，额外列产生 Penalty |
| 格式归一化 | 无统一口径 | 数字、日期、空值和文本规则固定并测试 |
| 聚合指标 | 仅统计 Runner 成功数量 | 输出总分、Mean Recall、任务状态和难度分数 |
| 可验证性 | 无正式测试目录 | 单元测试、CLI 集成测试和真实运行差分验证 |

## 验证过程与脱敏基线结果

评测模块目前包含 31 个自动化测试，覆盖精确匹配、列名及顺序变化、额外列、缺失列、
错误列、重复列、额外行、跨列行关系、损坏 CSV、缺失预测，以及数值四舍五入、千分位、
日期时间、时区和空值归一化。测试使用 Python 3.10 执行，`pytest`、Ruff 静态检查、
本次涉及文件的格式检查和 `git diff --check` 均通过。

为了确认实现不是仅在合成 fixture 上成立，本次还使用同一批真实预测分别运行了当前
实现和第三方参考项目 `kddcup2026-dataagents` 的 scorer。50 个任务的
`score`、`recall`、`penalty`、匹配列数和额外列数全部一致；Mean Recall 只存在
`5.55e-17` 的浮点求和顺序尾差，不影响任何可见结果。

完整 baseline 使用阿里云百炼的 `qwen3.5-35b-a3b`，配置为 `temperature=0`、
`max_steps=16`、`max_workers=2`、单任务超时 600 秒。API Key、本地配置、原始数据、
逐步 Trace 和预测文件均保留在被忽略的本地目录中，未写入本文档或版本控制。脱敏后的
结果如下：

| 指标 | 结果 |
|---|---:|
| 公开任务数 | 50 |
| 生成预测的任务 | 31 |
| Runner 成功率 | 62% |
| 总分 | **0.4840** |
| Mean Recall | 0.4867 |
| 满分任务 | 23 |
| 部分匹配任务 | 2 |
| 有预测但得 0 分 | 6 |
| 未生成预测 | 19 |

按难度统计，Easy、Medium、Hard 和 Extreme 的平均分分别为 `0.4800`、`0.5217`、
`0.4545` 和 `0.0000`。19 个未生成预测的任务中，16 个在 16 步内没有调用
`answer`，另外 3 个达到 600 秒硬超时。成功或失败任务均未出现 API 鉴权、429、损坏
CSV 或整批 Runner 崩溃。单任务耗时中位数约为 31.6 秒，完整批次墙钟时间约为 32
分钟。

这些数字只描述当前模型配置下的一次真实 baseline，不代表经过多次重复实验后的稳定
期望值。同一个任务在单独运行和批量运行中曾出现不同答案，说明即使
`temperature=0`，模型服务或 Agent 轨迹仍可能存在波动。后续对比实验应固定任务集、
模型配置和评分器，并通过重复运行报告均值与方差。

## 本次提交的作用与后续方向

本次提交没有宣称提高 Agent 正确率。它实现的是更基础也更关键的能力：让每一次改造都
可以被同一套确定性规则评估，让运行失败、答案错误和冗余输出分别暴露，并为回归检测
保留结构化证据。完整 baseline 表明，当前最突出的瓶颈不是 CSV 能否写出，而是 Agent
在有限步骤内无法稳定终止，以及合法 CSV 中仍可能包含错误、漏项或不符合 gold
粒度的结果。

下一阶段可以在不改变评分口径的前提下重构原生工具调用和状态机，再引入候选答案
Verifier 与纠错闭环。每个阶段都应复用本次 scorer 和固定回归集，分别比较协议错误率、
任务得分、未提交率、工具调用数、Token 和耗时。这样，后续 README、实验报告和项目
经历中的优化结论都能追溯到可复现的测量结果，而不是依赖主观观察。

本地复现命令如下；模型配置文件与密钥应继续保存在被忽略的
`configs/react_baseline.local.yaml` 中：

```bash
cd PHASE_1

uv sync --locked --extra dev
uv run pytest -q
uv run dabench run-benchmark --config configs/react_baseline.local.yaml
uv run dabench score-run artifacts/runs/<run_id> \
  --gold-dir data/public/output \
  --input-dir data/public/input \
  --verbose
```
