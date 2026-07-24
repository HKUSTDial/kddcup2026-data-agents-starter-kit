# Phase 1 原生工具调用改造记录

本次改造完成于 **2026 年 7 月 24 日**，开发分支为
`codex/phase1-native-tool-calling`。它延续本地评测链路和 Runner 可靠性改造，但不修改
评分规则、数据集、答案 Verifier、工具能力或分析策略，目标是把依赖 Prompt 约束的文本
JSON action 替换为 OpenAI-compatible 原生 `tools` / `tool_calls` 协议，并验证阿里云
百炼兼容接口能够完整执行该链路。

## 改造动机

此前模型必须在 fenced code block 中手写包含 `thought`、`action` 和 `action_input` 的
JSON。工具说明和参数示例以普通文本拼接进系统 Prompt，Runner 再自行抽取 fenced JSON
并检查顶层字段。工具注册表中的 `input_schema` 只是展示用示例，handler 会通过
`str()` 或 `int()` 转换部分输入，因此请求 schema、执行参数和 Prompt 文档并非同一个
事实来源。

这种设计容易出现 JSON 语法错误、额外文本、未知工具、缺失参数和静默类型转换。即便
接口本身支持 Function Calling，Agent 也没有使用 assistant `tool_calls` 和带匹配
`tool_call_id` 的 `tool` 消息，无法由 API 协议保证一次调用与一次观察正确配对。

## 实现后的协议链路

模型 Adapter 现在接收能够生成 OpenAI 工具定义的注册表，每次 Chat Completions 请求
传入同一组 `tools`，并使用 `tool_choice="auto"` 和
`parallel_tool_calls=False`。响应中的 call ID、函数名和原始参数字符串会先转换为
provider-neutral 数据类型，再交给 ReAct 循环：

```text
messages + tools
  → assistant.tool_calls[0]
  → JSON 解析
  → Pydantic 参数校验
  → handler
  → role=tool + matching tool_call_id
  → 下一轮请求或 answer 终止
```

该设计继续使用 OpenAI Python SDK 和已有 `agent.api_base`，没有引入 DashScope SDK
或供应商专用运行分支。百炼官方文档说明其 OpenAI-compatible Chat Completions 支持
`tools`、JSON Schema、`tool_calls` 和 `tool_call_id` 回传，且并行工具调用默认关闭：

- <https://help.aliyun.com/en/model-studio/qwen-function-calling>
- <https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions>

系统 Prompt 只删除旧 fenced JSON 格式要求，并说明必须通过原生接口每轮调用一个工具；
没有加入新的规划启发式或数据探索策略。工具描述和 JSON Schema 不再重复拼入 Prompt，
而是完全来自运行时注册表。原先文本 action 示例中隐含的最终答案形状约束，现由 `answer`
工具的 description 和 JSON Schema 直接承载：仅提交问题明确要求的最终列，不携带 join
key、筛选值、来源列、排名或中间统计列；额外列会受评分器惩罚。该条是协议迁移中的行为
等价恢复，不改变评分规则、Verifier 或工具执行逻辑。

## 类型校验与错误恢复

八个现有工具现在分别使用 Pydantic 输入模型，统一禁止额外字段并启用严格类型校验。
注册表使用同一个 `ToolSpec` 绑定工具名称、描述、输入模型、handler 和终止属性，由
输入模型直接生成 OpenAI-compatible JSON Schema。非法参数不会进入 handler。

可预期的模型或工具错误会转换为结构化、可恢复的工具结果：

| 错误码 | 含义 |
| --- | --- |
| `UNKNOWN_TOOL` | 模型返回了未注册的工具名 |
| `INVALID_ARGUMENTS_JSON` | `arguments` 不是合法 JSON |
| `ARGUMENTS_NOT_OBJECT` | 参数 JSON 顶层不是对象 |
| `ARGUMENT_VALIDATION_ERROR` | 缺字段、额外字段、类型或约束不满足 |
| `TOOL_EXECUTION_ERROR` | handler 执行时抛出异常 |

错误结果通过原始 call ID 作为 `tool` 消息返回，模型可以在下一轮修正。空
`tool_calls`、多个调用或缺少 call ID/函数名会记录为协议错误；系统不会从普通文本或
reasoning 中猜测并执行工具调用。若服务端违反串行设置返回多个调用，所有 call ID 都会
收到拒绝结果，避免只回复部分调用而生成非法消息历史。

## Trace 与事件兼容

`trace.json` 继续保留 `step_index`、`thought`、`action`、`action_input`、
`raw_response`、`observation` 和 `ok`。新增的 `tool_call_id` 与
`finish_reason` 是向后兼容字段；`raw_response` 现在保存稳定序列化的 assistant 原生
响应。模型通常不会在 assistant content 中返回显式 thought，因此 `thought` 可以为空。

`events.jsonl` 继续使用 `step_started`、模型请求、`tool_started`、
`tool_completed` / `tool_failed` 和 `step_completed` 等既有事件名。工具事件增加 call
ID、结构化错误码和可恢复标记，协议异常使用新增的 `protocol_error`。由于
`step_completed.step` 仍保存完整 step，任务硬超时后的部分步骤恢复逻辑无需修改。

旧 Trace 可以继续读取，但不同代码版本的实验不应复用同一个 `run_id`，避免在同一次
汇总中混合文本 action 和原生调用轨迹。

## 自动化验证

自动化测试覆盖：

- OpenAI 请求中的工具 schema、串行设置以及 assistant/tool 消息回放；
- call ID、函数名、参数字符串和 `finish_reason` 的响应解析；
- 八个工具 schema 的稳定生成和默认值；
- `answer` schema 中的最终列约束和原生 `columns` / `rows` 参数示例；
- 非法 JSON、非对象参数、严格类型、额外字段和未知工具；
- 参数校验失败不进入 handler，handler 异常可恢复；
- 正常多轮、参数纠正、空调用、多调用、`answer` 终止和步骤耗尽；
- Trace、事件、子进程超时、断点续跑、失败归档重跑和本地评分回归；
- 在全新 Python 解释器中直接导入工具注册表，防止循环导入回归。

共 56 项 pytest、Ruff 静态检查、相关文件格式检查和 `git diff --check` 均通过。
自动化测试不调用真实模型，也不依赖本地数据集或 API Key。

## 百炼真实实验

首先使用与上一轮相同的百炼模型配置运行单题 smoke test。任务在 10 个原生工具步骤后
成功提交答案；所有步骤均有 call ID，没有协议错误。一次 SQL 执行异常以
`TOOL_EXECUTION_ERROR` 返回，模型随后继续修正并完成任务，验证了真实接口下的错误
恢复链路。

随后使用 `max_steps=10`、`max_workers=2` 和 120 秒任务硬超时运行固定 11 题回归集。
整轮墙钟时间为 **191.324 秒**，5 个任务提交答案，6 个任务达到步骤上限。94 次模型
请求全部成功，94 个已完成 step 全部具有 call ID，没有协议错误。运行中发生 3 次工具
执行错误和 1 次参数校验错误，均作为可恢复结果返回。选中 11 题的平均分为 `0.2924`，
其中 4 题获得非零分；该局部平均分不能与完整 50 题总分直接比较。

最后使用正式配置运行 50 题，保持 `max_steps=16`、`max_workers=2`、20 秒模型请求
超时、一次应用层重试和 120 秒任务硬超时。脱敏结果如下：

| 指标 | 原生工具调用实验 |
| --- | ---: |
| 墙钟时间 | 1077.666 秒（17 分 58 秒） |
| Runner 成功 / 失败 | 34 / 16 |
| 累计任务时间 | 2098.901 秒 |
| 单任务耗时中位数 | 25.602 秒 |
| 模型请求尝试 | 500 |
| 成功模型请求平均 / P95 | 2.601 / 6.511 秒 |
| 已完成 step / 带 call ID | 490 / 490 |
| 原生协议错误 | 0 |
| 工具执行 / 参数校验错误 | 32 / 3 |
| 任务硬超时 | 6 |
| 总分 / Mean Recall | **0.4128** / 0.4233 |
| 满分 / 部分匹配 / 无匹配 / 缺失 | 14 / 9 / 11 / 16 |

16 个失败任务中，9 个达到步骤上限，6 个触发任务硬超时，1 个在模型请求显式重试后仍
超时。六个任务硬超时的最后活动分别为 3 个模型请求和 3 个 `execute_python`。全轮出现
4 次 `APITimeoutError` 和 3 次重试调度。

上一轮 Runner 可靠性正式实验为 15 分 1 秒、32 成功/18 失败、得分 0.4853。原生工具
调用本轮多提交了两个答案，且完全消除了文本 action 协议错误，但耗时增加约 2 分 57
秒，总分下降 0.0725。由于这是一次真实服务实验，模型输出和延迟存在波动，单轮结果不足
以把差异完全归因于协议；同时，这些数字也明确表明本 PR 不能宣称提高答案正确率或运行
速度。可以确认的收益是工具调用与观察具备原生 call ID 配对、参数在执行前得到严格
校验、错误可以结构化回传，以及全部 490 个完成步骤不再依赖 fenced JSON 解析。

按难度统计，Easy、Medium、Hard 和 Extreme 的平均分分别为 `0.4189`、`0.5249`、
`0.2076` 和 `0`。这些结果只用于记录本次协议迁移的真实表现，评分器、gold、预测文件
和答案验证逻辑均未修改。

### 答案输出契约的针对性验证

为验证降分是否与答案形状有关，随后只修改原生 `answer` 工具的 description 和 JSON
Schema：恢复单列最终答案参数示例，并明确禁止提交辅助列。没有改动 system prompt 的
探索/推理规则、评分器、数据集、Verifier、模型配置或 Runner。

回归集固定为 12 个“文本 action 正式实验满分、原生 50 题实验失分”的任务；其中 8 个在
原生实验中已经提交答案且存在答案形状问题，另外 4 个原生实验为缺失预测，保留它们以避免
将提交失败错误归因于工具描述。使用相同正式配置得到 run ID `20260724T062940Z`：

| 指标（12 题回归集） | 文本 action 基线 | 原生改造前 | 补强 `answer` 契约后 |
| --- | ---: | ---: | ---: |
| 分数总和 / 平均分 | 12.0000 / 1.0000 | 5.1583 / 0.4299 | 7.0000 / 0.5833 |
| 有预测任务数 | 12 | 8 | 10 |
| 额外列总数 | 0 | 11 | 3 |

对 8 个原生改造前已经有预测的任务，额外列从 11 个降为 1 个，7 个恢复满分；`task_196`
仍为无匹配，说明输出契约不能修复计算本身错误。回归集总计的剩余 3 个额外列来自本轮新
提交、但原生基线中缺失预测的两个任务，不能作为“同题答案形状”的直接比较。四个原生缺失
预测任务中两题仍缺失、两题改为无匹配，亦说明模型运行的非确定性和任务超时仍是独立变量。

该一次受控回归足以支持“丢失最终答案形状约束是原生改造退化的重要因素”，但不足以证明
全量 50 题分数必然提升。为验证该结论，随后使用相同正式配置完成第二轮 50 题全量实验，
run ID 为 `20260724T064248Z`：

| 指标 | 文本 action 基线 | 原生改造前 | 补强 `answer` 契约后 |
| --- | ---: | ---: | ---: |
| 墙钟时间 | 901.260 秒（15 分 1 秒） | 1077.666 秒（17 分 58 秒） | 887.745 秒（14 分 48 秒） |
| Runner 成功 / 失败 | 32 / 18 | 34 / 16 | 38 / 12 |
| 总分 / Mean Recall | 0.4853 / 0.4867 | 0.4128 / 0.4233 | **0.5303 / 0.5333** |
| 满分 / 部分匹配 / 无匹配 / 缺失 | 24 / 1 / 7 / 18 | 14 / 9 / 11 / 16 | 25 / 3 / 10 / 12 |
| 额外列总数 | 12 | 37 | 14 |

本轮的 12 个失败中，8 个达到步骤上限，4 个触发 120 秒任务硬超时。相比未补契约的原生
实验，总分提高 0.1175，且额外列明显回落；相比文本 action 基线，总分提高 0.0450。这是
当前配置下支持该迁移的实证结果，但真实服务输出和延迟仍有波动；若要量化稳定收益，仍应
在固定回归集或全量集重复多轮。实验 artifacts 仅保留在本地 ignored 目录，不纳入提交。

## 提交边界

本 PR 只包含 Phase 1 源码、测试、README 和本文档。它不包含 API Key、本地 endpoint
配置、数据集、模型响应、逐步 Trace、预测、评分产物或其他运行 artifacts，也不处理
Phase 2 视频输入。评分规则、答案 Verifier、Explorer、ETL、广泛 Prompt 策略优化和
新的 Runner 调度改造均留给后续独立 PR。
