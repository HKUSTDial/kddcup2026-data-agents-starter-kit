<div align="center">

# DataAgent-Bench Starter Kit

[English](README.md) | 中文

[![官方网站](https://img.shields.io/badge/Official%20Website-Visit%20dataagent.top-0ea5e9?style=for-the-badge&logo=googlechrome&logoColor=white&labelColor=0f172a)](https://dataagent.top)
[![Demo 数据集](https://img.shields.io/badge/Demo%20Dataset-Download%20Phase%201-f59e0b?style=for-the-badge&logo=googledrive&logoColor=white&labelColor=0f172a)](https://drive.google.com/file/d/1lICQVM_LfyQ5DMEIZjssq6aaOPTWCtNd/view?usp=share_link)
[![Discord](https://img.shields.io/badge/Discord-Join%20Community-5865F2?style=for-the-badge&logo=discord&logoColor=white&labelColor=0f172a)](https://discord.com/invite/7eFwJQN3Fx)

</div>

> 面向 KDD Cup 2026 DataAgent-Bench 挑战的官方 starter kit。仓库默认读取 `data/public/input/`，并为后续评测生成预测结果。

## Overview

| 项目 | 内容 |
| --- | --- |
| 数据输入 | `data/public/input/` |
| 公开 demo 标准答案 | `data/public/output/task_<id>/gold.csv` |
| hidden test 数据 | 仅提供 `input/`，不提供 `output/` |
| 入口命令 | `uv run dabench <command> --config PATH` |
| 默认输出目录 | `artifacts/runs/` |

## 快速开始

1. 请先按照 `uv` 官方安装指南安装 `uv`：
   - https://docs.astral.sh/uv/getting-started/installation/
2. 在 macOS 和 Linux 上，官方独立安装命令为：

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. 安装项目依赖：

   ```bash
   uv sync
   ```

4. 检查数据集根目录是否可见：

   ```bash
   uv run dabench status --config configs/react_baseline.example.yaml
   ```

5. 运行 baseline：

   ```bash
   uv run dabench run-benchmark --config configs/react_baseline.example.yaml
   ```

## 数据集

公开 demo 数据集默认位于 `data/public/input/`。每个任务目录结构如下：

```text
data/public/input/task_<id>/
├── task.json
└── context/
```

公开 demo 的标准答案文件单独放在 `data/public/output/task_<id>/gold.csv`。
hidden test set 只提供 `input/`，不会包含 `output/`。

`task.json` 包含：

- `task_id`
- `difficulty`
- `question`

`context/` 中可能包含一种或多种数据：

- CSV 文件
- JSON 文件
- SQLite / DB 文件
- 文本文档

## 配置

示例配置文件位于 `configs/react_baseline.example.yaml`。

```yaml
dataset:
  root_path: data/public/input

agent:
  model: YOUR_MODEL_NAME
  api_base: YOUR_API_BASE_URL
  api_key: YOUR_API_KEY
  max_steps: 16
  temperature: 0.0

run:
  output_dir: artifacts/runs
  run_id:
  max_workers: 4
  task_timeout_seconds: 600
```

配置字段说明：

| 字段 | 含义 |
| --- | --- |
| `dataset.root_path` | 公开 demo `input/` 数据集根目录。相对路径按项目根目录解析。 |
| `agent.model` | 模型名称。 |
| `agent.api_base` | OpenAI-compatible 接口根地址。 |
| `agent.api_key` | API key，直接从配置文件读取。 |
| `agent.max_steps` | 单个任务允许的最大 ReAct 步数。 |
| `agent.temperature` | 模型采样温度。 |
| `run.output_dir` | 运行产物输出目录。 |
| `run.run_id` | 可选，指定运行目录名。不传时默认使用 UTC 时间戳；必须是单个目录名，已存在会报错。 |
| `run.max_workers` | `run-benchmark` 并行 worker 数。 |
| `run.task_timeout_seconds` | 单个任务允许的最长墙钟时间。设为 `0` 或负数可关闭任务级超时。 |

## CLI

```bash
uv run dabench <command> --config PATH [options]
```

| 命令 | 作用 | 示例 |
| --- | --- | --- |
| `status` | 查看项目路径、配置路径、数据集根目录和公开任务数量。 | `uv run dabench status --config configs/react_baseline.example.yaml` |
| `inspect-task` | 查看任务元信息，并列出 `context/` 下可访问文件。 | `uv run dabench inspect-task task_1 --config configs/react_baseline.local.yaml` |
| `run-task` | 对单个任务运行 baseline，并写出结果。 | `uv run dabench run-task task_1 --config configs/react_baseline.local.yaml` |
| `run-benchmark` | 批量运行整个公开数据集。 | `uv run dabench run-benchmark --config configs/react_baseline.local.yaml` |

`run-benchmark` 还支持 `--limit N`，用于限制任务数量。

## Tools

当前暴露给模型的工具有：

| 工具 | 作用 | 输入 |
| --- | --- | --- |
| `list_context` | 列出 `context/` 下的文件和目录。 | `max_depth` |
| `read_csv` | 读取 CSV 预览。 | `path`、`max_rows` |
| `read_json` | 读取 JSON 预览。 | `path`、`max_chars` |
| `read_doc` | 读取文本文档预览。 | `path`、`max_chars` |
| `inspect_sqlite_schema` | 查看 SQLite / DB 文件中的表结构。 | `path` |
| `execute_context_sql` | 对 `context/` 内 SQLite / DB 文件执行只读 SQL。 | `path`、`sql`、`limit` |
| `execute_python` | 在任务 `context/` 目录内执行任意 Python 代码。 | `code` |
| `answer` | 提交最终答案表格并结束当前任务。 | `columns`、`rows` |

所有文件路径都必须是相对于任务 `context/` 目录的相对路径。

## 输出

每个任务运行后可能生成：

- `trace.json`
- `prediction.csv`

单任务产物路径：

```text
artifacts/runs/<run_id>/<task_id>/
├── trace.json
└── prediction.csv
```

批量运行还会额外生成：

```text
artifacts/runs/<run_id>/summary.json
```

## Contact

- 问题反馈： https://github.com/HKUSTDial/kddcup2026-data-agents-starter-kit/issues
- 官方网站： https://dataagent.top
- Discord： https://discord.com/invite/7eFwJQN3Fx
- 微信公众号：`数据智能与分析实验室 DIAL`

<div align="center">
  <table>
    <tr>
      <td align="center">
        <a href="https://dataagent.top">
          <img
            src="https://api.qrserver.com/v1/create-qr-code/?size=144x144&data=https://dataagent.top&bgcolor=ffffff&color=111827&margin=8"
            alt="Official website QR code"
            width="144"
          />
        </a>
        <br />
        官方网站
      </td>
      <td align="center">
        <a href="https://discord.com/invite/7eFwJQN3Fx">
          <img
            src="https://api.qrserver.com/v1/create-qr-code/?size=144x144&data=https://discord.com/invite/7eFwJQN3Fx&bgcolor=ffffff&color=111827&margin=8"
            alt="Discord QR code"
            width="144"
          />
        </a>
        <br />
        Discord
      </td>
      <td align="center">
        <img
          src="https://dataagent.top/HKUSTGZ_DIAL.jpg"
          alt="WeChat official account QR code"
          width="144"
        />
        <br />
        微信公众号
      </td>
    </tr>
  </table>
</div>

## 主要模块

| 模块 | 责任 |
| --- | --- |
| `src/data_agent_baseline/benchmark/dataset.py` | 公开数据集加载器 |
| `src/data_agent_baseline/tools/filesystem.py` | `list_context`、`read_csv`、`read_json`、`read_doc` |
| `src/data_agent_baseline/tools/python_exec.py` | `execute_python` |
| `src/data_agent_baseline/tools/sqlite.py` | `inspect_sqlite_schema`、`execute_context_sql` |
| `src/data_agent_baseline/tools/registry.py` | 工具注册与终止型 `answer` |
| `src/data_agent_baseline/agents/prompt.py` | system prompt、task prompt、observation prompt |
| `src/data_agent_baseline/agents/react.py` | 基于 JSON action 协议的 ReAct runtime |
| `src/data_agent_baseline/run/runner.py` | 单任务和批量运行逻辑 |
