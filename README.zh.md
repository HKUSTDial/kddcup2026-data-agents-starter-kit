<div align="center">

# KDD Cup 2026 DataAgent-Bench Starter Kit

[English](README.md) | 中文

[![官方网站](https://img.shields.io/badge/Official%20Website-Visit%20dataagent.top-0ea5e9?style=for-the-badge&logo=googlechrome&logoColor=white&labelColor=0f172a)](https://dataagent.top)
[![Discord](https://img.shields.io/badge/Discord-Join%20Community-5865F2?style=for-the-badge&logo=discord&logoColor=white&labelColor=0f172a)](https://discord.com/invite/7eFwJQN3Fx)

</div>

本仓库提供 KDD Cup 2026 DataAgent-Bench 的 baseline starter kit。仓库按比赛阶段组织，参赛者可以根据自己正在使用的数据格式进入对应目录。

## 仓库结构

| 目录 | 用途 |
| --- | --- |
| `PHASE_1/` | 第一阶段任务格式使用的 starter kit。 |
| `PHASE_2/` | 第二阶段任务格式和 demo release 使用的 starter kit。 |

每个阶段目录都是独立的，包含自己的 README、配置文件、源码、依赖锁文件和 baseline 命令行入口。

## 应该使用哪个目录？

如果你正在处理第一阶段任务格式，请使用 `PHASE_1/`。

如果你正在处理第二阶段 demo release 或第二阶段任务格式，请使用 `PHASE_2/`。第二阶段保留相同的基础 agent 工作流，同时支持更丰富的任务上下文文件。

## 快速开始

请先进入对应阶段目录，再按照该目录下的 README 操作。

```bash
cd PHASE_1
# 或
cd PHASE_2
```

然后在所选目录内安装依赖并运行 baseline：

```bash
uv sync
uv run dabench status --config configs/react_baseline.example.yaml
uv run dabench run-benchmark --config configs/react_baseline.example.yaml
```

具体的数据目录结构、配置字段、工具和输出路径，请查看各阶段目录下的 README。

## Baseline 概览

starter kit 中包含一个最小 ReAct-style data agent。baseline 保持简单：读取任务元信息，通过一组基础工具查看每个任务 `context/` 目录下的文件，调用 OpenAI-compatible 模型接口，并写出 `prediction.csv`。

参赛者可以在此基础上自行改造或替换 agent 逻辑。代码不会硬编码服务凭据，模型接口相关信息应通过配置或环境变量兼容的方式传入。

## 常见项目结构

每个阶段目录大致包含以下内容：

```text
configs/                         # baseline 示例配置
src/data_agent_baseline/          # baseline 源码
artifacts/                        # 本地运行产物，除 .gitkeep 外不应提交
README.md                         # 英文使用说明
README.zh.md                      # 中文使用说明
pyproject.toml                    # Python 项目元信息
uv.lock                           # 锁定的依赖版本
```

## 注意事项

- 请在 `PHASE_1/` 或 `PHASE_2/` 目录内运行命令，不要直接在仓库根目录运行。
- 本地运行产物请放在 `artifacts/` 下；这些文件不应提交到仓库。
- 在打包或提交方案前，请先阅读对应阶段目录下的 README。

## 工程记录

- [2026-07-23：Phase 1 本地评测链路改造](PHASE_1/docs/2026-07-23-phase1-evaluation-pipeline.md)
- [2026-07-24：Phase 1 运行可靠性与实验反馈周期改造](PHASE_1/docs/2026-07-24-runner-reliability.md)
- [2026-07-24：Phase 1 原生工具调用改造](PHASE_1/docs/2026-07-24-native-tool-calling.md)

## 联系方式

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

每个阶段目录内部使用相同的 baseline 结构。例如进入 `PHASE_1/` 或 `PHASE_2/` 后，核心模块包括：

| 模块 | 责任 |
| --- | --- |
| `src/data_agent_baseline/benchmark/dataset.py` | 数据集加载器 |
| `src/data_agent_baseline/tools/filesystem.py` | `list_context`、`read_csv`、`read_json`、`read_doc` |
| `src/data_agent_baseline/tools/python_exec.py` | `execute_python` |
| `src/data_agent_baseline/tools/sqlite.py` | `inspect_sqlite_schema`、`execute_context_sql` |
| `src/data_agent_baseline/tools/registry.py` | 工具注册与终止型 `answer` |
| `src/data_agent_baseline/agents/prompt.py` | system prompt 与 task prompt |
| `src/data_agent_baseline/agents/react.py` | ReAct runtime 与模型/工具协议 |
| `src/data_agent_baseline/run/runner.py` | 单任务和批量运行逻辑 |
