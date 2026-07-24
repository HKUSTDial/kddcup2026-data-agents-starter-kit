from __future__ import annotations

from data_agent_baseline.benchmark.schema import PublicTask


REACT_SYSTEM_PROMPT = """
You are a ReAct-style data agent.

You are solving a task from a public dataset. You may only inspect files inside the task's `context/` directory through the provided tools.

Rules:
1. Use tools to inspect the available context before answering.
2. Base your answer only on information you can observe through the provided tools.
3. The task is complete only when you call the `answer` tool.
4. The `answer` tool must receive a table with `columns` and `rows`.
5. Call exactly one tool in each turn through the native tool-calling interface.
6. Do not write or simulate tool calls in plain text or JSON.

Keep reasoning concise and grounded in the observed data.
""".strip()


def build_system_prompt(system_prompt: str | None = None) -> str:
    return system_prompt or REACT_SYSTEM_PROMPT


def build_task_prompt(task: PublicTask) -> str:
    return (
        f"Question: {task.question}\n"
        "All tool file paths are relative to the task context directory. "
        "When you have the final table, call the `answer` tool."
    )
