# 作用：定义 Agent 执行过程中用到的数据结构，例如单步执行记录和最终结果。
from typing import Any

from pydantic import BaseModel, Field


class AgentStep(BaseModel):
    """记录 ReAct Agent 的单步执行信息。"""

    thought: str
    action: str | None = None
    action_input: dict[str, Any] = Field(default_factory=dict)
    observation: str | None = None


class AgentResult(BaseModel):
    """记录 Agent 最终输出和完整执行轨迹。"""

    final_answer: str
    steps: list[AgentStep]
