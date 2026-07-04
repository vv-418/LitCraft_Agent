"""
作用：定义 API 接口的请求和响应数据结构，使用 Pydantic 做自动校验和文档生成。

包含三个主要模型：
- AgentStartRequest：前端发起综述任务时的请求体
- TaskStatusResponse：返回任务状态的响应体
- AgentResultResponse：返回最终结果和执行步骤的响应体
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AgentStartRequest(BaseModel):
    """启动 Agent 综述任务的请求参数。"""

    topic: str = Field(..., description="研究主题，如「Transformer 在自然语言处理中的应用」")
    year_from: str = Field(default="", description="年份过滤条件，只搜索该年份之后的文献（空字符串表示不限制）")
    save_pdf: bool = Field(default=False, description="是否在生成综述后自动保存为 PDF 文件")


class TaskStatusResponse(BaseModel):
    """任务状态的响应，用于前端轮询。"""

    task_id: str = Field(..., description="任务唯一 ID")
    status: str = Field(..., description="任务状态：running / done / error")
    created_at: str = Field(default="", description="任务创建时间")


class StepResponse(BaseModel):
    """单步 ReAct 执行记录的响应格式。"""

    thought: str = Field(default="", description="这一步模型的思考过程")
    action: str | None = Field(default=None, description="调用的工具名（空表示这是最终步骤）")
    action_input: dict[str, Any] = Field(default_factory=dict, description="工具输入参数")
    observation: str | None = Field(default=None, description="工具返回的观察结果（已截断）")


class AgentResultResponse(BaseModel):
    """Agent 执行结果的完整响应。"""

    task_id: str = Field(..., description="任务唯一 ID")
    topic: str = Field(..., description="研究主题")
    status: str = Field(..., description="最终状态：done / error")
    final_answer: str = Field(default="", description="最终文献综述正文（Markdown 格式）")
    steps: list[StepResponse] = Field(default_factory=list, description="完整 ReAct 执行轨迹")
    pdf_path: str | None = Field(default=None, description="PDF 文件路径（仅 save_pdf=True 时存在）")
    error: str | None = Field(default=None, description="错误信息（仅 status=error 时存在）")
    traceback: str | None = Field(default=None, description="完整错误堆栈（仅 status=error 时存在）")
    created_at: str = Field(default="", description="任务创建时间")


class TaskListItem(BaseModel):
    """任务历史列表中的一项。"""

    task_id: str = Field(..., description="任务唯一 ID")
    topic: str = Field(..., description="研究主题")
    status: str = Field(..., description="任务状态")
    created_at: str = Field(..., description="任务创建时间")
