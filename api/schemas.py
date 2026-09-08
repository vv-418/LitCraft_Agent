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
    output_dir: str = Field(
        default="",
        description="兼容旧参数：同时作为论文与综述的自定义根目录",
    )
    papers_output_dir: str = Field(
        default="",
        description="下载论文/插图保存目录。留空则用 output/日期/主题/{lit_source,figures}",
    )
    papers_filename: str = Field(
        default="",
        description="下载论文文件名前缀。留空则用「标题__网址.pdf」",
    )
    review_output_dir: str = Field(
        default="",
        description="最终综述 PDF 保存目录。留空则用 output/日期/主题/papers/",
    )
    review_filename: str = Field(
        default="",
        description="综述 PDF 文件名。留空则用「01主题_综述_时刻.pdf」",
    )
    per_source_limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="每个学术源（arXiv / Semantic Scholar / Google Scholar）单次最多保留篇数",
    )
    final_limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="多源合并去重后最终对外保留的论文篇数",
    )
    llm_model_id: str = Field(
        default="",
        description="覆盖默认模型 ID；须与 llm_api_key、llm_base_url 同时填写才生效",
    )
    llm_api_key: str = Field(
        default="",
        description="覆盖默认 API Key；三项未填齐则仍用本地 .env 模型",
    )
    llm_base_url: str = Field(
        default="",
        description="覆盖默认 OpenAI 兼容接口地址，如 https://api.deepseek.com/v1",
    )
    llm_source: str = Field(
        default="",
        description="local：未填项用 .env 补齐；online：须三项同时填写",
    )
    llm_timeout: int | None = Field(
        default=None,
        ge=1,
        le=600,
        description="覆盖请求超时秒数；留空则用默认",
    )


class LlmInfoResponse(BaseModel):
    """默认大模型的公开信息（不含密钥）。"""

    model: str = Field(default="", description="当前 .env 中的默认模型 ID")
    base_url: str = Field(default="", description="当前 .env 中的默认接口地址")
    timeout: int = Field(default=60, description="当前默认超时秒数")
    ready: bool = Field(default=False, description="默认三项是否已在服务器配齐")


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
    summary: dict[str, Any] = Field(
        default_factory=dict,
        description="面向用户的步骤摘要（标题、命中篇数、论文列表等）",
    )


class AgentResultResponse(BaseModel):
    """Agent 执行结果的完整响应。"""

    task_id: str = Field(..., description="任务唯一 ID")
    topic: str = Field(..., description="研究主题")
    status: str = Field(..., description="最终状态：done / error")
    final_answer: str = Field(default="", description="最终文献综述正文（Markdown 格式）")
    steps: list[StepResponse] = Field(default_factory=list, description="完整 ReAct 执行轨迹")
    pdf_path: str | None = Field(default=None, description="PDF 文件路径（仅 save_pdf=True 时存在）")
    output_folder: str | None = Field(default=None, description="本次任务根目录（日期/主题或自选论文目录）")
    papers_folder: str | None = Field(default=None, description="下载论文保存目录（lit_source）")
    review_folder: str | None = Field(default=None, description="综述 PDF 保存目录")
    error: str | None = Field(default=None, description="错误信息（仅 status=error 时存在）")
    traceback: str | None = Field(default=None, description="完整错误堆栈（仅 status=error 时存在）")
    created_at: str = Field(default="", description="任务创建时间")


class TaskListItem(BaseModel):
    """任务历史列表中的一项。"""

    task_id: str = Field(..., description="任务唯一 ID")
    topic: str = Field(..., description="研究主题")
    status: str = Field(..., description="任务状态")
    created_at: str = Field(..., description="任务创建时间")


class PickSavePathRequest(BaseModel):
    """弹出系统「另存为」对话框时的参数。"""

    kind: str = Field(default="review", description="papers=下载论文；review=综述 PDF")
    topic: str = Field(default="", description="用于预填综述默认文件名")
    initial_file: str = Field(default="", description="对话框里预填的文件名")


class AgentSavePdfRequest(BaseModel):
    """任务完成后，由用户手动保存综述 PDF。"""

    review_output_dir: str = Field(..., min_length=1, description="保存文件夹")
    review_filename: str = Field(default="", description="文件名，留空则用默认 01主题_综述_时刻")
