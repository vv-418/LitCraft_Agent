# 作用：定义工具系统的基础协议，规定一个工具应该有哪些信息和执行方法。
import asyncio
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ToolSpec(BaseModel):
    """工具说明，用来告诉大模型这个工具叫什么、能做什么、需要什么输入。"""

    name: str
    description: str
    input_schema: dict[str, Any]


class Tool(ABC):
    """所有工具都要继承的基础类。"""

    name: str
    description: str
    input_schema: dict[str, Any]

    # 作用：所有具体工具必须实现这个方法，用来执行工具逻辑并返回 observation。
    @abstractmethod
    def run(self, tool_input: dict[str, Any]) -> str:
        """执行工具，并返回一段文本形式的观察结果。"""

    async def async_run(self, tool_input: dict[str, Any]) -> str:
        """异步执行工具。默认用线程池包装同步 run()，子类可重写为纯异步实现。"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.run, tool_input)

    # 作用：把工具的名称、描述和输入格式打包成 ToolSpec，供 Prompt 使用。
    def spec(self) -> ToolSpec:
        """把工具信息转换成可以放进 Prompt 的结构化说明。"""
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
        )
