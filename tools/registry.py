# 作用：管理所有工具，负责工具注册、工具查找、工具执行和工具说明渲染。
import json
from typing import Any

from tools.base import Tool


class ToolRegistry:
    """工具注册表，负责保存工具、查找工具和执行工具。"""

    # 作用：初始化一个空的工具注册表。
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    # 作用：注册一个工具，让 Agent 可以通过工具名调用它。
    def register(self, tool: Tool) -> None:
        """注册一个工具，让 Agent 后续可以调用它。"""
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    # 作用：根据工具名查找工具，执行工具，并返回工具观察结果。
    def run(self, name: str, tool_input: dict[str, Any]) -> str:
        """根据工具名找到工具并执行，返回观察结果。"""
        tool = self._tools.get(name)
        if not tool:
            return f"Tool '{name}' not found. Available tools: {', '.join(self._tools)}"

        try:
            return tool.run(tool_input)
        except Exception as exc:
            return f"Tool '{name}' failed: {exc}"

    # 作用：把所有已注册工具的说明转换成 JSON 字符串，放进 Prompt 给大模型看。
    def render_descriptions(self) -> str:
        """把所有工具说明渲染成 JSON 字符串，提供给大模型阅读。"""
        specs = [tool.spec().model_dump() for tool in self._tools.values()]
        return json.dumps(specs, ensure_ascii=False, indent=2)
