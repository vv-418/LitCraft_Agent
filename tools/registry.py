# 作用：管理所有工具，负责工具注册、工具查找、工具执行和工具说明渲染。
# 每个工具在系统中只有一个精确的名字，LLM 必须使用该名字调用。
import json
from typing import Any

from tools.base import Tool


class ToolRegistry:
    """工具注册表，负责保存工具、查找工具和执行工具。"""

    # LLM 常把"生成综述"当成工具调用，实际上应该走 final_answer JSON
    _GENERATION_ALIASES = frozenset({
        "generate_overview", "generate_review", "write_review",
        "generate_summary", "write_summary", "generate_answer",
        "summarize", "generate", "write",
        "analyze_results", "summarize_results", "organize_results",
    })

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """注册工具，使用工具本身的 name 作为唯一名字。"""
        if tool.name not in self._tools:
            self._tools[tool.name] = tool

    def run(self, name: str, tool_input: dict[str, Any]) -> str:
        """根据工具名找到工具并执行，返回观察结果。"""
        # 拦截"生成综述"类工具名 → 提示 LLM 改用 final_answer
        if name.lower() in self._GENERATION_ALIASES:
            return (
                "不要用工具写综述。请调用 final_answer 结束检索"
                "（或在 JSON 协议下输出 {\"final_answer\": \"检索结束\"}）。"
            )
        tool = self._tools.get(name)
        if not tool:
            from utils.logger import get_logger
            log = get_logger(__name__)
            names = list(self._tools.keys())
            log.warning("未知工具调用: %s | 可用工具: %s", name, ", ".join(names))
            return f"Tool '{name}' not found. Available tools: {', '.join(names)}"
        try:
            return tool.run(tool_input)
        except Exception as exc:
            return f"Tool '{name}' failed: {exc}"

    _SYSTEM_OWNED_SEARCH_KEYS = frozenset({"limit", "per_source_limit", "year_from"})

    def openai_tools(self) -> list[dict[str, Any]]:
        """OpenAI / DeepSeek 兼容的 tools 列表（含结束检索的 final_answer）。"""
        tools: list[dict[str, Any]] = []
        for tool in self._tools.values():
            schema = tool.input_schema if isinstance(tool.input_schema, dict) else {
                "type": "object",
                "properties": {},
            }
            if schema.get("type") != "object":
                schema = {"type": "object", "properties": {}, "required": []}
            else:
                schema = {
                    "type": "object",
                    "properties": dict(schema.get("properties") or {}),
                    "required": list(schema.get("required") or []),
                }
                for key in self._SYSTEM_OWNED_SEARCH_KEYS:
                    schema["properties"].pop(key, None)
                schema["required"] = [k for k in schema["required"] if k not in self._SYSTEM_OWNED_SEARCH_KEYS]
            tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": (tool.description or "")[:512],
                    "parameters": schema,
                },
            })
        tools.append({
            "type": "function",
            "function": {
                "name": "final_answer",
                "description": (
                    "End the literature search phase. Do not write the full survey here; "
                    "the system will compose it afterwards."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Why retrieval can stop (one short sentence)",
                        }
                    },
                },
            },
        })
        return tools

    def render_descriptions(self) -> str:
        """把所有工具说明渲染成 JSON 字符串，提供给大模型阅读。"""
        specs = [tool.spec().model_dump() for tool in self._tools.values()]
        return json.dumps(specs, ensure_ascii=False, indent=2)
