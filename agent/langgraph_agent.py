# 作用：使用 LangGraph 实现是一个一步步推进的 Agent。
# 根据取址也是对的特征：LangGraph 提供了状态图管理、条件判断、自动流程控制
# 简化了原来的 ReAct 手写循环。
import json
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.constants import START

from agent.models import AgentResult, AgentStep
from agent.prompts import SYSTEM_PROMPT, build_user_prompt
from llm_client import LitCraftAgentsLLM
from tools.registry import ToolRegistry


class AgentState(TypedDict):
    """LangGraph 状态定义：记录 Agent 执行过程中的所有信息。"""
    topic: str
    steps: list[AgentStep]
    current_thought: str
    current_action: str
    current_action_input: dict
    current_observation: str
    is_final: bool
    final_answer: str
    step_count: int


class LangGraphAgent:
    """基于 LangGraph 的 Agent，使用状态图管理执行流程。"""

    def __init__(
        self,
        llm: LitCraftAgentsLLM,
        tools: ToolRegistry,
        max_steps: int = 6,
        temperature: float = 0,
    ) -> None:
        """初始化 Agent，构建 LangGraph 状态图。"""
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.temperature = temperature
        
        # 构建 LangGraph 状态图
        self.graph = self._build_graph()

    def _build_graph(self):
        """构建 LangGraph 状态图，定义 nodes 和 edges。"""
        graph = StateGraph(AgentState)
        
        # 定义 nodes
        graph.add_node("think", self._think_node)
        graph.add_node("tool", self._tool_node)
        graph.add_node("end", self._end_node)
        
        # 定义 edges：从 START 到 think
        graph.add_edge(START, "think")
        
        # 从 think 到 tool 或 end（基于是否有 final_answer）
        graph.add_conditional_edges(
            "think",
            self._should_end,
            {"end": "end", "tool": "tool"}
        )
        
        # 从 tool 回到 think（循环）
        graph.add_edge("tool", "think")
        
        # end node 到 END
        graph.add_edge("end", END)
        
        return graph.compile()

    def _think_node(self, state: AgentState) -> AgentState:
        """思考节点：调用 LLM 进行推理决策。"""
        user_prompt = build_user_prompt(
            topic=state["topic"],
            tools=self.tools,
            scratchpad=self._build_scratchpad(state["steps"]),
        )
        raw_response = self.llm.chat(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=self.temperature,
        )
        decision = self._parse_decision(raw_response)
        
        thought = str(decision.get("thought", "")).strip()
        state["current_thought"] = thought
        state["step_count"] += 1
        
        # 检查是否需要返回最终答案
        if "final_answer" in decision:
            state["is_final"] = True
            state["final_answer"] = str(decision["final_answer"]).strip()
        else:
            # 解析 action 和 action_input
            action = str(decision.get("action", "")).strip()
            action_input = decision.get("action_input", {})
            if not isinstance(action_input, dict):
                action_input = {"input": action_input}
            
            state["current_action"] = action
            state["current_action_input"] = action_input
            state["is_final"] = False
        
        return state

    def _tool_node(self, state: AgentState) -> AgentState:
        """工具节点：执行选定的工具。"""
        action = state["current_action"]
        action_input = state["current_action_input"]
        
        # 执行工具
        observation = self.tools.run(action, action_input)
        state["current_observation"] = observation
        
        # 记录这一步
        step = AgentStep(
            thought=state["current_thought"],
            action=action,
            action_input=action_input,
            observation=observation,
        )
        state["steps"].append(step)
        
        return state

    def _end_node(self, state: AgentState) -> AgentState:
        """结束节点：记录最终步骤（不包含 action）。"""
        # 记录最后的思考步骤
        step = AgentStep(thought=state["current_thought"])
        state["steps"].append(step)
        return state

    def _should_end(self, state: AgentState) -> str:
        """条件判断：是否应该结束循环。"""
        # 如果有 final_answer 或达到最大步数，就结束
        if state["is_final"] or state["step_count"] >= self.max_steps:
            return "end"
        return "tool"

    def run(self, topic: str) -> AgentResult:
        """执行 Agent：调用 LangGraph 状态图处理输入。"""
        initial_state = AgentState(
            topic=topic,
            steps=[],
            current_thought="",
            current_action="",
            current_action_input={},
            current_observation="",
            is_final=False,
            final_answer="",
            step_count=0,
        )
        
        # 执行图
        final_state = self.graph.invoke(initial_state)
        
        # 处理最终答案
        final_answer = final_state["final_answer"]
        if not final_answer:
            # 如果没有 final_answer（达到最大步数），使用默认回复
            final_answer = (
                "The agent reached the maximum step limit before producing a final answer. "
                "Try narrowing the topic or increasing max_steps."
            )
        
        return AgentResult(
            final_answer=final_answer,
            steps=final_state["steps"],
        )

    def _build_scratchpad(self, steps: list[AgentStep]) -> str:
        """把历史步骤整理成文本，放回下一轮 Prompt 里作为上下文。"""
        lines: list[str] = []
        for index, step in enumerate(steps, start=1):
            lines.append(f"Step {index}")
            lines.append(f"Thought: {step.thought}")
            if step.action:
                lines.append(f"Action: {step.action}")
                lines.append(
                    f"Action Input: {json.dumps(step.action_input, ensure_ascii=False)}"
                )
            if step.observation:
                lines.append(f"Observation: {step.observation}")
        return "\n".join(lines)

    def _parse_decision(self, raw_response: str) -> dict[str, Any]:
        """把大模型返回的 JSON 字符串解析成 Python 字典。
        
        处理 LLM 返回的 JSON 中可能包含的未转义控制字符（如 final_answer 中的 markdown 换行）。
        """
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.removeprefix("json").strip()
        try:
            # strict=False 允许字符串中包含控制字符（如 \n 换行）
            from json import JSONDecoder
            return JSONDecoder(strict=False).decode(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM must return valid JSON, got: {raw_response}") from exc
