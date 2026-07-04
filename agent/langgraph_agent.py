# 作用：使用 LangGraph 实现是一个一步步推进的 Agent。
# 根据取址也是对的特征：LangGraph 提供了状态图管理、条件判断、自动流程控制
# 简化了原来的 ReAct 手写循环。
import json
import logging
import sys
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.constants import START

from agent.models import AgentResult, AgentStep
from agent.prompts import SYSTEM_PROMPT, build_user_prompt
from llm_client import LitCraftAgentsLLM
from tools.registry import ToolRegistry
from utils.logger import get_logger

logger = get_logger("agent")


class AgentState(TypedDict):
    """LangGraph 状态定义：记录 Agent 执行过程中的所有信息。"""
    topic: str
    year_from: str
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
        year_from: str = "",
    ) -> None:
        """初始化 Agent，构建 LangGraph 状态图。"""
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.temperature = temperature
        self.year_from = year_from
        
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
        # "think" 分支用于强制 Agent 重试（跳过空动作直接再次思考）
        graph.add_conditional_edges(
            "think",
            self._should_end,
            {"end": "end", "tool": "tool", "think": "think"}
        )
        
        # 从 tool 回到 think（循环）
        graph.add_edge("tool", "think")
        
        # end node 到 END
        graph.add_edge("end", END)
        
        return graph.compile()

    def _think_node(self, state: AgentState) -> AgentState:
        """思考节点：调用 LLM 进行推理决策。"""
        step_num = state["step_count"] + 1
        max_steps = self.max_steps
        print(f"\n{'='*60}", flush=True, file=sys.stderr)
        print(f"  [AGENT] 🤔 第 {step_num}/{max_steps} 步 — LLM 思考中...", flush=True, file=sys.stderr)
        print(f"  [AGENT] 📋 主题: {state['topic'][:80]}", flush=True, file=sys.stderr)
        print(f"{'='*60}", flush=True, file=sys.stderr)

        logger.info("步骤 %d | 开始思考", step_num)
        user_prompt = build_user_prompt(
            topic=state["topic"],
            tools=self.tools,
            scratchpad=self._build_scratchpad(state["steps"]),
            year_from=state["year_from"],
        )
        try:
            raw_response = self.llm.chat(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=self.temperature,
            )
        except Exception as e:
            logger.error("LLM 调用失败 | error=%s", str(e), exc_info=True)
            # LLM 不可用（如 API 超时/500）时，生成兜底答案继续执行
            print(f"  [AGENT] ❌ LLM 调用失败，生成兜底答案", flush=True, file=sys.stderr)
            state["is_final"] = True
            state["final_answer"] = (
                f"# 关于「{state['topic']}」的文献综述\n\n"
                f"（注：当前 LLM 服务暂时不可用，以下内容基于已收集的搜索结果编写。）\n\n"
                + self._build_fallback_review(state["topic"], state.get("steps", []))
            )
            return state

        decision = self._parse_decision(raw_response)

        thought = str(decision.get("thought", "")).strip()
        state["current_thought"] = thought
        state["step_count"] += 1

        # 检查是否需要返回最终答案
        if "final_answer" in decision:
            state["is_final"] = True
            raw = decision["final_answer"]
            print(f"  [AGENT] ✅ 第 {step_num}/{max_steps} 步 → 生成最终答案!", flush=True, file=sys.stderr)
            # LLM 有时把 final_answer 输出为 dict（含 title/content 等字段）
            # 需要转成纯文本 Markdown
            if isinstance(raw, dict):
                lines = []
                title = raw.get("title", "") or raw.get("topic", "")
                if title:
                    lines.append(f"# {title}\n")
                content = raw.get("content", raw.get("sections", raw.get("text", "")))
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            sec_title = item.get("section_title", item.get("title", ""))
                            sec_body = item.get("content", item.get("text", ""))
                            if sec_title:
                                lines.append(f"\n## {sec_title}\n")
                            if isinstance(sec_body, list):
                                for b in sec_body:
                                    lines.append(f"- {b}")
                            elif sec_body:
                                lines.append(str(sec_body))
                        elif isinstance(item, str):
                            lines.append(item)
                elif isinstance(content, str):
                    lines.append(content)
                conclusion = raw.get("conclusion", "")
                if conclusion:
                    lines.append(f"\n{conclusion}")
                state["final_answer"] = "\n".join(lines).strip()
            else:
                state["final_answer"] = str(raw).strip()
            logger.info("步骤 %d | 生成最终答案 | 长度=%d",
                        state["step_count"], len(state["final_answer"]))
        else:
            # 解析 action 和 action_input
            action = str(decision.get("action", "")).strip()
            action_input = decision.get("action_input", {})
            if not isinstance(action_input, dict):
                action_input = {"input": action_input}

            # ── 兜底提取参数：LLM 可能把参数放在顶层而非 action_input ──
            # 常见情况 1: {"action": "search", "query": "...", "source": "..."}
            # 常见情况 2: {"action": "search", "action_input": {"parameters": {...}}}
            # 常见情况 3: {"action": "search", "action_input": {"query": "..."}}
            if not action_input or all(v == "" for v in action_input.values()):
                top_level_params = {
                    k: v for k, v in decision.items()
                    if k not in ("action", "thought", "final_answer", "action_input", "observation")
                }
                if top_level_params:
                    action_input = top_level_params
                    logger.info("从顶层提取 action_input 参数: %s", str(top_level_params)[:120])

            # 如果 action_input 里只有 "parameters" 一个键，展开它
            if "parameters" in action_input and isinstance(action_input["parameters"], dict):
                action_input = action_input["parameters"]
                logger.info("展开 parameters 参数: %s", str(action_input)[:120])

            # ── 展开 LLM 常用的 `input` 嵌套 ──────────────────
            # LLM 常把参数包在 input 键里：{"action": "search", "input": {"query": "..."}}
            if "input" in action_input and isinstance(action_input["input"], dict):
                nested = action_input.pop("input")
                # 将嵌套的参数提到顶层，不覆盖已有键
                for k, v in nested.items():
                    if k not in action_input:
                        action_input[k] = v
                logger.info("展开 input 嵌套: %s", str(action_input)[:120])

            # ── 工具名合法性 + 搜索查询覆盖 ────────────────────
            # LLM（尤其小模型）经常编造不存在的工具名。
            # 先自动纠错，如果纠错无效再由工具执行报错+循环检测兜底。
            topic_raw = state.get("topic", "").strip()
            if action:
                corrected = self._autocorrect_tool_name(action)
                if corrected != action:
                    logger.warning("自动纠错工具名: %s → %s", action, corrected)
                    print(f"  [AGENT] 🔄 工具名自动纠错: '{action}' → '{corrected}'", flush=True, file=sys.stderr)
                    action = corrected

            # 搜索查询：允许 LLM 使用自己的 query（DeepSeek 可自主选英文关键词）
            # 仅在 LLM 未提供 query 时兜底使用用户主题
            if self._is_search_action(action):
                llm_query = action_input.get("query", "").strip()
                if not llm_query and topic_raw:
                    action_input["query"] = topic_raw
                    logger.info("LLM 未提供搜索词，兜底使用用户主题: %.80s", topic_raw[:80])
                elif llm_query:
                    logger.info("LLM 自主搜索词: %.80s", llm_query[:80])
                action_input.pop("year_from", None)

            state["current_action"] = action
            state["current_action_input"] = action_input
            state["is_final"] = False
            thought_preview = thought[:80] if thought else "(无思考)"
            print(f"  [AGENT] 🛠 第 {step_num}/{max_steps} 步 → 调用工具: {action}", flush=True, file=sys.stderr)
            print(f"  [AGENT]   ╰ 参数: {str(action_input)[:120]}", flush=True, file=sys.stderr)
            if thought:
                print(f"  [AGENT]   ╰ 思考: {thought_preview}", flush=True, file=sys.stderr)
            logger.info("步骤 %d | 决策: action=%s | thought=%.80s",
                        state["step_count"], action, thought)
        return state

    def _tool_node(self, state: AgentState) -> AgentState:
        """工具节点：执行选定的工具。"""
        action = state["current_action"]
        action_input = state["current_action_input"]
        
        step_num = state["step_count"]
        print(f"\n{'─'*60}", flush=True, file=sys.stderr)
        print(f"  [AGENT] 🔧 第 {step_num}/{self.max_steps} 步 — 正在执行: {action}", flush=True, file=sys.stderr)
        print(f"{'─'*60}", flush=True, file=sys.stderr)

        # 强制注入 year_from：将用户指定的年份覆盖到搜索工具的 action_input 中
        year_from = state.get("year_from", "")
        if year_from and self._is_search_action(action):
            action_input["year_from"] = year_from
            print(f"  [INJECT] 强制设置 {action} 的 year_from={year_from}", flush=True, file=sys.stderr)
        
        # ── 参数名归一化：LLM 经常用单数/错误名 ──
        # source → sources, limit → max_results, query → query (不变)
        param_aliases = {
            "source": "sources",
            "year": "year_from",
            "max_results": "limit",
        }
        for wrong, correct in param_aliases.items():
            if wrong in action_input and correct not in action_input:
                action_input[correct] = action_input.pop(wrong)
        # sources 应该是列表，LLM 可能给字符串
        if "sources" in action_input and isinstance(action_input["sources"], str):
            action_input["sources"] = [action_input["sources"]]
        
        # 执行工具
        try:
            logger.info("步骤 %d | 执行工具 %s | input=%s",
                        state["step_count"], action, str(action_input)[:200])
            observation = self.tools.run(action, action_input)
        except Exception as e:
            logger.error("工具 %s 执行失败 | error=%s", action, str(e), exc_info=True)
            raise

        state["current_observation"] = observation

        obs_preview = str(observation)[:80].replace('\n', ' ')
        print(f"  [AGENT] ✅ 第 {step_num}/{self.max_steps} 步 — {action} 执行完毕", flush=True, file=sys.stderr)
        print(f"  [AGENT]   ╰ 结果: {obs_preview}", flush=True, file=sys.stderr)

        # 记录这一步
        step = AgentStep(
            thought=state["current_thought"],
            action=action,
            action_input=action_input,
            observation=observation,
        )
        state["steps"].append(step)

        # ── 工具结果失败检测（不限工具类型） ─────────────────────
        # 如果工具返回了错误/空结果，注入系统提示引导 LLM 转向不同的操作。
        # 不区分搜索工具还是下载工具——任何工具连续失败都应干预。
        result_is_empty = self._result_is_empty(str(observation))
        result_is_error = self._result_is_error(str(observation))

        if result_is_empty or result_is_error:
            fail_type = "错误" if result_is_error else "空结果"
            consecutive_fail = 0
            for s in reversed(state["steps"]):
                if s.action is None:
                    continue
                # 同类型工具失败才算连续（搜索/下载/解析等不同类型互不影响）
                if self._result_is_empty(str(s.observation or "")) or self._result_is_error(str(s.observation or "")):
                    consecutive_fail += 1
                else:
                    break

            print(f"  [AGENT] ⚠ 第 {step_num}/{self.max_steps} 步 — {action} 返回{fail_type}", flush=True, file=sys.stderr)
            print(f"  [AGENT]   ╰ 连续失败次数: {consecutive_fail}", flush=True, file=sys.stderr)

            # 注入系统提示
            if consecutive_fail <= 3:
                guidance = (
                    f"⚠️ 工具「{action}」返回了{fail_type}，请换用其他操作或关键词。"
                    f"不要重复调用同一个失败的工具。"
                )
            else:
                guidance = (
                    f"⚠️ 工具「{action}」已连续 {consecutive_fail} 次返回{fail_type}，"
                    f"请停止尝试该操作，转为使用其他可用工具，或直接输出 final_answer 结束。"
                )
            state["steps"].append(AgentStep(
                thought="[系统提示] " + guidance,
                action=None,
            ))
            logger.info("步骤 %d | 工具失败检测(%d次连续) → 已注入提示",
                        state["step_count"], consecutive_fail)

            # ── 强制兜底结束 ──────────────────────────────────
            if consecutive_fail >= 5:
                logger.warning("步骤 %d | 连续%d次工具失败 → 标记强制结束",
                               state["step_count"], consecutive_fail)
                state["is_final"] = True
                state["final_answer"] = ""

        logger.info("步骤 %d | 工具完成 | observation_len=%d",
                    state["step_count"], len(str(observation)))
        return state

    def _result_is_empty(self, obs: str) -> bool:
        """检查工具返回结果是否为空（0 篇论文）或工具调用失败（不存在/出错）。"""
        # 搜索成功但返回 0 篇论文
        empty_patterns = [
            '"total_count": 0',   # json.dumps(indent=2)
            '"total_count":0',    # json.dumps(indent=None)
            '"papers": []',       # 空论文列表
            '"success": []',      # 无成功来源
        ]
        # 工具调用失败（编造的工具名）— 支持英文和中文消息
        not_found_patterns = ["Tool '", "not found", "未知工具调用"]
        # 工具返回错误（空查询、速率限制、超时等）
        error_pattern = '"error":'

        return (
            any(p in obs for p in empty_patterns)
            or any(p in obs for p in not_found_patterns)
            or error_pattern in obs
        )

    def _result_is_error(self, obs: str) -> bool:
        """检查工具返回结果是否明确标记为错误（与空结果区分）。
        
        有些工具返回的错误信息不含 `"error":` JSON 格式（如工具执行异常），
        需要额外检测 HTTP 错误/异常描述等模式。
        """
        error_patterns = [
            "HTTP Error", "Timeout", "timeout", "速率限制",
            "Rate limit", "429", "500", "503",
            "Connection", "connection", "Read timed out",
            "File I/O error", "No such file",
        ]
        return any(p in obs for p in error_patterns)

    def _is_search_action(self, action: str) -> bool:
        """判断是否为搜索类工具。"""
        action_lower = action.lower()
        search_keywords = {"search", "retriev", "arxiv", "scholar"}
        return any(kw in action_lower for kw in search_keywords)

    def _autocorrect_tool_name(self, name: str) -> str:
        """自动纠错 LLM 编造的工具名为真实工具名。
        
        qwen2:7b 等小模型经常记不住 prompt 中的工具名，
        会编造 download_papers、search_arxiv 等变体。
        这里做模糊匹配：优先精确匹配，失败则尝试关键词映射。
        LLM 对此完全无感知。
        """
        # 精确匹配 → 直接用
        if name in self.tools._tools:
            return name

        name_lower = name.lower()

        # 关键词 → 真实工具名映射（内部自动纠错，不暴露给 LLM）
        correction_map: list[tuple[list[str], str | None]] = [
            # 下载类变体（download_papers、save_papers、fetch 等）
            (["download", "save", "fetch"], "paper_downloader"),
            # 搜索类变体（search_arxiv → arxiv_search 等）
            (["search_arxiv", "arxivsearch", "arxiv-search"], "arxiv_search"),
            (["search_semantic", "semanticsearch", "semantic-search"], "semantic_scholar"),
            (["search_google", "googlescholar", "google-search"], "google_scholar"),
            (["search_paper", "searchpaper", "search_for_paper", "search_for"], "multi_source_search"),
            # 解析类变体
            (["parse_pdf", "pdf_extract", "extract_pdf"], "pdf_parser"),
            # 分块类变体
            (["chunk_text", "split_text", "text_split"], "text_chunker"),
            # 存储类变体
            (["vector_search", "store_vector", "chroma_search"], "vector_store"),
            # 检索类变体
            (["hybrid_search", "rag_search", "retrieve"], "advanced_search"),
            # 分析/总结类变体 — 提示应使用 final_answer
            (["analyze", "summarize_result", "organize"], None),
            # 停止类变体 — 返回 None 表示不纠错，让失败检测兜底
            (["stop_search", "cancel", "exit", "stop"], None),
        ]

        for keywords, target in correction_map:
            if any(kw in name_lower for kw in keywords):
                if target is None:
                    return name
                return target

        # ── 兜底：含 "search" 或 "find" 的未知工具名 → multi_source_search
        # 覆盖任何 LLM 编造的搜索类变体（search_for_paper、search_paper等）
        if "search" in name_lower or "find" in name_lower:
            return "multi_source_search"

        return name

    def _build_fallback_review(self, topic: str, steps: list[AgentStep]) -> str:
        """LLM 不可用时，基于已搜索到的论文生成兜底答案。"""
        # 从 steps 中提取所有论文标题
        titles: list[str] = []
        for s in steps:
            if not s.observation:
                continue
            obs = str(s.observation)
            try:
                data = json.loads(obs)
                papers = data.get("papers", data.get("results", []))
                if isinstance(papers, list):
                    for p in papers:
                        t = p.get("title", "") if isinstance(p, dict) else str(p)
                        if t and t not in titles:
                            titles.append(t)
            except (json.JSONDecodeError, TypeError):
                pass

        parts = [f"## 研究背景\n\n{topic}是当前计算机视觉与智能交通领域的重要研究方向。\n"]
        if titles:
            parts.append("## 相关文献\n\n")
            for i, t in enumerate(titles, 1):
                parts.append(f"- [{i}] {t}\n")
        parts.append(f"\n共检索到 {len(titles)} 篇相关文献。由于 LLM 服务暂不可用，"
                     "无法自动生成完整综述文本。请稍后重试。\n")
        return "".join(parts)

    def _end_node(self, state: AgentState) -> AgentState:
        """结束节点：记录最终步骤（不包含 action）。"""
        print(f"\n{'='*60}", flush=True, file=sys.stderr)
        print(f"  [AGENT] 🏁 Agent 结束 | 总步骤数={state['step_count']}", flush=True, file=sys.stderr)
        if state.get("final_answer"):
            print(f"  [AGENT] 📝 最终答案长度: {len(state['final_answer'])} 字符", flush=True, file=sys.stderr)
        print(f"{'='*60}\n", flush=True, file=sys.stderr)
        # 记录最后的思考步骤
        step = AgentStep(thought=state["current_thought"])
        state["steps"].append(step)
        logger.info("Agent 结束 | 总步骤数=%d", state["step_count"])
        return state
    def _should_end(self, state: AgentState) -> str:
        """条件判断：是否应该结束循环。

        返回值：
            "end"   → 结束整个流程
            "tool"  → 执行当前选择的工具
            "think" → 跳过工具执行，直接回到思考节点（用于强制重试）
        """
        # 如果达到最大步数，强制结束
        if state["step_count"] >= self.max_steps:
            return "end"

        # ── 连续空搜索/错误过多 → 强制兜底结束 ─────────────────
        consecutive_empty = self._count_consecutive_empty_searches(state)
        if consecutive_empty >= 5:
            logger.warning("步骤 %d | 连续%d次空搜索/错误 → 强制结束",
                           state["step_count"], consecutive_empty)
            return "end"

        # ── 连续重复的失败工具调用 → 强制兜底结束 ────────────
        # LLM 可能卡在调用同一个不存在的工具（如 download_pdfs）上循环，
        # 即使不是搜索工具，连续失败也应强制结束。
        consecutive_fail = self._count_consecutive_identical_failures(state)
        if consecutive_fail >= 3:
            logger.warning("步骤 %d | 连续%d次相同失败动作 → 强制结束",
                           state["step_count"], consecutive_fail)
            return "end"

        # 如果 LLM 想返回最终答案，先检查是否过早放弃
        if state["is_final"]:
            search_count = sum(
                1 for s in state["steps"]
                if s.action and "search" in s.action.lower()
            )
            # 如果搜索次数 < 2 且最后一步观察是空结果 → 判定为过早放弃，强制继续尝试
            if search_count < 2 and self._last_search_was_empty(state):
                logger.warning(
                    "检测到过早放弃：仅尝试 %d 次搜索且结果为空，"
                    "覆盖 LLM 的 final_answer 决策，强制继续搜索",
                    search_count
                )
                state["is_final"] = False
                state["final_answer"] = ""
                state["current_action"] = ""
                state["current_action_input"] = {}
                state["current_thought"] = (
                    "⚠️ 上一步搜索返回了 0 篇论文，但仅仅尝试一次搜索就放弃为时过早。"
                    "请换用不同的关键词（如英文关键词）重新搜索，至少尝试 2~3 种不同查询后再做决定。"
                )
                # 将提示注入到 steps 中，这样 LLM 在下一轮 scratchpad 里能看到这条消息
                state["steps"].append(AgentStep(
                    thought="[系统提示] " + state["current_thought"]
                ))
                return "think"  # 跳过 tool 节点，直接回到 think 重新决策

            return "end"

        return "tool"

    def _last_search_was_empty(self, state: AgentState) -> bool:
        """检查最后一步搜索是否返回了 0 篇论文或全部来源均失败。"""
        steps = state["steps"]
        if not steps:
            return False
        last_step = steps[-1]
        if not last_step.observation:
            return False
        obs = str(last_step.observation)
        # 检查多种观测格式：total_count 为 0、papers 为空数组、所有来源失败
        patterns = [
            '"total_count": 0',      # json.dumps(indent=2) 格式
            '"total_count":0',       # json.dumps(indent=None) 格式
            '"papers": []',          # 空论文列表
            '"success": []',         # 无成功来源
        ]
        not_found = any(p in obs for p in ["Tool '", "not found", "未知工具调用"])
        is_error = '"error":' in obs
        is_empty = any(p in obs for p in patterns) or not_found or is_error
        logger.debug("_last_search_was_empty: action=%s | is_empty=%s | obs_preview=%s",
                     last_step.action, is_empty, obs[:120])
        return is_empty

    def _count_consecutive_empty_searches(self, state: AgentState) -> int:
        """从 steps 末尾向前追溯，统计连续空搜索/错误结果的步数（跳过系统提示步骤）。"""
        steps = state["steps"]
        count = 0
        for s in reversed(steps):
            if s.action is None:
                continue  # 跳过系统提示步骤
            if not self._is_search_action(s.action or ""):
                break
            if self._result_is_empty(str(s.observation or "")):
                count += 1
            else:
                break
        return count

    def _count_consecutive_identical_failures(self, state: AgentState) -> int:
        """统计连续相同工具调用失败的次数（任意工具类型）。
        
        用于检测 LLM 卡在重复调用同一不存在的工具（如 download_pdfs）的循环。
        只统计 steps 中已执行的步骤（有 observation 的），返回连续相同 action 且
        结果视为失败的次数。
        """
        steps = state["steps"]
        count = 0
        last_action = None
        for s in reversed(steps):
            if s.action is None:
                continue
            if last_action is None:
                last_action = s.action
            if s.action != last_action:
                break
            if self._result_is_empty(str(s.observation or "")):
                count += 1
            else:
                break
        return count

    def run(self, topic: str, year_from: str = "") -> AgentResult:
        """执行 Agent：调用 LangGraph 状态图处理输入。
        
        Args:
            topic: 研究主题
            year_from: 年份过滤条件，只搜索该年份之后的文献（空字符串表示不限制）
        """
        effective_year = year_from or self.year_from
        initial_state = AgentState(
            topic=topic,
            year_from=effective_year,
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
            # 达到最大步数但 LLM 未返回 final_answer → 强制 LLM 根据已有信息生成综述
            logger.warning("达到最大步数 %d 但未生成 final_answer，强制 LLM 汇总", self.max_steps)
            scratchpad = self._build_scratchpad(final_state["steps"])
            forced_prompt = (
                f"## 研究主题\n{final_state['topic']}\n\n"
                f"## 已收集到的信息\n"
                f"{scratchpad if scratchpad else '（暂无搜索结果，请基于已有知识回答）'}\n\n"
                f"## 指令\n"
                f"由于达到了最大执行步数，无法继续搜索新信息。请**基于以上已收集的信息**，"
                f"生成一份完整的文献综述。要求：\n"
                f"1. 如果没有任何搜索结果，请注明「搜索受限，以下内容基于已有知识编写」\n"
                f"2. 格式规范，包含引言、主体、结论\n"
                f"3. 引用已有论文信息\n"
                f"4. 用中文撰写，字数不少于 500 字\n\n"
                f"直接输出综述正文，不要额外解释。"
            )
            try:
                forced_answer = self.llm.chat(
                    system_prompt="你是一个专业的文献综述写作助手。请根据已提供的信息，生成一份结构完整的文献综述。",
                    user_prompt=forced_prompt,
                    temperature=0.3,
                )
                if forced_answer and forced_answer.strip():
                    final_answer = forced_answer.strip()
                    logger.info("强制汇总成功 | 长度=%d", len(final_answer))
                else:
                    raise ValueError("LLM 返回为空")
            except Exception as e:
                logger.error("强制汇总失败 | error=%s", str(e))
                final_answer = (
                    f"（达到最大步数 {self.max_steps}，未能生成完整综述。以下为搜索摘要）\n\n"
                    f"研究主题：{final_state['topic']}\n"
                    f"已执行步骤数：{len(final_state['steps'])}\n\n"
                    + (scratchpad[:2000] if scratchpad else "无搜索结果")
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
        
        处理 LLM 返回的 JSON 中可能包含的未转义字符（如 final_answer
        中的 markdown 含有未转义的双引号等）。使用多种策略：
        1. 标准 JSON 解析（strict=False）
        2. 宽松的手动字段提取（处理 final_answer 中未转义的双引号）
        """
        if raw_response is None or not raw_response.strip():
            raise ValueError(f"LLM returned empty response: {raw_response}")

        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.removeprefix("json").strip()

        # 策略1：尝试标准 JSON 解析
        try:
            from json import JSONDecoder
            return JSONDecoder(strict=False).decode(cleaned)
        except json.JSONDecodeError:
            pass

        # 策略2：手动提取字段（处理 final_answer 文本中含未转义双引号的情况）
        result: dict[str, Any] = {}

        # 提取 thought 字段（位于 "thought": "..." 之间）
        import re
        thought_match = re.search(r'("thought"\s*:\s*)"((?:[^"\\]|\\.)*)"', cleaned)
        if thought_match:
            thought_str = thought_match.group(2)
            # 解码转义字符
            thought_str = thought_str.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')
            result["thought"] = thought_str

        # 提取 final_answer 字段（可能是字符串 Markdown 或 JSON 对象）
        if '"final_answer"' in cleaned or '"final_answer":' in cleaned:
            prefix = '"final_answer":'
            idx_fa = cleaned.index(prefix) + len(prefix)
            fa_stripped = cleaned[idx_fa:].lstrip()

            if fa_stripped.startswith("{"):
                # final_answer 是 JSON 对象 → 提取 {} 内容
                brace_depth = 0
                brace_end = 0
                for i, ch in enumerate(fa_stripped):
                    if ch == "{":
                        brace_depth += 1
                    elif ch == "}":
                        brace_depth -= 1
                    if brace_depth == 0:
                        brace_end = i + 1
                        break
                if brace_end > 0:
                    try:
                        result["final_answer"] = json.loads(fa_stripped[:brace_end])
                    except json.JSONDecodeError:
                        result["final_answer"] = fa_stripped[:brace_end]
            else:
                # final_answer 是字符串 → 提取引号内的内容
                idx_quote = cleaned.index('"', idx_fa) + 1  # 跳过开头的 "
                # 从字符串末尾或最后 } 之前向前找未转义的闭合引号
                closing_brace = cleaned.rfind("}")
                if closing_brace == -1:
                    closing_brace = len(cleaned)
                quote_pos = closing_brace - 1
                while quote_pos >= idx_quote:
                    if cleaned[quote_pos] == '"' and (quote_pos == 0 or cleaned[quote_pos - 1] != '\\'):
                        break
                    quote_pos -= 1
                if quote_pos > idx_quote:
                    result["final_answer"] = cleaned[idx_quote:quote_pos]
        
        # 提取 action 字段
        action_match = re.search(r'("action"\s*:\s*)"((?:[^"\\]|\\.)*)"', cleaned)
        if action_match:
            result["action"] = action_match.group(2)

        # 提取 action_input 字段（可能是 { } 包围的 JSON 对象）
        if '"action_input"' in cleaned:
            try:
                # 找到 action_input 值的起点
                prefix2 = '"action_input":'
                idx2 = cleaned.index(prefix2) + len(prefix2)
                # 尝试提取 { } 中的内容
                brace_start = cleaned.index('{', idx2)
                brace_depth = 0
                brace_end = brace_start
                for i, ch in enumerate(cleaned[brace_start:], start=brace_start):
                    if ch == '{':
                        brace_depth += 1
                    elif ch == '}':
                        brace_depth -= 1
                    if brace_depth == 0:
                        brace_end = i + 1
                        break
                action_input_str = cleaned[brace_start:brace_end]
                result["action_input"] = json.loads(action_input_str)
            except (ValueError, json.JSONDecodeError):
                pass

        # 验证解析结果
        if "thought" not in result:
            result["thought"] = ""  # thought 可选，默认为空

        # 警告：LLM 不应输出 observation 字段（这是工具返回的）
        if "observation" in result:
            logger.warning(
                "LLM 返回了不应出现的 observation 字段，已忽略 | thought=%.40s",
                result.get("thought", "")[:40],
            )
            del result["observation"]

        if "final_answer" not in result and "action" not in result:
            # 尝试 final_answer 的最后一种检测方式：找 "final_answer": 后跟一个 "
            fallback_match = re.search(r'"final_answer"\s*:\s*"(.+)', cleaned, re.DOTALL)
            if fallback_match:
                raw_val = fallback_match.group(1)
                # 去掉末尾的 "} 或 " }
                raw_val = raw_val.rstrip()
                if raw_val.endswith('"'):
                    raw_val = raw_val[:-1]
                result["final_answer"] = raw_val.strip()
            else:
                raise ValueError(
                    f"LLM must return valid JSON with 'final_answer' or 'action', got: {raw_response[:500]}..."
                )

        return result
