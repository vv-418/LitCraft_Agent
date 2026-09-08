# 作用：使用 LangGraph 实现是一个一步步推进的 Agent。
# 根据取址也是对的特征：LangGraph 提供了状态图管理、条件判断、自动流程控制
# 简化了原来的 ReAct 手写循环。
import json
import logging
import re
import sys
from typing import Any, Callable, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.constants import START

from agent.memory import ResearchMemory
from agent.models import AgentResult, AgentStep
from agent.prompts import (
    JSON_REPAIR_PROMPT,
    REACT_SYSTEM,
    REACT_SYSTEM_NATIVE,
    REVIEW_MIN_CHARS,
    REVIEW_SECTIONS,
    WRITER_SECTION_SYSTEM,
    WRITER_SYSTEM,
    build_react_user,
    build_section_writer_user,
    build_writer_user,
    format_references,
    review_meets_standard,
    section_heading_md,
    section_number,
    section_search_query,
)
from utils.review_format import ensure_section_heading, polish_review_markdown, strip_inline_bibliographies
from llm_client import LLMTurn, LitCraftAgentsLLM
from tools.registry import ToolRegistry
from utils.logger import get_logger

logger = get_logger("agent")


class AgentState(TypedDict):
    """LangGraph 状态定义：记录 Agent 执行过程中的所有信息。"""
    topic: str
    year_from: str
    per_source_limit: int
    final_limit: int
    steps: list[AgentStep]
    current_thought: str
    current_action: str
    current_action_input: dict
    current_observation: str
    is_final: bool
    final_answer: str
    step_count: int
    memory: dict


class LangGraphAgent:
    """基于 LangGraph 的 Agent，使用状态图管理执行流程。"""

    def __init__(
        self,
        llm: LitCraftAgentsLLM,
        tools: ToolRegistry,
        max_steps: int = 6,
        temperature: float = 0,
        year_from: str = "",
        on_step: Callable[[AgentStep], None] | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """初始化 Agent，构建 LangGraph 状态图。"""
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.temperature = temperature
        self.year_from = year_from
        self.on_step = on_step
        self.on_progress = on_progress
        
        # 构建 LangGraph 状态图
        self.graph = self._build_graph()

    def _emit_step(self, state: AgentState, step: AgentStep) -> None:
        """追加一步并回调（供前端实时进度）。"""
        state["steps"].append(step)
        if self.on_step is None:
            return
        try:
            self.on_step(step)
        except Exception:
            logger.exception("on_step 回调失败（忽略，不影响主流程）")

    def _emit_progress(self, *, thought: str, action: str, action_input: dict) -> None:
        """工具开始执行时通知前端（不写入 Agent scratchpad）。"""
        if self.on_progress is None:
            return
        try:
            self.on_progress({
                "thought": thought or "",
                "action": action,
                "action_input": action_input or {},
            })
        except Exception:
            logger.exception("on_progress 回调失败（忽略）")

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

        mem = ResearchMemory(state.get("memory"))
        native_ok = not getattr(self.llm, "_tools_unsupported", False)
        user_prompt = build_react_user(
            topic=state["topic"],
            tools=self.tools,
            scratchpad=self._build_scratchpad(state["steps"]),
            year_from=state["year_from"],
            per_source_limit=int(state.get("per_source_limit") or 10),
            final_limit=int(state.get("final_limit") or 10),
            memory_block=mem.render(
                per_source_limit=int(state.get("per_source_limit") or 10),
                final_limit=int(state.get("final_limit") or 10),
            ),
            native_tools=native_ok,
        )
        try:
            decision = self._decide(user_prompt, native_ok=native_ok)
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
            # 解析 action 和 action_input（兼容 OpenAI 风格 name/parameters）
            decision = self._normalize_tool_decision(decision)
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
                    if k not in (
                        "action", "thought", "final_answer", "action_input",
                        "observation", "name", "parameters", "tool", "arguments",
                    )
                }
                if top_level_params:
                    action_input = top_level_params
                    logger.info("从顶层提取 action_input 参数: %s", str(top_level_params)[:120])

            # 如果 action_input 里只有 "parameters"/"arguments" 一个键，展开它
            for nest_key in ("parameters", "arguments"):
                if nest_key in action_input and isinstance(action_input[nest_key], dict):
                    action_input = action_input[nest_key]
                    logger.info("展开 %s 参数: %s", nest_key, str(action_input)[:120])
                    break

            # ── 展开 LLM 常用的 `input` 嵌套 ──────────────────
            # LLM 常把参数包在 input 键里：{"action": "search", "input": {"query": "..."}}
            if "input" in action_input and isinstance(action_input["input"], dict):
                nested = action_input.pop("input")
                # 将嵌套的参数提到顶层，不覆盖已有键
                for k, v in nested.items():
                    if k not in action_input:
                        action_input[k] = v
                logger.info("展开 input 嵌套: %s", str(action_input)[:120])

            topic_raw = state.get("topic", "").strip()
            if action:
                corrected = self._resolve_tool_name(action)
                if corrected != action:
                    logger.info("工具名映射: %s → %s", action, corrected)
                    action = corrected
            if action.lower() in ("final_answer", "finish"):
                state["is_final"] = True
                state["final_answer"] = "检索结束，请按体例成稿"
                print(f"  [AGENT] ✅ 第 {step_num}/{max_steps} 步 → 结束检索", flush=True, file=sys.stderr)
                return state

            # 检索词与篇数以用户设置为准：原生 FC 常会自填短词 query 和错误的 5
            if self._is_search_action(action):
                mem_now = ResearchMemory(state.get("memory"))
                given_q = str(action_input.get("query") or "").strip()
                first_search = not mem_now.queries
                if topic_raw and (first_search or not given_q):
                    if given_q and given_q != topic_raw:
                        print(
                            f"  [AGENT] 第一次搜索忽略模型自编 query「{given_q[:40]}」，改用主题",
                            flush=True,
                            file=sys.stderr,
                        )
                        logger.info("覆盖自编 query: %s → %s", given_q, topic_raw)
                    action_input["query"] = topic_raw
                user_year = str(state.get("year_from") or "").strip()
                if user_year.isdigit():
                    if str(action_input.get("year_from") or "").strip() != user_year:
                        logger.info("年份以用户设置为准: %s → %s", action_input.get("year_from"), user_year)
                    action_input["year_from"] = user_year
                else:
                    invented = action_input.pop("year_from", None)
                    action_input.pop("year", None)
                    if invented not in (None, "", 0, "0"):
                        print(
                            f"  [AGENT] 用户未限制年份，已忽略模型自填的 year_from={invented}",
                            flush=True,
                            file=sys.stderr,
                        )
                        logger.info("忽略模型自填 year_from=%s（用户未限制年份）", invented)
                per_source = int(state.get("per_source_limit") or 0)
                final_n = int(state.get("final_limit") or 0)
                if "multi_source" in (action or "").lower():
                    if final_n > 0:
                        action_input["limit"] = final_n
                    if per_source > 0:
                        action_input["per_source_limit"] = per_source
                elif per_source > 0:
                    action_input["limit"] = per_source

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

    def _decision_from_native(self, turn: LLMTurn) -> dict[str, Any]:
        name = (turn.tool_name or "").strip()
        args = turn.tool_args if isinstance(turn.tool_args, dict) else {}
        thought = (turn.content or "").strip() or str(args.get("reason") or args.get("thought") or "")
        if name.lower() in ("final_answer", "finish", "end"):
            return {
                "thought": thought or str(args.get("reason") or "检索结束"),
                "final_answer": "检索结束，请按体例成稿",
            }
        return {"thought": thought, "action": name, "action_input": args}

    def _parse_or_repair(self, raw_response: str) -> dict[str, Any]:
        try:
            return self._parse_decision(raw_response)
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning("决策不是 JSON，进行一次纠错重试 | error=%s | preview=%s", e, (raw_response or "")[:160])
            print("  [AGENT] ⚠ 未得到 JSON，按协议纠错重试一次…", flush=True, file=sys.stderr)
            try:
                repaired = self.llm.chat(
                    system_prompt=REACT_SYSTEM,
                    user_prompt=JSON_REPAIR_PROMPT + (raw_response or "")[:400],
                    temperature=0,
                    max_tokens=1024,
                    json_mode=True,
                    stop=["下一步指示", "请告知您下一步"],
                )
                return self._parse_decision(repaired)
            except Exception as e2:
                logger.warning("纠错后仍不是 JSON | error=%s", e2)
                print("  [AGENT] ⚠ 纠错失败，本步记为无效", flush=True, file=sys.stderr)
                return {
                    "thought": (raw_response or "")[:200],
                    "action": "",
                    "action_input": {},
                }

    def _decide(self, user_prompt: str, native_ok: bool) -> dict[str, Any]:
        """原生 function calling 优先；接口不支持或未给出 tool_call 时走 JSON。"""
        if native_ok:
            turn = self.llm.chat_turn(
                system_prompt=REACT_SYSTEM_NATIVE,
                user_prompt=user_prompt,
                temperature=self.temperature,
                max_tokens=1024,
                tools=self.tools.openai_tools(),
                stop=["下一步指示", "请告知您下一步"],
            )
            if turn.tool_name:
                print(
                    f"  [AGENT] 原生 tool_call → {turn.tool_name}",
                    flush=True,
                    file=sys.stderr,
                )
                return self._decision_from_native(turn)
            if turn.content:
                return self._parse_or_repair(turn.content)
        raw = self.llm.chat(
            system_prompt=REACT_SYSTEM,
            user_prompt=user_prompt,
            temperature=self.temperature,
            max_tokens=1024,
            json_mode=True,
            stop=["下一步指示", "请告知您下一步"],
        )
        return self._parse_or_repair(raw)

    def _tool_node(self, state: AgentState) -> AgentState:
        """工具节点：执行选定的工具。"""
        action = state["current_action"]
        action_input = state["current_action_input"]
        
        step_num = state["step_count"]
        print(f"\n{'─'*60}", flush=True, file=sys.stderr)
        print(f"  [AGENT] 🔧 第 {step_num}/{self.max_steps} 步 — 正在执行: {action}", flush=True, file=sys.stderr)
        print(f"{'─'*60}", flush=True, file=sys.stderr)

        if not action:
            observation = json.dumps(
                {
                    "error": "missing_action",
                    "message": "这一步没有 action。请返回带 action 的工具调用，或返回 final_answer。",
                },
                ensure_ascii=False,
            )
            state["current_observation"] = observation
            self._emit_step(state, AgentStep(
                thought=state.get("current_thought") or "",
                action="",
                action_input=action_input,
                observation=observation,
            ))
            return state

        # 参数名别名（不改变语义，只对齐 schema）
        param_aliases = {
            "source": "sources",
            "year": "year_from",
            "max_results": "limit",
        }
        for wrong, correct in param_aliases.items():
            if wrong in action_input and correct not in action_input:
                action_input[correct] = action_input.pop(wrong)
        if "sources" in action_input and isinstance(action_input["sources"], str):
            action_input["sources"] = [action_input["sources"]]

        # 同一 query 或已达互补检索上限时跳过，避免 20 步全在搜同一句话
        if self._is_web_search_action(action):
            mem = ResearchMemory(state.get("memory"))
            query = str(action_input.get("query") or "").strip()
            allowed, reason = mem.can_web_search(query)
            if not allowed:
                observation = json.dumps(
                    {
                        "status": "skipped_redundant_search",
                        "paper_count": len(mem.papers),
                        "queries": mem.queries,
                        "message": reason,
                    },
                    ensure_ascii=False,
                )
                print(f"  [AGENT] ⏭ 跳过联网搜索 | {reason[:80]}", flush=True, file=sys.stderr)
                state["current_observation"] = observation
                self._emit_step(state, AgentStep(
                    thought=state.get("current_thought") or "",
                    action=action,
                    action_input=action_input,
                    observation=observation,
                ))
                return state

        # 执行工具
        self._emit_progress(
            thought=state.get("current_thought") or "",
            action=action,
            action_input=action_input,
        )
        try:
            logger.info("步骤 %d | 执行工具 %s | input=%s",
                        state["step_count"], action, str(action_input)[:200])
            observation = self.tools.run(action, action_input)
        except Exception as e:
            logger.error("工具 %s 执行失败 | error=%s", action, str(e), exc_info=True)
            observation = json.dumps({"error": str(e), "action": action}, ensure_ascii=False)

        state["current_observation"] = observation

        obs_preview = str(observation)[:80].replace('\n', ' ')
        print(f"  [AGENT] ✅ 第 {step_num}/{self.max_steps} 步 — {action} 执行完毕", flush=True, file=sys.stderr)
        print(f"  [AGENT]   ╰ 结果: {obs_preview}", flush=True, file=sys.stderr)

        self._emit_step(state, AgentStep(
            thought=state["current_thought"],
            action=action,
            action_input=action_input,
            observation=observation,
        ))
        mem = ResearchMemory(state.get("memory"))
        mem.ingest(action, action_input or {}, observation)
        state["memory"] = mem.to_dict()
        logger.info(
            "步骤 %d | 工具完成 | observation_len=%d | memory_papers=%d",
            state["step_count"], len(str(observation)), len(mem.papers),
        )
        return state

    def _is_web_search_action(self, action: str) -> bool:
        return (action or "") in {
            "multi_source_search", "arxiv_search", "semantic_scholar", "google_scholar",
            "openalex", "crossref", "europe_pmc",
        }

    def _is_search_action(self, action: str) -> bool:
        """联网搜索类工具（不含本地 advanced_search）。"""
        return self._is_web_search_action(action)

    @staticmethod
    def _compact_observation_for_prompt(action: str, obs: str) -> str:
        """搜索观察太大时只保留篇数、题录和入库信息，避免截断后模型以为没搜到。"""
        if not obs or not action:
            return obs or ""
        raw = obs.strip()
        if not raw.startswith("{"):
            brace = raw.find("{")
            if brace < 0:
                return obs
            raw = raw[brace:]
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return obs
        if not isinstance(data, dict):
            return obs
        papers = data.get("papers") or data.get("results") or data.get("documents") or []
        if not isinstance(papers, list) or not papers:
            return obs
        compact_papers = []
        for paper in papers[:15]:
            if not isinstance(paper, dict):
                continue
            compact_papers.append({
                "title": paper.get("title") or paper.get("paper_title"),
                "year": paper.get("year"),
                "url": paper.get("url") or paper.get("pdf_url"),
                "local_path": paper.get("local_path"),
                "abstract": str(paper.get("abstract") or paper.get("snippet") or paper.get("text") or "")[:180],
            })
        compact = {
            "query": data.get("query"),
            "total_count": data.get("total_count", len(compact_papers)),
            "indexed_chunks": data.get("indexed_chunks"),
            "source_summary": data.get("source_summary"),
            "papers": compact_papers,
        }
        if data.get("status"):
            compact["status"] = data.get("status")
            compact["message"] = data.get("message")
        return json.dumps(compact, ensure_ascii=False)

    @staticmethod
    def _normalize_tool_decision(decision: dict[str, Any]) -> dict[str, Any]:
        """把 OpenAI / 通用 function-calling 风格归一成 action + action_input。

        兼容：
        - {"name": "tool", "parameters": {...}}
        - {"tool": "tool", "arguments": {...}}
        - {"function": {"name": "...", "arguments": {...}}}
        """
        if not isinstance(decision, dict):
            return {}

        out = dict(decision)

        # OpenAI ChatCompletions tool call 嵌套
        fn = out.get("function")
        if isinstance(fn, dict):
            if not out.get("action") and fn.get("name"):
                out["action"] = fn.get("name")
            args = fn.get("arguments")
            if args is not None and not out.get("action_input"):
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"input": args}
                if isinstance(args, dict):
                    out["action_input"] = args

        if not out.get("action"):
            for key in ("name", "tool", "tool_name", "function_name"):
                val = out.get(key)
                if isinstance(val, str) and val.strip():
                    out["action"] = val.strip()
                    break

        if not out.get("action_input"):
            for key in ("parameters", "arguments", "args", "input"):
                val = out.get(key)
                if isinstance(val, dict):
                    out["action_input"] = val
                    break
                if isinstance(val, str) and val.strip():
                    try:
                        parsed = json.loads(val)
                        if isinstance(parsed, dict):
                            out["action_input"] = parsed
                            break
                    except Exception:
                        pass

        return out

    def _resolve_tool_name(self, name: str) -> str:
        """只做精确名和少数常见别名映射，不把任意 search* 吞成同一工具。"""
        if name in self.tools._tools:
            return name
        aliases = {
            "arxiv": "arxiv_search",
            "s2": "semantic_scholar",
            "semantic-scholar": "semantic_scholar",
            "google-scholar": "google_scholar",
            "scholar": "google_scholar",
            "multi_search": "multi_source_search",
            "hybrid_search": "advanced_search",
            "retrieve": "advanced_search",
            "download_pdf": "paper_downloader",
            "parse_pdf": "pdf_parser",
        }
        return aliases.get(name.lower().replace(" ", "_"), name)

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

        parts = [f"## 已收集到的题录\n\n主题：{topic}\n"]
        if titles:
            parts.append("\n")
            for i, t in enumerate(titles, 1):
                parts.append(f"- [{i}] {t}\n")
        else:
            parts.append("\n本次尚未从工具中获得论文题录。\n")
        parts.append("\n当前大模型服务不可用，无法继续生成综述正文。请稍后重试。\n")
        return "".join(parts)

    @staticmethod
    def _unwrap_markdown(text: str) -> str:
        t = (text or "").strip()
        if t.startswith("```"):
            t = re.sub(r"^```(?:markdown|md)?\s*", "", t, flags=re.I)
            t = re.sub(r"\s*```$", "", t)
        return t.strip()

    def _collect_papers_from_steps(self, steps: list[AgentStep]) -> list[dict[str, Any]]:
        """从工具观察中收集去重后的论文题录。"""
        seen: set[str] = set()
        papers: list[dict[str, Any]] = []
        for step in steps or []:
            if not step.observation:
                continue
            raw = str(step.observation).strip()
            if not raw.startswith("{"):
                brace = raw.find("{")
                if brace < 0:
                    continue
                raw = raw[brace:]
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(data, dict):
                continue
            items = data.get("papers") or data.get("results") or data.get("documents") or []
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or item.get("paper_title") or "").strip()
                if not title:
                    continue
                key = title.lower()
                if key in seen:
                    continue
                seen.add(key)
                papers.append(item)
        return papers

    def _collection_candidates(self, topic: str, mem: ResearchMemory) -> list[str]:
        names: list[str] = []
        for raw in list(mem.collections) + list(mem.queries) + [topic]:
            name = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff_\-]", "_", (raw or "").strip())[:50]
            if name and name not in names:
                names.append(name)
        return names

    def _retrieve_compose_passages(
        self,
        topic: str,
        papers: list[dict[str, Any]],
        mem: ResearchMemory,
        top_k: int = 10,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        """成稿前从 Chroma 取原文块；失败则返回空，Writer 退回摘要。"""
        search_query = (query or topic or "").strip() or topic
        if not papers:
            return []
        whitelist = {
            str(p.get("title") or p.get("paper_title") or "").strip().lower()
            for p in papers
            if str(p.get("title") or p.get("paper_title") or "").strip()
        }
        title_to_cite = {}
        for i, paper in enumerate(papers, 1):
            title = str(paper.get("title") or paper.get("paper_title") or "").strip()
            if title:
                title_to_cite[title.lower()] = str(i)

        raw_chunks: list[dict[str, Any]] = []
        for cname in self._collection_candidates(topic, mem):
            try:
                obs = self.tools.run("advanced_search", {
                    "collection_name": cname,
                    "query": search_query,
                    "top_k": top_k,
                    "threshold": 0.0,
                })
            except Exception as e:
                logger.warning("成稿检索失败 | collection=%s | error=%s", cname, e)
                continue
            try:
                data = json.loads(obs) if isinstance(obs, str) else obs
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(data, dict) or data.get("status") == "error":
                continue
            chunks = data.get("chunks") or []
            if isinstance(chunks, list) and chunks:
                raw_chunks = [c for c in chunks if isinstance(c, dict)]
                print(
                    f"  [AGENT] 成稿检索 '{search_query[:40]}' @ '{cname}' → {len(raw_chunks)} 个片段",
                    flush=True,
                    file=sys.stderr,
                )
                break

        passages: list[dict[str, Any]] = []
        seen_text: set[str] = set()
        for chunk in raw_chunks:
            if str(chunk.get("modality") or "text") == "image":
                continue
            meta = chunk.get("metadata") or {}
            if not isinstance(meta, dict):
                meta = {}
            title = str(
                meta.get("title")
                or meta.get("custom_title")
                or chunk.get("source")
                or meta.get("source")
                or ""
            ).strip()
            content = str(chunk.get("content") or "").strip()
            if not content:
                continue
            key = content[:160]
            if key in seen_text:
                continue
            if whitelist and title and title.lower() not in whitelist:
                hit = next((t for t in whitelist if t[:20] and t[:20] in title.lower()), "")
                if not hit:
                    continue
                matched = next(
                    (
                        str(p.get("title") or "").strip()
                        for p in papers
                        if str(p.get("title") or "").strip().lower() == hit
                    ),
                    "",
                )
                if matched:
                    title = matched
            seen_text.add(key)
            passages.append({
                "title": title or str(meta.get("source") or ""),
                "content": content,
                "cite": title_to_cite.get(title.lower(), ""),
            })
            if len(passages) >= top_k:
                break
        return passages

    def _compose_review(
        self,
        topic: str,
        steps: list[AgentStep],
        extra_papers: list[dict[str, Any]] | None = None,
        memory: ResearchMemory | None = None,
    ) -> str:
        """按节 retrieve-and-write。单节没搜到或写失败不影响其他节；不因过短整篇作废。"""
        papers = self._collect_papers_from_steps(steps)
        seen = {
            str(p.get("title") or p.get("paper_title") or "").strip().lower()
            for p in papers
        }
        for paper in extra_papers or []:
            title = str(paper.get("title") or paper.get("paper_title") or "").strip()
            if title and title.lower() not in seen:
                seen.add(title.lower())
                papers.append(paper)
        mem = memory or ResearchMemory()
        parts: list[str] = []
        for spec in REVIEW_SECTIONS:
            heading = str(spec.get("heading") or "")
            q = section_search_query(topic, spec)
            passages: list[dict[str, Any]] = []
            try:
                passages = self._retrieve_compose_passages(
                    topic, papers, mem, top_k=6, query=q,
                )
            except Exception as e:
                logger.warning("分节检索失败，本节改用摘要 | heading=%s | error=%s", heading, e)
            if not passages:
                print(
                    f"  [AGENT] 「{heading}」无专属片段，仍写本节（依据摘要）",
                    flush=True,
                    file=sys.stderr,
                )
            else:
                print(
                    f"  [AGENT] 撰写「{heading}」| 片段 {len(passages)} | query={q[:60]}",
                    flush=True,
                    file=sys.stderr,
                )
            user_prompt = build_section_writer_user(
                topic, spec, papers, passages, prior_sections=parts,
            )
            section_md = ""
            min_section = max(80, int(spec.get("target_chars") or 400) // 4)
            for attempt in (1, 2):
                try:
                    raw = self.llm.chat(
                        system_prompt=WRITER_SECTION_SYSTEM,
                        user_prompt=user_prompt,
                        temperature=0.4,
                        timeout=180,
                        max_tokens=2048,
                    )
                    section_md = self._unwrap_markdown(raw).strip()
                except Exception as e:
                    logger.warning(
                        "分节写作失败 | heading=%s | attempt=%s | error=%s",
                        heading, attempt, e,
                    )
                    section_md = ""
                if len(section_md) >= min_section:
                    break
                if attempt == 1:
                    print(
                        f"  [AGENT] 「{heading}」过短，只重写本节，其他节保留",
                        flush=True,
                        file=sys.stderr,
                    )
            idx = section_number(spec)
            heading_md = section_heading_md(spec)
            if len(section_md) < 20:
                section_md = (
                    f"{heading_md}\n\n"
                    f"（本节未生成正文，依据摘要：相关工作见题录白名单，原文片段不足。）"
                )
            section_md = strip_inline_bibliographies(section_md)
            section_md = ensure_section_heading(section_md, idx, heading)
            parts.append(section_md)

        body = "\n\n".join(parts).strip()
        refs = format_references(papers)
        return polish_review_markdown(f"{body}\n\n{refs}", topic=topic)

    def _end_node(self, state: AgentState) -> AgentState:
        """结束节点：记录最终步骤（不包含 action）。"""
        print(f"\n{'='*60}", flush=True, file=sys.stderr)
        print(f"  [AGENT] 🏁 Agent 结束 | 总步骤数={state['step_count']}", flush=True, file=sys.stderr)
        if state.get("final_answer"):
            print(f"  [AGENT] 📝 最终答案长度: {len(state['final_answer'])} 字符", flush=True, file=sys.stderr)
        print(f"{'='*60}\n", flush=True, file=sys.stderr)
        # 记录最后的思考步骤
        step = AgentStep(thought=state["current_thought"])
        self._emit_step(state, step)
        logger.info("Agent 结束 | 总步骤数=%d", state["step_count"])
        return state

    def _should_end(self, state: AgentState) -> str:
        """ReAct 循环出口：模型给出 final_answer，或达到步数上限。"""
        if state["step_count"] >= self.max_steps:
            return "end"
        if state["is_final"]:
            return "end"
        return "tool"

    def run(
        self,
        topic: str,
        year_from: str = "",
        per_source_limit: int = 10,
        final_limit: int = 10,
    ) -> AgentResult:
        """执行 Agent：调用 LangGraph 状态图处理输入。
        
        Args:
            topic: 研究主题
            year_from: 年份过滤条件，只搜索该年份之后的文献（空字符串表示不限制）
            per_source_limit: 每个学术源单次最多保留篇数
            final_limit: 多源合并后最终对外保留篇数
        """
        effective_year = year_from or self.year_from
        initial_state = AgentState(
            topic=topic,
            year_from=effective_year,
            per_source_limit=max(1, min(int(per_source_limit or 10), 50)),
            final_limit=max(1, min(int(final_limit or 10), 50)),
            steps=[],
            current_thought="",
            current_action="",
            current_action_input={},
            current_observation="",
            is_final=False,
            final_answer="",
            step_count=0,
            memory=ResearchMemory().to_dict(),
        )
        
        # 执行图
        final_state = self.graph.invoke(initial_state)
        
        # 有文献则单独成稿（不在 ReAct JSON 里写 3000 字，避免又慢又被超时重跑）
        final_answer = self._unwrap_markdown(final_state.get("final_answer") or "")
        mem = ResearchMemory(final_state.get("memory"))
        papers = mem.papers or self._collect_papers_from_steps(final_state["steps"])
        paper_count = len(papers)
        need_compose = paper_count > 0 or not review_meets_standard(final_answer, paper_count)
        if need_compose and paper_count > 0:
            logger.info("开始成稿 | papers=%d | draft_chars=%d", paper_count, len(final_answer))
            print(
                f"  [AGENT] 检索结束，正在撰写综述（约 {REVIEW_MIN_CHARS} 字，本地模型可能需要数分钟）…",
                flush=True, file=sys.stderr,
            )
            self._emit_progress(
                thought="检索已结束，正在按体例撰写综述正文，请稍候。",
                action="write_review",
                action_input={"min_chars": REVIEW_MIN_CHARS, "papers": paper_count},
            )
            try:
                rewritten = self._compose_review(
                    final_state["topic"],
                    final_state["steps"],
                    extra_papers=mem.papers,
                    memory=mem,
                )
                if rewritten:
                    final_answer = rewritten
                    logger.info("成稿完成 | 长度=%d", len(final_answer))
            except Exception as e:
                logger.error("成稿失败 | error=%s", str(e))
                if not final_answer:
                    scratchpad = self._build_scratchpad(final_state["steps"])
                    final_answer = (
                        f"（未能生成完整综述。）\n\n研究主题：{final_state['topic']}\n"
                        + (scratchpad[:2000] if scratchpad else "无工具观察")
                    )
        elif not final_answer:
            scratchpad = self._build_scratchpad(final_state["steps"])
            final_answer = (
                f"（未能生成完整综述。）\n\n研究主题：{final_state['topic']}\n"
                + (scratchpad[:2000] if scratchpad else "无工具观察")
            )
        
        return AgentResult(
            final_answer=final_answer,
            steps=final_state["steps"],
        )

    # ── 滑动窗口 + 全局摘要 ─────────────────────────────────
    # scratchpad 结构:「全局摘要 + 最近 WINDOW_SIZE 步」
    # 旧步骤滑出窗口后批量压缩为一段摘要,LLM 始终能感知全部搜索成果;
    # 最近 N 步完整保留,保证推理时有完整的近期上下文。
    _MAX_OBSERVATION_CHARS = 4000   # 近期步骤 observation 安全截断
    _WINDOW_SIZE = 6                # 完整保留的最近步数

    # scratchpad 字符上限由模型上下文窗口动态计算(1 token ≈ 2 字符,预留 30% 给 system prompt + 输出)
    @property
    def _MAX_SCRATCHPAD_CHARS(self) -> int:
        """根据当前 LLM 的上下文窗口动态计算 scratchpad 字符上限。"""
        max_tokens = self.llm.max_context_tokens  # 例: gpt-4o → 128000
        # 70% 分配给输入上下文, 1 token ≈ 2 中文字符 / 4 英文字符, 取 2.5 作为中英混合保守估计
        reserved_chars = int(max_tokens * 2.5 * 0.7)
        return reserved_chars

    @staticmethod
    def _summarize_steps(steps: list[AgentStep]) -> str:
        """将滑出窗口的一批历史步骤批量压缩为一段全局摘要。

        摘要包含:
        - 统计信息:总步数、搜索次数、使用过的工具
        - 论文发现:从 observation JSON 中提取的论文标题列表
        - 关键思路:最后 3 步的 thought
        """
        if not steps:
            return ""

        tools_used: list[str] = []
        seen_tools: set[str] = set()
        search_count = 0
        paper_titles: list[str] = []
        paper_ids: set[str] = set()

        for step in steps:
            # 工具统计
            if step.action and step.action not in seen_tools:
                tools_used.append(step.action)
                seen_tools.add(step.action)
            if step.action and "search" in (step.action or "").lower():
                search_count += 1

            # 从 observation JSON 中提取论文标题
            if step.observation:
                try:
                    data = json.loads(step.observation)
                except (json.JSONDecodeError, TypeError):
                    continue
                papers = (
                    data.get("papers")
                    or data.get("results")
                    or data.get("documents")
                    or []
                )
                if isinstance(papers, list):
                    for p in papers[:10]:
                        title = p.get("title") or ""
                        pid = p.get("arxiv_id") or p.get("paper_id") or title
                        if pid and pid not in paper_ids:
                            paper_ids.add(pid)
                            if title:
                                paper_titles.append(title)

        # 组装摘要文本
        parts: list[str] = []
        parts.append(
            f"已执行 {len(steps)} 步 | 工具: {', '.join(tools_used) if tools_used else '无'}"
        )
        if search_count:
            parts.append(f"搜索 {search_count} 次")

        if paper_titles:
            titles_display = "; ".join(paper_titles[:12])
            suffix = f" 等共{len(paper_titles)}篇" if len(paper_titles) > 12 else ""
            parts.append(f"已发现论文: {titles_display}{suffix}")

        # 保留最后 3 步的思考链
        key_thoughts = [s.thought for s in steps if s.thought][-3:]
        if key_thoughts:
            parts.append("关键思路: " + " → ".join(key_thoughts))

        return "\n".join(parts)

    def _build_scratchpad(self, steps: list[AgentStep]) -> str:
        """构建 scratchpad:「全局摘要 + 最近 WINDOW_SIZE 步完整内容」。

        - 窗口内(最近 N 步):完整保留 thought / action / observation
          observation 超过 _MAX_OBSERVATION_CHARS 时安全截断(兜底)
        - 窗口外(更早的步骤):批量压缩为一段全局摘要
        - 总量超过 _MAX_SCRATCHPAD_CHARS 时丢弃摘要中的论文列表(最终兜底)
        """
        if not steps:
            return ""

        lines: list[str] = []

        # ── 1. 旧步骤 → 全局摘要 ──
        window = self._WINDOW_SIZE
        if len(steps) > window:
            old_steps = steps[:-window]
            summary = self._summarize_steps(old_steps)
            if summary:
                summary_block = (
                    f"[全局摘要 — 已压缩前 {len(old_steps)} 步]\n{summary}"
                )
                lines.append(summary_block)
        else:
            summary_block = ""

        # ── 2. 最近 N 步完整保留 ──
        recent_steps = steps[-window:]
        total_chars = len("\n".join(lines)) + 1 if lines else 0

        for i, step in enumerate(recent_steps):
            obs = step.observation or ""
            if step.action:
                obs = self._compact_observation_for_prompt(step.action, obs)
            if len(obs) > self._MAX_OBSERVATION_CHARS:
                obs = (
                    obs[: self._MAX_OBSERVATION_CHARS]
                    + f"\n... [已截断,原始 {len(step.observation or '')} 字符]"
                )

            # 计算实际步骤编号(从 1 开始连续编号)
            step_num = len(steps) - len(recent_steps) + i + 1
            step_text = f"Step {step_num}"
            if step.thought:
                step_text += f"\nThought: {step.thought}"
            if step.action:
                step_text += f"\nAction: {step.action}"
                step_text += (
                    f"\nAction Input: {json.dumps(step.action_input, ensure_ascii=False)}"
                )
            if obs:
                step_text += f"\nObservation: {obs}"

            step_chars = len(step_text) + 1

            # 总量兜底:摘要+近期步骤超限时截断摘要
            if total_chars + step_chars > self._MAX_SCRATCHPAD_CHARS:
                # 尝试从摘要中截断论文列表,保留统计信息
                if summary_block and lines:
                    summary_short = summary_block.split("\n")[0] + "\n..." 
                    lines[0] = summary_short
                    total_chars = len("\n".join(lines)) + 1
                    if total_chars + step_chars > self._MAX_SCRATCHPAD_CHARS:
                        break  # 已到极限,停止追加
                else:
                    break

            lines.append(step_text)
            total_chars += step_chars

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

        # 模型偶尔在 JSON 前加对话口吻：从第一个 { 起解析
        brace = cleaned.find("{")
        if brace < 0:
            raise ValueError(f"LLM returned no JSON object: {raw_response[:200]}")
        if brace > 0:
            cleaned = cleaned[brace:]

        # 策略1：从第一个 { 解析一个对象；尾部闲聊忽略
        try:
            from json import JSONDecoder
            obj, _end = JSONDecoder(strict=False).raw_decode(cleaned)
            if isinstance(obj, dict):
                obj.pop("observation", None)
                if "final_answer" in obj or "action" in obj:
                    return obj
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
