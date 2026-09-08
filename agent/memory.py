"""研究记忆：工作记忆（近期步骤）之外，始终放进 Prompt 的结构化档案。

对齐常见 Agent 记忆分层：
- Core / working：本文件的研究档案（题录、已用 query、入库块数），每步更新、永不滑出窗口
- Episodic：langgraph 里最近若干步的 Thought/Action/Observation
- Archival：Chroma 全文块，用 advanced_search 按需召回
"""
from __future__ import annotations

import json
import re
from typing import Any


_WS = re.compile(r"\s+")
MAX_DISTINCT_QUERIES = 3


def normalize_query(query: str) -> str:
    return _WS.sub(" ", (query or "").strip().lower())


def _parse_obs(observation: Any) -> dict[str, Any] | None:
    if not observation:
        return None
    raw = observation if isinstance(observation, str) else str(observation)
    text = raw.strip()
    if not text.startswith("{"):
        brace = text.find("{")
        if brace < 0:
            return None
        text = text[brace:]
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _paper_key(paper: dict[str, Any]) -> str:
    title = str(paper.get("title") or paper.get("paper_title") or "").strip().lower()
    if title:
        return title
    return str(paper.get("arxiv_id") or paper.get("paper_id") or paper.get("url") or "").strip().lower()


class ResearchMemory:
    """跨步骤累积的研究档案（core memory）。"""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        src = data or {}
        self.queries: list[str] = list(src.get("queries") or [])
        self.query_norms: list[str] = list(src.get("query_norms") or [])
        self.papers: list[dict[str, Any]] = list(src.get("papers") or [])
        self.indexed_chunks: int = int(src.get("indexed_chunks") or 0)
        self.web_searches: int = int(src.get("web_searches") or 0)
        self.collections: list[str] = list(src.get("collections") or [])

    def to_dict(self) -> dict[str, Any]:
        return {
            "queries": self.queries,
            "query_norms": self.query_norms,
            "papers": self.papers,
            "indexed_chunks": self.indexed_chunks,
            "web_searches": self.web_searches,
            "collections": self.collections,
        }

    def seen_query(self, query: str) -> bool:
        norm = normalize_query(query)
        return bool(norm) and norm in self.query_norms

    def can_web_search(self, query: str) -> tuple[bool, str]:
        """是否允许再做一轮联网搜索。同一 query 跳过；有文献后最多 MAX_DISTINCT_QUERIES 个不同 query。"""
        q = (query or "").strip()
        if not q:
            return False, "搜索缺少 query。"
        if self.seen_query(q):
            return False, (
                f"研究记忆里已经用过同一检索词「{q}」。"
                "请换一个互补 query（中英、同义、子方向），或 advanced_search / final_answer。"
            )
        if self.papers and len(self.query_norms) >= MAX_DISTINCT_QUERIES:
            return False, (
                f"已用 {len(self.query_norms)} 个不同检索词、记忆中有 {len(self.papers)} 篇文献。"
                "请 advanced_search 取证据，或 final_answer 结束检索。"
            )
        return True, ""

    def ingest(self, action: str, action_input: dict[str, Any], observation: Any) -> None:
        action = (action or "").strip()
        data = _parse_obs(observation)
        if data and data.get("status") == "skipped_redundant_search":
            return

        if action in {
            "multi_source_search", "arxiv_search", "semantic_scholar", "google_scholar",
            "openalex", "crossref", "europe_pmc",
        }:
            self.web_searches += 1
            query = str((action_input or {}).get("query") or (data or {}).get("query") or "").strip()
            norm = normalize_query(query)
            if query and norm not in self.query_norms:
                self.queries.append(query)
                self.query_norms.append(norm)

        if not data:
            return
        chunks = data.get("indexed_chunks")
        if isinstance(chunks, int) and chunks > self.indexed_chunks:
            self.indexed_chunks = chunks
        coll = str(data.get("collection_name") or data.get("collection") or "").strip()
        if coll and coll not in self.collections:
            self.collections.append(coll)

        items = data.get("papers") or data.get("results") or data.get("documents") or []
        if not isinstance(items, list):
            return
        known = {_paper_key(p) for p in self.papers if _paper_key(p)}
        for item in items:
            if not isinstance(item, dict):
                continue
            key = _paper_key(item)
            if not key or key in known:
                continue
            known.add(key)
            self.papers.append({
                "title": item.get("title") or item.get("paper_title") or "",
                "year": item.get("year") or "",
                "authors": item.get("authors") or [],
                "venue": item.get("venue") or item.get("journal") or "",
                "url": item.get("url") or "",
                "pdf_url": item.get("pdf_url") or "",
                "local_path": item.get("local_path") or "",
                "abstract": str(item.get("abstract") or item.get("snippet") or item.get("text") or "")[:400],
                "source": item.get("source") or "",
            })

    def render(self, per_source_limit: int = 10, final_limit: int = 10) -> str:
        lines = [
            "# 研究记忆（跨步骤累积，不会因为轨迹截断而丢失）",
            "",
            f"- 已用检索词 {len(self.queries)}/{MAX_DISTINCT_QUERIES}（有文献后最多 {MAX_DISTINCT_QUERIES} 个不同 query）",
            f"- 已收录文献 {len(self.papers)} 篇（单次多源搜索：各源最多约 {per_source_limit} 篇候选，合并去重后保留约 {final_limit} 篇）",
            f"- 已索引文本块 {self.indexed_chunks}",
        ]
        if self.queries:
            used = "；".join(self.queries)
            lines.append(f"- 检索词：{used}")
        lines.append("")
        if not self.papers:
            lines.append("（还没有题录。请先 multi_source_search。）")
            return "\n".join(lines)
        lines.append("已收录题录：")
        for i, paper in enumerate(self.papers, 1):
            title = paper.get("title") or "（无标题）"
            year = paper.get("year") or "?"
            path = "已下载" if paper.get("local_path") else "无全文"
            lines.append(f"{i}. {title} ({year}) [{path}]")
        return "\n".join(lines)
