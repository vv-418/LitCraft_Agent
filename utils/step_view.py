"""把 Agent 单步结果整理成前端可读摘要 + 可轮询的序列化结构。"""

from __future__ import annotations

import json
from typing import Any

from agent.models import AgentStep

_ACTION_LABELS = {
    "multi_source_search": "多源学术搜索",
    "arxiv_search": "arXiv 搜索",
    "semantic_scholar": "Semantic Scholar 搜索",
    "google_scholar": "Google Scholar 搜索",
    "openalex": "OpenAlex 搜索",
    "crossref": "Crossref 搜索",
    "europe_pmc": "Europe PMC / PubMed 搜索",
    "paper_downloader": "下载论文 PDF",
    "pdf_parser": "解析 PDF 全文",
    "text_chunker": "文本分块",
    "vector_store": "写入向量库",
    "advanced_search": "本地证据检索",
    "write_review": "撰写综述正文",
}


def _safe_json_loads(text: str | None) -> Any | None:
    if not text or not str(text).strip():
        return None
    raw = str(text).strip()
    # 截断尾巴时尽量找到完整 JSON 起点
    if not raw.startswith("{") and not raw.startswith("["):
        brace = raw.find("{")
        if brace >= 0:
            raw = raw[brace:]
    try:
        return json.loads(raw)
    except Exception:
        return None


def _paper_item(p: dict[str, Any]) -> dict[str, Any]:
    title = (
        p.get("title")
        or p.get("paper_title")
        or ""
    )
    return {
        "title": str(title)[:200],
        "year": str(p.get("year") or p.get("published") or "")[:12],
        "source": str(
            p.get("source")
            or (p.get("sources") or [""])[0]
            or ""
        )[:40],
        "url": str(p.get("url") or p.get("pdf_url") or p.get("arxiv_url") or "")[:300],
        "citation_count": p.get("citation_count") or p.get("citationCount") or 0,
    }


def _extract_papers(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    papers = data.get("papers") or data.get("results") or []
    if not isinstance(papers, list):
        return []
    out = []
    for p in papers:
        if isinstance(p, dict) and (p.get("title") or p.get("paper_title")):
            out.append(_paper_item(p))
    return out


def build_step_summary(step: AgentStep) -> dict[str, Any]:
    """面向用户的一步摘要（非原始 JSON）。"""
    action = (step.action or "").strip()
    thought = (step.thought or "").strip()
    obs = step.observation
    data = _safe_json_loads(obs if isinstance(obs, str) else None)

    if thought.startswith("[系统提示]"):
        return {
            "kind": "system",
            "title": "系统提示",
            "headline": thought.replace("[系统提示]", "").strip()[:200],
            "papers": [],
            "stats": {},
            "error": None,
        }

    if not action:
        return {
            "kind": "final",
            "title": "汇总 / 结束",
            "headline": (thought[:160] if thought else "本轮思考结束"),
            "papers": [],
            "stats": {},
            "error": None,
        }

    label = _ACTION_LABELS.get(action, action)
    err = None
    if isinstance(data, dict) and data.get("error"):
        err = str(data.get("error"))[:240]

    papers = _extract_papers(data)
    stats: dict[str, Any] = {}
    headline = ""

    if "search" in action or action in ("semantic_scholar", "google_scholar"):
        kind = "search"
        total = None
        if isinstance(data, dict):
            if "total_count" in data:
                total = data.get("total_count")
            elif "count" in data:
                total = data.get("count")
            elif papers:
                total = len(papers)
            by_source = data.get("search_results") or data.get("source_counts")
            src_sum = data.get("source_summary")
            if isinstance(src_sum, dict):
                ok = src_sum.get("success") or []
                fail = src_sum.get("failed") or []
                if isinstance(ok, list) and ok:
                    stats["sources_ok"] = ok
                if isinstance(fail, list) and fail:
                    stats["sources_fail"] = fail
            if isinstance(by_source, dict):
                stats["by_source"] = by_source
        if total is None:
            total = len(papers)
        stats["total"] = total
        if isinstance(data, dict) and data.get("status") == "skipped_redundant_search":
            stats["total"] = data.get("paper_count") or 0
            headline = str(data.get("message") or "已有文献，跳过重复搜索")
        elif err:
            headline = f"搜索失败：{err}"
        elif total == 0:
            headline = "未找到相关论文"
        else:
            extra = ""
            if stats.get("by_source"):
                parts = [f"{k} {v}" for k, v in list(stats["by_source"].items())[:4]]
                extra = "（" + " · ".join(parts) + "）"
            elif stats.get("sources_ok"):
                extra = "（来源：" + "、".join(str(x) for x in stats["sources_ok"][:4]) + "）"
            headline = f"找到 {total} 篇论文{extra}"
    elif "download" in action:
        kind = "download"
        if isinstance(data, dict):
            ok = data.get("success") or data.get("downloaded") or []
            fail = data.get("failed") or []
            if isinstance(ok, list) or isinstance(fail, list):
                n_ok = len(ok) if isinstance(ok, list) else 0
                n_fail = len(fail) if isinstance(fail, list) else 0
                stats = {"downloaded": n_ok, "failed": n_fail}
                if n_ok == 0:
                    headline = f"未下载到 PDF（成功 {n_ok}，失败 {n_fail}）"
                    err = err or "没有成功下载任何文件，后续无法解析入库"
                else:
                    headline = f"下载完成：成功 {n_ok}，失败 {n_fail}"
            elif data.get("status"):
                headline = f"下载状态：{data.get('status')}"
            elif err:
                headline = f"下载失败：{err}"
            else:
                headline = "已执行下载"
        else:
            headline = "已执行下载"
    elif "parser" in action or action == "pdf_parser":
        kind = "parse"
        if isinstance(data, dict) and str(data.get("status") or "").startswith("skipped"):
            headline = str(data.get("message") or "已跳过解析")[:200]
            err = err or "无可用 PDF，已跳过解析"
        else:
            headline = err or "已解析 PDF 文本"
    elif "chunk" in action:
        kind = "chunk"
        headline = err or "已完成文本分块"
    elif "vector" in action:
        kind = "index"
        headline = err or "已写入向量库"
    elif "advanced" in action or "retriev" in action:
        kind = "retrieve"
        papers = _extract_papers(data) if data else []
        stats["total"] = len(papers)
        headline = err or (f"检索到 {len(papers)} 条证据片段" if papers else "已执行本地检索")
    else:
        kind = "other"
        headline = err or (thought[:120] if thought else f"已执行 {label}")

    return {
        "kind": kind,
        "title": label,
        "headline": headline,
        "papers": papers[:30],
        "stats": stats,
        "error": err,
    }


def serialize_step(step: AgentStep, observation_limit: int = 8000) -> dict[str, Any]:
    """序列化一步，供 TaskManager / API 轮询。"""
    obs = str(step.observation) if step.observation is not None else None
    if obs and len(obs) > observation_limit:
        obs = obs[:observation_limit] + f"... [已截断，共 {len(step.observation)} 字符]"

    return {
        "thought": step.thought or "",
        "action": step.action,
        "action_input": step.action_input or {},
        "observation": obs,
        "summary": build_step_summary(step),
    }
