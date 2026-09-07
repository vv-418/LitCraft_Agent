"""OpenAlex 学术搜索（微软学术 MAG 后继，免费、无需密钥）。"""
from __future__ import annotations

import json
import os
from typing import Any

import requests

from tools.base import Tool


def _mailto() -> str:
    return (
        (os.getenv("OPENALEX_MAILTO") or os.getenv("CONTACT_EMAIL") or "").strip()
        or "litcraft@example.com"
    )


def reconstruct_abstract(inverted: dict | str | None) -> str:
    if not inverted:
        return ""
    if isinstance(inverted, str):
        return inverted.strip()
    if not isinstance(inverted, dict):
        return ""
    pairs: list[tuple[int, str]] = []
    for word, idxs in inverted.items():
        if not isinstance(idxs, list):
            continue
        for i in idxs:
            try:
                pairs.append((int(i), str(word)))
            except (TypeError, ValueError):
                continue
    pairs.sort()
    return " ".join(w for _, w in pairs)


class OpenAlexTool(Tool):
    """OpenAlex Works API：覆盖期刊 + 预印本，常带 OA PDF。"""

    name = "openalex"
    description = "Search OpenAlex (open scholarly graph) for papers by topic. Free, no API key."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Max papers", "default": 5},
            "year_from": {"type": "integer", "description": "Filter from year"},
        },
        "required": ["query"],
    }

    def __init__(self, max_results: int = 5, timeout: int = 25):
        self.max_results = max_results
        self.timeout = timeout
        self.base_url = "https://api.openalex.org/works"

    def _headers(self) -> dict[str, str]:
        return {"User-Agent": f"LitCraft-Agent/1.0 (mailto:{_mailto()})"}

    @staticmethod
    def _pdf_url(item: dict) -> str:
        oa = item.get("open_access") or {}
        for key in ("oa_url",):
            url = (oa.get(key) or "").strip()
            if url:
                return url
        loc = item.get("best_oa_location") or item.get("primary_location") or {}
        if isinstance(loc, dict):
            for key in ("pdf_url", "landing_page_url"):
                url = (loc.get(key) or "").strip()
                if url and url.lower().endswith(".pdf"):
                    return url
            pdf = (loc.get("pdf_url") or "").strip()
            if pdf:
                return pdf
        return ""

    def run(self, tool_input: dict[str, Any]) -> str:
        query = str(tool_input.get("query", "")).strip()
        if not query:
            return json.dumps({"error": "Query cannot be empty"}, ensure_ascii=False)

        from tools.search.query_optimizer import QueryOptimizer
        query = QueryOptimizer.optimize_for_search_static(query)
        limit = max(1, min(int(tool_input.get("limit", self.max_results) or self.max_results), 50))
        year_from = str(tool_input.get("year_from") or "").strip()

        params: dict[str, Any] = {
            "search": query,
            "per_page": limit,
            "mailto": _mailto(),
        }
        filters = ["has_abstract:true"]
        if year_from.isdigit():
            filters.append(f"from_publication_date:{year_from}-01-01")
        params["filter"] = ",".join(filters)

        try:
            print("[SEARCH] 搜索 OpenAlex...")
            resp = requests.get(
                self.base_url, params=params, headers=self._headers(), timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout:
            return json.dumps({"error": f"OpenAlex timed out after {self.timeout}s"}, ensure_ascii=False)
        except requests.exceptions.RequestException as e:
            return json.dumps({"error": f"OpenAlex request failed: {str(e)[:120]}"}, ensure_ascii=False)

        papers = []
        for item in data.get("results") or []:
            authorships = item.get("authorships") or []
            authors = []
            for a in authorships[:5]:
                name = ((a.get("author") or {}).get("display_name") or "").strip()
                if name:
                    authors.append(name)
            oa_id = str(item.get("id") or "")
            work_id = oa_id.rsplit("/", 1)[-1]
            doi = (item.get("doi") or "").replace("https://doi.org/", "")
            papers.append({
                "paper_id": work_id,
                "title": item.get("display_name") or item.get("title") or "",
                "authors": authors,
                "year": item.get("publication_year") or "",
                "abstract": reconstruct_abstract(item.get("abstract_inverted_index"))[:400],
                "venue": ((item.get("primary_location") or {}).get("source") or {}).get("display_name") or "",
                "citation_count": item.get("cited_by_count") or 0,
                "pdf_url": self._pdf_url(item),
                "doi": doi,
                "url": oa_id or (f"https://doi.org/{doi}" if doi else ""),
            })

        print(f"[OK] OpenAlex 找到 {len(papers)} 篇")
        return json.dumps(
            {"query": query, "count": len(papers), "papers": papers, "source": "openalex"},
            ensure_ascii=False,
            indent=2,
        )
