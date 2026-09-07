"""Crossref 元数据搜索（DOI 注册中心，写论文查文献最常用的底层源之一）。"""
from __future__ import annotations

import json
import os
import re
from typing import Any

import requests

from tools.base import Tool


def _mailto() -> str:
    return (
        (os.getenv("CROSSREF_MAILTO") or os.getenv("OPENALEX_MAILTO")
         or os.getenv("CONTACT_EMAIL") or "").strip()
        or "litcraft@example.com"
    )


_TAG = re.compile(r"<[^>]+>")


class CrossrefTool(Tool):
    """Crossref REST API：覆盖绝大多数正式发表论文的题录。"""

    name = "crossref"
    description = "Search Crossref for published papers by topic (DOI metadata). Free, no API key."
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
        self.base_url = "https://api.crossref.org/works"

    def _headers(self) -> dict[str, str]:
        return {
            "User-Agent": f"LitCraft-Agent/1.0 (mailto:{_mailto()})",
        }

    @staticmethod
    def _pdf_url(item: dict) -> str:
        for link in item.get("link") or []:
            if not isinstance(link, dict):
                continue
            ctype = (link.get("content-type") or "").lower()
            url = (link.get("URL") or "").strip()
            if url and "pdf" in ctype:
                return url
            if url.lower().endswith(".pdf"):
                return url
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
            "query": query,
            "rows": limit,
            "select": "DOI,title,author,published,abstract,container-title,is-referenced-by-count,URL,link,type",
        }
        if year_from.isdigit():
            params["filter"] = f"from-pub-date:{year_from}"

        try:
            print("[SEARCH] 搜索 Crossref...")
            resp = requests.get(
                self.base_url, params=params, headers=self._headers(), timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout:
            return json.dumps({"error": f"Crossref timed out after {self.timeout}s"}, ensure_ascii=False)
        except requests.exceptions.RequestException as e:
            return json.dumps({"error": f"Crossref request failed: {str(e)[:120]}"}, ensure_ascii=False)

        papers = []
        for item in ((data.get("message") or {}).get("items") or []):
            titles = item.get("title") or []
            title = titles[0] if titles else ""
            authors = []
            for a in (item.get("author") or [])[:5]:
                given = (a.get("given") or "").strip()
                family = (a.get("family") or "").strip()
                name = f"{given} {family}".strip() or (a.get("name") or "").strip()
                if name:
                    authors.append(name)
            year = ""
            published = item.get("published") or item.get("published-print") or item.get("published-online") or {}
            parts = (published.get("date-parts") or [[]])[0]
            if parts:
                year = str(parts[0])
            abstract = _TAG.sub(" ", item.get("abstract") or "").strip()[:400]
            doi = item.get("DOI") or ""
            venues = item.get("container-title") or []
            papers.append({
                "paper_id": doi,
                "title": title,
                "authors": authors,
                "year": year,
                "abstract": abstract,
                "venue": venues[0] if venues else "",
                "citation_count": item.get("is-referenced-by-count") or 0,
                "pdf_url": self._pdf_url(item),
                "doi": doi,
                "url": item.get("URL") or (f"https://doi.org/{doi}" if doi else ""),
            })

        print(f"[OK] Crossref 找到 {len(papers)} 篇")
        return json.dumps(
            {"query": query, "count": len(papers), "papers": papers, "source": "crossref"},
            ensure_ascii=False,
            indent=2,
        )
