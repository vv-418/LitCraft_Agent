"""Europe PMC 搜索（含 PubMed / PMC，常有全文 PDF 链接）。"""
from __future__ import annotations

import json
from typing import Any

import requests

from tools.base import Tool


class EuropePMCTool(Tool):
    """Europe PMC REST API：生物医学为主，也覆盖不少预印本与 OA 全文。"""

    name = "europe_pmc"
    description = "Search Europe PMC / PubMed for papers. Free, no API key; often includes OA PDF links."
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
        self.base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    @staticmethod
    def _pdf_url(item: dict) -> str:
        urls = ((item.get("fullTextUrlList") or {}).get("fullTextUrl")) or []
        if isinstance(urls, dict):
            urls = [urls]
        pdf_first = ""
        any_first = ""
        for u in urls:
            if not isinstance(u, dict):
                continue
            href = (u.get("url") or "").strip()
            if not href:
                continue
            style = (u.get("documentStyle") or "").lower()
            avail = (u.get("availability") or "").lower()
            if not any_first:
                any_first = href
            if "pdf" in style or href.lower().endswith(".pdf"):
                if avail in ("open access", "free", "oa", "") or "open" in avail:
                    return href
                if not pdf_first:
                    pdf_first = href
        return pdf_first or any_first

    def run(self, tool_input: dict[str, Any]) -> str:
        query = str(tool_input.get("query", "")).strip()
        if not query:
            return json.dumps({"error": "Query cannot be empty"}, ensure_ascii=False)

        from tools.search.query_optimizer import QueryOptimizer
        query = QueryOptimizer.optimize_for_search_static(query)
        limit = max(1, min(int(tool_input.get("limit", self.max_results) or self.max_results), 50))
        year_from = str(tool_input.get("year_from") or "").strip()
        epmc_q = query
        if year_from.isdigit():
            epmc_q = f"({query}) AND PUB_YEAR:[{year_from} TO 2100]"

        params = {
            "query": epmc_q,
            "format": "json",
            "pageSize": limit,
            "resultType": "core",
        }

        try:
            print("[SEARCH] 搜索 Europe PMC...")
            resp = requests.get(self.base_url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout:
            return json.dumps({"error": f"Europe PMC timed out after {self.timeout}s"}, ensure_ascii=False)
        except requests.exceptions.RequestException as e:
            return json.dumps({"error": f"Europe PMC request failed: {str(e)[:120]}"}, ensure_ascii=False)

        papers = []
        for item in ((data.get("resultList") or {}).get("result") or []):
            doi = item.get("doi") or ""
            pmid = str(item.get("pmid") or item.get("id") or "")
            authors = []
            raw_authors = item.get("authorString") or ""
            if raw_authors:
                authors = [a.strip() for a in raw_authors.split(",") if a.strip()][:5]
            pmcid = item.get("pmcid") or ""
            pdf = self._pdf_url(item)
            if not pdf and pmcid:
                pdf = f"https://europepmc.org/articles/{pmcid}?pdf=render"
            papers.append({
                "paper_id": pmid or doi,
                "title": item.get("title") or "",
                "authors": authors,
                "year": str(item.get("pubYear") or ""),
                "abstract": (item.get("abstractText") or "")[:400],
                "venue": item.get("journalTitle") or item.get("bookOrReportDetails") or "",
                "citation_count": int(item.get("citedByCount") or 0),
                "pdf_url": pdf,
                "doi": doi,
                "url": (
                    f"https://europepmc.org/article/MED/{pmid}" if pmid
                    else (f"https://doi.org/{doi}" if doi else "")
                ),
            })

        print(f"[OK] Europe PMC 找到 {len(papers)} 篇")
        return json.dumps(
            {"query": query, "count": len(papers), "papers": papers, "source": "europe_pmc"},
            ensure_ascii=False,
            indent=2,
        )
