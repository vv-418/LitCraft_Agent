# 作用：实现 Semantic Scholar API 搜索工具。
# 有 SEMANTIC_SCHOLAR_API_KEY 时带 x-api-key（官方约 1 req/s）；无密钥仍可匿名调用但更易限流。
import asyncio
import json
import os
import time
import random
from typing import Any
import requests

from tools.base import Tool


class SemanticScholarTool(Tool):
    """Semantic Scholar 学术搜索工具，通过官方 API。"""

    name = "semantic_scholar"
    description = "Search for academic papers on Semantic Scholar by keywords, authors, or topics."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query (topic, keywords, author)"},
            "limit": {"type": "integer", "description": "Maximum number of papers to return", "default": 5},
            "year_from": {"type": "integer", "description": "Filter papers from this year (optional)"},
            "year_to": {"type": "integer", "description": "Filter papers up to this year (optional)"},
        },
        "required": ["query"],
    }

    def __init__(self, max_results: int = 5, timeout: int = 30, api_key: str | None = None):
        """初始化 Semantic Scholar Tool。
        
        Args:
            max_results: 最多返回的论文数量
            timeout: 请求超时时间（秒）
            api_key: S2 API Key；默认读环境变量 SEMANTIC_SCHOLAR_API_KEY / S2_API_KEY
        """
        self.max_results = max_results
        self.timeout = timeout
        self.base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        self.api_key = (
            (api_key or "").strip()
            or (os.getenv("SEMANTIC_SCHOLAR_API_KEY") or "").strip()
            or (os.getenv("S2_API_KEY") or "").strip()
        )
        self.last_request_time = 0
        # 有 key：官方 1 req/s，略留余量；无 key：更保守
        self.min_delay = 1.1 if self.api_key else 3.5
        self.max_retries = 3

    def _headers(self) -> dict[str, str]:
        headers = {
            "User-Agent": "LitCraft-Agent/1.0 (academic literature review)",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    @staticmethod
    def _resolve_pdf_url(item: dict) -> str:
        """优先 openAccessPdf；其次 ArXiv externalId；不用 S2 网页 HTML。"""
        oa = item.get("openAccessPdf") or {}
        pdf_url = (oa.get("url") or "").strip() if isinstance(oa, dict) else ""
        if pdf_url and "semanticscholar.org/paper/" not in pdf_url.lower():
            return pdf_url
        ext = item.get("externalIds") or {}
        if isinstance(ext, dict):
            arxiv_id = (ext.get("ArXiv") or ext.get("arXiv") or "").strip()
            if arxiv_id:
                return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        return ""

    def _rate_limit(self):
        """实现速率限制：保持请求间隔至少 min_delay 秒。"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_delay:
            sleep_time = self.min_delay - elapsed + random.uniform(0, 0.15)
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _request_with_retry(self, params: dict) -> requests.Response:
        """带指数退避重试的请求。
        
        对 429（速率限制）和 5xx（服务端错误）自动重试，最多 max_retries 次。
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                self._rate_limit()
                
                response = requests.get(
                    self.base_url,
                    params=params,
                    headers=self._headers(),
                    timeout=self.timeout
                )
                
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 5))
                    wait = retry_after + random.uniform(0, 2)
                    print(f"[WARN]  Semantic Scholar 429 速率限制（尝试 {attempt + 1}/{self.max_retries + 1}），等待 {wait:.0f} 秒...")
                    time.sleep(wait)
                    last_exception = requests.exceptions.RequestException(
                        f"429 Too Many Requests (attempt {attempt + 1})"
                    )
                    continue
                
                response.raise_for_status()
                return response
                
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                wait = (2 ** (attempt + 1)) + random.uniform(0, 2)  # 指数退避: 2s, 4s, 8s...
                print(f"[WARN]  Semantic Scholar 请求失败（尝试 {attempt + 1}/{self.max_retries + 1}）：{str(e)[:60]}，{wait:.0f} 秒后重试...")
                time.sleep(wait)
                last_exception = e
        
        # 所有重试耗尽
        raise last_exception or requests.exceptions.RequestException("Max retries exhausted")

    def run(self, tool_input: dict[str, Any]) -> str:
        """搜索 Semantic Scholar 论文。
        
        Args:
            tool_input: 包含 'query' 和可选的 'limit', 'year_from', 'year_to' 字段
            
        Returns:
            JSON 字符串，包含论文列表
        """
        query = str(tool_input.get("query", "")).strip()
        # 查询优化：剥离冗余前缀、归一化
        from tools.search.query_optimizer import QueryOptimizer
        query = QueryOptimizer.optimize_for_search_static(query)
        limit = int(tool_input.get("limit", self.max_results))
        limit = max(1, min(limit, max(self.max_results, 50)))
        year_from = tool_input.get("year_from")
        year_to = tool_input.get("year_to")

        if not query:
            return json.dumps({"error": "Query cannot be empty"}, ensure_ascii=False)

        try:
            auth = "API key" if self.api_key else "anonymous"
            print(f"[SEARCH] 搜索 Semantic Scholar（{auth}）...")
            
            # 构建查询参数
            params = {
                "query": query,
                "limit": limit,
                "fields": "paperId,title,authors,year,abstract,venue,citationCount,openAccessPdf,externalIds",
            }
            
            # 如果指定了年份范围，添加到查询
            if year_from:
                params["year"] = f"{year_from}:"
            if year_to:
                if "year" in params:
                    params["year"] += f"{year_to}"
                else:
                    params["year"] = f":{year_to}"
            
            # 带重试的请求
            response = self._request_with_retry(params)
            
            data = response.json()
            
            if "data" not in data or not data["data"]:
                return json.dumps(
                    {"message": f"No papers found for query: {query}", "papers": []},
                    ensure_ascii=False
                )
            
            papers = []
            for item in data.get("data", []):
                # 提取作者名称
                authors = [author.get("name", "") for author in item.get("authors", [])[:5]]
                
                paper_info = {
                    "paper_id": item.get("paperId", ""),
                    "title": item.get("title", ""),
                    "authors": authors,
                    "year": item.get("year", ""),
                    "abstract": item.get("abstract", "")[:300] if item.get("abstract") else "",
                    "venue": item.get("venue", ""),
                    "citation_count": item.get("citationCount", 0),
                    "pdf_url": self._resolve_pdf_url(item),
                    "semantic_scholar_url": f"https://www.semanticscholar.org/paper/{item.get('paperId', '')}",
                }
                papers.append(paper_info)
            
            print(f"[OK] 成功找到 {len(papers)} 篇论文")
            return json.dumps(
                {"query": query, "count": len(papers), "papers": papers, "source": "semantic_scholar"},
                ensure_ascii=False,
                indent=2
            )
        
        except requests.exceptions.Timeout:
            return json.dumps(
                {"error": f"Semantic Scholar request timed out after {self.timeout} seconds"},
                ensure_ascii=False
            )
        except requests.exceptions.RequestException as e:
            return json.dumps(
                {"error": f"Failed to search Semantic Scholar: {str(e)}"},
                ensure_ascii=False
            )
        except json.JSONDecodeError:
            return json.dumps(
                {"error": "Failed to parse Semantic Scholar response"},
                ensure_ascii=False
            )
        except Exception as e:
            return json.dumps(
                {"error": f"Unexpected error: {str(e)}"},
                ensure_ascii=False
            )

    async def async_run(self, tool_input: dict[str, Any]) -> str:
        """纯异步搜索 Semantic Scholar，使用 aiohttp 替代 requests。"""
        import aiohttp

        query = str(tool_input.get("query", "")).strip()
        from tools.search.query_optimizer import QueryOptimizer
        query = QueryOptimizer.optimize_for_search_static(query)
        limit = int(tool_input.get("limit", self.max_results))
        limit = max(1, min(limit, max(self.max_results, 50)))
        year_from = tool_input.get("year_from")
        year_to = tool_input.get("year_to")

        if not query:
            return json.dumps({"error": "Query cannot be empty"}, ensure_ascii=False)

        params = {
            "query": query,
            "limit": limit,
            "fields": "paperId,title,authors,year,abstract,venue,citationCount,openAccessPdf,externalIds",
        }
        if year_from:
            params["year"] = f"{year_from}:"
        if year_to:
            params["year"] = params.get("year", "") + f"{year_to}"

        headers = self._headers()

        for attempt in range(self.max_retries + 1):
            try:
                # 与同步路径共用客户端侧限速
                elapsed = time.time() - self.last_request_time
                if elapsed < self.min_delay:
                    await asyncio.sleep(self.min_delay - elapsed + random.uniform(0, 0.15))
                self.last_request_time = time.time()

                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        self.base_url,
                        params=params,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        if response.status == 429:
                            retry_after = int(response.headers.get("Retry-After", 5))
                            wait = retry_after + random.uniform(0, 2)
                            print(f"[WARN]  S2 429 速率限制（尝试 {attempt + 1}/{self.max_retries + 1}），等待 {wait:.0f} 秒...")
                            await asyncio.sleep(wait)
                            continue

                        response.raise_for_status()
                        data = await response.json()

                if "data" not in data or not data["data"]:
                    return json.dumps(
                        {"message": f"No papers found for query: {query}", "papers": []},
                        ensure_ascii=False
                    )

                papers = []
                for item in data.get("data", []):
                    authors = [author.get("name", "") for author in item.get("authors", [])[:5]]
                    paper_info = {
                        "paper_id": item.get("paperId", ""),
                        "title": item.get("title", ""),
                        "authors": authors,
                        "year": item.get("year", ""),
                        "abstract": item.get("abstract", "")[:300] if item.get("abstract") else "",
                        "venue": item.get("venue", ""),
                        "citation_count": item.get("citationCount", 0),
                        "pdf_url": self._resolve_pdf_url(item),
                        "semantic_scholar_url": f"https://www.semanticscholar.org/paper/{item.get('paperId', '')}",
                    }
                    papers.append(paper_info)

                print(f"[OK] S2 异步搜索: 成功找到 {len(papers)} 篇论文")
                return json.dumps(
                    {"query": query, "count": len(papers), "papers": papers, "source": "semantic_scholar"},
                    ensure_ascii=False, indent=2
                )

            except (asyncio.TimeoutError, aiohttp.ServerTimeoutError):
                if attempt == self.max_retries:
                    return json.dumps(
                        {"error": f"Semantic Scholar request timed out after {self.timeout} seconds"},
                        ensure_ascii=False
                    )
                wait = (2 ** (attempt + 1)) + random.uniform(0, 2)
                await asyncio.sleep(wait)
                continue

            except aiohttp.ClientError as e:
                if attempt == self.max_retries:
                    return json.dumps(
                        {"error": f"Failed to search Semantic Scholar: {str(e)[:120]}"},
                        ensure_ascii=False
                    )
                wait = (2 ** (attempt + 1)) + random.uniform(0, 2)
                await asyncio.sleep(wait)
                continue

        return json.dumps(
            {"error": "Semantic Scholar: max retries exhausted"},
            ensure_ascii=False
        )
