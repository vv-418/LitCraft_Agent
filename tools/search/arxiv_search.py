# 作用：实现真实的 arXiv 论文搜索工具，调用 arxiv API 获取论文元数据，使用指数退避重试。
import asyncio
import json
import time
from typing import Any
import arxiv
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from tools.base import Tool


class ArxivSearchTool(Tool):
    """真实的 arXiv 论文搜索工具，带重试机制。"""

    name = "arxiv_search"
    description = "Search for academic papers on arXiv by keywords or topics."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query (topic, keywords, author)"},
            "limit": {"type": "integer", "description": "Maximum number of papers to return", "default": 5},
            "year_from": {"type": "integer", "description": "Filter papers from this year (optional)"},
        },
        "required": ["query"],
    }

    def __init__(self, max_results: int = 5, initial_delay: float = 5.0, max_retries: int = 5):
        """初始化 ArxivSearchTool。
        
        Args:
            max_results: 最多返回的论文数量
            initial_delay: 初始延迟（秒）
            max_retries: 最大重试次数
        """
        self.max_results = max_results
        self.initial_delay = initial_delay
        self.max_retries = max_retries
        self.last_request_time = 0

    def _exponential_backoff_sleep(self, retry_count: int):
        """指数退避睡眠。
        
        Args:
            retry_count: 当前重试次数（从 0 开始）
        """
        # 计算延迟时间：初始延迟 * 2^retry_count，最多 60 秒
        delay = min(self.initial_delay * (2 ** retry_count), 60)
        print(f"[WAIT] 等待 {delay:.1f} 秒后重试...")
        time.sleep(delay)

    def _rate_limit(self):
        """实现基本速率限制。"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.initial_delay:
            time.sleep(self.initial_delay - elapsed)
        self.last_request_time = time.time()

    def _create_client_with_timeout(self):
        """创建带有超时配置的 arXiv 客户端。"""
        # 创建 requests 会话并配置重试策略
        session = requests.Session()
        
        # 配置 HTTP 和 HTTPS 适配器，支持连接重试和超时
        retry_strategy = Retry(
            total=2,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # 设置全局超时（30秒连接，60秒读取）
        session.timeout = (30, 60)
        
        # 创建 arXiv 客户端，使用自定义会话
        client = arxiv.Client(
            page_size=5,  # 减少页面大小以减轻服务器负载
            delay_seconds=3,  # 减少延迟以加快响应
        )
        # 覆盖客户端的会话
        client._session = session
        return client

    def run(self, tool_input: dict[str, Any]) -> str:
        """搜索 arXiv 论文，并返回论文元数据，带重试机制。
        
        Args:
            tool_input: 包含 'query' 和可选的 'limit', 'year_from' 字段
            
        Returns:
            JSON 字符串，包含论文列表
        """
        query = str(tool_input.get("query", "")).strip()
        # 查询优化：剥离冗余前缀、归一化
        from tools.search.query_optimizer import QueryOptimizer
        query = QueryOptimizer.optimize_for_search_static(query)
        limit = int(tool_input.get("limit", self.max_results))
        limit = min(limit, self.max_results)
        year_from_raw = tool_input.get("year_from")
        year_from = None
        if year_from_raw is not None:
            try:
                year_from = int(str(year_from_raw).strip())
            except (ValueError, TypeError):
                year_from = None

        if not query:
            return json.dumps({"error": "Query cannot be empty"}, ensure_ascii=False)

        # 重试循环
        for attempt in range(self.max_retries):
            try:
                # 应用速率限制
                self._rate_limit()
                
                print(f"[SEARCH] 尝试搜索 arXiv (第 {attempt + 1}/{self.max_retries} 次)...")
                
                # 创建带超时配置的 arXiv 客户端
                client = self._create_client_with_timeout()
                
                # 构建搜索查询 - 使用更简单的查询以降低 API 负载
                search = arxiv.Search(
                    query=query,
                    max_results=limit,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending,
                )

                papers = []
                for entry in client.results(search):
                    # 年份过滤：如果指定了 year_from，跳过更早的论文
                    if year_from and entry.published.year < year_from:
                        continue
                    
                    paper_info = {
                        "arxiv_id": entry.entry_id.split("/abs/")[-1],
                        "title": entry.title,
                        "authors": [author.name for author in entry.authors[:5]],
                        "published": entry.published.strftime("%Y-%m-%d"),
                        "summary": entry.summary.replace("\n", " ").strip()[:300],
                        "pdf_url": entry.pdf_url,
                        "arxiv_url": entry.entry_id,
                    }
                    papers.append(paper_info)

                if not papers:
                    return json.dumps(
                        {"message": f"No papers found for query: {query}", "papers": []},
                        ensure_ascii=False
                    )

                print(f"[OK] 成功找到 {len(papers)} 篇论文")
                return json.dumps(
                    {"query": query, "count": len(papers), "papers": papers},
                    ensure_ascii=False,
                    indent=2
                )

            except Exception as e:
                error_msg = str(e)
                print(f"[ERROR] 第 {attempt + 1} 次尝试失败: {error_msg}")
                
                # 如果是最后一次重试，返回错误
                if attempt == self.max_retries - 1:
                    return json.dumps(
                        {"error": f"Failed to search arXiv after {self.max_retries} retries: {error_msg}"},
                        ensure_ascii=False
                    )
                
                # 执行指数退避
                self._exponential_backoff_sleep(attempt)
        
        return json.dumps(
            {"error": "Failed to search arXiv: Unknown error"},
            ensure_ascii=False
        )

    async def async_run(self, tool_input: dict[str, Any]) -> str:
        """纯异步搜索 arXiv 论文，直接调用 arXiv OAI-PMH API（aiohttp + XML 解析）。"""
        import aiohttp
        import urllib.parse
        import xml.etree.ElementTree as ET

        query = str(tool_input.get("query", "")).strip()
        from tools.search.query_optimizer import QueryOptimizer
        query = QueryOptimizer.optimize_for_search_static(query)
        limit = int(tool_input.get("limit", self.max_results))
        limit = min(limit, self.max_results)
        year_from_raw = tool_input.get("year_from")
        year_from = None
        if year_from_raw is not None:
            try:
                year_from = int(str(year_from_raw).strip())
            except (ValueError, TypeError):
                year_from = None

        if not query:
            return json.dumps({"error": "Query cannot be empty"}, ensure_ascii=False)

        # arXiv API query: all:word1+AND+all:word2
        encoded_words = [urllib.parse.quote(w, safe='') for w in query.strip().split()]
        search_query = "all:" + "+AND+all:".join(encoded_words) if encoded_words else ""

        url = (f"http://export.arxiv.org/api/query"
               f"?search_query={search_query}"
               f"&start=0&max_results={limit}"
               f"&sortBy=submittedDate&sortOrder=descending")

        # arXiv 要求 User-Agent
        headers = {"User-Agent": "LitCraft/1.0 (mailto:litcraft@example.com)"}
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers,
                                       timeout=aiohttp.ClientTimeout(total=60)) as response:
                    xml_text = await response.text()

            root = ET.fromstring(xml_text)
            papers = []
            for entry in root.findall("atom:entry", ns):
                published_str = entry.findtext("atom:published", "", ns)
                pub_year = int(published_str[:4]) if published_str and published_str[:4].isdigit() else 0
                if year_from and pub_year < year_from:
                    continue

                entry_id = entry.findtext("atom:id", "", ns)
                arxiv_id = entry_id.split("/abs/")[-1] if "/abs/" in entry_id else entry_id

                authors = []
                for author_el in entry.findall("atom:author", ns):
                    name = author_el.findtext("atom:name", "", ns)
                    if name:
                        authors.append(name)

                pdf_url = ""
                for link in entry.findall("atom:link", ns):
                    if link.get("title") == "pdf":
                        pdf_url = link.get("href", "")
                        break

                title = entry.findtext("atom:title", "", ns)
                summary = entry.findtext("atom:summary", "", ns)

                paper_info = {
                    "arxiv_id": arxiv_id,
                    "title": title.replace("\n", " ").strip() if title else "",
                    "authors": authors[:5],
                    "published": published_str[:10] if published_str else "",
                    "summary": summary.replace("\n", " ").strip()[:300] if summary else "",
                    "pdf_url": pdf_url,
                    "arxiv_url": entry_id,
                }
                papers.append(paper_info)

            if not papers:
                return json.dumps(
                    {"message": f"No papers found for query: {query}", "papers": []},
                    ensure_ascii=False
                )

            print(f"[OK] arXiv 异步搜索: 成功找到 {len(papers)} 篇论文")
            return json.dumps(
                {"query": query, "count": len(papers), "papers": papers},
                ensure_ascii=False, indent=2
            )

        except asyncio.TimeoutError:
            return json.dumps(
                {"error": f"arXiv request timed out after 60 seconds"},
                ensure_ascii=False
            )
        except aiohttp.ClientError as e:
            return json.dumps(
                {"error": f"arXiv HTTP error: {str(e)[:120]}"},
                ensure_ascii=False
            )
        except ET.ParseError as e:
            return json.dumps(
                {"error": f"Failed to parse arXiv XML: {str(e)[:120]}"},
                ensure_ascii=False
            )
        except Exception as e:
            return json.dumps(
                {"error": f"arXiv async search failed: {str(e)[:120]}"},
                ensure_ascii=False
            )
