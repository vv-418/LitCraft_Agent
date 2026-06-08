# 作用：实现 Semantic Scholar API 搜索工具，完全免费，无需密钥，每分钟 100 次请求。
import json
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

    def __init__(self, max_results: int = 5, timeout: int = 30):
        """初始化 Semantic Scholar Tool。
        
        Args:
            max_results: 最多返回的论文数量
            timeout: 请求超时时间（秒）
        """
        self.max_results = max_results
        self.timeout = timeout
        self.base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        self.last_request_time = 0
        self.min_delay = 3.5  # 最小请求延迟（秒），官方限制 100 req/min ≈ 0.6s，设 3.5s 更安全
        self.max_retries = 3  # 429 重试次数

    def _rate_limit(self):
        """实现速率限制：保持请求间隔至少 min_delay 秒。"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_delay:
            sleep_time = self.min_delay - elapsed + random.uniform(0, 0.5)
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
                
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                
                response = requests.get(
                    self.base_url,
                    params=params,
                    headers=headers,
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
        limit = int(tool_input.get("limit", self.max_results))
        limit = min(limit, self.max_results)
        year_from = tool_input.get("year_from")
        year_to = tool_input.get("year_to")

        if not query:
            return json.dumps({"error": "Query cannot be empty"}, ensure_ascii=False)

        try:
            print(f"[SEARCH] 搜索 Semantic Scholar...")
            
            # 构建查询参数
            params = {
                "query": query,
                "limit": limit,
                "fields": "paperId,title,authors,year,abstract,venue,citationCount,openAccessPdf"
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
                
                # 获取 PDF URL（如果可用）
                pdf_url = ""
                if item.get("openAccessPdf"):
                    pdf_url = item["openAccessPdf"].get("url", "")
                
                paper_info = {
                    "paper_id": item.get("paperId", ""),
                    "title": item.get("title", ""),
                    "authors": authors,
                    "year": item.get("year", ""),
                    "abstract": item.get("abstract", "")[:300] if item.get("abstract") else "",
                    "venue": item.get("venue", ""),
                    "citation_count": item.get("citationCount", 0),
                    "pdf_url": pdf_url,
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
