# 作用：实现 Google Scholar 搜索工具，基于 scholarly 库，带随机延迟以避免被封禁。
import json
import time
import random
from typing import Any

from tools.base import Tool


class GoogleScholarTool(Tool):
    """Google Scholar 学术搜索工具，基于 scholarly 库。"""

    name = "google_scholar"
    description = "Search for academic papers on Google Scholar by keywords or authors."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query (topic, keywords, author)"},
            "limit": {"type": "integer", "description": "Maximum number of papers to return", "default": 5},
            "year_from": {"type": "integer", "description": "Filter papers from this year (optional)"},
        },
        "required": ["query"],
    }

    def __init__(self, max_results: int = 5, min_delay: float = 3.0, max_delay: float = 10.0):
        """初始化 Google Scholar Tool。
        
        Args:
            max_results: 最多返回的论文数量
            min_delay: 最小延迟时间（秒）
            max_delay: 最大延迟时间（秒）
        """
        self.max_results = max_results
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.last_request_time = 0
        self.max_retries = 2  # 最大重试次数
        self._init_scholarly()

    def _init_scholarly(self):
        """初始化 scholarly 库，配置反爬和代理机制。"""
        try:
            import scholarly
            # 配置 scholarly 的超时
            scholarly.scholarly.TIMEOUT = 15
            
            # 通过环境变量配置代理（可选）
            #   设置方式： $env:GOOGLE_SCHOLAR_PROXY="http://127.0.0.1:7890"
            import os
            proxy = os.environ.get("GOOGLE_SCHOLAR_PROXY", "").strip()
            if proxy:
                print(f"[PROXY] 使用代理: {proxy}")
                scholarly.scholarly.PROXIES = {"http": proxy, "https": proxy}
            else:
                print("[INFO]  未配置代理（可通过 GOOGLE_SCHOLAR_PROXY 环境变量设置）")
            
            self.scholarly = scholarly
            self.scholarly_available = True
        except ImportError:
            print("[WARN]  Warning: scholarly 库未安装，Google Scholar 功能将不可用")
            print("   请运行：pip install scholarly")
            self.scholarly_available = False

    def _random_delay(self):
        """添加随机延迟以避免被 Google Scholar 封禁。"""
        delay = random.uniform(self.min_delay, self.max_delay)
        elapsed = time.time() - self.last_request_time
        
        if elapsed < delay:
            sleep_time = delay - elapsed
            print(f"[WAIT] 避免被封禁，等待 {sleep_time:.1f} 秒...")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

    def run(self, tool_input: dict[str, Any]) -> str:
        """搜索 Google Scholar 论文。
        
        Args:
            tool_input: 包含 'query' 和可选的 'limit', 'year_from' 字段
            
        Returns:
            JSON 字符串，包含论文列表
        """
        if not self.scholarly_available:
            return json.dumps(
                {"error": "scholarly library not available. Please install: pip install scholarly"},
                ensure_ascii=False
            )
        
        query = str(tool_input.get("query", "")).strip()
        # 查询优化：剥离冗余前缀、归一化
        from tools.search.query_optimizer import QueryOptimizer
        query = QueryOptimizer.optimize_for_search_static(query)
        limit = int(tool_input.get("limit", self.max_results))
        limit = min(limit, self.max_results)
        year_from = tool_input.get("year_from")

        if not query:
            return json.dumps({"error": "Query cannot be empty"}, ensure_ascii=False)

        try:
            # 带重试机制的搜索（应对 Google Scholar 的临时性拦截）
            papers = []
            count = 0
            last_exception = None
            
            for retry in range(self.max_retries + 1):
                try:
                    if retry > 0:
                        # 重试前增加更大的延迟
                        print(f"[RETRY] 第 {retry} 次重试 Google Scholar（等待 {10 * retry} 秒后）...")
                        time.sleep(10 * retry)
                    
                    # 添加随机延迟
                    self._random_delay()
                    
                    print(f"[SEARCH] 搜索 Google Scholar...")
                    
                    # 发送搜索查询
                    search_query = self.scholarly.scholarly.search_pubs(query)
                    
                    papers = []
                    count = 0
                    
                    for pub in search_query:
                        if count >= limit:
                            break
                        
                        try:
                            # scholarly 1.x+ 返回 Publication 对象，不是 dict
                            # 属性访问方式：pub.bib['title'] / pub.bib.get('year') / pub.pub_url
                            bib = getattr(pub, 'bib', {}) or {}
                            if not isinstance(bib, dict):
                                bib = {}
                            
                            # 年份过滤（可能存在于 bib['year']、bib['pub_year'] 或直接属性）
                            pub_year = str(bib.get('year') or bib.get('pub_year') or '')
                            if not pub_year:
                                pub_year = str(getattr(pub, 'year', '') or '')
                            if year_from and pub_year:
                                try:
                                    if int(pub_year) < year_from:
                                        continue
                                except (ValueError, TypeError):
                                    pass
                            
                            # 提取标题
                            title = bib.get('title', '') or str(getattr(pub, 'title', ''))
                            
                            # 提取作者
                            authors_raw = bib.get('author', [])
                            if isinstance(authors_raw, str):
                                authors = [a.strip() for a in authors_raw.split(' and ')][:5]
                            elif isinstance(authors_raw, list):
                                authors = authors_raw[:5]
                            else:
                                authors = []
                            
                            # 提取摘要
                            abstract = bib.get('abstract', '') or ''
                            abstract = str(abstract)[:300] if abstract else ''
                            
                            # 提取会议/期刊
                            venue = bib.get('venue', '') or ''
                            
                            # 提取 URL
                            url = getattr(pub, 'pub_url', '') or bib.get('url', '') or ''
                            gs_url = getattr(pub, 'url_scholarbib', '') or ''
                            
                            paper_info = {
                                "title": title,
                                "authors": authors,
                                "year": pub_year,
                                "abstract": abstract,
                                "venue": venue,
                                "url": url,
                                "google_scholar_url": gs_url,
                            }
                            papers.append(paper_info)
                            count += 1
                            
                            # 每次获取后添加延迟以避免连续请求
                            time.sleep(random.uniform(1.0, 2.5))
                        
                        except Exception as e:
                            print(f"[WARN]  处理论文时出错: {str(e)}")
                            continue
                    
                    # 如果成功获取结果，跳出重试循环
                    break
                    
                except Exception as e:
                    last_exception = e
                    error_msg = str(e)
                    if "blocked" in error_msg.lower() or "captcha" in error_msg.lower() or "503" in error_msg:
                        print(f"[WARN]  Google Scholar 可能已屏蔽该请求（{error_msg[:80]}）")
                        if retry < self.max_retries:
                            print(f"[INFO]  将在更长延迟后重试...")
                            continue
                    elif retry < self.max_retries:
                        print(f"[WARN]  Google Scholar 搜索出错（{error_msg[:80]}），{10 * (retry + 1)} 秒后重试...")
                        continue
                    else:
                        # 所有重试耗尽
                        break
            
            if last_exception and not papers:
                return json.dumps(
                    {
                        "error": f"Failed to search Google Scholar after {self.max_retries + 1} attempts: {str(last_exception)[:200]}",
                        "note": "Google Scholar has strict anti-scraping measures. Try using semantic_scholar or arxiv instead.",
                        "source": "google_scholar"
                    },
                    ensure_ascii=False
                )

            # 正常返回搜索结果
            result = {
                "results": papers,
                "total": len(papers),
                "source": "google_scholar"
            }
            return json.dumps(result, ensure_ascii=False)

        except Exception as e:
            return json.dumps(
                {
                    "error": f"Google Scholar search unexpected error: {str(e)[:200]}",
                    "source": "google_scholar"
                },
                ensure_ascii=False
            )
