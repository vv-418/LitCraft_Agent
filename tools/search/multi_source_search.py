# 作用：实现多源搜索融合器，整合 arXiv、Semantic Scholar、Google Scholar 的搜索结果，并去重排序。
import json
import hashlib
from typing import Any, Dict, List
from collections import defaultdict

from tools.base import Tool
from tools.search.arxiv_search import ArxivSearchTool
from tools.search.semantic_scholar import SemanticScholarTool
from tools.search.google_scholar import GoogleScholarTool


class MultiSourceSearchTool(Tool):
    """多源搜索融合器，整合多个学术搜索引擎的结果。"""

    name = "multi_source_search"
    description = "Search across multiple academic databases (arXiv, Semantic Scholar, Google Scholar) and merge results intelligently."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Maximum total papers to return", "default": 10},
            "sources": {
                "type": "array",
                "items": {"type": "string", "enum": ["arxiv", "semantic_scholar", "google_scholar", "all"]},
                "description": "Which sources to search from. 'all' searches all available sources.",
                "default": ["all"]
            },
        },
        "required": ["query"],
    }

    def __init__(self, max_results: int = 10):
        """初始化 Multi Source Search Tool。
        
        Args:
            max_results: 最多返回的论文数量
        """
        self.max_results = max_results
        self.arxiv_tool = ArxivSearchTool(max_results=5)
        self.semantic_tool = SemanticScholarTool(max_results=5)
        self.google_tool = GoogleScholarTool(max_results=5)

    def _normalize_paper(self, paper: dict, source: str) -> dict:
        """将不同来源的论文格式统一化。
        
        Args:
            paper: 原始论文数据
            source: 数据来源
            
        Returns:
            统一格式的论文数据
        """
        if source == "arxiv":
            return {
                "title": paper.get("title", ""),
                "authors": paper.get("authors", []),
                "year": paper.get("published", "")[:4],
                "abstract": paper.get("summary", ""),
                "source": "arXiv",
                "source_id": paper.get("arxiv_id", ""),
                "url": paper.get("arxiv_url", ""),
                "pdf_url": paper.get("pdf_url", ""),
                "venue": "",
                "citation_count": 0,
            }
        elif source == "semantic_scholar":
            return {
                "title": paper.get("title", ""),
                "authors": paper.get("authors", []),
                "year": str(paper.get("year", "")),
                "abstract": paper.get("abstract", ""),
                "source": "Semantic Scholar",
                "source_id": paper.get("paper_id", ""),
                "url": paper.get("semantic_scholar_url", ""),
                "pdf_url": paper.get("pdf_url", ""),
                "venue": paper.get("venue", ""),
                "citation_count": paper.get("citation_count", 0),
            }
        elif source == "google_scholar":
            return {
                "title": paper.get("title", ""),
                "authors": paper.get("authors", []),
                "year": str(paper.get("year", "")),
                "abstract": paper.get("abstract", ""),
                "source": "Google Scholar",
                "source_id": "",
                "url": paper.get("google_scholar_url", ""),
                "pdf_url": paper.get("url", ""),
                "venue": paper.get("venue", ""),
                "citation_count": 0,
            }
        else:
            return paper

    def _get_paper_hash(self, paper: dict) -> str:
        """生成论文的哈希值用于去重（基于标题）。"""
        title = paper.get("title", "").lower().strip()
        # 移除一些常见的停用词以提高匹配度
        title = title.replace(" the ", " ").replace(" a ", " ").replace(" and ", " ")
        return hashlib.md5(title.encode()).hexdigest()

    def _deduplicate_and_rank(self, papers: List[dict]) -> List[dict]:
        """对论文进行去重和排序。
        
        Args:
            papers: 论文列表
            
        Returns:
            去重并按相关性排序的论文列表
        """
        # 按哈希值去重，保留引用数最多的版本
        dedup_dict: Dict[str, dict] = {}
        
        for paper in papers:
            paper_hash = self._get_paper_hash(paper)
            
            if paper_hash not in dedup_dict:
                dedup_dict[paper_hash] = paper
            else:
                # 如果存在重复，保留引用数更多的版本
                existing = dedup_dict[paper_hash]
                if paper.get("citation_count", 0) > existing.get("citation_count", 0):
                    dedup_dict[paper_hash] = paper
                # 合并来源信息
                existing_sources = existing.get("sources", [existing.get("source", "")])
                paper_source = paper.get("source", "")
                if paper_source and paper_source not in existing_sources:
                    existing_sources.append(paper_source)
                    existing["sources"] = existing_sources
        
        # 转换回列表并按以下顺序排序：
        # 1. 来源数量（多源优先）
        # 2. 引用数量（引用多优先）
        # 3. 发布年份（最新优先）
        result = list(dedup_dict.values())
        result.sort(
            key=lambda x: (
                -len(x.get("sources", [x.get("source", "")])),  # 多源优先
                -x.get("citation_count", 0),  # 引用数多优先
                -int(x.get("year", "0") or "0"),  # 最新优先
            )
        )
        
        return result

    def run(self, tool_input: dict[str, Any]) -> str:
        """执行多源搜索。
        
        Args:
            tool_input: 包含 'query', 'limit', 'sources' 字段
            
        Returns:
            JSON 字符串，包含统一格式的论文列表
        """
        query = str(tool_input.get("query", "")).strip()
        limit = int(tool_input.get("limit", self.max_results))
        sources = tool_input.get("sources", ["all"])
        
        # 处理 'all' 选项
        if "all" in sources or not sources:
            sources = ["arxiv", "semantic_scholar", "google_scholar"]
        
        if not query:
            return json.dumps({"error": "Query cannot be empty"}, ensure_ascii=False)

        print(f"\n[SEARCH] 多源搜索启动")
        print(f"   查询: {query}")
        print(f"   来源: {', '.join(sources)}")
        print("-" * 60)
        
        all_papers = []
        search_results = {}
        source_errors = {}
        
        # 搜索 arXiv
        if "arxiv" in sources:
            try:
                print("[PAPERS] 从 arXiv 搜索...")
                result = self.arxiv_tool.run({"query": query, "limit": 5})
                data = json.loads(result)
                if "error" in data:
                    raise Exception(data["error"])
                arxiv_papers = data.get("papers", [])
                search_results["arxiv"] = len(arxiv_papers)
                print(f"  -> arXiv: 找到 {len(arxiv_papers)} 篇论文")
                
                for paper in arxiv_papers:
                    normalized = self._normalize_paper(paper, "arxiv")
                    all_papers.append(normalized)
            except Exception as e:
                error_msg = str(e)[:120]
                print(f"  -> arXiv: [FAIL] 失败 - {error_msg}")
                search_results["arxiv"] = 0
                source_errors["arxiv"] = error_msg
        
        # 搜索 Semantic Scholar
        if "semantic_scholar" in sources:
            try:
                print("[PAPERS] 从 Semantic Scholar 搜索...")
                result = self.semantic_tool.run({"query": query, "limit": 5})
                data = json.loads(result)
                if "error" in data:
                    raise Exception(data["error"])
                semantic_papers = data.get("papers", [])
                search_results["semantic_scholar"] = len(semantic_papers)
                print(f"  -> Semantic Scholar: 找到 {len(semantic_papers)} 篇论文")
                
                for paper in semantic_papers:
                    normalized = self._normalize_paper(paper, "semantic_scholar")
                    all_papers.append(normalized)
            except Exception as e:
                error_msg = str(e)[:120]
                print(f"  -> Semantic Scholar: [FAIL] 失败 - {error_msg}")
                search_results["semantic_scholar"] = 0
                source_errors["semantic_scholar"] = error_msg
        
        # 搜索 Google Scholar
        if "google_scholar" in sources:
            try:
                print("[PAPERS] 从 Google Scholar 搜索...")
                result = self.google_tool.run({"query": query, "limit": 5})
                data = json.loads(result)
                if "error" in data:
                    raise Exception(data["error"])
                google_papers = data.get("papers", [])
                search_results["google_scholar"] = len(google_papers)
                print(f"  -> Google Scholar: 找到 {len(google_papers)} 篇论文")
                
                for paper in google_papers:
                    normalized = self._normalize_paper(paper, "google_scholar")
                    all_papers.append(normalized)
            except Exception as e:
                error_msg = str(e)[:120]
                print(f"  -> Google Scholar: [FAIL] 失败 - {error_msg}")
                search_results["google_scholar"] = 0
                source_errors["google_scholar"] = error_msg
        
        # 去重和排序
        print("-" * 60)
        success_sources = [k for k, v in search_results.items() if v > 0]
        failed_sources = [k for k, v in search_results.items() if v == 0]
        print(f"搜索摘要: 成功=[{', '.join(success_sources) if success_sources else '无'}] 失败=[{', '.join(failed_sources) if failed_sources else '无'}]")
        print(f"总共获取 {len(all_papers)} 篇论文，正在去重和排序...")
        
        unique_papers = self._deduplicate_and_rank(all_papers)
        
        # 限制返回数量
        final_papers = unique_papers[:limit]
        
        print(f"[OK] 最终返回 {len(final_papers)} 篇论文")
        print("-" * 60 + "\n")
        
        result = {
            "query": query,
            "total_count": len(final_papers),
            "papers": final_papers,
            "source": "multi_source_search",
            "source_summary": {
                "success": success_sources,
                "failed": failed_sources,
            }
        }
        if source_errors:
            result["source_errors"] = source_errors
        
        return json.dumps(result, ensure_ascii=False, indent=2)
