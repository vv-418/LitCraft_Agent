# 作用：实现多源搜索融合器，整合 arXiv、Semantic Scholar、Google Scholar 的搜索结果，并去重排序。
import asyncio
import json
import hashlib
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from tools.base import Tool
from tools.search.arxiv_search import ArxivSearchTool
from tools.search.semantic_scholar import SemanticScholarTool
from tools.search.google_scholar import GoogleScholarTool
from tools.search.openalex import OpenAlexTool
from tools.search.crossref import CrossrefTool
from tools.search.europe_pmc import EuropePMCTool
from tools.search.search_cache import SearchCache
from tools.search.query_optimizer import QueryOptimizer
from tools.bilingual_query import build_search_queries, has_chinese

import chromadb


class MultiSourceSearchTool(Tool):
    """多源搜索融合器，整合多个学术搜索引擎的结果。"""

    name = "multi_source_search"
    description = "Search across academic databases (arXiv, OpenAlex, Crossref, Europe PMC, Semantic Scholar, optional Google Scholar) and merge results."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Maximum total papers to return after merge", "default": 10},
            "per_source_limit": {
                "type": "integer",
                "description": "Max papers to keep from each source per query round",
                "default": 10,
            },
            "sources": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "arxiv", "semantic_scholar", "google_scholar",
                        "openalex", "crossref", "europe_pmc", "all",
                    ],
                },
                "description": "Which sources to search. 'all' includes OpenAlex/Crossref/Europe PMC/arXiv/S2/Google Scholar (set SEARCH_INCLUDE_GOOGLE=0 to skip Scholar).",
                "default": ["all"]
            },
            "year_from": {"type": "string", "description": "Only return papers published from this year onwards (e.g., '2020')"},
        },
        "required": ["query"],
    }

    _HARD_LIMIT = 50

    def __init__(
        self,
        max_results: int = 10,
        per_source_limit: int = 10,
        use_cache: bool = True,
        use_semantic_dedup: bool = True,
        use_query_optimizer: bool = True,
        cache_ttl: int = 3600,
        auto_download: bool = True,
        auto_index: bool = False,
        papers_dir: str = "",
        figures_dir: str = "",
    ):
        """初始化 Multi Source Search Tool。

        Args:
            max_results: 合并后最多返回的论文数量
            per_source_limit: 每个学术源单次最多保留篇数
            use_cache: 是否启用搜索缓存
            use_semantic_dedup: 是否启用语义去重（标题 embedding 余弦相似度）
            use_query_optimizer: 是否启用查询清洗（停用短语剥离/长度归一化；双语用机器翻译）
            cache_ttl: 缓存 TTL（秒）
            auto_download: 是否自动下载论文 PDF
            auto_index: 是否自动解析 PDF 并存入向量数据库（auto_download=True 时生效）
            papers_dir: PDF 保存目录（综述任务传入 output/日期/主题/lit_source）
            figures_dir: 插图保存目录（空则用论文目录同级 figures）
        """
        self.max_results = max(1, min(int(max_results or 10), self._HARD_LIMIT))
        self.per_source_limit = max(1, min(int(per_source_limit or 10), self._HARD_LIMIT))
        self.use_cache = use_cache
        self.use_semantic_dedup = use_semantic_dedup
        self.use_query_optimizer = use_query_optimizer
        self.auto_download = auto_download
        self.auto_index = auto_index
        self.papers_dir = (papers_dir or "").strip()
        self.figures_dir = (figures_dir or "").strip()
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_skipped = 0
        self.semantic_dedup_removed = 0
        self.query_opt_modified = 0   # 查询被优化器修改的次数
        self.query_opt_bilingual = 0  # 触发双语搜索的次数

        self.arxiv_tool = ArxivSearchTool(max_results=self.per_source_limit)
        self.semantic_tool = SemanticScholarTool(max_results=self.per_source_limit)
        self.google_tool = GoogleScholarTool(max_results=self.per_source_limit)
        self.openalex_tool = OpenAlexTool(max_results=self.per_source_limit)
        self.crossref_tool = CrossrefTool(max_results=self.per_source_limit)
        self.europe_pmc_tool = EuropePMCTool(max_results=self.per_source_limit)

        # 查询优化器（始终初始化，统计信息始终可用）
        self._optimizer = QueryOptimizer()

        # 搜索缓存（惰性初始化，避免未启用缓存时创建文件）
        self._cache: Optional[SearchCache] = None
        self._cache_ttl = cache_ttl
        # Embedding 模型（惰性加载，语义去重专用）
        self._embedder = None

    def _get_cache(self) -> SearchCache:
        """惰性初始化 SearchCache。"""
        if self._cache is None:
            self._cache = SearchCache(
                db_path="./storage/search_cache.db",
                max_entries=500,
                ttl_seconds=self._cache_ttl,
            )
        return self._cache

    def _get_embedder(self):
        """惰性加载 sentence-transformer（共享单例）。"""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            from pathlib import Path
            # 直接使用本地缓存路径，避免 huggingface_hub 联网检查
            local_path = "./models/models--sentence-transformers--all-MiniLM-L6-v2"
            snapshot_dir = Path(local_path) / "snapshots"
            if snapshot_dir.exists():
                snapshots = sorted(snapshot_dir.iterdir())
                if snapshots:
                    model_path = str(snapshots[0])
                else:
                    model_path = "sentence-transformers/all-MiniLM-L6-v2"
            else:
                model_path = "sentence-transformers/all-MiniLM-L6-v2"
            self._embedder = SentenceTransformer(model_path, device="cpu")
        return self._embedder

    def _check_chroma_status(self, papers: list[dict]) -> None:
        """检查 Chroma 中是否已有这些论文的全文内容。

        为每篇论文添加 `chroma_indexed` 字段（True/False），
        让 LLM 知道可以直接用 advanced_search 检索，跳过下载→解析→分块。

        检查逻辑：用论文标题作为 collection_name 的前缀去 Chroma 查集合，
        如果该集合存在且有内容，就认为该论文已入库。
        """
        import re as _re
        try:
            client = chromadb.PersistentClient(
                path=str(self._chroma_db_path) if hasattr(self, "_chroma_db_path") else "./storage/chroma"
            )
            existing_collections = {c.name for c in client.list_collections()}
        except Exception:
            # Chroma 不可用时，标记所有论文为未索引
            for p in papers:
                p["chroma_indexed"] = False
            return

        for p in papers:
            title = p.get("title", "")
            if not title:
                p["chroma_indexed"] = False
                continue
            # 与 vector_store.py 中的清理逻辑一致
            clean = _re.sub(r'[^a-zA-Z0-9_\-\.]', '_', title)[:50].lower()
            found = any(
                existing_name == clean
                or clean in existing_name
                or existing_name in clean
                for existing_name in existing_collections
            )
            p["chroma_indexed"] = found

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
        elif source in ("openalex", "crossref", "europe_pmc"):
            display = {
                "openalex": "OpenAlex",
                "crossref": "Crossref",
                "europe_pmc": "Europe PMC",
            }[source]
            return {
                "title": paper.get("title", ""),
                "authors": paper.get("authors", []),
                "year": str(paper.get("year", "")),
                "abstract": paper.get("abstract", ""),
                "source": display,
                "source_id": paper.get("paper_id", "") or paper.get("doi", ""),
                "url": paper.get("url", ""),
                "pdf_url": paper.get("pdf_url", ""),
                "venue": paper.get("venue", ""),
                "citation_count": paper.get("citation_count", 0) or 0,
                "doi": paper.get("doi", ""),
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
        """两阶段去重：阶段1 精确MD5 → 阶段2 语义去重；排序贴近各源相关度序。

        Args:
            papers: 论文列表（append 顺序 ≈ 各源 API 返回先后）

        Returns:
            去重后的论文列表
        """
        # ── 阶段1：MD5 精确哈希去重（保留先出现的，合并多源；引用仅作同题替换）──
        dedup_dict: Dict[str, dict] = {}
        order: List[str] = []

        for idx, paper in enumerate(papers):
            paper = dict(paper)
            paper.setdefault("_appear_idx", idx)
            paper_hash = self._get_paper_hash(paper)

            if paper_hash not in dedup_dict:
                dedup_dict[paper_hash] = paper
                order.append(paper_hash)
            else:
                existing = dedup_dict[paper_hash]
                # 合并来源
                existing_sources = list(existing.get("sources") or [existing.get("source", "")])
                paper_source = paper.get("source", "")
                if paper_source and paper_source not in existing_sources:
                    existing_sources.append(paper_source)
                existing["sources"] = [s for s in existing_sources if s]
                # 同题时可用引用更高的元数据补全，但不打乱首次出现序
                if (paper.get("citation_count") or 0) > (existing.get("citation_count") or 0):
                    for k in ("citation_count", "abstract", "pdf_url", "url", "year", "authors"):
                        if paper.get(k):
                            existing[k] = paper[k]

        result = [dedup_dict[h] for h in order]

        # ── 阶段2：语义去重（标题 embedding 余弦相似度） ──────
        if self.use_semantic_dedup and len(result) > 1:
            result = self._semantic_dedup(result)

        # ── 排序：多源命中优先 → 保持出现序（≈源相关度）→ 引用/年份仅 tie-break ──
        result.sort(
            key=lambda x: (
                -len(x.get("sources") or [x.get("source", "")]),
                int(x.get("_appear_idx", 10**9)),
                -(x.get("citation_count") or 0),
                -int(str(x.get("year") or "0") or "0"),
            )
        )
        for p in result:
            p.pop("_appear_idx", None)

        return result

    def _semantic_dedup(self, papers: List[dict]) -> List[dict]:
        """标题语义去重：余弦相似度 > 0.85 视为重复，保留引用数高的。"""
        if len(papers) <= 1:
            return papers

        try:
            model = self._get_embedder()
            titles = [p.get("title", "") or "" for p in papers]
            # 过滤空标题
            valid_indices = [i for i, t in enumerate(titles) if t.strip()]
            if len(valid_indices) <= 1:
                return papers

            valid_titles = [titles[i] for i in valid_indices]
            embeddings = model.encode(valid_titles, normalize_embeddings=True)

            keep = [True] * len(valid_indices)
            for i in range(len(valid_indices)):
                if not keep[i]:
                    continue
                for j in range(i + 1, len(valid_indices)):
                    if not keep[j]:
                        continue
                    # 余弦相似度 = 归一化向量的点积
                    sim = float(embeddings[i] @ embeddings[j])
                    if sim > 0.85:
                        # 保留引用数高的
                        idx_i = valid_indices[i]
                        idx_j = valid_indices[j]
                        ci = papers[idx_i].get("citation_count", 0) or 0
                        cj = papers[idx_j].get("citation_count", 0) or 0
                        if cj > ci:
                            keep[i] = False
                            break  # i 被淘汰
                        else:
                            keep[j] = False  # j 被淘汰

            # 保留未被淘汰的
            kept_indices = {valid_indices[i] for i in range(len(valid_indices)) if keep[i]}
            before = len(papers)
            result = [p for i, p in enumerate(papers) if i in kept_indices]
            removed = before - len(result)
            if removed > 0:
                self.semantic_dedup_removed += removed
                print(f"  [DEDUP] 语义去重: 移除 {removed} 篇近似重复论文")
            return result

        except Exception as e:
            # 语义去重失败不应阻断流程
            print(f"  [DEDUP] 语义去重跳过（{str(e)[:60]}）")
            return papers

    def _filter_by_topic_relevance(
        self, papers: List[dict], original_query: str, threshold: float = 0.22,
        min_keep: int = 3,
    ) -> List[dict]:
        """按「标题+摘要」与原始主题的 embedding 相似度过滤并打分。

        过滤掉明显偏题论文；通过的论文写入 topic_relevance，供后续排序。
        若硬阈值后不足 min_keep 篇，按分数保留 Top-K（软保底，避免 0 篇）。
        """
        if not papers or not original_query:
            return papers

        min_keep = max(0, min(int(min_keep or 0), len(papers)))

        def _rank(scored: List[dict]) -> List[dict]:
            scored.sort(
                key=lambda x: (
                    -int(self._is_downloadable_pdf_url(x.get("pdf_url", "") or "")),
                    -float(x.get("topic_relevance", 0) or 0),
                    -len(x.get("sources", [x.get("source", "")])),
                    -int(x.get("citation_count", 0) or 0),
                    -int(x.get("year", "0") or "0"),
                )
            )
            return scored

        try:
            model = self._get_embedder()
            query_emb = model.encode([original_query], normalize_embeddings=True)[0]

            scored: List[dict] = []
            dropped = 0
            for paper in papers:
                title = paper.get("title", "") or ""
                abstract = (paper.get("abstract", "") or "")[:400]
                paper_text = f"{title}. {abstract}".strip()
                if not paper_text:
                    paper["topic_relevance"] = 0.0
                    scored.append(paper)
                    continue

                paper_emb = model.encode([paper_text], normalize_embeddings=True)[0]
                sim = float(query_emb @ paper_emb)
                paper["topic_relevance"] = round(sim, 4)
                if sim >= threshold:
                    scored.append(paper)
                else:
                    dropped += 1
                    paper["_below_threshold"] = True
                    scored.append(paper)  # 暂存，后面按需保底

            passed = [p for p in scored if not p.pop("_below_threshold", False)]
            if dropped:
                print(f"  [TOPIC] embedding 过滤偏题 {dropped} 篇（阈值={threshold}）")

            if len(passed) >= min_keep:
                return _rank(passed)

            # 软保底：硬过滤后太少时，按分数取 Top min_keep（含刚低于阈值的）
            all_ranked = _rank(list(scored))
            keep_n = max(min_keep, len(passed))
            soft = all_ranked[:keep_n]
            print(
                f"  [TOPIC] 硬过滤后仅 {len(passed)} 篇，软保底保留 Top-{len(soft)} "
                f"(最高分={soft[0].get('topic_relevance') if soft else 'n/a'})"
            )
            return soft

        except Exception as e:
            print(f"  [TOPIC] 主题过滤降级到 token 相似度: {e}")
            from tools.search.query_optimizer import QueryOptimizer

            filtered = []
            for paper in papers:
                title = paper.get("title", "") or ""
                abstract = (paper.get("abstract", "") or "")[:300]
                paper_text = f"{title} {abstract}".strip()
                if not paper_text:
                    paper["topic_relevance"] = 0.0
                    filtered.append(paper)
                    continue
                sim = QueryOptimizer.semantic_similarity(original_query, paper_text)
                paper["topic_relevance"] = round(sim, 4)
                filtered.append(paper)
            filtered = _rank(filtered)
            passed = [p for p in filtered if float(p.get("topic_relevance", 0) or 0) >= threshold]
            if len(passed) >= min_keep:
                return passed
            return filtered[: max(min_keep, len(passed))]

    def run(self, tool_input: dict[str, Any]) -> str:
        """执行多源搜索（异步并发实现，向下兼容）。"""
        # 安全调用 asyncio.run()：检测是否已有运行中的事件循环
        try:
            asyncio.get_running_loop()
            # 已有事件循环 → 在线程池中执行
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self._async_run(tool_input)).result()
        except RuntimeError:
            # 无事件循环 → 直接使用 asyncio.run()
            return asyncio.run(self._async_run(tool_input))

    # ── 异步并发核心 ──────────────────────────────────────────

    # 各来源超时配置（秒）
    _SOURCE_TIMEOUTS = {
        "arxiv": 60,
        "semantic_scholar": 40,
        "google_scholar": 70,
        "openalex": 25,
        "crossref": 25,
        "europe_pmc": 25,
    }
    _KNOWN_SOURCES = (
        "arxiv", "semantic_scholar", "google_scholar",
        "openalex", "crossref", "europe_pmc",
    )

    @staticmethod
    def _is_downloadable_pdf_url(url: str) -> bool:
        u = (url or "").strip().lower()
        if not u.startswith("http"):
            return False
        # S2 / Scholar 落地页不是 PDF
        bad_markers = (
            "semanticscholar.org/paper/",
            "scholar.google.",
            "/abs/",
            "login",
            "signin",
        )
        if any(m in u for m in bad_markers):
            return False
        return True

    def _queries_for_source(self, source: str, query_rounds: List[str]) -> List[str]:
        """按来源选查询：arXiv / Crossref / OpenAlex 更认英文。"""
        english_first = source in ("arxiv", "openalex", "crossref")
        if not english_first:
            return list(query_rounds)
        latin = [q for q in query_rounds if q and not has_chinese(q)]
        if latin:
            return latin
        from tools.bilingual_query import translate_to_english
        zh = next((q for q in query_rounds if has_chinese(q)), "")
        if zh:
            en = translate_to_english(zh)
            if en and not has_chinese(en):
                print(f"  [{source.upper()}] 中文主题改用英译检索: {en[:80]}")
                return [en]
        print(f"  [{source.upper()}] 无可用英文查询，跳过本轮")
        return []

    async def _async_run(self, tool_input: dict[str, Any]) -> str:
        """异步并发执行多源搜索，通过 asyncio.gather() 同时发起三个搜索请求。"""
        query = str(tool_input.get("query", "")).strip()
        limit = max(1, min(int(tool_input.get("limit", self.max_results) or self.max_results), self._HARD_LIMIT))
        per_source = max(
            1,
            min(
                int(tool_input.get("per_source_limit", self.per_source_limit) or self.per_source_limit),
                self._HARD_LIMIT,
            ),
        )
        # 单源工具用 max_results 做上限，按本次请求抬高以免被截断
        for _tool in (
            self.arxiv_tool, self.semantic_tool, self.google_tool,
            self.openalex_tool, self.crossref_tool, self.europe_pmc_tool,
        ):
            _tool.max_results = max(getattr(_tool, "max_results", 1), per_source)
        sources = tool_input.get("sources", ["all"])

        # ── 来源名归一化：LLM 常用显示名而不是内部名 ─────────
        _SOURCE_DISPLAY_TO_INTERNAL = {
            "arxiv": "arxiv",
            "google scholar": "google_scholar",
            "semantic scholar": "semantic_scholar",
            "google_scholar": "google_scholar",
            "semantic_scholar": "semantic_scholar",
            "openalex": "openalex",
            "open alex": "openalex",
            "crossref": "crossref",
            "europe pmc": "europe_pmc",
            "europe_pmc": "europe_pmc",
            "pubmed": "europe_pmc",
            "pmc": "europe_pmc",
            "all": "all",
        }
        if isinstance(sources, str):
            sources = [sources]
        normalized = []
        for s in sources:
            key = s.strip().lower()
            mapped = _SOURCE_DISPLAY_TO_INTERNAL.get(key, key)
            if mapped:  # 排除完全不认识的值
                normalized.append(mapped)

        if "all" in normalized or not normalized:
            normalized = ["openalex", "crossref", "europe_pmc", "arxiv", "semantic_scholar"]
            include_gs = os.getenv("SEARCH_INCLUDE_GOOGLE", "1").strip().lower() not in (
                "0", "false", "no", "off",
            )
            if include_gs:
                normalized.append("google_scholar")
            else:
                print("[SEARCH] 跳过 Google Scholar（SEARCH_INCLUDE_GOOGLE=0）")
        sources = list(dict.fromkeys(normalized))

        # ── 中文查询强制加入 arXiv ──
        if re.search(r'[\u4e00-\u9fff]', query) and "arxiv" not in sources:
            sources.append("arxiv")
            print(f"[FORCE] 中文查询 → 自动加入 arXiv 搜索")

        if not query:
            return json.dumps({"error": "Query cannot be empty"}, ensure_ascii=False)

        # ── 查询清洗 + 机器翻译多查询（不调大模型） ──────────
        original_topic = query
        if self.use_query_optimizer:
            opt_result = self._optimizer.optimize(query)
            if opt_result["was_modified"]:
                self.query_opt_modified += 1
                print(f"  [OPTIM] 查询已清洗: '{query[:60]}' → '{opt_result['optimized'][:60]}'")
                query = opt_result["optimized"]

        # 原主题 + 合格英译（中文双路）；英文仅原句 —— 对齐网站搜索框
        query_rounds = build_search_queries(query, max_queries=2)
        if len(query_rounds) > 1:
            self.query_opt_bilingual += 1
            print(f"  [BI-LING] 中文双路（原句+英译）: {query_rounds}")
        else:
            print(f"  [QUERY] 英文单路 / 仅原句: {query_rounds}")

        print(f"\n[SEARCH] ⚡ 并发多源搜索启动")
        print(f"   主题: {original_topic[:80]}")
        print(f"   查询轮次: {query_rounds}")
        print(f"   来源: {', '.join(sources)}")
        print(f"   单源上限: {per_source} | 合并后保留: {limit}")
        print("-" * 60)

        all_papers: list[dict] = []
        search_results: dict[str, int] = {}
        source_errors: dict[str, str] = {}

        async def _search_one(source: str, q: str) -> tuple[str, list[dict], str | None]:
            """并发执行单个来源的搜索（带缓存）。"""
            tool_map = {
                "arxiv": self.arxiv_tool,
                "semantic_scholar": self.semantic_tool,
                "google_scholar": self.google_tool,
                "openalex": self.openalex_tool,
                "crossref": self.crossref_tool,
                "europe_pmc": self.europe_pmc_tool,
            }
            tool = tool_map[source]
            timeout = self._SOURCE_TIMEOUTS.get(source, 30)
            year_from = str(tool_input.get("year_from", "")).strip()

            # ── 缓存检查 ────────────────────────────────────────
            if self.use_cache:
                cached = self._get_cache().get(source, q, year_from)
                if cached is not None:
                    self.cache_hits += 1
                    try:
                        data = json.loads(cached)
                        raw_papers = (
                            data.get("results", []) if source == "google_scholar"
                            else data.get("papers", [])
                        )
                        normalized = [self._normalize_paper(p, source) for p in raw_papers]
                        # ── 桥梁：检查 Chroma 中是否已有这些论文的内容 ──
                        try:
                            self._check_chroma_status(normalized)
                        except Exception:
                            pass
                        print(f"  -> {source}: 命中缓存 | {len(normalized)} 篇")
                        return source, normalized, None
                    except Exception:
                        pass  # 缓存损坏则回退到真实搜索
                else:
                    self.cache_misses += 1
            else:
                self.cache_skipped += 1

            print(f"  [PAPERS] 从 {source} 搜索（超时 {timeout}s）...")
            search_start = time.perf_counter()

            try:
                result_str = await asyncio.wait_for(
                    tool.async_run({"query": q, "limit": per_source, "year_from": year_from}),
                    timeout=timeout,
                )
                elapsed = time.perf_counter() - search_start

                data = json.loads(result_str)

                # 错误检查
                if "error" in data:
                    raise Exception(data["error"])

                # ── 写入缓存 ─────────────────────────────────────
                if self.use_cache:
                    self._get_cache().set(source, q, year_from, result_str)

                # 各来源返回格式不同
                raw_papers = (
                    data.get("results", []) if source == "google_scholar"
                    else data.get("papers", [])
                )

                normalized = [self._normalize_paper(p, source) for p in raw_papers]
                print(f"  -> {source}: 找到 {len(normalized)} 篇 ({elapsed:.1f}s)")
                return source, normalized, None

            except asyncio.TimeoutError:
                elapsed = time.perf_counter() - search_start
                print(f"  -> {source}: ⏱ TIMEOUT 超过 {timeout}s")
                return source, [], f"Timeout after {timeout}s"
            except Exception as e:
                elapsed = time.perf_counter() - search_start
                msg = str(e)[:120]
                print(f"  -> {source}: FAIL {msg} ({elapsed:.1f}s)")
                return source, [], msg

        # ── 并发发射所有搜索（支持多轮查询：中文+英文） ────────
        for round_idx, current_query in enumerate(query_rounds):
            if round_idx > 0:
                # 第二轮（双语）时加分隔线标记
                print(f"  ── 双语第 {round_idx + 1} 轮: '{current_query[:60]}' ──")

            tasks = []
            for s in sources:
                if s not in self._KNOWN_SOURCES:
                    continue
                src_queries = self._queries_for_source(s, [current_query])
                if not src_queries:
                    continue
                tasks.append(_search_one(s, src_queries[0]))
            if not tasks:
                continue
            round_results = await asyncio.gather(*tasks)

            # 汇总本轮结果
            for source, papers, error in round_results:
                all_papers.extend(papers)
                search_results.setdefault(source, 0)
                search_results[source] += len(papers)
                if error:
                    source_errors[source] = error

        # ── 失败来源重试：如果总论文数 < 3 且有来源失败 ────────
        total_before_retry = len(all_papers)
        if total_before_retry < 3 and source_errors:
            retry_sources = [s for s in sources if s in source_errors]
            if retry_sources:
                print(f"\n[RETRY] 论文数太少({total_before_retry})，重试失败的来源: {retry_sources}")
                retry_query = next(
                    (q for q in reversed(query_rounds) if q and not has_chinese(q)),
                    query_rounds[-1] if query_rounds else query,
                )
                retry_tasks = []
                for s in retry_sources:
                    qs = self._queries_for_source(s, query_rounds) or [retry_query]
                    retry_tasks.append(_search_one(s, qs[-1]))
                if retry_tasks:
                    import copy
                    saved = copy.copy(self._SOURCE_TIMEOUTS)
                    for s in retry_sources:
                        self._SOURCE_TIMEOUTS[s] = int(self._SOURCE_TIMEOUTS.get(s, 30) * 1.5)
                    retry_results = await asyncio.gather(*retry_tasks)
                    self._SOURCE_TIMEOUTS = saved
                    for source, papers, error in retry_results:
                        if papers:
                            all_papers.extend(papers)
                            search_results[source] = search_results.get(source, 0) + len(papers)
                            source_errors.pop(source, None)
                            print(f"  [RETRY] {source}: 重试成功，额外获取 {len(papers)} 篇")
                        else:
                            print(f"  [RETRY] {source}: 重试仍失败")

        # ── 去重和排序 ──────────────────────────────────────────
        print("-" * 60)
        success_sources = [k for k, v in search_results.items() if v > 0]
        failed_sources = [k for k, v in search_results.items() if v == 0]
        print(f"搜索摘要: 成功=[{', '.join(success_sources) if success_sources else '无'}] 失败=[{', '.join(failed_sources) if failed_sources else '无'}]")
        print(f"共获取 {len(all_papers)} 篇论文（{len(query_rounds)} 轮查询: {' + '.join(q[:40] for q in query_rounds)}），去重和排序中...")

        unique_papers = self._deduplicate_and_rank(all_papers)

        # ── 主题相关性过滤（默认开启，去掉明显偏题结果）──
        # SEARCH_TOPIC_FILTER=0 可关闭（仅对齐「不砍官网结果」调试场景）
        topic_filter_on = os.getenv("SEARCH_TOPIC_FILTER", "1").strip().lower() not in (
            "0", "false", "no", "off",
        )
        if unique_papers and topic_filter_on:
            # 优先用英译做 embedding 过滤（中文主题 vs 英文标题更稳）
            filter_query = next(
                (q for q in query_rounds if q and not has_chinese(q)),
                str(tool_input.get("query", original_topic)).strip() or original_topic,
            )
            before_topic_filter = len(unique_papers)
            min_keep = max(3, min(limit, 5))
            unique_papers = self._filter_by_topic_relevance(
                unique_papers, filter_query, threshold=0.22, min_keep=min_keep
            )
            topic_filtered = before_topic_filter - len(unique_papers)
            if topic_filtered > 0:
                print(f"[TOPIC] 主题过滤: 移除 {topic_filtered} 篇偏题论文 "
                      f"({before_topic_filter} -> {len(unique_papers)} 篇) | 过滤查询={filter_query[:60]}")
        elif unique_papers:
            print("[TOPIC] 主题过滤关闭（SEARCH_TOPIC_FILTER=0）")

        # 年份过滤
        year_from = str(tool_input.get("year_from", "")).strip()
        if year_from and year_from.isdigit():
            year_from_int = int(year_from)
            before = len(unique_papers)
            unique_papers = [
                p for p in unique_papers
                if p.get("year", "0").isdigit() and int(p.get("year", "0")) >= year_from_int
            ]
            print(f"[FILTER] 年份过滤: >= {year_from} ({before} -> {len(unique_papers)} 篇)")

        do_download = bool(tool_input.get("auto_download", self.auto_download))
        if do_download and not self.papers_dir:
            print("[DOWNLOAD] 未指定 papers_dir，跳过下载")
            do_download = False
        # 可下载 PDF 优先；自动下载时多取候选
        unique_papers.sort(
            key=lambda p: (
                -int(self._is_downloadable_pdf_url(p.get("pdf_url", "") or "")),
                -float(p.get("topic_relevance", 0) or 0),
                -int(p.get("citation_count", 0) or 0),
            )
        )
        candidate_cap = max(limit * 3, limit) if do_download else limit
        candidates = unique_papers[:candidate_cap]
        final_papers = candidates[:limit] if not do_download else []

        # ═══════════════════════════════════════════════════════════
        #  阶段1: 并发下载 PDF（4 线程）；只把「有正文」的计入最终列表
        # ═══════════════════════════════════════════════════════════
        downloaded = []
        if do_download:
            print(
                f"[DOWNLOAD] 候选池 {len(candidates)} 篇，4 线程；"
                f"凑够 {limit} 篇有正文即停，不再下剩余候选"
            )
            from tools.paper_downloader import PaperDownloaderTool
            from utils.review_output import paper_export_filename
            dl = PaperDownloaderTool(storage_path=self.papers_dir)
            stop_download = threading.Event()

            def _download_one(p: dict) -> tuple[str, str | None, dict]:
                title = p.get("title", "")[:60]
                p["downloaded"] = False
                p["has_content"] = False
                if stop_download.is_set():
                    return title, None, p
                pdf_url = (p.get("pdf_url", "") or "").strip()
                page_url = (p.get("url", "") or "").strip()
                # 绝不把 S2/Scholar 落地页当 PDF
                if self._is_downloadable_pdf_url(pdf_url):
                    url = pdf_url
                elif self._is_downloadable_pdf_url(page_url):
                    url = page_url
                else:
                    url = ""
                if not url:
                    print(f"    ⏭ {title}: 无可用 PDF URL，跳过下载")
                    return title, None, p
                if stop_download.is_set():
                    return title, None, p
                try:
                    save_name = paper_export_filename(
                        str(p.get("title") or "paper"),
                        url,
                    )
                    r = dl.run({"url": url, "filename": save_name})
                    dl_result = json.loads(r) if isinstance(r, str) and r.startswith("{") else {"status": "unknown"}
                    status = dl_result.get("status")
                    if status in ("success", "already_exists") and dl_result.get("has_content", True):
                        path = dl_result.get("path", "")
                        if path and os.path.exists(path) and os.path.getsize(path) >= 100:
                            p["downloaded"] = True
                            p["has_content"] = True
                            p["local_path"] = path
                            return title, dl_result.get("filename", ""), p
                    return title, None, p
                except Exception:
                    return title, None, p

            max_workers = 4
            next_i = 0
            pending: dict = {}
            submitted = 0

            def _submit_one(executor: ThreadPoolExecutor) -> None:
                nonlocal next_i, submitted
                if stop_download.is_set() or next_i >= len(candidates):
                    return
                paper = candidates[next_i]
                next_i += 1
                submitted += 1
                pending[executor.submit(_download_one, paper)] = paper

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for _ in range(min(max_workers, len(candidates))):
                    _submit_one(executor)
                while pending:
                    done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                    for future in done:
                        pending.pop(future, None)
                        try:
                            title, fname, p = future.result()
                        except Exception:
                            continue
                        if fname and p.get("has_content"):
                            print(f"    ✅ {title} -> {fname}")
                            downloaded.append(fname)
                        elif not stop_download.is_set():
                            print(f"    ❌ {title}（无有效正文或下载失败）")

                        n_ok = sum(1 for c in candidates if c.get("has_content"))
                        if n_ok >= limit and not stop_download.is_set():
                            stop_download.set()
                            for queued in list(pending):
                                if queued.cancel():
                                    pending.pop(queued, None)
                            skipped = len(candidates) - next_i
                            print(
                                f"[DOWNLOAD] 已凑够 {limit} 篇有正文，停止提交"
                                f"（未开始 {skipped} 篇，进行中最多 {len(pending)} 篇会下完）"
                            )
                        if not stop_download.is_set():
                            _submit_one(executor)

            with_content = [p for p in candidates if p.get("has_content")]
            final_papers = with_content[:limit]
            print(
                f"[DOWNLOAD] 完成：尝试 {submitted} / 候选 {len(candidates)} 篇，"
                f"有正文 {len(with_content)} → 最终保留 {len(final_papers)} 篇"
            )
            if not final_papers and candidates:
                print("[DOWNLOAD] ⚠ 候选均无有效正文，最终列表为空（不再返回空壳文献）")
        else:
            final_papers = candidates[:limit]

        # ═══════════════════════════════════════════════════════════
        #  阶段2: 并发解析 → 分块 → 批量向量入库（4 线程流水线）
        # ═══════════════════════════════════════════════════════════
        indexed_count = 0
        collection_name = ""
        if tool_input.get("auto_index", self.auto_index):
            from tools.pdf_parser import PDFParserTool
            from tools.text_chunker import TextChunkerTool

            parser_tool = PDFParserTool(
                storage_path=self.papers_dir,
                figures_path=self.figures_dir,
            )
            chunker_tool = TextChunkerTool(chunk_size=512, chunk_overlap=128)

            all_chunks: list[dict] = []
            all_images: list[dict] = []
            index_lock = threading.Lock()
            downloadable = [p for p in final_papers if p.get("downloaded") and p.get("local_path")]

            if downloadable:
                print(f"[INDEX] 🔍 并发解析+分块 {len(downloadable)} 篇论文（4 线程）...")

                def _index_one(paper: dict) -> None:
                    """单篇论文：解析 → 分块 / 抽图（在线程池中并行执行）。"""
                    title = paper.get("title", "")[:60]
                    local_path = paper.get("local_path", "")
                    if not local_path or not os.path.exists(local_path):
                        return
                    try:
                        parse_result = parser_tool.run({"pdf_path": local_path})
                        data = json.loads(parse_result) if isinstance(parse_result, str) else parse_result
                        if not isinstance(data, dict) or data.get("status") == "error":
                            print(f"    ⚠ {title}: 解析失败")
                            return

                        content = data.get("content", "")
                        full_text = ""
                        images: list = []
                        if isinstance(content, dict):
                            full_text = content.get("full_text", "") or ""
                            images = list(content.get("images") or [])
                        elif isinstance(content, str):
                            full_text = content

                        if images:
                            for im in images:
                                im["source"] = Path(local_path).name
                            with index_lock:
                                all_images.extend(images)
                            print(f"    🖼️  {title}: {len(images)} 张插图")

                        if not (full_text or "").strip():
                            print(f"    ⚠ {title}: 解析后无正文，跳过入库")
                            paper["has_content"] = False
                            return

                        chunk_result = chunker_tool.run({
                            "text": full_text,
                            "source": Path(local_path).name,
                            "metadata": {
                                "title": paper.get("title", ""),
                                "authors": ", ".join(paper.get("authors", [])),
                                "year": paper.get("year", ""),
                                "path": local_path,
                            },
                        })
                        chunk_data = json.loads(chunk_result) if isinstance(chunk_result, str) else chunk_result
                        chunks = chunk_data.get("chunks", []) if isinstance(chunk_data, dict) else []

                        if chunks:
                            with index_lock:
                                all_chunks.extend(chunks)
                            paper["indexed_chunks"] = len(chunks)
                            print(f"    📄 {title}: {len(chunks)} 个块")
                        else:
                            paper["has_content"] = False
                            print(f"    ⚠ {title}: 分块为空")
                    except Exception as e:
                        print(f"    ❌ {title}: 索引失败 - {e}")

                with ThreadPoolExecutor(max_workers=4) as executor:
                    list(executor.map(_index_one, downloadable))

                # ── 批量写入向量数据库 ──
                collection_name = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff_\-]', '_', query)[:50]
                store_tool = None
                if all_chunks or all_images:
                    from tools.vector_store import VectorStoreTool
                    store_tool = VectorStoreTool(db_path="./storage/chroma", local_model_cache="./models")

                if all_chunks and store_tool is not None:
                    store_result = store_tool.run({
                        "action": "add",
                        "collection_name": collection_name,
                        "chunks": all_chunks,
                    })
                    store_data = json.loads(store_result) if isinstance(store_result, str) else {}
                    added = store_data.get("chunks_added", 0) if isinstance(store_data, dict) else 0
                    indexed_count = added
                    print(f"[INDEX] ✅ 向量入库完成: {added} 个块 → 集合 '{collection_name}'")
                    try:
                        indexed_titles = [p.get("title", "") for p in downloaded]
                        indexed_titles = [t for t in indexed_titles if t]
                        if indexed_titles and self.use_cache:
                            for source_name in set(p.get("source", "") for p in downloaded):
                                source_key = {
                                    "arXiv": "arxiv", "Semantic Scholar": "semantic_scholar",
                                    "Google Scholar": "google_scholar",
                                    "OpenAlex": "openalex", "Crossref": "crossref",
                                    "Europe PMC": "europe_pmc",
                                }.get(source_name, source_name)
                                year_from = str(tool_input.get("year_from", "")).strip()
                                for q in query_rounds:
                                    self._get_cache().update_chroma_status(source_key, q, year_from, indexed_titles)
                            print(f"[CACHE] 已标记 {len(indexed_titles)} 篇论文的 Chroma 状态")
                    except Exception:
                        pass
                elif not all_chunks:
                    print(f"[INDEX] ⏭ 无有效文本块可入库")

                if all_images and store_tool is not None:
                    img_result = store_tool.run({
                        "action": "add_images",
                        "collection_name": collection_name,
                        "images": all_images,
                    })
                    img_data = json.loads(img_result) if isinstance(img_result, str) else {}
                    print(
                        f"[INDEX] 🖼️ 多模态入库: status={img_data.get('status')} "
                        f"images={img_data.get('images', 0)} → '{collection_name}_mm'"
                    )
            else:
                print(f"[INDEX] ⏭ 没有已下载的论文可索引")

        # 自动下载场景：最终只返回仍有有效正文的论文（索引失败且无正文的剔除）
        if do_download:
            before_ret = len(final_papers)
            final_papers = [p for p in final_papers if p.get("has_content")]
            if before_ret != len(final_papers):
                print(f"[FILTER] 剔除无正文论文: {before_ret} -> {len(final_papers)} 篇")

        print(f"[OK] 最终返回 {len(final_papers)} 篇论文 "
              f"(已下载 {len(downloaded)} 篇, 已索引 {indexed_count} 个块)")
        print("-" * 60 + "\n")

        result = {
            "query": query,
            "total_count": len(final_papers),
            "papers": final_papers,
            "source": "multi_source_search",
            "source_summary": {
                "success": success_sources,
                "failed": failed_sources,
            },
            "indexed_chunks": indexed_count,
        }
        if collection_name:
            result["collection_name"] = collection_name
        if source_errors:
            result["source_errors"] = source_errors

        return json.dumps(result, ensure_ascii=False, indent=2)
