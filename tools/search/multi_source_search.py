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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tools.base import Tool
from tools.search.arxiv_search import ArxivSearchTool
from tools.search.semantic_scholar import SemanticScholarTool
from tools.search.google_scholar import GoogleScholarTool
from tools.search.search_cache import SearchCache
from tools.search.query_optimizer import QueryOptimizer

import chromadb


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
            "year_from": {"type": "string", "description": "Only return papers published from this year onwards (e.g., '2020')"},
        },
        "required": ["query"],
    }

    def __init__(
        self,
        max_results: int = 10,
        use_cache: bool = True,
        use_semantic_dedup: bool = True,
        use_query_optimizer: bool = True,
        cache_ttl: int = 3600,
        auto_download: bool = True,
        auto_index: bool = False,
    ):
        """初始化 Multi Source Search Tool。

        Args:
            max_results: 最多返回的论文数量
            use_cache: 是否启用搜索缓存
            use_semantic_dedup: 是否启用语义去重（标题 embedding 余弦相似度）
            use_query_optimizer: 是否启用 LLM 查询优化（停用短语剥离/双语候选/长度归一化）
            cache_ttl: 缓存 TTL（秒）
            auto_download: 是否自动下载论文 PDF
            auto_index: 是否自动解析 PDF 并存入向量数据库（auto_download=True 时生效）
        """
        self.max_results = max_results
        self.use_cache = use_cache
        self.use_semantic_dedup = use_semantic_dedup
        self.use_query_optimizer = use_query_optimizer
        self.auto_download = auto_download
        self.auto_index = auto_index
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_skipped = 0
        self.semantic_dedup_removed = 0
        self.query_opt_modified = 0   # 查询被优化器修改的次数
        self.query_opt_bilingual = 0  # 触发双语搜索的次数

        self.arxiv_tool = ArxivSearchTool(max_results=10)
        self.semantic_tool = SemanticScholarTool(max_results=10)
        self.google_tool = GoogleScholarTool(max_results=10)

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
        else:
            return paper

    def _get_paper_hash(self, paper: dict) -> str:
        """生成论文的哈希值用于去重（基于标题）。"""
        title = paper.get("title", "").lower().strip()
        # 移除一些常见的停用词以提高匹配度
        title = title.replace(" the ", " ").replace(" a ", " ").replace(" and ", " ")
        return hashlib.md5(title.encode()).hexdigest()

    def _deduplicate_and_rank(self, papers: List[dict]) -> List[dict]:
        """两阶段去重：阶段1 精确MD5 → 阶段2 语义去重。

        Args:
            papers: 论文列表

        Returns:
            去重并按相关性排序的论文列表
        """
        # ── 阶段1：MD5 精确哈希去重 ────────────────────────────
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

        result = list(dedup_dict.values())

        # ── 阶段2：语义去重（标题 embedding 余弦相似度） ──────
        if self.use_semantic_dedup and len(result) > 1:
            result = self._semantic_dedup(result)

        # ── 排序：多源优先 → 引用数多优先 → 最新优先 ───────────
        result.sort(
            key=lambda x: (
                -len(x.get("sources", [x.get("source", "")])),  # 多源优先
                -x.get("citation_count", 0),  # 引用数多优先
                -int(x.get("year", "0") or "0"),  # 最新优先
            )
        )

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
        self, papers: List[dict], original_query: str, threshold: float = 0.15
    ) -> List[dict]:
        """按主题相关性过滤论文，移除与原始查询主题偏离的结果。

        使用轻量级 token 余弦相似度（与 QueryOptimizer 同一算法），
        计算每篇论文的「标题 + 摘要」与原始查询的语义相似度，
        过滤掉低于阈值的结果。

        Args:
            papers: 去重排序后的论文列表
            original_query: 用户的原始查询（用于语义相似度计算）
            threshold: 相似度阈值，低于此值的论文被移除

        Returns:
            过滤后的论文列表
        """
        if not papers or not original_query:
            return papers

        from tools.search.query_optimizer import QueryOptimizer

        filtered = []
        for paper in papers:
            # 构造论文的语义表示文本
            title = paper.get("title", "") or ""
            abstract = paper.get("abstract", "") or ""
            # 取摘要前 300 字符（避免过长文本影响性能）
            abstract_short = abstract[:300] if abstract else ""
            paper_text = f"{title} {abstract_short}".strip()

            if not paper_text:
                # 无标题无摘要的论文保留（无法判断相关性）
                filtered.append(paper)
                continue

            sim = QueryOptimizer.semantic_similarity(original_query, paper_text)
            if sim >= threshold:
                paper["_topic_similarity"] = round(sim, 4)
                filtered.append(paper)
            # else: sim < threshold, 主题偏离，移除

        return filtered

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
        "arxiv": 60,            # 指数退避最多 5 次，最慢约 60s
        "semantic_scholar": 40,
        "google_scholar": 50,
    }

    async def _async_run(self, tool_input: dict[str, Any]) -> str:
        """异步并发执行多源搜索，通过 asyncio.gather() 同时发起三个搜索请求。"""
        query = str(tool_input.get("query", "")).strip()
        limit = int(tool_input.get("limit", self.max_results))
        sources = tool_input.get("sources", ["all"])

        # ── 来源名归一化：LLM 常用显示名而不是内部名 ─────────
        _SOURCE_DISPLAY_TO_INTERNAL = {
            "arxiv": "arxiv",
            "google scholar": "google_scholar",
            "semantic scholar": "semantic_scholar",
            "google_scholar": "google_scholar",
            "semantic_scholar": "semantic_scholar",
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
            normalized = ["arxiv", "semantic_scholar", "google_scholar"]
        sources = normalized

        # ── 中文查询强制加入 arXiv ──
        if re.search(r'[\u4e00-\u9fff]', query) and "arxiv" not in sources:
            sources.append("arxiv")
            print(f"[FORCE] 中文查询 → 自动加入 arXiv 搜索")

        if not query:
            return json.dumps({"error": "Query cannot be empty"}, ensure_ascii=False)

        # ── 查询优化（双语搜索：始终同时搜索中英文） ──────────
        query_rounds: list[str] = [query]
        if self.use_query_optimizer:
            opt_result = self._optimizer.optimize(query)
            optimized = opt_result["optimized"]
            bilingual = opt_result["bilingual_candidates"]

            if opt_result["was_modified"]:
                self.query_opt_modified += 1
                print(f"  [OPTIM] 查询已优化: '{query[:60]}' → '{optimized[:60]}'")
                query = optimized
                query_rounds = [query]

            # 无论查询是中文还是英文，都追加另一种语言的关键词
            # 这样能同时覆盖中英文文献
            # 但需先验证双语候选项与原始查询的语义相关性，防止主题偏离
            if len(bilingual) > 1:
                alt_lang = bilingual[1]  # 优化器自动生成的另一种语言版本
                # 语义验证：确保双语候选与原始查询主题相关
                alt_sim = QueryOptimizer.semantic_similarity(query, alt_lang)
                if alt_sim >= QueryOptimizer.SEMANTIC_FIDELITY_THRESHOLD:
                    if alt_lang != query and alt_lang not in query_rounds:
                        self.query_opt_bilingual += 1
                        query_rounds.append(alt_lang)
                        print(f"  [BI-LING] 追加 {'英文' if opt_result['has_chinese'] else '中文'}搜索: "
                              f"'{alt_lang[:60]}' (相似度={alt_sim:.2f})")
                else:
                    self.query_opt_bilingual += 1
                    print(f"  [BI-LING] ⚠️ 拒绝双语候选: '{alt_lang[:60]}' "
                          f"语义相似度={alt_sim:.2f} < 阈值{QueryOptimizer.SEMANTIC_FIDELITY_THRESHOLD}")

        print(f"\n[SEARCH] ⚡ 并发多源搜索启动")
        print(f"   查询: {query}")
        print(f"   来源: {', '.join(sources)}")
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
                    tool.async_run({"query": q, "limit": tool.max_results}),
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

            tasks = [
                _search_one(s, current_query)
                for s in sources
                if s in ("arxiv", "semantic_scholar", "google_scholar")
            ]
            round_results = await asyncio.gather(*tasks)

            # 汇总本轮结果
            for source, papers, error in round_results:
                all_papers.extend(papers)
                search_results.setdefault(source, 0)
                search_results[source] += len(papers)
                if error:
                    source_errors[source] = error

        # ── 失败来源重试：如果总论文数 < 3 且有来源失败 ────────
        # 单来源重试（用 timeout*1.5 增加生存率）
        total_before_retry = len(all_papers)
        if total_before_retry < 3 and source_errors:
            retry_sources = [s for s in sources if s in source_errors]
            if retry_sources:
                print(f"\n[RETRY] 论文数太少({total_before_retry})，重试失败的来源: {retry_sources}")
                last_query = query_rounds[-1] if query_rounds else query
                retry_tasks = [
                    _search_one(s, last_query)
                    for s in retry_sources
                ]
                # 加长超时
                import copy
                saved = copy.copy(self._SOURCE_TIMEOUTS)
                for s in retry_sources:
                    self._SOURCE_TIMEOUTS[s] = int(self._SOURCE_TIMEOUTS.get(s, 30) * 1.5)
                retry_results = await asyncio.gather(*retry_tasks)
                # 恢复超时
                self._SOURCE_TIMEOUTS = saved
                for source, papers, error in retry_results:
                    if papers:
                        all_papers.extend(papers)
                        search_results[source] = search_results.get(source, 0) + len(papers)
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

        # ── 主题相关性过滤：使用语义相似度过滤掉与原始查询主题偏离的论文 ──
        # 使用标题+摘要的组合文本与原始查询进行语义相似度比较
        if unique_papers and len(unique_papers) > 2:
            original_query_for_filter = tool_input.get("query", query)
            before_topic_filter = len(unique_papers)
            unique_papers = self._filter_by_topic_relevance(
                unique_papers, original_query_for_filter, threshold=0.15
            )
            topic_filtered = before_topic_filter - len(unique_papers)
            if topic_filtered > 0:
                print(f"[TOPIC] 主题过滤: 移除 {topic_filtered} 篇主题偏离论文 "
                      f"({before_topic_filter} -> {len(unique_papers)} 篇)")

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

        final_papers = unique_papers[:limit]

        # ═══════════════════════════════════════════════════════════
        #  阶段1: 并发下载 PDF（4 线程）
        # ═══════════════════════════════════════════════════════════
        downloaded = []
        if tool_input.get("auto_download", self.auto_download):
            print(f"[DOWNLOAD] 并发下载 {len(final_papers)} 篇论文的 PDF（4 线程）...")
            from tools.paper_downloader import PaperDownloaderTool
            dl = PaperDownloaderTool(storage_path="./storage/papers")

            def _download_one(p: dict) -> tuple[str, str | None, dict | None]:
                url = p.get("pdf_url", "") or p.get("url", "")
                title = p.get("title", "")[:60]
                if not url:
                    return title, None, p
                try:
                    r = dl.run({"url": url})
                    dl_result = json.loads(r) if isinstance(r, str) and r.startswith("{") else {"status": "unknown"}
                    if dl_result.get("status") == "success":
                        fname = dl_result.get("filename", "")
                        p["downloaded"] = True
                        p["local_path"] = dl_result.get("path", "")
                        return title, fname, p
                    elif dl_result.get("status") == "already_exists":
                        fname = dl_result.get("filename", "")
                        p["downloaded"] = True
                        return title, fname, p
                    else:
                        return title, None, p
                except Exception as e:
                    return title, None, p

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(_download_one, p): p for p in final_papers}
                for future in as_completed(futures):
                    title, fname, p = future.result()
                    if fname:
                        print(f"    ✅ {title} -> {fname}")
                        downloaded.append(fname)
                    else:
                        print(f"    ❌ {title}")
            print(f"[DOWNLOAD] 完成：成功 {len(downloaded)} / 共 {len(final_papers)} 篇")

        # ═══════════════════════════════════════════════════════════
        #  阶段2: 并发解析 → 分块 → 批量向量入库（4 线程流水线）
        # ═══════════════════════════════════════════════════════════
        indexed_count = 0
        if tool_input.get("auto_index", self.auto_index):
            from tools.pdf_parser import PDFParserTool
            from tools.text_chunker import TextChunkerTool

            parser_tool = PDFParserTool(storage_path="./storage/papers")
            chunker_tool = TextChunkerTool(chunk_size=512, chunk_overlap=128)

            all_chunks: list[dict] = []
            index_lock = threading.Lock()
            downloadable = [p for p in final_papers if p.get("downloaded") and p.get("local_path")]

            if downloadable:
                print(f"[INDEX] 🔍 并发解析+分块 {len(downloadable)} 篇论文（4 线程）...")

                def _index_one(paper: dict) -> None:
                    """单篇论文：解析 → 分块（在线程池中并行执行）。"""
                    title = paper.get("title", "")[:60]
                    local_path = paper.get("local_path", "")
                    if not local_path or not os.path.exists(local_path):
                        return
                    try:
                        # ── 解析 PDF/文本 ──
                        parse_result = parser_tool.run({"pdf_path": local_path})
                        data = json.loads(parse_result) if isinstance(parse_result, str) else parse_result
                        if not isinstance(data, dict) or data.get("status") == "error":
                            print(f"    ⚠ {title}: 解析失败")
                            return

                        # 提取全文（支持两种返回格式）
                        content = data.get("content", "")
                        if isinstance(content, dict):
                            full_text = content.get("full_text", "")
                        elif isinstance(content, str):
                            full_text = content
                        else:
                            full_text = ""
                        if not full_text or not full_text.strip():
                            return

                        # ── 分块 ──
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
                            print(f"    📄 {title}: {len(chunks)} 个块")
                    except Exception as e:
                        print(f"    ❌ {title}: 索引失败 - {e}")

                with ThreadPoolExecutor(max_workers=4) as executor:
                    list(executor.map(_index_one, downloadable))

                # ── 批量写入向量数据库 ──
                if all_chunks:
                    from tools.vector_store import VectorStoreTool
                    store_tool = VectorStoreTool(db_path="./storage/chroma", local_model_cache="./models")
                    # 用查询主题作为集合名（清理后）
                    collection_name = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff_\-]', '_', query)[:50]
                    store_result = store_tool.run({
                        "action": "add",
                        "collection_name": collection_name,
                        "chunks": all_chunks,
                    })
                    store_data = json.loads(store_result) if isinstance(store_result, str) else {}
                    added = store_data.get("chunks_added", 0) if isinstance(store_data, dict) else 0
                    indexed_count = added
                    print(f"[INDEX] ✅ 向量入库完成: {added} 个块 → 集合 '{collection_name}'")
                    # ── 桥梁：更新缓存，标记这些论文已入库 Chroma ──
                    try:
                        indexed_titles = [p.get("title", "") for p in downloaded]
                        indexed_titles = [t for t in indexed_titles if t]
                        if indexed_titles and self.use_cache:
                            for source_name in set(p.get("source", "") for p in downloaded):
                                source_key = {
                                    "arXiv": "arxiv", "Semantic Scholar": "semantic_scholar",
                                    "Google Scholar": "google_scholar",
                                }.get(source_name, source_name)
                                year_from = str(tool_input.get("year_from", "")).strip()
                                for q in query_rounds:
                                    self._get_cache().update_chroma_status(source_key, q, year_from, indexed_titles)
                            print(f"[CACHE] 已标记 {len(indexed_titles)} 篇论文的 Chroma 状态")
                    except Exception:
                        pass
                else:
                    print(f"[INDEX] ⏭ 无有效文本块可入库")
            else:
                print(f"[INDEX] ⏭ 没有已下载的论文可索引")

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
        if source_errors:
            result["source_errors"] = source_errors

        return json.dumps(result, ensure_ascii=False, indent=2)
