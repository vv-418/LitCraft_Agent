# 向量库证据检索。
# 默认：Dense 基线（评测上最优且快）。
# 可选：RETRIEVAL_MODE=hybrid → Dense(+BM25)→RRF→Cross-Encoder 精排。

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tools.base import Tool
from tools.bilingual_query import build_vector_queries, has_chinese
from tools.reranker import Reranker, get_reranker
from tools.vector_store import VectorStoreTool
from llm_client import LitCraftAgentsLLM

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = None  # type: ignore


_RRF_K = 60

_ZH_STOP = {
    "的", "了", "吗", "呢", "啊", "吧", "嘛", "是", "不", "很", "都", "也", "就", "还",
    "在", "有", "和", "与", "或", "及", "等", "被", "把", "让", "给", "对", "从", "向",
    "什么", "怎么", "如何", "是否", "可以", "一个", "这个", "那个", "哪些", "为什么",
    "主要", "想", "知道", "一下", "请问", "谢谢", "帮忙", "感觉", "觉得",
}


def _retrieval_mode() -> str:
    """dense（默认）| hybrid。"""
    mode = (os.getenv("RETRIEVAL_MODE") or "dense").strip().lower()
    return mode if mode in {"dense", "hybrid"} else "dense"


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    text = text.lower()
    tokens = re.findall(r"[a-z0-9_]+", text)
    for span in re.findall(r"[\u4e00-\u9fff]+", text):
        chars = [c for c in span if c not in _ZH_STOP]
        tokens.extend(chars)
        if len(chars) >= 2:
            tokens.extend("".join(chars[i : i + 2]) for i in range(len(chars) - 1))
        if len(chars) >= 3:
            tokens.extend("".join(chars[i : i + 3]) for i in range(len(chars) - 2))
    return tokens


def _rrf_fuse_weighted(
    weighted_lists: List[Tuple[List[str], float]], k: int = _RRF_K
) -> Dict[str, float]:
    scores: Dict[str, float] = defaultdict(float)
    for ranked, weight in weighted_lists:
        w = float(weight)
        if w <= 0 or not ranked:
            continue
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] += w / (k + rank)
    return dict(scores)


def _split_clauses(text: str, limit: int = 2) -> List[str]:
    """长中文问句切 1–2 条子句，作低权重 Dense 补充。"""
    parts = re.split(r"[，,。．？\?！!；;、\n]+", text or "")
    out: List[str] = []
    seen = set()
    for p in parts:
        p = p.strip()
        if len(p) < 10:
            continue
        if p in seen or p == (text or "").strip():
            continue
        seen.add(p)
        out.append(p)
        if len(out) >= limit:
            break
    return out


class AdvancedRetrieval(Tool):
    """默认 Dense 基线；RETRIEVAL_MODE=hybrid 时启用 RRF+重排。"""

    name = "advanced_search"
    description = (
        "Retrieve evidence chunks from the vector store. "
        "Default: dense baseline (fast, best measured recall). "
        "Set RETRIEVAL_MODE=hybrid for dense+BM25 RRF + cross-encoder rerank."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "collection_name": {"type": "string", "description": "Collection name to search in"},
            "query": {"type": "string", "description": "User research topic"},
            "top_k": {"type": "integer", "description": "Final chunks to return (default: 8)"},
            "threshold": {
                "type": "number",
                "description": "Min topic relevance after rerank (default: 0.32; 0 disables)",
            },
            "use_mmr": {
                "type": "boolean",
                "description": "Enable light MMR after rerank (default false)",
            },
        },
        "required": ["collection_name", "query"],
    }

    def __init__(
        self,
        vector_store: Optional[VectorStoreTool] = None,
        llm: Optional[LitCraftAgentsLLM] = None,
        num_hypotheses: int = 3,
        num_query_variants: int = 3,
        topic_threshold: float = 0.32,
        mmr_lambda: float = 0.85,
        use_mmr: bool = False,
        reranker: Optional[Reranker] = None,
        recall_pool: int = 80,
        rerank_top_n: int = 0,
    ):
        self.vector_store = vector_store or VectorStoreTool()
        self.llm = llm
        self.num_hypotheses = num_hypotheses
        self.num_query_variants = num_query_variants
        self.topic_threshold = topic_threshold
        self.mmr_lambda = mmr_lambda
        self.use_mmr = use_mmr
        self.reranker = reranker or get_reranker()
        self.recall_pool = max(20, int(recall_pool))
        env_n = os.getenv("RERANK_TOP_N", "").strip()
        self.rerank_top_n = int(rerank_top_n or env_n or 50)
        self._bm25_cache: Dict[str, Tuple[List[str], Dict[str, Dict[str, Any]], Any]] = {}

    def run(self, tool_input: Dict[str, Any]) -> str:
        collection_name = tool_input.get("collection_name")
        query = str(tool_input.get("query") or "").strip()
        top_k = int(tool_input.get("top_k", 8))
        threshold = float(tool_input.get("threshold", self.topic_threshold))
        use_mmr = bool(tool_input.get("use_mmr", self.use_mmr))
        results = self._retrieve(collection_name, query, top_k, threshold, use_mmr=use_mmr)
        return json.dumps(results, ensure_ascii=False, indent=2)

    # ── BM25 ──────────────────────────────────────────────────

    def _get_bm25_index(self, collection_name: str):
        if collection_name in self._bm25_cache:
            return self._bm25_cache[collection_name]

        if BM25Okapi is None:
            print("[WARN] 未安装 rank_bm25，跳过稀疏检索")
            empty = ([], {}, None)
            self._bm25_cache[collection_name] = empty
            return empty

        docs = self.vector_store.get_collection_documents(collection_name)
        if not docs:
            empty = ([], {}, None)
            self._bm25_cache[collection_name] = empty
            return empty

        doc_ids: List[str] = []
        id_to_doc: Dict[str, Dict[str, Any]] = {}
        corpus: List[List[str]] = []
        for d in docs:
            meta = d.get("metadata") or {}
            cid = str(meta.get("chunk_id") or d.get("id") or "")
            if not cid:
                continue
            doc_ids.append(cid)
            id_to_doc[cid] = {"content": d.get("content") or "", "metadata": meta}
            corpus.append(_tokenize(d.get("content") or ""))

        bm25 = BM25Okapi(corpus) if corpus else None
        packed = (doc_ids, id_to_doc, bm25)
        self._bm25_cache[collection_name] = packed
        print(f"[INFO] BM25 就绪 | docs={len(doc_ids)}")
        return packed

    def _bm25_rank(self, collection_name: str, query: str, top_k: int) -> List[str]:
        doc_ids, _, bm25 = self._get_bm25_index(collection_name)
        if bm25 is None or not doc_ids:
            return []
        scores = bm25.get_scores(_tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [doc_ids[i] for i, s in ranked if s > 0]

    # ── Dense ─────────────────────────────────────────────────

    def _dense_rank(
        self, collection_name: str, query: str, top_k: int
    ) -> Tuple[List[str], Dict[str, Dict[str, Any]], Dict[str, float]]:
        id_to_doc: Dict[str, Dict[str, Any]] = {}
        sims: Dict[str, float] = {}
        ranked: List[str] = []
        sr = json.loads(self.vector_store._search(collection_name, query, top_k=top_k, threshold=0))
        if sr.get("status") != "success":
            return [], {}, {}
        for chunk in sr.get("chunks", []):
            meta = chunk.get("metadata") or {}
            cid = str(meta.get("chunk_id") or "")
            if not cid:
                continue
            sim = float(chunk.get("similarity") or 0.0)
            ranked.append(cid)
            sims[cid] = max(sims.get(cid, 0.0), sim)
            id_to_doc[cid] = {"content": chunk.get("content"), "metadata": meta}
        return ranked, id_to_doc, sims

    # ── 主题门槛 + MMR ────────────────────────────────────────

    def _topic_scores(self, topic: str, id_to_doc: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        model = self.vector_store.get_model_for_text(topic)
        ids = list(id_to_doc.keys())
        if not ids:
            return {}
        topic_emb = model.encode([topic], normalize_embeddings=True)[0]
        previews = [(id_to_doc[i].get("content") or "")[:500] for i in ids]
        chunk_embs = model.encode(previews, normalize_embeddings=True)
        return {cid: float(np.dot(topic_emb, chunk_embs[j])) for j, cid in enumerate(ids)}

    def _mmr_select(
        self,
        candidate_ids: List[str],
        id_to_doc: Dict[str, Dict[str, Any]],
        relevance: Dict[str, float],
        top_k: int,
        query: str = "",
    ) -> List[str]:
        if not candidate_ids:
            return []
        model = self.vector_store.get_model_for_text(query or "")
        texts = [(id_to_doc[i].get("content") or "")[:500] for i in candidate_ids]
        embs = model.encode(texts, normalize_embeddings=True)
        emb_map = {cid: embs[j] for j, cid in enumerate(candidate_ids)}

        selected: List[str] = []
        remaining = list(candidate_ids)
        while remaining and len(selected) < top_k:
            best_id = None
            best_score = -1e9
            for cid in remaining:
                rel = relevance.get(cid, 0.0)
                if not selected:
                    score = rel
                else:
                    max_sim = max(float(np.dot(emb_map[cid], emb_map[s])) for s in selected)
                    score = self.mmr_lambda * rel - (1 - self.mmr_lambda) * max_sim
                if score > best_score:
                    best_score = score
                    best_id = cid
            if best_id is None:
                break
            selected.append(best_id)
            remaining.remove(best_id)
        return selected

    # ── 主检索 ────────────────────────────────────────────────

    def _retrieve(
        self,
        collection_name: str,
        query: str,
        top_k: int,
        topic_threshold: float,
        use_mmr: bool = False,
    ) -> Dict[str, Any]:
        mode = _retrieval_mode()
        if mode == "dense":
            return self._retrieve_dense(collection_name, query, top_k, topic_threshold)

        return self._retrieve_hybrid(
            collection_name, query, top_k, topic_threshold, use_mmr=use_mmr
        )

    def _image_rank(
        self,
        collection_name: str,
        query: str,
        top_m: int,
    ) -> Tuple[List[str], Dict[str, Dict[str, Any]], Dict[str, float]]:
        """CLIP 文本 query → `*_mm` 插图召回。不可用时返回空。"""
        ranked: List[str] = []
        id_to_doc: Dict[str, Dict[str, Any]] = {}
        sims: Dict[str, float] = {}
        if top_m <= 0:
            return ranked, id_to_doc, sims
        try:
            raw = json.loads(
                self.vector_store.search_images(
                    collection_name, query, top_k=top_m, threshold=0.0
                )
            )
        except Exception as e:
            print(f"   [WARN] 多模态检索跳过: {e}")
            return ranked, id_to_doc, sims

        status = raw.get("status")
        if status == "skipped":
            print(f"   [MM] multimodal skipped: {raw.get('reason', '')}")
            return ranked, id_to_doc, sims
        if status not in ("success", None):
            return ranked, id_to_doc, sims

        for c in raw.get("chunks") or []:
            meta = dict(c.get("metadata") or {})
            cid = str(meta.get("chunk_id") or "")
            if not cid:
                continue
            path = meta.get("path", "")
            page = meta.get("page", "?")
            sim = float(c.get("similarity") or 0.0)
            meta["modality"] = "image"
            content = (
                f"[FIGURE p.{page}] {path}\n"
                f"(multimodal image hit; cite as figure if relevant; sim={sim:.4f})"
            )
            ranked.append(cid)
            id_to_doc[cid] = {"content": content, "metadata": meta}
            sims[cid] = sim
        if ranked:
            print(f"   [MM] 图像召回 {len(ranked)} 张 (top_m={top_m})")
        return ranked, id_to_doc, sims

    def _retrieve_dense(
        self,
        collection_name: str,
        query: str,
        top_k: int,
        topic_threshold: float,
    ) -> Dict[str, Any]:
        """评测最优路径：文本 Dense + 可选 CLIP 图像加权 RRF。"""
        print("\n[SEARCH] Dense 基线检索（RETRIEVAL_MODE=dense）")
        print(f"   主题: {query[:80]}")
        ranked, id_to_doc, sims = self._dense_rank(collection_name, query, max(top_k, top_k))

        # 辅路：图像 Top-M（M≈max(2, top_k//3)），权重低于文本，避免冲掉正文
        img_m = max(2, top_k // 3)
        mm_ranked, mm_docs, mm_sims = self._image_rank(collection_name, query, img_m)
        for cid, doc in mm_docs.items():
            id_to_doc[cid] = doc
        for cid, s in mm_sims.items():
            sims[cid] = max(sims.get(cid, 0.0), s)

        text_w = 1.0
        image_w = 0.35
        fuse_lists: List[Tuple[List[str], float]] = [(ranked, text_w)]
        if mm_ranked:
            fuse_lists.append((mm_ranked, image_w))
            print(f"   [MM] 加权 RRF 融合 text_w={text_w} image_w={image_w}")
        rrf = _rrf_fuse_weighted(fuse_lists)
        fused = sorted(rrf.keys(), key=lambda c: rrf[c], reverse=True)
        # 保证文本 Top 仍占主导席位，再补插图
        merged: List[str] = []
        seen = set()
        for cid in ranked[:top_k] + fused:
            if cid in id_to_doc and cid not in seen:
                seen.add(cid)
                merged.append(cid)

        if topic_threshold > 0 and merged:
            # 图像命中不做主题门槛过滤（CLIP 分与文本分不可比）
            topic_scores = self._topic_scores(
                query,
                {c: id_to_doc[c] for c in merged if (id_to_doc[c].get("metadata") or {}).get("modality") != "image"},
            )
            kept = []
            for c in merged:
                meta = id_to_doc[c].get("metadata") or {}
                if meta.get("modality") == "image":
                    kept.append(c)
                elif topic_scores.get(c, 0.0) >= topic_threshold:
                    kept.append(c)
            if not kept:
                kept = merged
            for c in merged:
                if c not in topic_scores:
                    topic_scores[c] = sims.get(c, 0.0)
        else:
            topic_scores = {c: sims.get(c, 0.0) for c in merged}
            kept = merged

        chunks = []
        for cid in kept[:top_k]:
            meta = id_to_doc[cid].get("metadata") or {}
            chunks.append({
                "chunk_id": cid,
                "topic_relevance": round(topic_scores.get(cid, 0.0), 4),
                "similarity": round(sims.get(cid, 0.0), 4),
                "rrf_score": round(rrf.get(cid, 0.0), 6),
                "content": id_to_doc[cid].get("content"),
                "metadata": meta,
                "modality": meta.get("modality", "text"),
            })
        return {
            "status": "success",
            "strategy": "dense_baseline",
            "query": query,
            "queries": [query],
            "results_count": len(chunks),
            "chunks": chunks,
        }

    def _retrieve_hybrid(
        self,
        collection_name: str,
        query: str,
        top_k: int,
        topic_threshold: float,
        use_mmr: bool = False,
    ) -> Dict[str, Any]:
        print("\n[SEARCH] Hybrid 召回 + Cross-Encoder 精排（RETRIEVAL_MODE=hybrid）")
        print(f"   主题: {query[:80]}")

        queries = build_vector_queries(query, max_queries=2)
        original = queries[0] if queries else query
        zh = has_chinese(original)
        pool = max(self.recall_pool, top_k * 5)
        rerank_n = max(self.rerank_top_n, top_k)

        weighted_lists: List[Tuple[List[str], float]] = []
        id_to_doc: Dict[str, Dict[str, Any]] = {}
        dense_sim: Dict[str, float] = defaultdict(float)

        def _absorb(ranked, docs, sims, weight: float):
            if ranked:
                weighted_lists.append((ranked, weight))
            for cid, doc in docs.items():
                id_to_doc[cid] = doc
            for cid, s in sims.items():
                dense_sim[cid] = max(dense_sim[cid], s)

        # Phase1: Dense（中文略高权重）
        dense_w = 1.5 if zh else 1.0
        print(f"[PHASE1] Dense 召回 pool={pool} weight={dense_w}")
        ranked, docs, sims = self._dense_rank(collection_name, original, pool)
        _absorb(ranked, docs, sims, dense_w)
        primary_dense = list(ranked)

        # 长中文：1–2 条子句低权重补召回（扩大候选，供重排挑选）
        if zh and len(original) >= 40:
            for i, clause in enumerate(_split_clauses(original, limit=2), 1):
                print(f"   clause Dense {i}: {clause[:60]}")
                r, d, s = self._dense_rank(collection_name, clause, pool)
                _absorb(r, d, s, 0.6)

        # BM25：英文打开；中文强 Dense 场景关闭稀疏通道（避免噪声进重排池）
        bm25_docs = {}
        if not zh:
            bm25_w = 1.0
            print(f"[PHASE1b] BM25 召回 weight={bm25_w}")
            bm25_ranked = self._bm25_rank(collection_name, original, pool)
            if bm25_ranked:
                weighted_lists.append((bm25_ranked, bm25_w))
            _, bm25_docs, _ = self._get_bm25_index(collection_name)
            for cid in bm25_ranked:
                if cid not in id_to_doc and cid in bm25_docs:
                    id_to_doc[cid] = bm25_docs[cid]

            for q in queries[1:]:
                r, d, s = self._dense_rank(collection_name, q, pool)
                _absorb(r, d, s, 0.7)
                br = self._bm25_rank(collection_name, q, pool)
                if br:
                    weighted_lists.append((br, 0.5))
                for cid in br:
                    if cid not in id_to_doc and cid in bm25_docs:
                        id_to_doc[cid] = bm25_docs[cid]
        else:
            print("[PHASE1b] 跳过 BM25（中文：Dense 宽召回 + 重排）")

        # 真·多模态辅路：CLIP 文本→图像，低权重并入 RRF（不替换文本基线）
        img_m = max(2, top_k // 3)
        mm_ranked, mm_docs, mm_sims = self._image_rank(collection_name, original, img_m)
        if mm_ranked:
            _absorb(mm_ranked, mm_docs, mm_sims, 0.35)
            print(f"[PHASE1c] 多模态图像召回 {len(mm_ranked)} weight=0.35")

        if not weighted_lists or not id_to_doc:
            return {
                "status": "success",
                "strategy": "hybrid_rrf_rerank",
                "query": query,
                "queries": queries,
                "results_count": 0,
                "chunks": [],
            }

        print("[PHASE2] 加权 RRF 融合")
        rrf = _rrf_fuse_weighted(weighted_lists)
        fused_ids = sorted(
            (cid for cid in rrf if cid in id_to_doc),
            key=lambda c: rrf[c],
            reverse=True,
        )
        # 保证原查询 Dense Top 进入重排池（避免稀疏通道挤掉高相关）
        cand_set = []
        seen = set()
        for cid in primary_dense[:rerank_n] + fused_ids:
            if cid in id_to_doc and cid not in seen:
                seen.add(cid)
                cand_set.append(cid)
            if len(cand_set) >= rerank_n:
                break
        candidates = cand_set
        print(f"   候选进入重排: {len(candidates)}")

        # Phase3: Cross-Encoder 精排（用原始主题，截断过长 query）
        print(f"[PHASE3] Rerank top_n={len(candidates)} → keep {top_k}")
        rerank_query = original if len(original) <= 256 else original[:256]
        passages = [(id_to_doc[c].get("content") or "") for c in candidates]
        reranked = self.reranker.rerank(rerank_query, passages, top_k=len(candidates))
        rerank_scores = {candidates[i]: score for i, score in reranked}
        ordered = [candidates[i] for i, _ in reranked]

        # Dense 序 ∩ 重排序再 RRF：精排提升精度，同时不轻易丢掉 Dense 已召回的高相关
        print("[PHASE3b] Dense序 + 重排序 二次 RRF")
        dense_for_fuse = [c for c in primary_dense if c in set(candidates)]
        fuse_w_dense = 1.0
        fuse_w_rerank = 1.2
        final_rrf = _rrf_fuse_weighted([
            (dense_for_fuse, fuse_w_dense),
            (ordered, fuse_w_rerank),
        ])
        ordered = sorted(final_rrf.keys(), key=lambda c: final_rrf[c], reverse=True)

        # Phase4: 可选主题门槛（评测 threshold=0 时跳过）
        if topic_threshold > 0:
            print(f"[PHASE4] 主题门槛 threshold={topic_threshold}")
            subset = {cid: id_to_doc[cid] for cid in ordered[: max(top_k * 3, top_k)]}
            topic_scores = self._topic_scores(query, subset)
            kept = [cid for cid in ordered if topic_scores.get(cid, 0.0) >= topic_threshold]
            if not kept:
                kept = ordered[: max(top_k * 2, top_k)]
        else:
            print("[PHASE4] 跳过主题门槛")
            topic_scores = {cid: dense_sim.get(cid, 0.0) for cid in ordered}
            kept = ordered

        # 相关性：二次 RRF 分 + 重排分
        max_rs = max(rerank_scores.values()) if rerank_scores else 1.0
        min_rs = min(rerank_scores.values()) if rerank_scores else 0.0
        span = (max_rs - min_rs) or 1.0
        max_fr = max(final_rrf.values()) if final_rrf else 1.0
        relevance = {}
        for cid in kept:
            rn = (rerank_scores.get(cid, min_rs) - min_rs) / span
            fr = final_rrf.get(cid, 0.0) / max_fr
            relevance[cid] = 0.6 * fr + 0.4 * rn
        kept_sorted = sorted(kept, key=lambda c: relevance.get(c, 0.0), reverse=True)

        if use_mmr:
            print("[PHASE5] 轻度 MMR")
            selected = self._mmr_select(
                kept_sorted[: max(top_k * 4, top_k)],
                id_to_doc,
                relevance,
                top_k,
                query=query,
            )
        else:
            print("[PHASE5] 按重排分截断")
            selected = kept_sorted[:top_k]

        # 精排可能挤掉插图：保留少量 CLIP 命中（文本仍占多数席位）
        if mm_ranked:
            room = max(1, min(2, top_k // 3))
            extras = [c for c in mm_ranked if c in id_to_doc and c not in selected][:room]
            if extras:
                selected = selected[: max(0, top_k - len(extras))] + extras
                print(f"   [MM] 保留 {len(extras)} 张图像命中进入最终结果")

        chunks = []
        for cid in selected:
            meta = id_to_doc[cid].get("metadata") or {}
            chunks.append({
                "chunk_id": cid,
                "rerank_score": round(rerank_scores.get(cid, 0.0), 6),
                "topic_relevance": round(topic_scores.get(cid, 0.0), 4),
                "rrf_score": round(rrf.get(cid, 0.0), 6),
                "similarity": round(dense_sim.get(cid, 0.0), 4),
                "content": id_to_doc[cid].get("content"),
                "metadata": meta,
                "modality": meta.get("modality", "text"),
            })

        return {
            "status": "success",
            "strategy": "hybrid_rrf_rerank",
            "query": query,
            "queries": queries,
            "reranker": self.reranker.model_path if self.reranker.available else None,
            "use_mmr": use_mmr,
            "results_count": len(chunks),
            "chunks": chunks,
        }
