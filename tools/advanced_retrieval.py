# 作用：实现高级检索策略（HyDE + MQE），提升向量检索的准确率和召回率。
# LLM 生成内容始终同时覆盖中英文，确保双语文献都能被检索到。

import json
import re
from typing import Any, List, Dict, Optional
from collections import defaultdict

from tools.base import Tool
from tools.vector_store import VectorStoreTool
from llm_client import LitCraftAgentsLLM


class AdvancedRetrieval(Tool):
    """高级检索工具：支持 HyDE + MQE 混合策略。"""

    name = "advanced_search"
    description = "Advanced retrieval using HyDE + MQE strategies to find relevant documents"
    input_schema = {
        "type": "object",
        "properties": {
            "collection_name": {"type": "string", "description": "Collection name to search in"},
            "query": {"type": "string", "description": "User query/research topic"},
            "top_k": {"type": "integer", "description": "Number of final results to return (default: 5)"},
            "strategy": {
                "type": "string",
                "enum": ["hyde", "mque", "hybrid"],
                "description": "Retrieval strategy: hyde, mque, or hybrid (default: hybrid)"
            },
            "threshold": {"type": "number", "description": "Similarity threshold (0-1, default: 0.3)"},
        },
        "required": ["collection_name", "query"],
    }

    def __init__(
        self,
        vector_store: Optional[VectorStoreTool] = None,
        llm: Optional[LitCraftAgentsLLM] = None,
        num_hypotheses: int = 3,
        num_query_variants: int = 3,
    ):
        self.vector_store = vector_store or VectorStoreTool()
        self.llm = llm or LitCraftAgentsLLM()
        self.num_hypotheses = num_hypotheses
        self.num_query_variants = num_query_variants

    def run(self, tool_input: Dict[str, Any]) -> str:
        collection_name = tool_input.get("collection_name")
        query = tool_input.get("query")
        top_k = tool_input.get("top_k", 5)
        strategy = tool_input.get("strategy", "hybrid")
        threshold = tool_input.get("threshold", 0.3)
        if strategy == "hyde":
            results = self._hyde_search(collection_name, query, top_k, threshold)
        elif strategy == "mque":
            results = self._mque_search(collection_name, query, top_k, threshold)
        else:
            results = self._hybrid_search(collection_name, query, top_k, threshold)
        return json.dumps(results, ensure_ascii=False, indent=2)

    # ── HyDE：假设文档生成（始终中英文混合）────────────────────

    def _filter_by_topic_relevance(self, query: str, chunks: list[dict], min_relevance: float = 0.35) -> list[dict]:
        """用原始 query 的 embedding 对检索结果做二次主题相关性验证。

        即使 HyDE/MQE 检索出的 chunk 与假设文档相似度高，
        也要验证它们是否与用户原始主题真正相关。
        """
        if not chunks:
            return chunks
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            model = SentenceTransformer("all-MiniLM-L6-v2")
            query_emb = model.encode([query], normalize_embeddings=True)
            verified = []
            dropped = 0
            for c in chunks:
                content_preview = c["content"][:200]
                chunk_emb = model.encode([content_preview], normalize_embeddings=True)
                relevance = float(np.dot(query_emb[0], chunk_emb[0]))
                if relevance >= min_relevance:
                    c["topic_relevance"] = round(relevance, 4)
                    verified.append(c)
                else:
                    dropped += 1
            if dropped > 0:
                print(f"[TOPIC_FILTER] 过滤掉 {dropped} 个主题相关性过低的 chunk (阈值={min_relevance})")
            return verified
        except Exception as e:
            print(f"[TOPIC_FILTER] 主题过滤异常，保留全部结果: {e}")
            return chunks

    def _generate_hypotheses(self, query: str) -> List[str]:
        """用 LLM 生成假设文档摘要，严格围绕研究主题。"""
        prompt = (
            f"Please generate {self.num_hypotheses} hypothetical paper abstracts that would be directly relevant to this topic.\n"
            "The abstracts must be STRICTLY about the given topic — do NOT drift to adjacent or related fields.\n"
            "摘要必须**严格围绕**给定的研究主题，不得偏离到相关但不同的领域。\n\n"
            f"Research topic / 研究主题：{query}\n\n"
            "Requirements / 要求：\n"
            "1. Cover BOTH Chinese and English academic perspectives / 必须同时包含中英文\n"
            "2. Each abstract MUST include the core keywords of the topic / 每条摘要必须包含主题的核心关键词\n"
            "3. 50-100 words each / 每条 50-100 字\n"
            "4. Format: one per line starting with \"- \" / 每行一个，用 \"- \" 开头\n\n"
            "Return only the abstract list. / 只返回摘要列表。"
        )
        response = self.llm.think(messages=[{"role": "user", "content": prompt}], temperature=0.7)
        hypotheses = [
            line[2:].strip() for line in response.split("\n")
            if line.strip().startswith("- ") and line[2:].strip()
        ]
        print(f"[INFO] HyDE 生成了 {len(hypotheses)} 个假设文档")
        return hypotheses[:self.num_hypotheses]

    # ── MQE：查询变体生成（始终中英文混合）─────────────────────

    def _generate_query_variants(self, query: str) -> List[str]:
        """用 LLM 生成查询变体（中英文混合），严格围绕同一研究主题。"""
        prompt = (
            f"Generate {self.num_query_variants} search query variants that capture the same research intent.\n"
            "请同时生成中文和英文的查询变体，确保中英文文献都能被覆盖。\n"
            "⚠️ 所有变体必须严格围绕同一研究主题，不得偏离到其他领域。\n\n"
            f"Original query / 原始查询：{query}\n\n"
            "Requirements / 要求：\n"
            "1. Cover BOTH Chinese and English perspectives / 必须同时包含中英文\n"
            "2. Include: synonymous rewrites, concept expansion, related angles\n"
            "3. Each variant MUST contain the core concept of the original query / 每个变体必须包含原始查询的核心概念\n"
            "4. Format: one per line starting with \"- \" / 每行一个，用 \"- \" 开头\n\n"
            "Return only the query list. / 只返回查询列表。"
        )
        response = self.llm.think(messages=[{"role": "user", "content": prompt}], temperature=0.7)
        variants = [
            line[2:].strip() for line in response.split("\n")
            if line.strip().startswith("- ") and line[2:].strip()
        ]
        print(f"[INFO] MQE 生成了 {len(variants)} 个查询变体")
        return variants[:self.num_query_variants]

    # ── 检索核心 ──────────────────────────────────────────────

    def _search_and_collect(self, collection_name, query_texts: list[str], top_k: int, threshold: float):
        all_results = defaultdict(lambda: {"similarity_scores": [], "metadata": None, "content": None})
        for i, qt in enumerate(query_texts, 1):
            print(f"   搜索 {i}/{len(query_texts)}: {qt[:50]}...")
            sr = json.loads(self.vector_store._search(collection_name, qt, top_k=top_k, threshold=0))
            if sr.get("status") == "success":
                for chunk in sr.get("chunks", []):
                    cid = chunk["metadata"]["chunk_id"]
                    all_results[cid]["similarity_scores"].append(chunk["similarity"])
                    all_results[cid]["metadata"] = chunk["metadata"]
                    all_results[cid]["content"] = chunk["content"]
        return all_results

    def _fuse_results(self, all_results, threshold: float, top_k: int) -> list[dict]:
        fused = []
        for cid, data in all_results.items():
            if data["similarity_scores"]:
                sim = max(data["similarity_scores"])
                if sim >= threshold:
                    fused.append({
                        "chunk_id": cid,
                        "similarity": round(sim, 4),
                        "content": data["content"],
                        "metadata": data["metadata"],
                    })
        fused.sort(key=lambda x: x["similarity"], reverse=True)
        return fused[:top_k]

    def _hyde_search(self, collection_name, query, top_k=5, threshold=0.3) -> Dict[str, Any]:
        print(f"\n[SEARCH] 使用 HyDE 策略搜索...")
        hyps = self._generate_hypotheses(query)
        results = self._search_and_collect(collection_name, hyps, top_k * 2, threshold)
        fused = self._fuse_results(results, threshold, top_k)
        fused = self._filter_by_topic_relevance(query, fused, min_relevance=0.35)
        return {"status": "success", "strategy": "hyde", "query": query,
                "hypotheses_count": len(hyps), "results_count": len(fused), "chunks": fused}

    def _mque_search(self, collection_name, query, top_k=5, threshold=0.3) -> Dict[str, Any]:
        print(f"\n[SEARCH] 使用 MQE 策略搜索...")
        variants = self._generate_query_variants(query)
        results = self._search_and_collect(collection_name, variants, top_k * 2, threshold)
        fused = self._fuse_results(results, threshold, top_k)
        fused = self._filter_by_topic_relevance(query, fused, min_relevance=0.35)
        return {"status": "success", "strategy": "mque", "query": query,
                "variants_count": len(variants), "results_count": len(fused), "chunks": fused}

    def _hybrid_search(self, collection_name, query, top_k=5, threshold=0.3) -> Dict[str, Any]:
        print(f"\n[SEARCH] 使用混合策略 (HyDE + MQE) 搜索...")
        print(f"\n[PHASE1] HyDE 假设文档检索")
        hyps = self._generate_hypotheses(query)
        h_results = self._search_and_collect(collection_name, hyps, top_k * 3, threshold)
        print(f"\n[PHASE2] MQE 查询变体检索")
        variants = self._generate_query_variants(query)
        m_results = self._search_and_collect(collection_name, variants, top_k * 3, threshold)
        print(f"\n[PHASE3] 融合 HyDE + MQE（MAX 池化）")
        all_scores = {}
        for cid in set(list(h_results.keys()) + list(m_results.keys())):
            h_best = max(h_results[cid]["similarity_scores"]) if h_results[cid]["similarity_scores"] else 0.0
            m_best = max(m_results[cid]["similarity_scores"]) if m_results[cid]["similarity_scores"] else 0.0
            if max(h_best, m_best) >= threshold:
                meta = h_results.get(cid) or m_results.get(cid)
                all_scores[cid] = {"similarity": round(max(h_best, m_best), 4), "content": meta["content"], "metadata": meta["metadata"]}
        fused = sorted(all_scores.values(), key=lambda x: x["similarity"], reverse=True)[:top_k]
        # 主题相关性后验过滤
        fused = self._filter_by_topic_relevance(query, fused, min_relevance=0.35)
        return {"status": "success", "strategy": "hybrid", "query": query,
                "hypotheses_count": len(hyps), "variants_count": len(variants),
                "results_count": len(fused), "chunks": fused}
