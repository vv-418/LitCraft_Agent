# 作用：实现高级检索策略（HyDE + MQE），提升向量检索的准确率和召回率
import json
import re
from typing import Any, List, Dict, Optional
from collections import defaultdict

from tools.base import Tool
from tools.vector_store import VectorStoreTool
from llm_client import LitCraftAgentsLLM


class AdvancedRetrieval(Tool):
    """
    高级检索工具：支持 HyDE（假设文档嵌入）+ MQE（多查询扩展）混合策略。
    
    HyDE（Hypothetical Document Embeddings）：
    - 使用 LLM 根据查询生成假设相关文档
    - 对假设文档进行 embedding 和检索
    - 通常能获得更好的语义匹配效果
    
    MQE（Multi-Query Expansion）：
    - 使用 LLM 根据查询生成多个变体查询
    - 对每个变体查询进行检索
    - 通过融合结果提高召回率
    
    混合策略：
    - 同时应用 HyDE 和 MQE
    - 对所有结果进行重排和融合
    - 返回高质量的综合结果
    """

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
        """
        初始化高级检索工具。

        Args:
            vector_store: VectorStoreTool 实例
            llm: LitCraftAgentsLLM 实例
            num_hypotheses: HyDE 生成的假设文档数量
            num_query_variants: MQE 生成的查询变体数量
        """
        self.vector_store = vector_store or VectorStoreTool()
        self.llm = llm or LitCraftAgentsLLM()
        self.num_hypotheses = num_hypotheses
        self.num_query_variants = num_query_variants

    def run(self, tool_input: Dict[str, Any]) -> str:
        """执行高级检索。"""
        collection_name = tool_input.get("collection_name")
        query = tool_input.get("query")
        top_k = tool_input.get("top_k", 5)
        strategy = tool_input.get("strategy", "hybrid")
        threshold = tool_input.get("threshold", 0.3)

        if strategy == "hyde":
            results = self._hyde_search(collection_name, query, top_k, threshold)
        elif strategy == "mque":
            results = self._mque_search(collection_name, query, top_k, threshold)
        else:  # hybrid
            results = self._hybrid_search(collection_name, query, top_k, threshold)

        return json.dumps(results, ensure_ascii=False, indent=2)

    def _generate_hypotheses(self, query: str) -> List[str]:
        """
        使用 LLM 生成与查询相关的假设文档。

        HyDE 的核心思想：
        - 假设这些假设文档是与查询相关的真实文档
        - 对这些假设文档进行 embedding
        - 用这些 embeddings 去搜索真实文档
        - 效果通常比直接 embedding 查询更好
        """
        prompt = f"""请根据以下研究问题，生成 {self.num_hypotheses} 个假设的相关文档摘要。

研究问题：{query}

要求：
1. 每个假设文档应该是与研究问题高度相关的真实学术内容
2. 文档应该包含关键概念、方法、结果等具体信息
3. 每个假设文档长度 50-100 字左右
4. 返回格式：每行一个假设文档，用 "- " 开头

只返回假设文档列表，不要其他说明。"""

        response = self.llm.think(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7  # 提高创意性
        )

        # 解析响应
        hypotheses = []
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                hypothesis = line[2:].strip()
                if hypothesis:
                    hypotheses.append(hypothesis)

        print(f"[INFO] HyDE 生成了 {len(hypotheses)} 个假设文档")
        return hypotheses[:self.num_hypotheses]

    def _generate_query_variants(self, query: str) -> List[str]:
        """
        使用 LLM 生成查询的多个变体。

        MQE 的核心思想：
        - 一个查询可能有多种表达方式
        - 生成同义查询、扩展查询、聚焦查询等
        - 对每个变体进行搜索，然后融合结果
        - 提高召回率和鲁棒性
        """
        prompt = f"""请根据以下研究问题，生成 {self.num_query_variants} 个表达不同但含义相关的查询变体。

原始问题：{query}

要求：
1. 变体应该覆盖不同的角度和表达方式
2. 包括：同义改写、概念扩展、相关问题等
3. 每个变体应该是一个完整的查询
4. 返回格式：每行一个变体，用 "- " 开头

只返回查询列表，不要其他说明。"""

        response = self.llm.think(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7
        )

        # 解析响应
        variants = []
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                variant = line[2:].strip()
                if variant:
                    variants.append(variant)

        print(f"[INFO] MQE 生成了 {len(variants)} 个查询变体")
        return variants[:self.num_query_variants]

    def _hyde_search(
        self,
        collection_name: str,
        query: str,
        top_k: int = 5,
        threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        HyDE 检索策略：生成假设文档并检索。
        """
        print(f"\n[SEARCH] 使用 HyDE 策略搜索...")

        # 生成假设文档
        hypotheses = self._generate_hypotheses(query)

        # 对每个假设文档进行搜索
        all_results = defaultdict(lambda: {"similarity_scores": [], "metadata": None, "content": None})

        for i, hypothesis in enumerate(hypotheses, 1):
            print(f"   检索假设文档 {i}/{len(hypotheses)}: {hypothesis[:50]}...")

            # 使用向量检索搜索
            search_result = json.loads(
                self.vector_store._search(collection_name, hypothesis, top_k=top_k * 2, threshold=0)
            )

            if search_result.get("status") == "success":
                for chunk in search_result.get("chunks", []):
                    chunk_id = chunk["metadata"]["chunk_id"]
                    all_results[chunk_id]["similarity_scores"].append(chunk["similarity"])
                    all_results[chunk_id]["metadata"] = chunk["metadata"]
                    all_results[chunk_id]["content"] = chunk["content"]

        # 融合结果：计算平均相似度并重排
        fused_results = []
        for chunk_id, data in all_results.items():
            avg_similarity = sum(data["similarity_scores"]) / len(data["similarity_scores"])
            if avg_similarity >= threshold:
                fused_results.append({
                    "chunk_id": chunk_id,
                    "similarity": round(avg_similarity, 4),
                    "hypothesis_count": len(data["similarity_scores"]),
                    "content": data["content"],
                    "metadata": data["metadata"]
                })

        # 按相似度排序
        fused_results.sort(key=lambda x: x["similarity"], reverse=True)

        return {
            "status": "success",
            "strategy": "hyde",
            "query": query,
            "hypotheses_count": len(hypotheses),
            "results_count": len(fused_results[:top_k]),
            "chunks": fused_results[:top_k]
        }

    def _mque_search(
        self,
        collection_name: str,
        query: str,
        top_k: int = 5,
        threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        MQE 检索策略：生成查询变体并融合结果。
        """
        print(f"\n[SEARCH] 使用 MQE 策略搜索...")

        # 生成查询变体
        variants = self._generate_query_variants(query)

        # 对每个变体进行搜索
        all_results = defaultdict(lambda: {"similarity_scores": [], "metadata": None, "content": None})

        for i, variant in enumerate(variants, 1):
            print(f"   检索变体 {i}/{len(variants)}: {variant[:50]}...")

            search_result = json.loads(
                self.vector_store._search(collection_name, variant, top_k=top_k * 2, threshold=0)
            )

            if search_result.get("status") == "success":
                for chunk in search_result.get("chunks", []):
                    chunk_id = chunk["metadata"]["chunk_id"]
                    all_results[chunk_id]["similarity_scores"].append(chunk["similarity"])
                    all_results[chunk_id]["metadata"] = chunk["metadata"]
                    all_results[chunk_id]["content"] = chunk["content"]

        # 融合结果：计算平均相似度并重排
        fused_results = []
        for chunk_id, data in all_results.items():
            avg_similarity = sum(data["similarity_scores"]) / len(data["similarity_scores"])
            if avg_similarity >= threshold:
                fused_results.append({
                    "chunk_id": chunk_id,
                    "similarity": round(avg_similarity, 4),
                    "variant_count": len(data["similarity_scores"]),
                    "content": data["content"],
                    "metadata": data["metadata"]
                })

        # 按相似度排序
        fused_results.sort(key=lambda x: x["similarity"], reverse=True)

        return {
            "status": "success",
            "strategy": "mque",
            "query": query,
            "variants_count": len(variants),
            "results_count": len(fused_results[:top_k]),
            "chunks": fused_results[:top_k]
        }

    def _hybrid_search(
        self,
        collection_name: str,
        query: str,
        top_k: int = 5,
        threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        混合策略：同时使用 HyDE 和 MQE。

        流程：
        1. HyDE 生成假设文档并检索
        2. MQE 生成查询变体并检索
        3. 融合两种策略的结果
        4. 最终返回综合排名结果
        """
        print(f"\n[SEARCH] 使用混合策略 (HyDE + MQE) 搜索...")

        # 第一阶段：HyDE 检索
        print(f"\n[PHASE1] 第一阶段：HyDE 假设文档检索")
        hypotheses = self._generate_hypotheses(query)
        hyde_results = defaultdict(lambda: {"similarity_scores": [], "metadata": None, "content": None})

        for hypothesis in hypotheses:
            search_result = json.loads(
                self.vector_store._search(collection_name, hypothesis, top_k=top_k * 3, threshold=0)
            )

            if search_result.get("status") == "success":
                for chunk in search_result.get("chunks", []):
                    chunk_id = chunk["metadata"]["chunk_id"]
                    hyde_results[chunk_id]["similarity_scores"].append(chunk["similarity"] * 0.5)  # 权重 0.5
                    hyde_results[chunk_id]["metadata"] = chunk["metadata"]
                    hyde_results[chunk_id]["content"] = chunk["content"]

        # 第二阶段：MQE 检索
        print(f"\n[PHASE2] 第二阶段：MQE 查询变体检索")
        variants = self._generate_query_variants(query)
        mque_results = defaultdict(lambda: {"similarity_scores": [], "metadata": None, "content": None})

        for variant in variants:
            search_result = json.loads(
                self.vector_store._search(collection_name, variant, top_k=top_k * 3, threshold=0)
            )

            if search_result.get("status") == "success":
                for chunk in search_result.get("chunks", []):
                    chunk_id = chunk["metadata"]["chunk_id"]
                    mque_results[chunk_id]["similarity_scores"].append(chunk["similarity"] * 0.5)  # 权重 0.5
                    mque_results[chunk_id]["metadata"] = chunk["metadata"]
                    mque_results[chunk_id]["content"] = chunk["content"]

        # 第三阶段：融合结果
        print(f"\n[PHASE3] 第三阶段：融合 HyDE 和 MQE 结果")
        all_results = {}

        # 合并 HyDE 结果
        for chunk_id, data in hyde_results.items():
            if chunk_id not in all_results:
                all_results[chunk_id] = {
                    "content": data["content"],
                    "metadata": data["metadata"],
                    "scores": []
                }
            all_results[chunk_id]["scores"].extend(data["similarity_scores"])

        # 合并 MQE 结果
        for chunk_id, data in mque_results.items():
            if chunk_id not in all_results:
                all_results[chunk_id] = {
                    "content": data["content"],
                    "metadata": data["metadata"],
                    "scores": []
                }
            all_results[chunk_id]["scores"].extend(data["similarity_scores"])

        # 计算最终相似度并排序
        fused_results = []
        for chunk_id, data in all_results.items():
            if data["scores"]:
                final_similarity = sum(data["scores"]) / len(data["scores"])
                if final_similarity >= threshold:
                    fused_results.append({
                        "chunk_id": chunk_id,
                        "similarity": round(final_similarity, 4),
                        "hybrid_sources": {
                            "hyde": len([s for s in hyde_results.get(chunk_id, {}).get("similarity_scores", [])]),
                            "mque": len([s for s in mque_results.get(chunk_id, {}).get("similarity_scores", [])])
                        },
                        "content": data["content"],
                        "metadata": data["metadata"]
                    })

        fused_results.sort(key=lambda x: x["similarity"], reverse=True)

        return {
            "status": "success",
            "strategy": "hybrid",
            "query": query,
            "hypotheses_count": len(hypotheses),
            "variants_count": len(variants),
            "results_count": len(fused_results[:top_k]),
            "chunks": fused_results[:top_k]
        }
