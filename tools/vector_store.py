# 作用：实现向量数据库管理，使用 Chroma 存储和检索论文内容。
import json
import os
from pathlib import Path
from typing import Any, List, Dict, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from tools.base import Tool


class VectorStoreTool(Tool):
    """使用 Chroma 管理向量数据库的工具。"""

    name = "vector_store"
    description = "Store and retrieve text chunks in a vector database using Chroma."
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "search", "delete"],
                "description": "Action to perform: add (store), search (query), or delete (remove collection)"
            },
            "collection_name": {"type": "string", "description": "Collection name (e.g., paper title)"},
            "chunks": {"type": "array", "description": "Text chunks to add (required for 'add' action)"},
            "query": {"type": "string", "description": "Query text (required for 'search' action)"},
            "top_k": {"type": "integer", "description": "Number of results to return for search (default: 5)"},
            "threshold": {"type": "number", "description": "Similarity threshold for search (0-1, default: 0.0)"},
        },
        "required": ["action", "collection_name"],
    }

    def __init__(
        self,
        db_path: str = "./storage/chroma",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        local_model_cache: str = "./models"
    ):
        """初始化 VectorStoreTool。
        
        Args:
            db_path: Chroma 数据库路径
            embedding_model: Sentence Transformer 模型名称
            device: 计算设备 ('cpu' 或 'cuda')
            local_model_cache: 本地模型缓存目录（替代 C: 盘缓存）
        """
        # 设置模型缓存目录到项目内（而不是 C 盘用户目录）
        model_cache_path = Path(local_model_cache)
        model_cache_path.mkdir(parents=True, exist_ok=True)
        
        # 设置环境变量，让 sentence-transformers 和 transformers 使用本地缓存
        os.environ['TRANSFORMERS_CACHE'] = str(model_cache_path.absolute())
        os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(model_cache_path.absolute())
        # 默认优先尝试离线模式（但允许首次下载时联网）
        if 'HF_HUB_OFFLINE' not in os.environ:
            os.environ['HF_HUB_OFFLINE'] = '0'  # 首次下载允许，之后会缓存
        
        print(f"[CACHE] 模型缓存目录: {model_cache_path.absolute()}")
        
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化 Chroma 客户端
        print(f"[DB] 初始化向量数据库: {self.db_path}")
        # New Chroma Settings: enable persistent mode and set persist directory
        settings = Settings(
            is_persistent=True,
            persist_directory=str(self.db_path),
            anonymized_telemetry=False
        )

        # Initialize client with settings
        self.client = chromadb.Client(settings=settings)
        
        # 加载 Embedding 模型
        print(f"[MODEL] 加载 Embedding 模型: {embedding_model}")
        
        # 优先使用 HuggingFace Hub 缓存目录中的模型（离线模式）
        hf_cache_dir = Path(local_model_cache) / f"models--{embedding_model.replace('/', '--')}"
        snapshot_dirs = []
        if hf_cache_dir.exists():
            snapshots_dir = hf_cache_dir / "snapshots"
            if snapshots_dir.exists():
                snapshot_dirs = sorted(snapshots_dir.iterdir())
        
        if snapshot_dirs:
            local_path = str(snapshot_dirs[0])
            print(f"   从本地缓存加载: {local_path}")
            try:
                self.embedding_model = SentenceTransformer(local_path, device=device)
                self.model_name = embedding_model
                self.model_dim = self.embedding_model.get_embedding_dimension()
                print(f"[OK] 模型加载成功，维度: {self.model_dim}")
            except Exception as e:
                print(f"   本地缓存加载失败: {e}，尝试在线加载...")
                self.embedding_model = SentenceTransformer(embedding_model, device=device)
                self.model_name = embedding_model
                self.model_dim = self.embedding_model.get_embedding_dimension()
                print(f"[OK] 模型加载成功")
                print(f"   维度: {self.model_dim}")
        else:
            # 未找到本地缓存，尝试在线加载
            try:
                self.embedding_model = SentenceTransformer(embedding_model, device=device)
                self.model_name = embedding_model
                self.model_dim = self.embedding_model.get_embedding_dimension()
                print(f"[OK] 模型加载成功")
                print(f"   维度: {self.model_dim}")
            except Exception as e:
                error_msg = str(e)
                print(f"[ERROR] 加载失败: {error_msg}")
                print(f"   尝试使用降级模型...")
                try:
                    self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
                    self.model_name = "all-MiniLM-L6-v2"
                    self.model_dim = 384
                    print(f"[OK] 降级模型加载成功，维度: {self.model_dim}")
                except Exception as e2:
                    print(f"[ERROR] 降级模型也失败: {e2}")
                    raise

    def _sanitize_collection_name(self, name: str) -> str:
        """清理集合名称，使其符合 Chroma 要求。
        
        Args:
            name: 原始名称
            
        Returns:
            清理后的名称
        """
        # Chroma 要求: 3-63个字符，只能包含字母、数字、下划线、横线和点
        import re
        
        # 替换不允许的字符
        name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', name)
        
        # 确保长度在 3-63 之间
        name = name[:63]
        if len(name) < 3:
            name = ("x" * (3 - len(name))) + name
        
        return name.lower()

    def _add_chunks(self, collection_name: str, chunks: List[Dict[str, Any]]) -> str:
        """将文本块添加到向量数据库。
        
        Args:
            collection_name: 集合名称
            chunks: 文本块列表
            
        Returns:
            JSON 字符串，包含添加结果
        """
        try:
            # 清理集合名称
            clean_name = self._sanitize_collection_name(collection_name)
            
            # 获取或创建集合
            collection = self.client.get_or_create_collection(
                name=clean_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            print(f"[ADD] 向集合 '{clean_name}' 添加 {len(chunks)} 个块...")
            
            # 准备数据
            ids = []
            documents = []
            metadatas = []
            embeddings = []
            
            for chunk in chunks:
                chunk_id = f"{clean_name}_{chunk['chunk_id']}"
                ids.append(chunk_id)
                documents.append(chunk['content'])
                
                # 构建元数据
                metadata = {
                    "chunk_id": str(chunk['chunk_id']),
                    "length": str(chunk.get('length', len(chunk['content']))),
                    "source": chunk.get('source', '')
                }
                
                # 添加自定义元数据
                if 'metadata' in chunk and isinstance(chunk['metadata'], dict):
                    for key, value in chunk['metadata'].items():
                        if isinstance(value, (str, int, float, bool)):
                            metadata[f"custom_{key}"] = str(value)
                
                metadatas.append(metadata)
                
                # 生成 embedding
                embedding = self.embedding_model.encode(chunk['content']).tolist()
                embeddings.append(embedding)
            
            # 添加到 Chroma
            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            # 注意：chromadb 1.5.x 在 is_persistent=True 时自动持久化
            
            print(f"[OK] 成功添加 {len(chunks)} 个块")
            
            return json.dumps(
                {
                    "status": "success",
                    "action": "add",
                    "collection": clean_name,
                    "chunks_added": len(chunks),
                    "total_chunks": collection.count()
                },
                ensure_ascii=False
            )
        
        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR] 添加失败: {error_msg}")
            return json.dumps(
                {
                    "status": "error",
                    "action": "add",
                    "error": error_msg
                },
                ensure_ascii=False
            )

    def _search(self, collection_name: str, query: str, top_k: int = 5, threshold: float = 0.0) -> str:
        """在向量数据库中搜索相似的文本。
        
        Args:
            collection_name: 集合名称
            query: 查询文本
            top_k: 返回结果数量
            threshold: 相似度阈值
            
        Returns:
            JSON 字符串，包含搜索结果
        """
        try:
            # 清理集合名称
            clean_name = self._sanitize_collection_name(collection_name)
            
            # 获取集合
            collection = self.client.get_collection(name=clean_name)
            
            print(f"[SEARCH] 在集合 '{clean_name}' 中搜索: '{query[:50]}...'")
            
            # 生成查询 embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # 搜索
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["embeddings", "documents", "metadatas", "distances"]
            )
            
            # 处理结果
            retrieved_chunks = []
            for i, doc in enumerate(results.get("documents", [[]])[0]):
                distance = results.get("distances", [[]])[0][i]
                # 距离转换为相似度（余弦距离: 1 - distance）
                similarity = 1 - distance
                
                if similarity >= threshold:
                    metadata = results.get("metadatas", [[]])[0][i]
                    retrieved_chunks.append({
                        "rank": i + 1,
                        "similarity": round(similarity, 4),
                        "content": doc,
                        "metadata": metadata
                    })
            
            print(f"[OK] 搜索完成，找到 {len(retrieved_chunks)} 个相关块")
            
            return json.dumps(
                {
                    "status": "success",
                    "action": "search",
                    "collection": clean_name,
                    "query": query,
                    "results_count": len(retrieved_chunks),
                    "chunks": retrieved_chunks
                },
                ensure_ascii=False,
                indent=2
            )
        
        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR] 搜索失败: {error_msg}")
            return json.dumps(
                {
                    "status": "error",
                    "action": "search",
                    "error": error_msg
                },
                ensure_ascii=False
            )

    def _delete_collection(self, collection_name: str) -> str:
        """删除一个集合。
        
        Args:
            collection_name: 集合名称
            
        Returns:
            JSON 字符串，包含删除结果
        """
        try:
            clean_name = self._sanitize_collection_name(collection_name)
            
            self.client.delete_collection(name=clean_name)
            # 注意：chromadb 1.5.x 在 is_persistent=True 时自动持久化
            
            print(f"[OK] 集合 '{clean_name}' 已删除")
            
            return json.dumps(
                {
                    "status": "success",
                    "action": "delete",
                    "collection": clean_name,
                    "message": f"Collection '{clean_name}' deleted"
                },
                ensure_ascii=False
            )
        
        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR] 删除失败: {error_msg}")
            return json.dumps(
                {
                    "status": "error",
                    "action": "delete",
                    "error": error_msg
                },
                ensure_ascii=False
            )

    def run(self, tool_input: dict[str, Any]) -> str:
        """执行向量数据库操作。
        
        Args:
            tool_input: 包含操作信息的字典
            
        Returns:
            JSON 字符串，包含操作结果
        """
        action = str(tool_input.get("action", "")).lower().strip()
        collection_name = str(tool_input.get("collection_name", "")).strip()
        
        if not action or not collection_name:
            return json.dumps(
                {"error": "action and collection_name are required"},
                ensure_ascii=False
            )
        
        if action == "add":
            chunks = tool_input.get("chunks", [])
            if not chunks:
                return json.dumps(
                    {"error": "chunks are required for 'add' action"},
                    ensure_ascii=False
                )
            return self._add_chunks(collection_name, chunks)
        
        elif action == "search":
            query = str(tool_input.get("query", "")).strip()
            if not query:
                return json.dumps(
                    {"error": "query is required for 'search' action"},
                    ensure_ascii=False
                )
            top_k = int(tool_input.get("top_k", 5))
            threshold = float(tool_input.get("threshold", 0.0))
            return self._search(collection_name, query, top_k, threshold)
        
        elif action == "delete":
            return self._delete_collection(collection_name)
        
        else:
            return json.dumps(
                {"error": f"Unknown action: {action}"},
                ensure_ascii=False
            )
