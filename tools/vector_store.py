# Chroma 向量库：单一多语 embedding（默认 bge-m3）→ 每主题一个文本集合。
# 插图仍走独立 CLIP 集合 `{base}_mm`（与文本维度无关）。
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from tools.base import Tool

_DEFAULT_MULTILINGUAL = "./models/bge-m3"
_DEFAULT_HUB = "BAAI/bge-m3"


class VectorStoreTool(Tool):
    """Chroma + 多语 Dense embedding（bge-m3）；可选 CLIP 插图集合。"""

    name = "vector_store"
    description = (
        "Store and retrieve text chunks in Chroma using a multilingual embedding model "
        "(default: bge-m3). Optional image collection via CLIP (*_mm)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "search", "delete", "add_images", "search_images"],
                "description": "Action: add / search / delete / add_images / search_images",
            },
            "collection_name": {"type": "string", "description": "Collection name (e.g., topic)"},
            "chunks": {"type": "array", "description": "Text chunks to add"},
            "images": {"type": "array", "description": "Image metadata for add_images"},
            "query": {"type": "string", "description": "Query text"},
            "top_k": {"type": "integer", "description": "Number of results (default: 5)"},
            "threshold": {"type": "number", "description": "Similarity threshold (default: 0.0)"},
        },
        "required": ["action", "collection_name"],
    }

    def __init__(
        self,
        db_path: str = "./storage/chroma",
        embedding_model: str = "",
        embedding_model_zh: str = "",  # deprecated, ignored
        device: str = "cpu",
        local_model_cache: str = "./models",
        bilingual_index: bool = False,  # deprecated: always single multilingual index
        index_langs: list[str] | None = None,  # deprecated
    ):
        """初始化 VectorStoreTool。

        Args:
            db_path: Chroma 路径
            embedding_model: 多语模型路径/ID（默认 EMBEDDING_MODEL 或 ./models/bge-m3）
            device: cpu / cuda
            local_model_cache: 本地模型目录
        """
        if bilingual_index or index_langs:
            print(
                "[WARN] bilingual_index/index_langs 已废弃：P2 使用单一多语集合 "
                f"(bilingual_index={bilingual_index}, index_langs={index_langs})",
                flush=True,
            )
        if embedding_model_zh:
            print("[WARN] embedding_model_zh 已废弃，请改用 EMBEDDING_MODEL=bge-m3", flush=True)

        model_cache_path = Path(local_model_cache)
        model_cache_path.mkdir(parents=True, exist_ok=True)

        os.environ["TRANSFORMERS_CACHE"] = str(model_cache_path.absolute())
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(model_cache_path.absolute())
        # 允许从本地目录加载；缺模型时由 _load 尝试联网/报错
        if not os.getenv("HF_HUB_OFFLINE"):
            pass

        print(f"[CACHE] 模型缓存目录: {model_cache_path.absolute()}")

        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.local_model_cache = str(model_cache_path)
        self.device = device
        # 兼容旧调用方
        self.bilingual_index = False
        self.index_langs = ["multi"]

        print(f"[DB] 初始化向量数据库: {self.db_path}")
        settings = Settings(
            is_persistent=True,
            persist_directory=str(self.db_path),
            anonymized_telemetry=False,
        )
        self.client = chromadb.Client(settings=settings)

        model_name = (
            (embedding_model or "").strip()
            or (os.getenv("EMBEDDING_MODEL") or "").strip()
            or (os.getenv("EMBEDDING_MODEL_MULTI") or "").strip()
            or _DEFAULT_MULTILINGUAL
        )
        print(f"[MODEL] 加载多语 Embedding: {model_name}", flush=True)
        self.embedding_model = self._load_sentence_transformer(model_name, device)
        self.model_name = model_name
        self.model_dim = self._embedding_dim(self.embedding_model)
        print(f"[OK] 多语模型就绪，维度: {self.model_dim} | {self.model_name}", flush=True)

        # 兼容旧属性名（advanced_retrieval / 评测脚本）
        self.embedding_model_en = self.embedding_model
        self.embedding_model_zh = self.embedding_model
        self.model_name_en = self.model_name
        self.model_name_zh = self.model_name
        self.model_dim_en = self.model_dim
        self.model_dim_zh = self.model_dim

    @staticmethod
    def _embedding_dim(model) -> int:
        if hasattr(model, "get_embedding_dimension"):
            return int(model.get_embedding_dimension())
        return int(model.get_sentence_embedding_dimension())

    def _load_sentence_transformer(self, model_id: str, device: str) -> SentenceTransformer:
        """从本地目录 / HF cache / 模型名加载 SentenceTransformer。"""
        cache = Path(self.local_model_cache)
        candidates: List[Path] = []

        p = Path(model_id)
        if p.exists():
            candidates.append(p)

        local_dir = cache / Path(model_id).name
        if local_dir.exists() and local_dir not in candidates:
            candidates.append(local_dir)

        # 常见默认：models/bge-m3
        default_local = cache / "bge-m3"
        if default_local.exists() and default_local not in candidates:
            candidates.append(default_local)

        hub_name = model_id.replace("/", "--")
        if not hub_name.startswith("models--"):
            hub_name = f"models--{hub_name}"
        hub_dir = cache / hub_name
        if hub_dir.exists():
            snap = hub_dir / "snapshots"
            if snap.exists():
                candidates.extend(sorted(snap.iterdir()))

        last_err: Optional[Exception] = None
        offline = os.getenv("HF_HUB_OFFLINE", "1").strip() not in ("0", "false", "False")
        for path in candidates:
            try:
                print(f"   从本地加载: {path}")
                if offline:
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    os.environ["TRANSFORMERS_OFFLINE"] = "1"
                return SentenceTransformer(str(path), device=device)
            except Exception as e:
                last_err = e
                print(f"   本地加载失败: {e}")

        allow_download = os.getenv("EMBEDDING_ALLOW_DOWNLOAD", "0").strip() in ("1", "true", "True", "yes")
        if allow_download or not candidates:
            try:
                os.environ.pop("HF_HUB_OFFLINE", None)
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
                hub_id = model_id if "/" in model_id else _DEFAULT_HUB
                print(f"   尝试从 Hub 加载: {hub_id}")
                return SentenceTransformer(hub_id, device=device)
            except Exception as e:
                last_err = e

        raise RuntimeError(
            f"无法加载多语 embedding「{model_id}」。"
            f"请下载到 {_DEFAULT_MULTILINGUAL}（modelscope: BAAI/bge-m3）。"
            f" last_err={last_err}"
        )

    def detect_lang(self, text: str) -> str:
        """兼容旧接口；多语模式下不再用于路由。"""
        from tools.bilingual_query import has_chinese

        return "zh" if has_chinese(text) else "en"

    def get_model_for_text(self, text: str = ""):
        """返回唯一的多语 embedding 模型。"""
        return self.embedding_model

    def _sanitize_collection_name(self, name: str) -> str:
        name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)
        name = name.strip("._-")
        name = re.sub(r"_+", "_", name)
        name = name[:63]
        if len(name) < 3:
            name = (name + "xxx")[:3]
        if name[0] in "._-":
            name = "c" + name[1:]
        if name[-1] in "._-":
            name = name[:-1] + "x"
        return name.lower()

    def _lang_collection_name(self, base: str, lang: str) -> str:
        """兼容旧调用；多语主集合不再使用语言后缀。"""
        if lang in ("multi", "m3", ""):
            return base
        return f"{base}_{lang}"

    def _resolve_text_collection(self, base: str):
        """解析文本集合：优先新单集合；仅当不存在时提示旧 _zh/_en（不可混用）。"""
        try:
            return self.client.get_collection(name=base), base
        except Exception:
            pass
        # 探测旧双集合，给出明确迁移提示（维度不兼容，不能直接查）
        legacy = []
        for suffix in ("_zh", "_en"):
            try:
                self.client.get_collection(name=f"{base}{suffix}")
                legacy.append(f"{base}{suffix}")
            except Exception:
                pass
        if legacy:
            raise RuntimeError(
                f"未找到多语集合 '{base}'，但发现旧双语集合 {legacy}。"
                f"P2（bge-m3）维度与旧索引不兼容，请用 benchmark/seed_chroma.py --rebuild 重建。"
            )
        return None, base

    def _encode_texts(self, texts: List[str]) -> List[List[float]]:
        arr = self.embedding_model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return arr.tolist()

    def _build_metadatas(self, clean_base: str, chunks: List[Dict[str, Any]]):
        ids, documents, metadatas = [], [], []
        for chunk in chunks:
            ids.append(f"{clean_base}_{chunk['chunk_id']}")
            documents.append(chunk["content"])
            metadata = {
                "chunk_id": str(chunk["chunk_id"]),
                "length": str(chunk.get("length", len(chunk["content"]))),
                "source": chunk.get("source", ""),
                "embedding": "bge-m3",
            }
            if "metadata" in chunk and isinstance(chunk["metadata"], dict):
                for key, value in chunk["metadata"].items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata[f"custom_{key}"] = str(value)
            metadatas.append(metadata)
        return ids, documents, metadatas

    def _add_chunks(self, collection_name: str, chunks: List[Dict[str, Any]]) -> str:
        """写入单一多语文本集合 `{base}`。"""
        try:
            clean_base = self._sanitize_collection_name(collection_name)
            ids, documents, metadatas = self._build_metadatas(clean_base, chunks)
            collection = self.client.get_or_create_collection(
                name=clean_base,
                metadata={
                    "hnsw:space": "cosine",
                    "embedding": "multilingual",
                    "model": str(self.model_name)[:120],
                },
            )
            embeddings = self._encode_texts(documents)
            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            print(f"[OK] 文本入库 {len(chunks)} chunks → {clean_base} (dim={self.model_dim})")
            return json.dumps(
                {
                    "status": "success",
                    "action": "add",
                    "collection": clean_base,
                    "collections_written": [clean_base],
                    "chunks_added": len(chunks),
                    "total_chunks": collection.count(),
                    "embedding_model": self.model_name,
                    "dim": self.model_dim,
                },
                ensure_ascii=False,
            )
        except Exception as e:
            print(f"[ERROR] 添加失败: {e}")
            return json.dumps({"status": "error", "action": "add", "error": str(e)}, ensure_ascii=False)

    def _search(self, collection_name: str, query: str, top_k: int = 5, threshold: float = 0.0) -> str:
        """多语 Dense 检索（不再按中/英路由）。"""
        try:
            clean_base = self._sanitize_collection_name(collection_name)
            try:
                collection, cname = self._resolve_text_collection(clean_base)
            except RuntimeError as e:
                return json.dumps({"status": "error", "action": "search", "error": str(e)}, ensure_ascii=False)
            if collection is None:
                return json.dumps(
                    {"status": "error", "action": "search", "error": f"Collection not found: {cname}"},
                    ensure_ascii=False,
                )

            print(f"[SEARCH][multi] 集合 '{cname}' | query='{query[:50]}...'")
            query_embedding = self._encode_texts([query])[0]
            return self._search_with_embedding(
                collection, cname, "multi", query, query_embedding, top_k, threshold
            )
        except Exception as e:
            print(f"[ERROR] 搜索失败: {e}")
            return json.dumps({"status": "error", "action": "search", "error": str(e)}, ensure_ascii=False)

    def _search_with_embedding(
        self,
        collection,
        cname: str,
        lang: str,
        query: str,
        query_embedding: list,
        top_k: int,
        threshold: float,
    ) -> str:
        n = min(top_k, max(collection.count(), 1))
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        retrieved_chunks = []
        docs = results.get("documents", [[]])[0]
        for i, doc in enumerate(docs):
            distance = results.get("distances", [[]])[0][i]
            similarity = 1 - distance
            if similarity >= threshold:
                retrieved_chunks.append({
                    "rank": i + 1,
                    "similarity": round(similarity, 4),
                    "content": doc,
                    "metadata": results.get("metadatas", [[]])[0][i],
                    "lang": lang,
                })

        print(f"[OK] 搜索完成，找到 {len(retrieved_chunks)} 个相关块")
        return json.dumps(
            {
                "status": "success",
                "action": "search",
                "collection": cname,
                "lang": lang,
                "query": query,
                "results_count": len(retrieved_chunks),
                "chunks": retrieved_chunks,
            },
            ensure_ascii=False,
            indent=2,
        )

    def search_by_embedding(
        self,
        collection_name: str,
        query_embedding,
        top_k: int = 5,
        threshold: float = 0.0,
        lang: str = "",
        query_label: str = "[prf]",
    ) -> str:
        """用已有向量检索（PRF / 查询向量改写）。"""
        try:
            clean_base = self._sanitize_collection_name(collection_name)
            collection, cname = self._resolve_text_collection(clean_base)
            if collection is None:
                return json.dumps(
                    {"status": "error", "action": "search", "error": f"Collection not found: {cname}"},
                    ensure_ascii=False,
                )
            emb = query_embedding.tolist() if hasattr(query_embedding, "tolist") else list(query_embedding)
            print(f"[SEARCH][multi] 集合 '{cname}' | embedding-query={query_label}")
            return self._search_with_embedding(
                collection, cname, lang or "multi", query_label, emb, top_k, threshold
            )
        except Exception as e:
            print(f"[ERROR] embedding 搜索失败: {e}")
            return json.dumps({"status": "error", "action": "search", "error": str(e)}, ensure_ascii=False)

    def get_collection_documents(self, collection_name: str) -> List[Dict[str, Any]]:
        """拉取文本集合文档。"""
        try:
            clean_base = self._sanitize_collection_name(collection_name)
            collection, _ = self._resolve_text_collection(clean_base)
            if collection is None:
                return []
            data = collection.get(include=["documents", "metadatas"])
            ids = data.get("ids") or []
            documents = data.get("documents") or []
            metadatas = data.get("metadatas") or []
            results: List[Dict[str, Any]] = []
            for i, doc_id in enumerate(ids):
                results.append({
                    "id": doc_id,
                    "content": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                })
            return results
        except Exception as e:
            print(f"[WARN] 拉取集合文档失败: {e}")
            return []

    def _delete_collection(self, collection_name: str) -> str:
        try:
            clean_base = self._sanitize_collection_name(collection_name)
            deleted = []
            for name in (
                clean_base,
                f"{clean_base}_mm",
                f"{clean_base}_en",  # 清理旧双语索引
                f"{clean_base}_zh",
            ):
                try:
                    self.client.delete_collection(name=name)
                    deleted.append(name)
                except Exception:
                    pass
            print(f"[OK] 已删除集合: {deleted or '无'}")
            return json.dumps(
                {"status": "success", "action": "delete", "deleted": deleted},
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps({"status": "error", "action": "delete", "error": str(e)}, ensure_ascii=False)

    def add_images(self, collection_name: str, images: List[Dict[str, Any]]) -> str:
        """将 PDF 插图写入多模态集合 `{base}_mm`（CLIP 图像向量）。"""
        try:
            from tools.multimodal_embedder import get_multimodal_embedder

            mm = get_multimodal_embedder()
            if not mm.available:
                return json.dumps(
                    {
                        "status": "skipped",
                        "action": "add_images",
                        "reason": "multimodal model unavailable",
                        "images": 0,
                    },
                    ensure_ascii=False,
                )

            clean_base = self._sanitize_collection_name(collection_name)
            cname = f"{clean_base}_mm"
            paths = []
            metas = []
            ids = []
            docs = []
            max_images = int(os.getenv("MULTIMODAL_MAX_IMAGES", "24") or "24")
            for img in images:
                if len(paths) >= max(1, max_images):
                    break
                p = str(img.get("path") or "").strip()
                if not p or not Path(p).exists():
                    continue
                try:
                    w = int(img.get("width") or 0)
                    h = int(img.get("height") or 0)
                except (TypeError, ValueError):
                    w, h = 0, 0
                if (w and w < 80) or (h and h < 80):
                    continue
                if w and h and w * h > 8_000_000:
                    continue
                iid = str(img.get("image_id") or Path(p).stem)
                paths.append(p)
                ids.append(f"{clean_base}_{iid}_mm")
                docs.append(f"[FIGURE] page={img.get('page', '')} path={p}")
                metas.append({
                    "chunk_id": iid,
                    "modality": "image",
                    "page": str(img.get("page", "")),
                    "path": p,
                    "width": str(img.get("width", "")),
                    "height": str(img.get("height", "")),
                    "source": str(img.get("source", "")),
                })

            if not paths:
                return json.dumps(
                    {"status": "success", "action": "add_images", "images": 0, "collection": cname},
                    ensure_ascii=False,
                )

            embeddings = mm.encode_images(paths).tolist()
            collection = self.client.get_or_create_collection(
                name=cname,
                metadata={"hnsw:space": "cosine", "modality": "image"},
            )
            collection.upsert(
                ids=ids,
                documents=docs,
                metadatas=metas,
                embeddings=embeddings,
            )
            print(f"[OK] 多模态入库 {len(paths)} 张图 → {cname}")
            return json.dumps(
                {
                    "status": "success",
                    "action": "add_images",
                    "collection": cname,
                    "images": len(paths),
                    "total": collection.count(),
                },
                ensure_ascii=False,
            )
        except Exception as e:
            print(f"[ERROR] 多模态入库失败: {e}")
            return json.dumps({"status": "error", "action": "add_images", "error": str(e)}, ensure_ascii=False)

    def search_images(self, collection_name: str, query: str, top_k: int = 5, threshold: float = 0.0) -> str:
        """用文本 query 的 CLIP 向量检索插图集合。"""
        try:
            from tools.multimodal_embedder import get_multimodal_embedder

            mm = get_multimodal_embedder()
            if not mm.available:
                return json.dumps(
                    {"status": "skipped", "action": "search_images", "chunks": [], "reason": "no multimodal model"},
                    ensure_ascii=False,
                )

            clean_base = self._sanitize_collection_name(collection_name)
            cname = f"{clean_base}_mm"
            try:
                collection = self.client.get_collection(name=cname)
            except Exception:
                return json.dumps(
                    {"status": "success", "action": "search_images", "collection": cname, "chunks": []},
                    ensure_ascii=False,
                )

            q_emb = mm.encode_texts([query])[0].tolist()
            n = min(top_k, max(collection.count(), 1))
            results = collection.query(
                query_embeddings=[q_emb],
                n_results=n,
                include=["documents", "metadatas", "distances"],
            )
            chunks = []
            docs = results.get("documents", [[]])[0]
            for i, doc in enumerate(docs):
                distance = results.get("distances", [[]])[0][i]
                similarity = 1 - distance
                if similarity < threshold:
                    continue
                meta = results.get("metadatas", [[]])[0][i] or {}
                chunks.append({
                    "rank": i + 1,
                    "similarity": round(similarity, 4),
                    "content": doc,
                    "metadata": meta,
                    "modality": "image",
                })
            print(f"[OK] 多模态检索命中 {len(chunks)} 张图 | {cname}")
            return json.dumps(
                {
                    "status": "success",
                    "action": "search_images",
                    "collection": cname,
                    "query": query,
                    "chunks": chunks,
                },
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps({"status": "error", "action": "search_images", "error": str(e)}, ensure_ascii=False)

    def run(self, tool_input: dict[str, Any]) -> str:
        action = str(tool_input.get("action", "")).lower().strip()
        collection_name = str(tool_input.get("collection_name", "")).strip()

        if not action or not collection_name:
            return json.dumps({"error": "action and collection_name are required"}, ensure_ascii=False)

        if action == "add":
            chunks = tool_input.get("chunks", [])
            if not chunks:
                return json.dumps({"error": "chunks are required for 'add' action"}, ensure_ascii=False)
            return self._add_chunks(collection_name, chunks)

        if action == "add_images":
            images = tool_input.get("images", [])
            if not images:
                return json.dumps({"error": "images are required for 'add_images'"}, ensure_ascii=False)
            return self.add_images(collection_name, images)

        if action == "search":
            query = str(tool_input.get("query", "")).strip()
            if not query:
                return json.dumps({"error": "query is required for 'search' action"}, ensure_ascii=False)
            top_k = int(tool_input.get("top_k", 5))
            threshold = float(tool_input.get("threshold", 0.0))
            return self._search(collection_name, query, top_k, threshold)

        if action == "search_images":
            query = str(tool_input.get("query", "")).strip()
            top_k = int(tool_input.get("top_k", 5))
            threshold = float(tool_input.get("threshold", 0.0))
            return self.search_images(collection_name, query, top_k, threshold)

        if action == "delete":
            return self._delete_collection(collection_name)

        return json.dumps({"error": f"Unknown action: {action}"}, ensure_ascii=False)
