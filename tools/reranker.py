# Cross-Encoder 重排：bge-reranker-v2-m3（本地）。
# 用法：一阶段宽召回后，对 (query, passage) 打分再截断 Top-K。

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

_DEFAULT_MODEL = "./models/bge-reranker-v2-m3"


def _auto_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class Reranker:
    """懒加载 CrossEncoder；失败时 degrade 为按原序返回。"""

    def __init__(
        self,
        model_path: str = "",
        device: str = "",
        max_length: int = 512,
    ):
        self.model_path = (
            (model_path or os.getenv("RERANKER_MODEL") or _DEFAULT_MODEL).strip()
        )
        self.device = (device or os.getenv("RERANKER_DEVICE") or "").strip() or _auto_device()
        self.max_length = max_length
        self._model = None
        self._load_error: Optional[str] = None

    @property
    def available(self) -> bool:
        self._ensure_loaded()
        return self._model is not None

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._load_error is not None:
            return
        path = Path(self.model_path)
        if not path.exists():
            self._load_error = f"reranker model not found: {path}"
            print(f"[WARN] {self._load_error}")
            return
        try:
            # 强制离线，避免误拉 Hub
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from sentence_transformers import CrossEncoder

            print(f"[MODEL] 加载 Reranker: {path.resolve()}")
            self._model = CrossEncoder(
                str(path),
                device=self.device,
                max_length=self.max_length,
            )
            print("[OK] Reranker 就绪")
        except Exception as e:
            self._load_error = str(e)
            print(f"[WARN] Reranker 加载失败，将跳过精排: {e}")

    def rerank(
        self,
        query: str,
        passages: Sequence[str],
        top_k: int = 10,
        truncate_chars: int = 500,
    ) -> List[Tuple[int, float]]:
        """对 passages 重排，返回 [(原下标, score), ...] 按分数降序，长度 ≤ top_k。"""
        if not passages:
            return []
        n = len(passages)
        top_k = max(1, min(int(top_k), n))
        self._ensure_loaded()
        if self._model is None:
            return [(i, float(n - i)) for i in range(top_k)]

        pairs = [
            (query or "", (p or "")[:truncate_chars])
            for p in passages
        ]
        try:
            scores = self._model.predict(pairs, show_progress_bar=False)
        except Exception as e:
            print(f"[WARN] Reranker 推理失败，回退原序: {e}")
            return [(i, float(n - i)) for i in range(top_k)]

        ranked = sorted(
            enumerate(float(s) for s in scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]


_default_reranker: Optional[Reranker] = None


def get_reranker() -> Reranker:
    global _default_reranker
    if _default_reranker is None:
        _default_reranker = Reranker()
    return _default_reranker
