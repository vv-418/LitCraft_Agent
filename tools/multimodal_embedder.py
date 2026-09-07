# 真·多模态图文向量：CLIP 文本与图像同一向量空间。
# 默认本地 ./models/clip-ViT-B-32；MULTIMODAL_ENABLED=0 时直接不可用。
# 支持两种本地布局：
#   1) sentence-transformers/clip-ViT-B-32（含 modules.json）
#   2) HuggingFace/transformers OpenAI CLIP（含 config.json model_type=clip）

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import numpy as np

_DEFAULT_LOCAL = "./models/clip-ViT-B-32"
_DEFAULT_HUB = "sentence-transformers/clip-ViT-B-32"


def _env_true(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _looks_like_st_clip(path: Path) -> bool:
    return (path / "modules.json").exists()


def _looks_like_hf_clip(path: Path) -> bool:
    cfg = path / "config.json"
    if not cfg.exists():
        return False
    try:
        import json

        data = json.loads(cfg.read_text(encoding="utf-8"))
        return str(data.get("model_type", "")).lower() == "clip" or "clip" in str(
            data.get("architectures", [])
        ).lower()
    except Exception:
        return False


class MultimodalEmbedder:
    """CLIP 图文编码器（懒加载）。不可用时 available=False。"""

    def __init__(self, model_path: str = "", device: str = "cpu"):
        self.model_path = (model_path or os.getenv("MULTIMODAL_MODEL") or _DEFAULT_LOCAL).strip()
        self.device = device
        self._backend: Optional[str] = None  # "st" | "hf"
        self._model: Any = None
        self._processor: Any = None
        self._dim = 0
        self._load_error: Optional[str] = None
        self._disabled = not _env_true("MULTIMODAL_ENABLED", "1")

    @property
    def available(self) -> bool:
        if self._disabled:
            return False
        self._ensure_loaded()
        return self._model is not None

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        return self._dim

    def _ensure_loaded(self) -> None:
        if self._disabled:
            self._load_error = "MULTIMODAL_ENABLED=0"
            return
        if self._model is not None or self._load_error is not None:
            return

        candidates: List[tuple[str, bool]] = []  # (path_or_id, is_local)
        local = Path(self.model_path)
        if local.exists():
            candidates.append((str(local.resolve()), True))
        default_local = Path(_DEFAULT_LOCAL)
        if default_local.exists() and str(default_local.resolve()) not in {c[0] for c in candidates}:
            candidates.append((str(default_local.resolve()), True))
        if _env_true("MULTIMODAL_ALLOW_DOWNLOAD", "0"):
            candidates.append((_DEFAULT_HUB, False))

        if not candidates:
            self._load_error = f"CLIP not found at {self.model_path}"
            print(
                f"[WARN] 多模态未就绪: {self._load_error}\n"
                f"       请下载到 {_DEFAULT_LOCAL}，或设 MULTIMODAL_ALLOW_DOWNLOAD=1"
            )
            return

        last_err: Optional[Exception] = None
        for cand, is_local in candidates:
            p = Path(cand) if is_local else None
            attempts = []
            if is_local and p is not None:
                # 残缺 ST 目录（仅有 modules.json）优先走 HF CLIP
                if _looks_like_hf_clip(p):
                    attempts.append("hf")
                if _looks_like_st_clip(p):
                    attempts.append("st")
                if not attempts:
                    attempts = ["st", "hf"]
            else:
                attempts = ["st"]

            for kind in attempts:
                try:
                    if is_local:
                        os.environ["HF_HUB_OFFLINE"] = "1"
                        os.environ["TRANSFORMERS_OFFLINE"] = "1"
                    if kind == "st":
                        if self._try_load_sentence_transformers(cand):
                            return
                    else:
                        if self._try_load_transformers_clip(cand):
                            return
                except Exception as e:
                    last_err = e
                    self._model = None
                    self._processor = None
                    self._backend = None
                    print(f"   [WARN] CLIP {kind} 加载失败，尝试下一路由: {e}")
                    continue

        self._load_error = str(last_err) if last_err else "load failed"
        print(f"[WARN] 多模态模型加载失败（跳过图文向量）: {self._load_error}")

    def _try_load_sentence_transformers(self, cand: str) -> bool:
        from sentence_transformers import SentenceTransformer

        print(f"[MODEL] 加载多模态 CLIP (sentence-transformers): {cand}")
        model = SentenceTransformer(cand, device=self.device)
        probe = model.encode(["probe"], convert_to_numpy=True, normalize_embeddings=True)
        self._model = model
        self._backend = "st"
        self._dim = int(np.asarray(probe).shape[-1])
        print(f"[OK] 多模态模型就绪，维度: {self._dim}")
        return True

    def _try_load_transformers_clip(self, cand: str) -> bool:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        print(f"[MODEL] 加载多模态 CLIP (transformers): {cand}")
        processor = CLIPProcessor.from_pretrained(cand, local_files_only=True)
        model = CLIPModel.from_pretrained(cand, local_files_only=True)
        model.eval()
        model.to(self.device)
        with torch.no_grad():
            feats = self._hf_text_features(model, processor, ["probe"])
            dim = int(feats.shape[-1])
        self._model = model
        self._processor = processor
        self._backend = "hf"
        self._dim = dim
        print(f"[OK] 多模态模型就绪，维度: {self._dim}")
        return True

    @staticmethod
    def _hf_text_features(model, processor, texts: Sequence[str]):
        """兼容新版 transformers：get_text_features 可能返回 BaseModelOutputWithPooling。"""
        import torch

        inputs = processor(text=list(texts), return_tensors="pt", padding=True, truncation=True)
        device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        text_out = model.text_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = text_out.pooler_output
        feats = model.text_projection(pooled)
        return feats / feats.norm(dim=-1, keepdim=True)

    @staticmethod
    def _hf_image_features(model, processor, images):
        import torch

        inputs = processor(images=list(images), return_tensors="pt", padding=True)
        device = next(model.parameters()).device
        pixel_values = inputs["pixel_values"].to(device)
        vision_out = model.vision_model(pixel_values=pixel_values)
        pooled = vision_out.pooler_output
        feats = model.visual_projection(pooled)
        return feats / feats.norm(dim=-1, keepdim=True)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not self.available:
            raise RuntimeError(self._load_error or "multimodal model unavailable")
        if self._backend == "st":
            arr = self._model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
            return np.asarray(arr, dtype=np.float32)

        import torch

        assert self._processor is not None
        with torch.no_grad():
            feats = self._hf_text_features(self._model, self._processor, texts)
            return feats.detach().cpu().numpy().astype(np.float32)

    def encode_images(self, image_paths: Sequence[Union[str, Path]]) -> np.ndarray:
        if not self.available:
            raise RuntimeError(self._load_error or "multimodal model unavailable")
        from PIL import Image

        images = [Image.open(p).convert("RGB") for p in image_paths]
        if self._backend == "st":
            arr = self._model.encode(images, convert_to_numpy=True, normalize_embeddings=True)
            return np.asarray(arr, dtype=np.float32)

        import torch

        assert self._processor is not None
        with torch.no_grad():
            feats = self._hf_image_features(self._model, self._processor, images)
            return feats.detach().cpu().numpy().astype(np.float32)


_default_mm: Optional[MultimodalEmbedder] = None


def get_multimodal_embedder() -> MultimodalEmbedder:
    global _default_mm
    if _default_mm is None:
        _default_mm = MultimodalEmbedder()
    return _default_mm
