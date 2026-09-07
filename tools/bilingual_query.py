"""
双语 / 多查询构造（零 LLM）。

站外学术搜索：中文主题 = 原句 + 合格英译（最多两路）；英文主题 = 仅原句。
本地向量检索另用 build_vector_queries（可更保守）。

在线翻译（TRANSLATE_ONLINE=1，默认开启）：
  auto 顺序：百度（若配置密钥）→ MyMemory（免密钥）→ Google（短超时）→ 词典。
  国内网络下 Google 常不可达；百度/有道官方 API 需自行申请密钥，不可免注册直连。
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import List

_RE_CHINESE = re.compile(r"[\u4e00-\u9fff]")
_RE_LATIN_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\-+/]{1,}")
_RE_ACRONYM = re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)?\b")

_STOP_LATIN = {
    "the", "and", "for", "with", "from", "that", "this", "based", "application",
    "of", "in", "on", "to", "a", "an", "or", "as", "by", "at", "is", "are",
}

# 常见中文学术短语 → 英文（兜底词典，翻译失败时用）
_PHRASE_MAP = {
    "检索增强生成": "retrieval augmented generation",
    "检索增强": "retrieval augmented",
    "大语言模型": "large language model",
    "语言模型": "language model",
    "自然语言处理": "natural language processing",
    "知识图谱": "knowledge graph",
    "问答": "question answering",
    "医学问答": "medical question answering",
    "多模态": "multimodal",
    "推荐系统": "recommender system",
    "联邦学习": "federated learning",
    "对比学习": "contrastive learning",
    "预训练": "pretraining",
    "微调": "fine-tuning",
    "注意力机制": "attention mechanism",
    "自注意力": "self attention",
    "图神经网络": "graph neural network",
    "目标检测": "object detection",
    "语义分割": "semantic segmentation",
    "信息检索": "information retrieval",
    "向量检索": "vector retrieval",
    "密集检索": "dense retrieval",
    "文献综述": "literature review",
    "综述": "survey",
    "Transformer": "Transformer",
    "跨模态": "cross modal",
    "重排序": "reranking",
    "应用": "application",
    "中的应用": "application in",
    "基于": "based on",
    # 交通 / 气象 / 感知
    "高速公路": "highway",
    "快速路": "expressway",
    "智能交通": "intelligent transportation",
    "交通流": "traffic flow",
    "路况": "road condition",
    "天气状况": "weather condition",
    "气象": "meteorology",
    "天气": "weather",
    "雨雾": "rain fog",
    "能见度": "visibility",
    "路面湿滑": "wet slippery road",
    "积雪": "snow cover",
    "结冰": "icing",
    "识别": "recognition",
    "检测": "detection",
    "分类": "classification",
    "计算机视觉": "computer vision",
    "深度学习": "deep learning",
    "卷积神经网络": "convolutional neural network",
}


def has_chinese(text: str) -> bool:
    return bool(_RE_CHINESE.search(text or ""))


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    return text.strip('.,;:!?"\'')


def _meaningful_latin_tokens(text: str) -> List[str]:
    return [
        t for t in _RE_LATIN_TOKEN.findall(text or "")
        if len(t) >= 3 and t.lower() not in _STOP_LATIN
    ]


def _call_with_timeout(fn, timeout_sec: float):
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(fn)
        return fut.result(timeout=timeout_sec)


def _glossary_translate(text: str) -> str:
    replaced = text
    for zh, en in sorted(_PHRASE_MAP.items(), key=lambda x: -len(x[0])):
        if zh in replaced:
            replaced = replaced.replace(zh, f" {en} ")
    replaced = _RE_CHINESE.sub(" ", replaced)
    replaced = _clean(re.sub(r"\s+", " ", replaced))
    if replaced:
        return replaced
    terms = _RE_LATIN_TOKEN.findall(text)
    return _clean(" ".join(terms))


def _online_backends() -> List[str]:
    """解析 TRANSLATE_PROVIDER：auto | mymemory | google | baidu。"""
    raw = (os.getenv("TRANSLATE_PROVIDER") or "auto").strip().lower()
    if raw in ("mymemory", "google", "baidu"):
        return [raw]
    # auto：国内优先免密钥 MyMemory；配置了百度密钥则优先百度；Google 放最后（常超时）
    order: List[str] = []
    if os.getenv("BAIDU_TRANSLATE_APPID") and os.getenv("BAIDU_TRANSLATE_SECRET"):
        order.append("baidu")
    order.extend(["mymemory", "google"])
    return order


def _try_online_translate(text: str) -> str:
    timeout = float(os.getenv("TRANSLATE_TIMEOUT", "8") or 8)
    for backend in _online_backends():
        try:
            if backend == "mymemory":
                from deep_translator import MyMemoryTranslator

                out = _call_with_timeout(
                    lambda: MyMemoryTranslator(source="zh-CN", target="en-US").translate(text),
                    timeout,
                )
            elif backend == "google":
                from deep_translator import GoogleTranslator

                out = _call_with_timeout(
                    lambda: GoogleTranslator(source="auto", target="en").translate(text),
                    timeout,
                )
            elif backend == "baidu":
                from deep_translator import BaiduTranslator

                appid = os.getenv("BAIDU_TRANSLATE_APPID", "").strip()
                secret = os.getenv("BAIDU_TRANSLATE_SECRET", "").strip()
                if not appid or not secret:
                    continue
                out = _call_with_timeout(
                    lambda: BaiduTranslator(
                        appid=appid, appkey=secret, source="zh", target="en"
                    ).translate(text),
                    timeout,
                )
            else:
                continue
            out = _clean(out or "")
            if out and out.lower() != text.lower() and _meaningful_latin_tokens(out):
                print(f"[TRANSLATE] {backend}: {text[:40]} → {out[:80]}")
                return out
        except FuturesTimeout:
            print(f"[TRANSLATE] {backend} 超时（>{timeout}s），尝试下一后端")
        except Exception as e:
            print(f"[TRANSLATE] {backend} 失败: {type(e).__name__}: {str(e)[:80]}")
    return ""


def translate_to_english(text: str, use_online: bool | None = None) -> str:
    """中文 → 英文。

    TRANSLATE_ONLINE 默认开启（未设置或非 0）。设 0/false 则仅用词典。
    """
    text = _clean(text)
    if not text:
        return ""
    if not has_chinese(text):
        return text

    if use_online is None:
        use_online = os.getenv("TRANSLATE_ONLINE", "1").strip().lower() not in (
            "0", "false", "no", "off",
        )

    if use_online:
        online = _try_online_translate(text)
        if online:
            return online

    return _glossary_translate(text)


def extract_keyword_query(text: str) -> str:
    """抽出缩写 + 拉丁技术词，适合 BM25 / 学术库关键词检索。"""
    text = text or ""
    acronyms = _RE_ACRONYM.findall(text)
    latin = [t for t in _RE_LATIN_TOKEN.findall(text) if len(t) > 2]
    # 去重保序
    seen = set()
    tokens: List[str] = []
    for t in acronyms + latin:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            tokens.append(t)
    return _clean(" ".join(tokens[:12]))


def build_search_queries(topic: str, max_queries: int = 2) -> List[str]:
    """站外学术搜索查询构造（对齐「网站搜索框」：少改写）。

    - 中文：原句 + 合格英译（最多 2）；英译不达标则只保留原句
    - 英文：仅原句（不加关键词核）
    """
    original = _clean(topic)
    if not original:
        return []

    limit = max(1, min(int(max_queries or 2), 2))
    queries: List[str] = [original]

    if not has_chinese(original):
        return queries[:limit]

    en = translate_to_english(original)
    if not en or en.lower() == original.lower():
        return queries[:limit]

    meaningful = _meaningful_latin_tokens(en)
    # 至少 2 个有效学术词，避免 "of based on" 一类噪声
    if len(meaningful) < 2:
        return queries[:limit]

    if en.lower() not in {q.lower() for q in queries}:
        queries.append(en)
    return queries[:limit]


def build_vector_queries(topic: str, max_queries: int = 3) -> List[str]:
    """向量库检索专用查询：比网页检索更保守，优先保证召回不被噪声稀释。

    - 始终保留原句
    - 英文主题：可加关键词核
    - 中文主题：仅当英译含足够术语时才加入（学术中英对照场景）；
      日常中文问答默认只用原句，由 Dense 多语模型负责召回
    """
    original = _clean(topic)
    if not original:
        return []

    queries: List[str] = [original]
    if not has_chinese(original):
        kw = extract_keyword_query(original)
        if kw and kw.lower() != original.lower() and len(kw) >= 3:
            queries.append(kw)
        return queries[:max_queries]

    # 中文：只有「像样的」英译才进入向量多路（避免 of/based on 碎片）
    en = translate_to_english(original)
    if en:
        meaningful = _meaningful_latin_tokens(en)
        # 至少 3 个有效英文词，才像学术主题翻译
        if len(meaningful) >= 3 and en.lower() not in {q.lower() for q in queries}:
            queries.append(en)
            kw = extract_keyword_query(en)
            if kw and kw.lower() not in {q.lower() for q in queries} and len(kw) >= 3:
                queries.append(kw)

    return queries[:max_queries]
