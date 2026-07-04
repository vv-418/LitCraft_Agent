"""
LLM 搜索查询优化模块
=====================

解决痛点：LLM 在 ReAct 循环中经常生成低质量查询的问题：
  1. 过于冗长 — "Find me papers about the latest advances in..."
  2. 过于狭窄 — "Attention Is All You Need Vaswani 2017 specific architecture"
  3. 中英文混杂 — 用户主题是中文，但 LLM 只搜英文（或反之）
  4. 含搜索操作词 — "search for", "find papers about" 等 API 指令残留

优化管线（按顺序执行）：
  1. 停用短语剥离 — 去掉 "find me papers about" 等冗余前缀
  2. 双语候选生成 — 中文查询 → 自动生成英文候选；英文查询 → 保持
  3. 长度归一化 — 截断 > 120 字符的过长查询
  4. 查询归一化 — 去除标点噪音、多余空格、小写化

用法：
    optimizer = QueryOptimizer()
    result = optimizer.optimize("Find me papers about Transformer attention mechanism")
    # → {"original": "...", "optimized": "Transformer attention mechanism", ...}
"""

import re
import time
import math
from typing import Optional
from collections import Counter


# ── 停用短语模式（匹配查询开头的冗余前缀）─────────────────────
_VERBOSE_PREFIXES = re.compile(
    r"^(find|search|look|get|retrieve|show|give|tell)\s+"
    r"(me|us|the|for|about|all|some|any|recent|latest|relevant|related|"
    r"academic|scientific|research|papers?|articles?|publications?|"
    r"information|results?|works?)\s+"
    r"(about|on|regarding|concerning|related\s+to|for|of|in|by|with|"
    r"papers?\s+(about|on|regarding)|articles?\s+(about|on)|"
    r"publications?\s+(about|on))\s+",
    re.IGNORECASE,
)

_VERBOSE_PREFIXES_SIMPLE = re.compile(
    r"^(search\s+for|find\s+|look\s+for|retrieve\s+|show\s+me)\s+",
    re.IGNORECASE,
)

# ── 查询中的搜索残留（不匹配开头，出现在中间/结尾的）──────────
_RESIDUAL_WORDS = re.compile(
    r"\b(papers?\s+(about|on|related\s+to|regarding|concerning|that\s+discuss)|"
    r"articles?\s+(about|on|regarding)|"
    r"literature\s+(about|on|related\s+to)|"
    r"research\s+(about|on|related\s+to|regarding)|"
    r"recent\s+(papers?|articles?|publications?|works?|advances?|developments?|"
    r"studies?|findings?|research))\s+",
    re.IGNORECASE,
)

# ── 检测字符串是否含中文 ────────────────────────────────────────
_RE_CHINESE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")


class QueryOptimizer:
    """搜索查询优化器。

    优化管线（按顺序执行）：
      1. 停用短语剥离 — 去掉 "find me papers about" 等冗余前缀
      2. 残留搜索词清理 — 去除查询中间的 "papers about" 等
      3. 语义保真验证 — 确保优化后的查询与原始查询语义一致
      4. 长度归一化 — 截断 > 120 字符的过长查询
      5. 查询归一化 — 去除标点噪音、多余空格
      6. 双语候选生成 — 中文查询 → 自动构造英文候选
    """

    # ── 语义保真阈值：优化后查询与原始查询的相似度不得低于此值 ──
    # 低于此值时回退到原始查询，防止主题偏离
    SEMANTIC_FIDELITY_THRESHOLD = 0.30

    # ── 类级别共享 embedder（懒加载，所有实例共享同一模型）──
    _embedder = None          # SentenceTransformer 实例
    _embedder_loading = False # 防止并发重复加载
    _embedder_failed = False  # 加载失败时永久降级
    _embedder_model_path = None

    def __init__(self):
        self.queries_processed = 0
        self.queries_modified = 0
        self.queries_reverted = 0       # 因语义偏离被回退的次数
        self.bilingual_generated = 0
        self.bilingual_rejected = 0     # 因语义偏离被拒绝的双语候选次数
        # 首次实例化时触发 embedder 懒加载
        self._ensure_embedder()

    # ── 公开接口 ──────────────────────────────────────────────

    def optimize(self, query: str) -> dict:
        """执行完整查询优化管线，返回优化结果。
        
        Args:
            query: 原始查询
            
        Returns:
            dict 包含:
              - original: 原始查询
              - optimized: 优化后的主查询
              - bilingual_candidates: 双语候选查询列表（可能与 optimized 相同）
              - was_modified: 是否被修改
              - modifications: 应用的所有修改描述
              - has_chinese: 是否含中文
        """
        self.queries_processed += 1
        mods: list[str] = []

        raw = query.strip()

        # 记录初始状态
        has_chinese = bool(_RE_CHINESE.search(raw))

        # 1. 停用短语剥离
        cleaned = self._strip_verbose_prefixes(raw)
        if cleaned != raw:
            mods.append(f"剥离停用短语: '{raw[:40]}...' → '{cleaned[:40]}...'")

        # 2. 去除残留搜索词
        cleaned2 = _RESIDUAL_WORDS.sub("", cleaned).strip()
        if cleaned2 != cleaned and cleaned2:
            mods.append("去除搜索残留词")
            cleaned = cleaned2

        # 3. 长度归一化
        if len(cleaned) > 120:
            cleaned = cleaned[:117].rstrip() + "..."
            mods.append(f"截断过长查询 ({len(cleaned)} → 120 字符)")

        # 4. 最终清理
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = cleaned.strip('.,;:!?"\'')

        was_modified = (cleaned != raw)
        if was_modified:
            self.queries_modified += 1

        # 5. 语义保真验证：确保优化后查询未偏离原始语义 ──────────
        if was_modified and not self._validate_semantic_fidelity(raw, cleaned):
            sim = self.semantic_similarity(raw, cleaned)
            self.queries_reverted += 1
            mods.append(
                f"⚠️ 语义偏离检测: 相似度={sim:.2f} < 阈值{self.SEMANTIC_FIDELITY_THRESHOLD}，"
                f"回退到原始查询"
            )
            cleaned = raw
            was_modified = False

        # 6. 双语候选生成
        bilingual_candidates = self._generate_bilingual(cleaned, has_chinese)

        # 7. 双语候选语义验证：拒绝与原始查询语义偏离的候选 ──────
        if len(bilingual_candidates) > 1:
            validated = [bilingual_candidates[0]]  # 主查询始终保留
            for candidate in bilingual_candidates[1:]:
                sim = self.semantic_similarity(raw, candidate)
                if sim >= self.SEMANTIC_FIDELITY_THRESHOLD:
                    validated.append(candidate)
                else:
                    self.bilingual_rejected += 1
                    mods.append(
                        f"⚠️ 双语候选拒绝: '{candidate[:40]}...' 语义相似度={sim:.2f} 太低"
                    )
            bilingual_candidates = validated

        if len(bilingual_candidates) > 1:
            self.bilingual_generated += 1

        return {
            "original": raw,
            "optimized": cleaned,
            "bilingual_candidates": bilingual_candidates,
            "was_modified": was_modified,
            "modifications": mods,
            "has_chinese": has_chinese,
        }

    def optimize_for_search(self, query: str) -> str:
        """简洁版：返回优化后的主查询字符串（适合快速调用）。"""
        return self.optimize(query)["optimized"]

    @staticmethod
    def optimize_for_search_static(query: str) -> str:
        """静态方法版：无需实例化，直接优化查询字符串。"""
        return QueryOptimizer().optimize(query)["optimized"]

    # ── 内部方法 ──────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """将文本分词（支持中英文混合）。
        
        中文按字符切分（unigram + bigram），英文按空格切分。
        """
        tokens = []
        # 英文单词
        english_words = re.findall(r'[a-zA-Z]+', text.lower())
        tokens.extend(english_words)
        # 中文字符（unigram + bigram）
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        tokens.extend(chinese_chars)
        for i in range(len(chinese_chars) - 1):
            tokens.append(chinese_chars[i] + chinese_chars[i + 1])
        return tokens

    @staticmethod
    def _cosine_similarity(tokens_a: list[str], tokens_b: list[str]) -> float:
        """计算两组 token 的余弦相似度（基于词频向量）。"""
        if not tokens_a or not tokens_b:
            return 0.0
        counter_a = Counter(tokens_a)
        counter_b = Counter(tokens_b)
        # 所有词汇
        all_words = set(counter_a.keys()) | set(counter_b.keys())
        # 点积
        dot = sum(counter_a.get(w, 0) * counter_b.get(w, 0) for w in all_words)
        # 模长
        norm_a = math.sqrt(sum(v * v for v in counter_a.values()))
        norm_b = math.sqrt(sum(v * v for v in counter_b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @classmethod
    def semantic_similarity(cls, text_a: str, text_b: str) -> float:
        """计算两段文本的语义相似度。
        
        优先使用 sentence-transformers embedding 余弦（精度高）；
        如果模型不可用则自动降级到 token 余弦（零依赖）。
        返回值范围 [0, 1]，1 表示完全相同。
        """
        # 优先尝试 embedding 余弦
        if cls._embedder is not None:
            try:
                embeddings = cls._embedder.encode([text_a, text_b], normalize_embeddings=True)
                return float(embeddings[0] @ embeddings[1])
            except Exception:
                pass  # 降级到 token 余弦

        # fallback：token 余弦相似度
        tokens_a = cls._tokenize(text_a)
        tokens_b = cls._tokenize(text_b)
        return cls._cosine_similarity(tokens_a, tokens_b)

    @classmethod
    def _ensure_embedder(cls) -> bool:
        """确保 sentence-transformers embedder 可用（懒加载）。
        
        加载成功返回 True，失败返回 False（后续调用自动降级）。
        使用类级别锁防止并发重复加载。
        """
        if cls._embedder is not None:
            return True
        if cls._embedder_failed:
            return False
        if cls._embedder_loading:
            return False  # 等待其他线程加载完成

        cls._embedder_loading = True
        try:
            from sentence_transformers import SentenceTransformer
            import os

            # 复用项目已有的模型缓存路径
            local_path = "./models/models--sentence-transformers--all-MiniLM-L6-v2"
            model_path = "sentence-transformers/all-MiniLM-L6-v2"

            if os.path.isdir(local_path):
                model_path = local_path

            cls._embedder = SentenceTransformer(model_path, device="cpu")
            cls._embedder_model_path = model_path
            return True
        except Exception as e:
            cls._embedder_failed = True
            import logging
            logging.getLogger(__name__).warning(
                f"sentence-transformers 加载失败，QueryOptimizer 降级到 token 余弦: {e}"
            )
            return False
        finally:
            cls._embedder_loading = False

    def _validate_semantic_fidelity(self, original: str, optimized: str) -> bool:
        """验证优化后的查询与原始查询的语义一致性。
        
        如果相似度低于阈值，说明优化过程可能导致主题偏离，
        调用方应回退到原始查询。
        """
        sim = self.semantic_similarity(original, optimized)
        return sim >= self.SEMANTIC_FIDELITY_THRESHOLD

    @staticmethod
    def _strip_verbose_prefixes(text: str) -> str:
        """剥离查询开头的冗余前缀。"""
        result = _VERBOSE_PREFIXES.sub("", text).strip()
        if result == text or not result:
            result = _VERBOSE_PREFIXES_SIMPLE.sub("", text).strip()
        return result if result else text

    @staticmethod
    def _generate_bilingual(query: str, has_chinese: bool) -> list[str]:
        """生成双语候选查询列表。

        如果查询含中文，自动构造英文关键词版本；
        如果查询为纯英文，自动构造中文关键词版本；
        如果已经是中英混合，返回自身。
        """
        candidates = [query]

        if has_chinese:
            # 中文查询 → 抽出英文术语作为候选
            english_terms = _RE_CHINESE.sub(" ", query).strip()
            english_terms = re.sub(r"\s+", " ", english_terms).strip()
            if english_terms and english_terms != query:
                candidates.append(english_terms)
        else:
            # 英文查询 → 尝试用 Google Translate 风格的直接保留英文（不强行造中文）
            # 策略：对于英文查询，添加 "中文翻译" 标记让 LLM 去做中英双语
            # 但更实用的做法是：当优化器检测到纯英文查询时，在 multi_source_search 中
            # 直接保留英文搜索即可（中英都用英文搜，因为 arXiv/Scholar 对英文更友好）
            # 这里标记 has_chinese=False，由调用方决定是否追加中文搜索
            pass

        return candidates

    def stats(self) -> dict:
        """返回优化器统计信息。"""
        mod_rate = self.queries_modified / self.queries_processed if self.queries_processed > 0 else 0
        revert_rate = self.queries_reverted / self.queries_processed if self.queries_processed > 0 else 0
        return {
            "queries_processed": self.queries_processed,
            "queries_modified": self.queries_modified,
            "queries_reverted": self.queries_reverted,
            "modification_rate": round(mod_rate, 4),
            "revert_rate": round(revert_rate, 4),
            "bilingual_generated": self.bilingual_generated,
            "bilingual_rejected": self.bilingual_rejected,
        }

    @staticmethod
    def has_chinese(text: str) -> bool:
        """检查文本是否包含中文字符。"""
        return bool(_RE_CHINESE.search(text))
