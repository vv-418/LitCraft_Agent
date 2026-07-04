"""
搜索缓存层：提供 LRU + TTL 的持久化缓存，降低重复搜索 API 调用。

工作方式：
- 以 (source, query_md5, year_from) 三元组为键
- SQLite 持久化，进程重启后缓存不丢失
- LRU 淘汰：超出 max_entries 时淘汰最久未访问的条目
- TTL 过期：超过 ttl_seconds 的条目自动视为过期

缓存键示例：
    "arxiv|a7f3c2...|2024"
    "semantic_scholar|8b1d5e...|"
"""

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional


class SearchCache:
    """搜索缓存，LRU + TTL + SQLite 持久化。"""

    def __init__(
        self,
        db_path: str = "./storage/search_cache.db",
        max_entries: int = 500,
        ttl_seconds: int = 3600,
    ):
        """
        Args:
            db_path: SQLite 数据库路径
            max_entries: 最大缓存条目数，超出后淘汰 LRU
            ttl_seconds: 缓存 TTL（秒），默认 1 小时
        """
        self.db_path = Path(db_path)
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()

        # 统计信息
        self.hits = 0
        self.misses = 0

        # 初始化数据库
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """创建缓存表并执行 LRU 清理。"""
        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS search_cache (
                    cache_key TEXT PRIMARY KEY,
                    result_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    accessed_at REAL NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_accessed ON search_cache(accessed_at)"
            )
            conn.commit()
        self._evict_expired()

    def _evict_expired(self):
        """淘汰过期条目和超出 LRU 限制的条目。"""
        cutoff = time.time() - self.ttl_seconds
        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            # 删除过期的
            conn.execute("DELETE FROM search_cache WHERE created_at < ?", (cutoff,))
            # 如果仍然超出上限，删除最久未访问的
            count = conn.execute("SELECT COUNT(*) FROM search_cache").fetchone()[0]
            if count > self.max_entries:
                excess = count - self.max_entries
                conn.execute(
                    """DELETE FROM search_cache WHERE cache_key IN (
                        SELECT cache_key FROM search_cache
                        ORDER BY accessed_at ASC
                        LIMIT ?
                    )""",
                    (excess,),
                )
            conn.commit()

    @staticmethod
    def _make_key(source: str, query: str, year_from: str = "") -> str:
        """生成缓存键：(source, query_md5, year_from)。"""
        q_md5 = hashlib.md5(query.lower().strip().encode()).hexdigest()
        return f"{source}|{q_md5}|{year_from}"

    def get(self, source: str, query: str, year_from: str = "") -> Optional[str]:
        """获取缓存结果（如果存在且未过期）。"""
        key = self._make_key(source, query, year_from)
        now = time.time()
        cutoff = now - self.ttl_seconds

        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT result_json, created_at FROM search_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()

        if row is None:
            self.misses += 1
            return None

        result_json, created_at = row

        if created_at < cutoff:
            # TTL 过期：删除
            with self._lock, sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("DELETE FROM search_cache WHERE cache_key = ?", (key,))
                conn.commit()
            self.misses += 1
            return None

        # 更新访问时间
        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "UPDATE search_cache SET accessed_at = ? WHERE cache_key = ?",
                (now, key),
            )
            conn.commit()

        self.hits += 1
        return result_json

    def set(self, source: str, query: str, year_from: str, result_json: str) -> None:
        """写入缓存结果。"""
        key = self._make_key(source, query, year_from)
        now = time.time()

        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO search_cache
                   (cache_key, result_json, created_at, accessed_at)
                   VALUES (?, ?, ?, ?)""",
                (key, result_json, now, now),
            )
            conn.commit()

    def update_chroma_status(self, source: str, query: str, year_from: str,
                             titles_indexed: list[str]) -> None:
        """标记某些论文已入库 Chroma，在 result_json 中为这些论文加上 indexed=True。

        Args:
            titles_indexed: 已成功入库 Chroma 的论文标题列表
        """
        cached = self.get(source, query, year_from)
        if cached is None:
            return
        try:
            data = json.loads(cached)
            # 兼容两种缓存格式
            papers_key = "results" if "results" in data else "papers"
            papers = data.get(papers_key, [])
            title_set = set(t.strip().lower() for t in titles_indexed)
            for p in papers:
                title = (p.get("title") or "").strip().lower()
                if title in title_set:
                    p["chroma_indexed"] = True
            data[papers_key] = papers
            # 写回缓存
            key = self._make_key(source, query, year_from)
            now = time.time()
            with self._lock, sqlite3.connect(str(self.db_path)) as conn:
                conn.execute(
                    "UPDATE search_cache SET result_json = ?, accessed_at = ? WHERE cache_key = ?",
                    (json.dumps(data, ensure_ascii=False), now, key),
                )
                conn.commit()
        except (json.JSONDecodeError, TypeError):
            pass

        # 每次写入后检查是否超出上限
        self._evict_expired()

    def stats(self) -> dict:
        """返回缓存统计信息。"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            entry_count = conn.execute(
                "SELECT COUNT(*) FROM search_cache"
            ).fetchone()[0]
            oldest = conn.execute(
                "SELECT MIN(accessed_at) FROM search_cache"
            ).fetchone()[0]
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 4),
            "entries": entry_count,
            "oldest_entry_timestamp": oldest or 0,
            "max_entries": self.max_entries,
            "ttl_seconds": self.ttl_seconds,
        }

    def clear(self) -> None:
        """清空所有缓存。"""
        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("DELETE FROM search_cache")
            conn.commit()
        self.hits = 0
        self.misses = 0
